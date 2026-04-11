import argparse
import os
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np
import torch
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    from segment_anything import SamPredictor, sam_model_registry
except ImportError:
    print("segment_anything is not installed. Run: pip install segment-anything")
    raise SystemExit(1)


CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Batch-generate SAM masks and masked images in a mirrored fish/frame folder structure."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=base_dir / "final stages clean no repeat",
        help="Path to source fish/frame dataset.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=base_dir / "masked_images_SAM",
        help="Path where masked output folders will be created.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=base_dir / "sam_vit_b_01ec64.pth",
        help="Path to SAM checkpoint (vit_b).",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=1280,
        help="Resize longest image side for SAM inference to reduce VRAM usage (0 disables resize).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--disable-background-modeling",
        action="store_true",
        help="Disable background modeling stage (enabled by default).",
    )
    parser.add_argument(
        "--bg-samples",
        type=int,
        default=80,
        help="Maximum number of frames per fish/view used to build background model (0 = use all).",
    )
    parser.add_argument(
        "--bg-motion-proximity-factor",
        type=float,
        default=0.18,
        help="Prompt pixels farther than this fraction of fish bbox from motion map are suppressed.",
    )
    parser.add_argument(
        "--bg-guided-min-ratio",
        type=float,
        default=0.22,
        help="Minimum kept ratio of prompt after background guidance; otherwise fallback to original prompt.",
    )
    parser.add_argument(
        "--disable-visibility-enhance",
        action="store_true",
        help="Disable contrast enhancement before SAM (enabled by default).",
    )
    parser.add_argument(
        "--negative-points",
        type=int,
        default=10,
        help="Number of background negative prompt points around fish prompt to suppress tape/reflections.",
    )
    parser.add_argument(
        "--negative-margin-ratio",
        type=float,
        default=0.35,
        help="Margin ratio around fish prompt bbox used to place negative points.",
    )
    parser.add_argument(
        "--distance-factor",
        type=float,
        default=0.32,
        help="Max allowed SAM expansion from prompt mask as a fraction of prompt bbox size.",
    )
    parser.add_argument(
        "--distance-floor",
        type=int,
        default=12,
        help="Minimum pixel distance allowed from prompt mask during refinement.",
    )
    parser.add_argument(
        "--distance-cap",
        type=int,
        default=72,
        help="Maximum pixel distance allowed from prompt mask during refinement.",
    )
    parser.add_argument(
        "--bridge-break-kernel",
        type=int,
        default=5,
        help="Odd kernel size to break thin bridges (fish-to-tape links) before component filtering.",
    )
    parser.add_argument(
        "--prompt-dilate-kernel",
        type=int,
        default=31,
        help="Odd kernel size for dilating prompt mask while removing far-away noise components.",
    )
    parser.add_argument(
        "--min-component-area",
        type=int,
        default=250,
        help="Minimum connected component area in SAM mask (adaptive floor also uses 1%% of prompt area).",
    )
    parser.add_argument(
        "--min-overlap-ratio",
        type=float,
        default=0.05,
        help="Minimum overlap ratio between a SAM component and dilated prompt mask.",
    )
    parser.add_argument(
        "--fallback-prompt-ratio",
        type=float,
        default=0.20,
        help="Fallback to prompt mask if refined SAM mask area is below this ratio of prompt area.",
    )
    parser.add_argument(
        "--fish",
        type=str,
        default="",
        help="Optional comma-separated fish folder names to process (e.g. fish01,fish2).",
    )
    parser.add_argument(
        "--max-frames-per-fish",
        type=int,
        default=0,
        help="Optional frame limit per fish for quick validation (0 = all frames).",
    )
    return parser.parse_args()


def ensure_checkpoint(checkpoint_path: Path) -> None:
    if checkpoint_path.exists():
        return
    print(f"Checkpoint not found. Downloading to: {checkpoint_path}")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(CHECKPOINT_URL, str(checkpoint_path))


def largest_contour_bbox(binary_mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def sanitize_odd_kernel(kernel_size: int, minimum: int = 3) -> int:
    k = max(minimum, int(kernel_size))
    if k % 2 == 0:
        k += 1
    return k


def mask_centroid(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    return np.array([float(xs.mean()), float(ys.mean())], dtype=np.float32)


def enhance_low_visibility_for_sam(bgr_image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((l_channel, a_channel, b_channel))
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    denoised = cv2.bilateralFilter(enhanced_bgr, d=5, sigmaColor=35, sigmaSpace=35)
    blurred = cv2.GaussianBlur(denoised, (0, 0), 1.2)
    sharpened = cv2.addWeighted(denoised, 1.25, blurred, -0.25, 0)
    return sharpened


def build_background_model_from_paths(image_paths: list[Path], max_samples: int) -> np.ndarray | None:
    valid_paths = [p for p in image_paths if p.exists()]
    if not valid_paths:
        return None

    max_samples = int(max_samples)
    if max_samples > 0 and len(valid_paths) > max_samples:
        sample_indices = np.linspace(0, len(valid_paths) - 1, max_samples, dtype=np.int32)
        selected_paths = [valid_paths[int(idx)] for idx in sample_indices]
    else:
        selected_paths = valid_paths

    frames_u8: list[np.ndarray] = []
    target_hw: tuple[int, int] | None = None
    for img_path in selected_paths:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        if target_hw is None:
            target_hw = frame.shape[:2]
        elif frame.shape[:2] != target_hw:
            frame = cv2.resize(frame, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_AREA)

        frames_u8.append(frame)

    if not frames_u8:
        return None

    stack = np.stack(frames_u8, axis=0)
    background = np.median(stack, axis=0).astype(np.uint8)
    return background


def build_fish_background_models(
    frame_dirs: list[Path],
    view_specs: list[dict[str, str]],
    max_samples: int,
) -> dict[str, np.ndarray | None]:
    models: dict[str, np.ndarray | None] = {}
    for spec in view_specs:
        raw_name = spec["raw_name"]
        image_paths = [frame_dir / raw_name for frame_dir in frame_dirs]
        models[raw_name] = build_background_model_from_paths(image_paths=image_paths, max_samples=max_samples)
    return models


def compute_motion_prior_from_background(raw_bgr: np.ndarray, background_bgr: np.ndarray | None) -> np.ndarray:
    if background_bgr is None:
        h, w = raw_bgr.shape[:2]
        return np.zeros((h, w), dtype=np.uint8)

    if background_bgr.shape[:2] != raw_bgr.shape[:2]:
        background_bgr = cv2.resize(
            background_bgr,
            (raw_bgr.shape[1], raw_bgr.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    raw_blur = cv2.GaussianBlur(raw_bgr, (5, 5), 0)
    bg_blur = cv2.GaussianBlur(background_bgr, (5, 5), 0)

    bgr_diff = cv2.absdiff(raw_blur, bg_blur)
    gray_diff = cv2.cvtColor(bgr_diff, cv2.COLOR_BGR2GRAY)

    raw_hsv = cv2.cvtColor(raw_blur, cv2.COLOR_BGR2HSV)
    bg_hsv = cv2.cvtColor(bg_blur, cv2.COLOR_BGR2HSV)
    sat_diff = cv2.absdiff(raw_hsv[:, :, 1], bg_hsv[:, :, 1])
    val_diff = cv2.absdiff(raw_hsv[:, :, 2], bg_hsv[:, :, 2])

    motion = cv2.max(gray_diff, sat_diff)
    motion = cv2.max(motion, val_diff)
    motion = cv2.GaussianBlur(motion, (5, 5), 0)
    return motion


def guide_prompt_with_background_model(
    raw_bgr: np.ndarray,
    prompt_mask_u8: np.ndarray,
    background_bgr: np.ndarray | None,
    motion_proximity_factor: float,
    guided_min_ratio: float,
) -> tuple[np.ndarray, np.ndarray | None]:
    prompt_bin = (prompt_mask_u8 > 127).astype(np.uint8)
    prompt_area = int(prompt_bin.sum())
    if prompt_area == 0:
        return prompt_mask_u8, None

    motion_u8 = compute_motion_prior_from_background(raw_bgr=raw_bgr, background_bgr=background_bgr)
    if motion_u8.max() <= 0:
        return prompt_mask_u8, None

    otsu_thr, _ = cv2.threshold(motion_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    motion_thr = max(6, int(round(0.75 * float(otsu_thr))))
    motion_fg = (motion_u8 >= motion_thr).astype(np.uint8)

    motion_fg = cv2.morphologyEx(
        motion_fg,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    motion_fg = cv2.morphologyEx(
        motion_fg,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    motion_fg = cv2.dilate(
        motion_fg,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )

    bbox = largest_contour_bbox(prompt_mask_u8)
    if bbox is None:
        return prompt_mask_u8, (motion_u8.astype(np.float32) / 255.0)

    box_w = max(1, int(round(bbox[2] - bbox[0])))
    box_h = max(1, int(round(bbox[3] - bbox[1])))

    motion_proximity_factor = max(0.05, float(motion_proximity_factor))
    proximity_px = int(round(max(box_w, box_h) * motion_proximity_factor))
    proximity_px = min(max(4, proximity_px), 24)

    distance_to_motion = cv2.distanceTransform((motion_fg == 0).astype(np.uint8), cv2.DIST_L2, 3)
    near_motion = distance_to_motion <= float(proximity_px)

    core_prompt = cv2.erode(
        prompt_bin,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    guided_prompt = np.logical_and(prompt_bin > 0, near_motion)
    guided_prompt = np.logical_or(guided_prompt, core_prompt > 0)
    guided_prompt = cv2.morphologyEx(
        guided_prompt.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )

    guided_min_ratio = float(np.clip(guided_min_ratio, 0.05, 1.0))
    guided_area = int(guided_prompt.sum())
    if guided_area < max(64, int(prompt_area * guided_min_ratio)):
        guided_prompt = prompt_bin

    motion_prior = motion_u8.astype(np.float32) / 255.0
    return ((guided_prompt * 255).astype(np.uint8)), motion_prior


def deduplicate_points(points_xy: list[tuple[float, float]], min_distance: float = 3.0) -> list[tuple[float, float]]:
    kept: list[tuple[float, float]] = []
    min_distance_sq = float(min_distance * min_distance)
    for x_coord, y_coord in points_xy:
        keep = True
        for kept_x, kept_y in kept:
            dx = x_coord - kept_x
            dy = y_coord - kept_y
            if (dx * dx) + (dy * dy) < min_distance_sq:
                keep = False
                break
        if keep:
            kept.append((x_coord, y_coord))
    return kept


def build_prompt_points(
    prompt_mask_u8: np.ndarray,
    bbox_xyxy: np.ndarray,
    negative_points: int,
    negative_margin_ratio: float,
    motion_prior: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    prompt_bin = (prompt_mask_u8 > 127).astype(np.uint8)
    ys, xs = np.where(prompt_bin > 0)
    if xs.size == 0:
        return None, None

    positives: list[tuple[float, float]] = [
        (float(xs.mean()), float(ys.mean())),
        (float(xs[np.argmin(xs)]), float(ys[np.argmin(xs)])),
        (float(xs[np.argmax(xs)]), float(ys[np.argmax(xs)])),
        (float(xs[np.argmin(ys)]), float(ys[np.argmin(ys)])),
        (float(xs[np.argmax(ys)]), float(ys[np.argmax(ys)])),
    ]
    positives = deduplicate_points(positives, min_distance=3.0)

    h, w = prompt_mask_u8.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy]
    x1 = int(np.clip(x1, 0, max(0, w - 1)))
    x2 = int(np.clip(x2, 0, max(0, w - 1)))
    y1 = int(np.clip(y1, 0, max(0, h - 1)))
    y2 = int(np.clip(y2, 0, max(0, h - 1)))

    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    margin_x = max(4, int(round(box_w * max(0.05, negative_margin_ratio))))
    margin_y = max(4, int(round(box_h * max(0.05, negative_margin_ratio))))

    ex1 = max(0, x1 - margin_x)
    ex2 = min(w - 1, x2 + margin_x)
    ey1 = max(0, y1 - margin_y)
    ey2 = min(h - 1, y2 + margin_y)
    cx = (ex1 + ex2) // 2
    cy = (ey1 + ey2) // 2

    negatives: list[tuple[float, float]] = []
    edge_points = [
        (ex1, ey1), (cx, ey1), (ex2, ey1),
        (ex1, cy),
        (ex2, cy),
        (ex1, ey2), (cx, ey2), (ex2, ey2),
    ]
    for px, py in edge_points:
        if prompt_bin[py, px] == 0:
            negatives.append((float(px), float(py)))

    negative_points = max(0, int(negative_points))
    if len(negatives) < negative_points:
        inner = cv2.dilate(prompt_bin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
        ring_kernel_size = sanitize_odd_kernel(int(round(0.18 * max(box_w, box_h))), minimum=9)
        outer = cv2.dilate(
            prompt_bin,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_kernel_size, ring_kernel_size)),
            iterations=1,
        )
        ring = np.logical_and(outer > 0, inner == 0)
        ring_ys, ring_xs = np.where(ring)
        if ring_xs.size > 0:
            step = max(1, ring_xs.size // max(1, negative_points - len(negatives)))
            for idx in range(0, ring_xs.size, step):
                negatives.append((float(ring_xs[idx]), float(ring_ys[idx])))
                if len(negatives) >= negative_points:
                    break

    if motion_prior is not None and motion_prior.shape == prompt_mask_u8.shape:
        filtered_negatives: list[tuple[float, float]] = []
        for x_coord, y_coord in negatives:
            x_idx = int(np.clip(int(round(x_coord)), 0, w - 1))
            y_idx = int(np.clip(int(round(y_coord)), 0, h - 1))
            if float(motion_prior[y_idx, x_idx]) < 0.18:
                filtered_negatives.append((x_coord, y_coord))
        if filtered_negatives:
            negatives = filtered_negatives

    negatives = deduplicate_points(negatives, min_distance=3.0)[:negative_points]
    all_points = positives + negatives
    labels = ([1] * len(positives)) + ([0] * len(negatives))
    if not all_points:
        return None, None

    point_coords = np.array(all_points, dtype=np.float32)
    point_labels = np.array(labels, dtype=np.int32)
    return point_coords, point_labels


def compute_prompt_distance_prior(
    prompt_mask_u8: np.ndarray,
    distance_factor: float,
    distance_floor: int,
    distance_cap: int,
) -> tuple[np.ndarray, float]:
    prompt_bin = (prompt_mask_u8 > 127).astype(np.uint8)
    distance_map = cv2.distanceTransform((prompt_bin == 0).astype(np.uint8), cv2.DIST_L2, 3)

    bbox = largest_contour_bbox(prompt_mask_u8)
    if bbox is None:
        return distance_map, float(max(0, int(distance_floor)))

    box_w = max(1, int(round(bbox[2] - bbox[0])))
    box_h = max(1, int(round(bbox[3] - bbox[1])))

    distance_factor = max(0.05, float(distance_factor))
    distance_floor = max(0, int(distance_floor))
    distance_cap = max(0, int(distance_cap))

    max_allowed = max(distance_floor, int(round(max(box_w, box_h) * distance_factor)))
    if distance_cap > 0:
        max_allowed = min(max_allowed, distance_cap)
    return distance_map, float(max_allowed)


def select_best_candidate_mask(
    candidate_masks: np.ndarray,
    predicted_ious: np.ndarray,
    prompt_mask_u8: np.ndarray,
    prompt_distance_map: np.ndarray,
    max_prompt_distance: float,
    motion_prior: np.ndarray | None,
) -> np.ndarray:
    prompt_bool = prompt_mask_u8 > 0
    prompt_area = float(prompt_bool.sum())
    prompt_center = mask_centroid(prompt_mask_u8)
    image_diag = float(np.hypot(prompt_mask_u8.shape[0], prompt_mask_u8.shape[1]))

    best_idx = 0
    best_score = -1e9

    for idx, candidate in enumerate(candidate_masks):
        cand_bool = candidate.astype(bool)
        cand_area = float(cand_bool.sum())
        if cand_area <= 0:
            continue

        if prompt_area > 0:
            intersection = float(np.logical_and(cand_bool, prompt_bool).sum())
            union = float(np.logical_or(cand_bool, prompt_bool).sum())
            iou_with_prompt = intersection / union if union > 0 else 0.0
            coverage_of_prompt = intersection / prompt_area
            area_ratio = cand_area / prompt_area
        else:
            iou_with_prompt = 0.0
            coverage_of_prompt = 0.0
            area_ratio = 1.0

        if max_prompt_distance > 0:
            outside_pixels = float(np.logical_and(cand_bool, prompt_distance_map > max_prompt_distance).sum())
            outside_ratio = outside_pixels / cand_area
        else:
            outside_ratio = 0.0

        motion_penalty = 0.0
        if motion_prior is not None and motion_prior.shape == prompt_mask_u8.shape:
            motion_values = motion_prior[cand_bool]
            if motion_values.size > 0:
                low_motion_ratio = float(np.mean(motion_values < 0.15))
                motion_penalty = low_motion_ratio

        cand_center = mask_centroid(cand_bool.astype(np.uint8))
        if prompt_center is not None and cand_center is not None and image_diag > 0:
            center_dist = float(np.linalg.norm(cand_center - prompt_center))
            center_score = 1.0 - min(1.0, center_dist / image_diag)
        else:
            center_score = 0.0

        model_iou = float(predicted_ious[idx]) if idx < len(predicted_ious) else 0.0
        large_area_penalty = max(0.0, area_ratio - 2.5)

        score = (
            0.50 * model_iou
            + 0.25 * iou_with_prompt
            + 0.20 * coverage_of_prompt
            + 0.10 * center_score
            - 0.10 * min(1.0, large_area_penalty / 2.5)
            - 0.30 * outside_ratio
            - 0.22 * motion_penalty
        )

        if score > best_score:
            best_score = score
            best_idx = idx

    return candidate_masks[best_idx].astype(np.uint8)


def keep_best_component_near_prompt(
    mask_u8: np.ndarray,
    prompt_mask_u8: np.ndarray,
    min_component_area: int,
    min_overlap_ratio: float,
    prompt_dilate_kernel: int,
) -> np.ndarray:
    mask_bin = (mask_u8 > 127).astype(np.uint8)
    prompt_bin = (prompt_mask_u8 > 127).astype(np.uint8)

    if mask_bin.sum() == 0:
        return mask_bin

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    if num_labels <= 1:
        return mask_bin

    k = sanitize_odd_kernel(prompt_dilate_kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dilated_prompt = cv2.dilate(prompt_bin, kernel, iterations=1)

    prompt_area = int(prompt_bin.sum())
    adaptive_min_area = max(int(min_component_area), int(prompt_area * 0.01))
    min_overlap_ratio = float(np.clip(min_overlap_ratio, 0.0, 1.0))

    best_label = -1
    best_score = -1.0

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < adaptive_min_area:
            continue

        component = labels == label
        overlap_dilated = int(np.logical_and(component, dilated_prompt > 0).sum())
        overlap_ratio = overlap_dilated / float(area)
        if overlap_ratio < min_overlap_ratio:
            continue

        overlap_prompt = int(np.logical_and(component, prompt_bin > 0).sum())
        score = overlap_prompt + (0.15 * overlap_dilated) + (0.01 * area)
        if score > best_score:
            best_score = score
            best_label = label

    if best_label == -1:
        # Fallback: keep the component with highest overlap with the prompt, tie-break by area.
        best_overlap = -1
        best_area = -1
        for label in range(1, num_labels):
            component = labels == label
            overlap_prompt = int(np.logical_and(component, prompt_bin > 0).sum())
            area = int(stats[label, cv2.CC_STAT_AREA])
            if overlap_prompt > best_overlap or (overlap_prompt == best_overlap and area > best_area):
                best_overlap = overlap_prompt
                best_area = area
                best_label = label

    return (labels == best_label).astype(np.uint8)


def refine_mask(
    mask_u8: np.ndarray,
    prompt_mask_u8: np.ndarray,
    min_component_area: int,
    min_overlap_ratio: float,
    prompt_dilate_kernel: int,
    fallback_prompt_ratio: float,
    distance_factor: float,
    distance_floor: int,
    distance_cap: int,
    bridge_break_kernel: int,
) -> np.ndarray:
    prompt_bin = (prompt_mask_u8 > 127).astype(np.uint8)
    work_mask = (mask_u8 > 127).astype(np.uint8)

    k_bridge = sanitize_odd_kernel(bridge_break_kernel)
    if k_bridge >= 3:
        work_mask = cv2.morphologyEx(
            work_mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_bridge, k_bridge)),
        )

    distance_map, max_prompt_distance = compute_prompt_distance_prior(
        prompt_mask_u8=prompt_mask_u8,
        distance_factor=distance_factor,
        distance_floor=distance_floor,
        distance_cap=distance_cap,
    )
    if max_prompt_distance > 0:
        clipped = np.logical_and(work_mask > 0, distance_map <= max_prompt_distance)
        clipped = np.logical_or(clipped, np.logical_and(work_mask > 0, prompt_bin > 0))

        clipped_area = int(clipped.sum())
        prompt_area = int(prompt_bin.sum())
        min_required = max(64, int(prompt_area * 0.25))
        if clipped_area >= min_required:
            work_mask = clipped.astype(np.uint8)

    kept_component = keep_best_component_near_prompt(
        mask_u8=(work_mask * 255).astype(np.uint8),
        prompt_mask_u8=prompt_mask_u8,
        min_component_area=min_component_area,
        min_overlap_ratio=min_overlap_ratio,
        prompt_dilate_kernel=prompt_dilate_kernel,
    )

    cleaned = kept_component
    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )

    if max_prompt_distance > 0:
        cleaned = np.logical_and(cleaned > 0, distance_map <= max_prompt_distance).astype(np.uint8)
        cleaned = np.logical_or(cleaned > 0, np.logical_and(kept_component > 0, prompt_bin > 0)).astype(np.uint8)

    prompt_area = int(prompt_bin.sum())
    cleaned_area = int(cleaned.sum())
    fallback_prompt_ratio = float(np.clip(fallback_prompt_ratio, 0.0, 1.0))

    if prompt_area > 0 and cleaned_area < max(64, int(prompt_area * fallback_prompt_ratio)):
        cleaned = prompt_bin

    return (cleaned * 255).astype(np.uint8)


def resize_image_and_box(
    bgr_image: np.ndarray, bbox_xyxy: np.ndarray, max_side: int
) -> tuple[np.ndarray, np.ndarray, float]:
    if max_side <= 0:
        return bgr_image, bbox_xyxy, 1.0

    height, width = bgr_image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return bgr_image, bbox_xyxy, 1.0

    scale = max_side / float(longest)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = cv2.resize(bgr_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    resized_bbox = bbox_xyxy * scale
    return resized, resized_bbox, scale


def predict_sam_mask(
    predictor: SamPredictor,
    raw_bgr: np.ndarray,
    prompt_mask_u8: np.ndarray,
    prompt_bbox_xyxy: np.ndarray,
    max_side: int,
    background_bgr: np.ndarray | None,
    enable_background_modeling: bool,
    bg_motion_proximity_factor: float,
    bg_guided_min_ratio: float,
    enable_visibility_enhance: bool,
    negative_points: int,
    negative_margin_ratio: float,
    prompt_dilate_kernel: int,
    min_component_area: int,
    min_overlap_ratio: float,
    fallback_prompt_ratio: float,
    distance_factor: float,
    distance_floor: int,
    distance_cap: int,
    bridge_break_kernel: int,
) -> np.ndarray:
    prompt_for_model = prompt_mask_u8
    motion_prior_full: np.ndarray | None = None
    if enable_background_modeling:
        prompt_for_model, motion_prior_full = guide_prompt_with_background_model(
            raw_bgr=raw_bgr,
            prompt_mask_u8=prompt_mask_u8,
            background_bgr=background_bgr,
            motion_proximity_factor=bg_motion_proximity_factor,
            guided_min_ratio=bg_guided_min_ratio,
        )

    prompt_bbox_for_model = largest_contour_bbox(prompt_for_model)
    if prompt_bbox_for_model is None:
        prompt_bbox_for_model = prompt_bbox_xyxy

    sam_bgr, sam_bbox, scale = resize_image_and_box(raw_bgr, prompt_bbox_for_model, max_side)
    prompt_mask_for_sam = prompt_for_model
    motion_prior_for_sam = motion_prior_full
    if scale != 1.0:
        sam_h, sam_w = sam_bgr.shape[:2]
        prompt_mask_for_sam = cv2.resize(
            prompt_for_model,
            (sam_w, sam_h),
            interpolation=cv2.INTER_NEAREST,
        )
        if motion_prior_full is not None:
            motion_prior_for_sam = cv2.resize(
                motion_prior_full,
                (sam_w, sam_h),
                interpolation=cv2.INTER_LINEAR,
            )

    point_coords, point_labels = build_prompt_points(
        prompt_mask_u8=prompt_mask_for_sam,
        bbox_xyxy=sam_bbox,
        negative_points=negative_points,
        negative_margin_ratio=negative_margin_ratio,
        motion_prior=motion_prior_for_sam,
    )

    score_distance_map, score_max_prompt_distance = compute_prompt_distance_prior(
        prompt_mask_u8=prompt_mask_for_sam,
        distance_factor=distance_factor,
        distance_floor=distance_floor,
        distance_cap=distance_cap,
    )

    sam_input_bgr = sam_bgr
    if enable_visibility_enhance:
        sam_input_bgr = enhance_low_visibility_for_sam(sam_input_bgr)

    sam_rgb = cv2.cvtColor(sam_input_bgr, cv2.COLOR_BGR2RGB)

    predictor.set_image(sam_rgb)
    with torch.no_grad():
        masks, predicted_ious, _ = predictor.predict(
            box=sam_bbox,
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )

    best_mask = select_best_candidate_mask(
        candidate_masks=masks,
        predicted_ious=predicted_ious,
        prompt_mask_u8=prompt_mask_for_sam,
        prompt_distance_map=score_distance_map,
        max_prompt_distance=score_max_prompt_distance,
        motion_prior=motion_prior_for_sam,
    )
    sam_mask = refine_mask(
        mask_u8=(best_mask.astype(np.uint8) * 255),
        prompt_mask_u8=prompt_mask_for_sam,
        min_component_area=min_component_area,
        min_overlap_ratio=min_overlap_ratio,
        prompt_dilate_kernel=prompt_dilate_kernel,
        fallback_prompt_ratio=fallback_prompt_ratio,
        distance_factor=distance_factor,
        distance_floor=distance_floor,
        distance_cap=distance_cap,
        bridge_break_kernel=bridge_break_kernel,
    )

    if scale != 1.0:
        h, w = raw_bgr.shape[:2]
        sam_mask = cv2.resize(sam_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    return sam_mask


def apply_binary_mask(raw_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    out = np.zeros_like(raw_bgr)
    foreground = mask_u8 > 127
    out[foreground] = raw_bgr[foreground]
    return out


def process_view(
    predictor: SamPredictor,
    source_frame_dir: Path,
    output_frame_dir: Path,
    raw_name: str,
    prompt_mask_name: str,
    out_mask_name: str,
    out_masked_name: str,
    max_side: int,
    background_bgr: np.ndarray | None,
    enable_background_modeling: bool,
    bg_motion_proximity_factor: float,
    bg_guided_min_ratio: float,
    enable_visibility_enhance: bool,
    negative_points: int,
    negative_margin_ratio: float,
    prompt_dilate_kernel: int,
    min_component_area: int,
    min_overlap_ratio: float,
    fallback_prompt_ratio: float,
    distance_factor: float,
    distance_floor: int,
    distance_cap: int,
    bridge_break_kernel: int,
    overwrite: bool,
) -> str:
    output_frame_dir.mkdir(parents=True, exist_ok=True)
    out_mask_path = output_frame_dir / out_mask_name
    out_masked_path = output_frame_dir / out_masked_name

    if not overwrite and out_mask_path.exists() and out_masked_path.exists():
        return "skipped"

    raw_path = source_frame_dir / raw_name
    prompt_mask_path = source_frame_dir / prompt_mask_name
    if not raw_path.exists() or not prompt_mask_path.exists():
        return "missing_inputs"

    raw_bgr = cv2.imread(str(raw_path))
    prompt_mask = cv2.imread(str(prompt_mask_path), cv2.IMREAD_GRAYSCALE)
    if raw_bgr is None or prompt_mask is None:
        return "read_failed"

    bbox = largest_contour_bbox(prompt_mask)
    if bbox is None:
        return "no_prompt_contour"

    sam_mask = predict_sam_mask(
        predictor=predictor,
        raw_bgr=raw_bgr,
        prompt_mask_u8=prompt_mask,
        prompt_bbox_xyxy=bbox,
        max_side=max_side,
        background_bgr=background_bgr,
        enable_background_modeling=enable_background_modeling,
        bg_motion_proximity_factor=bg_motion_proximity_factor,
        bg_guided_min_ratio=bg_guided_min_ratio,
        enable_visibility_enhance=enable_visibility_enhance,
        negative_points=negative_points,
        negative_margin_ratio=negative_margin_ratio,
        prompt_dilate_kernel=prompt_dilate_kernel,
        min_component_area=min_component_area,
        min_overlap_ratio=min_overlap_ratio,
        fallback_prompt_ratio=fallback_prompt_ratio,
        distance_factor=distance_factor,
        distance_floor=distance_floor,
        distance_cap=distance_cap,
        bridge_break_kernel=bridge_break_kernel,
    )
    masked_image = apply_binary_mask(raw_bgr, sam_mask)

    cv2.imwrite(str(out_mask_path), sam_mask)
    cv2.imwrite(str(out_masked_path), masked_image)
    return "processed"


def main() -> None:
    args = parse_args()

    ensure_checkpoint(args.checkpoint)

    if not args.source_root.exists():
        print(f"Source root not found: {args.source_root}")
        raise SystemExit(1)

    if torch.cuda.is_available():
        device = "cuda"
        props = torch.cuda.get_device_properties(0)
        print(
            f"Using GPU: {torch.cuda.get_device_name(0)} "
            f"({props.total_memory / 1e9:.1f} GB VRAM)"
        )
    else:
        device = "cpu"
        print("CUDA not found. Running on CPU (slower).")

    sam = sam_model_registry["vit_b"](checkpoint=str(args.checkpoint))
    sam.to(device=device)
    predictor = SamPredictor(sam)

    fish_dirs = sorted(
        [p for p in args.source_root.iterdir() if p.is_dir() and p.name.lower().startswith("fish")],
        key=lambda p: p.name,
    )

    fish_filter = {item.strip() for item in args.fish.split(",") if item.strip()}
    if fish_filter:
        fish_dirs = [p for p in fish_dirs if p.name in fish_filter]

    if not fish_dirs:
        print("No fish folders found in source root.")
        raise SystemExit(1)

    print(f"Source: {args.source_root}")
    print(f"Output: {args.output_root}")
    print(f"Fish folders detected: {len(fish_dirs)}")

    counts = {
        "processed": 0,
        "skipped": 0,
        "missing_inputs": 0,
        "read_failed": 0,
        "no_prompt_contour": 0,
    }

    view_specs = [
        {
            "raw_name": "01_top_raw.jpg",
            "prompt_name": "40_top_final_mask.png",
            "out_mask": "01_top_SAM_mask.png",
            "out_masked": "01_top_raw_masked_SAM.png",
        },
        {
            "raw_name": "21_front_raw.jpg",
            "prompt_name": "41_front_final_mask.png",
            "out_mask": "21_front_SAM_mask.png",
            "out_masked": "21_front_raw_masked_SAM.png",
        },
    ]

    for fish_dir in tqdm(fish_dirs, desc="Fish"):
        out_fish_dir = args.output_root / fish_dir.name
        frame_dirs = sorted(
            [p for p in fish_dir.iterdir() if p.is_dir() and p.name.startswith("frame_")],
            key=lambda p: p.name,
        )
        if args.max_frames_per_fish > 0:
            frame_dirs = frame_dirs[: args.max_frames_per_fish]

        background_models = build_fish_background_models(
            frame_dirs=frame_dirs,
            view_specs=view_specs,
            max_samples=args.bg_samples,
        )

        for frame_dir in tqdm(frame_dirs, desc=f"{fish_dir.name} frames", leave=False):
            out_frame_dir = out_fish_dir / frame_dir.name
            out_frame_dir.mkdir(parents=True, exist_ok=True)

            for spec in view_specs:
                status = process_view(
                    predictor=predictor,
                    source_frame_dir=frame_dir,
                    output_frame_dir=out_frame_dir,
                    raw_name=spec["raw_name"],
                    prompt_mask_name=spec["prompt_name"],
                    out_mask_name=spec["out_mask"],
                    out_masked_name=spec["out_masked"],
                    max_side=args.max_side,
                    background_bgr=background_models.get(spec["raw_name"]),
                    enable_background_modeling=(not args.disable_background_modeling),
                    bg_motion_proximity_factor=args.bg_motion_proximity_factor,
                    bg_guided_min_ratio=args.bg_guided_min_ratio,
                    enable_visibility_enhance=(not args.disable_visibility_enhance),
                    negative_points=args.negative_points,
                    negative_margin_ratio=args.negative_margin_ratio,
                    prompt_dilate_kernel=args.prompt_dilate_kernel,
                    min_component_area=args.min_component_area,
                    min_overlap_ratio=args.min_overlap_ratio,
                    fallback_prompt_ratio=args.fallback_prompt_ratio,
                    distance_factor=args.distance_factor,
                    distance_floor=args.distance_floor,
                    distance_cap=args.distance_cap,
                    bridge_break_kernel=args.bridge_break_kernel,
                    overwrite=args.overwrite,
                )
                counts[status] = counts.get(status, 0) + 1

            if device == "cuda":
                torch.cuda.empty_cache()

    print("\nCompleted SAM export.")
    print("Per-view status counts:")
    for key, value in counts.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
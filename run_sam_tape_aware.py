"""
Tape-Aware Fish Segmentation using SAM (Segment Anything Model).

Segments fish from images where red and green tapes act as noise.
Pipeline: tape detection -> tape removal -> preprocessing -> SAM -> postprocessing.

Output folder: masked_images_SAM_tape_aware/
"""

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

# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Tape-aware SAM fish segmentation.")
    p.add_argument("--source-root", type=Path, default=base / "final stages clean no repeat")
    p.add_argument("--output-root", type=Path, default=base / "masked_images_SAM_tape_aware")
    p.add_argument("--checkpoint", type=Path, default=base / "sam_vit_b_01ec64.pth")
    p.add_argument("--max-side", type=int, default=1024, help="Resize longest side for SAM (0=no resize).")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--fish", type=str, default="", help="Comma-separated fish folders to process.")
    p.add_argument("--max-frames-per-fish", type=int, default=0, help="Limit frames per fish (0=all).")
    p.add_argument("--debug", action="store_true", help="Save intermediate debug images.")
    # Tape detection
    p.add_argument("--tape-dilate", type=int, default=5, help="Dilation iterations for tape mask safety margin.")
    # SAM prompting
    p.add_argument("--num-positive-points", type=int, default=7)
    p.add_argument("--num-negative-points", type=int, default=14)
    p.add_argument("--negative-margin-ratio", type=float, default=0.35)
    # Postprocessing
    p.add_argument("--morph-close-k", type=int, default=7, help="Closing kernel for mask cleanup.")
    p.add_argument("--morph-open-k", type=int, default=5, help="Opening kernel for mask cleanup.")
    p.add_argument("--min-component-area", type=int, default=300)
    p.add_argument("--bridge-break-kernel", type=int, default=7)
    p.add_argument("--distance-factor", type=float, default=0.30)
    p.add_argument("--distance-floor", type=int, default=12)
    p.add_argument("--distance-cap", type=int, default=64)
    p.add_argument("--fallback-prompt-ratio", type=float, default=0.20)
    # Background modeling
    p.add_argument("--bg-samples", type=int, default=80)
    return p.parse_args()


def ensure_checkpoint(path: Path) -> None:
    if path.exists():
        return
    print(f"Downloading SAM checkpoint to {path} ...")
    path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(CHECKPOINT_URL, str(path))

# ---------------------------------------------------------------------------
#  1. Tape Detection (HSV-based red & green)
# ---------------------------------------------------------------------------

def detect_tape_mask(bgr: np.ndarray, dilate_iters: int = 5) -> np.ndarray:
    """Detect red and green tape regions via HSV thresholding.

    Returns a binary uint8 mask (255 = tape).
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Red tape: wraps around hue=0/180
    red_lo1 = np.array([0, 70, 50], dtype=np.uint8)
    red_hi1 = np.array([10, 255, 255], dtype=np.uint8)
    red_lo2 = np.array([170, 70, 50], dtype=np.uint8)
    red_hi2 = np.array([180, 255, 255], dtype=np.uint8)
    red_mask = cv2.inRange(hsv, red_lo1, red_hi1) | cv2.inRange(hsv, red_lo2, red_hi2)

    # Green tape
    green_lo = np.array([35, 50, 40], dtype=np.uint8)
    green_hi = np.array([85, 255, 255], dtype=np.uint8)
    green_mask = cv2.inRange(hsv, green_lo, green_hi)

    tape = red_mask | green_mask

    # Conservative morphology: close small gaps, then dilate to create safety buffer
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tape = cv2.morphologyEx(tape, cv2.MORPH_CLOSE, kernel_close)

    if dilate_iters > 0:
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        tape = cv2.dilate(tape, kernel_dilate, iterations=dilate_iters)

    return tape

# ---------------------------------------------------------------------------
#  2. Tape Removal (inpainting, conservative)
# ---------------------------------------------------------------------------

def remove_tape(bgr: np.ndarray, tape_mask: np.ndarray, prompt_mask: np.ndarray | None = None) -> np.ndarray:
    """Remove tape from image via inpainting.

    If a prompt_mask (fish region) is provided, we protect fish pixels from
    being inpainted by excluding the prompt interior from the tape mask.
    """
    inpaint_mask = tape_mask.copy()

    # Protect fish pixels: don't inpaint pixels that are clearly fish
    if prompt_mask is not None:
        fish_core = cv2.erode(
            (prompt_mask > 127).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=2,
        )
        inpaint_mask = cv2.bitwise_and(inpaint_mask, cv2.bitwise_not(fish_core))

    if inpaint_mask.max() == 0:
        return bgr.copy()

    # Telea inpainting for clean background fill
    cleaned = cv2.inpaint(bgr, inpaint_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    return cleaned

# ---------------------------------------------------------------------------
#  3. Preprocessing (Gaussian blur + CLAHE)
# ---------------------------------------------------------------------------

def preprocess_image(bgr: np.ndarray) -> np.ndarray:
    """Apply Gaussian blur + CLAHE contrast enhancement."""
    # Light Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(bgr, (3, 3), 0)

    # CLAHE on L channel in LAB space
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_ch = clahe.apply(l_ch)
    enhanced = cv2.merge((l_ch, a_ch, b_ch))
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # Light sharpening
    blur_for_sharp = cv2.GaussianBlur(enhanced_bgr, (0, 0), 1.5)
    sharpened = cv2.addWeighted(enhanced_bgr, 1.3, blur_for_sharp, -0.3, 0)
    return sharpened

# ---------------------------------------------------------------------------
#  4. Resize for SAM
# ---------------------------------------------------------------------------

def resize_for_sam(
    bgr: np.ndarray, max_side: int
) -> tuple[np.ndarray, float]:
    """Resize so longest side ≈ max_side. Returns (resized, scale_factor)."""
    if max_side <= 0:
        return bgr, 1.0
    h, w = bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return bgr, 1.0
    scale = max_side / float(longest)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA), scale


def scale_mask_back(mask: np.ndarray, orig_hw: tuple[int, int]) -> np.ndarray:
    """Resize binary mask back to original resolution."""
    return cv2.resize(mask, (orig_hw[1], orig_hw[0]), interpolation=cv2.INTER_NEAREST)

# ---------------------------------------------------------------------------
#  5. ROI / Bounding Box
# ---------------------------------------------------------------------------

def get_fish_bbox(prompt_mask: np.ndarray, pad_ratio: float = 0.10) -> np.ndarray | None:
    """Get bounding box from prompt mask with padding. Returns xyxy float32 or None."""
    contours, _ = cv2.findContours(
        (prompt_mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    img_h, img_w = prompt_mask.shape[:2]
    pad_x = int(w * pad_ratio)
    pad_y = int(h * pad_ratio)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img_w, x + w + pad_x)
    y2 = min(img_h, y + h + pad_y)
    return np.array([x1, y1, x2, y2], dtype=np.float32)

# ---------------------------------------------------------------------------
#  6. SAM Segmentation with fallback
# ---------------------------------------------------------------------------

def _mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def _odd(k: int, minimum: int = 3) -> int:
    k = max(minimum, int(k))
    return k if k % 2 == 1 else k + 1


def _deduplicate_pts(pts: list[tuple[float, float]], min_d: float = 3.0) -> list[tuple[float, float]]:
    kept: list[tuple[float, float]] = []
    for x, y in pts:
        if all((x - kx) ** 2 + (y - ky) ** 2 >= min_d ** 2 for kx, ky in kept):
            kept.append((x, y))
    return kept


def build_positive_points(
    prompt_mask: np.ndarray, n_points: int
) -> list[tuple[float, float]]:
    """Generate positive prompt points from the prompt mask interior."""
    bin_mask = (prompt_mask > 127).astype(np.uint8)
    ys, xs = np.where(bin_mask > 0)
    if xs.size == 0:
        return []

    pts: list[tuple[float, float]] = []

    # Centroid
    pts.append((float(xs.mean()), float(ys.mean())))

    # Extremes (left, right, top, bottom)
    pts.append((float(xs[np.argmin(xs)]), float(ys[np.argmin(xs)])))
    pts.append((float(xs[np.argmax(xs)]), float(ys[np.argmax(xs)])))
    pts.append((float(xs[np.argmin(ys)]), float(ys[np.argmin(ys)])))
    pts.append((float(xs[np.argmax(ys)]), float(ys[np.argmax(ys)])))

    # Add midpoint-extremes for more coverage
    mid_x, mid_y = float(xs.mean()), float(ys.mean())
    left_half = xs < mid_x
    right_half = xs >= mid_x
    if left_half.any():
        lx, ly = xs[left_half], ys[left_half]
        pts.append((float(lx.mean()), float(ly.mean())))
    if right_half.any():
        rx, ry = xs[right_half], ys[right_half]
        pts.append((float(rx.mean()), float(ry.mean())))

    pts = _deduplicate_pts(pts)
    return pts[:n_points]


def build_negative_points(
    prompt_mask: np.ndarray,
    tape_mask: np.ndarray,
    bbox: np.ndarray,
    n_points: int,
    margin_ratio: float,
) -> list[tuple[float, float]]:
    """Generate negative prompt points: tape regions + background edges."""
    h, w = prompt_mask.shape[:2]
    bin_prompt = (prompt_mask > 127).astype(np.uint8)
    bin_tape = (tape_mask > 127).astype(np.uint8) if tape_mask is not None else np.zeros_like(bin_prompt)

    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    mx = max(4, int(bw * margin_ratio))
    my = max(4, int(bh * margin_ratio))

    ex1, ey1 = max(0, x1 - mx), max(0, y1 - my)
    ex2, ey2 = min(w - 1, x2 + mx), min(h - 1, y2 + my)

    neg_pts: list[tuple[float, float]] = []

    # 1) Points ON tape that are NOT fish
    tape_not_fish = cv2.bitwise_and(bin_tape, cv2.bitwise_not(bin_prompt))
    tape_ys, tape_xs = np.where(tape_not_fish > 0)
    if tape_xs.size > 0:
        n_tape = min(n_points // 2, tape_xs.size)
        indices = np.linspace(0, tape_xs.size - 1, n_tape, dtype=int)
        for idx in indices:
            neg_pts.append((float(tape_xs[idx]), float(tape_ys[idx])))

    # 2) Edge points around expanded bbox
    cx, cy = (ex1 + ex2) // 2, (ey1 + ey2) // 2
    edge_pts = [
        (ex1, ey1), (cx, ey1), (ex2, ey1),
        (ex1, cy), (ex2, cy),
        (ex1, ey2), (cx, ey2), (ex2, ey2),
    ]
    for px, py in edge_pts:
        px = int(np.clip(px, 0, w - 1))
        py = int(np.clip(py, 0, h - 1))
        if bin_prompt[py, px] == 0:
            neg_pts.append((float(px), float(py)))

    # 3) Ring sampling if we need more
    if len(neg_pts) < n_points:
        inner = cv2.dilate(bin_prompt, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
        ring_k = _odd(int(0.18 * max(bw, bh)), 9)
        outer = cv2.dilate(bin_prompt, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ring_k, ring_k)), iterations=1)
        ring = np.logical_and(outer > 0, inner == 0)
        ring_ys, ring_xs = np.where(ring)
        if ring_xs.size > 0:
            step = max(1, ring_xs.size // max(1, n_points - len(neg_pts)))
            for i in range(0, ring_xs.size, step):
                neg_pts.append((float(ring_xs[i]), float(ring_ys[i])))
                if len(neg_pts) >= n_points:
                    break

    neg_pts = _deduplicate_pts(neg_pts)
    return neg_pts[:n_points]


def run_sam(
    predictor: SamPredictor,
    bgr_for_sam: np.ndarray,
    prompt_mask_sam: np.ndarray,
    tape_mask_sam: np.ndarray | None,
    bbox_sam: np.ndarray,
    n_positive: int,
    n_negative: int,
    neg_margin_ratio: float,
) -> np.ndarray:
    """Run SAM predictor with box + point prompts, with retry fallback."""

    pos_pts = build_positive_points(prompt_mask_sam, n_positive)
    neg_pts = build_negative_points(prompt_mask_sam, tape_mask_sam, bbox_sam, n_negative, neg_margin_ratio)

    all_pts = pos_pts + neg_pts
    labels = [1] * len(pos_pts) + [0] * len(neg_pts)

    point_coords = np.array(all_pts, dtype=np.float32) if all_pts else None
    point_labels = np.array(labels, dtype=np.int32) if all_pts else None

    rgb = cv2.cvtColor(bgr_for_sam, cv2.COLOR_BGR2RGB)
    predictor.set_image(rgb)

    # -- Attempt 1: box + all points --
    with torch.no_grad():
        masks, ious, _ = predictor.predict(
            box=bbox_sam,
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
    best = _select_best_mask(masks, ious, prompt_mask_sam, bbox_sam)

    # -- Fallback: if mask is too small or too large, retry with box only --
    prompt_area = float((prompt_mask_sam > 127).sum())
    best_area = float(best.sum())
    if prompt_area > 0:
        ratio = best_area / prompt_area
        if ratio < 0.15 or ratio > 4.0:
            with torch.no_grad():
                masks2, ious2, _ = predictor.predict(
                    box=bbox_sam,
                    point_coords=None,
                    point_labels=None,
                    multimask_output=True,
                )
            best2 = _select_best_mask(masks2, ious2, prompt_mask_sam, bbox_sam)
            best2_area = float(best2.sum())
            ratio2 = best2_area / prompt_area if prompt_area > 0 else 1.0
            # Pick whichever is closer to prompt area
            if abs(ratio2 - 1.0) < abs(ratio - 1.0):
                best = best2

    # -- Fallback 2: retry with only positive points --
    best_area = float(best.sum())
    if prompt_area > 0:
        ratio = best_area / prompt_area
        if ratio < 0.15 or ratio > 4.0:
            pos_only_coords = np.array(pos_pts, dtype=np.float32) if pos_pts else None
            pos_only_labels = np.array([1] * len(pos_pts), dtype=np.int32) if pos_pts else None
            with torch.no_grad():
                masks3, ious3, _ = predictor.predict(
                    box=bbox_sam,
                    point_coords=pos_only_coords,
                    point_labels=pos_only_labels,
                    multimask_output=True,
                )
            best3 = _select_best_mask(masks3, ious3, prompt_mask_sam, bbox_sam)
            best3_area = float(best3.sum())
            ratio3 = best3_area / prompt_area if prompt_area > 0 else 1.0
            if abs(ratio3 - 1.0) < abs(ratio - 1.0):
                best = best3

    return best


def _select_best_mask(
    masks: np.ndarray, ious: np.ndarray, prompt_mask: np.ndarray, bbox: np.ndarray
) -> np.ndarray:
    """Score and pick the best SAM candidate mask."""
    prompt_bin = prompt_mask > 127
    prompt_area = float(prompt_bin.sum())
    prompt_center = _mask_centroid(prompt_mask)
    diag = float(np.hypot(prompt_mask.shape[0], prompt_mask.shape[1]))

    best_idx, best_score = 0, -1e9
    for i, cand in enumerate(masks):
        cb = cand.astype(bool)
        ca = float(cb.sum())
        if ca <= 0:
            continue

        # IoU with prompt
        inter = float(np.logical_and(cb, prompt_bin).sum())
        union = float(np.logical_or(cb, prompt_bin).sum())
        iou_prompt = inter / union if union > 0 else 0.0
        coverage = inter / prompt_area if prompt_area > 0 else 0.0
        area_ratio = ca / prompt_area if prompt_area > 0 else 1.0

        # Center proximity
        cc = _mask_centroid(cb.astype(np.uint8))
        center_score = 0.0
        if prompt_center and cc and diag > 0:
            d = np.hypot(cc[0] - prompt_center[0], cc[1] - prompt_center[1])
            center_score = 1.0 - min(1.0, d / diag)

        model_iou = float(ious[i]) if i < len(ious) else 0.0
        penalty = max(0.0, area_ratio - 2.0)

        score = (
            0.45 * model_iou
            + 0.25 * iou_prompt
            + 0.20 * coverage
            + 0.10 * center_score
            - 0.15 * min(1.0, penalty / 3.0)
        )
        if score > best_score:
            best_score = score
            best_idx = i

    return masks[best_idx].astype(np.uint8)

# ---------------------------------------------------------------------------
#  7. Postprocessing / Mask Refinement
# ---------------------------------------------------------------------------

def refine_mask(
    raw_mask: np.ndarray,
    prompt_mask: np.ndarray,
    tape_mask: np.ndarray | None,
    close_k: int,
    open_k: int,
    min_area: int,
    bridge_k: int,
    dist_factor: float,
    dist_floor: int,
    dist_cap: int,
    fallback_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Refine SAM raw mask → cleaned binary mask.

    Returns (cleaned_mask_255, raw_mask_255).
    """
    prompt_bin = (prompt_mask > 127).astype(np.uint8)
    work = raw_mask.copy()

    # -- Subtract tape from SAM mask --
    if tape_mask is not None:
        tape_bin = (tape_mask > 127).astype(np.uint8)
        # Only remove tape pixels that are NOT inside prompt (fish) region
        tape_to_remove = cv2.bitwise_and(tape_bin, cv2.bitwise_not(prompt_bin))
        work = cv2.bitwise_and(work, cv2.bitwise_not(tape_to_remove))

    # -- Bridge breaking: separate fish from tape blobs --
    bk = _odd(bridge_k)
    if bk >= 3:
        work = cv2.morphologyEx(work, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bk, bk)))

    # -- Distance clipping: restrict mask to vicinity of prompt --
    dist_map = cv2.distanceTransform((prompt_bin == 0).astype(np.uint8), cv2.DIST_L2, 3)
    bbox = get_fish_bbox(prompt_mask, pad_ratio=0.0)
    if bbox is not None:
        bw = max(1, int(bbox[2] - bbox[0]))
        bh = max(1, int(bbox[3] - bbox[1]))
        max_dist = max(dist_floor, int(max(bw, bh) * dist_factor))
        if dist_cap > 0:
            max_dist = min(max_dist, dist_cap)
        clipped = np.logical_and(work > 0, dist_map <= max_dist)
        clipped = np.logical_or(clipped, np.logical_and(work > 0, prompt_bin > 0))
        if clipped.sum() >= max(64, int(prompt_bin.sum() * 0.25)):
            work = clipped.astype(np.uint8)

    # -- Keep largest connected component near prompt --
    work_255 = (work * 255 if work.max() <= 1 else work).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (work_255 > 127).astype(np.uint8), connectivity=8
    )
    if n_labels > 1:
        dilate_k = _odd(31)
        dilated_prompt = cv2.dilate(prompt_bin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k)))
        best_label, best_score = -1, -1.0
        for lbl in range(1, n_labels):
            area = int(stats[lbl, cv2.CC_STAT_AREA])
            if area < max(min_area, int(prompt_bin.sum() * 0.01)):
                continue
            comp = (labels == lbl)
            overlap = int(np.logical_and(comp, dilated_prompt > 0).sum())
            if overlap / float(area) < 0.05:
                continue
            direct_overlap = int(np.logical_and(comp, prompt_bin > 0).sum())
            sc = direct_overlap + 0.15 * overlap + 0.01 * area
            if sc > best_score:
                best_score = sc
                best_label = lbl
        if best_label >= 0:
            work = (labels == best_label).astype(np.uint8)
        else:
            work = (work_255 > 127).astype(np.uint8)

    # -- Morphological cleanup --
    ck = _odd(close_k)
    ok = _odd(open_k)
    work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck)))
    work = cv2.morphologyEx(work, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ok, ok)))

    # -- Second distance clip after cleanup --
    if bbox is not None:
        work = np.logical_and(work > 0, dist_map <= max_dist).astype(np.uint8)
        work = np.logical_or(work > 0, np.logical_and((labels == best_label) if best_label >= 0 else work > 0, prompt_bin > 0)).astype(np.uint8)

    # -- Fallback to prompt if mask collapsed --
    prompt_area = int(prompt_bin.sum())
    if prompt_area > 0 and int(work.sum()) < max(64, int(prompt_area * fallback_ratio)):
        work = prompt_bin

    # -- Optional contour smoothing --
    contours, _ = cv2.findContours(work.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        smooth = np.zeros_like(work, dtype=np.uint8)
        for cnt in contours:
            epsilon = 0.002 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            cv2.drawContours(smooth, [approx], -1, 1, thickness=cv2.FILLED)
        work = smooth

    raw_out = (raw_mask * 255).astype(np.uint8) if raw_mask.max() <= 1 else raw_mask
    cleaned_out = (work * 255).astype(np.uint8) if work.max() <= 1 else work
    return cleaned_out, raw_out

# ---------------------------------------------------------------------------
#  8. Background Modeling (median-of-frames)
# ---------------------------------------------------------------------------

def build_background_model(image_paths: list[Path], max_samples: int) -> np.ndarray | None:
    """Build a median background model from sampled frames."""
    valid = [p for p in image_paths if p.exists()]
    if not valid:
        return None
    if max_samples > 0 and len(valid) > max_samples:
        idxs = np.linspace(0, len(valid) - 1, max_samples, dtype=np.int32)
        valid = [valid[int(i)] for i in idxs]

    frames = []
    target_hw = None
    for p in valid:
        f = cv2.imread(str(p))
        if f is None:
            continue
        if target_hw is None:
            target_hw = f.shape[:2]
        elif f.shape[:2] != target_hw:
            f = cv2.resize(f, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_AREA)
        frames.append(f)

    if not frames:
        return None
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def build_fish_backgrounds(
    frame_dirs: list[Path], view_specs: list[dict], max_samples: int
) -> dict[str, np.ndarray | None]:
    models = {}
    for spec in view_specs:
        raw = spec["raw_name"]
        paths = [d / raw for d in frame_dirs]
        models[raw] = build_background_model(paths, max_samples)
    return models


def compute_motion_map(frame_bgr: np.ndarray, bg_bgr: np.ndarray | None) -> np.ndarray:
    """Multi-channel motion map: max(gray_diff, sat_diff, val_diff)."""
    h, w = frame_bgr.shape[:2]
    if bg_bgr is None:
        return np.zeros((h, w), dtype=np.uint8)
    bg = bg_bgr
    if bg.shape[:2] != (h, w):
        bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_AREA)

    fb = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
    bb = cv2.GaussianBlur(bg, (5, 5), 0)

    gray_d = cv2.cvtColor(cv2.absdiff(fb, bb), cv2.COLOR_BGR2GRAY)
    fh = cv2.cvtColor(fb, cv2.COLOR_BGR2HSV)
    bh = cv2.cvtColor(bb, cv2.COLOR_BGR2HSV)
    sat_d = cv2.absdiff(fh[:, :, 1], bh[:, :, 1])
    val_d = cv2.absdiff(fh[:, :, 2], bh[:, :, 2])

    motion = cv2.max(gray_d, cv2.max(sat_d, val_d))
    return cv2.GaussianBlur(motion, (5, 5), 0)

# ---------------------------------------------------------------------------
#  9. Output writing
# ---------------------------------------------------------------------------

def overlay_mask(bgr: np.ndarray, mask: np.ndarray, color: tuple = (0, 255, 0), alpha: float = 0.35) -> np.ndarray:
    """Create a semi-transparent overlay of the mask on the image."""
    out = bgr.copy()
    fg = mask > 127
    overlay = out.copy()
    overlay[fg] = color
    cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
    return out


def apply_mask(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply binary mask: black background where mask == 0."""
    out = np.zeros_like(bgr)
    fg = mask > 127
    out[fg] = bgr[fg]
    return out

# ---------------------------------------------------------------------------
#  10. Per-view pipeline
# ---------------------------------------------------------------------------

def process_view(
    predictor: SamPredictor,
    source_dir: Path,
    output_dir: Path,
    raw_name: str,
    prompt_name: str,
    out_mask_name: str,
    out_masked_name: str,
    background_bgr: np.ndarray | None,
    args: argparse.Namespace,
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_path = output_dir / out_mask_name
    masked_path = output_dir / out_masked_name

    if not args.overwrite and mask_path.exists() and masked_path.exists():
        return "skipped"

    raw_path = source_dir / raw_name
    prompt_path = source_dir / prompt_name
    if not raw_path.exists() or not prompt_path.exists():
        return "missing_inputs"

    raw_bgr = cv2.imread(str(raw_path))
    prompt_mask = cv2.imread(str(prompt_path), cv2.IMREAD_GRAYSCALE)
    if raw_bgr is None or prompt_mask is None:
        return "read_failed"

    orig_hw = raw_bgr.shape[:2]

    # --- Step 1: Detect tape ---
    tape_mask = detect_tape_mask(raw_bgr, dilate_iters=args.tape_dilate)

    # --- Step 2: Remove tape from image ---
    cleaned_bgr = remove_tape(raw_bgr, tape_mask, prompt_mask)

    # --- Step 3: Preprocess ---
    preprocessed = preprocess_image(cleaned_bgr)

    # --- Step 4: Get fish bbox from prompt ---
    bbox = get_fish_bbox(prompt_mask, pad_ratio=0.10)
    if bbox is None:
        return "no_prompt_contour"

    # --- Step 5: Resize for SAM ---
    sam_bgr, scale = resize_for_sam(preprocessed, args.max_side)
    sam_bbox = bbox * scale if scale != 1.0 else bbox

    # Resize prompt and tape masks for SAM resolution
    sam_h, sam_w = sam_bgr.shape[:2]
    prompt_sam = cv2.resize(prompt_mask, (sam_w, sam_h), interpolation=cv2.INTER_NEAREST) if scale != 1.0 else prompt_mask
    tape_sam = cv2.resize(tape_mask, (sam_w, sam_h), interpolation=cv2.INTER_NEAREST) if scale != 1.0 else tape_mask

    # --- Step 6: SAM segmentation ---
    sam_raw = run_sam(
        predictor, sam_bgr, prompt_sam, tape_sam, sam_bbox,
        args.num_positive_points, args.num_negative_points, args.negative_margin_ratio,
    )

    # --- Step 7: Refine mask ---
    cleaned_mask, raw_sam_mask = refine_mask(
        raw_mask=sam_raw,
        prompt_mask=prompt_sam,
        tape_mask=tape_sam,
        close_k=args.morph_close_k,
        open_k=args.morph_open_k,
        min_area=args.min_component_area,
        bridge_k=args.bridge_break_kernel,
        dist_factor=args.distance_factor,
        dist_floor=args.distance_floor,
        dist_cap=args.distance_cap,
        fallback_ratio=args.fallback_prompt_ratio,
    )

    # --- Scale back to original resolution ---
    if scale != 1.0:
        cleaned_mask = scale_mask_back(cleaned_mask, orig_hw)
        raw_sam_mask = scale_mask_back(raw_sam_mask, orig_hw)
        tape_sam_full = tape_mask  # already at original res
    else:
        tape_sam_full = tape_mask

    # --- Final tape subtraction at full resolution (safety net) ---
    fish_core = cv2.erode(
        (prompt_mask > 127).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    tape_outside_fish = cv2.bitwise_and(
        (tape_sam_full > 127).astype(np.uint8),
        cv2.bitwise_not(fish_core),
    )
    cleaned_mask_bin = (cleaned_mask > 127).astype(np.uint8)
    cleaned_mask_bin = cv2.bitwise_and(cleaned_mask_bin, cv2.bitwise_not(tape_outside_fish))
    # Restore any fish-core pixels that were accidentally removed
    cleaned_mask_bin = np.logical_or(cleaned_mask_bin > 0, np.logical_and(cleaned_mask > 127, fish_core > 0)).astype(np.uint8)
    cleaned_mask = (cleaned_mask_bin * 255).astype(np.uint8)

    # --- Generate outputs ---
    segmented_fish = apply_mask(raw_bgr, cleaned_mask)

    cv2.imwrite(str(mask_path), cleaned_mask)
    cv2.imwrite(str(masked_path), segmented_fish)

    # --- Debug outputs ---
    if args.debug:
        debug_dir = output_dir / "debug"
        debug_dir.mkdir(exist_ok=True)
        prefix = raw_name.replace("_raw.jpg", "")
        cv2.imwrite(str(debug_dir / f"{prefix}_tape_mask.png"), tape_mask)
        cv2.imwrite(str(debug_dir / f"{prefix}_tape_removed.jpg"), cleaned_bgr)
        cv2.imwrite(str(debug_dir / f"{prefix}_preprocessed.jpg"), preprocessed)
        cv2.imwrite(str(debug_dir / f"{prefix}_sam_raw.png"), raw_sam_mask)
        cv2.imwrite(str(debug_dir / f"{prefix}_cleaned_mask.png"), cleaned_mask)
        overlay = overlay_mask(raw_bgr, cleaned_mask)
        cv2.imwrite(str(debug_dir / f"{prefix}_overlay.jpg"), overlay)

    return "processed"

# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    ensure_checkpoint(args.checkpoint)

    if not args.source_root.exists():
        print(f"Source root not found: {args.source_root}")
        raise SystemExit(1)

    args.output_root.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {torch.cuda.get_device_name(0)} ({props.total_memory / 1e9:.1f} GB)")
    else:
        print("Running on CPU (slower).")

    sam = sam_model_registry["vit_b"](checkpoint=str(args.checkpoint))
    sam.to(device=device)
    predictor = SamPredictor(sam)

    fish_dirs = sorted(
        [p for p in args.source_root.iterdir() if p.is_dir() and p.name.lower().startswith("fish")],
        key=lambda p: p.name,
    )
    if args.fish:
        keep = {s.strip() for s in args.fish.split(",") if s.strip()}
        fish_dirs = [d for d in fish_dirs if d.name in keep]

    if not fish_dirs:
        print("No fish folders found.")
        raise SystemExit(1)

    view_specs = [
        {"raw_name": "01_top_raw.jpg",   "prompt_name": "40_top_final_mask.png",
         "out_mask": "01_top_SAM_mask.png",  "out_masked": "01_top_raw_masked_SAM.png"},
        {"raw_name": "21_front_raw.jpg",  "prompt_name": "41_front_final_mask.png",
         "out_mask": "21_front_SAM_mask.png", "out_masked": "21_front_raw_masked_SAM.png"},
    ]

    print(f"Source : {args.source_root}")
    print(f"Output : {args.output_root}")
    print(f"Fish   : {len(fish_dirs)}")
    print(f"Debug  : {'ON' if args.debug else 'OFF'}")

    counts: dict[str, int] = {}

    for fish_dir in tqdm(fish_dirs, desc="Fish"):
        out_fish = args.output_root / fish_dir.name
        frame_dirs = sorted(
            [p for p in fish_dir.iterdir() if p.is_dir() and p.name.startswith("frame_")],
            key=lambda p: p.name,
        )
        if args.max_frames_per_fish > 0:
            frame_dirs = frame_dirs[:args.max_frames_per_fish]

        # Build per-view background models
        bg_models = build_fish_backgrounds(frame_dirs, view_specs, args.bg_samples)

        for frame_dir in tqdm(frame_dirs, desc=f"{fish_dir.name}", leave=False):
            out_frame = out_fish / frame_dir.name
            for spec in view_specs:
                status = process_view(
                    predictor=predictor,
                    source_dir=frame_dir,
                    output_dir=out_frame,
                    raw_name=spec["raw_name"],
                    prompt_name=spec["prompt_name"],
                    out_mask_name=spec["out_mask"],
                    out_masked_name=spec["out_masked"],
                    background_bgr=bg_models.get(spec["raw_name"]),
                    args=args,
                )
                counts[status] = counts.get(status, 0) + 1

            if device == "cuda":
                torch.cuda.empty_cache()

    print("\n=== Tape-Aware SAM Export Complete ===")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

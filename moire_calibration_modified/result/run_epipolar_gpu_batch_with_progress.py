import argparse
import math
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import kornia as K
from kornia.feature import LoFTR
from tqdm import tqdm
from openpyxl import Workbook
from openpyxl.utils import get_column_letter

# ============================================================
# Filename patterns (dataset naming convention)
# ============================================================
PAIR_RE = re.compile(
    r"^(?P<idx>\d{4})_(?P<base>gt|moire)_(?P<augtype>blackbox|rotation|translation)_(?P<k>\d+)\.jpg$",
    re.IGNORECASE
)
ORIG_RE = re.compile(r"^(?P<idx>\d{4})_(?P<base>gt|moire)\.jpg$", re.IGNORECASE)


# ============================================================
# File discovery
# ============================================================
def find_originals(test_dir: Path) -> Dict[Tuple[str, str], Path]:
    mapping: Dict[Tuple[str, str], Path] = {}
    for p in test_dir.iterdir():
        if p.is_file():
            m = ORIG_RE.match(p.name)
            if m:
                mapping[(m.group("idx"), m.group("base").lower())] = p
    return mapping


def find_augmented(aug_dir: Path) -> List[Tuple[str, str, str, str, Path]]:
    items = []
    for p in aug_dir.iterdir():
        if p.is_file():
            m = PAIR_RE.match(p.name)
            if m:
                items.append((
                    m.group("idx"),
                    m.group("base").lower(),
                    m.group("augtype").lower(),
                    m.group("k"),
                    p
                ))
    items.sort(key=lambda t: (t[0], t[1], t[2], int(t[3])))
    return items


# ============================================================
# Utilities
# ============================================================
def autosize_columns(ws):
    for col in ws.columns:
        max_len = 0
        col_letter = get_column_letter(col[0].column)
        for cell in col:
            if cell.value is not None:
                max_len = max(max_len, len(str(cell.value)))
        ws.column_dimensions[col_letter].width = min(max_len + 2, 60)


def read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(str(path))
    return img


def resize_long_side(img: np.ndarray, max_side: int = 640) -> np.ndarray:
    """
    Resize image so that max(H,W) == max_side (keeping aspect ratio).
    If already smaller than max_side, return as-is.
    """
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1.0:
        return img
    new_w = int(round(w * s))
    new_h = int(round(h * s))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def to_torch_gray01(img: np.ndarray, device: torch.device) -> torch.Tensor:
    # (H,W) uint8 -> (1,1,H,W) float32 in [0,1]
    return torch.from_numpy(img).to(device=device, dtype=torch.float32).div(255.0).unsqueeze(0).unsqueeze(0)


# ============================================================
# Epipolar geometry helpers (GPU)
# ============================================================
def sampson_distance_torch(F: torch.Tensor, pts1: torch.Tensor, pts2: torch.Tensor) -> torch.Tensor:
    """
    F: (3,3)
    pts1, pts2: (N,2) pixel coords (float)
    return: (N,) Sampson distance^2
    """
    ones = torch.ones((pts1.shape[0], 1), device=pts1.device, dtype=pts1.dtype)
    x1 = torch.cat([pts1, ones], dim=1)  # Nx3
    x2 = torch.cat([pts2, ones], dim=1)  # Nx3

    Fx1 = (F @ x1.t()).t()       # Nx3
    Ftx2 = (F.t() @ x2.t()).t()  # Nx3
    x2tFx1 = torch.sum(x2 * Fx1, dim=1)  # N

    denom = Fx1[:, 0]**2 + Fx1[:, 1]**2 + Ftx2[:, 0]**2 + Ftx2[:, 1]**2
    denom = torch.clamp(denom, min=1e-12)
    return (x2tFx1**2) / denom


def rms_sampson_px(F: torch.Tensor, pts1: torch.Tensor, pts2: torch.Tensor) -> float:
    d2 = sampson_distance_torch(F, pts1, pts2)
    return float(torch.sqrt(torch.mean(d2)).item())


def rotation_angle_deg(R: np.ndarray) -> float:
    tr = float(np.trace(R))
    c = (tr - 1.0) / 2.0
    c = max(-1.0, min(1.0, c))
    return float(math.degrees(math.acos(c)))


# ============================================================
# GPU RANSAC for Fundamental matrix (Kornia)
# ============================================================
def estimate_F_ransac_opencv(mkpts0: torch.Tensor, mkpts1: torch.Tensor,
                            thresh_px: float, confidence: float):
    # mkpts0/mkpts1: (N,2) on GPU -> CPU numpy
    pts0 = mkpts0.detach().cpu().numpy().astype(np.float64)
    pts1 = mkpts1.detach().cpu().numpy().astype(np.float64)

    F, mask = cv2.findFundamentalMat(
        pts0, pts1,
        method=cv2.FM_RANSAC,
        ransacReprojThreshold=thresh_px,
        confidence=confidence
    )
    if F is None or mask is None:
        raise RuntimeError("findFundamentalMat failed")

    inliers = mask.ravel().astype(bool)  # (N,)
    F_t = torch.from_numpy(F).to(device=mkpts0.device, dtype=torch.float32)
    inliers_t = torch.from_numpy(inliers).to(device=mkpts0.device)
    return F_t, inliers_t


# ============================================================
# Per-pair processing
# ============================================================
@torch.inference_mode()
def process_pair_gpu(
    loftr: LoFTR,
    orig_path: Path,
    aug_path: Path,
    device: torch.device,
    ransac_thresh_px: float,
    ransac_max_iters: int,
    ransac_conf: float,
    K_intr: Optional[np.ndarray],
    max_side: int,
    verbose: bool = False
) -> Dict:
    if verbose:
        print("    [1/4] Load & resize")

    # (1) Read + resize for speed
    g1 = resize_long_side(read_gray(orig_path), max_side=max_side)
    g2 = resize_long_side(read_gray(aug_path), max_side=max_side)

    t1 = to_torch_gray01(g1, device)
    t2 = to_torch_gray01(g2, device)

    if verbose:
        print(f"        img0: {g1.shape[1]}x{g1.shape[0]}  img1: {g2.shape[1]}x{g2.shape[0]}")
        print("    [2/4] LoFTR matching (GPU, fp16 autocast)")

    # (2) fp16 autocast to reduce VRAM + speed up
    use_amp = (device.type == "cuda")
    with torch.cuda.amp.autocast(enabled=use_amp):
        out = loftr({"image0": t1, "image1": t2})

    mkpts0 = out["keypoints0"]  # (M,2)
    mkpts1 = out["keypoints1"]

    num_matches = int(mkpts0.shape[0])
    if num_matches < 32:
        raise RuntimeError(f"Too few matches: {num_matches}")

    if verbose:
        print(f"        matches: {num_matches}")
        print("    [3/4] Fundamental matrix (GPU RANSAC)")

    # (3) F estimation via GPU RANSAC
    F, inliers = estimate_F_ransac_opencv(mkpts0, mkpts1, ransac_thresh_px, ransac_conf)

    in0 = mkpts0[inliers]
    in1 = mkpts1[inliers]
    num_inliers = int(in0.shape[0])
    if num_inliers < 16:
        raise RuntimeError(f"Too few inliers: {num_inliers}")

    if verbose:
        print(f"        inliers: {num_inliers}")
        print("    [4/4] RMS Sampson error")

    rms_px = rms_sampson_px(F, in0, in1)
    d2 = sampson_distance_torch(F, in0, in1)
    mean_px = float(torch.mean(torch.sqrt(d2)).item())
    median_px = float(torch.median(torch.sqrt(d2)).item())

    rot_deg = None
    tdir = (None, None, None)

    # Optional: recoverPose (CPU OpenCV) if K is provided
    if K_intr is not None:
        Fcpu = F.detach().cpu().numpy().astype(np.float64)
        E = K_intr.T @ Fcpu @ K_intr

        pts0 = in0.detach().cpu().numpy().astype(np.float64)
        pts1 = in1.detach().cpu().numpy().astype(np.float64)

        _, R, t, _ = cv2.recoverPose(E, pts0, pts1, K_intr)
        rot_deg = rotation_angle_deg(R)
        t = t.reshape(3)
        t = t / (np.linalg.norm(t) + 1e-12)
        tdir = (float(t[0]), float(t[1]), float(t[2]))

    Fcpu_f = F.detach().cpu().numpy()

    return {
        "num_matches": num_matches,
        "num_inliers": num_inliers,
        "inlier_ratio": float(num_inliers / max(1, num_matches)),
        "rms_reproj_px": float(rms_px),       # RMS Sampson distance (px)
        "mean_epi_px": float(mean_px),
        "median_epi_px": float(median_px),
        "rotation_angle_deg": rot_deg,
        "t_dir_x": tdir[0],
        "t_dir_y": tdir[1],
        "t_dir_z": tdir[2],
        "F_00": float(Fcpu_f[0, 0]), "F_01": float(Fcpu_f[0, 1]), "F_02": float(Fcpu_f[0, 2]),
        "F_10": float(Fcpu_f[1, 0]), "F_11": float(Fcpu_f[1, 1]), "F_12": float(Fcpu_f[1, 2]),
        "F_20": float(Fcpu_f[2, 0]), "F_21": float(Fcpu_f[2, 1]), "F_22": float(Fcpu_f[2, 2]),
        "has_K": (K_intr is not None),
        "max_side": int(max_side),
    }


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=r"C:\Users\USER\Desktop\project\calibration\moire\datasets\test")
    parser.add_argument("--aug_dir", default="augmented_9x")
    parser.add_argument("--out_xlsx", default="epipolar_gpu_results_basic.xlsx")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    # Speed/Memory controls
    parser.add_argument("--max_side", type=int, default=640, help="resize so max(H,W)=max_side (recommended: 640 or 960)")
    parser.add_argument("--empty_cache_every", type=int, default=100, help="call torch.cuda.empty_cache() every N pairs (0 to disable)")

    # RANSAC
    parser.add_argument("--ransac_thresh_px", type=float, default=1.0)
    parser.add_argument("--ransac_max_iters", type=int, default=2000)
    parser.add_argument("--ransac_conf", type=float, default=0.999)

    # Optional intrinsics
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)

    parser.add_argument("--skip_missing", action="store_true")
    parser.add_argument("--verbose_first", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    if device.type == "cuda":
        print(f"[INFO] CUDA device: {torch.cuda.get_device_name(0)}")

    K_intr = None
    if all(v is not None for v in [args.fx, args.fy, args.cx, args.cy]):
        K_intr = np.array([[args.fx, 0, args.cx],
                           [0, args.fy, args.cy],
                           [0, 0, 1]], dtype=np.float64)
        print("[INFO] Intrinsics K provided -> will run recoverPose")

    root = Path(args.root)
    aug_dir = root / args.aug_dir
    out_xlsx = Path(args.out_xlsx)
    if not out_xlsx.is_absolute():
        out_xlsx = root / out_xlsx

    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")
    if not aug_dir.exists():
        raise FileNotFoundError(f"Augmented dir not found: {aug_dir}")

    originals = find_originals(root)
    augmented = find_augmented(aug_dir)
    print(f"[INFO] Found originals: {len(originals)} | augmented pairs: {len(augmented)}")

    # LoFTR model (pretrained weights will download on first run)
    loftr = LoFTR(pretrained="outdoor").to(device).eval()

    rows: List[Dict] = []

    for i, (idx, base, augtype, k, aug_path) in enumerate(
        tqdm(augmented, desc="GPU Epipolar Calibration", unit="pair"), start=1
    ):
        orig_path = originals.get((idx, base))

        if orig_path is None:
            msg = f"MISSING ORIGINAL: expected {idx}_{base}.jpg for {aug_path.name}"
            if args.skip_missing:
                continue
            rows.append({
                "idx": idx, "base": base, "augtype": augtype, "aug_k": k,
                "orig_path": None, "aug_path": str(aug_path),
                "error": msg
            })
            continue

        t0 = time.time()
        try:
            metrics = process_pair_gpu(
                loftr=loftr,
                orig_path=orig_path,
                aug_path=aug_path,
                device=device,
                ransac_thresh_px=args.ransac_thresh_px,
                ransac_max_iters=args.ransac_max_iters,
                ransac_conf=args.ransac_conf,
                K_intr=K_intr,
                max_side=args.max_side,
                verbose=(args.verbose_first and i == 1)
            )
            elapsed = time.time() - t0

            row = {
                "idx": idx,
                "base": base,
                "augtype": augtype,
                "aug_k": k,
                "orig_path": str(orig_path),
                "aug_path": str(aug_path),
                "elapsed_sec": round(elapsed, 3),
                **metrics
            }
            rows.append(row)

        except Exception as e:
            rows.append({
                "idx": idx, "base": base, "augtype": augtype, "aug_k": k,
                "orig_path": str(orig_path), "aug_path": str(aug_path),
                "error": f"{type(e).__name__}: {e}"
            })

        # (3) Periodic cache cleanup to reduce VRAM pressure
        if device.type == "cuda" and args.empty_cache_every > 0 and (i % args.empty_cache_every == 0):
            torch.cuda.empty_cache()

    # Write Excel
    wb = Workbook()
    ws = wb.active
    ws.title = "epipolar_gpu_results"

    headers = sorted({k for r in rows for k in r.keys()})
    ws.append(headers)
    for r in rows:
        ws.append([r.get(h, None) for h in headers])

    autosize_columns(ws)
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    wb.save(str(out_xlsx))

    print(f"\n[DONE] Saved: {out_xlsx}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()

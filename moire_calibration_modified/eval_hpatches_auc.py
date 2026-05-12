"""HPatches homography estimation benchmark (Table-2 style).

This script evaluates a matcher by:
  1) predicting correspondences for HPatches image pairs
  2) estimating homography with RANSAC
  3) computing corner reprojection error (px)
  4) reporting AUC@{3,5,10}px (HPatches convention)

It is intentionally self-contained and reuses the same LoFTR/ORB wrappers
as your MoiréPose runner, but WITHOUT MoiréPose-specific parts.

Expected HPatches folder structure (official):
  <hpatches_root>/hpatches-sequences-release/<sequence_name>/
    1.ppm ... 6.ppm
    H_1_2 ... H_1_6   (3x3 homographies)

NOTE on homography direction:
  In many HPatches releases, H_1_i maps points from image i -> image 1.
  For cv2.findHomography(pts1, ptsi) which yields H: 1 -> i,
  you likely need H_gt = inv(H_1_i). Verify once with a quick sanity check.

Example:
  python eval_hpatches_auc.py --hpatches_root /path/to/hpatches-sequences-release \
      --matcher loftr --device cuda --loftr_pretrained outdoor --topk 2000
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


# -----------------------------
# Optional (GPU) matcher: LoFTR (Kornia)
# -----------------------------
try:
    import torch
    import kornia.feature as KF
    _HAS_LOFTR = True
except Exception:
    torch = None
    KF = None
    _HAS_LOFTR = False


def load_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(str(path))
    return img


# -----------------------------
# ORB matching (CPU)
# -----------------------------
def orb_match(g0: np.ndarray, g1: np.ndarray, nfeatures: int = 3000, max_matches: int = 2000):
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp0, des0 = orb.detectAndCompute(g0, None)
    kp1, des1 = orb.detectAndCompute(g1, None)

    if des0 is None or des1 is None or len(kp0) < 30 or len(kp1) < 30:
        return (np.zeros((0, 2), np.float64), np.zeros((0, 2), np.float64), np.zeros((0,), np.float64))

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des0, des1)
    matches = sorted(matches, key=lambda m: m.distance)[:max_matches]
    pts0 = np.float64([kp0[m.queryIdx].pt for m in matches]) if matches else np.zeros((0, 2), np.float64)
    pts1 = np.float64([kp1[m.trainIdx].pt for m in matches]) if matches else np.zeros((0, 2), np.float64)
    # Convert distance to a "confidence" (larger is better)
    conf = np.float64([1.0 / (m.distance + 1e-6) for m in matches]) if matches else np.zeros((0,), np.float64)
    return pts0, pts1, conf


# -----------------------------
# LoFTR matching (GPU)
# -----------------------------
class LoFTRMatcher:
    def __init__(self, device: str = "cuda", pretrained: str = "outdoor"):
        if not _HAS_LOFTR:
            raise RuntimeError("LoFTR requested but kornia/torch is not installed.")
        if device.startswith("cuda") and (not torch.cuda.is_available()):
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        self.device = torch.device(device)
        self.matcher = KF.LoFTR(pretrained=pretrained).to(self.device).eval()

    @torch.inference_mode()
    def match(self, g0: np.ndarray, g1: np.ndarray, max_side: int = 640):
        def resize_keep(img: np.ndarray, max_side_: int):
            h, w = img.shape[:2]
            s = max_side_ / max(h, w)
            if s >= 1.0:
                return img, 1.0
            out = cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
            return out, s

        g0r, s0 = resize_keep(g0, max_side)
        g1r, s1 = resize_keep(g1, max_side)

        t0 = torch.from_numpy(g0r).float()[None, None] / 255.0
        t1 = torch.from_numpy(g1r).float()[None, None] / 255.0
        t0 = t0.to(self.device)
        t1 = t1.to(self.device)

        out = self.matcher({"image0": t0, "image1": t1})
        mkpts0 = out["keypoints0"].detach().cpu().numpy()
        mkpts1 = out["keypoints1"].detach().cpu().numpy()
        conf = out.get("confidence", None)
        if conf is None:
            conf = np.ones((mkpts0.shape[0],), np.float32)
        else:
            conf = conf.detach().cpu().numpy().reshape(-1)

        # scale back to original image coordinates
        if mkpts0.shape[0] > 0:
            mkpts0[:, 0] /= s0
            mkpts0[:, 1] /= s0
            mkpts1[:, 0] /= s1
            mkpts1[:, 1] /= s1

        return mkpts0.astype(np.float64), mkpts1.astype(np.float64), conf.astype(np.float64)


def topk(pts0: np.ndarray, pts1: np.ndarray, conf: np.ndarray, k: int):
    if k <= 0 or pts0.shape[0] <= k:
        return pts0, pts1, conf
    idx = np.argsort(-conf)[:k]
    return pts0[idx], pts1[idx], conf[idx]


def estimate_h(pts0: np.ndarray, pts1: np.ndarray, ransac_thr_px: float = 3.0) -> Optional[np.ndarray]:
    if pts0.shape[0] < 4:
        return None
    H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, ransac_thr_px)
    return H


def corners(w: int, h: int) -> np.ndarray:
    # (x,y) corners in pixel coordinates
    return np.array([[0.0, 0.0], [w - 1.0, 0.0], [w - 1.0, h - 1.0], [0.0, h - 1.0]], dtype=np.float64)


def corner_reproj_error_px(H_est: np.ndarray, H_gt: np.ndarray, w: int, h: int) -> float:
    c = corners(w, h).reshape(1, 4, 2)
    p_est = cv2.perspectiveTransform(c.astype(np.float32), H_est.astype(np.float32)).reshape(4, 2)
    p_gt = cv2.perspectiveTransform(c.astype(np.float32), H_gt.astype(np.float32)).reshape(4, 2)
    return float(np.mean(np.linalg.norm(p_est - p_gt, axis=1)))


def auc_at(errors_px: np.ndarray, thr_px: float) -> float:
    """Exact AUC of the recall-vs-threshold curve, normalized to [0,100]."""
    e = np.asarray(errors_px, dtype=np.float64)
    return float(np.mean(np.clip(1.0 - (e / float(thr_px)), 0.0, 1.0)) * 100.0)


def iter_hpatches_pairs(root: Path) -> Iterable[Tuple[Path, Path, np.ndarray]]:
    """Yield (img1, img_i, H_gt_1_to_i)."""
    seq_dirs = [p for p in root.iterdir() if p.is_dir()]
    seq_dirs.sort(key=lambda p: p.name)
    for seq in seq_dirs:
        # image extension may vary; try common ones
        def img_path(i: int) -> Optional[Path]:
            for ext in ("ppm", "png", "jpg", "jpeg"):
                p = seq / f"{i}.{ext}"
                if p.exists():
                    return p
            return None

        img1 = img_path(1)
        if img1 is None:
            continue

        for i in range(2, 7):
            imgi = img_path(i)
            if imgi is None:
                continue
            H_file = seq / f"H_1_{i}"
            if not H_file.exists():
                continue
            H_1_i = np.loadtxt(str(H_file)).astype(np.float64)
            # Common HPatches convention: H_1_i maps i -> 1.
            # We need 1 -> i to compare against cv2.findHomography(pts1, ptsi).
            try:
                H_gt = np.linalg.inv(H_1_i)
            except np.linalg.LinAlgError:
                continue

            yield img1, imgi, H_gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hpatches_root", required=True, help="Path to hpatches-sequences-release")

    ap.add_argument("--matcher", choices=["orb", "loftr"], default="loftr")
    ap.add_argument("--topk", type=int, default=0, help="Keep top-K matches by confidence (0=keep all)")
    ap.add_argument("--ransac_thr_px", type=float, default=3.0)

    # ORB
    ap.add_argument("--orb_nfeatures", type=int, default=3000)
    ap.add_argument("--orb_max_matches", type=int, default=2000)

    # LoFTR
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--loftr_pretrained", default="outdoor", choices=["outdoor", "indoor"])
    ap.add_argument("--loftr_max_side", type=int, default=640)

    args = ap.parse_args()

    root = Path(args.hpatches_root)
    if not root.exists():
        raise FileNotFoundError(str(root))

    loftr = None
    if args.matcher == "loftr":
        loftr = LoFTRMatcher(device=args.device, pretrained=args.loftr_pretrained)

    errors: List[float] = []
    n_total = 0
    n_fail = 0

    for img1_path, imgi_path, H_gt in iter_hpatches_pairs(root):
        n_total += 1
        g1 = load_gray(img1_path)
        gi = load_gray(imgi_path)

        if args.matcher == "loftr":
            pts0, pts1, conf = loftr.match(g1, gi, max_side=args.loftr_max_side)
        else:
            pts0, pts1, conf = orb_match(g1, gi, nfeatures=args.orb_nfeatures, max_matches=args.orb_max_matches)

        pts0, pts1, conf = topk(pts0, pts1, conf, args.topk)

        H_est = estimate_h(pts0, pts1, ransac_thr_px=args.ransac_thr_px)
        if H_est is None:
            n_fail += 1
            errors.append(float("inf"))
            continue

        h, w = g1.shape[:2]
        errors.append(corner_reproj_error_px(H_est, H_gt, w=w, h=h))

    e = np.array(errors, dtype=np.float64)
    auc3 = auc_at(e, 3.0)
    auc5 = auc_at(e, 5.0)
    auc10 = auc_at(e, 10.0)

    print("=== HPatches Homography Estimation (AUC of corner reprojection error) ===")
    print(f"Matcher: {args.matcher}   topk={args.topk}   ransac_thr={args.ransac_thr_px}px")
    print(f"Pairs: {n_total}   Failures (no H): {n_fail}")
    print(f"AUC@3px : {auc3:.2f}")
    print(f"AUC@5px : {auc5:.2f}")
    print(f"AUC@10px: {auc10:.2f}")


if __name__ == "__main__":
    main()

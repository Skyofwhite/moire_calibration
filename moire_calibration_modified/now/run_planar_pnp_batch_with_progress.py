import argparse
import math
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
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


def resize_long_side(img: np.ndarray, max_side: int = 1280) -> Tuple[np.ndarray, float]:
    """
    Resize image so that max(H,W) == max_side (keeping aspect ratio).
    Returns resized_img, scale (resized = original * scale).
    If already smaller: scale=1.0
    """
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1.0:
        return img, 1.0
    new_w = int(round(w * s))
    new_h = int(round(h * s))
    out = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return out, s


def ensure_K_scaled(K: np.ndarray, scale: float) -> np.ndarray:
    """
    If we resized image by scale, intrinsics must be scaled too:
      fx, fy, cx, cy multiply by scale
    """
    Ks = K.copy().astype(np.float64)
    Ks[0, 0] *= scale
    Ks[1, 1] *= scale
    Ks[0, 2] *= scale
    Ks[1, 2] *= scale
    return Ks


# ============================================================
# Feature detection & matching
# ============================================================
def create_detector(detector: str):
    detector = detector.lower()
    if detector == "sift":
        if hasattr(cv2, "SIFT_create"):
            return cv2.SIFT_create()
        raise RuntimeError("SIFT not available. Install opencv-contrib-python or use ORB.")
    if detector == "orb":
        return cv2.ORB_create(nfeatures=5000)
    raise ValueError("detector must be 'orb' or 'sift'")


def match_features(detector_name: str, img0: np.ndarray, img1: np.ndarray):
    det = create_detector(detector_name)
    kp0, des0 = det.detectAndCompute(img0, None)
    kp1, des1 = det.detectAndCompute(img1, None)

    if des0 is None or des1 is None or len(kp0) < 20 or len(kp1) < 20:
        return [], [], [], 0, 0

    if detector_name == "sift":
        # FLANN + ratio test
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        knn = flann.knnMatch(des0, des1, k=2)

        good = []
        for m, n in knn:
            if m.distance < 0.75 * n.distance:
                good.append(m)
        matches = good
    else:
        # ORB
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des0, des1)
        matches = sorted(matches, key=lambda m: m.distance)[:2000]

    pts0 = np.float64([kp0[m.queryIdx].pt for m in matches]) if matches else np.zeros((0, 2), np.float64)
    pts1 = np.float64([kp1[m.trainIdx].pt for m in matches]) if matches else np.zeros((0, 2), np.float64)
    return pts0, pts1, matches, len(kp0), len(kp1)


# ============================================================
# Planar PnP (screen plane Z=0)
# ============================================================
def plane_points_from_ref_pixels(pts_ref_px: np.ndarray, sx_m_per_px: float, sy_m_per_px: float) -> np.ndarray:
    """
    Convert reference image pixel coords (u,v) to plane coordinates (X,Y,0) in meters.
    """
    X = pts_ref_px[:, 0] * sx_m_per_px
    Y = pts_ref_px[:, 1] * sy_m_per_px
    Z = np.zeros_like(X)
    return np.stack([X, Y, Z], axis=1).astype(np.float64)


def reprojection_rms_px(object_points: np.ndarray, image_points: np.ndarray,
                        rvec: np.ndarray, tvec: np.ndarray,
                        K: np.ndarray, dist: np.ndarray) -> float:
    proj, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    err = proj - image_points
    d2 = np.sum(err * err, axis=1)
    return float(np.sqrt(np.mean(d2)))


def roll_from_R(R: np.ndarray) -> float:
    """
    Define 'roll' as rotation about camera Z axis in ZYX convention.
    (This is one reasonable definition for consistency; not identical to MoiréPose's internal roll necessarily.)
    """
    # yaw-pitch-roll (ZYX): roll = atan2(R21, R22) with certain conventions
    # We’ll compute roll around X in camera frame (common robotics): roll = atan2(R[2,1], R[2,2])
    return float(math.atan2(R[2, 1], R[2, 2]))


def pose_from_planar_pnp(
    pts_ref_px: np.ndarray,
    pts_img_px: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    sx_m_per_px: float,
    sy_m_per_px: float,
    pnp_reproj_thresh_px: float,
    pnp_conf: float,
    pnp_iters: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns rvec, tvec, inlier_mask (bool array length N)
    """
    obj_pts = plane_points_from_ref_pixels(pts_ref_px, sx_m_per_px, sy_m_per_px)  # Nx3
    img_pts = pts_img_px.astype(np.float64)

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=obj_pts,
        imagePoints=img_pts,
        cameraMatrix=K,
        distCoeffs=dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
        reprojectionError=pnp_reproj_thresh_px,
        confidence=pnp_conf,
        iterationsCount=pnp_iters
    )
    if not ok or inliers is None or len(inliers) < 12:
        raise RuntimeError("solvePnPRansac failed or too few inliers")

    mask = np.zeros((obj_pts.shape[0],), dtype=bool)
    mask[inliers.flatten()] = True
    return rvec, tvec, mask


def rvec_tvec_to_pose(rvec: np.ndarray, tvec: np.ndarray):
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)
    return R, t


# ============================================================
# Per-pair processing
# ============================================================
def process_pair_planar_pnp(
    orig_path: Path,
    aug_path: Path,
    detector: str,
    K: np.ndarray,
    dist: np.ndarray,
    sx_m_per_px: float,
    sy_m_per_px: float,
    max_side: int,
    pnp_reproj_thresh_px: float,
    pnp_conf: float,
    pnp_iters: int,
    verbose: bool = False
) -> Dict:
    g0 = read_gray(orig_path)
    g1 = read_gray(aug_path)

    g0s, s0 = resize_long_side(g0, max_side=max_side)
    g1s, s1 = resize_long_side(g1, max_side=max_side)

    # We must use a single consistent scale for both images to keep "ref pixel -> plane meters" stable.
    # Use min scale (more downscale) to avoid mismatch.
    s = min(s0, s1)
    if s != s0:
        g0s = cv2.resize(g0, (int(round(g0.shape[1]*s)), int(round(g0.shape[0]*s))), interpolation=cv2.INTER_AREA)
    if s != s1:
        g1s = cv2.resize(g1, (int(round(g1.shape[1]*s)), int(round(g1.shape[0]*s))), interpolation=cv2.INTER_AREA)

    Ks = ensure_K_scaled(K, s)

    pts0, pts1, matches, n_kp0, n_kp1 = match_features(detector, g0s, g1s)
    num_matches = int(len(matches))
    if num_matches < 40:
        raise RuntimeError(f"Too few matches: {num_matches}")

    # Optional: filter obvious outliers with homography RANSAC first (stabilizes PnP in planar case)
    H, hmask = cv2.findHomography(pts0, pts1, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if hmask is None:
        raise RuntimeError("findHomography failed")
    hmask = hmask.ravel().astype(bool)
    pts0_f = pts0[hmask]
    pts1_f = pts1[hmask]
    if pts0_f.shape[0] < 30:
        raise RuntimeError(f"Too few homography inliers: {pts0_f.shape[0]}")

    # Solve PnP (plane Z=0) using ref pixels as plane coords
    rvec_aug, tvec_aug, pnp_inliers = pose_from_planar_pnp(
        pts_ref_px=pts0_f,
        pts_img_px=pts1_f,
        K=Ks,
        dist=dist,
        sx_m_per_px=sx_m_per_px / s,  # because pts0_f is in resized pixels; convert back to original px scale
        sy_m_per_px=sy_m_per_px / s,
        pnp_reproj_thresh_px=pnp_reproj_thresh_px,
        pnp_conf=pnp_conf,
        pnp_iters=pnp_iters
    )

    # Compute reprojection RMS on inliers
    obj_pts = plane_points_from_ref_pixels(pts0_f, (sx_m_per_px / s), (sy_m_per_px / s))
    rms_aug = reprojection_rms_px(obj_pts[pnp_inliers], pts1_f[pnp_inliers], rvec_aug, tvec_aug, Ks, dist)

    R_aug, t_aug = rvec_tvec_to_pose(rvec_aug, tvec_aug)
    roll_aug = roll_from_R(R_aug)

    # For "orig" pose: treat orig image as reference view of the plane.
    # Using identity mapping (pts_ref -> same pixel) makes pose estimation ill-posed without additional constraints,
    # so we define orig pose as (R=I, t=[0,0,0]) in the screen frame for delta computation convenience.
    # This is the standard way to report "pose of aug w.r.t reference screen frame".
    R_orig = np.eye(3, dtype=np.float64)
    t_orig = np.zeros((3,), dtype=np.float64)
    roll_orig = 0.0

    # Deltas (aug - orig) in screen frame
    dx, dy, dz = (t_aug - t_orig).tolist()

    # A simple "angle delta" proxy: rotation angle between R_orig and R_aug
    R_rel = R_aug @ R_orig.T
    tr = float(np.trace(R_rel))
    c = max(-1.0, min(1.0, (tr - 1.0) / 2.0))
    dtheta_deg = float(math.degrees(math.acos(c)))

    return {
        "num_kp_orig": int(n_kp0),
        "num_kp_aug": int(n_kp1),
        "num_matches": num_matches,
        "num_h_inliers": int(pts0_f.shape[0]),
        "pnp_inliers": int(np.sum(pnp_inliers)),
        "inlier_ratio": float(np.sum(pnp_inliers) / max(1, pts0_f.shape[0])),
        # MoiréPose-like outputs
        "orig_rms_reproj_px": None,                 # not meaningful here (we anchor orig as reference)
        "aug_rms_reproj_px": float(rms_aug),
        "orig_x_m": float(t_orig[0]),
        "orig_y_m": float(t_orig[1]),
        "orig_z_m": float(t_orig[2]),
        "orig_roll_rad": float(roll_orig),
        "aug_x_m": float(t_aug[0]),
        "aug_y_m": float(t_aug[1]),
        "aug_z_m": float(t_aug[2]),
        "aug_roll_rad": float(roll_aug),
        "delta_x_m": float(dx),
        "delta_y_m": float(dy),
        "delta_z_m": float(dz),
        "delta_w_angle_deg": float(dtheta_deg),
        "max_side": int(max_side),
    }


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--aug_dir", default="augmented_9x")
    parser.add_argument("--out_xlsx", default="planar_pnp_results_basic.xlsx")

    # Matching
    parser.add_argument("--detector", default="orb", choices=["orb", "sift"])
    parser.add_argument("--max_side", type=int, default=1280)

    # Intrinsics (required for metric pose)
    parser.add_argument("--fx", type=float, required=True)
    parser.add_argument("--fy", type=float, required=True)
    parser.add_argument("--cx", type=float, required=True)
    parser.add_argument("--cy", type=float, required=True)

    # Distortion (optional; default 0)
    parser.add_argument("--k1", type=float, default=0.0)
    parser.add_argument("--k2", type=float, default=0.0)
    parser.add_argument("--p1", type=float, default=0.0)
    parser.add_argument("--p2", type=float, default=0.0)
    parser.add_argument("--k3", type=float, default=0.0)

    # Screen metric scale: meters per pixel
    # If you know screen physical size and resolution:
    #   sx = screen_width_m / screen_width_px
    #   sy = screen_height_m / screen_height_px
    parser.add_argument("--sx_m_per_px", type=float, required=True)
    parser.add_argument("--sy_m_per_px", type=float, required=True)

    # PnP RANSAC
    parser.add_argument("--pnp_reproj_thresh_px", type=float, default=3.0)
    parser.add_argument("--pnp_conf", type=float, default=0.999)
    parser.add_argument("--pnp_iters", type=int, default=2000)

    parser.add_argument("--skip_missing", action="store_true")
    parser.add_argument("--verbose_first", action="store_true")
    args = parser.parse_args()

    Kmat = np.array([[args.fx, 0, args.cx],
                     [0, args.fy, args.cy],
                     [0, 0, 1]], dtype=np.float64)
    dist = np.array([args.k1, args.k2, args.p1, args.p2, args.k3], dtype=np.float64)

    root = Path(args.root)
    aug_dir = root / args.aug_dir
    out_xlsx = Path(args.out_xlsx)
    if not out_xlsx.is_absolute():
        out_xlsx = root / out_xlsx

    originals = find_originals(root)
    augmented = find_augmented(aug_dir)
    print(f"[INFO] Found originals: {len(originals)} | augmented pairs: {len(augmented)}")

    rows: List[Dict] = []
    for i, (idx, base, augtype, k, aug_path) in enumerate(
        tqdm(augmented, desc="Planar PnP Calibration", unit="pair"), start=1
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
            metrics = process_pair_planar_pnp(
                orig_path=orig_path,
                aug_path=aug_path,
                detector=args.detector,
                K=Kmat,
                dist=dist,
                sx_m_per_px=args.sx_m_per_px,
                sy_m_per_px=args.sy_m_per_px,
                max_side=args.max_side,
                pnp_reproj_thresh_px=args.pnp_reproj_thresh_px,
                pnp_conf=args.pnp_conf,
                pnp_iters=args.pnp_iters,
                verbose=(args.verbose_first and i == 1)
            )
            elapsed = time.time() - t0
            rows.append({
                "idx": idx, "base": base, "augtype": augtype, "aug_k": k,
                "orig_path": str(orig_path),
                "aug_path": str(aug_path),
                "elapsed_sec": round(elapsed, 3),
                **metrics
            })
        except Exception as e:
            rows.append({
                "idx": idx, "base": base, "augtype": augtype, "aug_k": k,
                "orig_path": str(orig_path), "aug_path": str(aug_path),
                "error": f"{type(e).__name__}: {e}"
            })

    wb = Workbook()
    ws = wb.active
    ws.title = "planar_pnp_results"

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

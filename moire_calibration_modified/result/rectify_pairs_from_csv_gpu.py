import os
import csv
import math
import argparse
from typing import Tuple, Optional

import numpy as np
import cv2
import time

# --- Optional: OpenCV CUDA (works only if your OpenCV has CUDA) ---
HAS_CV2_CUDA = hasattr(cv2, "cuda") and hasattr(cv2.cuda, "warpPerspective")

# --- PyTorch GPU warp ---
import torch
import torch.nn.functional as F


def rodrigues_rotate(v: np.ndarray, axis: np.ndarray, theta: float) -> np.ndarray:
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    v = v.reshape(3)
    return (
        v * math.cos(theta)
        + np.cross(axis, v) * math.sin(theta)
        + axis * (np.dot(axis, v)) * (1.0 - math.cos(theta))
    )


def build_camera_axes_from_pose(position_xyz: np.ndarray, roll_rad: float) -> np.ndarray:
    P = position_xyz.astype(np.float64).reshape(3)
    w = -P
    w = w / (np.linalg.norm(w) + 1e-12)

    ex = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    u0 = ex - (np.dot(ex, w) * w)
    nu0 = np.linalg.norm(u0)
    if nu0 < 1e-9:
        ey = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        u0 = ey - (np.dot(ey, w) * w)
        u0 = u0 / (np.linalg.norm(u0) + 1e-12)
    else:
        u0 = u0 / nu0

    u = rodrigues_rotate(u0, w, roll_rad)
    u = u / (np.linalg.norm(u) + 1e-12)

    v = np.cross(w, u)
    v = v / (np.linalg.norm(v) + 1e-12)

    R_sc = np.stack([u, v, w], axis=1)  # columns
    return R_sc


def homography_screen_to_image(K: np.ndarray, position_xyz: np.ndarray, roll_rad: float) -> np.ndarray:
    R_sc = build_camera_axes_from_pose(position_xyz, roll_rad)
    R_cs = R_sc.T
    C_s = position_xyz.reshape(3, 1).astype(np.float64)
    t_cs = -R_cs @ C_s

    r1 = R_cs[:, [0]]
    r2 = R_cs[:, [1]]
    Rt_plane = np.concatenate([r1, r2, t_cs], axis=1)

    H = K @ Rt_plane
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return H


def safe_imread(path: str) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        return None
    # Windows 한글 경로 대응
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def safe_imwrite(path: str, img: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ext = os.path.splitext(path)[1].lower() or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    buf.tofile(path)


def parse_K(args, img_shape: Tuple[int, int, int]) -> np.ndarray:
    h, w = img_shape[:2]
    if args.fx is not None and args.fy is not None:
        fx, fy = float(args.fx), float(args.fy)
        cx = float(args.cx) if args.cx is not None else (w / 2.0)
        cy = float(args.cy) if args.cy is not None else (h / 2.0)
    else:
        # fallback(품질↓): 반드시 실제 fx,fy 넣는 걸 권장
        fx = fy = max(w, h) * 1.2
        cx, cy = w / 2.0, h / 2.0

    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]], dtype=np.float64)


# -------------------- GPU warp via PyTorch --------------------
def warp_perspective_torch(
    img_bgr: np.ndarray,
    H: np.ndarray,
    out_hw: Tuple[int, int],
    device: torch.device,
) -> np.ndarray:
    """
    Warp img by homography H (maps source->dest) to size out_hw=(H_out, W_out).
    Uses grid_sample on GPU/CPU.
    """
    H_out, W_out = out_hw
    src = img_bgr

    # BGR uint8 -> float32 tensor (1,3,H,W) in [0,1]
    src_t = torch.from_numpy(src).to(device=device, dtype=torch.float32)
    src_t = src_t.permute(2, 0, 1).unsqueeze(0) / 255.0  # (1,3,H,W)

    H_t = torch.from_numpy(H).to(device=device, dtype=torch.float32)

    # We need a grid that maps output pixels to input coords: x_src ~ H^{-1} x_dst
    Hinv = torch.linalg.inv(H_t)

    # Create destination pixel grid (x,y,1) in pixel coords
    ys, xs = torch.meshgrid(
        torch.arange(H_out, device=device, dtype=torch.float32),
        torch.arange(W_out, device=device, dtype=torch.float32),
        indexing="ij",
    )
    ones = torch.ones_like(xs)
    dst_h = torch.stack([xs, ys, ones], dim=-1)  # (H_out,W_out,3)
    dst_h = dst_h.reshape(-1, 3).T  # (3, N)

    src_h = Hinv @ dst_h  # (3,N)
    src_x = src_h[0, :] / (src_h[2, :] + 1e-12)
    src_y = src_h[1, :] / (src_h[2, :] + 1e-12)

    # Normalize to [-1,1] for grid_sample
    H_src, W_src = src.shape[:2]
    grid_x = (src_x / (W_src - 1)) * 2 - 1
    grid_y = (src_y / (H_src - 1)) * 2 - 1
    grid = torch.stack([grid_x, grid_y], dim=1).reshape(H_out, W_out, 2)
    grid = grid.unsqueeze(0)  # (1,H_out,W_out,2)

    warped = F.grid_sample(
        src_t, grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True
    )

    # Back to uint8 BGR
    warped = (warped.clamp(0, 1) * 255.0).byte()
    warped_np = warped.squeeze(0).permute(1, 2, 0).contiguous().cpu().numpy()
    return warped_np


def warp_perspective_opencv_cuda(img_bgr: np.ndarray, H: np.ndarray, out_size_wh: Tuple[int, int]) -> np.ndarray:
    """
    Use OpenCV CUDA warpPerspective if available.
    out_size_wh = (W,H)
    """
    gpu = cv2.cuda_GpuMat()
    gpu.upload(img_bgr)
    H32 = H.astype(np.float32)
    warped_gpu = cv2.cuda.warpPerspective(gpu, H32, out_size_wh, flags=cv2.INTER_LINEAR)
    return warped_gpu.download()


def count_valid_rows(csv_path: str) -> int:
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return sum(1 for _ in reader)

def fmt_sec(sec: float) -> str:
    sec = max(0.0, float(sec))
    if sec < 60:
        return f"{sec:.1f}s"
    m = int(sec // 60)
    s = sec - m * 60
    if m < 60:
        return f"{m}m {s:.0f}s"
    h = int(m // 60)
    m2 = m - h * 60
    return f"{h}h {m2}m"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="results_basic.csv or results_patch.csv")
    ap.add_argument("--out_dir", required=True, help="output directory")
    ap.add_argument("--fx", type=float, default=None)
    ap.add_argument("--fy", type=float, default=None)
    ap.add_argument("--cx", type=float, default=None)
    ap.add_argument("--cy", type=float, default=None)
    ap.add_argument("--save_diff", action="store_true")
    ap.add_argument("--max_rows", type=int, default=None)

    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--prefer_opencv_cuda", action="store_true")
    ap.add_argument("--log_every", type=int, default=10, help="N개 처리마다 진행 로그 출력")
    args = ap.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # torch device 선택
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 전체 row 수(진행률용)
    total_rows = count_valid_rows(args.csv)
    if args.max_rows is not None:
        total_rows = min(total_rows, args.max_rows)

    summary_csv = os.path.join(out_dir, "rectify_summary.csv")
    fieldnames = [
        "row_id", "orig_path", "aug_path",
        "warp_saved_path", "diff_saved_path",
        "mean_abs_diff", "rms_abs_diff",
        "orig_rms_reproj_px", "aug_rms_reproj_px",
        "orig_x_m", "orig_y_m", "orig_z_m", "orig_roll_rad",
        "aug_x_m", "aug_y_m", "aug_z_m", "aug_roll_rad",
        "augtype", "aug_k", "patch_mode", "best_patch_index",
        "best_patch_score_rms_reproj_px",
        "warp_backend", "device",
        "status", "error"
    ]

    t0 = time.time()
    done = 0
    ok_cnt = 0
    fail_cnt = 0

    with open(args.csv, "r", newline="", encoding="utf-8") as f_in, \
         open(summary_csv, "w", newline="", encoding="utf-8") as f_out:

        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for row_id, row in enumerate(reader):
            if args.max_rows is not None and row_id >= args.max_rows:
                break

            orig_path = row.get("orig_path", "")
            aug_path = row.get("aug_path", "")
            status = "OK"
            err = ""

            try:
                orig_img = safe_imread(orig_path)
                aug_img = safe_imread(aug_path)
                if orig_img is None or aug_img is None:
                    raise FileNotFoundError("orig_img or aug_img is None (missing file or unreadable)")

                H_out, W_out = orig_img.shape[:2]
                K = parse_K(args, orig_img.shape)

                # Pose
                orig_pos = np.array([float(row["orig_x_m"]), float(row["orig_y_m"]), float(row["orig_z_m"])], dtype=np.float64)
                aug_pos  = np.array([float(row["aug_x_m"]),  float(row["aug_y_m"]),  float(row["aug_z_m"])],  dtype=np.float64)
                orig_roll = float(row["orig_roll_rad"])
                aug_roll  = float(row["aug_roll_rad"])

                Hs2i_orig = homography_screen_to_image(K, orig_pos, orig_roll)
                Hs2i_aug  = homography_screen_to_image(K, aug_pos,  aug_roll)
                H_aug_to_orig = Hs2i_orig @ np.linalg.inv(Hs2i_aug)

                base_name = os.path.splitext(os.path.basename(aug_path))[0]
                warp_path = os.path.join(out_dir, f"{row_id:06d}_{base_name}_to_orig.png")
                if os.path.exists(warp_path):
                    continue

                # backend 선택
                if args.prefer_opencv_cuda and HAS_CV2_CUDA and device.type == "cuda":
                    warped = warp_perspective_opencv_cuda(aug_img, H_aug_to_orig, (W_out, H_out))
                    warp_backend = "opencv_cuda"
                else:
                    warped = warp_perspective_torch(aug_img, H_aug_to_orig, (H_out, W_out), device)
                    warp_backend = f"torch_{device.type}"

                safe_imwrite(warp_path, warped)

                diff_path = ""
                mean_abs, rms_abs = "", ""
                if args.save_diff:
                    diff = cv2.absdiff(orig_img, warped)
                    diff_path = os.path.join(out_dir, f"{row_id:06d}_{base_name}_absdiff.png")
                    safe_imwrite(diff_path, diff)

                    diff_g = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY).astype(np.float64)
                    mean_abs = float(np.mean(diff_g))
                    rms_abs = float(np.sqrt(np.mean(diff_g ** 2)))

                writer.writerow({
                    "row_id": row_id,
                    "orig_path": orig_path,
                    "aug_path": aug_path,
                    "warp_saved_path": warp_path,
                    "diff_saved_path": diff_path,
                    "mean_abs_diff": mean_abs,
                    "rms_abs_diff": rms_abs,
                    "orig_rms_reproj_px": row.get("orig_rms_reproj_px", ""),
                    "aug_rms_reproj_px": row.get("aug_rms_reproj_px", ""),
                    "orig_x_m": row.get("orig_x_m", ""),
                    "orig_y_m": row.get("orig_y_m", ""),
                    "orig_z_m": row.get("orig_z_m", ""),
                    "orig_roll_rad": row.get("orig_roll_rad", ""),
                    "aug_x_m": row.get("aug_x_m", ""),
                    "aug_y_m": row.get("aug_y_m", ""),
                    "aug_z_m": row.get("aug_z_m", ""),
                    "aug_roll_rad": row.get("aug_roll_rad", ""),
                    "augtype": row.get("augtype", ""),
                    "aug_k": row.get("aug_k", ""),
                    "patch_mode": row.get("patch_mode", ""),
                    "best_patch_index": row.get("best_patch_index", ""),
                    "best_patch_score_rms_reproj_px": row.get("best_patch_score_rms_reproj_px", ""),
                    "warp_backend": warp_backend,
                    "device": str(device),
                    "status": status,
                    "error": err
                })
                ok_cnt += 1

            except Exception as e:
                status = "FAIL"
                err = str(e)
                fail_cnt += 1
                writer.writerow({
                    "row_id": row_id,
                    "orig_path": orig_path,
                    "aug_path": aug_path,
                    "warp_saved_path": "",
                    "diff_saved_path": "",
                    "mean_abs_diff": "",
                    "rms_abs_diff": "",
                    "orig_rms_reproj_px": row.get("orig_rms_reproj_px", ""),
                    "aug_rms_reproj_px": row.get("aug_rms_reproj_px", ""),
                    "orig_x_m": row.get("orig_x_m", ""),
                    "orig_y_m": row.get("orig_y_m", ""),
                    "orig_z_m": row.get("orig_z_m", ""),
                    "orig_roll_rad": row.get("orig_roll_rad", ""),
                    "aug_x_m": row.get("aug_x_m", ""),
                    "aug_y_m": row.get("aug_y_m", ""),
                    "aug_z_m": row.get("aug_z_m", ""),
                    "aug_roll_rad": row.get("aug_roll_rad", ""),
                    "augtype": row.get("augtype", ""),
                    "aug_k": row.get("aug_k", ""),
                    "patch_mode": row.get("patch_mode", ""),
                    "best_patch_index": row.get("best_patch_index", ""),
                    "best_patch_score_rms_reproj_px": row.get("best_patch_score_rms_reproj_px", ""),
                    "warp_backend": "",
                    "device": str(device),
                    "status": status,
                    "error": err
                })

            done += 1
            # ---- progress log ----
            if done % args.log_every == 0 or done == 1 or done == total_rows:
                elapsed = time.time() - t0
                ips = done / max(elapsed, 1e-9)  # images/sec
                remain = (total_rows - done) / max(ips, 1e-9)
                pct = 100.0 * done / max(total_rows, 1)
                print(
                    f"[{done}/{total_rows}] {pct:6.2f}% | "
                    f"elapsed={fmt_sec(elapsed)} | eta={fmt_sec(remain)} | "
                    f"speed={ips:.2f} img/s | ok={ok_cnt} fail={fail_cnt}"
                )

    total_elapsed = time.time() - t0
    print(f"\n[OK] out_dir: {out_dir}")
    print(f"[OK] summary_csv: {summary_csv}")
    print(f"[DONE] total={done}, ok={ok_cnt}, fail={fail_cnt}, time={fmt_sec(total_elapsed)}")
    print(f"[INFO] torch device: {device}, opencv_cuda_available={HAS_CV2_CUDA}")


if __name__ == "__main__":
    main()
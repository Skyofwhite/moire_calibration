"""ChArUco/ArUco ground-truth pose helpers for real camera-screen experiments.

Use fiducials only to measure ground truth (or as explicit baselines).  The
moire-based method should still run on fiducial-free screen content.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from moire_geometry import CameraIntrinsics, PoseRT


@dataclass
class CharucoConfig:
    squares_x: int = 8
    squares_y: int = 5
    square_length_m: float = 0.04
    marker_length_m: float = 0.03
    dictionary_name: str = "DICT_4X4_100"


def _require_aruco():
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("cv2.aruco is unavailable. Install opencv-contrib-python.")
    return cv2.aruco


def get_aruco_dictionary(name: str):
    aruco = _require_aruco()
    if not hasattr(aruco, name):
        raise ValueError(f"Unknown ArUco dictionary: {name}")
    return aruco.getPredefinedDictionary(getattr(aruco, name))


def create_charuco_board(cfg: CharucoConfig):
    aruco = _require_aruco()
    dictionary = get_aruco_dictionary(cfg.dictionary_name)
    if hasattr(aruco, "CharucoBoard"):
        return aruco.CharucoBoard((cfg.squares_x, cfg.squares_y), cfg.square_length_m, cfg.marker_length_m, dictionary)
    # OpenCV 4.5 legacy API
    return aruco.CharucoBoard_create(cfg.squares_x, cfg.squares_y, cfg.square_length_m, cfg.marker_length_m, dictionary)


def render_charuco_png(out_path: str | Path, cfg: CharucoConfig, size_px: Tuple[int, int] = (1920, 1080)) -> None:
    board = create_charuco_board(cfg)
    if hasattr(board, "generateImage"):
        img = board.generateImage(size_px)
    else:
        img = board.draw(size_px)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def estimate_charuco_pose(image_path: str | Path, intr: CameraIntrinsics, cfg: CharucoConfig) -> Tuple[PoseRT, dict]:
    aruco = _require_aruco()
    board = create_charuco_board(cfg)
    dictionary = get_aruco_dictionary(cfg.dictionary_name)
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detector_params = aruco.DetectorParameters()
    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(dictionary, detector_params)
        marker_corners, marker_ids, rejected = detector.detectMarkers(gray)
    else:
        marker_corners, marker_ids, rejected = aruco.detectMarkers(gray, dictionary, parameters=detector_params)

    if marker_ids is None or len(marker_ids) == 0:
        raise RuntimeError("No ArUco markers detected in ChArUco image")

    # Interpolate ChArUco corners.
    _, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
        marker_corners, marker_ids, gray, board, intr.K, intr.dist
    )
    if charuco_ids is None or charuco_corners is None or len(charuco_ids) < 4:
        raise RuntimeError(f"Too few ChArUco corners: {0 if charuco_ids is None else len(charuco_ids)}")

    # Preferred OpenCV 4.8+ path: board.matchImagePoints + solvePnP.
    if hasattr(board, "matchImagePoints"):
        obj_pts, img_pts = board.matchImagePoints(charuco_corners, charuco_ids)
        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, intr.K, intr.dist, flags=cv2.SOLVEPNP_ITERATIVE)
    else:
        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.zeros((3, 1), dtype=np.float64)
        ok, rvec, tvec = aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, board, intr.K, intr.dist, rvec, tvec)

    if not ok:
        raise RuntimeError("ChArUco solvePnP failed")

    pose = PoseRT.from_arrays(rvec, tvec)
    diag = {
        "n_markers": int(len(marker_ids)),
        "n_charuco_corners": int(len(charuco_ids)),
        "image_path": str(image_path),
        "config": asdict(cfg),
    }
    return pose, diag


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--intrinsics_json", required=True)
    ap.add_argument("--out_pose_json", required=True)
    ap.add_argument("--out_diag_json", default=None)
    ap.add_argument("--render_board", default=None, help="optional output PNG path for the ChArUco board")
    ap.add_argument("--squares_x", type=int, default=8)
    ap.add_argument("--squares_y", type=int, default=5)
    ap.add_argument("--square_length_m", type=float, default=0.04)
    ap.add_argument("--marker_length_m", type=float, default=0.03)
    ap.add_argument("--dictionary_name", default="DICT_4X4_100")
    args = ap.parse_args()

    cfg = CharucoConfig(
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_length_m=args.square_length_m,
        marker_length_m=args.marker_length_m,
        dictionary_name=args.dictionary_name,
    )
    if args.render_board:
        render_charuco_png(args.render_board, cfg)

    intr = CameraIntrinsics.from_json(args.intrinsics_json)
    pose, diag = estimate_charuco_pose(args.image, intr, cfg)
    pose.to_json(args.out_pose_json)
    if args.out_diag_json:
        Path(args.out_diag_json).write_text(json.dumps(diag, indent=2), encoding="utf-8")
    print(json.dumps({"pose": asdict(pose), "diagnostics": diag}, indent=2))


if __name__ == "__main__":
    main()

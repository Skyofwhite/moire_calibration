"""Geometry utilities for camera-to-screen extrinsic pose evaluation.

This module is intentionally independent from the moiré FFT code.  It provides
unit-testable conventions for projection, ray-plane intersection, and absolute
pose metrics.  Use this for real GT experiments and for the redesigned
ray-plane object-point lifting path.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np


@dataclass
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    dist_coeffs: Optional[Tuple[float, ...]] = None

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def dist(self) -> Optional[np.ndarray]:
        if self.dist_coeffs is None:
            return None
        return np.asarray(self.dist_coeffs, dtype=np.float64).reshape(-1, 1)

    @classmethod
    def from_json(cls, path: str | Path) -> "CameraIntrinsics":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


@dataclass
class ScreenModel:
    width_px: int
    height_px: int
    pixel_pitch_m: float
    plane_z: float = 0.0

    @property
    def width_m(self) -> float:
        return float(self.width_px * self.pixel_pitch_m)

    @property
    def height_m(self) -> float:
        return float(self.height_px * self.pixel_pitch_m)

    def corner_points(self, centered: bool = True) -> np.ndarray:
        if centered:
            x0, x1 = -self.width_m / 2.0, self.width_m / 2.0
            y0, y1 = -self.height_m / 2.0, self.height_m / 2.0
        else:
            x0, x1 = 0.0, self.width_m
            y0, y1 = 0.0, self.height_m
        z = float(self.plane_z)
        return np.array([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]], dtype=np.float64)


@dataclass
class PoseRT:
    """OpenCV-style world(screen)-to-camera pose: X_c = R_cw X_w + t_cw."""

    rvec: Tuple[float, float, float]
    tvec: Tuple[float, float, float]

    @property
    def R_cw(self) -> np.ndarray:
        R, _ = cv2.Rodrigues(np.asarray(self.rvec, dtype=np.float64).reshape(3, 1))
        return R

    @property
    def t_cw(self) -> np.ndarray:
        return np.asarray(self.tvec, dtype=np.float64).reshape(3, 1)

    @property
    def R_wc(self) -> np.ndarray:
        return self.R_cw.T

    @property
    def C_w(self) -> np.ndarray:
        return (-(self.R_wc @ self.t_cw)).reshape(3)

    @classmethod
    def from_arrays(cls, rvec: np.ndarray, tvec: np.ndarray) -> "PoseRT":
        return cls(tuple(np.asarray(rvec, dtype=np.float64).reshape(3)), tuple(np.asarray(tvec, dtype=np.float64).reshape(3)))

    @classmethod
    def from_json(cls, path: str | Path) -> "PoseRT":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if "rvec" in data and "tvec" in data:
            return cls(tuple(data["rvec"]), tuple(data["tvec"]))
        raise ValueError(f"Unsupported pose JSON format: {path}")

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


def project_screen_points(points_screen: np.ndarray, pose: PoseRT, intr: CameraIntrinsics) -> np.ndarray:
    pts = np.asarray(points_screen, dtype=np.float64).reshape(-1, 3)
    proj, _ = cv2.projectPoints(
        pts,
        np.asarray(pose.rvec, dtype=np.float64).reshape(3, 1),
        np.asarray(pose.tvec, dtype=np.float64).reshape(3, 1),
        intr.K,
        intr.dist,
    )
    return proj.reshape(-1, 2)


def backproject_pixels_to_rays(pixels: np.ndarray, intr: CameraIntrinsics, undistort: bool = True) -> np.ndarray:
    px = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    if undistort and intr.dist is not None:
        und = cv2.undistortPoints(px.reshape(-1, 1, 2), intr.K, intr.dist, P=intr.K).reshape(-1, 2)
        px = und
    pts_h = np.concatenate([px, np.ones((px.shape[0], 1), dtype=np.float64)], axis=1)
    rays = (np.linalg.inv(intr.K) @ pts_h.T).T
    rays /= np.linalg.norm(rays, axis=1, keepdims=True) + 1e-12
    return rays


def ray_plane_intersections(
    pixels: np.ndarray,
    pose: PoseRT,
    intr: CameraIntrinsics,
    plane_z: float = 0.0,
    undistort: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Intersect image rays with screen plane Z=plane_z in screen coordinates."""
    rays_cam = backproject_pixels_to_rays(pixels, intr, undistort=undistort)
    rays_world = (pose.R_wc @ rays_cam.T).T
    C = pose.C_w
    denom = rays_world[:, 2]
    valid = np.isfinite(denom) & (np.abs(denom) > 1e-12)
    lam = np.full((rays_world.shape[0],), np.nan, dtype=np.float64)
    lam[valid] = (float(plane_z) - C[2]) / denom[valid]
    valid = valid & np.isfinite(lam) & (lam > 0)
    X = C[None, :] + lam[:, None] * rays_world
    X[:, 2] = float(plane_z)
    X[~valid] = np.nan
    return X, valid


def rotation_error_deg(pose_est: PoseRT, pose_gt: PoseRT) -> float:
    R_rel = pose_est.R_cw @ pose_gt.R_cw.T
    cos_angle = (np.trace(R_rel) - 1.0) / 2.0
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    return math.degrees(math.acos(cos_angle))


def translation_error_mm(pose_est: PoseRT, pose_gt: PoseRT) -> float:
    return float(np.linalg.norm(pose_est.C_w - pose_gt.C_w) * 1000.0)


def depth_error_mm(pose_est: PoseRT, pose_gt: PoseRT, normal: Iterable[float] = (0.0, 0.0, 1.0)) -> float:
    n = np.asarray(tuple(normal), dtype=np.float64)
    n /= np.linalg.norm(n) + 1e-12
    return float(abs(np.dot(pose_est.C_w - pose_gt.C_w, n)) * 1000.0)


def lateral_error_mm(pose_est: PoseRT, pose_gt: PoseRT, normal: Iterable[float] = (0.0, 0.0, 1.0)) -> float:
    n = np.asarray(tuple(normal), dtype=np.float64)
    n /= np.linalg.norm(n) + 1e-12
    d = pose_est.C_w - pose_gt.C_w
    lateral = d - np.dot(d, n) * n
    return float(np.linalg.norm(lateral) * 1000.0)


def screen_corner_reprojection_error_px(
    pose_est: PoseRT,
    pose_gt: PoseRT,
    intr: CameraIntrinsics,
    screen: ScreenModel,
) -> dict:
    corners = screen.corner_points(centered=True)
    p_est = project_screen_points(corners, pose_est, intr)
    p_gt = project_screen_points(corners, pose_gt, intr)
    err = np.linalg.norm(p_est - p_gt, axis=1)
    return {"mean_px": float(np.mean(err)), "median_px": float(np.median(err)), "max_px": float(np.max(err))}


def pose_metric_report(pose_est: PoseRT, pose_gt: PoseRT, intr: CameraIntrinsics, screen: Optional[ScreenModel] = None) -> dict:
    out = {
        "rotation_error_deg": rotation_error_deg(pose_est, pose_gt),
        "translation_error_mm": translation_error_mm(pose_est, pose_gt),
        "depth_error_mm": depth_error_mm(pose_est, pose_gt),
        "lateral_error_mm": lateral_error_mm(pose_est, pose_gt),
    }
    if screen is not None:
        out.update({f"screen_corner_{k}": v for k, v in screen_corner_reprojection_error_px(pose_est, pose_gt, intr, screen).items()})
    return out

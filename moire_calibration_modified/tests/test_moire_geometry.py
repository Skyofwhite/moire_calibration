import math

import cv2
import numpy as np

from moire_geometry import CameraIntrinsics, PoseRT, ScreenModel, project_screen_points, ray_plane_intersections, rotation_error_deg, translation_error_mm


def test_ray_plane_roundtrip_identity_like_pose():
    intr = CameraIntrinsics(width=640, height=480, fx=500.0, fy=500.0, cx=320.0, cy=240.0)
    # Camera at z=1 looking toward screen plane z=0.  R=diag(1,-1,-1) maps world to camera with positive camera z.
    R = np.diag([1.0, -1.0, -1.0]).astype(np.float64)
    rvec, _ = cv2.Rodrigues(R)
    tvec = np.array([[0.0], [0.0], [1.0]])
    pose = PoseRT.from_arrays(rvec, tvec)

    pts_screen = np.array([[0.0, 0.0, 0.0], [0.1, 0.05, 0.0], [-0.1, -0.05, 0.0]], dtype=np.float64)
    px = project_screen_points(pts_screen, pose, intr)
    lifted, valid = ray_plane_intersections(px, pose, intr, plane_z=0.0)
    assert valid.all()
    assert np.allclose(lifted, pts_screen, atol=1e-8)


def test_pose_errors_zero_for_same_pose():
    pose = PoseRT((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    assert rotation_error_deg(pose, pose) < 1e-9
    assert translation_error_mm(pose, pose) < 1e-9


def test_screen_corners_have_four_points():
    screen = ScreenModel(width_px=100, height_px=50, pixel_pitch_m=0.001)
    corners = screen.corner_points()
    assert corners.shape == (4, 3)
    assert math.isclose(corners[:, 2].sum(), 0.0)

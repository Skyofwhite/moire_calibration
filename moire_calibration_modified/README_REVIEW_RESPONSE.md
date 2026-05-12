# Review-response code update

This modified codebase keeps the original MoiréPose-style scripts but adds the changes needed for a stronger resubmission:

## Main technical changes

1. **Reliability-aware patch selection**
   - `moirepose_calib.py` no longer selects patches from internal RMS only by default.
   - New patch score combines:
     - internal 4-POI RMS,
     - FFT peak SNR,
     - peak sharpness,
     - POI distance coefficient of variation,
     - physical infeasibility of the moiré distance model,
     - radial penalty for off-center patches.
   - Legacy RMS-only scoring remains available with `--patch_score_mode internal_rms`.

2. **Crop-dependent metric scale fix**
   - The `a` scale is now computed as `a = d0 * c / f * half_px`.
   - This removes the previous one-third-center-crop scale assumption from patch mode.

3. **Ray-plane object-point lifting**
   - `run_moirepose_hybrid_refine_gpu_with_progress_best.py` now defaults to `--object_lift ray_plane`.
   - The old Eq. (13) fronto-parallel lift is preserved for ablation with `--object_lift fronto_parallel`.

4. **Full geometry utilities**
   - New `moire_geometry.py` provides typed camera intrinsics, screen model, OpenCV-style pose, projection, ray-plane intersection, and absolute pose metrics.

5. **Real GT evaluation support**
   - New `charuco_gt.py` estimates ground-truth pose from a screen-displayed ChArUco board.
   - New `real_gt_eval.py` evaluates estimated poses against GT using rotation, translation, depth, lateral, and screen-corner reprojection errors.

## Recommended commands

### Hybrid runner with new defaults

```bash
python run_moirepose_hybrid_refine_gpu_with_progress_best.py \
  --root /path/to/data \
  --patch_mode \
  --patch_score_mode reliability \
  --object_lift ray_plane \
  --matcher loftr \
  --refine_backend torch
```

### Ablate the old Eq. (13) lifting

```bash
python run_moirepose_hybrid_refine_gpu_with_progress_best.py \
  --root /path/to/data \
  --patch_mode \
  --patch_score_mode internal_rms \
  --object_lift fronto_parallel
```

### Render / estimate ChArUco GT

```bash
python charuco_gt.py \
  --image sample/charuco_gt.png \
  --intrinsics_json sample/camera_intrinsics.json \
  --out_pose_json sample/gt_pose_camera_to_screen.json \
  --out_diag_json sample/charuco_diag.json
```

### Evaluate real GT samples

```bash
python real_gt_eval.py \
  --root /path/to/MoireScreenPose-GT \
  --est_name estimated_pose_camera_to_screen.json \
  --gt_name gt_pose_camera_to_screen.json
```

## Important reporting rule

`hybrid_rms_before_px`, `hybrid_rms_after_px`, and `rms_reproj_px` are **not** metric pose accuracy.  They are internal or correspondence-level errors.  For the revised paper, report absolute pose accuracy from `real_gt_eval.py` whenever GT is available.

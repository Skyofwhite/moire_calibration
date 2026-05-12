"""Evaluate estimated camera-to-screen poses against real GT pose JSON files.

Expected sample layout (recommended):

sample_000/
  camera_intrinsics.json
  screen_model.json            # optional
  gt_pose_camera_to_screen.json
  estimated_pose_camera_to_screen.json

The pose JSON format is the one written by moire_geometry.PoseRT:
  {"rvec": [..], "tvec": [..]}

This script intentionally reports absolute pose metrics separately from
correspondence-level RMS.  Do not use inlier RMS as metric pose accuracy.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from moire_geometry import CameraIntrinsics, PoseRT, ScreenModel, pose_metric_report


def _read_screen(path: Path) -> Optional[ScreenModel]:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return ScreenModel(**data)


def evaluate_sample(sample_dir: Path, est_name: str, gt_name: str) -> Dict:
    intr = CameraIntrinsics.from_json(sample_dir / "camera_intrinsics.json")
    screen = _read_screen(sample_dir / "screen_model.json")
    gt = PoseRT.from_json(sample_dir / gt_name)
    est = PoseRT.from_json(sample_dir / est_name)
    report = pose_metric_report(est, gt, intr, screen=screen)
    report.update({"sample_id": sample_dir.name, "sample_dir": str(sample_dir)})
    return report


def summarize(rows: List[Dict]) -> Dict:
    keys = [k for k in rows[0].keys() if k not in {"sample_id", "sample_dir"}]
    out: Dict = {"n": len(rows)}
    for k in keys:
        vals = np.array([float(r[k]) for r in rows if r.get(k) is not None and np.isfinite(float(r[k]))], dtype=np.float64)
        if vals.size == 0:
            continue
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_median"] = float(np.median(vals))
        out[f"{k}_p90"] = float(np.percentile(vals, 90))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="directory containing per-sample folders")
    ap.add_argument("--est_name", default="estimated_pose_camera_to_screen.json")
    ap.add_argument("--gt_name", default="gt_pose_camera_to_screen.json")
    ap.add_argument("--out_csv", default="real_gt_pose_metrics.csv")
    ap.add_argument("--out_summary_json", default="real_gt_pose_summary.json")
    args = ap.parse_args()

    root = Path(args.root)
    rows: List[Dict] = []
    for sample_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        required = [sample_dir / "camera_intrinsics.json", sample_dir / args.gt_name, sample_dir / args.est_name]
        if not all(p.exists() for p in required):
            continue
        rows.append(evaluate_sample(sample_dir, args.est_name, args.gt_name))

    if not rows:
        raise RuntimeError(f"No evaluable samples found under {root}")

    out_csv = Path(args.out_csv)
    if not out_csv.is_absolute():
        out_csv = root / out_csv
    headers = list(rows[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    out_summary = Path(args.out_summary_json)
    if not out_summary.is_absolute():
        out_summary = root / out_summary
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps({"csv": str(out_csv), "summary_json": str(out_summary), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()

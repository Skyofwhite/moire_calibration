import argparse
import csv
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# moirepose_calib.py 에 있는 calibrate_pair / DEVICE_PARAMS 를 사용
# (같은 폴더에 moirepose_calib.py 가 있어야 함)
from moirepose_calib import calibrate_pair, DEVICE_PARAMS


PAIR_RE = re.compile(
    r"^(?P<idx>\d{4})_(?P<base>gt|moire)_(?P<augtype>blackbox|rotation|translation)_(?P<k>\d+)\.jpg$",
    re.IGNORECASE
)

ORIG_RE = re.compile(r"^(?P<idx>\d{4})_(?P<base>gt|moire)\.jpg$", re.IGNORECASE)


def find_originals(test_dir: Path) -> Dict[Tuple[str, str], Path]:
    """
    Map: (idx, base) -> path, where base in {gt, moire}
    """
    mapping: Dict[Tuple[str, str], Path] = {}
    for p in test_dir.iterdir():
        if not p.is_file():
            continue
        m = ORIG_RE.match(p.name)
        if not m:
            continue
        idx = m.group("idx")
        base = m.group("base").lower()
        mapping[(idx, base)] = p
    return mapping


def find_augmented(aug_dir: Path) -> List[Tuple[str, str, str, str, Path]]:
    """
    Return list of tuples:
    (idx, base, augtype, k, path)
    """
    items = []
    for p in aug_dir.iterdir():
        if not p.is_file():
            continue
        m = PAIR_RE.match(p.name)
        if not m:
            continue
        idx = m.group("idx")
        base = m.group("base").lower()
        augtype = m.group("augtype").lower()
        k = m.group("k")
        items.append((idx, base, augtype, k, p))
    # sort for stable output
    items.sort(key=lambda t: (t[0], t[1], t[2], int(t[3])))
    return items


def safe_get(d: dict, path: List[str], default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def to_row(result: dict, idx: str, base: str, augtype: str, k: str,
           orig_path: Path, aug_path: Path) -> Dict:
    pr = result.get("pair_result", {})
    debug = result.get("debug", {})

    # 핵심 결과
    delta_pos = pr.get("delta_position_m", (None, None, None))
    delta_w_deg = pr.get("delta_w_angle_deg", None)

    orig_rms = safe_get(pr, ["orig", "rms_reproj_px"], None)
    aug_rms = safe_get(pr, ["aug", "rms_reproj_px"], None)

    # pose (선택적으로 저장)
    orig_pose = safe_get(pr, ["orig", "pose"], {}) or {}
    aug_pose = safe_get(pr, ["aug", "pose"], {}) or {}

    # patch mode 정보
    patch_mode = bool(debug.get("patch_mode", False))
    best_patch_idx = None
    best_patch_score = None
    chosen_patch_box = None
    if patch_mode:
        pd = safe_get(debug, ["patch_debug"], {}) or {}
        best_patch_idx = pd.get("best_patch_index", None)
        best_patch_score = pd.get("best_patch_score_rms_reproj_px", None)
        chosen_patch_box = debug.get("chosen_patch_box", None)

    row = {
        "idx": idx,
        "base": base,                 # gt or moire
        "augtype": augtype,           # blackbox/rotation/translation
        "aug_k": k,                   # 1/2/3...
        "orig_path": str(orig_path),
        "aug_path": str(aug_path),

        # RMS Reprojection Error (px)
        "orig_rms_reproj_px": orig_rms,
        "aug_rms_reproj_px": aug_rms,

        # Relative results
        "delta_x_m": delta_pos[0] if isinstance(delta_pos, (list, tuple)) and len(delta_pos) == 3 else None,
        "delta_y_m": delta_pos[1] if isinstance(delta_pos, (list, tuple)) and len(delta_pos) == 3 else None,
        "delta_z_m": delta_pos[2] if isinstance(delta_pos, (list, tuple)) and len(delta_pos) == 3 else None,
        "delta_w_angle_deg": delta_w_deg,

        # Optional: absolute poses
        "orig_x_m": orig_pose.get("x", None),
        "orig_y_m": orig_pose.get("y", None),
        "orig_z_m": orig_pose.get("z", None),
        "orig_roll_rad": orig_pose.get("roll_theta_c", None),

        "aug_x_m": aug_pose.get("x", None),
        "aug_y_m": aug_pose.get("y", None),
        "aug_z_m": aug_pose.get("z", None),
        "aug_roll_rad": aug_pose.get("roll_theta_c", None),

        # Patch mode diagnostics
        "patch_mode": patch_mode,
        "best_patch_index": best_patch_idx,
        "best_patch_score_rms_reproj_px": best_patch_score,
        "chosen_patch_box": str(chosen_patch_box) if chosen_patch_box is not None else None,
    }
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=r"C:\Users\USER\Desktop\project\calibration\moire\datasets\test",
        help="test 폴더 경로"
    )
    parser.add_argument(
        "--aug_dir",
        default="augmented_9x",
        help="root 아래 augmented 폴더명"
    )
    parser.add_argument(
        "--out_csv",
        default="moirepose_batch_results.csv",
        help="저장할 CSV 파일명(상대경로면 root 기준)"
    )
    parser.add_argument(
        "--patch_mode",
        action="store_true",
        help="5x5 패치 전수 탐색 후 최소 RMS 패치를 선택하는 모드"
    )
    parser.add_argument(
        "--skip_missing",
        action="store_true",
        help="원본이 없는 생성이미지는 스킵(기본은 에러 표시)"
    )
    args = parser.parse_args()

    root = Path(args.root)
    aug_dir = root / args.aug_dir
    out_csv = Path(args.out_csv)
    if not out_csv.is_absolute():
        out_csv = root / out_csv

    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")
    if not aug_dir.exists():
        raise FileNotFoundError(f"Augmented dir not found: {aug_dir}")

    originals = find_originals(root)
    augmented = find_augmented(aug_dir)

    rows: List[Dict] = []
    total = len(augmented)

    for t_i, (idx, base, augtype, k, aug_path) in enumerate(augmented, start=1):
        orig_path = originals.get((idx, base), None)

        if orig_path is None:
            msg = f"[{t_i}/{total}] MISSING ORIGINAL for {aug_path.name} -> expected {idx}_{base}.jpg"
            if args.skip_missing:
                print(msg + " (skip)")
                continue
            else:
                print(msg + " (will record error)")
                rows.append({
                    "idx": idx, "base": base, "augtype": augtype, "aug_k": k,
                    "orig_path": None, "aug_path": str(aug_path),
                    "error": msg
                })
                continue

        print(f"[{t_i}/{total}] RUN  orig={orig_path.name}  aug={aug_path.name}  patch_mode={args.patch_mode}")

        try:
            result = calibrate_pair(str(orig_path), str(aug_path), DEVICE_PARAMS, patch_mode=args.patch_mode)
            row = to_row(result, idx, base, augtype, k, orig_path, aug_path)
            rows.append(row)
        except Exception as e:
            rows.append({
                "idx": idx, "base": base, "augtype": augtype, "aug_k": k,
                "orig_path": str(orig_path), "aug_path": str(aug_path),
                "error": str(e)
            })
            print(f"  -> ERROR: {e}")

    # CSV 저장
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # 필드 통일(오류 row 포함)
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    fieldnames = sorted(all_keys)

    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\nSaved: {out_csv}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()

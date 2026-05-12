import argparse
import json
import math
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ============================================================
# 0) YOU MUST SET THESE PARAMETERS (from device manuals / metadata)
# ============================================================
# MoiréPose paper requires: focal length f, CFA frequency fc, screen frequency fs.
# In the paper, fc and fs are "determined from COTS manuals in advance". :contentReference[oaicite:2]{index=2}
#
# Practical defaults (YOU SHOULD OVERRIDE):
# - CFA pixel pitch c [meters] (sensor pixel size)
# - Screen pixel pitch ps [meters]
# - focal length f [meters]
#
# NOTE:
# A common approximation for Bayer CFA grating period along x/y is 2*c (because RGGB repeats every 2 pixels).
# Then fc ≈ 1 / (2*c) in cycles/m. This is a practical approximation (not guaranteed for every CFA).
# Screen grating period is typically 1 screen pixel => fs ≈ 1 / ps in cycles/m.

DEVICE_PARAMS = {
    "f_m": 4.3e-3,          # focal length in meters (example: 4.3 mm)
    "cfa_pitch_m": 1.4e-6,  # sensor pixel pitch in meters (example: 1.4 μm)
    "screen_pitch_m": 205.8e-6,  # screen pixel pitch in meters (example: 205.8 μm)
    "assume_bayer_cfa_period_2px": True,  # if True: fc = 1/(2*cfa_pitch)
}

# ============================================================
# 1) Dataclasses for outputs
# ============================================================

@dataclass
class MoireFeatures:
    # spatial freq in cycles per "center-part side length" (raw) and cycles/m (metric)
    fm_raw: float
    theta_rad: float
    fm_cyc_per_m: float
    # Review-response diagnostics.  These are intentionally kept in the
    # per-POI feature record so patch selection can be based on moiré reliability
    # rather than on the 4-point internal RMS only.
    peak_snr: float = 0.0
    peak_sharpness: float = 0.0
    peak_score: float = 0.0
    physical_infeasibility: float = 0.0
    valid: bool = True

@dataclass
class Pose6DoF:
    # camera position in screen coords (x,y,z), meters
    x: float
    y: float
    z: float
    # unit axes in screen coords
    u: Tuple[float, float, float]
    v: Tuple[float, float, float]
    w: Tuple[float, float, float]
    # roll angle (theta_c) in radians
    roll_theta_c: float

@dataclass
class CalibResult:
    image_path: str
    center_crop_box: Tuple[int, int, int, int]  # x0,y0,x1,y1 in original image
    # POI features
    features_by_poi: Dict[str, MoireFeatures]
    # distances from camera to POIs (meters)
    distances_by_poi: Dict[str, float]
    # estimated pose
    pose: Pose6DoF
    # reprojection error in pixels (internal consistency)
    rms_reproj_px: float
    # Additional diagnostics used by reliability-aware patch search.
    # NOTE: rms_reproj_px is an internal consistency score, not metric pose error.
    diagnostics: Dict = field(default_factory=dict)

@dataclass
class PairResult:
    orig: CalibResult
    aug: CalibResult
    # relative pose (aug wrt orig) in screen coords approximation
    delta_position_m: Tuple[float, float, float]
    # angle difference between w axes (deg)
    delta_w_angle_deg: float

# ============================================================
# 2) Image preprocessing (paper §5.1 inspired) :contentReference[oaicite:3]{index=3}
#    - center crop: side length = (short_side / 3)
#    - histogram eq
#    - adaptive illumination/gamma (simplified)
#    - binarization
#    - median filter
# ============================================================

def center_crop_one_third(img_bgr: np.ndarray) -> Tuple[np.ndarray, Tuple[int,int,int,int]]:
    h, w = img_bgr.shape[:2]
    short = min(h, w)
    side = short // 3
    cx, cy = w // 2, h // 2
    x0 = max(0, cx - side // 2)
    y0 = max(0, cy - side // 2)
    x1 = min(w, x0 + side)
    y1 = min(h, y0 + side)
    crop = img_bgr[y0:y1, x0:x1].copy()
    return crop, (x0, y0, x1, y1)

def preprocess_for_moire(crop_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    # histogram equalization
    eq = cv2.equalizeHist(gray)

    # simplified illumination compensation:
    # subtract a blurred version to reduce vignetting-like low-frequency shading
    blur = cv2.GaussianBlur(eq, (0, 0), sigmaX=15, sigmaY=15)
    high = cv2.addWeighted(eq, 1.5, blur, -0.5, 0)

    # adaptive threshold (binarization)
    thr = cv2.adaptiveThreshold(
        high, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        51, 2
    )

    # median filter to suppress salt-and-pepper
    med = cv2.medianBlur(thr, 5)
    return med

# ============================================================
# 3) FFT spectrum and moiré peak extraction (paper §5.2, Eq.(5)) :contentReference[oaicite:4]{index=4}
#    - compute magnitude spectrum
#    - suppress DC neighborhood
#    - find bright peaks and select dominant symmetric pair(s)
#    - derive (fm, theta)
# ============================================================

def fft_spectrum(binary_img: np.ndarray) -> np.ndarray:
    f = np.fft.fft2(binary_img.astype(np.float32))
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    mag = np.log1p(mag)
    return mag

def _suppress_dc(mag: np.ndarray, r: int = 10) -> np.ndarray:
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    out = mag.copy()
    out[cy - r: cy + r + 1, cx - r: cx + r + 1] = 0
    return out

def extract_moire_features_from_roi(roi_bin: np.ndarray, cfa_pitch_m: float) -> MoireFeatures:
    """
    Returns MoireFeatures:
    - fm_raw: sqrt(u^2+v^2) where u,v in frequency plane pixel coordinates (paper Eq.(5))
    - theta_rad: atan2(v,u)
    - fm_cyc_per_m: convert using xi = 1/(c*Rm) (paper §5.2) :contentReference[oaicite:5]{index=5}
    """
    mag = fft_spectrum(roi_bin)
    mag = _suppress_dc(mag, r=max(6, min(roi_bin.shape)//40))

    # normalize for robust thresholding
    m = mag / (mag.max() + 1e-9)

    # threshold to get candidate peaks
    thresh = np.quantile(m[m > 0], 0.995) if np.any(m > 0) else 0.0
    mask = (m >= max(thresh, 0.6)).astype(np.uint8)

    # find connected components, pick the largest bright blob(s) far from center
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        # fallback: take global maximum
        y, x = np.unravel_index(np.argmax(m), m.shape)
        peaks = [(x, y)]
    else:
        # score by area * mean intensity and distance from center
        h, w = m.shape
        cx, cy = w / 2, h / 2
        candidates = []
        for i in range(1, num):
            x, y = centroids[i]
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 5:
                continue
            # distance from DC
            dist = math.hypot(x - cx, y - cy)
            # mean intensity in component
            comp_mask = (labels == i)
            mean_int = float(m[comp_mask].mean())
            score = area * mean_int * (dist + 1.0)
            candidates.append((score, x, y))
        candidates.sort(reverse=True, key=lambda t: t[0])
        peaks = [(int(round(x)), int(round(y))) for _, x, y in candidates[:6]]

    # Choose the best peak and use its vector from center
    h, w = m.shape
    cx, cy = w // 2, h // 2

    best = None
    best_score = -1.0
    best_intensity = 0.0
    for (px, py) in peaks:
        u = px - cx
        v = py - cy
        dist = math.hypot(u, v)
        if dist < min(h, w) * 0.05:
            continue
        inten = float(m[py, px])
        score = dist * inten
        if score > best_score:
            best_score = score
            best = (u, v)
            best_intensity = inten

    if best is None:
        best = (1.0, 0.0)
        best_score = 0.0
        best_intensity = 0.0

    u, v = float(best[0]), float(best[1])
    fm_raw = math.hypot(u, v)
    theta = math.atan2(v, u)

    # Reliability diagnostics.  These are not part of MoiréPose, but they make
    # the patch search less heuristic by exposing how clear and sharp the chosen
    # FFT peak is.  The SNR is robust because it is measured against the median
    # and MAD-like spread of the non-zero normalized spectrum.
    nz = m[m > 0]
    if nz.size > 0:
        med = float(np.median(nz))
        mad = float(np.median(np.abs(nz - med))) + 1e-9
        peak_snr = float((best_intensity - med) / mad)
        # sharpness: peak height minus the local 95th-percentile background in a
        # small annulus around the selected peak.
        px = int(round(u + cx)); py = int(round(v + cy))
        yy, xx = np.ogrid[:h, :w]
        rr = np.sqrt((xx - px) ** 2 + (yy - py) ** 2)
        local = m[(rr >= 3) & (rr <= 10)]
        bg95 = float(np.percentile(local, 95)) if local.size > 0 else med
        peak_sharpness = float(max(best_intensity - bg95, 0.0))
    else:
        peak_snr = 0.0
        peak_sharpness = 0.0

    # Convert fm_raw (cycles per ROI side length) to cycles/m using xi = 1/(c*Rm) in the paper.
    # Here Rm = ROI side pixels; c = CFA pixel pitch (meters).
    Rm = float(min(roi_bin.shape[:2]))
    xi = 1.0 / (cfa_pitch_m * Rm)
    fm_cyc_per_m = fm_raw * xi

    return MoireFeatures(
        fm_raw=fm_raw,
        theta_rad=theta,
        fm_cyc_per_m=fm_cyc_per_m,
        peak_snr=peak_snr,
        peak_sharpness=peak_sharpness,
        peak_score=float(best_score),
        valid=bool(fm_raw > 0),
    )

# ============================================================
# 4) MoiréPose distance model (paper Eq.(9)) :contentReference[oaicite:6]{index=6}
# ============================================================

def compute_fc_fs(params: Dict) -> Tuple[float, float]:
    c = params["cfa_pitch_m"]
    ps = params["screen_pitch_m"]
    if params.get("assume_bayer_cfa_period_2px", True):
        fc = 1.0 / (2.0 * c)  # cycles/m
    else:
        fc = 1.0 / c
    fs = 1.0 / ps
    return fc, fs

def distance_from_moire(f_m: float, theta_m: float, f_cam: float, fc: float, fs: float) -> float:
    """
    Implements paper Eq.(9):
    d = f/fs * ( sqrt(fc^2 - (fm*sinθ)^2 ) - fm*cosθ )
    where all frequencies are cycles/m, f is meters => d in meters.
    """
    d, _penalty, _inside = distance_from_moire_diagnostic(f_m, theta_m, f_cam, fc, fs)
    return d


def distance_from_moire_diagnostic(
    f_m: float,
    theta_m: float,
    f_cam: float,
    fc: float,
    fs: float,
) -> Tuple[float, float, float]:
    """Distance model with explicit physical-feasibility diagnostics.

    The previous implementation silently clamped negative square-root arguments.
    That is numerically convenient, but it hides an important signal: a patch whose
    spectrum is inconsistent with the device frequencies should be penalized by the
    patch selector.  We still clamp for a finite distance estimate, but return a
    normalized infeasibility penalty as well.
    """
    s = f_m * math.sin(theta_m)
    c = f_m * math.cos(theta_m)
    inside_raw = fc * fc - s * s
    penalty = float(max(-inside_raw, 0.0) / (fc * fc + 1e-12))
    inside = max(inside_raw, 0.0)
    d = (f_cam / fs) * (math.sqrt(inside) - c)
    return float(d), penalty, float(inside_raw)

# ============================================================
# 5) POI / ROI selection (paper §6.2.1/6.2.2) :contentReference[oaicite:7]{index=7}
#    - Choose 4 POIs on boundary of center crop: IU, ID, IL, IR
#    - ROI centered at each POI; ROI size determined by 4*f' (Nyquist+margin)
# ============================================================

def poi_locations_on_crop(crop_shape: Tuple[int,int], margin: int = 4) -> Dict[str, Tuple[int,int]]:
    h, w = crop_shape[:2]
    cx, cy = w // 2, h // 2
    # boundary of crop
    IU = (cx, margin)
    ID = (cx, h - 1 - margin)
    IL = (margin, cy)
    IR = (w - 1 - margin, cy)
    return {"IU": IU, "ID": ID, "IL": IL, "IR": IR}

def crop_square(img: np.ndarray, center_xy: Tuple[int,int], side: int) -> np.ndarray:
    h, w = img.shape[:2]
    cx, cy = center_xy
    r = side // 2
    x0 = max(0, cx - r); x1 = min(w, cx + r)
    y0 = max(0, cy - r); y1 = min(h, cy + r)
    patch = img[y0:y1, x0:x1].copy()
    # force square by center cropping if needed
    ph, pw = patch.shape[:2]
    m = min(ph, pw)
    patch = patch[(ph-m)//2:(ph-m)//2+m, (pw-m)//2:(pw-m)//2+m]
    return patch

def estimate_roi_size_by_nyquist(bin_crop: np.ndarray, poi_xy: Tuple[int,int], cfa_pitch_m: float) -> int:
    # start with a coarse ROI
    base = max(64, min(bin_crop.shape[:2]) // 3)
    roi0 = crop_square(bin_crop, poi_xy, base)
    feat0 = extract_moire_features_from_roi(roi0, cfa_pitch_m)

    # paper: R' = 4 f' (pixels), with f' being temporal spatial freq (cycles per ROI side) :contentReference[oaicite:8]{index=8}
    # Here we use fm_raw as f' (same unit: cycles per ROI side)
    R = int(max(64, min(bin_crop.shape[:2]) - 2, 4.0 * feat0.fm_raw))
    # clamp
    R = int(np.clip(R, 64, min(bin_crop.shape[:2])))
    # keep even
    if R % 2 == 1:
        R += 1
    return R

# ============================================================
# 6) Position via multilateral optimization (paper Eq.(10)) :contentReference[oaicite:9]{index=9}
# ============================================================

def solve_position_multilateral(S: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Solve P = argmin Σ (||S_i - P|| - d_i)^2
    Simple Gauss-Newton iteration without external deps.
    """
    # initial guess: (0,0,mean(d))
    P = np.array([0.0, 0.0, float(np.mean(d))], dtype=np.float64)

    for _ in range(30):
        r = S - P[None, :]
        dist = np.linalg.norm(r, axis=1) + 1e-9
        residual = dist - d  # (n,)

        # Jacobian: ∂(dist_i)/∂P = -(S_i - P)/dist_i
        J = -(r / dist[:, None])  # (n,3)

        # GN step: (J^T J) dp = -J^T residual
        A = J.T @ J + 1e-6 * np.eye(3)
        b = -J.T @ residual
        dp = np.linalg.solve(A, b)

        P = P + dp
        if np.linalg.norm(dp) < 1e-6:
            break
    return P

# ============================================================
# 7) Posture (paper §7, Eq.(12),(13)) :contentReference[oaicite:10]{index=10}
# ============================================================

def rodrigues_rotate(v: np.ndarray, axis: np.ndarray, theta: float) -> np.ndarray:
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    v = v.astype(np.float64)
    return (v * math.cos(theta) +
            np.cross(axis, v) * math.sin(theta) +
            axis * (np.dot(axis, v)) * (1.0 - math.cos(theta)))

def estimate_roll_theta_c(theta_m: float, d0: float, f_cam: float, fc: float, fs: float) -> float:
    """
    Paper Eq.(12):
    tan(theta_m) = sin(theta_c) / (cos(theta_c) - (d0*fs)/(f*fc))
    Solve for theta_c numerically.
    """
    k = (d0 * fs) / (f_cam * fc + 1e-12)

    # solve g(tc)=0 via 1D search
    def g(tc: float) -> float:
        return math.tan(theta_m) - (math.sin(tc) / (math.cos(tc) - k + 1e-12))

    # bracket search over [-pi, pi]
    best_tc = 0.0
    best_val = 1e9
    for tc in np.linspace(-math.pi, math.pi, 721):
        val = abs(g(tc))
        if val < best_val:
            best_val = val
            best_tc = float(tc)
    # local refine
    tc = best_tc
    for _ in range(20):
        eps = 1e-5
        g0 = g(tc)
        dg = (g(tc + eps) - g(tc - eps)) / (2 * eps)
        if abs(dg) < 1e-9:
            break
        step = g0 / dg
        tc = tc - step
        if abs(step) < 1e-7:
            break
    return float(tc)

def estimate_posture_axes(P: np.ndarray, theta_c: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Paper: ŵ = OP / ||OP||.
    Define m̂ as projection of OP onto horizontal plane (y=0) and perpendicular to ŵ.
    Then û = rotate(m̂ around ŵ by theta_c), v̂ = ŵ × û. :contentReference[oaicite:11]{index=11}
    """
    w_hat = P / (np.linalg.norm(P) + 1e-12)

    # Choose m in plane y=0 and perpendicular to w: dot(m,w)=0, with my=0.
    wx, wy, wz = w_hat
    # let m = [1,0, -wx/wz] if |wz| big else [ -wz/wx,0,1]
    if abs(wz) > 1e-6:
        m = np.array([1.0, 0.0, -wx / (wz + 1e-12)])
    else:
        m = np.array([-wz / (wx + 1e-12), 0.0, 1.0])

    m_hat = m / (np.linalg.norm(m) + 1e-12)

    u_hat = rodrigues_rotate(m_hat, w_hat, theta_c)
    u_hat = u_hat / (np.linalg.norm(u_hat) + 1e-12)
    v_hat = np.cross(w_hat, u_hat)
    v_hat = v_hat / (np.linalg.norm(v_hat) + 1e-12)
    return u_hat, v_hat, w_hat

# ============================================================
# 8) Reprojection error (px)
#    - Internal consistency: project screen POIs -> expected pixel locations (in crop)
#    - Use pinhole projection, principal point at crop center, focal length in pixels
# ============================================================

def focal_px(f_m: float, cfa_pitch_m: float) -> float:
    return float(f_m / cfa_pitch_m)

def project_points_screen_to_crop_px(
    S_screen: np.ndarray,
    P_cam_screen: np.ndarray,
    u_hat: np.ndarray,
    v_hat: np.ndarray,
    w_hat: np.ndarray,
    crop_wh: Tuple[int,int],
    f_px: float
) -> np.ndarray:
    """
    Screen coords = world.
    Camera axes (u,v,w) are expressed in screen coords (world frame). Columns of R_wc.
    Convert world point to camera coords: Xc = R_cw * (S - P), where R_cw = R_wc^T.
    Then x = f*Xc.x/Xc.z + cx, y = f*Xc.y/Xc.z + cy
    """
    R_wc = np.stack([u_hat, v_hat, w_hat], axis=1)  # 3x3
    R_cw = R_wc.T
    X = (S_screen - P_cam_screen[None, :])
    Xc = (R_cw @ X.T).T  # Nx3

    w, h = crop_wh
    cx, cy = w / 2.0, h / 2.0

    # avoid divide-by-zero
    z = Xc[:, 2] + 1e-12
    x = f_px * (Xc[:, 0] / z) + cx
    y = f_px * (Xc[:, 1] / z) + cy
    return np.stack([x, y], axis=1)

def rms_reprojection_error_px(observed_px: np.ndarray, predicted_px: np.ndarray) -> float:
    """
    Standard RMS Reprojection Error (px):
    RMS = sqrt( (1/N) * sum_i ||x_i - xhat_i||^2 )
        = sqrt( mean( (du)^2 + (dv)^2 ) )
    """
    e = observed_px - predicted_px  # Nx2
    sq = (e[:, 0] ** 2) + (e[:, 1] ** 2)  # Nx
    return float(np.sqrt(np.mean(sq)))

# ============================================================
# 9) Single-image MoiréPose calibration
# ============================================================

def moirepose_calibrate_single(img_path: str, params: Dict, override_crop: Optional[Tuple[int,int,int,int]] = None) -> CalibResult:
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(img_path)

    if override_crop is None:
        crop, box = center_crop_one_third(img)
    else:
        x0, y0, x1, y1 = override_crop
        crop = img[y0:y1, x0:x1].copy()
        box = override_crop

    bin_crop = preprocess_for_moire(crop)
    pois = poi_locations_on_crop(bin_crop.shape)

    f_cam = float(params["f_m"])
    cfa_pitch = float(params["cfa_pitch_m"])
    fc, fs = compute_fc_fs(params)

    features_by_poi: Dict[str, MoireFeatures] = {}
    distances_by_poi: Dict[str, float] = {}
    distance_diagnostics_by_poi: Dict[str, Dict[str, float]] = {}

    # for each POI, determine ROI size, compute features
    for name, xy in pois.items():
        R = estimate_roi_size_by_nyquist(bin_crop, xy, cfa_pitch)
        roi = crop_square(bin_crop, xy, R)
        feat = extract_moire_features_from_roi(roi, cfa_pitch)
        d, phys_penalty, sqrt_arg = distance_from_moire_diagnostic(
            feat.fm_cyc_per_m, feat.theta_rad, f_cam, fc, fs
        )
        feat.physical_infeasibility = float(phys_penalty)
        feat.valid = bool(np.isfinite(d) and d > 0 and phys_penalty < 1.0)
        features_by_poi[name] = feat
        distances_by_poi[name] = float(d)
        distance_diagnostics_by_poi[name] = {
            "physical_infeasibility": float(phys_penalty),
            "sqrt_argument": float(sqrt_arg),
            "roi_side_px": float(R),
        }

    # Distance selection rule (paper §6.2.2):
    # - IL, IR: use horizontal spline cluster => our simplified extractor returns one dominant peak;
    # - IU, ID: use vertical spline cluster.
    # Here: we keep the same 4 distances as is (practical simplification).

    # d0: distance to virtual origin O (paper uses the screen point projected to image center).
    # Practical: use mean of 4 POI distances.
    d0 = float(np.mean(list(distances_by_poi.values())))

    # Determine "a" from the selected crop geometry.  The earlier code used
    # d0*Lc/(6f), inherited from the one-third-center-crop assumption.  In patch
    # mode that makes the object-point scale inconsistent.  We instead use the
    # paper's crop-dependent form: a = d0 * c/f * half, where half is the POI
    # effective half-size in pixels.
    crop_h, crop_w = bin_crop.shape[:2]
    margin_px = 4.0
    half_px = max((min(crop_h, crop_w) / 2.0) - margin_px, 1.0)
    a = float(d0 * cfa_pitch / (f_cam + 1e-12) * half_px)

    # screen POI coordinates (SU,SD,SL,SR)
    S = np.array([
        [0.0,  a, 0.0],   # IU -> SU
        [0.0, -a, 0.0],   # ID -> SD
        [-a, 0.0, 0.0],   # IL -> SL
        [ a, 0.0, 0.0],   # IR -> SR
    ], dtype=np.float64)
    d_arr = np.array([
        distances_by_poi["IU"],
        distances_by_poi["ID"],
        distances_by_poi["IL"],
        distances_by_poi["IR"],
    ], dtype=np.float64)

    P = solve_position_multilateral(S, d_arr)  # camera position in screen coords

    # posture
    # pick theta_m from center-ish POI to estimate roll (practical: use mean of POI angles)
    theta_m_mean = float(np.mean([features_by_poi[k].theta_rad for k in ["IU","ID","IL","IR"]]))
    theta_c = estimate_roll_theta_c(theta_m_mean, d0, f_cam, fc, fs)
    u_hat, v_hat, w_hat = estimate_posture_axes(P, theta_c)

    # Reprojection error:
    # Observed pixel locations of POIs on crop are known (IU/ID/IL/IR).
    obs = np.array([pois["IU"], pois["ID"], pois["IL"], pois["IR"]], dtype=np.float64)
    fpx = focal_px(f_cam, cfa_pitch)
    pred = project_points_screen_to_crop_px(
        S_screen=S,
        P_cam_screen=P,
        u_hat=u_hat, v_hat=v_hat, w_hat=w_hat,
        crop_wh=(crop_w, crop_h),
        f_px=fpx
    )
    rms_px = rms_reprojection_error_px(obs, pred)

    pose = Pose6DoF(
        x=float(P[0]), y=float(P[1]), z=float(P[2]),
        u=(float(u_hat[0]), float(u_hat[1]), float(u_hat[2])),
        v=(float(v_hat[0]), float(v_hat[1]), float(v_hat[2])),
        w=(float(w_hat[0]), float(w_hat[1]), float(w_hat[2])),
        roll_theta_c=float(theta_c),
    )

    return CalibResult(
        image_path=img_path,
        center_crop_box=box,
        features_by_poi=features_by_poi,
        distances_by_poi=distances_by_poi,
        pose=pose,
        rms_reproj_px=float(rms_px),
        diagnostics={
            "d0_m": float(d0),
            "a_m": float(a),
            "half_px": float(half_px),
            "distance_variance_m2": float(np.var(list(distances_by_poi.values()))),
            "distance_cv": float(np.std(list(distances_by_poi.values())) / (abs(d0) + 1e-12)),
            "mean_peak_snr": float(np.mean([f.peak_snr for f in features_by_poi.values()])),
            "mean_peak_sharpness": float(np.mean([f.peak_sharpness for f in features_by_poi.values()])),
            "mean_physical_infeasibility": float(np.mean([f.physical_infeasibility for f in features_by_poi.values()])),
            "distance_diagnostics_by_poi": distance_diagnostics_by_poi,
            "metric_warning": "rms_reproj_px is internal consistency only; it is not absolute pose accuracy.",
        }
    )

# ============================================================
# 10) Patch mode: 5x5 patches, pick min reprojection error patch as "center"
# ============================================================

def split_into_grid_boxes(img_shape: Tuple[int,int], grid: Tuple[int, int] = (5, 5)) -> List[Tuple[int,int,int,int]]:
    h, w = img_shape[:2]
    boxes = []
    gy, gx = grid
    ys = np.linspace(0, h, gy + 1).astype(int)
    xs = np.linspace(0, w, gx + 1).astype(int)
    for iy in range(gy):
        for ix in range(gx):
            y0, y1 = ys[iy], ys[iy+1]
            x0, x1 = xs[ix], xs[ix+1]
            boxes.append((int(x0), int(y0), int(x1), int(y1)))
    return boxes

def split_into_5x5_boxes(img_shape: Tuple[int,int]) -> List[Tuple[int,int,int,int]]:
    return split_into_grid_boxes(img_shape, grid=(5, 5))


def _mean_feature_diag(res: CalibResult, attr: str, default: float = 0.0) -> float:
    vals = []
    for f in res.features_by_poi.values():
        vals.append(float(getattr(f, attr, default)))
    return float(np.mean(vals)) if vals else default


def reliability_patch_score(
    orig_res: CalibResult,
    aug_res: CalibResult,
    box: Tuple[int, int, int, int],
    img_shape: Tuple[int, int],
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[str, float]]:
    """Patch score used for review-response experiments.

    The old patch selector minimized the four-POI internal RMS only.  Reviewers
    correctly noted that this is a weak heuristic.  This score still includes the
    internal geometric consistency term, but also adds spectral confidence,
    distance consistency, physical feasibility, and an optional radial penalty for
    off-center patches where distortion/vignetting is more likely.

    Lower is better.  The units are px-like because the internal RMS remains the
    dominant term; auxiliary terms are diagnostic penalties/bonuses.
    """
    if weights is None:
        weights = {
            "geo": 1.0,
            "snr": 2.0,
            "sharpness": 20.0,
            "dist_cv": 80.0,
            "physical": 300.0,
            "radial": 25.0,
        }

    geo = 0.5 * (float(orig_res.rms_reproj_px) + float(aug_res.rms_reproj_px))
    snr = 0.5 * (_mean_feature_diag(orig_res, "peak_snr") + _mean_feature_diag(aug_res, "peak_snr"))
    sharp = 0.5 * (_mean_feature_diag(orig_res, "peak_sharpness") + _mean_feature_diag(aug_res, "peak_sharpness"))

    dist_cv = 0.5 * (
        float(orig_res.diagnostics.get("distance_cv", 0.0)) +
        float(aug_res.diagnostics.get("distance_cv", 0.0))
    )
    phys = 0.5 * (
        _mean_feature_diag(orig_res, "physical_infeasibility") +
        _mean_feature_diag(aug_res, "physical_infeasibility")
    )

    h, w = img_shape[:2]
    x0, y0, x1, y1 = box
    pcx, pcy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    radial = math.hypot(pcx - w / 2.0, pcy - h / 2.0) / (math.hypot(w / 2.0, h / 2.0) + 1e-12)

    score = (
        weights.get("geo", 1.0) * geo
        - weights.get("snr", 0.0) * math.log1p(max(snr, 0.0))
        - weights.get("sharpness", 0.0) * max(sharp, 0.0)
        + weights.get("dist_cv", 0.0) * max(dist_cv, 0.0)
        + weights.get("physical", 0.0) * max(phys, 0.0)
        + weights.get("radial", 0.0) * max(radial, 0.0)
    )

    diag = {
        "score": float(score),
        "internal_rms_avg_px": float(geo),
        "mean_peak_snr": float(snr),
        "mean_peak_sharpness": float(sharp),
        "distance_cv": float(dist_cv),
        "physical_infeasibility": float(phys),
        "radial_penalty": float(radial),
    }
    return float(score), diag


def patch_calibration_min_reproj(
    orig_path: str,
    aug_path: str,
    params: Dict,
    grid: Tuple[int, int] = (5, 5),
    score_mode: str = "reliability",
    score_weights: Optional[Dict[str, float]] = None,
) -> Tuple[int, Dict]:
    # read once to get shape
    img = cv2.imread(aug_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(aug_path)
    boxes = split_into_grid_boxes(img.shape, grid=grid)

    patch_scores = []
    patch_outputs = {}

    for idx, box in enumerate(boxes):
        try:
            # For patch-mode, we treat the patch itself as the "center crop" for MoiréPose pipeline.
            # (It matches user request: "각각 25개의 패치로 분해한 다음 ... 캘리브레이션".)
            aug_res = moirepose_calibrate_single(aug_path, params, override_crop=box)
            orig_res = moirepose_calibrate_single(orig_path, params, override_crop=box)

            if score_mode == "internal_rms":
                score = 0.5 * (aug_res.rms_reproj_px + orig_res.rms_reproj_px)
                score_diag = {"score": float(score), "internal_rms_avg_px": float(score)}
            else:
                score, score_diag = reliability_patch_score(
                    orig_res, aug_res, box=box, img_shape=img.shape, weights=score_weights
                )

            patch_scores.append((score, idx))
            patch_outputs[idx] = {
                "box": box,
                "score": float(score),
                "score_mode": score_mode,
                "score_diagnostics": score_diag,
                "orig_rms_reproj_px": orig_res.rms_reproj_px,
                "aug_rms_reproj_px": aug_res.rms_reproj_px,
                "orig_diagnostics": orig_res.diagnostics,
                "aug_diagnostics": aug_res.diagnostics,
            }
        except Exception as e:
            patch_scores.append((1e9, idx))
            patch_outputs[idx] = {"box": box, "error": str(e)}

    patch_scores.sort(key=lambda t: t[0])
    best_score, best_idx = patch_scores[0]
    patch_outputs["best_patch_index"] = best_idx
    patch_outputs["best_patch_score"] = float(best_score)
    patch_outputs["grid"] = tuple(grid)
    patch_outputs["score_mode"] = score_mode
    return best_idx, patch_outputs

# ============================================================
# 11) Pair processing + reporting
# ============================================================

def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return math.degrees(math.acos(dot))

def calibrate_pair(
    orig_path: str,
    aug_path: str,
    params: Dict,
    patch_mode: bool,
    patch_grid: Tuple[int, int] = (5, 5),
    patch_score_mode: str = "reliability",
) -> Dict:
    if patch_mode:
        best_idx, patch_debug = patch_calibration_min_reproj(
            orig_path, aug_path, params, grid=patch_grid, score_mode=patch_score_mode
        )

        # use best patch as "image center"
        img = cv2.imread(aug_path, cv2.IMREAD_COLOR)
        boxes = split_into_grid_boxes(img.shape, grid=patch_grid)
        best_box = boxes[best_idx]

        orig_res = moirepose_calibrate_single(orig_path, params, override_crop=best_box)
        aug_res = moirepose_calibrate_single(aug_path, params, override_crop=best_box)

        debug = {
            "patch_mode": True,
            "patch_grid": tuple(patch_grid),
            "patch_score_mode": patch_score_mode,
            "patch_debug": patch_debug,
            "chosen_patch_box": best_box,
        }
    else:
        orig_res = moirepose_calibrate_single(orig_path, params, override_crop=None)
        aug_res = moirepose_calibrate_single(aug_path, params, override_crop=None)
        debug = {"patch_mode": False}

    P0 = np.array([orig_res.pose.x, orig_res.pose.y, orig_res.pose.z], dtype=np.float64)
    P1 = np.array([aug_res.pose.x, aug_res.pose.y, aug_res.pose.z], dtype=np.float64)
    dP = (P1 - P0)

    w0 = np.array(orig_res.pose.w, dtype=np.float64)
    w1 = np.array(aug_res.pose.w, dtype=np.float64)
    dw_deg = angle_between(w0, w1)

    pair = PairResult(
        orig=orig_res,
        aug=aug_res,
        delta_position_m=(float(dP[0]), float(dP[1]), float(dP[2])),
        delta_w_angle_deg=float(dw_deg),
    )

    out = {
        "device_params": params,
        "pair_result": {
            "orig": asdict(pair.orig),
            "aug": asdict(pair.aug),
            "delta_position_m": pair.delta_position_m,
            "delta_w_angle_deg": pair.delta_w_angle_deg,
        },
        "debug": debug
    }
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig", required=True, help="path to original image, e.g., 0000_gt.jpg")
    parser.add_argument("--aug", required=True, help="path to augmented image, e.g., 0000_gt_blackbox_1.jpg")
    parser.add_argument("--patch_mode", action="store_true", help="enable 5x5 patch search and choose min reproj patch")
    parser.add_argument("--patch_grid", default="5,5", help="patch grid as rows,cols; e.g. 3,3 or 5,5")
    parser.add_argument("--patch_score_mode", default="reliability", choices=["reliability", "internal_rms"])
    parser.add_argument("--out_json", default=None, help="output json path (optional)")
    args = parser.parse_args()

    grid = tuple(int(x.strip()) for x in args.patch_grid.split(","))
    if len(grid) != 2:
        raise ValueError("--patch_grid must have the form rows,cols")
    result = calibrate_pair(
        args.orig,
        args.aug,
        DEVICE_PARAMS,
        args.patch_mode,
        patch_grid=(grid[0], grid[1]),
        patch_score_mode=args.patch_score_mode,
    )

    txt = json.dumps(result, indent=2, ensure_ascii=False)
    print(txt)
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            f.write(txt)

if __name__ == "__main__":
    main()

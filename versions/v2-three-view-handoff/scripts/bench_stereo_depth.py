"""Compare RealSense D435 native depth with OpenCV StereoSGBM on rectified IR frames."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import time
import cv2
import numpy as np


def compute_sgbm_with_soft_lrc(
    ir_left: np.ndarray,
    ir_right: np.ndarray,
    baseline_m: float,
    fx_px: float,
    num_disparities: int = 96,
    block_size: int = 5,
    lrc_sigma_px: float = 1.5,
) -> dict:
    """Compute OpenCV CPU StereoSGBM left+right disparity & F26 SoftLRC continuous confidence."""
    t0 = time.perf_counter()
    gL = cv2.cvtColor(ir_left, cv2.COLOR_RGB2GRAY) if ir_left.ndim == 3 else ir_left
    gR = cv2.cvtColor(ir_right, cv2.COLOR_RGB2GRAY) if ir_right.ndim == 3 else ir_right

    sgbm = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * block_size * block_size,
        P2=32 * block_size * block_size,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disp_L = sgbm.compute(gL, gR).astype(np.float32) / 16.0

    # Right disparity without opencv-contrib via horizontally flipped images
    disp_R_flip = sgbm.compute(gR[:, ::-1], gL[:, ::-1]).astype(np.float32) / 16.0
    disp_R = disp_R_flip[:, ::-1]

    H, W = disp_L.shape
    uu = np.arange(W, dtype=np.float32)[None, :].repeat(H, axis=0)
    match_u = np.round(uu - disp_L).astype(np.int32)
    valid = (disp_L > 0.5) & (match_u >= 0) & (match_u < W)
    mu_clip = np.clip(match_u, 0, W - 1)
    disp_R_at_match = disp_R[np.arange(H)[:, None], mu_clip]
    lr_err = np.where(valid & (disp_R_at_match > 0.5), np.abs(disp_L - disp_R_at_match), 10.0)

    # F26 Stereo Anywhere continuous SoftLRC reliability in [0, 1]
    soft_lrc = np.where(valid, np.exp(-(lr_err ** 2) / (2.0 * lrc_sigma_px ** 2)), 0.0)
    depth_m = np.where(valid & (soft_lrc >= 0.5), (fx_px * baseline_m) / np.maximum(disp_L, 0.1), np.nan)
    ms = (time.perf_counter() - t0) * 1000.0
    return {
        "depth_m": depth_m.astype(np.float32),
        "soft_lrc_conf": soft_lrc.astype(np.float32),
        "valid_pct": round(100.0 * float(np.mean(np.isfinite(depth_m))), 2),
        "runtime_ms": round(ms, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--left", required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--baseline-m", type=float, default=0.050)
    ap.add_argument("--fx-px", type=float, default=424.0)
    ap.add_argument(
        "--native-depth",
        help="Optional 16-bit D435 depth image from the same capture, aligned to the left IR image",
    )
    ap.add_argument(
        "--native-depth-scale-m", type=float, default=0.001,
        help="Metres per stored depth unit (default: 0.001)",
    )
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    L = cv2.imread(args.left, cv2.IMREAD_UNCHANGED)
    R = cv2.imread(args.right, cv2.IMREAD_UNCHANGED)
    if L is None or R is None:
        raise SystemExit("Could not load --left / --right IR images")
    res = compute_sgbm_with_soft_lrc(L, R, args.baseline_m, args.fx_px)
    has_ort = importlib.util.find_spec("onnxruntime") is not None
    report = {
        "sgbm_soft_lrc": {k: v for k, v in res.items() if isinstance(v, (int, float))},
        "las2_onnx": {"available": has_ort, "reason": "" if has_ort else "onnxruntime not installed locally"},
        "fast_foundation_stereo": {"available": False, "reason": "Requires a compatible GPU, inference runtime and model weights"},
    }
    if args.native_depth:
        if not np.isfinite(args.native_depth_scale_m) or args.native_depth_scale_m <= 0:
            ap.error("--native-depth-scale-m must be finite and positive")
        native_depth = cv2.imread(args.native_depth, cv2.IMREAD_UNCHANGED)
        if native_depth is None or native_depth.dtype != np.uint16 or native_depth.shape != L.shape[:2]:
            ap.error("--native-depth must be a readable 16-bit image on the left IR pixel grid")

        native_valid = native_depth > 0
        shared_valid = native_valid & np.isfinite(res["depth_m"])
        native_m = native_depth[shared_valid] * args.native_depth_scale_m
        difference_mm = 1000.0 * np.abs(res["depth_m"][shared_valid] - native_m)
        report["d435_native_comparison"] = {
            "native_valid_pct": round(100.0 * float(np.mean(native_valid)), 2),
            "overlap_pct": round(100.0 * float(np.mean(shared_valid)), 2),
            "overlap_pixels": int(difference_mm.size),
            "abs_difference_mm_p50": round(float(np.median(difference_mm)), 2) if difference_mm.size else None,
            "abs_difference_mm_p95": round(float(np.percentile(difference_mm, 95)), 2) if difference_mm.size else None,
        }

    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

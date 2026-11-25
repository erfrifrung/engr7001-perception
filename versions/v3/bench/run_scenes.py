"""Run one method over a saved scene split using a given code tree (v2-baseline or v3) and write JSONL.

Usage: python run_scenes.py --root ../../v2-baseline --scenes DIR --method v2base_mc --out FILE
Methods: naive | v2base_mc | v2base_wpmv | tuned_mc | tuned_wpmv | v3prove_mc | v3enum_mc | v3prove_wpmv |
         v3prove_mc_nocarve
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--scenes", required=True)
ap.add_argument("--method", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--repeat", type=int, default=1)
args = ap.parse_args()
sys.path.insert(0, str(Path(args.root).resolve()))

from wp3_live import ViewInput, preset_config, run_perception_epoch  # noqa: E402
from wp3_live.dwell_gate import DwellStatus  # noqa: E402
from wp3_live.geometry_cache import CameraModelCache, deproject_roi_points  # noqa: E402
from wp3_live.heightmap_core import HeightmapGrid, build_heightmap_fast, extract_height_band_components  # noqa: E402

import cv2  # noqa: E402

FAR_ROBOT = {"link6": ([1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]), "link_tcp": ([1.0, 1.0, 0.8], [0.0, 0.0, 0.0, 1.0])}
DWELL = DwellStatus(True, True, True, True, 0.0, 0.1, 0.2, 80.0, "ok")
cache = CameraModelCache()


def cams(Kw, Ks):
    cw = cache.get("wrist", 1280, 720, Kw, np.zeros(5), "wrist_color_optical_frame", 0.001)
    cs = cache.get("tripod", 640, 480, Ks, np.zeros(5), "tripod_color_optical_frame", 0.001)
    return cw, cs


def naive(cfg, cw, Tw, rgb, d):
    """Lower bound: band component -> min-area rectangle -> equal split into round(area / block) pieces."""
    dep = deproject_roi_points(d, rgb, cw, Tw, cfg.roi_xyz_min, cfg.roi_xyz_max, need_pts_cam=False)
    grid = HeightmapGrid.from_roi(cfg.roi_xyz_min[:2], cfg.roi_xyz_max[:2], 0.002)
    hm = build_heightmap_fast(dep["pts_tbl_f32"], dep["rgb_u8"], grid)
    out = []
    B = cfg.block
    for comp in extract_height_band_components(hm, B.height_m, min_area_cells=100):
        pts = comp["pts_tbl"]
        (cx, cy), (w, h), ang = cv2.minAreaRect(pts[:, :2].astype(np.float32))
        if w < h:
            w, h, ang = h, w, ang + 90
        k = int(round(w * h / (B.major_m * B.minor_m)))
        if k < 1:
            continue
        yaw = np.radians(ang)
        ux = np.array([np.cos(yaw), np.sin(yaw)])
        uy = np.array([-np.sin(yaw), np.cos(yaw)])
        if h > 1.5 * B.minor_m:          # pieces stacked across the short side
            for i in range(k):
                c = np.array([cx, cy]) + uy * ((i + 0.5) * h / k - h / 2)
                out.append(dict(xy=c.tolist(), z=float(np.median(pts[:, 2])) - B.height_m / 2, yaw=float(yaw),
                                dims=[w * 1000, h / k * 1000], src="naive"))
        else:
            for i in range(k):
                c = np.array([cx, cy]) + ux * ((i + 0.5) * w / k - w / 2)
                out.append(dict(xy=c.tolist(), z=float(np.median(pts[:, 2])) - B.height_m / 2, yaw=float(yaw),
                                dims=[w / k * 1000, h * 1000], src="naive"))
    return out, []


def method_cfg(m):
    cfg = preset_config("printed_jenga")
    cfg.calib_epoch = "synthetic"
    pol = "whole_partition_mv" if m.endswith("wpmv") else "merged_corroborated"
    if m.startswith(("tuned", "v3")):
        cfg.enable_v3_regions = m.startswith("v3")
        if "slowdep" in m:
            cfg.v3_fast_deproject = False
        if m.startswith("v3"):
            cfg.v3_solver = "enumerate" if "enum" in m else ("adaptive" if "adapt" in m else "prove")
            if "nocarve" in m:
                cfg.v3_view_carving = False
            if "noguard" in m:
                cfg.v3_slab_guard = False
            if "slowdep" in m:
                cfg.v3_fast_deproject = False
            if "lowbudget" in m:          # stress: timeouts must only cost coverage, never correctness
                cfg.v3_max_nodes, cfg.v3_frame_budget_ms, cfg.v3_region_budget_ms = 150, 15.0, 8.0
    return cfg, pol


index = json.loads((Path(args.scenes) / "index.json").read_text())
cfg, pol = method_cfg(args.method)
with open(args.out, "w") as fh:
    for rep in range(args.repeat):
        for k, sc in enumerate(index):
            z = np.load(Path(args.scenes) / sc["file"])
            cw, cs = cams(z["Kw"], z["Ks"])
            t0 = time.perf_counter()
            if args.method == "naive":
                poses, amb = naive(cfg, cw, z["Tw"], z["rgb_w"], z["d_w"])
                timings = {"total_ms": round((time.perf_counter() - t0) * 1000, 3)}
            else:
                stamp = 1_000_000_000 * (k + 1)
                r = run_perception_epoch(
                    wrist=ViewInput("wrist", cw, z["rgb_w"], z["d_w"], z["Tw"], stamp, stamp),
                    side=ViewInput("tripod", cs, z["rgb_s"], z["d_s"], z["Ts"], stamp, stamp),
                    config=cfg, policy=pol, dwell=DWELL, epoch_id=k + 1,
                    robot_links=FAR_ROBOT,   # TF available, robot parked outside the ROI (same for every method)
                )
                poses = [dict(xy=list(f.xy_m), z=f.z_center_m, yaw=f.yaw_rad, dims=list(f.dims_mm), src=s)
                         for f, s in zip(r.poses, r.pose_sources)]
                amb = [dict(xy=list(a.xy_m), reason=a.reason) for a in r.ambiguous_clusters]
                timings = r.stage_timings_ms
            fh.write(json.dumps(dict(file=sc["file"], rep=rep, method=args.method, poses=poses, amb=amb,
                                     timings=timings, wall_ms=round((time.perf_counter() - t0) * 1000, 3))) + "\n")

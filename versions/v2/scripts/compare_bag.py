"""Shared multi-method matched bag comparison runner (v2 historical controls + live-core policies)."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np


IMPL_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = IMPL_ROOT.parent
V1_DIR = OUTPUT_ROOT / "WP3-Live-Implementation"
V2_DIR = OUTPUT_ROOT / "WP3-Heightmap-2026-11-18" / "v2"
if str(IMPL_ROOT) not in sys.path:
    sys.path.insert(0, str(IMPL_ROOT))

from wp3_live import (  # noqa: E402
    EpochTemporalCache,
    ViewInput,
    config_hash,
    configure_single_thread_libs,
    load_runtime_config,
    run_perception_epoch,
    sha256_files,
)
from wp3_live.bag_reader import chain_tf, load_bag_fast  # noqa: E402
from wp3_live.dwell_gate import WristDwellGate  # noqa: E402
from wp3_live.fusion_tracker import WP3TemporalTracker  # noqa: E402
from wp3_live.geometry_cache import CameraModelCache  # noqa: E402
from wp3_live.overlay import render_overlay  # noqa: E402
from wp3_live.scoring import aggregate_method_summary, load_ground_truth_labels, score_epoch_against_labels  # noqa: E402
from wp3_live.side_verifier import _bilinear_sample_valid_depth  # noqa: E402

configure_single_thread_libs()


def _pair_side_to_wrist(wrist_pairs: list[dict], side_pairs: list[dict], max_skew_s: float) -> list[dict | None]:
    """Nearest-stamp pairing of side frames to wrist frames within max_skew_s."""
    if not side_pairs:
        return [None] * len(wrist_pairs)
    s_stamps = np.array([p["header_ns"] for p in side_pairs], dtype=np.int64)
    out: list[dict | None] = []
    max_skew_ns = int(round(max_skew_s * 1e9))
    for wp in wrist_pairs:
        idx = int(np.searchsorted(s_stamps, wp["header_ns"]))
        cand = []
        if idx < len(s_stamps):
            cand.append(idx)
        if idx > 0:
            cand.append(idx - 1)
        best_i = min(cand, key=lambda i: abs(int(s_stamps[i]) - wp["header_ns"]))
        if abs(int(s_stamps[best_i]) - wp["header_ns"]) <= max_skew_ns:
            out.append(side_pairs[best_i])
        else:
            out.append(None)
    return out


def _register_board_T(frames: list[dict], info: dict, cam_model, cols: int, rows: int, pitch_m: float):
    obj = np.array(
        [[(x - (cols - 1) / 2.0) * pitch_m, ((rows - 1) / 2.0 - y) * pitch_m, 0.0] for y in range(rows) for x in range(cols)],
        dtype=np.float32,
    )
    K = np.asarray(info["K"], dtype=np.float64)
    D = np.asarray(info["D"], dtype=np.float64)
    sample_frames = frames[:: max(1, len(frames) // 12)][:12] if len(frames) > 12 else frames
    poses, corners_list, frames_used, rms = [], [], [], []
    ref_corners = None
    for p in sample_frames:
        gray = cv2.cvtColor(p["rgb"], cv2.COLOR_RGB2GRAY)
        ok, c = cv2.findChessboardCornersSB(gray, (cols, rows), flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        if not ok:
            continue
        c = c.reshape(-1, 2)
        if ref_corners is None:
            ref_corners = c.copy()
        elif float(np.mean((c[::-1] - ref_corners) ** 2)) < float(np.mean((c - ref_corners) ** 2)):
            c = c[::-1].copy()
        ok, rv, tv = cv2.solvePnP(obj, c, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        proj, _ = cv2.projectPoints(obj, rv, tv, K, D)
        rms.append(float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - c) ** 2, 1)))))
        R, _ = cv2.Rodrigues(rv)
        poses.append((R, tv.ravel()))
        corners_list.append(c)
        frames_used.append(p)
    if len(poses) < 3:
        raise RuntimeError(f"Board registration failed ({cam_model.role}): poses={len(poses)} (need >= 3)")

    R_mean = np.mean([R for R, _ in poses], axis=0)
    U, _, Vt = np.linalg.svd(R_mean)
    R_cb = U @ Vt
    if np.linalg.det(R_cb) < 0:
        U[:, -1] *= -1.0
        R_cb = U @ Vt
    t_cb = np.median([t for _, t in poses], axis=0)
    cam_from_board = np.eye(4, dtype=np.float64)
    cam_from_board[:3, :3] = R_cb
    cam_from_board[:3, 3] = t_cb
    T = np.linalg.inv(cam_from_board)
    if T[2, 3] < 0.0:
        # Rotate 180 deg around board X if solvePnP chose downward normal so +Z points toward camera
        T = np.diag([1.0, -1.0, -1.0, 1.0]) @ T

    quad_idx = [9, 14, 38, 33] if (cols, rows) == (8, 6) else [0, cols - 1, rows * cols - 1, (rows - 1) * cols]
    normal_errs_mm: list[np.ndarray] = []
    for p, c in zip(frames_used, corners_list):
        mask = np.zeros(p["rgb"].shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.rint(c[quad_idx]).astype(np.int32), 1)
        v_px, u_px = np.nonzero(mask)
        if len(v_px) == 0:
            continue
        d_m = p["depth_u16"][v_px, u_px].astype(np.float32) * np.float32(cam_model.depth_scale_m)
        ok_d = (d_m > 0.15) & (d_m < 2.0)
        if not np.any(ok_d):
            continue
        rays = cam_model.rays_f32[v_px[ok_d], u_px[ok_d]].astype(np.float64)
        z_err_mm = (d_m[ok_d].astype(np.float64) * (rays @ T[2, :3]) + T[2, 3]) * 1000.0
        normal_errs_mm.append(z_err_mm)
    med_norm_mm = round(float(np.median(np.concatenate(normal_errs_mm))), 3) if normal_errs_mm else None

    return T, (T[2, :3].copy(), float(T[2, 3])), {
        "board_frames": len(poses),
        "reproj_rms_px_median": round(float(np.median(rms)), 3),
        "board_normal_depth_error_mm_p50": med_norm_mm,
        "cam_pos_table_m": [round(float(x), 4) for x in T[:3, 3]],
    }


def _disambiguate_side_board_180(
    T_wrist_tbl: np.ndarray,
    T_side_0: np.ndarray,
    paired_reg_frames: list[tuple[dict, dict]],
    cam_wrist,
    cam_side,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Resolve 180 deg Z symmetry of even-grid (e.g. 8x6) checkerboard via 3D mutual depth consistency."""
    pts_list: list[np.ndarray] = []
    Rw = T_wrist_tbl[:3, :3].astype(np.float32)
    tw = T_wrist_tbl[:3, 3].astype(np.float32)
    per_frame_cap = max(200, 4000 // max(1, len(paired_reg_frames)))
    for wp, _ in paired_reg_frames:
        d_w = wp["depth_u16"].astype(np.float32) * np.float32(cam_wrist.depth_scale_m)
        v = (d_w > 0.15) & (d_w < 1.50)
        if not np.any(v):
            continue
        pc_w = cam_wrist.rays_f32[v] * d_w[v, None]
        pt_w = pc_w @ Rw.T + tw
        above = (pt_w[:, 2] > 0.015) & (pt_w[:, 2] < 0.15) & (np.abs(pt_w[:, 0]) < 0.35) & (np.abs(pt_w[:, 1]) < 0.35)
        cand = pt_w[above]
        if len(cand) > per_frame_cap:
            cand = cand[:: max(1, len(cand) // per_frame_cap)][:per_frame_cap]
        if len(cand) > 0:
            pts_list.append(cand)

    T_side_180 = np.diag([-1.0, -1.0, 1.0, 1.0]) @ T_side_0
    if not pts_list:
        return T_side_0, {"available": False, "reason": "no_above_table_wrist_points", "flipped_180": False}

    def _eval_cand(T_s: np.ndarray) -> tuple[float, float, int]:
        T_cam_tbl = np.linalg.inv(T_s)
        R_ct = T_cam_tbl[:3, :3].astype(np.float64)
        t_ct = T_cam_tbl[:3, 3].astype(np.float64)
        rvec0 = np.zeros((3, 1), dtype=np.float64)
        tvec0 = np.zeros((3, 1), dtype=np.float64)
        all_errs: list[np.ndarray] = []
        for idx, (_, sp) in enumerate(paired_reg_frames):
            if idx >= len(pts_list):
                break
            pt = pts_list[idx].astype(np.float64)
            pc_s = pt @ R_ct.T + t_ct
            in_front = pc_s[:, 2] > 0.15
            if np.count_nonzero(in_front) < 20:
                continue
            pc_s = pc_s[in_front]
            uv, _ = cv2.projectPoints(pc_s, rvec0, tvec0, cam_side.K_f64, cam_side.D_f64)
            uv = uv.reshape(-1, 2)
            d_side_m = sp["depth_u16"].astype(np.float32) * np.float32(cam_side.depth_scale_m)
            d_meas, d_ok = _bilinear_sample_valid_depth(d_side_m, uv, max_jump_m=0.04)
            valid = d_ok & (d_meas > 0.15)
            if np.any(valid):
                all_errs.append(np.abs(d_meas[valid].astype(np.float64) - pc_s[valid, 2]))
        if not all_errs:
            return 0.0, 9999.0, 0
        errs = np.concatenate(all_errs)
        return float(np.mean(errs < 0.025)), float(np.median(errs) * 1000.0), int(len(errs))

    f0, e0, n0 = _eval_cand(T_side_0)
    f180, e180, n180 = _eval_cand(T_side_180)
    flip = (f180, -e180) > (f0, -e0)
    return (T_side_180 if flip else T_side_0), {
        "available": True,
        "n_pts_0": n0,
        "n_pts_180": n180,
        "inlier_frac_0": round(f0, 4),
        "inlier_frac_180": round(f180, 4),
        "med_err_0_mm": round(e0, 2),
        "med_err_180_mm": round(e180, 2),
        "flipped_180": bool(flip),
    }


def _try_run_frozen_v2(
    epochs: list[dict],
    cam_wrist,
    T_wrist_table: np.ndarray | None,
    cfg,
    labels: list[dict],
    sym_deg: float,
    frame_mode: str,
    dropped: dict[str, int],
    warmup_frames: int = 0,
) -> dict[str, Any]:
    """Run unmodified frozen WP3-Heightmap-2026-11-18/v2 H1 & H2 on the exact same epochs."""
    v2_path = V2_DIR / "wp3_heightmap.py"
    if not v2_path.is_file():
        return {"available": False, "reason": f"missing {v2_path}"}
    spec = importlib.util.spec_from_file_location("frozen_wp3_heightmap_v2", str(v2_path))
    if spec is None or spec.loader is None:
        return {"available": False, "reason": "could not load v2 module"}
    v2 = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = v2
    spec.loader.exec_module(v2)

    p = v2.Params(
        cell_m=cfg.thresholds.cell_m,
        block_l_m=cfg.block.length_m,
        block_w_m=cfg.block.width_m,
        block_h_m=cfg.block.height_m,
        band_lo=cfg.thresholds.band_lo_factor,
        band_hi=cfg.thresholds.band_hi_factor,
    )
    grid = v2.Grid(
        cfg.roi_xyz_min[0], cfg.roi_xyz_min[1],
        int(np.ceil((cfg.roi_xyz_max[0] - cfg.roi_xyz_min[0]) / p.cell_m)),
        int(np.ceil((cfg.roi_xyz_max[1] - cfg.roi_xyz_min[1]) / p.cell_m)),
        p.cell_m,
    )
    triples = np.random.default_rng(18).integers(0, 2048, size=(120, 3))
    out: dict[str, Any] = {"available": True, "v2_H1": {}, "v2_H2": {}}
    records: dict[str, list[dict]] = {"v2_H1": [], "v2_H2": []}

    for idx, ep in enumerate(epochs):
        wp = ep["wrist"]
        d_m = wp["depth_u16"].astype(np.float32) * np.float32(cam_wrist.depth_scale_m)
        pc = cam_wrist.rays_f32 * d_m[:, :, None]
        valid = (d_m > 0.18) & (d_m < 1.60)
        if T_wrist_table is not None and frame_mode != "plane":
            T_tc = T_wrist_table
        else:
            fit = v2.fit_plane(pc[valid][::9], triples)
            if fit is None:
                for k in ("v2_H1", "v2_H2"):
                    records[k].append({"status": "no_plane"})
                continue
            T_tc = v2.plane_frame(fit[0], fit[1])
        pt = (pc.reshape(-1, 3) @ T_tc[:3, :3].T + T_tc[:3, 3]).reshape(cam_wrist.height, cam_wrist.width, 3)
        sel = (
            valid & (pt[..., 0] >= cfg.roi_xyz_min[0]) & (pt[..., 0] <= cfg.roi_xyz_max[0])
            & (pt[..., 1] >= cfg.roi_xyz_min[1]) & (pt[..., 1] <= cfg.roi_xyz_max[1])
            & (pt[..., 2] >= cfg.roi_xyz_min[2]) & (pt[..., 2] <= cfg.roi_xyz_max[2])
        ).reshape(-1)
        pts_sel = pt.reshape(-1, 3)[sel]
        rgb_sel = wp["rgb"].reshape(-1, 3)[sel]
        for key, var in (("v2_H1", "H1"), ("v2_H2", "H2")):
            tic = time.perf_counter()
            Hm, Cm, Vm = v2.heightmap(pts_sel, rgb_sel, grid)
            regs, lab = v2.detect(Hm, Cm, grid, p, variant=var, V=Vm)
            ms = (time.perf_counter() - tic) * 1000.0
            inst = v2.instances(regs)
            # Reuse physical-membership scoring logic via minimal wrapper
            amb = [r for r in regs if r.status in ("ambiguous_merged", "unexpected_shape")]
            rec = _score_v2_record(ep["t_rel_s"], inst, amb, lab, grid, labels, sym_deg, ms)
            records[key].append(rec)

    for k in ("v2_H1", "v2_H2"):
        out[k] = aggregate_method_summary(records[k], len(epochs), dropped, warmup_frames=warmup_frames)
    return out


def _try_run_frozen_v1(
    epochs: list[dict],
    cam_wrist,
    cam_side,
    T_wrist_tbl: np.ndarray | None,
    T_side_tbl: np.ndarray | None,
    config_path: str | Path,
    labels: list[dict],
    sym_deg: float,
    frame_mode: str,
    dwell_by_idx: list[Any],
    dropped: dict[str, int],
    warmup_frames: int = 0,
) -> dict[str, Any]:
    """Run unmodified frozen WP3-Live-Implementation (v1) controls on identical scheduled epochs."""
    if not (V1_DIR / "wp3_live" / "__init__.py").is_file():
        return {"available": False, "reason": f"missing {V1_DIR / 'wp3_live'}"}

    saved_mods = {k: sys.modules[k] for k in list(sys.modules) if k == "wp3_live" or k.startswith("wp3_live.")}
    saved_path = list(sys.path)
    try:
        for k in list(saved_mods):
            del sys.modules[k]
        sys.path = [str(V1_DIR)] + [p for p in saved_path if p != str(IMPL_ROOT)]
        v1_pkg = importlib.import_module("wp3_live")
        v1_geom = importlib.import_module("wp3_live.geometry_cache")
        v1_dwell = importlib.import_module("wp3_live.dwell_gate")

        cfg_v1 = v1_pkg.load_runtime_config(config_path)
        cc_v1 = v1_geom.CameraModelCache()
        cw1 = cc_v1.get("wrist", cam_wrist.width, cam_wrist.height, cam_wrist.K_f64, cam_wrist.D_f64, cam_wrist.frame_id, cfg_v1.wrist.depth_scale_m) if cam_wrist is not None else None
        cs1 = cc_v1.get("tripod", cam_side.width, cam_side.height, cam_side.K_f64, cam_side.D_f64, cam_side.frame_id, cfg_v1.side.depth_scale_m) if cam_side is not None else None
        triples = np.random.default_rng(18).integers(0, 2048, size=(120, 3))

        out: dict[str, Any] = {"available": True}
        for pol in ("merged_corroborated", "merged_wrist_rgb"):
            if cw1 is None or cs1 is None or T_side_tbl is None:
                continue
            key = f"v1_both__{pol}"
            recs: list[dict[str, Any]] = []
            for ep_idx, ep in enumerate(epochs):
                wp, sp = ep["wrist"], ep["side"]
                w_vi = v1_pkg.ViewInput(role="wrist", cam=cw1, rgb_img=wp["rgb"], depth_raw=wp["depth_u16"], T_table_cam=T_wrist_tbl, stamp_ns=wp["header_ns"], rx_ns=wp["rx_ns"], pose_frame_mode="plane" if frame_mode == "plane" else "table") if wp is not None else None
                s_vi = v1_pkg.ViewInput(role="tripod", cam=cs1, rgb_img=sp["rgb"], depth_raw=sp["depth_u16"], T_table_cam=T_side_tbl, stamp_ns=sp["header_ns"], rx_ns=sp["rx_ns"], pose_frame_mode="table") if sp is not None else None
                dw = dwell_by_idx[ep_idx]
                dv1 = v1_dwell.DwellStatus(**vars(dw)) if dw is not None else None
                res = v1_pkg.run_perception_epoch(wrist=w_vi, side=s_vi, config=cfg_v1, policy=pol, dwell=dv1, rng_triples=triples, epoch_id=ep_idx + 1)
                sc = score_epoch_against_labels(ep["t_rel_s"], res, labels, sym_deg) if labels else None
                recs.append({"idx": ep["idx"], "t": round(ep["t_rel_s"], 3), "status": res.status, "n_out": len(res.poses), "n_amb": len(res.ambiguous_clusters), "stage_timings_ms": res.stage_timings_ms, "score": sc})
            out[key] = aggregate_method_summary(recs, len(epochs), dropped, warmup_frames=warmup_frames)
        return out
    finally:
        for k in list(sys.modules):
            if k == "wp3_live" or k.startswith("wp3_live."):
                del sys.modules[k]
        sys.modules.update(saved_mods)
        sys.path = saved_path


def _score_v2_record(t_rel_s: float, inst, amb, lab, grid, labels, sym_deg: float, ms: float) -> dict[str, Any]:
    from wp3_live.policies import AmbiguousCluster, EpochResult
    from wp3_live.proposals import BlockFit

    dummy_pts = np.zeros((0, 3), dtype=np.float32)
    fits = [
        BlockFit(
            xy_m=r.xy_m, z_center_m=r.top_mm / 1000.0, z_top_m=r.top_mm / 1000.0,
            yaw_rad=np.radians(r.yaw_deg), yaw_deg_mod=r.yaw_deg % sym_deg,
            dims_mm=r.dims_mm, fill=1.0, size_err=0.0, n_points=10, points_xyz=dummy_pts,
        )
        for r in inst
    ]
    amb_list = [
        AmbiguousCluster(xy_m=a.xy_m, span_mm=a.dims_mm, top_z_m=a.top_mm / 1000.0, fill=1.0,
                         n_est=a.n_est, candidate_counts=[a.n_est], candidate_axes_deg=[], reason=a.status)
        for a in amb
    ]
    ep = EpochResult(
        policy="v2", status="ok", primary_role="wrist", fallback_mode="wrist_only", stamp_ns=0,
        poses=fits, pose_sources=["v2"] * len(fits), ambiguous_clusters=amb_list, slot_occupancies=[],
        dwell=None, stage_timings_ms={"total_ms": round(ms, 3)}, evidence_log={}, hm_grid=None, label_map=lab,
    )
    # Build minimal grid adapter for label_map lookup
    class _GridWrap:
        def __init__(self, g):
            self.g = g; self.ny = g.ny; self.nx = g.nx
        def world_to_ij(self, xy):
            return self.g.ij(xy)
    ep.hm_grid = _GridWrap(grid)  # type: ignore[assignment]
    sc = score_epoch_against_labels(t_rel_s, ep, labels, sym_deg) if labels else None
    return {
        "t": round(t_rel_s, 3),
        "status": "ok",
        "n_out": len(fits),
        "n_amb": len(amb_list),
        "stage_timings_ms": {"total_ms": round(ms, 3)},
        "score": sc,
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Shared matched comparison of WP3 perception policies on one ROS 2 SQLite bag.")
    ap.add_argument("bag", help="Path to .db3 file or bag folder")
    ap.add_argument("--out", required=True, help="Output folder inside WP3-Live-Implementation/results/...")
    ap.add_argument("--config", default=str(IMPL_ROOT / "ros2_ws/src/wp3_live_perception/config/printed_jenga_live.json"))
    ap.add_argument("--frame", choices=("tf", "board", "plane"), default="tf")
    ap.add_argument("--tf-json", default="", help="Optional recalibration JSON supplying diagnostics.{wrist,tripod}.board_from_optical")
    ap.add_argument("--register", nargs=2, type=float, default=(0.0, 3.0))
    ap.add_argument("--board", default="8x6")
    ap.add_argument("--pitch-mm", type=float, default=20.0)
    ap.add_argument("--labels", default="")
    ap.add_argument("--labels-bag", default="")
    ap.add_argument("--policies", default="depth_only,astra_permissive,merged_corroborated,merged_wrist_rgb,whole_partition_mv,generic_watershed")
    ap.add_argument("--views", default="wrist,both", help="View ablations when side camera exists: wrist,side,both")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--warmup-frames", type=int, default=5, help="Initial OK frames excluded from timing p50/p95/p99 percentiles")
    ap.add_argument("--temporal-skip", choices=("off", "on"), default="off", help="Enable stationary-frame ROI skip during main sweep")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / "predictions"; pred_dir.mkdir(exist_ok=True)
    ov_dir = out_dir / "overlays"; ov_dir.mkdir(exist_ok=True)

    cfg = load_runtime_config(args.config)
    bag = load_bag_fast(args.bag, [("wrist", cfg.wrist.namespace), ("side", cfg.side.namespace)], use_aligned_depth=True)

    wrist_cam_data = bag.cameras["wrist"]
    side_cam_data = bag.cameras["side"]
    cam_cache = CameraModelCache()
    cam_wrist = None
    if wrist_cam_data.info is not None:
        i_w = wrist_cam_data.info
        cam_wrist = cam_cache.get("wrist", i_w["width"], i_w["height"], i_w["K"], i_w["D"], i_w["frame_id"], cfg.wrist.depth_scale_m)
    cam_side = None
    if side_cam_data.info is not None:
        i_s = side_cam_data.info
        cam_side = cam_cache.get("tripod", i_s["width"], i_s["height"], i_s["K"], i_s["D"], i_s["frame_id"], cfg.side.depth_scale_m)

    if cam_wrist is None and cam_side is None:
        raise RuntimeError(f"Neither wrist nor side camera streams found in {args.bag}")

    T_wrist_tbl = None
    T_side_tbl = None
    reg_info: dict[str, Any] = {"frame_mode": args.frame}
    ref_plane = None
    triples = np.random.default_rng(18).integers(0, 2048, size=(120, 3))

    if args.tf_json:
        tf_data = json.loads(Path(args.tf_json).read_text(encoding="utf-8"))
        diag = tf_data.get("diagnostics", {})
        if cam_wrist is not None and "wrist" in diag and "board_from_optical" in diag["wrist"]:
            T_wrist_tbl = np.asarray(diag["wrist"]["board_from_optical"], dtype=np.float64).reshape(4, 4)
            ref_plane = (T_wrist_tbl[2, :3].copy(), float(T_wrist_tbl[2, 3]))
        if cam_side is not None and "tripod" in diag and "board_from_optical" in diag["tripod"]:
            T_side_tbl = np.asarray(diag["tripod"]["board_from_optical"], dtype=np.float64).reshape(4, 4)
        reg_info["tf_json"] = str(args.tf_json)
    elif args.frame == "tf":
        if cam_wrist is not None:
            T_wrist_tbl = chain_tf(bag.tf_static, cfg.frame_id, cam_wrist.frame_id)
        if cam_side is not None:
            try:
                T_side_tbl = chain_tf(bag.tf_static, cfg.frame_id, cam_side.frame_id)
            except KeyError:
                T_side_tbl = None
    elif args.frame == "board" and cam_wrist is not None:
        cols, rows = (int(x) for x in args.board.lower().split("x"))
        win_w = [p for p in wrist_cam_data.pairs if args.register[0] <= (p["rx_ns"] - bag.t0_ns) * 1e-9 <= args.register[1]]
        T_wrist_tbl, ref_plane, b_rep = _register_board_T(win_w, wrist_cam_data.info, cam_wrist, cols, rows, args.pitch_mm / 1000.0)
        reg_info["board_registration_wrist"] = b_rep
        if cam_side is not None and side_cam_data.pairs:
            win_s = [p for p in side_cam_data.pairs if args.register[0] <= (p["rx_ns"] - bag.t0_ns) * 1e-9 <= args.register[1]]
            if len(win_s) >= 3:
                T_side_0, _, s_rep = _register_board_T(win_s, side_cam_data.info, cam_side, cols, rows, args.pitch_mm / 1000.0)
                paired_win = _pair_side_to_wrist(win_w, side_cam_data.pairs, cfg.thresholds.max_pair_skew_s)
                reg_pairs = [(wp, sp) for wp, sp in zip(win_w, paired_win) if sp is not None]
                sample_pairs = reg_pairs[:: max(1, len(reg_pairs) // 5)][:5]
                T_side_tbl, ori_chk = _disambiguate_side_board_180(T_wrist_tbl, T_side_0, sample_pairs, cam_wrist, cam_side)
                reg_info["board_registration_side"] = s_rep
                reg_info["side_orientation_check"] = ori_chk

    # Schedule identical input epochs across ALL methods
    base_pairs = wrist_cam_data.pairs if wrist_cam_data.pairs else side_cam_data.pairs
    paired_side = _pair_side_to_wrist(wrist_cam_data.pairs, side_cam_data.pairs, cfg.thresholds.max_pair_skew_s) if wrist_cam_data.pairs else base_pairs

    labels = load_ground_truth_labels(args.labels, args.labels_bag or None) if args.labels else []
    epochs: list[dict[str, Any]] = []
    for idx, wp in enumerate(base_pairs):
        t_rel_s = (wp["rx_ns"] - bag.t0_ns) * 1e-9
        # Only score frames inside label windows if labels provided (plus ALL frames if unscored)
        if args.max_frames and len(epochs) >= args.max_frames:
            break
        epochs.append({
            "idx": idx,
            "t_rel_s": t_rel_s,
            "wrist": wp if wrist_cam_data.pairs else None,
            "side": paired_side[idx] if wrist_cam_data.pairs else wp,
        })

    # Pre-compute dwell gate ONCE monotonically across sorted streams so bounded deques don't evict mid-bag
    dwell_by_idx: list[Any] = [None] * len(epochs)
    if (bag.imu_gyro or bag.joint_states) and cam_wrist is not None:
        js_sorted = sorted(bag.joint_states, key=lambda x: x[0])
        gyro_sorted = sorted(bag.imu_gyro, key=lambda x: x[0])
        dwell_gate = WristDwellGate(require_fk=bool(js_sorted))
        j_cur = 0
        g_cur = 0
        for ep_idx, ep in enumerate(epochs):
            wp = ep["wrist"]
            if wp is None:
                continue
            stamp_s = wp["header_ns"] * 1e-9
            cutoff_s = stamp_s + 0.03
            while j_cur < len(js_sorted) and js_sorted[j_cur][0] <= cutoff_s:
                dwell_gate.push_joint_state(js_sorted[j_cur][0], js_sorted[j_cur][1])
                j_cur += 1
            while g_cur < len(gyro_sorted) and gyro_sorted[g_cur][0] <= cutoff_s:
                dwell_gate.push_imu_gyro(gyro_sorted[g_cur][0], gyro_sorted[g_cur][1])
                g_cur += 1
            gray_w = cv2.cvtColor(wp["rgb"], cv2.COLOR_RGB2GRAY)
            dwell_by_idx[ep_idx] = dwell_gate.evaluate(stamp_s, gray_w)

    view_modes = [v.strip() for v in args.views.split(",") if v.strip()]
    if cam_side is None or T_side_tbl is None:
        view_modes = [v for v in view_modes if v == "wrist"] or ["wrist"]
    if cam_wrist is None:
        view_modes = ["side"]

    policy_list = [p.strip() for p in args.policies.split(",") if p.strip()]
    sym_deg = cfg.block.yaw_symmetry_deg

    all_summaries: dict[str, Any] = {}
    timings_export: dict[str, Any] = {}
    run_cfg = replace(cfg, enable_temporal_skip=(args.temporal_skip == "on"))

    def _run_sweep_variant(run_key: str, vmode: str, pol: str, use_cfg, synth_dwell_if_missing: bool = False) -> None:
        tracker = WP3TemporalTracker(use_cfg.block, use_cfg.thresholds)
        tcache = EpochTemporalCache()
        dwell_fallback = WristDwellGate(require_fk=False) if synth_dwell_if_missing else None
        recs: list[dict[str, Any]] = []
        saved_ov = 0
        with open(pred_dir / f"{run_key}.jsonl", "w", encoding="utf-8") as pfh:
            for ep_idx, ep in enumerate(epochs):
                w_vi = None
                s_vi = None
                if vmode in ("wrist", "both") and ep["wrist"] is not None and cam_wrist is not None:
                    wp = ep["wrist"]
                    w_vi = ViewInput(
                        role="wrist", cam=cam_wrist, rgb_img=wp["rgb"], depth_raw=wp["depth_u16"],
                        T_table_cam=T_wrist_tbl, stamp_ns=wp["header_ns"], rx_ns=wp["rx_ns"],
                        pose_frame_mode="plane" if args.frame == "plane" else "table",
                    )
                if vmode in ("side", "both") and ep["side"] is not None and cam_side is not None and T_side_tbl is not None:
                    sp = ep["side"]
                    s_vi = ViewInput(
                        role="tripod", cam=cam_side, rgb_img=sp["rgb"], depth_raw=sp["depth_u16"],
                        T_table_cam=T_side_tbl, stamp_ns=sp["header_ns"], rx_ns=sp["rx_ns"],
                        pose_frame_mode="table",
                    )

                dwell = dwell_by_idx[ep_idx] if w_vi is not None else None
                if dwell is None and w_vi is not None and dwell_fallback is not None:
                    gray_w = cv2.cvtColor(ep["wrist"]["rgb"], cv2.COLOR_RGB2GRAY)
                    dwell = dwell_fallback.evaluate(w_vi.stamp_ns * 1e-9, gray_w)

                res = run_perception_epoch(
                    wrist=w_vi, side=s_vi, config=use_cfg, policy=pol, dwell=dwell,
                    ref_plane=ref_plane, rng_triples=triples, epoch_id=ep_idx + 1,
                    temporal_cache=tcache,
                )
                side_fresh = s_vi is not None and res.fallback_mode == "dual_view"
                tracks = tracker.update(
                    res.poses, res.pose_sources, res.stamp_ns, ep["t_rel_s"],
                    side_fresh_global=side_fresh, pose_side_visible=res.pose_side_visible,
                )
                sc = score_epoch_against_labels(ep["t_rel_s"], res, labels, sym_deg) if labels else None

                rec = {
                    "idx": ep["idx"],
                    "t": round(ep["t_rel_s"], 3),
                    "status": res.status,
                    "fallback_mode": res.fallback_mode,
                    "n_out": len(res.poses),
                    "n_amb": len(res.ambiguous_clusters),
                    "stage_timings_ms": res.stage_timings_ms,
                    "score": sc,
                }
                recs.append(rec)
                pfh.write(json.dumps({
                    "t": rec["t"],
                    "status": res.status,
                    "poses": [{"xy_m": p.xy_m, "z_m": p.z_center_m, "yaw_deg": p.yaw_deg_mod} for p in res.poses],
                    "ambiguous": [{"xy_m": a.xy_m, "n_est": a.n_est, "reason": a.reason} for a in res.ambiguous_clusters],
                    "slots": len(res.slot_occupancies),
                    "timings_ms": res.stage_timings_ms,
                }) + "\n")

                if saved_ov < 3 and res.status == "ok" and (sc is not None or not labels) and (saved_ov == 0 or ep_idx == len(epochs) // 2 or res.ambiguous_clusters):
                    ov = render_overlay(w_vi.rgb_img if w_vi else None, s_vi.rgb_img if s_vi else None, res, tracks, use_cfg.block)
                    cv2.imwrite(str(ov_dir / f"{run_key}_f{ep['idx']:04d}.png"), ov)
                    saved_ov += 1

        summ = aggregate_method_summary(recs, len(epochs), wrist_cam_data.dropped, warmup_frames=args.warmup_frames)
        all_summaries[run_key] = summ
        timings_export[run_key] = summ.get("timings_ms", {})

    for vmode in view_modes:
        for pol in policy_list:
            # Side-only doesn't benefit from sweeping wrist RGB policies beyond depth_only & merged_corroborated
            if vmode == "side" and pol not in ("depth_only", "merged_corroborated", "generic_watershed"):
                continue
            _run_sweep_variant(f"{vmode}__{pol}", vmode, pol, run_cfg, synth_dwell_if_missing=False)

    # Also measure live stationary-ROI skip throughput explicitly alongside cold per-frame compute
    if args.temporal_skip == "off" and "whole_partition_mv" in policy_list:
        if "both" in view_modes and cam_side is not None and T_side_tbl is not None:
            _run_sweep_variant(
                "both__whole_partition_mv_live_skip", "both", "whole_partition_mv",
                replace(cfg, enable_temporal_skip=True), synth_dwell_if_missing=True,
            )
        elif "wrist" in view_modes:
            _run_sweep_variant(
                "wrist__whole_partition_mv_live_skip", "wrist", "whole_partition_mv",
                replace(cfg, enable_temporal_skip=True), synth_dwell_if_missing=True,
            )

    # Run unmodified frozen v1 controls (WP3-Live-Implementation) when present on disk
    if cam_wrist is not None and wrist_cam_data.pairs:
        v1_res = _try_run_frozen_v1(
            epochs, cam_wrist, cam_side, T_wrist_tbl, T_side_tbl, args.config,
            labels, sym_deg, args.frame, dwell_by_idx, wrist_cam_data.dropped,
            warmup_frames=args.warmup_frames,
        )
        if v1_res.get("available"):
            for k_v1, val_v1 in v1_res.items():
                if k_v1 != "available":
                    all_summaries[k_v1] = val_v1
                    timings_export[k_v1] = val_v1.get("timings_ms", {})

    # Run unmodified frozen v2 comparators if wrist stream is present
    if cam_wrist is not None and wrist_cam_data.pairs:
        v2_res = _try_run_frozen_v2(
            epochs, cam_wrist, T_wrist_tbl, cfg, labels, sym_deg, args.frame, wrist_cam_data.dropped,
            warmup_frames=args.warmup_frames,
        )
        if v2_res.get("available"):
            all_summaries["wrist__v2_H1"] = v2_res["v2_H1"]
            all_summaries["wrist__v2_H2"] = v2_res["v2_H2"]
            timings_export["wrist__v2_H1"] = v2_res["v2_H1"].get("timings_ms", {})
            timings_export["wrist__v2_H2"] = v2_res["v2_H2"].get("timings_ms", {})

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "bag": str(bag.bag_path),
        "config_path": str(args.config),
        "config_hash": config_hash(run_cfg),
        "warmup_frames": args.warmup_frames,
        "temporal_skip_sweep": args.temporal_skip,
        "registration": reg_info,
        "file_sha256": sha256_files(list((IMPL_ROOT / "wp3_live").glob("*.py")) + [Path(__file__)]),
    }
    (out_dir / "manifest_sha256.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "timings.json").write_text(json.dumps(timings_export, indent=2), encoding="utf-8")
    (out_dir / "comparison_report.json").write_text(json.dumps({"manifest": manifest, "methods": all_summaries}, indent=2), encoding="utf-8")

    # Build concise decision_table.md
    lines = [
        f"# Matched Comparison Report — `{Path(args.bag).name}`",
        "",
        f"- **Profile:** `{cfg.profile_name}` (`frame={args.frame}`, `config_hash={manifest['config_hash']}`, `warmup_excluded={args.warmup_frames}`)",
        f"- **Epochs:** `{len(epochs)}` | **Wrist dropped:** `{wrist_cam_data.dropped}` | **Side pairs:** `{len(side_cam_data.pairs)}`",
        "",
        f"| Run (`view__policy`) | OK / Total | Membership 15 mm | Merge Fr | False-Conf % | Legacy 15 mm (Extra) | XY p50/p95 mm | Total ms p50/p95/p99 (excl {args.warmup_frames} warmup) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k, s in all_summaries.items():
        t_tot = s.get("timings_ms", {}).get("total_ms", {})
        tms = f"{t_tot.get('p50', '-')}/{t_tot.get('p95', '-')}/{t_tot.get('p99', '-')}"
        sc = s.get("scored")
        if sc:
            xy = sc.get("xy_err_mm_15") or {}
            xys = f"{xy.get('median', '-')}/{xy.get('p95', '-')}"
            lines.append(
                f"| `{k}` | {s['frames_ok']}/{s['frames_total']} | "
                f"{sc['membership_hits_15']}/{sc['opportunities']} ({sc['membership_recall_pct']}%) | "
                f"{sc['frames_with_merge']} | {sc['false_confident_rate_pct']}% | "
                f"{sc['legacy_hit_15']}/{sc['opportunities']} ({sc['extra_15']}) | {xys} | {tms} |"
            )
        else:
            lines.append(f"| `{k}` | {s['frames_ok']}/{s['frames_total']} | unscored | - | - | - | - | {tms} |")
    md_text = "\n".join(lines) + "\n"
    (out_dir / "decision_table.md").write_text(md_text, encoding="utf-8")
    print(md_text)


if __name__ == "__main__":
    main()

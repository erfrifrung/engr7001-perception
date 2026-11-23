"""Shared detector entry point with interchangeable split/corroboration policies and per-stage timings."""
from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any

import cv2
import numpy as np

from .config import RuntimeConfig
from .dwell_gate import DwellStatus
from .geometry_cache import (
    CameraModel,
    check_frozen_plane,
    deproject_roi_points,
    filter_robot_links,
    fit_table_plane_fast,
    plane_to_local_table_tf,
)
from .heightmap_core import HeightmapGrid, build_heightmap_fast, extract_height_band_components
from .proposals import (
    BlockFit,
    ImageProjectionContext,
    PartitionHypothesis,
    enumerate_partition_hypotheses,
    fit_cuboid_top,
    segment_scalar_watershed,
)
from .side_verifier import (
    CroppedSilhouetteVerifier,
    SideViewData,
    SideViewVerifier,
    SlotOccupancy,
    detect_side_facade_slots,
)


ALL_POLICIES = (
    "depth_only",
    "astra_permissive",
    "merged_corroborated",
    "merged_wrist_rgb",
    "generic_watershed",
)


@dataclass
class ViewInput:
    """One synchronised/selected camera observation ready for the shared detector core."""

    role: str                    # "wrist" | "tripod" | "third"
    cam: CameraModel
    rgb_img: np.ndarray          # (H, W, 3) uint8 RGB
    depth_raw: np.ndarray        # (H, W) uint16 mm or float32 m
    T_table_cam: np.ndarray | None
    stamp_ns: int
    rx_ns: int
    is_fresh: bool = True
    is_frozen: bool = False
    pose_frame_mode: str = "table"   # "table" | "plane"


@dataclass
class AmbiguousCluster:
    xy_m: tuple[float, float]
    span_mm: tuple[float, float]
    top_z_m: float
    fill: float
    n_est: int
    candidate_counts: list[int]
    candidate_axes_deg: list[float]
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpochResult:
    policy: str
    status: str                  # "ok" | "no_plane" | "roi_empty" | "camera_moved" | "unqualified"
    primary_role: str
    fallback_mode: str           # "dual_view" | "wrist_only" | "side_only" | "degraded"
    stamp_ns: int
    poses: list[BlockFit]
    pose_sources: list[str]
    ambiguous_clusters: list[AmbiguousCluster]
    slot_occupancies: list[SlotOccupancy]
    dwell: DwellStatus | None
    stage_timings_ms: dict[str, float]
    evidence_log: dict[str, Any]
    T_table_primary: np.ndarray | None = None
    primary_img_ctx: ImageProjectionContext | None = None
    side_img_ctx: ImageProjectionContext | None = None
    hm_grid: HeightmapGrid | None = None
    label_map: np.ndarray | None = None


def _select_primary_and_side(
    wrist: ViewInput | None,
    side: ViewInput | None,
    dwell: DwellStatus | None,
) -> tuple[ViewInput | None, ViewInput | None, str]:
    wrist_ok = wrist is not None and wrist.is_fresh and not wrist.is_frozen
    side_ok = side is not None and side.is_fresh and not side.is_frozen
    wrist_moving = wrist_ok and (dwell is not None and not dwell.stationary)

    if wrist_ok and side_ok and not wrist_moving:
        return wrist, side, "dual_view"
    if wrist_ok and not wrist_moving:
        return wrist, None, "wrist_only"
    if side_ok:
        return side, None, "side_only"
    if wrist_ok:
        return wrist, side if side_ok else None, "wrist_unqualified"
    return None, None, "no_valid_stream"


def _resolve_oversized_region(
    policy: str,
    pts_k: np.ndarray,
    hyps: list[PartitionHypothesis],
    n_est: int,
    block_l_m: float,
    block_w_m: float,
    require_unique_square: bool,
) -> tuple[list[BlockFit], list[str], AmbiguousCluster | None, dict[str, Any]]:
    """Apply interchangeable policy logic over the EXACT same enumerated multi-hypothesis evidence."""
    top_z = float(np.median(pts_k[:, 2]))
    cen_xy = (float(pts_k[:, 0].mean()), float(pts_k[:, 1].mean()))
    if not hyps:
        return [], [], AmbiguousCluster(
            xy_m=cen_xy, span_mm=(0.0, 0.0), top_z_m=top_z, fill=0.0, n_est=n_est,
            candidate_counts=[n_est], candidate_axes_deg=[], reason="no_valid_cuboid_partition",
        ), {"hypotheses": 0}

    def diag_summary(h: PartitionHypothesis) -> dict[str, Any]:
        return {
            "axis_deg": h.axis_deg,
            "k": h.k_blocks,
            "score": round(h.score, 3),
            "depth_step": h.has_validated_depth_step,
            "depth_trough": h.has_depth_trough,
            "rgb_ok": h.has_strong_rgb,
            "rgb_cont": round(h.mean_rgb_continuity, 3),
            "side_corr": h.has_side_corroboration,
            "side_contra": h.has_side_contradiction,
            "periodic_alias": h.has_periodic_alias,
        }

    if policy == "depth_only":
        eligible = [h for h in hyps if h.has_validated_depth_step]
    elif policy == "astra_permissive":
        # Mirrors heightmap_seam_pipeline.py:653 (depth_step OR depth_trough OR rgb_ok)
        eligible = [h for h in hyps if (h.has_validated_depth_step or h.has_depth_trough or h.has_strong_rgb)]
    elif policy == "merged_corroborated":
        # LEADING CANDIDATE: validated depth step OR (strong RGB + side/motion corroboration, no stripe alias, no contradiction)
        eligible = [
            h for h in hyps
            if h.has_validated_depth_step
            or (
                h.has_strong_rgb
                and h.has_side_corroboration
                and not h.has_side_contradiction
                and not h.has_periodic_alias
            )
        ]
    elif policy == "merged_wrist_rgb":
        # Controlled ablation: allows high-continuity texture-guarded RGB cuts without side corroboration,
        # while still respecting an explicit side-view CONTRADICTION or periodic stripe alias.
        eligible = [
            h for h in hyps
            if h.has_validated_depth_step
            or (
                h.has_strong_rgb
                and h.mean_rgb_continuity >= 0.70
                and not h.has_side_contradiction
                and not h.has_periodic_alias
            )
        ]
    else:
        eligible = []

    if not eligible:
        reason = (
            "side_contradicted_rgb_cut" if any(h.has_side_contradiction for h in hyps)
            else ("periodic_stripe_alias_rejected" if any(h.has_periodic_alias for h in hyps)
                  else ("uncorroborated_coplanar_rgb_only" if any(h.has_strong_rgb for h in hyps)
                        else "unresolved_flush_joint"))
        )
        return [], [], AmbiguousCluster(
            xy_m=cen_xy, span_mm=(0.0, 0.0), top_z_m=top_z, fill=0.0, n_est=n_est,
            candidate_counts=sorted({h.k_blocks for h in hyps}),
            candidate_axes_deg=sorted({h.axis_deg for h in hyps}),
            reason=reason,
            evidence={"candidates": [diag_summary(h) for h in hyps[:4]]},
        ), {"hypotheses": len(hyps), "eligible": 0, "reason": reason}

    # Check competing orthogonal 90-degree square partitions (e.g., 75x75 mm 3-block layouts!)
    best = eligible[0]
    if require_unique_square and len(eligible) >= 2:
        for alt in eligible[1:]:
            d_ang = min(abs(alt.axis_deg - best.axis_deg), 180.0 - abs(alt.axis_deg - best.axis_deg))
            # If an orthogonal (roughly 90-deg) partition has close evidence score, refuse to guess!
            if abs(d_ang - 90.0) <= 20.0 and (best.score - alt.score) < 0.08:
                return [], [], AmbiguousCluster(
                    xy_m=cen_xy, span_mm=(75.0, 75.0), top_z_m=top_z, fill=0.0, n_est=n_est,
                    candidate_counts=sorted({h.k_blocks for h in eligible}),
                    candidate_axes_deg=sorted({best.axis_deg, alt.axis_deg}),
                    reason="ambiguous_orthogonal_square_partitions",
                    evidence={"best": diag_summary(best), "competing": diag_summary(alt)},
                ), {"hypotheses": len(hyps), "eligible": len(eligible), "reason": "ambiguous_square"}

    src_tag = (
        "split:depth" if best.has_validated_depth_step
        else ("split:rgb+side" if best.has_side_corroboration else "split:rgb_only")
    )
    return list(best.child_fits), [src_tag] * len(best.child_fits), None, {
        "hypotheses": len(hyps),
        "eligible": len(eligible),
        "chosen": diag_summary(best),
    }


def run_perception_epoch(
    *,
    wrist: ViewInput | None,
    side: ViewInput | None,
    config: RuntimeConfig,
    policy: str = "merged_corroborated",
    dwell: DwellStatus | None = None,
    robot_links: dict | None = None,
    ref_plane: tuple[np.ndarray, float] | None = None,
    rng_triples: np.ndarray | None = None,
    epoch_id: int = 0,
    extra_step_check: bool = True,
) -> EpochResult:
    """Shared detector entry point for live node and bag comparison runner."""
    t_total_0 = time.perf_counter()
    if rng_triples is None:
        rng_triples = np.random.default_rng(18).integers(0, 2048, size=(120, 3))

    primary, secondary, fallback_mode = _select_primary_and_side(wrist, side, dwell)
    if primary is None:
        return EpochResult(
            policy=policy, status="no_valid_stream", primary_role="none", fallback_mode=fallback_mode,
            stamp_ns=0, poses=[], pose_sources=[], ambiguous_clusters=[], slot_occupancies=[],
            dwell=dwell, stage_timings_ms={"total_ms": round((time.perf_counter() - t_total_0) * 1000.0, 3)},
            evidence_log={"reason": fallback_mode},
        )

    th = config.thresholds
    block = config.block
    timings: dict[str, float] = {}

    # Stage 1 & 2: Deprojection + table-plane fit/check (ROI-cropped BEFORE dense XYZ!)
    t0 = time.perf_counter()
    T_tc = primary.T_table_cam
    plane_dist_mm = None
    plane_sup_pct = None

    # First check or fit table plane when in plane mode or verifying board stability
    use_precrop = config.use_roi_precrop and (T_tc is not None)
    deproj = deproject_roi_points(
        primary.depth_raw,
        primary.rgb_img,
        primary.cam,
        T_tc,
        config.roi_xyz_min,
        config.roi_xyz_max,
        min_depth_m=config.wrist.min_depth_m if primary.role == "wrist" else config.side.min_depth_m,
        max_depth_m=config.wrist.max_depth_m if primary.role == "wrist" else config.side.max_depth_m,
        pad_px=config.roi_pad_px,
        use_roi_precrop=use_precrop,
        extra_frac_crop=config.extra_roi_crop_frac if not use_precrop else None,
    )
    timings["deproject_roi_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)

    t_plane0 = time.perf_counter()
    pts_cam = deproj["pts_cam_f32"]
    pts_tbl = deproj["pts_tbl_f32"]
    rgb_pts = deproj["rgb_u8"]

    if T_tc is None or primary.pose_frame_mode == "plane":
        fit = fit_table_plane_fast(pts_cam.astype(np.float64), rng_triples, config.plane_tolerance_m, config.min_plane_support)
        if fit is None:
            timings["plane_ms"] = round((time.perf_counter() - t_plane0) * 1000.0, 3)
            timings["total_ms"] = round((time.perf_counter() - t_total_0) * 1000.0, 3)
            return EpochResult(
                policy=policy, status="no_plane", primary_role=primary.role, fallback_mode=fallback_mode,
                stamp_ns=primary.stamp_ns, poses=[], pose_sources=[], ambiguous_clusters=[],
                slot_occupancies=[], dwell=dwell, stage_timings_ms=timings,
                evidence_log={"crop_frac": round(deproj["crop_frac"], 3)},
            )
        n_cam, d_cam, sup, med_mm = fit
        plane_dist_mm = med_mm
        plane_sup_pct = round(100.0 * sup, 1)
        if T_tc is None:
            T_tc = plane_to_local_table_tf(n_cam, d_cam)
            R = T_tc[:3, :3].astype(np.float32)
            t = T_tc[:3, 3].astype(np.float32)
            pts_tbl = pts_cam @ R.T + t
            rmin = np.asarray(config.roi_xyz_min, dtype=np.float32)
            rmax = np.asarray(config.roi_xyz_max, dtype=np.float32)
            in_box = (
                (pts_tbl[:, 0] >= rmin[0]) & (pts_tbl[:, 0] <= rmax[0])
                & (pts_tbl[:, 1] >= rmin[1]) & (pts_tbl[:, 1] <= rmax[1])
                & (pts_tbl[:, 2] >= rmin[2]) & (pts_tbl[:, 2] <= rmax[2])
            )
            pts_tbl = pts_tbl[in_box]
            rgb_pts = rgb_pts[in_box]
    elif ref_plane is not None:
        ok_plane, sup, med_mm = check_frozen_plane(pts_cam.astype(np.float64), ref_plane[0], ref_plane[1], config.plane_tolerance_m, 0.20)
        plane_dist_mm = med_mm
        plane_sup_pct = round(100.0 * sup, 1)
        if not ok_plane:
            timings["plane_ms"] = round((time.perf_counter() - t_plane0) * 1000.0, 3)
            timings["total_ms"] = round((time.perf_counter() - t_total_0) * 1000.0, 3)
            return EpochResult(
                policy=policy, status="camera_moved", primary_role=primary.role, fallback_mode=fallback_mode,
                stamp_ns=primary.stamp_ns, poses=[], pose_sources=[], ambiguous_clusters=[],
                slot_occupancies=[], dwell=dwell, stage_timings_ms=timings,
                evidence_log={"plane_sup": sup, "plane_med_mm": med_mm},
            )
    timings["plane_ms"] = round((time.perf_counter() - t_plane0) * 1000.0, 3)

    # Stage 3: Robot self-filter
    t_rf0 = time.perf_counter()
    removed_robot = 0
    if config.enable_robot_self_filter and robot_links:
        pts_tbl, rgb_pts, removed_robot = filter_robot_links(
            pts_tbl, robot_links, rgb_pts, tcp_offset_z_m=config.tcp_offset_z_m,
        )
    timings["robot_filter_ms"] = round((time.perf_counter() - t_rf0) * 1000.0, 3)

    if len(pts_tbl) < 40:
        timings["total_ms"] = round((time.perf_counter() - t_total_0) * 1000.0, 3)
        return EpochResult(
            policy=policy, status="roi_empty", primary_role=primary.role, fallback_mode=fallback_mode,
            stamp_ns=primary.stamp_ns, poses=[], pose_sources=[], ambiguous_clusters=[],
            slot_occupancies=[], dwell=dwell, stage_timings_ms=timings,
            evidence_log={"removed_robot": removed_robot},
        )

    # Prepare side camera context & slot occupancies (available even when primary is wrist!)
    t_side0 = time.perf_counter()
    side_data: SideViewData | None = None
    side_slots: list[SlotOccupancy] = []
    if secondary is not None and secondary.T_table_cam is not None:
        sd_m = (
            secondary.depth_raw.astype(np.float32, copy=False)
            if np.issubdtype(secondary.depth_raw.dtype, np.floating)
            else secondary.depth_raw.astype(np.float32) * np.float32(secondary.cam.depth_scale_m)
        )
        s_ctx = ImageProjectionContext.from_rgb(secondary.rgb_img, secondary.cam.K_f64, secondary.cam.D_f64, secondary.T_table_cam)
        side_data = SideViewData(
            cam=secondary.cam, T_table_cam=secondary.T_table_cam, rgb_img=secondary.rgb_img,
            depth_m=sd_m, stamp_ns=secondary.stamp_ns, img_ctx=s_ctx,
            is_fresh=secondary.is_fresh, is_frozen=secondary.is_frozen,
        )
        side_slots = detect_side_facade_slots(
            side_data, block, th, layers=min(4, block.layers),
            tower_origin_xy=(config.row_origin_xyz_m[0], config.row_origin_xyz_m[1]),
            tower_yaw_rad=float(np.radians(config.row_yaw_deg)),
        )
    elif primary.role != "wrist" and primary.T_table_cam is not None:
        sd_m = (
            primary.depth_raw.astype(np.float32, copy=False)
            if np.issubdtype(primary.depth_raw.dtype, np.floating)
            else primary.depth_raw.astype(np.float32) * np.float32(primary.cam.depth_scale_m)
        )
        p_ctx = ImageProjectionContext.from_rgb(primary.rgb_img, primary.cam.K_f64, primary.cam.D_f64, primary.T_table_cam)
        s_only = SideViewData(
            cam=primary.cam, T_table_cam=primary.T_table_cam, rgb_img=primary.rgb_img,
            depth_m=sd_m, stamp_ns=primary.stamp_ns, img_ctx=p_ctx,
            is_fresh=primary.is_fresh, is_frozen=primary.is_frozen,
        )
        side_slots = detect_side_facade_slots(
            s_only, block, th, layers=min(4, block.layers),
            tower_origin_xy=(config.row_origin_xyz_m[0], config.row_origin_xyz_m[1]),
            tower_yaw_rad=float(np.radians(config.row_yaw_deg)),
        )
    side_verifier = SideViewVerifier(side_data, th)
    timings["side_facade_ms"] = round((time.perf_counter() - t_side0) * 1000.0, 3)

    # Stage 4: Orthographic fast O(N) heightmap + components
    t_hm0 = time.perf_counter()
    grid = HeightmapGrid.from_roi(config.roi_xyz_min[:2], config.roi_xyz_max[:2], th.cell_m)
    hm = build_heightmap_fast(pts_tbl, rgb_pts, grid, zlim=(config.roi_xyz_min[2], config.roi_xyz_max[2]))
    a1_cells = (block.length_m * block.width_m) / (th.cell_m ** 2)
    min_cells = max(8, int(round(th.min_area_frac * a1_cells)))
    components = extract_height_band_components(
        hm, block.height_m, th.band_lo_factor, th.band_hi_factor, min_area_cells=min_cells,
    )
    timings["heightmap_ms"] = round((time.perf_counter() - t_hm0) * 1000.0, 3)

    # Prepare lazy primary RGB projection context (zero full-frame image warping!)
    primary_ctx = ImageProjectionContext.from_rgb(primary.rgb_img, primary.cam.K_f64, primary.cam.D_f64, T_tc)

    # Generic scalar watershed control branch
    t_seg0 = time.perf_counter()
    confident_with_src: list[tuple[BlockFit, str]] = []
    ambiguous: list[AmbiguousCluster] = []
    split_logs: list[dict[str, Any]] = []
    row_yaw_rad = float(np.radians(config.row_yaw_deg))
    label_map = np.zeros((grid.ny, grid.nx), dtype=np.int32)

    if policy == "generic_watershed":
        fg_union = np.zeros((grid.ny, grid.nx), dtype=bool)
        for comp in components:
            fg_union |= comp["mask"]
        ws_labels, n_seeds = segment_scalar_watershed(hm.h_filled, hm.color_rgb, hm.color_valid, fg_union, step_barrier_m=th.depth_step_min_m)
        label_map = ws_labels
        cell_lut = ws_labels.reshape(-1)
        for sid in range(1, ws_labels.max() + 1):
            m_sid = ws_labels == sid
            if int(m_sid.sum()) < min_cells:
                continue
            pts_k = hm.pts_tbl[cell_lut[hm.pts_cell_flat] == sid]
            fit_single, diag = fit_cuboid_top(pts_k, block, row_yaw_rad=row_yaw_rad)
            if fit_single is not None:
                fit_single.cell_mask = m_sid
                confident_with_src.append((fit_single, "watershed_single"))
            else:
                ambiguous.append(AmbiguousCluster(
                    xy_m=(float(pts_k[:, 0].mean()), float(pts_k[:, 1].mean())),
                    span_mm=diag.get("dims_mm", (0.0, 0.0)),
                    top_z_m=float(np.median(pts_k[:, 2])),
                    fill=float(diag.get("fill", 0.0)),
                    n_est=max(1, int(round(int(m_sid.sum()) / max(1.0, a1_cells)))),
                    candidate_counts=[], candidate_axes_deg=[],
                    reason="watershed_unresolved_multi_or_noncuboid",
                ))
        timings["proposals_and_rgb_ms"] = round((time.perf_counter() - t_seg0) * 1000.0, 3)
    else:
        for comp in components:
            pts_k = comp["pts_tbl"]
            m_comp = comp["mask"]
            fit_single, diag = fit_cuboid_top(pts_k, block, row_yaw_rad=row_yaw_rad)
            if fit_single is not None:
                fit_single.cell_mask = m_comp
                confident_with_src.append((fit_single, "single"))
                label_map[m_comp] = len(confident_with_src)
                continue

            dims_mm = diag.get("dims_mm", (0.0, 0.0))
            area_ratio = float(comp["area_cells"]) / max(1.0, a1_cells)
            n_est = max(1, int(round(area_ratio / 0.85)))
            max_maj = block.major_m * 1000.0 * (1.0 + block.size_tol)
            max_min = block.minor_m * 1000.0 * (1.0 + block.size_tol)
            min_min = block.minor_m * 1000.0 * (1.0 - block.size_tol)
            is_oversized = (dims_mm[0] > max_maj or dims_mm[1] > max_min) and dims_mm[1] >= 0.72 * min_min

            if is_oversized:
                hyps = enumerate_partition_hypotheses(
                    pts_k,
                    None if policy == "depth_only" else primary_ctx,
                    block,
                    th,
                    row_yaw_rad=row_yaw_rad,
                    side_verifier=side_verifier if secondary is not None else None,
                )
                child_fits, child_srcs, amb, slog = _resolve_oversized_region(
                    policy, pts_k, hyps, n_est, block.length_m, block.width_m, config.require_unique_square_evidence,
                )
                split_logs.append(slog)
                if child_fits:
                    for cf, cs in zip(child_fits, child_srcs):
                        confident_with_src.append((cf, cs))
                        # Paint nearest label cells inside comp
                        if len(cf.points_xyz):
                            ri, rj = grid.world_to_ij(cf.points_xyz[:, :2])
                            ok_rc = (ri >= 0) & (ri < grid.ny) & (rj >= 0) & (rj < grid.nx)
                            label_map[ri[ok_rc], rj[ok_rc]] = len(confident_with_src)
                elif amb is not None:
                    amb.span_mm = dims_mm
                    amb.fill = float(diag.get("fill", 0.0))
                    ambiguous.append(amb)
        timings["proposals_and_rgb_ms"] = round((time.perf_counter() - t_seg0) * 1000.0, 3)

    # Stage 5: Optional cropped §7a silhouette/depth verifier (veto-only on sampled FRESH + splits)
    t_ver0 = time.perf_counter()
    verifier_stats = {"checked": 0, "vetoed": 0, "skipped_unseen": len(confident_with_src)}
    if config.use_side_verifier and side_data is not None and policy not in ("astra_permissive",):
        verifier = CroppedSilhouetteVerifier(th)
        kept, vetoed, verifier_stats = verifier.verify_confident_outputs(confident_with_src, side_data, block, epoch_id=epoch_id)
        for vf, vreason in vetoed:
            ambiguous.append(AmbiguousCluster(
                xy_m=vf.xy_m, span_mm=vf.dims_mm, top_z_m=vf.z_top_m, fill=vf.fill, n_est=1,
                candidate_counts=[1], candidate_axes_deg=[round(vf.yaw_deg_mod, 1)], reason=vreason,
            ))
        confident_with_src = kept
    timings["verifier_ms"] = round((time.perf_counter() - t_ver0) * 1000.0, 3)
    timings["total_ms"] = round((time.perf_counter() - t_total_0) * 1000.0, 3)

    poses = [f for f, _ in confident_with_src]
    pose_sources = [s for _, s in confident_with_src]
    return EpochResult(
        policy=policy,
        status="ok",
        primary_role=primary.role,
        fallback_mode=fallback_mode,
        stamp_ns=primary.stamp_ns,
        poses=poses,
        pose_sources=pose_sources,
        ambiguous_clusters=ambiguous,
        slot_occupancies=side_slots,
        dwell=dwell,
        stage_timings_ms=timings,
        evidence_log={
            "crop_frac": round(deproj["crop_frac"], 3),
            "bbox_2d": deproj["bbox_2d"],
            "pts_in_roi": int(len(pts_tbl)),
            "removed_robot_pts": removed_robot,
            "plane_dist_mm": plane_dist_mm,
            "plane_sup_pct": plane_sup_pct,
            "split_logs": split_logs,
            "verifier": verifier_stats,
        },
        T_table_primary=T_tc,
        primary_img_ctx=primary_ctx,
        side_img_ctx=side_data.img_ctx if side_data is not None else None,
        hm_grid=grid,
        label_map=label_map,
    )

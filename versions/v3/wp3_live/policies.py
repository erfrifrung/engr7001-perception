"""Shared detector entry point with interchangeable split/corroboration policies and per-stage timings."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import time
from typing import Any

import cv2
import numpy as np

from .config import BlockSpec, PolicyThresholds, RuntimeConfig, config_hash
from .dwell_gate import DwellStatus
from .fast_ops import configure_single_thread_libs, fast_median_1d
from .geometry_cache import (
    CameraModel,
    check_frozen_plane,
    deproject_roi_points,
    deproject_roi_points_fast,
    filter_robot_links,
    filter_robot_links_fast,
    fit_table_plane_fast,
    plane_to_local_table_tf,
    project_workspace_roi_2d,
)
from .heightmap_core import HeightmapGrid, build_heightmap_fast, extract_height_band_components
from .proposals import (
    BlockFit,
    ImageProjectionContext,
    PartitionHypothesis,
    _make_partition_candidate,
    enumerate_partition_hypotheses,
    evaluate_cut_cues,
    evaluate_warm_start_hypothesis,
    fit_cuboid_top,
    propose_hypotheses_from_2d_masks,
    segment_scalar_watershed,
)
from .side_verifier import (
    CroppedSilhouetteVerifier,
    SideViewData,
    SideViewVerifier,
    SlotOccupancy,
    TwoSideCutVerifier,
    compute_target_side_visibility,
    detect_side_facade_slots,
    score_partition_multiview,
)
from .temporal_cache import EpochTemporalCache, WarmStartEntry
from .tiling import ViewDepth
from .v3_regions import robot_unknown_cells, run_v3_regions

_VIEW_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="wp3_view")
_RNG_TRIPLES_DEFAULT = np.random.default_rng(18).integers(0, 2048, size=(120, 3))


ALL_POLICIES = (
    "depth_only",
    "astra_permissive",
    "merged_corroborated",
    "merged_wrist_rgb",
    "whole_partition_mv",
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
    pose_side_visible: list[bool] = field(default_factory=list)
    T_table_primary: np.ndarray | None = None
    primary_img_ctx: ImageProjectionContext | None = None
    side_img_ctx: ImageProjectionContext | None = None
    third_img_ctx: ImageProjectionContext | None = None
    hm_grid: HeightmapGrid | None = None
    label_map: np.ndarray | None = None


def _select_primary_and_side(
    wrist: ViewInput | None,
    side: ViewInput | None,
    dwell: DwellStatus | None,
    max_cross_view_skew_s: float = 0.08,
) -> tuple[ViewInput | None, ViewInput | None, str, dict[str, Any]]:
    wrist_ok = wrist is not None and wrist.is_fresh and not wrist.is_frozen
    side_ok = side is not None and side.is_fresh and not side.is_frozen
    wrist_moving = wrist_ok and (dwell is not None and not dwell.stationary)
    diag: dict[str, Any] = {}

    if wrist_ok and side_ok:
        skew_s = abs(int(wrist.stamp_ns) - int(side.stamp_ns)) * 1e-9  # type: ignore[union-attr]
        diag["cross_view_skew_ms"] = round(skew_s * 1000.0, 2)
        if skew_s > max_cross_view_skew_s:
            side_ok = False
            diag["side_rejected_reason"] = "cross_view_skew"

    if wrist_ok and side_ok and not wrist_moving:
        return wrist, side, "dual_view", diag
    if wrist_ok and not wrist_moving:
        return wrist, None, "wrist_only", diag
    if side_ok:
        return side, None, "side_only", diag
    if wrist_ok:
        return wrist, None, "wrist_unqualified", diag
    return None, None, "no_valid_stream", diag


def _resolve_oversized_region(
    policy: str,
    pts_k: np.ndarray,
    hyps: list[PartitionHypothesis],
    n_est: int,
    config: RuntimeConfig,
    primary_ctx: ImageProjectionContext | None,
    wrist_depth_m: np.ndarray | None,
    side_data: SideViewData | None,
    row_yaw_rad: float = 0.0,
    third_data: SideViewData | None = None,
) -> tuple[list[BlockFit], list[str], AmbiguousCluster | None, dict[str, Any], PartitionHypothesis | None]:
    """Apply interchangeable policy logic over the EXACT same enumerated multi-hypothesis evidence."""
    top_z = fast_median_1d(pts_k[:, 2])
    cen_xy = (float(pts_k[:, 0].mean()), float(pts_k[:, 1].mean()))
    if not hyps:
        return [], [], AmbiguousCluster(
            xy_m=cen_xy, span_mm=(0.0, 0.0), top_z_m=top_z, fill=0.0, n_est=n_est,
            candidate_counts=[n_est], candidate_axes_deg=[], reason="no_valid_cuboid_partition",
        ), {"hypotheses": 0}, None

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

    block = config.block
    th = config.thresholds
    require_unique_square = config.require_unique_square_evidence

    if policy == "whole_partition_mv":
        cand_hyps = [
            h for h in hyps
            if (not h.has_periodic_alias)
            and (not h.has_side_contradiction)
            and (h.has_validated_depth_step or h.has_side_corroboration or (h.has_strong_rgb and h.mean_rgb_continuity >= th.rgb_corroborated_continuity_min))
        ]
        if not cand_hyps:
            reason = (
                "side_contradicted_rgb_cut" if any(h.has_side_contradiction for h in hyps)
                else ("periodic_stripe_alias_rejected" if any(h.has_periodic_alias for h in hyps)
                      else "unresolved_flush_joint")
            )
            return [], [], AmbiguousCluster(
                xy_m=cen_xy, span_mm=(0.0, 0.0), top_z_m=top_z, fill=0.0, n_est=n_est,
                candidate_counts=sorted({h.k_blocks for h in hyps}),
                candidate_axes_deg=sorted({h.axis_deg for h in hyps}),
                reason=reason,
                evidence={"candidates": [diag_summary(h) for h in hyps[:4]]},
            ), {"hypotheses": len(hyps), "eligible": 0, "reason": reason}, None

        # Score intact envelope hypothesis + all valid candidate splits on mutual views
        intact_fit, _ = fit_cuboid_top(pts_k, block, row_yaw_rad=row_yaw_rad)
        if intact_fit is not None:
            mv_intact = score_partition_multiview([intact_fit], [], primary_ctx, wrist_depth_m, side_data, block, th, is_intact=True, third_side=third_data)["mv_score"]
        else:
            # Oversized intact region incurs cuboid size penalty proportional to excess span
            env_err = abs(n_est - 1.0)
            env_dummy = BlockFit(cen_xy, top_z - block.height_m / 2.0, top_z, row_yaw_rad, 0.0, (block.major_m * 1000.0, n_est * block.minor_m * 1000.0), 0.9, env_err, len(pts_k), pts_k)
            mv_intact = score_partition_multiview([env_dummy], [], primary_ctx, wrist_depth_m, side_data, block, th, is_intact=True, third_side=third_data)["mv_score"]

        scored = []
        for h in cand_hyps:
            mv = score_partition_multiview(h.child_fits, h.cuts, primary_ctx, wrist_depth_m, side_data, block, th, is_intact=False, third_side=third_data)
            scored.append((mv["mv_score"], mv, h))
        scored.sort(key=lambda item: -item[0])
        best_mv, best_mv_diag, best = scored[0]

        # On coplanar square clusters (width ~ length) with NO depth step and NO side corroboration,
        # withhold confident decision when square ambiguity guard is active (mirrors obscured square test).
        span_xy = np.ptp(pts_k[:, :2], axis=0)
        near_square_cluster = min(float(span_xy[0]), float(span_xy[1])) >= 0.78 * block.major_m
        if require_unique_square and near_square_cluster and (not best.has_validated_depth_step) and (not best.has_side_corroboration):
            return [], [], AmbiguousCluster(
                xy_m=cen_xy, span_mm=(75.0, 75.0), top_z_m=top_z, fill=0.0, n_est=n_est,
                candidate_counts=sorted({h.k_blocks for h in cand_hyps}),
                candidate_axes_deg=sorted({h.axis_deg for h in cand_hyps}),
                reason="uncorroborated_coplanar_square_mv",
                evidence={"best_mv": best_mv_diag, "best": diag_summary(best)},
            ), {"hypotheses": len(hyps), "eligible": len(cand_hyps), "reason": "uncorroborated_coplanar_square_mv"}, None

        if (best_mv - mv_intact) < th.mv_partition_margin:
            return [], [], AmbiguousCluster(
                xy_m=cen_xy, span_mm=(0.0, 0.0), top_z_m=top_z, fill=0.0, n_est=n_est,
                candidate_counts=[1, best.k_blocks], candidate_axes_deg=[best.axis_deg],
                reason="mv_intact_vs_split_ambiguous",
                evidence={"mv_intact": round(mv_intact, 3), "mv_split": best_mv_diag},
            ), {"hypotheses": len(hyps), "eligible": len(cand_hyps), "reason": "mv_intact_ambiguous"}, None

        for alt_mv, alt_diag, alt in scored[1:]:
            d_ang = min(abs(alt.axis_deg - best.axis_deg), 180.0 - abs(alt.axis_deg - best.axis_deg))
            if abs(d_ang - 90.0) <= 20.0 and (best_mv - alt_mv) < th.mv_partition_margin:
                return [], [], AmbiguousCluster(
                    xy_m=cen_xy, span_mm=(75.0, 75.0), top_z_m=top_z, fill=0.0, n_est=n_est,
                    candidate_counts=sorted({h.k_blocks for h in cand_hyps}),
                    candidate_axes_deg=sorted({best.axis_deg, alt.axis_deg}),
                    reason="mv_orthogonal_square_ambiguous",
                    evidence={"best_mv": best_mv_diag, "competing_mv": alt_diag},
                ), {"hypotheses": len(hyps), "eligible": len(cand_hyps), "reason": "mv_orthogonal_ambiguous"}, None

        src_tag = (
            "split:depth" if best.has_validated_depth_step
            else ("split:mv+side" if best.has_side_corroboration else "split:mv_rgb")
        )
        return list(best.child_fits), [src_tag] * len(best.child_fits), None, {
            "hypotheses": len(hyps), "eligible": len(cand_hyps), "mv": best_mv_diag, "chosen": diag_summary(best),
        }, best

    if policy == "depth_only":
        eligible = [h for h in hyps if h.has_validated_depth_step]
    elif policy == "astra_permissive":
        eligible = [h for h in hyps if (h.has_validated_depth_step or h.has_depth_trough or h.has_strong_rgb)]
    elif policy == "merged_corroborated":
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
        eligible = [
            h for h in hyps
            if h.has_validated_depth_step
            or (
                h.has_strong_rgb
                and h.mean_rgb_continuity >= th.rgb_corroborated_continuity_min
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
        ), {"hypotheses": len(hyps), "eligible": 0, "reason": reason}, None

    best = eligible[0]
    if require_unique_square and len(eligible) >= 2:
        for alt in eligible[1:]:
            d_ang = min(abs(alt.axis_deg - best.axis_deg), 180.0 - abs(alt.axis_deg - best.axis_deg))
            if abs(d_ang - 90.0) <= 20.0 and (best.score - alt.score) < 0.08:
                return [], [], AmbiguousCluster(
                    xy_m=cen_xy, span_mm=(75.0, 75.0), top_z_m=top_z, fill=0.0, n_est=n_est,
                    candidate_counts=sorted({h.k_blocks for h in eligible}),
                    candidate_axes_deg=sorted({best.axis_deg, alt.axis_deg}),
                    reason="ambiguous_orthogonal_square_partitions",
                    evidence={"best": diag_summary(best), "competing": diag_summary(alt)},
                ), {"hypotheses": len(hyps), "eligible": len(eligible), "reason": "ambiguous_square"}, None

    src_tag = (
        "split:depth" if best.has_validated_depth_step
        else ("split:rgb+side" if best.has_side_corroboration else "split:rgb_only")
    )
    return list(best.child_fits), [src_tag] * len(best.child_fits), None, {
        "hypotheses": len(hyps),
        "eligible": len(eligible),
        "chosen": diag_summary(best),
    }, best



def _occlusion_truncated_hypotheses(
    pts_k: np.ndarray,
    comp_mask: np.ndarray,
    h_filled: np.ndarray,
    grid: HeightmapGrid,
    primary_ctx: ImageProjectionContext | None,
    block: BlockSpec,
    th: PolicyThresholds,
    side_verifier: Any,
    *,
    ring_cells: int = 2,
    min_occluder_cells: int = 3,
) -> list[tuple[PartitionHypothesis, np.ndarray]]:
    """Hypotheses carving ONE fully visible block from an oversized component truncated by an occluder.

    Requirements (all must hold): the component is longer than one block along the cut normal but the
    remainder is shorter than a block (so no K=2/3 hypothesis exists); the carved side fits a cuboid;
    the cut shows a depth step, depth trough or native-RGB seam; the remainder's outline touches
    occluder cells (no depth, or taller than the component top by half a block); the carved block's
    own outline touches none. Returns (hypothesis, remainder point mask) pairs; acceptance is still
    decided by the policy in ``_resolve_oversized_region``.
    """
    xy = np.ascontiguousarray(pts_k[:, :2], dtype=np.float32)
    if len(xy) < 40:
        return []
    z_top = fast_median_1d(pts_k[:, 2])
    ri, rj = grid.world_to_ij(pts_k[:, :2])
    ok_ij = (ri >= 0) & (ri < grid.ny) & (rj >= 0) & (rj < grid.nx)
    occluder = (~np.isfinite(h_filled)) | (h_filled > z_top + 0.5 * block.height_m)
    occluder &= ~comp_mask
    kern = np.ones((2 * ring_cells + 1, 2 * ring_cells + 1), np.uint8)

    def ring_hits(sel: np.ndarray) -> int:
        cells = np.zeros(comp_mask.shape, dtype=np.uint8)
        cells[ri[sel & ok_ij], rj[sel & ok_ij]] = 1
        ring = cv2.dilate(cells, kern).astype(bool) & ~cells.astype(bool)
        return int(np.count_nonzero(ring & occluder))

    rect = cv2.minAreaRect(xy)
    box = cv2.boxPoints(rect)
    out: list[tuple[PartitionHypothesis, np.ndarray]] = []
    for v in (box[1] - box[0], box[2] - box[1]):
        nv = float(np.linalg.norm(v))
        if nv < 1e-6:
            continue
        d = (v / nv).astype(np.float32)
        perp_d = np.array([-d[1], d[0]], dtype=np.float32)
        along = xy @ d
        perp = xy @ perp_d
        lo_a, hi_a = (float(x) for x in np.percentile(along, (4.0, 96.0)))
        span = (hi_a - lo_a) + block.edge_loss_m
        axis_deg = float(np.degrees(np.arctan2(float(d[1]), float(d[0]))) % 180.0)
        for unit in sorted({block.major_m, block.minor_m}):
            remainder_len = span - unit
            if not (span > unit * (1.0 + block.size_tol) and remainder_len < unit * (1.0 - block.size_tol)):
                continue
            for from_low in (True, False):
                cut = lo_a + unit - 0.5 * block.edge_loss_m if from_low else hi_a - unit + 0.5 * block.edge_loss_m
                cue = evaluate_cut_cues(pts_k, along, perp, cut, d, perp_d, primary_ctx, th, 0)
                if not (cue.depth_step_ok or cue.depth_trough_ok or cue.rgb_ok):
                    continue
                ref = cut + (cue.best_offset_m if cue.rgb_ok else 0.0)
                child_sel = along < ref if from_low else along >= ref
                rem_sel = ~child_sel
                if np.count_nonzero(rem_sel) < 12:
                    continue
                fit, _ = fit_cuboid_top(pts_k[child_sel], block, row_yaw_rad=float(np.radians(axis_deg)))
                if fit is None:
                    continue
                if ring_hits(child_sel) > 0 or ring_hits(rem_sel) < min_occluder_cells:
                    continue
                hyp = _make_partition_candidate(axis_deg, 2, [cue], [fit], fit.size_err, side_verifier, [ref], block)
                out.append((hyp, rem_sel))
    out.sort(key=lambda item: -item[0].score)
    return out

def _v3_region_cache(temporal_cache, config: RuntimeConfig):
    """Region verdict cache lives on the caller's temporal cache (live node / replay); keyed by calib + config."""
    if temporal_cache is None or not config.v3_region_cache:
        return None
    from .tiling import RegionCache
    rc = getattr(temporal_cache, "v3_region_cache", None)
    if rc is None:
        rc = RegionCache()
        temporal_cache.v3_region_cache = rc
    rc.bind((config.calib_epoch, config_hash(config)))
    return rc


def _depth_to_float32_roi(
    depth_raw: np.ndarray,
    cam: CameraModel,
    bbox: tuple[int, int, int, int] | None,
) -> np.ndarray:
    if np.issubdtype(depth_raw.dtype, np.floating):
        return depth_raw.astype(np.float32, copy=False)
    if bbox is None:
        return depth_raw.astype(np.float32) * np.float32(cam.depth_scale_m)
    H, W = depth_raw.shape[:2]
    u0, v0, u1, v1 = bbox
    if u0 == 0 and v0 == 0 and u1 == W and v1 == H:
        return depth_raw.astype(np.float32) * np.float32(cam.depth_scale_m)
    out = np.zeros((H, W), dtype=np.float32)
    out[v0:v1, u0:u1] = depth_raw[v0:v1, u0:u1].astype(np.float32) * np.float32(cam.depth_scale_m)
    return out


def _prepare_side_view_and_slots(
    view: ViewInput,
    config: RuntimeConfig,
    block: BlockSpec,
    th: PolicyThresholds,
    *,
    detect_slots: bool = True,
) -> tuple[SideViewData, list[SlotOccupancy], tuple[int, int, int, int]]:
    assert view.T_table_cam is not None
    bbox_s = (
        project_workspace_roi_2d(view.cam, view.T_table_cam, config.roi_xyz_min, config.roi_xyz_max, pad_px=40)
        if config.use_roi_precrop
        else (0, 0, view.cam.width, view.cam.height)
    )
    if bbox_s is None:
        bbox_s = (0, 0, view.cam.width, view.cam.height)
    sd_m = _depth_to_float32_roi(view.depth_raw, view.cam, bbox_s)
    inv_bundle = view.cam.get_projection_bundle(view.T_table_cam)
    s_ctx = ImageProjectionContext.from_rgb(
        view.rgb_img, view.cam.K_f64, view.cam.D_f64, view.T_table_cam, inv_rvec_tvec=inv_bundle,
    )
    s_data = SideViewData(
        cam=view.cam, T_table_cam=view.T_table_cam, rgb_img=view.rgb_img,
        depth_m=sd_m, stamp_ns=view.stamp_ns, img_ctx=s_ctx,
        is_fresh=view.is_fresh, is_frozen=view.is_frozen,
    )
    slots: list[SlotOccupancy] = []
    if detect_slots:
        slots = detect_side_facade_slots(
            s_data, block, th, layers=min(4, block.layers),
            tower_origin_xy=(config.row_origin_xyz_m[0], config.row_origin_xyz_m[1]),
            tower_yaw_rad=float(np.radians(config.row_yaw_deg)),
        )
    return s_data, slots, bbox_s


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
    external_masks: list[np.ndarray] | None = None,
    temporal_cache: EpochTemporalCache | None = None,
    third: ViewInput | None = None,
) -> EpochResult:
    """Shared detector entry point for live node and bag comparison runner."""
    t_total_0 = time.perf_counter()
    if rng_triples is None:
        rng_triples = _RNG_TRIPLES_DEFAULT

    th = config.thresholds
    primary, secondary, fallback_mode, view_diag = _select_primary_and_side(
        wrist, side, dwell, max_cross_view_skew_s=th.max_cross_view_skew_s,
    )
    if primary is None:
        return EpochResult(
            policy=policy, status="no_valid_stream", primary_role="none", fallback_mode=fallback_mode,
            stamp_ns=0, poses=[], pose_sources=[], ambiguous_clusters=[], slot_occupancies=[],
            dwell=dwell, stage_timings_ms={"total_ms": round((time.perf_counter() - t_total_0) * 1000.0, 3)},
            evidence_log={"reason": fallback_mode, **view_diag},
        )

    third_for_cuts = False
    if third is not None:
        if policy in ("depth_only", "generic_watershed", "astra_permissive"):
            view_diag["third_cut_status"] = "policy_uses_no_third_cut_evidence"
        elif primary.role != "wrist" or primary.T_table_cam is None:
            view_diag["third_cut_status"] = "no_registered_primary_wrist"
        elif third.T_table_cam is None or not third.is_fresh or third.is_frozen:
            view_diag["third_cut_status"] = "unavailable_or_unregistered"
        else:
            skew_s = abs(primary.stamp_ns - third.stamp_ns) * 1e-9
            view_diag["third_skew_ms"] = round(skew_s * 1000.0, 2)
            third_for_cuts = skew_s <= th.max_cross_view_skew_s
            view_diag["third_cut_status"] = "eligible" if third_for_cuts else "cross_view_skew"

    block = config.block
    min_d_p = config.wrist.min_depth_m if primary.role == "wrist" else config.side.min_depth_m
    max_d_p = config.wrist.max_depth_m if primary.role == "wrist" else config.side.max_depth_m
    min_d_s = config.side.min_depth_m if (secondary is None or secondary.role != "wrist") else config.wrist.min_depth_m
    max_d_s = config.side.max_depth_m if (secondary is None or secondary.role != "wrist") else config.wrist.max_depth_m

    if temporal_cache is not None and config.enable_temporal_skip:
        cached_ep = temporal_cache.try_reuse_stationary_epoch(
            primary=primary,
            secondary=secondary,
            policy=policy,
            calib_epoch=config.calib_epoch,
            fallback_mode=fallback_mode,
            dwell=dwell,
            max_skips=config.temporal_skip_max_frames,
            depth_tol_m=config.temporal_skip_depth_tol_m,
            min_d_p=min_d_p,
            max_d_p=max_d_p,
            min_d_s=min_d_s,
            max_d_s=max_d_s,
            has_external_masks=bool(external_masks),
            third=third if third_for_cuts else None,
        )
        if cached_ep is not None:
            return cached_ep

    timings: dict[str, float] = {}
    side_future = None
    if secondary is not None and secondary.T_table_cam is not None and config.enable_parallel_views:
        side_future = _VIEW_EXECUTOR.submit(_prepare_side_view_and_slots, secondary, config, block, th, detect_slots=True)

    # Stage 1 & 2: Deprojection + table-plane fit/check (ROI-cropped BEFORE dense XYZ!)
    t0 = time.perf_counter()
    T_tc = primary.T_table_cam
    plane_dist_mm = None
    plane_sup_pct = None

    use_precrop = config.use_roi_precrop and (T_tc is not None)
    need_pts_cam = bool(T_tc is None or primary.pose_frame_mode == "plane" or ref_plane is not None)
    deproj = (deproject_roi_points_fast if config.v3_fast_deproject else deproject_roi_points)(
        primary.depth_raw,
        primary.rgb_img,
        primary.cam,
        T_tc,
        config.roi_xyz_min,
        config.roi_xyz_max,
        min_depth_m=min_d_p,
        max_depth_m=max_d_p,
        pad_px=config.roi_pad_px,
        use_roi_precrop=use_precrop,
        extra_frac_crop=config.extra_roi_crop_frac if not use_precrop else None,
        need_pts_cam=need_pts_cam,
    )
    timings["deproject_roi_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)

    def _finish_early(status: str, ev: dict[str, Any]) -> EpochResult:
        if side_future is not None:
            try:
                side_future.result()
            except Exception:
                pass
        timings["total_ms"] = round((time.perf_counter() - t_total_0) * 1000.0, 3)
        return EpochResult(
            policy=policy, status=status, primary_role=primary.role, fallback_mode=fallback_mode,
            stamp_ns=primary.stamp_ns, poses=[], pose_sources=[], ambiguous_clusters=[],
            slot_occupancies=[], dwell=dwell, stage_timings_ms=timings, evidence_log={**ev, **view_diag},
        )

    t_plane0 = time.perf_counter()
    pts_cam = deproj["pts_cam_f32"]
    pts_tbl = deproj["pts_tbl_f32"]
    rgb_pts = deproj["rgb_u8"]

    if T_tc is None or primary.pose_frame_mode == "plane":
        fit = fit_table_plane_fast(pts_cam.astype(np.float64), rng_triples, config.plane_tolerance_m, config.min_plane_support)
        if fit is None:
            timings["plane_ms"] = round((time.perf_counter() - t_plane0) * 1000.0, 3)
            return _finish_early("no_plane", {"crop_frac": round(deproj["crop_frac"], 3)})
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
            return _finish_early("camera_moved", {"plane_sup": sup, "plane_med_mm": med_mm})
    timings["plane_ms"] = round((time.perf_counter() - t_plane0) * 1000.0, 3)

    # Stage 3: Robot self-filter
    t_rf0 = time.perf_counter()
    removed_robot = 0
    if config.enable_robot_self_filter and robot_links:
        pts_tbl, rgb_pts, removed_robot = (filter_robot_links_fast if config.v3_fast_deproject else filter_robot_links)(
            pts_tbl, robot_links, rgb_pts, tcp_offset_z_m=config.tcp_offset_z_m,
        )
    timings["robot_filter_ms"] = round((time.perf_counter() - t_rf0) * 1000.0, 3)

    if len(pts_tbl) < 40:
        return _finish_early("roi_empty", {"removed_robot": removed_robot})

    # Stage 4: Orthographic fast O(N) heightmap + components (runs concurrently with side worker if active!)
    t_hm0 = time.perf_counter()
    grid = HeightmapGrid.from_roi(config.roi_xyz_min[:2], config.roi_xyz_max[:2], th.cell_m)
    hm = build_heightmap_fast(pts_tbl, rgb_pts, grid, zlim=(config.roi_xyz_min[2], config.roi_xyz_max[2]))
    a1_cells = (block.length_m * block.width_m) / (th.cell_m ** 2)
    min_cells = max(8, int(round(th.min_area_frac * a1_cells)))
    components = extract_height_band_components(
        hm, block.height_m, th.band_lo_factor, th.band_hi_factor, min_area_cells=min_cells,
        top_band_from_upper_cells=config.top_band_from_upper_cells,
    )
    timings["heightmap_ms"] = round((time.perf_counter() - t_hm0) * 1000.0, 3)

    # Collect or run side view & slot occupancies before proposal corroboration
    t_side0 = time.perf_counter()
    side_data: SideViewData | None = None
    side_slots: list[SlotOccupancy] = []
    side_bbox: tuple[int, int, int, int] | None = None
    if side_future is not None:
        side_data, side_slots, side_bbox = side_future.result()
    elif secondary is not None and secondary.T_table_cam is not None:
        side_data, side_slots, side_bbox = _prepare_side_view_and_slots(secondary, config, block, th, detect_slots=True)
    elif primary.role != "wrist" and primary.T_table_cam is not None:
        _, side_slots, _ = _prepare_side_view_and_slots(primary, config, block, th, detect_slots=True)
    side_verifier = SideViewVerifier(side_data, th)
    two_side_verifier = None
    third_data: SideViewData | None = None
    third_bbox: tuple[int, int, int, int] | None = None
    if third_for_cuts:
        third_data, _, third_bbox = _prepare_side_view_and_slots(third, config, block, th, detect_slots=False)
        two_side_verifier = TwoSideCutVerifier(side_verifier, SideViewVerifier(third_data, th))
    cut_verifier = two_side_verifier or (side_verifier if secondary is not None else None)
    timings["side_facade_ms"] = round((time.perf_counter() - t_side0) * 1000.0, 3)

    # Prepare lazy primary RGB projection context (zero full-frame image warping!)
    p_inv_bundle = primary.cam.get_projection_bundle(T_tc)
    primary_ctx = ImageProjectionContext.from_rgb(
        primary.rgb_img, primary.cam.K_f64, primary.cam.D_f64, T_tc, inv_rvec_tvec=p_inv_bundle,
    )

    # Generic scalar watershed control branch
    t_seg0 = time.perf_counter()
    confident_with_src: list[tuple[BlockFit, str]] = []
    ambiguous: list[AmbiguousCluster] = []
    split_logs: list[dict[str, Any]] = []
    warm_starts_new: list[WarmStartEntry] = []
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
                    top_z_m=fast_median_1d(pts_k[:, 2]),
                    fill=float(diag.get("fill", 0.0)),
                    n_est=max(1, int(round(int(m_sid.sum()) / max(1.0, a1_cells)))),
                    candidate_counts=[], candidate_axes_deg=[],
                    reason="watershed_unresolved_multi_or_noncuboid",
                ))
        timings["proposals_and_rgb_ms"] = round((time.perf_counter() - t_seg0) * 1000.0, 3)
    else:
        w_depth_m = None
        if policy == "whole_partition_mv" and primary.role == "wrist" and any(True for _ in components):
            w_depth_m = _depth_to_float32_roi(primary.depth_raw, primary.cam, deproj["bbox_2d"] if use_precrop else None)

        unresolved: list[tuple[dict, Any]] = []
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
                sv = cut_verifier
                comp_cen_xy = (float(pts_k[:, 0].mean()), float(pts_k[:, 1].mean()))
                near_square = min(dims_mm[0], dims_mm[1]) >= 0.78 * block.major_m * 1000.0

                child_fits: list[BlockFit] = []
                child_srcs: list[str] = []
                amb: AmbiguousCluster | None = None
                slog: dict[str, Any] = {}
                chosen_hyp: PartitionHypothesis | None = None
                used_warm_start = False

                if (
                    temporal_cache is not None
                    and config.enable_partition_warm_start
                    and not external_masks
                    and policy != "depth_only"
                    and not (config.require_unique_square_evidence and near_square)
                ):
                    ws_ent = temporal_cache.lookup_warm_start(policy, config.calib_epoch, fallback_mode, comp_cen_xy, dims_mm, third_active=third_for_cuts)
                    if ws_ent is not None:
                        ws_hyp = evaluate_warm_start_hypothesis(
                            pts_k, primary_ctx, block, th,
                            direction=ws_ent.direction,
                            perp_dir=ws_ent.perp_dir,
                            axis_deg=ws_ent.axis_deg,
                            k_blocks=ws_ent.k_blocks,
                            refined_cuts_m=ws_ent.refined_cuts_m,
                            side_verifier=sv,
                        )
                        if ws_hyp is not None:
                            ws_cf, ws_cs, ws_amb, ws_slog, ws_best = _resolve_oversized_region(
                                policy, pts_k, [ws_hyp], n_est, config, primary_ctx, w_depth_m, side_data, row_yaw_rad=row_yaw_rad, third_data=third_data,
                            )
                            ws_err_sum = sum(cf.size_err for cf in ws_cf)
                            if ws_cf and ws_amb is None and len(ws_cf) == ws_ent.k_blocks and ws_err_sum <= (1.25 * ws_ent.size_err + 0.05):
                                child_fits, child_srcs, amb, slog, chosen_hyp = ws_cf, ws_cs, ws_amb, {**ws_slog, "warm_start_hit": True}, ws_best
                                used_warm_start = True

                if not used_warm_start:
                    hyps = enumerate_partition_hypotheses(
                        pts_k,
                        None if policy == "depth_only" else primary_ctx,
                        block,
                        th,
                        row_yaw_rad=row_yaw_rad,
                        side_verifier=sv,
                    )
                    if external_masks and policy != "depth_only":
                        m_hyps = propose_hypotheses_from_2d_masks(
                            pts_k, external_masks, primary_ctx, block, th, row_yaw_rad=row_yaw_rad, side_verifier=sv,
                            grouping=config.mask_grouping, suppress_merged=config.mask_suppress_merged,
                        )
                        if m_hyps:
                            hyps.extend(m_hyps)
                            hyps.sort(key=lambda h: -h.score)
                    child_fits, child_srcs, amb, slog, chosen_hyp = _resolve_oversized_region(
                        policy, pts_k, hyps, n_est, config, primary_ctx, w_depth_m, side_data, row_yaw_rad=row_yaw_rad, third_data=third_data,
                    )
                    if (
                        not child_fits and not hyps and config.enable_occlusion_truncated_partition
                        and policy in ("whole_partition_mv", "merged_corroborated", "merged_wrist_rgb")
                    ):
                        occ = _occlusion_truncated_hypotheses(
                            pts_k, m_comp, hm.h_filled, grid, primary_ctx, block, th, sv,
                        )
                        if occ:
                            o_fits, _, o_amb, o_slog, o_best = _resolve_oversized_region(
                                policy, pts_k, [h for h, _ in occ], n_est, config, primary_ctx, w_depth_m, side_data,
                                row_yaw_rad=row_yaw_rad, third_data=third_data,
                            )
                            slog = {**slog, "occlusion_truncated": {"candidates": len(occ), "accepted": bool(o_fits), **o_slog}}
                            if o_fits and o_best is not None:
                                rem = next(r for h, r in occ if h is o_best)
                                rem_pts = pts_k[rem]
                                child_fits, child_srcs, chosen_hyp = o_fits, ["split:occlusion_truncated"] * len(o_fits), None
                                rem_span = np.ptp(rem_pts[:, :2] @ np.array(o_best.cuts[0].direction, dtype=np.float64))
                                ambiguous.append(AmbiguousCluster(
                                    xy_m=(float(rem_pts[:, 0].mean()), float(rem_pts[:, 1].mean())),
                                    span_mm=(round(1000.0 * float(rem_span), 1), 0.0), top_z_m=fast_median_1d(rem_pts[:, 2]),
                                    fill=0.0, n_est=1, candidate_counts=[1], candidate_axes_deg=[o_best.axis_deg],
                                    reason="occluded_partial_remainder",
                                    evidence={"carved_block_xy_m": [round(v, 5) for v in o_fits[0].xy_m]},
                                ))
                                amb = None

                split_logs.append(slog)
                if child_fits:
                    if chosen_hyp is not None and chosen_hyp.cuts:
                        c0 = chosen_hyp.cuts[0]
                        warm_starts_new.append(WarmStartEntry(
                            centroid_xy=comp_cen_xy,
                            span_mm=dims_mm,
                            axis_deg=chosen_hyp.axis_deg,
                            direction=c0.direction,
                            perp_dir=c0.perp_dir,
                            k_blocks=chosen_hyp.k_blocks,
                            refined_cuts_m=tuple(float(cv.cut_val_m + (cv.best_offset_m if cv.rgb_ok else 0.0)) for cv in chosen_hyp.cuts),
                            size_err=float(sum(cf.size_err for cf in child_fits)),
                        ))
                    for cf, cs in zip(child_fits, child_srcs):
                        confident_with_src.append((cf, cs))
                        if len(cf.points_xyz):
                            ri, rj = grid.world_to_ij(cf.points_xyz[:, :2])
                            ok_rc = (ri >= 0) & (ri < grid.ny) & (rj >= 0) & (rj < grid.nx)
                            label_map[ri[ok_rc], rj[ok_rc]] = len(confident_with_src)
                elif amb is not None:
                    amb.span_mm = dims_mm
                    amb.fill = float(diag.get("fill", 0.0))
                    ambiguous.append(amb)
                    unresolved.append((comp, amb))
                else:
                    unresolved.append((comp, None))
            else:
                unresolved.append((comp, None))
        timings["proposals_and_rgb_ms"] = round((time.perf_counter() - t_seg0) * 1000.0, 3)

    # Stage 4b (V3): explain components the V2 path left unresolved (bounded, deadline-aware)
    if (
        policy not in ("generic_watershed", "depth_only", "astra_permissive")
        and config.enable_v3_regions
        and unresolved
    ):
        t_v3 = time.perf_counter()
        views: list[ViewDepth] = []
        if config.v3_view_carving:
            if primary.T_table_cam is not None or T_tc is not None:
                wd = _depth_to_float32_roi(primary.depth_raw, primary.cam, None)
                views.append(ViewDepth(primary.role, primary.cam.K_f64, np.linalg.inv(T_tc), wd))
            for sd in (side_data, third_data):
                if sd is not None and sd.is_fresh and not sd.is_frozen:
                    views.append(ViewDepth("fixed", sd.cam.K_f64, np.linalg.inv(sd.T_table_cam), sd.depth_m))
        r_unk = (
            robot_unknown_cells(grid, robot_links, tcp_offset_z_m=config.tcp_offset_z_m)
            if (config.v3_robot_unknown and robot_links) else None
        )

        def _cluster(xy, comp, reason):
            pts = comp["pts_tbl"]
            return AmbiguousCluster(
                xy_m=(float(xy[0]), float(xy[1])), span_mm=(0.0, 0.0), top_z_m=fast_median_1d(pts[:, 2]),
                fill=0.0, n_est=max(1, int(round(float(comp["area_cells"]) / max(1.0, a1_cells)))),
                candidate_counts=[], candidate_axes_deg=[], reason=reason,
            )

        def _cue(pts, nrm, cut_val):
            d32 = np.asarray(nrm, dtype=np.float32)
            pd = np.array([-d32[1], d32[0]], dtype=np.float32)
            xy = np.ascontiguousarray(pts[:, :2], dtype=np.float32)
            # geometry predicts the seam to ~1 mm; look for appearance/depth evidence within +-0.8 mm of it
            # (V2's 1.5 mm offset lattice can straddle a narrow dark-gap-dark seam)
            cue = None
            for dc in (0.0, -0.0008, 0.0008):
                cue = evaluate_cut_cues(pts, xy @ d32, xy @ pd, float(cut_val) + dc, d32, pd, primary_ctx, th, 0)
                if cue.depth_step_ok or cue.depth_trough_ok:
                    return True, "depth"
                if cue.rgb_ok and not cue.periodic_texture_alias:
                    return True, "rgb"
            if cut_verifier is not None:
                sv = cut_verifier.verify_cut(cue, float(cut_val), block.height_m)
                if sv.status == "CORROBORATED":
                    return True, "side"
            if external_masks and config.v3_mask_seam_cue:
                # externally proposed instance masks may only confirm a seam between two geometrically proven
                # blocks: each side's projected top points must fall predominantly in a *different* mask
                side_a = (xy @ d32) < cut_val
                uv, _ = cv2.projectPoints(pts.astype(np.float64), primary_ctx.rvec_ct, primary_ctx.tvec_ct,
                                          primary_ctx.K, primary_ctx.D)
                uv = np.round(uv.reshape(-1, 2)).astype(np.int64)
                Hh, Ww = primary_ctx.gray_img.shape[:2]
                inb = (uv[:, 0] >= 0) & (uv[:, 0] < Ww) & (uv[:, 1] >= 0) & (uv[:, 1] < Hh)
                if np.count_nonzero(inb) >= 40:
                    lab = np.full(len(pts), -1)
                    for mi, mk in enumerate(external_masks):
                        if mk.shape[:2] != (Hh, Ww):
                            continue
                        hit = inb.copy()
                        hit[inb] = mk[uv[inb, 1], uv[inb, 0]] > 0
                        lab[hit & (lab == -1)] = mi
                    def major(sel):
                        v = lab[sel & inb]
                        v = v[v >= 0]
                        if len(v) < 20:
                            return None, 0.0
                        cnt = np.bincount(v)
                        return int(cnt.argmax()), float(cnt.max()) / max(1, np.count_nonzero(sel & inb))
                    ma, fa = major(side_a)
                    mb, fb = major(~side_a)
                    if ma is not None and mb is not None and ma != mb and fa >= 0.8 and fb >= 0.8:
                        return True, "mask"
            return False, "none"

        v3_stats = run_v3_regions(
            unresolved=unresolved, hm=hm, grid=grid, block=block, config=config, views=views,
            ambiguous=ambiguous, confident_with_src=confident_with_src, label_map=label_map,
            robot_unknown=r_unk, make_cluster=_cluster, t_frame0=t_total_0, cue_fn=_cue,
            region_cache=_v3_region_cache(temporal_cache, config),
        )
        timings["v3_regions_ms"] = round((time.perf_counter() - t_v3) * 1000.0, 3)
        split_logs.append({"v3": {k: v for k, v in v3_stats.items() if k != "ms"}})

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

    pose_side_vis = [
        compute_target_side_visibility(f, side_data, block, th)
        or (third_for_cuts and compute_target_side_visibility(f, third_data, block, th))
        for f, _ in confident_with_src
    ]
    timings["verifier_ms"] = round((time.perf_counter() - t_ver0) * 1000.0, 3)
    timings["total_ms"] = round((time.perf_counter() - t_total_0) * 1000.0, 3)

    poses = [f for f, _ in confident_with_src]
    pose_sources = [s for _, s in confident_with_src]
    ev_log = {
        "crop_frac": round(deproj["crop_frac"], 3),
        "bbox_2d": deproj["bbox_2d"],
        "pts_in_roi": int(len(pts_tbl)),
        "removed_robot_pts": removed_robot,
        "plane_dist_mm": plane_dist_mm,
        "plane_sup_pct": plane_sup_pct,
        "split_logs": split_logs,
        "verifier": verifier_stats,
        **({"third_cut_verdicts": two_side_verifier.third_verdicts} if two_side_verifier is not None else {}),
        **view_diag,
    }
    ep_res = EpochResult(
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
        evidence_log=ev_log,
        pose_side_visible=pose_side_vis,
        T_table_primary=T_tc,
        primary_img_ctx=primary_ctx,
        side_img_ctx=side_data.img_ctx if side_data is not None else None,
        third_img_ctx=third_data.img_ctx if third_data is not None else None,
        hm_grid=grid,
        label_map=label_map,
    )
    if temporal_cache is not None:
        if verifier_stats.get("vetoed", 0) > 0:
            temporal_cache.invalidate_warm_starts()
        else:
            temporal_cache.record_warm_starts(policy, config.calib_epoch, fallback_mode, warm_starts_new, third_active=third_for_cuts)
        if config.enable_temporal_skip:
            temporal_cache.store_epoch(
                result=ep_res,
                primary=primary,
                secondary=secondary,
                policy=policy,
                calib_epoch=config.calib_epoch,
                p_bbox=tuple(deproj["bbox_2d"]),
                s_bbox=side_bbox,
                min_d_p=min_d_p,
                max_d_p=max_d_p,
                min_d_s=min_d_s,
                max_d_s=max_d_s,
                third=third if third_for_cuts else None,
                t_bbox=third_bbox,
            )
    return ep_res

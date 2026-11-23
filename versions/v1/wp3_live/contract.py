"""Atomic epoch-stamped v2 contract builder with 6-DOF validity bitmask & snapshot-commit fields."""
from __future__ import annotations

from typing import Any
import numpy as np

from .config import RuntimeConfig, config_hash
from .fusion_tracker import TrackedBlock
from .policies import EpochResult


def build_atomic_scene_payload(
    epoch: EpochResult,
    tracks: list[TrackedBlock],
    cfg: RuntimeConfig,
    *,
    epoch_id: int,
    publish_stamp_ns: int,
    code_hash: str,
    camera_health: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble one atomic scene snapshot combining Gemini's contract fields & Astra's 6-DOF bitmask."""
    yaw_sym = cfg.block.yaw_symmetry_deg
    z_ok_dwell = epoch.dwell is None or epoch.dwell.stationary

    tracks_out = []
    for tr in tracks:
        f = tr.fit
        z_valid = bool(tr.state in ("MEASURED", "DISPLACED") and np.isfinite(f.z_center_m) and z_ok_dwell)
        xy_valid = bool(tr.state in ("MEASURED", "DISPLACED") and np.isfinite(f.xy_m[0]) and np.isfinite(f.xy_m[1]))
        yaw_valid = bool(xy_valid and not cfg.block.is_square and np.isfinite(f.yaw_rad))
        tracks_out.append({
            "block_id": tr.block_name,
            "track_id": tr.track_id,
            "state": tr.state,
            "confirmed": tr.confirmed,
            "mds_score": tr.mds_score,
            "pose_table_m": [round(f.xy_m[0], 5), round(f.xy_m[1], 5), round(f.z_center_m, 5)],
            "z_top_table_rel_m": round(f.z_top_m, 5),
            "pose_base_m": None,  # Explicitly None unless base transform is separately qualified
            "yaw_rad": round(float(f.yaw_rad), 5),
            "yaw_deg_mod": round(float(f.yaw_deg_mod), 2),
            "yaw_symmetry_deg": yaw_sym,
            "dims_mm": list(f.dims_mm),
            "validity_mask": {
                "x": xy_valid,
                "y": xy_valid,
                "z": z_valid,
                "roll": False,
                "pitch": False,
                "yaw": yaw_valid,
            },
            "covariance_available": f.covariance_xy_m2 is not None,
            "observation_stamp_ns": int(tr.observation_stamp_ns),
            "source_role": epoch.primary_role,
            "evidence_source": tr.source,
            "target_side_fresh": tr.target_side_fresh,
            "snapshot_commit_eligible": tr.snapshot_commit_eligible,
        })

    amb_out = [
        {
            "cluster_id": f"AMB_{idx:02d}",
            "envelope_xy_m": [round(c.xy_m[0], 5), round(c.xy_m[1], 5)],
            "span_mm": list(c.span_mm),
            "top_z_m": round(c.top_z_m, 5),
            "fill": round(c.fill, 3),
            "n_est": int(c.n_est),
            "candidate_counts": list(c.candidate_counts),
            "candidate_axes_deg": list(c.candidate_axes_deg),
            "reason": c.reason,
            "evidence": c.evidence,
        }
        for idx, c in enumerate(epoch.ambiguous_clusters, start=1)
    ]

    slots_out = [
        {
            "layer": s.layer,
            "slot": s.slot,
            "state": s.state,
            "visibility_ratio_r": s.visibility_ratio_r,
            "median_depth_delta_mm": s.median_depth_delta_mm,
            "expected_range_m": s.expected_range_m,
        }
        for s in epoch.slot_occupancies
    ]

    return {
        "schema_version": "wp3_scene_observation_v2.0",
        "epoch_id": int(epoch_id),
        "frame_id": cfg.frame_id,
        "publish_stamp_ns": int(publish_stamp_ns),
        "observation_stamp_ns": int(epoch.stamp_ns),
        "policy": epoch.policy,
        "status": epoch.status,
        "primary_role": epoch.primary_role,
        "fallback_mode": epoch.fallback_mode,
        "calibration_epoch": cfg.calib_epoch,
        "config_hash": config_hash(cfg),
        "code_hash": code_hash,
        "camera_health": camera_health or {},
        "dwell": {
            "stationary": epoch.dwell.stationary,
            "fk_qualified": epoch.dwell.fk_qualified,
            "imu_corroborated": epoch.dwell.imu_corroborated,
            "reason": epoch.dwell.reason,
        } if epoch.dwell else None,
        "tracks": tracks_out,
        "ambiguous_clusters": amb_out,
        "slot_occupancies": slots_out,
        "stage_timings_ms": epoch.stage_timings_ms,
        "evidence": epoch.evidence_log,
    }

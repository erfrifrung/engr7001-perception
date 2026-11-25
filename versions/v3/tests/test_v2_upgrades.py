"""Regression tests for v2 upgrades: valid depth interp, invisible side seams, cross-view skew, ray AABB slots, whole-partition MV, target visibility & async mask worker."""
from __future__ import annotations

from pathlib import Path
import sys
import time
import numpy as np

IMPL_ROOT = Path(__file__).resolve().parents[1]
if str(IMPL_ROOT) not in sys.path:
    sys.path.insert(0, str(IMPL_ROOT))
sys.path.insert(0, str(IMPL_ROOT / "scripts"))

from wp3_live import AsyncMaskWorker, SAM2MaskChallenger, ViewInput, preset_config, run_perception_epoch  # noqa: E402
from wp3_live.dwell_gate import DwellStatus  # noqa: E402
from wp3_live.fusion_tracker import WP3TemporalTracker  # noqa: E402
from wp3_live.geometry_cache import CameraModelCache  # noqa: E402
from wp3_live.proposals import BlockFit, ImageProjectionContext, _bilinear_sample_valid_depth  # noqa: E402
from wp3_live.side_verifier import SideViewData, detect_side_facade_slots  # noqa: E402
from run_printed_jenga_suite import make_synthetic_cameras, render_synthetic_scene  # noqa: E402


def preset_config(name, _orig=preset_config):  # noqa: F811
    """V2-parity tests exercise V2 mechanisms: the V3 region stage is disabled here (see tests/test_v3_*.py)."""
    cfg = _orig(name)
    cfg.enable_v3_regions = False
    return cfg


def test_bilinear_depth_excludes_zero_and_jumps():
    d = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    uv = np.array([[0.5, 0.5]], dtype=np.float32)
    # Must NEVER average 0.0 and 1.0 into 0.5! Sum of valid weights = 0.5 -> valid corners are both 1.0 -> returns 1.0
    val, ok = _bilinear_sample_valid_depth(d, uv, require_all_4=False)
    assert ok[0] and abs(float(val[0]) - 1.0) < 1e-4

    # If valid weight is < 0.5 (mostly invalid 0.0), or jump exceeds 40 mm, MUST return False
    uv_bad = np.array([[0.2, 0.5]], dtype=np.float32)
    val_bad, ok_bad = _bilinear_sample_valid_depth(d, uv_bad, require_all_4=False)
    assert not ok_bad[0] and np.isnan(val_bad[0])

    d_jump = np.array([[0.40, 1.00], [0.40, 1.00]], dtype=np.float32)
    _, ok_jump = _bilinear_sample_valid_depth(d_jump, uv, max_jump_m=0.04)
    assert not ok_jump[0]


def test_invisible_side_seam_inconclusive_and_cross_view_skew():
    cfg = preset_config("printed_jenga")
    cfg.calib_epoch = "v2_test"
    cache = CameraModelCache()
    cam_w, Tw, cam_s, Ts = make_synthetic_cameras(cache)
    dwell = DwellStatus(True, True, True, True, 0.0, 0.1, 0.2, 90.0, "ok")

    # True flush pair WITHOUT visible dark line on side view (0 contrast on side face!)
    rw, dw, rs, ds = render_synthetic_scene(
        cam_w, Tw, cam_s, Ts,
        [
            {"cx": 0.0, "cy": -0.0125, "z0": 0.0, "z1": 0.015, "L": 0.075, "W": 0.025, "yaw_rad": 0.0},
            {"cx": 0.0, "cy": 0.0125, "z0": 0.0, "z1": 0.015, "L": 0.075, "W": 0.025, "yaw_rad": 0.0},
        ],
        seams_top=[{"p0": (-0.037, 0.0), "p1": (0.037, 0.0), "z": 0.015, "half_w_m": 0.0009}],
        end_breaks=[],
    )
    res_wrgb = run_perception_epoch(
        wrist=ViewInput("wrist", cam_w, rw, dw, Tw, 1_000_000_000, 1_000_000_000),
        side=ViewInput("tripod", cam_s, rs, ds, Ts, 1_000_000_000, 1_000_000_000),
        config=cfg, policy="merged_wrist_rgb", dwell=dwell,
    )
    # Invisible side seam MUST be INCONCLUSIVE (NOT CONTRADICTED), so merged_wrist_rgb resolves 2 blocks!
    assert len(res_wrgb.poses) == 2

    res_corr = run_perception_epoch(
        wrist=ViewInput("wrist", cam_w, rw, dw, Tw, 1_000_000_000, 1_000_000_000),
        side=ViewInput("tripod", cam_s, rs, ds, Ts, 1_000_000_000, 1_000_000_000),
        config=cfg, policy="merged_corroborated", dwell=dwell,
    )
    assert len(res_corr.ambiguous_clusters) == 1
    assert res_corr.ambiguous_clusters[0].reason != "side_contradicted_rgb_cut"

    # Cross-view skew gate inside shared core: 250 ms skew MUST reject side & degrade to wrist_only!
    res_skew = run_perception_epoch(
        wrist=ViewInput("wrist", cam_w, rw, dw, Tw, 1_000_000_000, 1_000_000_000),
        side=ViewInput("tripod", cam_s, rs, ds, Ts, 1_250_000_000, 1_250_000_000),
        config=cfg, policy="merged_corroborated", dwell=dwell,
    )
    assert res_skew.fallback_mode == "wrist_only"
    assert res_skew.evidence_log.get("side_rejected_reason") == "cross_view_skew"


def test_ray_aabb_recessed_vs_empty_slot_and_whole_partition_mv():
    cfg = preset_config("printed_jenga")
    cache = CameraModelCache()
    cam_w, Tw, cam_s, Ts = make_synthetic_cameras(cache)
    dwell = DwellStatus(True, True, True, True, 0.0, 0.1, 0.2, 90.0, "ok")

    # Build Layer 1 viewed into the exposed -X end facade (~20 deg elevation, slight azimuth offset):
    # slot -1 OCCUPIED (cx=0), slot 0 RECESSED by +15 mm inward (cx=+0.015), slot +1 OBSERVED_EMPTY!
    pos_facade = np.array([-0.55, -0.10, 0.20], dtype=np.float64)
    fwd = -pos_facade / np.linalg.norm(pos_facade)
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0])); right /= np.linalg.norm(right)
    down = np.cross(fwd, right); down /= np.linalg.norm(down)
    Ts_facade = np.eye(4, dtype=np.float64)
    Ts_facade[:3, 0] = right; Ts_facade[:3, 1] = down; Ts_facade[:3, 2] = fwd; Ts_facade[:3, 3] = pos_facade

    blks = [
        {"cx": 0.0, "cy": -0.025, "z0": 0.0, "z1": 0.015, "L": 0.075, "W": 0.025, "yaw_rad": 0.0},
        {"cx": 0.015, "cy": 0.0, "z0": 0.0, "z1": 0.015, "L": 0.075, "W": 0.025, "yaw_rad": 0.0},
        {"cx": -0.025, "cy": 0.0, "z0": 0.015, "z1": 0.030, "L": 0.075, "W": 0.025, "yaw_rad": np.pi / 2},
        {"cx": 0.0, "cy": 0.0, "z0": 0.015, "z1": 0.030, "L": 0.075, "W": 0.025, "yaw_rad": np.pi / 2},
        {"cx": 0.025, "cy": 0.0, "z0": 0.015, "z1": 0.030, "L": 0.075, "W": 0.025, "yaw_rad": np.pi / 2},
    ]
    _, _, rs, ds = render_synthetic_scene(cam_w, Tw, cam_s, Ts_facade, blks)
    s_data = SideViewData(cam_s, Ts_facade, rs, ds.astype(np.float32) * 0.001, 100, ImageProjectionContext.from_rgb(rs, cam_s.K_f64, cam_s.D_f64, Ts_facade))
    slots = {(s.layer, s.slot): s for s in detect_side_facade_slots(s_data, cfg.block, cfg.thresholds, layers=2)}
    assert slots[(1, -1)].state == "OCCUPIED"
    assert slots[(1, 0)].state == "RECESSED"
    # Verify empty center tunnel (cx=0 removed) reports OBSERVED_EMPTY
    blks_empty = [blks[0], blks[2], blks[3], blks[4]]
    _, _, rse, dse = render_synthetic_scene(cam_w, Tw, cam_s, Ts_facade, blks_empty)
    se_data = SideViewData(cam_s, Ts_facade, rse, dse.astype(np.float32) * 0.001, 101, ImageProjectionContext.from_rgb(rse, cam_s.K_f64, cam_s.D_f64, Ts_facade))
    slots_e = {(s.layer, s.slot): s for s in detect_side_facade_slots(se_data, cfg.block, cfg.thresholds, layers=2)}
    assert slots_e[(1, 0)].state == "OBSERVED_EMPTY"

    # whole_partition_mv resolves rotated 90-deg 75x75 square when ends visible & rejects 2-false-stripe solid!
    rw90, dw90, rs90, ds90 = render_synthetic_scene(
        cam_w, Tw, cam_s, Ts,
        [
            {"cx": -0.025, "cy": 0.0, "z0": 0.0, "z1": 0.015, "L": 0.075, "W": 0.025, "yaw_rad": np.pi / 2},
            {"cx": 0.0, "cy": 0.0, "z0": 0.0, "z1": 0.015, "L": 0.075, "W": 0.025, "yaw_rad": np.pi / 2},
            {"cx": 0.025, "cy": 0.0, "z0": 0.0, "z1": 0.015, "L": 0.075, "W": 0.025, "yaw_rad": np.pi / 2},
        ],
        seams_top=[{"p0": (-0.0125, -0.037), "p1": (-0.0125, 0.037), "z": 0.015}, {"p0": (0.0125, -0.037), "p1": (0.0125, 0.037), "z": 0.015}],
        end_breaks=[{"x": -0.0125, "y": -0.0375, "z0": 0.0, "z1": 0.015}, {"x": 0.0125, "y": -0.0375, "z0": 0.0, "z1": 0.015}],
    )
    res_mv = run_perception_epoch(
        wrist=ViewInput("wrist", cam_w, rw90, dw90, Tw, 200, 200),
        side=ViewInput("tripod", cam_s, rs90, ds90, Ts, 200, 200),
        config=cfg, policy="whole_partition_mv", dwell=dwell,
    )
    assert len(res_mv.poses) == 3

    rw_f, dw_f, rs_f, ds_f = render_synthetic_scene(
        cam_w, Tw, cam_s, Ts,
        [{"cx": 0.0, "cy": 0.0, "z0": 0.0, "z1": 0.015, "L": 0.075, "W": 0.075, "yaw_rad": 0.0}],
        seams_top=[{"p0": (-0.037, -0.0125), "p1": (0.037, -0.0125), "z": 0.015}, {"p0": (-0.037, 0.0125), "p1": (0.037, 0.0125), "z": 0.015}],
    )
    res_mv_false = run_perception_epoch(
        wrist=ViewInput("wrist", cam_w, rw_f, dw_f, Tw, 300, 300),
        side=ViewInput("tripod", cam_s, rs_f, ds_f, Ts, 300, 300),
        config=cfg, policy="whole_partition_mv", dwell=dwell,
    )
    assert len(res_mv_false.poses) == 0 and len(res_mv_false.ambiguous_clusters) == 1


def test_per_target_side_visibility_identity_ambiguity_and_async_mask_worker():
    cfg = preset_config("printed_jenga")
    cache = CameraModelCache()
    cam_w, Tw, cam_s, Ts = make_synthetic_cameras(cache)
    dwell = DwellStatus(True, True, True, True, 0.0, 0.1, 0.2, 90.0, "ok")

    # 1. Per-target side visibility: occluded side target gets pose_side_visible=False even when side camera is fresh
    rw, dw, rs, ds = render_synthetic_scene(
        cam_w, Tw, cam_s, Ts,
        [{"cx": 0.0, "cy": 0.0, "z0": 0.0, "z1": 0.015, "L": 0.075, "W": 0.025, "yaw_rad": 0.0}],
        occlude_side=True,
    )
    res_occ = run_perception_epoch(
        wrist=ViewInput("wrist", cam_w, rw, dw, Tw, 100, 100),
        side=ViewInput("tripod", cam_s, rs, ds, Ts, 100, 100),
        config=cfg, policy="merged_corroborated", dwell=dwell,
    )
    assert len(res_occ.poses) == 1 and res_occ.pose_side_visible == [False]
    tr = WP3TemporalTracker(cfg.block, cfg.thresholds)
    tracks = tr.update(res_occ.poses, res_occ.pose_sources, 100, 1.0, side_fresh_global=True, pose_side_visible=res_occ.pose_side_visible)
    assert not tracks[0].target_side_fresh and not tracks[0].snapshot_commit_eligible

    # 2. Hidden-swap / nearby identity ambiguity after occlusion
    tr2 = WP3TemporalTracker(cfg.block, cfg.thresholds)
    fA = BlockFit((0.0, 0.0), 0.0075, 0.015, 0.0, 0.0, (75.0, 25.0), 1.0, 0.0, 40, np.zeros((40, 3)))
    fB = BlockFit((0.0, 0.026), 0.0075, 0.015, 0.0, 0.0, (75.0, 25.0), 1.0, 0.0, 40, np.zeros((40, 3)))
    tr2.update([fA, fB], ["single", "single"], 1_000_000_000, 1.0, pose_side_visible=[True, True])
    tr2.update([], [], 1_200_000_000, 1.2, pose_side_visible=[])  # both OCCLUDED
    fA_back = BlockFit((0.0, 0.002), 0.0075, 0.015, 0.0, 0.0, (75.0, 25.0), 1.0, 0.0, 40, np.zeros((40, 3)))
    out_back = tr2.update([fA_back, fB], ["single", "single"], 1_400_000_000, 1.4, pose_side_visible=[True, True])
    assert any(t.identity_ambiguous and not t.snapshot_commit_eligible for t in out_back)

    # 3. AsyncMaskWorker with callable mock & metric cuboid gating via external_masks
    rw_p, dw_p, rs_p, ds_p = render_synthetic_scene(
        cam_w, Tw, cam_s, Ts,
        [
            {"cx": 0.0, "cy": -0.0125, "z0": 0.0, "z1": 0.015, "L": 0.075, "W": 0.025, "yaw_rad": 0.0},
            {"cx": 0.0, "cy": 0.0125, "z0": 0.0, "z1": 0.015, "L": 0.075, "W": 0.025, "yaw_rad": 0.0},
        ],
        seams_top=[{"p0": (-0.037, 0.0), "p1": (0.037, 0.0), "z": 0.015, "half_w_m": 0.0009}],
        end_breaks=[{"x": -0.0375, "y": 0.0, "z0": 0.0, "z1": 0.015}, {"x": 0.0375, "y": 0.0, "z0": 0.0, "z1": 0.015}],
    )
    H, W = rw_p.shape[:2]
    def mock_seg(rgb, box):
        m1 = np.zeros((H, W), dtype=np.uint8); m1[: H // 2, :] = 1
        m2 = np.zeros((H, W), dtype=np.uint8); m2[H // 2 :, :] = 1
        return [m1, m2]

    ch = SAM2MaskChallenger(backend="callable", callable_fn=mock_seg)
    worker = AsyncMaskWorker(ch)
    try:
        assert worker.submit(epoch_id=10, observation_stamp_ns=500, calib_epoch="c1", region_key="global", rgb_img=rw_p, box_xyxy=(0, 0, W, H))
        res_m = None
        for _ in range(30):
            res_m = worker.poll_matching(epoch_id=10, calib_epoch="c1", region_key="global")
            if res_m is not None:
                break
            time.sleep(0.01)
        assert res_m is not None and len(res_m.masks_u8) == 2
        assert worker.poll_matching(epoch_id=20, calib_epoch="c1", region_key="global", max_epoch_lag=2) is None
        assert worker.poll_matching(epoch_id=10, calib_epoch="c2", region_key="global") is None
        ep_ext = run_perception_epoch(
            wrist=ViewInput("wrist", cam_w, rw_p, dw_p, Tw, 500, 500),
            side=ViewInput("tripod", cam_s, rs_p, ds_p, Ts, 500, 500),
            config=cfg, policy="merged_corroborated", dwell=dwell, external_masks=res_m.masks_u8,
        )
        assert len(ep_ext.poses) == 2
    finally:
        worker.stop()

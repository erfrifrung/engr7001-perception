"""Regression tests for printed Jenga shapes, stripes, 75x75 square layouts, watershed, ROI efficiency, tracking & dropouts."""
from __future__ import annotations

from pathlib import Path
import sys
import time
import numpy as np

IMPL_ROOT = Path(__file__).resolve().parents[1]
if str(IMPL_ROOT) not in sys.path:
    sys.path.insert(0, str(IMPL_ROOT))
sys.path.insert(0, str(IMPL_ROOT / "scripts"))

from wp3_live import ViewInput, preset_config, run_perception_epoch  # noqa: E402
from wp3_live.dwell_gate import DwellStatus, WristDwellGate, estimate_so3_clock_offset  # noqa: E402
from wp3_live.fusion_tracker import StreamWatchdog, WP3TemporalTracker, associate_late_hypotheses  # noqa: E402
from wp3_live.geometry_cache import CameraModelCache, deproject_roi_points, filter_robot_links  # noqa: E402
from wp3_live.proposals import segment_scalar_watershed  # noqa: E402
from wp3_live.side_verifier import CroppedSilhouetteVerifier, SideViewData, detect_side_facade_slots  # noqa: E402
from wp3_live.proposals import ImageProjectionContext  # noqa: E402
from run_printed_jenga_suite import make_synthetic_cameras, render_synthetic_scene  # noqa: E402


H_BLOCK = 0.015


def test_printed_rotated_square_layouts_and_false_stripes():
    cfg = preset_config("printed_jenga")
    cfg.calib_epoch = "test_calib"
    cfg.require_fk_for_moving_wrist = False
    cache = CameraModelCache()
    cam_w, Tw, cam_s, Ts = make_synthetic_cameras(cache)
    dwell = DwellStatus(True, True, True, True, 0.0, 0.1, 0.2, 80.0, "ok")

    # 1. Rotated 30 deg single 75x25x15 mm printed block
    rw, dw, rs, ds = render_synthetic_scene(
        cam_w, Tw, cam_s, Ts,
        [{"cx": 0.0, "cy": 0.0, "z0": 0.0, "z1": H_BLOCK, "L": 0.075, "W": 0.025, "yaw_rad": np.radians(30.0)}],
    )
    res_single = run_perception_epoch(
        wrist=ViewInput("wrist", cam_w, rw, dw, Tw, 100, 100),
        side=ViewInput("tripod", cam_s, rs, ds, Ts, 100, 100),
        config=cfg, policy="merged_corroborated", dwell=dwell,
    )
    assert len(res_single.poses) == 1
    assert abs((res_single.poses[0].yaw_deg_mod - 30.0 + 90.0) % 180.0 - 90.0) < 4.0

    # 2. Two false stripes over continuous 75x75 material: corroborated MUST NOT false-split!
    rw, dw, rs, ds = render_synthetic_scene(
        cam_w, Tw, cam_s, Ts,
        [{"cx": 0.0, "cy": 0.0, "z0": 0.0, "z1": H_BLOCK, "L": 0.075, "W": 0.075, "yaw_rad": 0.0}],
        seams_top=[
            {"p0": (-0.037, -0.0125), "p1": (0.037, -0.0125), "z": H_BLOCK, "half_w_m": 0.0009},
            {"p0": (-0.037, 0.0125), "p1": (0.037, 0.0125), "z": H_BLOCK, "half_w_m": 0.0009},
        ],
    )
    res_2stripe = run_perception_epoch(
        wrist=ViewInput("wrist", cam_w, rw, dw, Tw, 200, 200),
        side=ViewInput("tripod", cam_s, rs, ds, Ts, 200, 200),
        config=cfg, policy="merged_corroborated", dwell=dwell,
    )
    assert len(res_2stripe.poses) == 0
    assert len(res_2stripe.ambiguous_clusters) == 1

    # 3. 75x75 3-block Layout A (0 deg) vs Layout B (90 deg) with ends visible -> BOTH resolve cleanly
    for yaw_deg in (0.0, 90.0):
        rad = np.radians(yaw_deg)
        c, s = np.cos(rad), np.sin(rad)
        blks = [
            {"cx": -s * off, "cy": c * off, "z0": 0.0, "z1": H_BLOCK, "L": 0.075, "W": 0.025, "yaw_rad": rad}
            for off in (-0.025, 0.0, 0.025)
        ]
        # Seam segments along each joint and end breaks on BOTH ends
        u = np.array([c, s]); p = np.array([-s, c])
        seams = [
            {"p0": tuple((-0.037 * u + off * p).tolist()), "p1": tuple((0.037 * u + off * p).tolist()), "z": H_BLOCK, "half_w_m": 0.0009}
            for off in (-0.0125, 0.0125)
        ]
        breaks = [
            {"x": float(-0.0375 * u[0] + off * p[0]), "y": float(-0.0375 * u[1] + off * p[1]), "z0": 0.0, "z1": H_BLOCK}
            for off in (-0.0125, 0.0125)
        ] + [
            {"x": float(0.0375 * u[0] + off * p[0]), "y": float(0.0375 * u[1] + off * p[1]), "z0": 0.0, "z1": H_BLOCK}
            for off in (-0.0125, 0.0125)
        ]
        rw, dw, rs, ds = render_synthetic_scene(cam_w, Tw, cam_s, Ts, blks, seams, breaks)
        res_sq = run_perception_epoch(
            wrist=ViewInput("wrist", cam_w, rw, dw, Tw, 300, 300),
            side=ViewInput("tripod", cam_s, rs, ds, Ts, 300, 300),
            config=cfg, policy="merged_corroborated", dwell=dwell,
        )
        assert len(res_sq.poses) == 3, f"Expected 3 blocks for 75x75 yaw={yaw_deg}, got {len(res_sq.poses)} amb={res_sq.ambiguous_clusters}"
        for po in res_sq.poses:
            err = min((po.yaw_deg_mod - yaw_deg) % 180.0, (yaw_deg - po.yaw_deg_mod) % 180.0)
            assert err < 5.0

    # 4. 75x75 flush square with ends obscured under merged_corroborated MUST abstain!
    rw, dw, rs, ds = render_synthetic_scene(
        cam_w, Tw, cam_s, Ts,
        [{"cx": 0.0, "cy": off, "z0": 0.0, "z1": H_BLOCK, "L": 0.075, "W": 0.025, "yaw_rad": 0.0} for off in (-0.025, 0.0, 0.025)],
        [{"p0": (-0.037, -0.0125), "p1": (0.037, -0.0125), "z": H_BLOCK}, {"p0": (-0.037, 0.0125), "p1": (0.037, 0.0125), "z": H_BLOCK}],
        occlude_side=True,
    )
    res_obs = run_perception_epoch(
        wrist=ViewInput("wrist", cam_w, rw, dw, Tw, 400, 400),
        side=ViewInput("tripod", cam_s, rs, ds, Ts, 400, 400),
        config=cfg, policy="merged_corroborated", dwell=dwell,
    )
    assert len(res_obs.poses) == 0 and len(res_obs.ambiguous_clusters) == 1

    # 5. Real 3.5 mm physical step MUST resolve even without side camera
    rw, dw, rs, ds = render_synthetic_scene(
        cam_w, Tw, cam_s, Ts,
        [
            {"cx": 0.0, "cy": -0.0125, "z0": 0.0, "z1": H_BLOCK, "L": 0.075, "W": 0.025, "yaw_rad": 0.0},
            {"cx": 0.0, "cy": 0.0125, "z0": 0.0, "z1": H_BLOCK + 0.0035, "L": 0.075, "W": 0.025, "yaw_rad": 0.0},
        ],
        occlude_side=True,
    )
    res_step = run_perception_epoch(
        wrist=ViewInput("wrist", cam_w, rw, dw, Tw, 500, 500),
        side=None,
        config=cfg, policy="merged_corroborated", dwell=dwell,
    )
    assert len(res_step.poses) == 2 and all(s == "split:depth" for s in res_step.pose_sources)


def test_roi_precrop_parity_and_speedup():
    cfg = preset_config("printed_jenga")
    cfg.require_fk_for_moving_wrist = False
    cache = CameraModelCache()
    cam_w, Tw, cam_s, Ts = make_synthetic_cameras(cache)
    rw, dw, _, _ = render_synthetic_scene(
        cam_w, Tw, cam_s, Ts,
        [{"cx": 0.02, "cy": -0.01, "z0": 0.0, "z1": H_BLOCK, "L": 0.075, "W": 0.025, "yaw_rad": 0.2}],
    )
    vi_full = ViewInput("wrist", cam_w, rw, dw, Tw, 100, 100)
    # Wait: check crop_frac with 0.30 m ROI (which fills ~75% at 0.45 m wrist) OR narrower ROI (which crops tightly)
    cfg_tight = preset_config("printed_jenga")
    cfg_tight.require_fk_for_moving_wrist = False
    cfg_tight.roi_xyz_min = (-0.10, -0.10, -0.02)
    cfg_tight.roi_xyz_max = (0.10, 0.10, 0.08)
    t0 = time.perf_counter()
    for _ in range(10):
        res_roi = run_perception_epoch(wrist=vi_full, side=None, config=cfg_tight, policy="depth_only")
    t_roi = time.perf_counter() - t0

    cfg_full = preset_config("printed_jenga")
    cfg_full.require_fk_for_moving_wrist = False
    cfg_full.roi_xyz_min = (-0.10, -0.10, -0.02)
    cfg_full.roi_xyz_max = (0.10, 0.10, 0.08)
    cfg_full.use_roi_precrop = False
    t1 = time.perf_counter()
    for _ in range(10):
        res_all = run_perception_epoch(wrist=vi_full, side=None, config=cfg_full, policy="depth_only")
    t_all = time.perf_counter() - t1

    assert len(res_roi.poses) == len(res_all.poses) == 1
    assert np.hypot(res_roi.poses[0].xy_m[0] - res_all.poses[0].xy_m[0], res_roi.poses[0].xy_m[1] - res_all.poses[0].xy_m[1]) < 1e-4
    assert res_roi.evidence_log["crop_frac"] < 0.45
    assert t_roi <= t_all


def test_scalar_watershed_flush_stays_single_step_splits():
    H = np.zeros((40, 40), dtype=np.float32)
    rgb = np.full((40, 40, 3), 180, dtype=np.uint8)
    valid = np.ones((40, 40), dtype=bool)
    fg = np.zeros((40, 40), dtype=bool)
    fg[5:35, 8:32] = True
    # Case 1: Flat coplanar flush rectangle with strong RGB seam line -> MUST stay 1 seed!
    H[fg] = 0.015
    rgb[5:35, 19:21] = 30
    lab_flush, n_seeds = segment_scalar_watershed(H, rgb, valid, fg)
    assert n_seeds == 1 and len(set(lab_flush[fg].tolist()) - {0}) == 1

    # Case 2: Clear 4 mm height step across x=20 -> MUST split into 2 seeds/regions!
    H[5:35, 20:32] = 0.019
    lab_step, n_seeds_step = segment_scalar_watershed(H, rgb, valid, fg, step_barrier_m=0.002)
    assert n_seeds_step == 2 and len(set(lab_step[fg].tolist()) - {0}) == 2


def test_side_slots_and_cropped_verifier_veto():
    cfg = preset_config("printed_jenga")
    cache = CameraModelCache()
    cam_w, Tw, cam_s, Ts = make_synthetic_cameras(cache)
    # Build 2-layer tower missing Layer 1 slot 0 (center observed empty from side, slot +1 & -1 occupied)
    blks = [
        {"cx": 0.0, "cy": -0.025, "z0": 0.0, "z1": 0.015, "L": 0.075, "W": 0.025, "yaw_rad": 0.0},
        {"cx": 0.0, "cy": 0.025, "z0": 0.0, "z1": 0.015, "L": 0.075, "W": 0.025, "yaw_rad": 0.0},
        {"cx": -0.025, "cy": 0.0, "z0": 0.015, "z1": 0.030, "L": 0.075, "W": 0.025, "yaw_rad": np.pi / 2},
        {"cx": 0.0, "cy": 0.0, "z0": 0.015, "z1": 0.030, "L": 0.075, "W": 0.025, "yaw_rad": np.pi / 2},
        {"cx": 0.025, "cy": 0.0, "z0": 0.015, "z1": 0.030, "L": 0.075, "W": 0.025, "yaw_rad": np.pi / 2},
    ]
    _, _, rs, ds = render_synthetic_scene(cam_w, Tw, cam_s, Ts, blks)
    s_ctx = ImageProjectionContext.from_rgb(rs, cam_s.K_f64, cam_s.D_f64, Ts)
    s_data = SideViewData(cam_s, Ts, rs, ds.astype(np.float32) * 0.001, 100, s_ctx)
    slots = detect_side_facade_slots(s_data, cfg.block, cfg.thresholds, layers=2)
    by_key = {(s.layer, s.slot): s for s in slots}
    assert by_key[(1, 0)].state == "OBSERVED_EMPTY"
    assert by_key[(1, -1)].state in ("OCCUPIED", "DISPLACED")
    assert by_key[(2, -1)].state == "OCCUPIED"

    # Verifier veto test: phantom block predicted at (0.0, 0.0) on empty table must be vetoed!
    _, _, rs_empty, ds_empty = render_synthetic_scene(cam_w, Tw, cam_s, Ts, [])
    s_empty = SideViewData(cam_s, Ts, rs_empty, ds_empty.astype(np.float32) * 0.001, 200,
                           ImageProjectionContext.from_rgb(rs_empty, cam_s.K_f64, cam_s.D_f64, Ts))
    from wp3_live.proposals import BlockFit
    phantom = BlockFit((0.0, 0.0), 0.0075, 0.015, 0.0, 0.0, (75.0, 25.0), 1.0, 0.0, 50, np.zeros((50, 3)))
    ver = CroppedSilhouetteVerifier(cfg.thresholds)
    kept, vetoed, _ = ver.verify_confident_outputs([(phantom, "split:rgb_only")], s_empty, cfg.block, epoch_id=1)
    assert len(kept) == 0 and len(vetoed) == 1


def test_dwell_gate_fk_primary_tracker_and_frozen_stream():
    # 1. Galilean test: linear joint motion with ZERO IMU gyro MUST FAIL dwell gate!
    gate = WristDwellGate(require_fk=True)
    for k in range(10):
        t = k * 0.02
        gate.push_imu_gyro(t, np.array([0.0, 0.0, 0.0]))  # linear motion: zero rotation
        gate.push_joint_state(t, np.array([0.1 * t, 0.0, 0.0, 0.0, 0.0, 0.0]))  # moving at 0.1 rad/s!
    st_mov = gate.evaluate(0.18)
    assert not st_mov.stationary and st_mov.reason == "fk_missing_or_moving"

    # 2. Temporal tracker: 1.8 s hold preserves unmodified stamp; 9 mm block shift fails snapshot_commit
    cfg = preset_config("printed_jenga")
    tr = WP3TemporalTracker(cfg.block, cfg.thresholds)
    from wp3_live.proposals import BlockFit
    f0 = BlockFit((0.0, 0.0), 0.0075, 0.015, 0.0, 0.0, (75.0, 25.0), 1.0, 0.0, 50, np.zeros((50, 3)))
    out0 = tr.update([f0], ["single"], stamp_ns=1_000_000_000, now_s=1.0, side_fresh_global=True)
    assert out0[0].snapshot_commit_eligible and out0[0].observation_stamp_ns == 1_000_000_000

    # Occluded for 1.2 s (< 1.8 s hold): MUST stay OCCLUDED & stamp_ns unchanged!
    out_occ = tr.update([], [], stamp_ns=2_200_000_000, now_s=2.2, side_fresh_global=True)
    assert out_occ[0].state == "OCCLUDED" and out_occ[0].observation_stamp_ns == 1_000_000_000

    # Single block shifts 9 mm -> state DISPLACED & snapshot_commit_eligible=False!
    f_shift = BlockFit((0.009, 0.0), 0.0075, 0.015, 0.0, 0.0, (75.0, 25.0), 1.0, 0.0, 50, np.zeros((50, 3)))
    out_sh = tr.update([f_shift], ["single"], stamp_ns=2_500_000_000, now_s=2.5, side_fresh_global=True)
    assert out_sh[0].state == "DISPLACED" and not out_sh[0].snapshot_commit_eligible

    # 3. Frozen stream watchdog: repeated identical payload > freeze_timeout_s triggers frozen=True
    wd = StreamWatchdog(max_age_s=0.35, freeze_timeout_s=0.5)
    buf = bytes(range(256)) * 100
    h1 = wd.update("tripod", 1_000_000_000, 1.0, 1.0, buf)
    assert not h1.frozen
    h2 = wd.update("tripod", 1_000_000_000, 1.6, 1.6, buf)
    assert h2.frozen

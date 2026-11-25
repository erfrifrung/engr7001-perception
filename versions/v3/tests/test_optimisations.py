"""Unit & regression tests for Opus 5.5 speedups: fast_ops parity, temporal skip/warm-start, ROI alignment & async IO."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time
import warnings

import cv2
import numpy as np

IMPL_ROOT = Path(__file__).resolve().parents[1]
if str(IMPL_ROOT) not in sys.path:
    sys.path.insert(0, str(IMPL_ROOT))
sys.path.insert(0, str(IMPL_ROOT / "scripts"))

from wp3_live import EpochTemporalCache, ViewInput, preset_config, run_perception_epoch  # noqa: E402
from wp3_live.dwell_gate import DwellStatus  # noqa: E402
from wp3_live.fast_ops import (  # noqa: E402
    _heightmap_max_z_py,
    _meyer_watershed_py,
    bilinear_sample_fast,
    fast_median_1d,
    fast_nanmedian_3rows,
    fast_nanmedian_cols,
    fast_nanmedian_rows,
    fast_percentile_pair_1d,
    fast_percentiles_2col,
    fast_rolling_median_1d,
    heightmap_max_z_fast,
    meyer_watershed_2d_fast,
)
from wp3_live.geometry_cache import (  # noqa: E402
    CameraModelCache,
    deproject_raw_depth_roi_with_color,
    project_workspace_roi_2d,
)
from wp3_live.io_worker import AsyncIOArtifactWriter  # noqa: E402
from wp3_live.scoring import aggregate_method_summary  # noqa: E402
from run_printed_jenga_suite import make_synthetic_cameras, render_synthetic_scene  # noqa: E402


def test_fast_ops_parity_and_zero_all_nan_warnings():
    rng = np.random.default_rng(42)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)

        # 1. fast_median_1d across small & medium sizes
        for n in (1, 2, 3, 4, 7, 50):
            arr = rng.normal(10.0, 3.0, size=n).astype(np.float32)
            assert abs(fast_median_1d(arr) - float(np.median(arr))) < 1e-4
        assert np.isnan(fast_median_1d(np.empty(0, dtype=np.float32)))

        # 2. fast_percentiles_2col & fast_percentile_pair_1d
        pts2 = rng.uniform(-0.1, 0.1, size=(137, 2)).astype(np.float32)
        lo_f, hi_f = fast_percentiles_2col(pts2, 4.0, 96.0)
        lo_ref, hi_ref = np.percentile(pts2, [4.0, 96.0], axis=0)
        assert np.allclose(lo_f, lo_ref, atol=1e-5)
        assert np.allclose(hi_f, hi_ref, atol=1e-5)

        p1 = rng.uniform(-0.08, 0.08, size=89).astype(np.float32)
        p_lo, p_hi = fast_percentile_pair_1d(p1, 6.0, 94.0)
        ref_lo, ref_hi = np.percentile(p1, [6.0, 94.0])
        assert abs(p_lo - float(ref_lo)) < 1e-5 and abs(p_hi - float(ref_hi)) < 1e-5

        # 3. fast_nanmedian_rows with mixed NaNs + all-NaN row (MUST NOT warn!)
        mat = rng.uniform(0.0, 255.0, size=(9, 21)).astype(np.float32)
        mat[0, :5] = np.nan
        mat[4, :] = np.nan
        res_rows = fast_nanmedian_rows(mat)
        assert np.isnan(res_rows[4])
        for r in range(9):
            if r == 4:
                continue
            fin = mat[r][np.isfinite(mat[r])]
            assert abs(float(res_rows[r]) - float(np.median(fin))) < 1e-4

        # 4. fast_nanmedian_3rows (0, 1, 2, 3 valid combinations, ZERO warnings!)
        m3 = rng.uniform(0.0, 255.0, size=(3, 12)).astype(np.float32)
        m3[0, 1] = np.nan
        m3[:2, 2] = np.nan
        m3[:, 3] = np.nan
        res3 = fast_nanmedian_3rows(m3)
        assert np.isnan(res3[3])
        for c in range(12):
            if c == 3:
                continue
            fin = m3[:, c][np.isfinite(m3[:, c])]
            assert abs(float(res3[c]) - float(np.median(fin))) < 1e-4

        # 5. fast_nanmedian_cols
        col_arr = rng.uniform(0.0, 255.0, size=(15, 3)).astype(np.float32)
        col_arr[::3, 1] = np.nan
        rc = fast_nanmedian_cols(col_arr)
        for j in range(3):
            assert abs(float(rc[j]) - float(np.median(col_arr[:, j][np.isfinite(col_arr[:, j])]))) < 1e-4

        # 6. fast_rolling_median_1d matching boundary-truncated slice windows
        prof = rng.uniform(50.0, 200.0, size=31).astype(np.float32)
        prof[5] = np.nan
        prof[30] = np.nan
        med_r, cnt_r = fast_rolling_median_1d(prof, half=4, min_count=4)
        for k in range(len(prof)):
            w = prof[max(0, k - 4) : min(len(prof), k + 5)]
            w = w[np.isfinite(w)]
            assert cnt_r[k] == len(w)
            if len(w) >= 4:
                assert abs(float(med_r[k]) - float(np.median(w))) < 1e-4
            else:
                assert np.isnan(med_r[k])

        # 7. bilinear_sample_fast parity on uint8 & float32 cv2.remap
        img_u8 = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
        uv = np.column_stack([rng.uniform(1.0, 61.0, 40), rng.uniform(1.0, 61.0, 40)]).astype(np.float32)
        out_u8, in_u8 = bilinear_sample_fast(img_u8, uv)
        assert bool(in_u8.all())
        out_f32, in_f32 = bilinear_sample_fast(img_u8.astype(np.float32), uv, use_cv2_remap=True)
        assert bool(in_f32.all())
        assert np.max(np.abs(out_u8 - out_f32)) < 6.0


def test_temporal_skip_refresh_invalidation_and_warm_start():
    cfg = preset_config("printed_jenga")
    cfg.calib_epoch = "tcache_test"
    cfg.require_fk_for_moving_wrist = False
    cfg.enable_temporal_skip = True
    cfg.temporal_skip_max_frames = 3
    cfg.enable_partition_warm_start = True

    cache = CameraModelCache()
    cam_w, Tw, cam_s, Ts = make_synthetic_cameras(cache)
    dwell = DwellStatus(True, True, True, True, 0.0, 0.1, 0.2, 85.0, "ok")

    rw, dw, rs, ds = render_synthetic_scene(
        cam_w, Tw, cam_s, Ts,
        [
            {"cx": 0.0, "cy": -0.0125, "z0": 0.0, "z1": 0.015, "L": 0.075, "W": 0.025, "yaw_rad": 0.0},
            {"cx": 0.0, "cy": 0.0125, "z0": 0.0, "z1": 0.015, "L": 0.075, "W": 0.025, "yaw_rad": 0.0},
        ],
        seams_top=[{"p0": (-0.037, 0.0), "p1": (0.037, 0.0), "z": 0.015, "half_w_m": 0.0009}],
        end_breaks=[{"x": -0.0375, "y": 0.0, "z0": 0.0, "z1": 0.015}, {"x": 0.0375, "y": 0.0, "z0": 0.0, "z1": 0.015}],
    )

    tcache = EpochTemporalCache()
    # Frame 1: cold compute + store
    r1 = run_perception_epoch(
        wrist=ViewInput("wrist", cam_w, rw, dw, Tw, 100, 100),
        side=ViewInput("tripod", cam_s, rs, ds, Ts, 100, 100),
        config=cfg, policy="whole_partition_mv", dwell=dwell, epoch_id=1, temporal_cache=tcache,
    )
    assert len(r1.poses) == 2 and not r1.evidence_log.get("temporal_cache_hit")

    # Frames 2..4: stationary cache hits (zero proposal compute; full-FOV dual synthetic ROI << cold)
    for k in (2, 3, 4):
        rk = run_perception_epoch(
            wrist=ViewInput("wrist", cam_w, rw, dw, Tw, k * 100, k * 100),
            side=ViewInput("tripod", cam_s, rs, ds, Ts, k * 100, k * 100),
            config=cfg, policy="whole_partition_mv", dwell=dwell, epoch_id=k, temporal_cache=tcache,
        )
        assert rk.evidence_log.get("temporal_cache_hit") is True
        assert len(rk.poses) == 2 and rk.stamp_ns == k * 100
        assert rk.stage_timings_ms["proposals_and_rgb_ms"] == 0.0
        assert rk.stage_timings_ms["total_ms"] < 30.0
        assert rk.stage_timings_ms["total_ms"] < r1.stage_timings_ms["total_ms"]

    # Frame 5: exceeds max_skips=3 -> forced full refresh WITH warm-start hit!
    r5 = run_perception_epoch(
        wrist=ViewInput("wrist", cam_w, rw, dw, Tw, 500, 500),
        side=ViewInput("tripod", cam_s, rs, ds, Ts, 500, 500),
        config=cfg, policy="whole_partition_mv", dwell=dwell, epoch_id=5, temporal_cache=tcache,
    )
    assert not r5.evidence_log.get("temporal_cache_hit")
    assert r5.evidence_log["split_logs"][0].get("warm_start_hit") is True
    assert len(r5.poses) == 2

    # Frame 6: block disappears / scene moves -> immediate invalidation!
    rw_e, dw_e, rs_e, ds_e = render_synthetic_scene(cam_w, Tw, cam_s, Ts, [])
    r6 = run_perception_epoch(
        wrist=ViewInput("wrist", cam_w, rw_e, dw_e, Tw, 600, 600),
        side=ViewInput("tripod", cam_s, rs_e, ds_e, Ts, 600, 600),
        config=cfg, policy="whole_partition_mv", dwell=dwell, epoch_id=6, temporal_cache=tcache,
    )
    assert not r6.evidence_log.get("temporal_cache_hit")
    assert len(r6.poses) == 0

    # Dual-view D435 static side depth jitter (~9-10 mm) still hits skip, while side RGB occlusion invalidates
    tcache_mv = EpochTemporalCache()
    r7 = run_perception_epoch(
        wrist=ViewInput("wrist", cam_w, rw, dw, Tw, 700, 700),
        side=ViewInput("tripod", cam_s, rs, ds, Ts, 700, 700),
        config=cfg, policy="whole_partition_mv", dwell=dwell, epoch_id=7, temporal_cache=tcache_mv,
    )
    assert not r7.evidence_log.get("temporal_cache_hit") and len(r7.poses) == 2

    rng_ns = np.random.default_rng(123)
    ds_jitter = ds.copy()
    m_ds = (ds_jitter > 0) & (ds_jitter < 5000)
    noise_mm = rng_ns.integers(-9, 10, size=ds.shape)
    ds_jitter[m_ds] = np.clip(ds_jitter[m_ds].astype(np.int32) + noise_mm[m_ds], 200, 4900).astype(np.uint16)
    r8 = run_perception_epoch(
        wrist=ViewInput("wrist", cam_w, rw, dw, Tw, 800, 800),
        side=ViewInput("tripod", cam_s, rs, ds_jitter, Ts, 800, 800),
        config=cfg, policy="whole_partition_mv", dwell=dwell, epoch_id=8, temporal_cache=tcache_mv,
    )
    assert r8.evidence_log.get("temporal_cache_hit") is True and len(r8.poses) == 2

    s_bbox = project_workspace_roi_2d(cam_s, Ts, cfg.roi_xyz_min, cfg.roi_xyz_max, cfg.roi_pad_px)
    assert s_bbox is not None
    rs_occ = rs.copy()
    rs_occ[s_bbox[1] : s_bbox[3], s_bbox[0] : s_bbox[2]] = 0
    r9 = run_perception_epoch(
        wrist=ViewInput("wrist", cam_w, rw, dw, Tw, 900, 900),
        side=ViewInput("tripod", cam_s, rs_occ, ds, Ts, 900, 900),
        config=cfg, policy="whole_partition_mv", dwell=dwell, epoch_id=9, temporal_cache=tcache_mv,
    )
    assert not r9.evidence_log.get("temporal_cache_hit")


def test_raw_depth_roi_color_projection_and_async_io_and_numba_fallback(tmp_path):
    # 1. ROI-only raw depth-to-colour projection without RealSense align_depth
    cache = CameraModelCache()
    cam_c, Tw_c, _, _ = make_synthetic_cameras(cache)
    # Offset depth optical frame by dx = +25 mm relative to colour optical frame
    Kd = np.array([[420.0, 0.0, 320.0], [0.0, 420.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    cam_d = cache.get("wrist_depth", 640, 480, Kd, np.zeros(5), "wrist_depth_optical_frame", 0.001)
    T_color_depth = np.eye(4, dtype=np.float64)
    T_color_depth[0, 3] = -0.025  # P_c = T_color_depth @ P_d => Tw_d = Tw_c @ T_color_depth
    Tw_d = Tw_c @ T_color_depth

    # Render colour at cam_c and depth at cam_d for one bright blue/orange block vs gray table
    rw_c, _, _, _ = render_synthetic_scene(
        cam_c, Tw_c, cam_d, Tw_d,
        [{"cx": 0.0, "cy": 0.0, "z0": 0.0, "z1": 0.015, "L": 0.075, "W": 0.025, "yaw_rad": 0.0, "color": (30, 180, 240)}],
    )
    # Render depth directly from cam_d pose using rays
    _, dd_u16, _, _ = render_synthetic_scene(
        cam_d, Tw_d, cam_c, Tw_c,
        [{"cx": 0.0, "cy": 0.0, "z0": 0.0, "z1": 0.015, "L": 0.075, "W": 0.025, "yaw_rad": 0.0}],
    )
    deproj = deproject_raw_depth_roi_with_color(
        dd_u16, cam_d, Tw_d, rw_c, cam_c, T_color_depth,
        (-0.10, -0.10, -0.02), (0.10, 0.10, 0.08),
    )
    pts = deproj["pts_tbl_f32"]
    rgbs = deproj["rgb_u8"]
    top_pts = pts[:, 2] > 0.010
    assert int(np.count_nonzero(top_pts)) >= 40
    # Sampled colour on block top must recover the distinctive (30, 180, 240) block colour!
    med_rgb = np.median(rgbs[top_pts], axis=0)
    assert float(np.linalg.norm(med_rgb - np.array([30.0, 180.0, 240.0]))) < 20.0

    # 2. AsyncIOArtifactWriter non-blocking PNG backpressure & lossless JSONL
    jpath = tmp_path / "out.jsonl"
    writer = AsyncIOArtifactWriter(jpath, max_pending_png=2, flush_every=3)
    dummy_bgr = np.zeros((240, 320, 3), dtype=np.uint8)
    t0 = time.perf_counter()
    for idx in range(15):
        writer.write_jsonl(json.dumps({"ep": idx}))
        writer.write_png(tmp_path / f"ov_{idx:02d}.png", dummy_bgr)
    submit_ms = (time.perf_counter() - t0) * 1000.0
    writer.stop(timeout_s=2.0)
    assert submit_ms < 200.0
    lines = jpath.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 15

    # 3. Heightmap & watershed fast hooks match Python reference + scoring warmup/p99 check
    flat = np.array([0, 0, 3, 5, 3], dtype=np.int32)
    z = np.array([0.01, 0.02, 0.015, 0.03, 0.018], dtype=np.float32)
    h1, c1 = np.empty(6, dtype=np.float32), np.empty(6, dtype=np.int32)
    h2, c2 = np.empty(6, dtype=np.float32), np.empty(6, dtype=np.int32)
    heightmap_max_z_fast(flat, z, h1, c1)
    _heightmap_max_z_py(flat, z, h2, c2)
    assert np.array_equal(c1, c2) and np.allclose(h1, h2)

    elev = np.array([[0.0, 1.0, 2.0, 0.0], [0.0, 1.0, 2.0, 0.0], [0.0, 1.0, 2.0, 0.0]], dtype=np.float32)
    markers = np.array([[1, 0, 0, 2], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int32)
    mask = np.ones((3, 4), dtype=bool)
    assert np.array_equal(meyer_watershed_2d_fast(elev, markers, mask), _meyer_watershed_py(elev, markers, mask))

    recs = [{"status": "ok", "stage_timings_ms": {"total_ms": 100.0 if i < 5 else 10.0 + i}} for i in range(15)]
    summ = aggregate_method_summary(recs, 15, {}, warmup_frames=5)
    assert summ["warmup_excluded"] == 5
    assert "p99" in summ["timings_ms"]["total_ms"] and summ["timings_ms"]["total_ms"]["p95"] < 30.0

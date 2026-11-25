"""Golden-output replay regression gate: exact poses, partitions, abstentions & slots must match."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

IMPL_ROOT = Path(__file__).resolve().parents[1]
if str(IMPL_ROOT) not in sys.path:
    sys.path.insert(0, str(IMPL_ROOT))
sys.path.insert(0, str(IMPL_ROOT / "scripts"))

from wp3_live import ViewInput, load_runtime_config, preset_config, run_perception_epoch  # noqa: E402
from wp3_live.bag_reader import chain_tf, load_bag_fast  # noqa: E402
from wp3_live.dwell_gate import DwellStatus  # noqa: E402
from wp3_live.geometry_cache import CameraModelCache  # noqa: E402
from wp3_live.scoring import sym_yaw_err_deg  # noqa: E402
from run_printed_jenga_suite import build_printed_scenarios, make_synthetic_cameras, render_synthetic_scene  # noqa: E402


def preset_config(name, _orig=preset_config):  # noqa: F811
    """V2-parity tests exercise V2 mechanisms: the V3 region stage is disabled here (see tests/test_v3_*.py)."""
    cfg = _orig(name)
    cfg.enable_v3_regions = False
    return cfg


def load_runtime_config(path, *a, _orig=load_runtime_config, **k):  # noqa: F811
    cfg = _orig(path, *a, **k)
    cfg.enable_v3_regions = False
    return cfg

GOLDEN_PATH = Path(__file__).resolve().parent / "golden_baseline_v2.json"
_DEFAULT_BAG = Path.home() / "wp3-lab-staging/three_cam_demo_bag/three_cam_demo_bag_0.db3"


def _resolve_baseline_bag() -> Path:
    candidates: list[Path] = []
    env_bag = os.environ.get("WP3_GOLDEN_BAG_PATH", "").strip()
    if env_bag:
        candidates.append(Path(env_bag))
    candidates.append(_DEFAULT_BAG)
    for cand in candidates:
        if cand.is_file():
            return cand
    return _DEFAULT_BAG


BAG_BASELINE_PATH = _resolve_baseline_bag()
BAG_CFG_PATH = IMPL_ROOT / "ros2_ws/src/wp3_live_perception/config/historical_props_12nov.json"

SYN_POLICIES = ("whole_partition_mv", "merged_corroborated", "merged_wrist_rgb", "generic_watershed")
BAG_POLICIES = ("whole_partition_mv", "merged_corroborated")


def _serialize_epoch(res) -> dict[str, Any]:
    poses = sorted(
        [
            {
                "xy_m": [round(float(p.xy_m[0]), 6), round(float(p.xy_m[1]), 6)],
                "z_center_m": round(float(p.z_center_m), 6),
                "yaw_deg_mod": round(float(p.yaw_deg_mod), 4),
            }
            for p in res.poses
        ],
        key=lambda d: (d["xy_m"][0], d["xy_m"][1], d["z_center_m"]),
    )
    ambs = sorted(
        [
            {
                "xy_m": [round(float(a.xy_m[0]), 5), round(float(a.xy_m[1]), 5)],
                "n_est": int(a.n_est),
                "reason": str(a.reason),
            }
            for a in res.ambiguous_clusters
        ],
        key=lambda d: (d["xy_m"][0], d["xy_m"][1], d["n_est"]),
    )
    slots = [
        {"layer": int(s.layer), "slot": int(s.slot), "state": str(s.state)}
        for s in sorted(res.slot_occupancies, key=lambda x: (x.layer, x.slot))
    ]
    return {
        "status": str(res.status),
        "fallback_mode": str(res.fallback_mode),
        "n_out": len(res.poses),
        "n_amb": len(res.ambiguous_clusters),
        "pose_sources": sorted(str(s) for s in res.pose_sources),
        "poses": poses,
        "ambiguous": ambs,
        "slots": slots,
    }


def _assert_epoch_matches(actual: dict[str, Any], expected: dict[str, Any], *, sym_deg: float, ctx: str) -> None:
    assert actual["status"] == expected["status"], f"{ctx}: status {actual['status']} != {expected['status']}"
    assert actual["fallback_mode"] == expected["fallback_mode"], f"{ctx}: fallback {actual['fallback_mode']} != {expected['fallback_mode']}"
    assert actual["n_out"] == expected["n_out"], f"{ctx}: n_out {actual['n_out']} != {expected['n_out']}"
    assert actual["n_amb"] == expected["n_amb"], f"{ctx}: n_amb {actual['n_amb']} != {expected['n_amb']}"
    assert actual["pose_sources"] == expected["pose_sources"], f"{ctx}: pose_sources {actual['pose_sources']} != {expected['pose_sources']}"
    assert [a["reason"] for a in actual["ambiguous"]] == [a["reason"] for a in expected["ambiguous"]], f"{ctx}: amb reasons differ"
    assert [a["n_est"] for a in actual["ambiguous"]] == [a["n_est"] for a in expected["ambiguous"]], f"{ctx}: amb n_est differ"
    assert actual["slots"] == expected["slots"], f"{ctx}: slot states differ"

    p_act = actual["poses"]
    p_exp = expected["poses"]
    if p_exp:
        cost = np.array(
            [
                [math.hypot(a["xy_m"][0] - b["xy_m"][0], a["xy_m"][1] - b["xy_m"][1]) for b in p_exp]
                for a in p_act
            ],
            dtype=np.float64,
        )
        rr, cc = linear_sum_assignment(cost)
        for r, c in zip(rr, cc):
            xy_err_m = float(cost[r, c])
            z_err_m = abs(p_act[r]["z_center_m"] - p_exp[c]["z_center_m"])
            y_err_deg = sym_yaw_err_deg(p_act[r]["yaw_deg_mod"], p_exp[c]["yaw_deg_mod"], sym_deg)
            assert xy_err_m <= 0.0005, f"{ctx}: pose XY drift {xy_err_m * 1000.0:.3f} mm > 0.5 mm"
            assert z_err_m <= 0.0005, f"{ctx}: pose Z drift {z_err_m * 1000.0:.3f} mm > 0.5 mm"
            assert y_err_deg <= 0.5, f"{ctx}: pose yaw drift {y_err_deg:.3f} deg > 0.5 deg"


def _run_synthetic_golden() -> dict[str, dict[str, Any]]:
    cfg = preset_config("printed_jenga")
    cfg.calib_epoch = "synthetic_exact"
    cfg.require_fk_for_moving_wrist = False
    cache = CameraModelCache()
    cam_w, Tw, cam_s, Ts = make_synthetic_cameras(cache)
    dwell_ok = DwellStatus(True, True, True, True, 0.0, 0.1, 0.2, 80.0, "ok")
    out: dict[str, dict[str, Any]] = {}
    for sc_idx, sc in enumerate(build_printed_scenarios()):
        rgb_w, d_w, rgb_s, d_s = render_synthetic_scene(
            cam_w, Tw, cam_s, Ts, sc["blocks"], sc.get("seams"), sc.get("end_breaks"), sc.get("occlude_side", False),
        )
        vi_w = ViewInput("wrist", cam_w, rgb_w, d_w, Tw, (sc_idx + 1) * 1_000_000_000, (sc_idx + 1) * 1_000_000_000)
        vi_s = ViewInput("tripod", cam_s, rgb_s, d_s, Ts, (sc_idx + 1) * 1_000_000_000, (sc_idx + 1) * 1_000_000_000)
        sc_map: dict[str, Any] = {}
        for pol in SYN_POLICIES:
            res = run_perception_epoch(wrist=vi_w, side=vi_s, config=cfg, policy=pol, dwell=dwell_ok, epoch_id=sc_idx + 1)
            sc_map[pol] = _serialize_epoch(res)
        out[sc["name"]] = sc_map
    return out


def _pair_side(wrist_pairs: list[dict], side_pairs: list[dict], max_skew_s: float) -> list[dict | None]:
    if not side_pairs:
        return [None] * len(wrist_pairs)
    s_stamps = np.array([p["header_ns"] for p in side_pairs], dtype=np.int64)
    max_skew_ns = int(round(max_skew_s * 1e9))
    out: list[dict | None] = []
    for wp in wrist_pairs:
        idx = int(np.searchsorted(s_stamps, wp["header_ns"]))
        cand = []
        if idx < len(s_stamps):
            cand.append(idx)
        if idx > 0:
            cand.append(idx - 1)
        best_i = min(cand, key=lambda i: abs(int(s_stamps[i]) - wp["header_ns"]))
        out.append(side_pairs[best_i] if abs(int(s_stamps[best_i]) - wp["header_ns"]) <= max_skew_ns else None)
    return out


def _run_bag_golden(max_frames: int = 20) -> dict[str, list[dict[str, Any]]]:
    cfg = load_runtime_config(BAG_CFG_PATH)
    bag = load_bag_fast(BAG_BASELINE_PATH, [("wrist", cfg.wrist.namespace), ("side", cfg.side.namespace)], use_aligned_depth=True)
    cc = CameraModelCache()
    iw = bag.cameras["wrist"].info
    isd = bag.cameras["side"].info
    cw = cc.get("wrist", iw["width"], iw["height"], iw["K"], iw["D"], iw["frame_id"], cfg.wrist.depth_scale_m)
    cs = cc.get("tripod", isd["width"], isd["height"], isd["K"], isd["D"], isd["frame_id"], cfg.side.depth_scale_m)
    Tw = chain_tf(bag.tf_static, cfg.frame_id, cw.frame_id)
    Ts = chain_tf(bag.tf_static, cfg.frame_id, cs.frame_id)
    wp_list = bag.cameras["wrist"].pairs[:max_frames]
    sp_list = _pair_side(wp_list, bag.cameras["side"].pairs, cfg.thresholds.max_pair_skew_s)
    triples = np.random.default_rng(18).integers(0, 2048, size=(120, 3))

    out: dict[str, list[dict[str, Any]]] = {}
    for pol in BAG_POLICIES:
        recs: list[dict[str, Any]] = []
        for idx, (wp, sp) in enumerate(zip(wp_list, sp_list)):
            w_vi = ViewInput("wrist", cw, wp["rgb"], wp["depth_u16"], Tw, wp["header_ns"], wp["rx_ns"])
            s_vi = ViewInput("tripod", cs, sp["rgb"], sp["depth_u16"], Ts, sp["header_ns"], sp["rx_ns"]) if sp is not None else None
            res = run_perception_epoch(wrist=w_vi, side=s_vi, config=cfg, policy=pol, rng_triples=triples, epoch_id=idx + 1)
            recs.append(_serialize_epoch(res))
        out[pol] = recs
    return out


def capture_golden() -> None:
    payload: dict[str, Any] = {
        "synthetic": _run_synthetic_golden(),
        "bag_12nov_baseline_20": _run_bag_golden(20) if BAG_BASELINE_PATH.is_file() else {},
    }
    GOLDEN_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote golden baseline to {GOLDEN_PATH} (bag_epochs={len(payload['bag_12nov_baseline_20'].get('whole_partition_mv', []))})")


def test_golden_synthetic_printed_jenga_parity() -> None:
    assert GOLDEN_PATH.is_file(), f"Missing {GOLDEN_PATH}"
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))["synthetic"]
    actual = _run_synthetic_golden()
    assert set(actual.keys()) == set(golden.keys())
    for sc_name, sc_expected in golden.items():
        for pol, exp_ep in sc_expected.items():
            _assert_epoch_matches(actual[sc_name][pol], exp_ep, sym_deg=180.0, ctx=f"syn:{sc_name}:{pol}")


def test_golden_bag_12nov_baseline_parity() -> None:
    if not BAG_BASELINE_PATH.is_file():
        pytest.skip("12 Nov baseline bag not present on disk")
    assert GOLDEN_PATH.is_file(), f"Missing {GOLDEN_PATH}"
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8")).get("bag_12nov_baseline_20", {})
    assert golden, "Empty bag_12nov_baseline_20 in golden baseline JSON"
    actual = _run_bag_golden(len(golden[BAG_POLICIES[0]]))
    for pol, exp_list in golden.items():
        assert len(actual[pol]) == len(exp_list)
        for idx, exp_ep in enumerate(exp_list):
            _assert_epoch_matches(actual[pol][idx], exp_ep, sym_deg=90.0, ctx=f"bag:{pol}:ep{idx}")


if __name__ == "__main__":
    if "--capture" in sys.argv:
        capture_golden()

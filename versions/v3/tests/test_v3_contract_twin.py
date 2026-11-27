"""Offline contract replay through the digital twin's own JSON handler (no ROS / MoveIt needed).

`wp3_scene_bridge.on_scene_json` and `to_world` are executed from the unchanged twin source (../twin_fixture) with a
stub node; `wp3_link.BlockTracker` is the twin's real ROS-free tracker.  Checks that V3 output keeps the v2.1 schema
(same keys/types as recorded V2 payloads), that the bridge places V3 blocks at pose_table_m, and that held /
ambiguous / identity-unverified tracks are not committed with --commit-eligible-only.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
import time
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _twin_dir() -> Path:
    """Twin (MoveIt) code stays OUTSIDE the perception tree for the zero-MoveIt audit.

    Order: $WP3_TWIN_DIR, a sibling ``twin_fixture`` (original V3 bundle), the portfolio's ``integration/``.
    """
    import os
    env = os.environ.get("WP3_TWIN_DIR", "").strip()
    for cand in ([Path(env)] if env else []) + [ROOT.parent / "twin_fixture", ROOT.parents[1] / "integration"]:
        if (cand / "wp3_scene_bridge.py").is_file():
            return cand
    return ROOT.parents[1] / "integration"


FIX = _twin_dir()
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bench")); sys.path.insert(0, str(FIX))

import wp3_link  # noqa: E402  (twin code, ROS-free)
from scenes import box, cameras, render  # noqa: E402
from wp3_live import ViewInput, preset_config, run_perception_epoch  # noqa: E402
from wp3_live.contract import build_atomic_scene_payload  # noqa: E402
from wp3_live.dwell_gate import DwellStatus  # noqa: E402
from wp3_live.fusion_tracker import WP3TemporalTracker  # noqa: E402


class _Pose:
    def __init__(self):
        self.position = types.SimpleNamespace(x=0.0, y=0.0, z=0.0)
        self.orientation = types.SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)


def _bridge(commit_eligible_only: bool):
    src = (FIX / "wp3_scene_bridge.py").read_text()
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    fns = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ("on_scene_json", "to_world")]
    ns = {"json": json, "time": time, "np": np, "Pose": _Pose, "scene_id": wp3_link.scene_id,
          "rotz_quat": wp3_link.rotz_quat, "matrix_from_quat": wp3_link.matrix_from_quat,
          "quat_from_matrix": wp3_link.quat_from_matrix}
    mod = ast.Module(body=fns, type_ignores=[])
    exec(compile(mod, "wp3_scene_bridge(on_scene_json,to_world)", "exec"), ns)
    b = types.SimpleNamespace()
    b.frame = "jenga_table"            # identical frame -> identity transform branch of to_world
    b.args = types.SimpleNamespace(commit_eligible_only=commit_eligible_only)
    b.tracker = wp3_link.BlockTracker()
    b.warns = []
    b.warn = lambda key, text, every=5.0: b.warns.append(text)
    b.on_scene_json = types.MethodType(ns["on_scene_json"], b)
    b.to_world = types.MethodType(ns["to_world"], b)
    return b


def _v3_payloads():
    cw, Tw, cs, Ts = cameras()
    from sequences import seq_defs
    pair = seq_defs()["stationary_seamed_pair"][0][0]      # same geometry as the replay benchmark
    cfg = preset_config("printed_jenga")
    cfg.enable_v3_regions = True                       # V3.1: opt in explicitly (library default is the V2 path)
    cfg.v3_frame_budget_ms = cfg.v3_region_budget_ms = 1e9   # deterministic node cap only
    tr = WP3TemporalTracker(cfg.block, cfg.thresholds, persistent_joint_occlusion=True, reanchor_after=2)
    rng = np.random.default_rng(2)
    out = []
    for k in range(3):
        rw, dw = render(cw, Tw, pair, rng, noise_m=0.00045)
        rs, ds = render(cs, Ts, pair, rng, noise_m=0.0008)
        st = (k + 1) * 66_666_667
        ep = run_perception_epoch(wrist=ViewInput("wrist", cw, rw, dw, Tw, st, st), side=ViewInput("tripod", cs, rs, ds, Ts, st, st),
                                  config=cfg, policy="merged_corroborated",
                                  dwell=DwellStatus(True, True, True, True, 0.0, 0.1, 0.2, 80.0, "ok"), epoch_id=k + 1)
        tracks = tr.update(ep.poses, ep.pose_sources, ep.stamp_ns, st * 1e-9 + 0.03, side_fresh_global=True,
                           pose_side_visible=ep.pose_side_visible)
        out.append(json.loads(json.dumps(build_atomic_scene_payload(ep, tracks, cfg, epoch_id=k + 1,
                                                                    publish_stamp_ns=st + 30_000_000, code_hash="test"))))
    return out


def _types(d):
    return {k: type(v).__name__ for k, v in d.items()}


def test_v3_payload_schema_matches_recorded_v2():
    rec = json.loads((FIX / "display_replay_scene_observations.json").read_text())
    v3 = _v3_payloads()
    assert all(p["schema_version"] == "wp3_scene_observation_v2.1" for p in v3)
    assert set(rec[0]) == set(v3[0]), set(rec[0]) ^ set(v3[0])
    assert all(p["frame_id"] == "jenga_table" for p in v3)
    for k in ("epoch_id", "publish_stamp_ns", "observation_stamp_ns", "tracks", "ambiguous_clusters", "slot_occupancies"):
        assert _types(rec[0])[k] == _types(v3[0])[k]
    t = v3[-1]["tracks"][0]
    for k in ("pose_table_m", "yaw_rad", "dims_mm", "validity_mask", "observation_stamp_ns", "state",
              "snapshot_commit_eligible", "identity_ambiguous"):
        assert k in t
    assert t["observation_stamp_ns"] <= v3[-1]["publish_stamp_ns"]


def test_twin_bridge_places_v3_blocks_and_parses_recorded_v2():
    rec = json.loads((FIX / "display_replay_scene_observations.json").read_text())
    b = _bridge(commit_eligible_only=True)
    for p in rec:
        b.on_scene_json(types.SimpleNamespace(data=json.dumps(p)))
    assert not b.warns
    v3 = _v3_payloads()
    for p in v3:
        b.on_scene_json(types.SimpleNamespace(data=json.dumps(p)))
    assert not b.warns
    published = {wp3_link.scene_id("wp3_blocks", t["track_id"]): t for t in v3[-1]["tracks"]
                 if t["state"] in ("MEASURED", "DISPLACED") and t["snapshot_commit_eligible"]}
    assert len(published) == 2, "V3 should publish the seamed flush pair"
    for key, t in published.items():
        dims, xyz, quat, _ = b.tracker.latest[key]
        assert np.allclose(xyz, t["pose_table_m"], atol=1e-9)
        # V3.1: dims are measured where both opposite edges are FREE-bordered (were always nominal)
        assert abs(max(dims[:2]) - 0.075) < 0.0025 and abs(min(dims[:2]) - 0.025) < 0.0025
    acts = b.tracker.actions(time.monotonic())
    assert {a[1] for a in acts if a[0] in ("add", "move")} >= set(published)


def _msg(base, tracks, status="ok"):
    return types.SimpleNamespace(data=json.dumps(dict(base, tracks=tracks, status=status)))


def test_bridge_removes_retired_tracks_and_ignores_bad_status():
    """V3.1: a track missing from the atomic snapshot is retired -> its collision object goes; a non-ok epoch
    (e.g. 'uncalibrated') changes nothing (it is not evidence that blocks left)."""
    base = _v3_payloads()[-1]                      # built once: the steps below must run inside stale_s
    t0 = dict(base["tracks"][0], track_id=11)
    t1 = dict(base["tracks"][0], track_id=12, pose_table_m=[0.05, 0.05, t0["pose_table_m"][2]])
    m_both, m_bad, m_one = _msg(base, [t0, t1]), _msg(base, [], status="uncalibrated"), _msg(base, [t0])
    b = _bridge(commit_eligible_only=False)
    b.on_scene_json(m_both)
    first = b.tracker.actions(time.monotonic())
    assert {a[1] for a in first if a[0] == "add"} == {"wp3_blocks_11", "wp3_blocks_12"}
    b.on_scene_json(m_bad)
    assert b.tracker.actions(time.monotonic()) == []
    assert any("uncalibrated" in w for w in b.warns)
    b.on_scene_json(m_one)                         # 12 retired upstream
    assert ("remove", "wp3_blocks_12") in b.tracker.actions(time.monotonic())


def test_bridge_keeps_identity_ambiguous_blocks_as_obstacles_by_default():
    base = _v3_payloads()[-1]
    amb = dict(base["tracks"][0], track_id=21, identity_ambiguous=True, snapshot_commit_eligible=False, state="MEASURED")
    b = _bridge(commit_eligible_only=False)
    b.on_scene_json(_msg(base, [amb]))
    assert "wp3_blocks_21" in b.tracker.latest


def test_block_tracker_attached_and_rejected_updates_are_not_lost():
    tr = wp3_link.BlockTracker()
    q = (0.0, 0.0, 0.0, 1.0)
    tr.observe("a", (0.075, 0.025, 0.015), (0.0, 0.0, 0.0), q, 0.0)
    tr.observe("b", (0.075, 0.025, 0.015), (0.1, 0.0, 0.0), q, 0.0)
    assert {k for _, k, *_ in tr.actions(0.1)} == {"a", "b"}
    tr.retain_only({"a"})                     # b retired while it is in the gripper
    assert tr.actions(0.2, skip={"b"}) == []  # nothing applied for an attached object ...
    assert ("remove", "b") in tr.actions(0.3)  # ... and the removal is still pending after release
    tr.observe("a", (0.075, 0.025, 0.015), (0.0, 0.02, 0.0), q, 0.4)
    snap = tr.snapshot()
    assert tr.actions(0.5)[0][0] == "move"
    tr.restore(snap)                          # MoveIt rejected the diff
    assert tr.actions(0.6)[0][0] == "move"     # retried, not lost


def test_block_tracker_holds_removal_just_after_release():
    """Twin v5 + V3.1: a block retired upstream while in the gripper is not removed the instant it is released
    (hold_removals), and a re-observation during the hold cancels the removal."""
    tr = wp3_link.BlockTracker()
    q = (0.0, 0.0, 0.0, 1.0)
    tr.observe("b", (0.075, 0.025, 0.015), (0.1, 0.0, 0.0), q, 0.0)
    tr.actions(0.1)
    tr.retain_only(set())                                   # retired while held
    assert tr.actions(0.2, skip={"b"}) == []
    assert tr.actions(0.3, hold_removals={"b"}) == []        # just released: removal held
    tr.observe("b", (0.075, 0.025, 0.015), (0.12, 0.05, 0.0), q, 0.4)   # camera re-reports it at the drop pose
    acts = tr.actions(0.5, hold_removals={"b"})
    assert [a[0] for a in acts] == ["move"] and ("remove", "b") not in acts


def test_bridge_skips_held_and_ineligible_tracks():
    v3 = _v3_payloads()[-1]
    t0 = dict(v3["tracks"][0])
    held = dict(t0, state="OCCLUDED", track_id=901)
    unverified = dict(t0, identity_ambiguous=True, snapshot_commit_eligible=False, track_id=902)
    p = dict(v3, tracks=[held, unverified])
    b = _bridge(commit_eligible_only=True)
    calls = []
    b.tracker.observe = lambda key, *a, **k: calls.append(key)
    b.on_scene_json(types.SimpleNamespace(data=json.dumps(p)))
    assert calls == []


def test_bridge_drops_held_poses_on_calibration_epoch_change():
    base = _v3_payloads()[-1]
    t0 = dict(base["tracks"][0], track_id=31, state="MEASURED", snapshot_commit_eligible=True)
    t0_held = dict(t0, state="OCCLUDED", snapshot_commit_eligible=False)
    t_new = dict(base["tracks"][0], track_id=32, state="MEASURED", snapshot_commit_eligible=True,
                 pose_table_m=[0.06, 0.04, t0["pose_table_m"][2]])
    b = _bridge(commit_eligible_only=False)
    m1 = dict(base, calibration_epoch="calib_1", tracks=[t0], status="ok")
    m2 = dict(base, calibration_epoch="calib_2", tracks=[t0_held, t_new], status="ok")
    b.on_scene_json(types.SimpleNamespace(data=json.dumps(m1)))
    acts1 = b.tracker.actions(time.monotonic())
    assert {a[1] for a in acts1 if a[0] == "add"} == {"wp3_blocks_31"}
    b.on_scene_json(types.SimpleNamespace(data=json.dumps(m2)))
    acts2 = b.tracker.actions(time.monotonic())
    assert ("remove", "wp3_blocks_31") in acts2
    assert any(a[0] == "add" and a[1] == "wp3_blocks_32" for a in acts2)
    assert not b.warns

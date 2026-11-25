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
FIX = ROOT.parent / "twin_fixture"   # twin (MoveIt) code stays OUTSIDE the perception tree: zero-MoveIt audit
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
        assert abs(max(dims[:2]) - 0.075) < 1e-6 and abs(min(dims[:2]) - 0.025) < 1e-6
    acts = b.tracker.actions(time.monotonic())
    assert {a[1] for a in acts if a[0] in ("add", "move")} >= set(published)


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

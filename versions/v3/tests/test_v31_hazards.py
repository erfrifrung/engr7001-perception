"""V3.1 evidence: reproductions of known limitations (offline, deterministic, no ROS).

1. Healthy-camera prolonged occlusion: a block hidden (e.g. under the arm) for > retire_s while the cameras are
   fine is retired by the tracker and then removed from the MoveIt collision world by the bridge, although nobody
   observed the space to be empty.
2. Calibration-epoch change: the JSON bridge ignores calibration_epoch, so a held (OCCLUDED / STALE) track keeps a
   pose computed under the previous calibration.
3. The live node caches fixed-camera transforms per calib_epoch, and calib_epoch is fixed for the node's lifetime:
   re-registering the board while the node runs does not reach the tripod / third views until a restart.

Each limitation has a strict xfail asserting the SAFE behaviour (flips to a failure when fixed) plus a passing test
that records the current, measured behaviour for the FMEA.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time
import types

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wp3_live import preset_config  # noqa: E402
from wp3_live.fusion_tracker import WP3TemporalTracker  # noqa: E402
from wp3_live.geometry_cache import StaticTFCache  # noqa: E402
from wp3_live.proposals import BlockFit  # noqa: E402
from test_v3_contract_twin import _bridge  # noqa: E402

CFG = preset_config("printed_jenga")
H = CFG.block.height_m
FPS = 15.0


def _fit(x=0.0, y=0.0):
    return BlockFit(xy_m=(x, y), z_center_m=H / 2, z_top_m=H, yaw_rad=0.0, yaw_deg_mod=0.0, dims_mm=(75.0, 25.0),
                    fill=1.0, size_err=0.0, n_points=500, points_xyz=np.zeros((0, 3)), covariance_xy_m2=None)


def _payload(tracks, status="ok", epoch="cal_A"):
    out = []
    for t in tracks:
        out.append(dict(track_id=t.track_id, block_id=t.block_name, state=t.state,
                        pose_table_m=[t.fit.xy_m[0], t.fit.xy_m[1], t.fit.z_center_m], z_top_table_rel_m=t.fit.z_top_m,
                        yaw_rad=t.fit.yaw_rad, dims_mm=list(t.fit.dims_mm), snapshot_commit_eligible=t.snapshot_commit_eligible,
                        identity_ambiguous=t.identity_ambiguous, observation_stamp_ns=t.observation_stamp_ns))
    return types.SimpleNamespace(data=json.dumps(dict(schema_version="wp3_scene_observation_v2.1", frame_id="jenga_table",
                                                      status=status, fallback_mode="dual_view", calibration_epoch=epoch,
                                                      tracks=out, ambiguous_clusters=[])))


def occlusion_timeline(visible_s=1.0, hidden_s=4.0):
    """Block seen for visible_s, then not detected for hidden_s with healthy cameras (status ok, dual_view).
    Returns per-frame (t_since_hidden, tracker state or None, bridge has the obstacle)."""
    tr = WP3TemporalTracker(CFG.block, CFG.thresholds, persistent_joint_occlusion=True, reanchor_after=2)
    b = _bridge(commit_eligible_only=False)
    t0 = 1.0
    out = []
    n_vis, n_hid = int(visible_s * FPS), int(hidden_s * FPS)
    for k in range(n_vis + n_hid):
        t = t0 + k / FPS
        fits = [_fit()] if k < n_vis else []
        tracks = tr.update(fits, ["single"] * len(fits), int(t * 1e9), t, side_fresh_global=True)
        b.on_scene_json(_payload(tracks))
        b.tracker.actions(time.monotonic())
        state = tracks[0].state if tracks else None
        out.append((round(t - (t0 + n_vis / FPS), 3), state, "wp3_blocks_1" in b.tracker.applied))
    return out


def test_occlusion_timeline_is_as_documented():
    tl = occlusion_timeline()
    first = lambda pred: next((t for t, s, present in tl if pred(s, present)), None)
    t_occl = first(lambda s, p: s == "OCCLUDED")
    t_stale = first(lambda s, p: s == "STALE")
    t_retired = first(lambda s, p: s is None and t_occl is not None)
    t_removed = first(lambda s, p: not p and t_occl is not None)
    print(f"\nhidden at t=0 with healthy cameras: OCCLUDED at {t_occl} s, STALE at {t_stale} s, "
          f"retired at {t_retired} s, removed from the MoveIt collision world at {t_removed} s")
    th = CFG.thresholds
    assert t_occl is not None and t_occl <= 1 / FPS + 1e-6
    assert t_stale is not None and abs(t_stale - th.occlusion_hold_s) <= 1.5 / FPS
    assert t_retired is not None and abs(t_retired - th.retire_s) <= 1.5 / FPS
    assert t_removed == t_retired           # the bridge removes the obstacle in the same snapshot


@pytest.mark.xfail(strict=True, reason=(
    "KNOWN LIMITATION (FMEA): with healthy cameras a block hidden for more than retire_s (3.0 s), e.g. under the "
    "arm, is retired and removed from the MoveIt collision world although its space was never observed empty. "
    "Proper fix (post-M1): retire only when the footprint is observed FREE; keep 'unobserved' as UNKNOWN occupancy."))
def test_hidden_block_stays_an_obstacle_until_observed_empty():
    tl = occlusion_timeline(hidden_s=4.0)
    assert tl[-1][2], "obstacle removed although its space was never observed empty"


@pytest.mark.xfail(strict=True, reason=(
    "KNOWN LIMITATION: the JSON bridge ignores calibration_epoch. After a recalibration, a held (OCCLUDED/STALE) "
    "track keeps the pose measured under the previous calibration. Workaround: restart WP3 and the bridge after any "
    "recalibration."))
def test_bridge_drops_held_poses_from_a_previous_calibration():
    tr = WP3TemporalTracker(CFG.block, CFG.thresholds)
    b = _bridge(commit_eligible_only=False)
    tracks = tr.update([_fit(0.01, 0.0)], ["single"], int(1e9), 1.0, side_fresh_global=True)
    b.on_scene_json(_payload(tracks, epoch="cal_A"))
    b.tracker.actions(time.monotonic())
    tracks = tr.update([], [], int(1.2e9), 1.2, side_fresh_global=True)          # now held (OCCLUDED)
    assert tracks[0].state == "OCCLUDED"
    b.on_scene_json(_payload(tracks, epoch="cal_B"))                             # recalibrated meanwhile
    b.tracker.actions(time.monotonic())
    assert "wp3_blocks_1" not in b.tracker.applied, "old-calibration pose still used as an obstacle"


def test_fixed_camera_transform_is_cached_for_the_node_lifetime():
    """Documents current behaviour: StaticTFCache only invalidates when calib_epoch changes, and the live node's
    calib_epoch comes from the config file at start-up, so re-published board TFs are not picked up until restart."""
    c = StaticTFCache()
    old, new = np.eye(4), np.eye(4)
    new[0, 3] = 0.01
    c.put("cal_A", "jenga_table", "tripod_link", old)
    assert np.allclose(c.get("cal_A", "jenga_table", "tripod_link"), old)      # a new TF for cal_A is never looked up
    assert c.get("cal_B", "jenga_table", "tripod_link") is None                # only a new epoch forces a fresh lookup

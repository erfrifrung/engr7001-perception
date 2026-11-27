"""V3.2 tracker regressions.

1. The duplicate-track spiral seen in the 26 Nov live soak (3203 ids for 3 objects in 180 s).
2. Fault #12: a block hidden for > 3 s with healthy cameras was retired and removed from MoveIt. V3.2 keeps it until
   its footprint is SEEN empty (or hidden_retire_max_s).
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_v31_tracker import CFG, H, _fit, _step  # noqa: E402
from wp3_live import load_runtime_config  # noqa: E402
from wp3_live.free_space import FreeSpaceMap  # noqa: E402
from wp3_live.fusion_tracker import WP3TemporalTracker, tracker_from_config  # noqa: E402
from wp3_live.heightmap_core import HeightmapGrid  # noqa: E402

FPS = 15


def _tr32(**kw):
    opts = dict(persistent_joint_occlusion=True, reanchor_after=2, merge_overlapping=True)
    opts.update(kw)
    return WP3TemporalTracker(CFG.block, CFG.thresholds, **opts)


# ------------------------------------------------------------------------------------------- 1. duplicate spiral
def _run_split_detection(tr, frames_after=45):
    """A block, then ONE frame where it is detected twice 1 mm apart, then the normal single detection again."""
    k = 0
    for _ in range(5):
        _step(tr, [_fit(0.0)], k); k += 1
    _step(tr, [_fit(0.0), _fit(0.001)], k); k += 1          # near-tie: the track's row is vetoed, both spawn
    ids_seen = set()
    out = []
    for _ in range(frames_after):                             # 3 s at 15 Hz
        out = _step(tr, [_fit(0.0)], k); k += 1
        ids_seen.update(t.track_id for t in out)
    return out, ids_seen


def test_v31_behaviour_reproduces_the_duplicate_spiral():
    tr = _tr32(merge_overlapping=False)
    out, ids_seen = _run_split_detection(tr)
    # one physical block: V3.1 spawns a new id on (almost) every frame and holds a pile of duplicates
    assert len(ids_seen) > 30
    assert len(out) > 30


def test_v32_one_block_keeps_one_id_after_a_split_detection():
    tr = _tr32()
    out, ids_seen = _run_split_detection(tr)
    assert [t.track_id for t in out] == [1]
    assert ids_seen == {1}
    (b,) = out
    assert b.state == "MEASURED" and b.snapshot_commit_eligible
    assert tr.merged_total >= 1


def test_v32_merged_track_is_never_a_target_on_the_merge_frame():
    tr = _tr32()
    for k in range(5):
        _step(tr, [_fit(0.0)], k)
    (b,) = _step(tr, [_fit(0.0), _fit(0.001)], 5)
    assert b.track_id == 1 and not b.snapshot_commit_eligible


def test_v32_touching_blocks_are_never_merged():
    tr = _tr32()
    w = CFG.block.minor_m
    for k in range(10):                                       # side by side (25 mm) and end to end (75 mm)
        out = _step(tr, [_fit(0.0), _fit(w), _fit(0.0, y=CFG.block.major_m)], k)
    assert len(out) == 3 and tr.merged_total == 0


def test_v32_blocks_on_different_layers_are_never_merged():
    tr = _tr32()
    for k in range(10):
        out = _step(tr, [_fit(0.0, z_top=H), _fit(0.0, z_top=2 * H)], k)
    assert len(out) == 2 and tr.merged_total == 0


def test_v32_yaw_flip_in_place_does_not_leave_a_duplicate():
    """A detection at the same spot whose yaw jumps past the 30 deg gate (e.g. a square prop with the block model)."""
    tr = _tr32()
    k = 0
    for _ in range(5):
        _step(tr, [_fit(0.0, yaw=0.0)], k); k += 1
    for _ in range(45):
        yaw = 0.0 if k % 2 else np.pi / 2
        out = _step(tr, [_fit(0.0, yaw=yaw)], k); k += 1
        assert len(out) == 1, f"frame {k}: {len(out)} tracks for one block"
    assert out[0].track_id == 1


# ----------------------------------------------------------------------------------- 2. observed-free retirement
GRID = HeightmapGrid.from_roi((-0.16, -0.16), (0.16, 0.16), 0.002)


def _hm(fill=np.nan, patches=()):
    """A heightmap: `fill` everywhere, then axis-aligned (x0, x1, y0, y1, h) patches."""
    h = np.full((GRID.ny, GRID.nx), fill, dtype=np.float32)
    for x0, x1, y0, y1, z in patches:
        c0, c1 = int((x0 - GRID.x0) / GRID.cell_m), int((x1 - GRID.x0) / GRID.cell_m)
        r0, r1 = int((y0 - GRID.y0) / GRID.cell_m), int((y1 - GRID.y0) / GRID.cell_m)
        h[r0:r1, c0:c1] = z
    return FreeSpaceMap(grid=GRID, h_max=h, stamp_ns=0)


L, W = CFG.block.major_m, CFG.block.minor_m
TABLE = _hm(0.0005)                                                     # table visible everywhere, nothing on it
BLOCK_THERE = _hm(0.0005, [(-L / 2, L / 2, -W / 2, W / 2, H)])          # the block's top is still seen
ARM_OVER = _hm(0.0005, [(-0.05, 0.05, -0.03, 0.03, 0.20)])             # an occluder above the footprint
NO_DEPTH = _hm(np.nan)                                                  # robot self-filter / no depth: unobserved


def _seq(tr, k0, n, fits, fs):
    out = None
    for k in range(k0, k0 + n):
        t = 1.0 + k / FPS
        out = tr.update(fits, ["test"] * len(fits), int(t * 1e9), t, side_fresh_global=True, free_space=fs)
    return out, k0 + n


def test_footprint_probe():
    fp = (0.0, 0.0)
    assert TABLE.footprint_free(fp, 0.0, H, CFG.block)
    assert not BLOCK_THERE.footprint_free(fp, 0.0, H, CFG.block)
    assert not ARM_OVER.footprint_free(fp, 0.0, H, CFG.block)
    assert not NO_DEPTH.footprint_free(fp, 0.0, H, CFG.block)
    # an upper-layer block removed: the layer below is now its footprint's top
    assert BLOCK_THERE.footprint_free(fp, 0.0, 2 * H, CFG.block)
    # half the footprint unobserved: not enough evidence
    half = _hm(np.nan, [(0.0, 0.16, -0.16, 0.16, 0.0005)])
    assert not half.footprint_free(fp, 0.0, H, CFG.block)


def test_v31_behaviour_reproduces_fault_12():
    tr = _tr32(retire_on_observed_free=False)
    _, k = _seq(tr, 0, 5, [_fit(0.0)], BLOCK_THERE)
    out, _ = _seq(tr, k, int(4.0 * FPS), [], ARM_OVER)                  # hidden 4 s under the arm
    assert out == []                                                    # V3.1: retired after 3 s


def test_v32_block_hidden_under_the_arm_stays_an_obstacle():
    tr = _tr32(retire_on_observed_free=True)
    _, k = _seq(tr, 0, 5, [_fit(0.0)], BLOCK_THERE)
    out, k = _seq(tr, k, int(20.0 * FPS), [], ARM_OVER)                 # 20 s under the arm
    (b,) = out
    assert b.track_id == 1 and b.state == "STALE" and not b.snapshot_commit_eligible
    out, _ = _seq(tr, k, int(5.0 * FPS), [], NO_DEPTH)                  # then 5 s with no depth at all
    assert [t.track_id for t in out] == [1]


def test_v32_block_hidden_then_seen_again_keeps_its_id():
    tr = _tr32(retire_on_observed_free=True)
    _, k = _seq(tr, 0, 5, [_fit(0.0)], BLOCK_THERE)
    _, k = _seq(tr, k, int(10.0 * FPS), [], ARM_OVER)
    out, _ = _seq(tr, k, 3, [_fit(0.0)], BLOCK_THERE)
    (b,) = out
    assert b.track_id == 1 and b.state == "MEASURED"


def test_v32_removed_block_is_retired_once_its_space_is_seen_empty():
    tr = _tr32(retire_on_observed_free=True, free_confirm_frames=3)
    _, k = _seq(tr, 0, 5, [_fit(0.0)], BLOCK_THERE)
    out, k = _seq(tr, k, 2, [], TABLE)
    assert len(out) == 1                                                # 2 empty epochs: not yet
    out, _ = _seq(tr, k, 1, [], TABLE)
    assert out == [] and tr.retired_free_total == 1                     # 3rd: retired (~0.2 s, not 3 s)


def test_v32_empty_evidence_must_be_consecutive():
    tr = _tr32(retire_on_observed_free=True, free_confirm_frames=3)
    _, k = _seq(tr, 0, 5, [_fit(0.0)], BLOCK_THERE)
    for fs in (TABLE, TABLE, ARM_OVER, TABLE, TABLE, None, TABLE, TABLE):   # never 3 in a row
        out, k = _seq(tr, k, 1, [], fs)
    assert len(out) == 1


def test_v32_unseen_cap_still_applies():
    tr = _tr32(retire_on_observed_free=True, hidden_retire_max_s=10.0)
    _, k = _seq(tr, 0, 5, [_fit(0.0)], BLOCK_THERE)
    out, k = _seq(tr, k, int(9.0 * FPS), [], ARM_OVER)
    assert len(out) == 1
    out, _ = _seq(tr, k, int(2.0 * FPS), [], ARM_OVER)
    assert out == []


def test_v32_config_defaults_and_json_switches(tmp_path):
    cfg = load_runtime_config(None)
    tr = tracker_from_config(cfg)
    assert tr.merge_overlapping and tr.retire_on_observed_free and tr.hidden_retire_max_s == 60.0
    p = tmp_path / "v31.json"
    p.write_text('{"tracker_merge_overlapping": false, "retire_on_observed_free": false}')
    tr = tracker_from_config(load_runtime_config(str(p)))
    assert not tr.merge_overlapping and not tr.retire_on_observed_free


def test_v32_displaced_block_stays_displaced_across_duplicate_merge():
    tr = _tr32()
    for k in range(5):
        _step(tr, [_fit(0.0)], k)
    out = _step(tr, [_fit(0.020)], 5)
    assert out[0].state == "DISPLACED"
    out = _step(tr, [_fit(0.020), _fit(0.021)], 6)
    assert len(out) == 1 and out[0].track_id == 1
    assert out[0].state == "DISPLACED"
    assert tr.merged_total >= 1

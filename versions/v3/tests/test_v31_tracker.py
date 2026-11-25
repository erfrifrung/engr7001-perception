"""V3.1 tracker regressions: layer gate, slide-safe re-anchoring, retired-track bookkeeping."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wp3_live import preset_config  # noqa: E402
from wp3_live.fusion_tracker import WP3TemporalTracker  # noqa: E402
from wp3_live.proposals import BlockFit  # noqa: E402

CFG = preset_config("printed_jenga")
H = CFG.block.height_m


def _fit(x, y=0.0, z_top=H, yaw=0.0) -> BlockFit:
    return BlockFit(xy_m=(float(x), float(y)), z_center_m=z_top - H / 2, z_top_m=z_top, yaw_rad=yaw,
                    yaw_deg_mod=float(np.degrees(yaw) % 180.0), dims_mm=(75.0, 25.0), fill=1.0, size_err=0.0,
                    n_points=500, points_xyz=np.zeros((0, 3)), covariance_xy_m2=None)


def _step(tr, fits, k, dt=1 / 15):
    t = 1.0 + k * dt
    return tr.update(fits, ["test"] * len(fits), int(t * 1e9), t, side_fresh_global=True)


def test_block_on_another_layer_gets_a_new_identity():
    tr = WP3TemporalTracker(CFG.block, CFG.thresholds)
    (a,) = _step(tr, [_fit(0.0, z_top=H)], 0)
    # same (x, y, yaw), two layers higher: a different physical block
    out = _step(tr, [_fit(0.0, z_top=3 * H)], 1)
    ids = {t.track_id: t for t in out}
    assert len(ids) == 2
    assert ids[a.track_id].state == "OCCLUDED"


def test_steady_slide_never_reanchors_mid_motion():
    tr = WP3TemporalTracker(CFG.block, CFG.thresholds, reanchor_after=2)
    k = 0
    for _ in range(3):
        (b,) = _step(tr, [_fit(0.0)], k); k += 1
    assert b.state == "MEASURED"
    x = 0.0
    for _ in range(20):            # 1.5 mm/frame = 22.5 mm/s at 15 Hz: consecutive frames are < 2 mm apart
        x += 0.0015
        (b,) = _step(tr, [_fit(x)], k); k += 1
        if x >= 0.008:
            assert b.state == "DISPLACED", f"re-anchored while sliding at x={x * 1000:.1f} mm"
            assert not b.snapshot_commit_eligible
    for _ in range(3):             # block stops: re-anchors after reanchor_after consistent frames
        (b,) = _step(tr, [_fit(x)], k); k += 1
    assert b.state == "MEASURED"


def test_retired_track_bookkeeping_is_dropped_but_peer_stays_unverified():
    tr = WP3TemporalTracker(CFG.block, CFG.thresholds, persistent_joint_occlusion=True, reanchor_after=2)
    out = _step(tr, [_fit(-0.04), _fit(0.04)], 0)
    a_id, b_id = sorted(t.track_id for t in out)
    _step(tr, [], 1)                                   # both hidden together
    k = 2
    for _ in range(int((CFG.thresholds.retire_s + 0.5) * 15)):   # only one comes back; the other retires
        out = _step(tr, [_fit(-0.04)], k); k += 1
    live = {t.track_id for t in out}
    assert live == {a_id}
    assert b_id not in tr._co_hidden and b_id not in tr._unverified
    (a,) = out
    assert a.identity_ambiguous and b_id in a.ambiguous_peer_ids   # may be the block that left: stays unverified

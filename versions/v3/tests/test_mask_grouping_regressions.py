"""Regressions for defects reproduced in the WP3 V2 external-mask path and multi-layer band selection.

All fixtures are synthetic RGB-D renders with ideal modal top-face masks (no learned model involved).
Each test pins the legacy behaviour that failed and the behaviour of the fix, so the ablation stays visible.
"""
from __future__ import annotations

import itertools
from pathlib import Path
import sys

import numpy as np
import pytest

IMPL_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(IMPL_ROOT), str(IMPL_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run_printed_jenga_suite import make_synthetic_cameras, render_synthetic_scene  # noqa: E402
from wp3_live import policies  # noqa: E402
from wp3_live.config import preset_config  # noqa: E402
from wp3_live.dwell_gate import DwellStatus  # noqa: E402
from wp3_live.geometry_cache import CameraModelCache  # noqa: E402
from wp3_live.proposals import suppress_merged_masks  # noqa: E402


def preset_config(name, _orig=preset_config):  # noqa: F811
    """V2-parity tests exercise V2 mechanisms: the V3 region stage is disabled here (see tests/test_v3_*.py)."""
    cfg = _orig(name)
    cfg.enable_v3_regions = False
    return cfg

H = 0.015
CAM_W, T_W, CAM_S, T_S = make_synthetic_cameras(CameraModelCache())
DWELL = DwellStatus(True, True, True, True, 0.0, 0.1, 0.2, 80.0, "ok")


def _blk(cx, cy, yaw_deg=0.0, z0=0.0):
    return {"cx": cx, "cy": cy, "z0": z0, "z1": z0 + H, "L": 0.075, "W": 0.025, "yaw_rad": float(np.radians(yaw_deg))}


def _top_mask(b, depth, lo=0.0, hi=1.01):
    rays = CAM_W.rays_f32 @ T_W[:3, :3].T
    o = T_W[:3, 3]
    t = (b["z1"] - o[2]) / rays[:, :, 2]
    xy = o[:2] + t[:, :, None] * rays[:, :, :2]
    d = xy - [b["cx"], b["cy"]]
    a = b["yaw_rad"]
    along = (d[..., 0] * np.cos(a) + d[..., 1] * np.sin(a)) / b["L"] + 0.5
    across = -d[..., 0] * np.sin(a) + d[..., 1] * np.cos(a)
    vis = np.abs(depth.astype(np.float32) * 0.001 - t) < 0.0025
    return vis & (along >= max(0.0, lo)) & (along < hi) & (along <= 1.0) & (np.abs(across) <= b["W"] / 2)


def _cfg(grouping="search", suppress=True, top_band=True):
    cfg = preset_config("printed_jenga")
    cfg.calib_epoch = "synthetic_exact"
    cfg.require_fk_for_moving_wrist = False
    cfg.mask_grouping = grouping
    cfg.mask_suppress_merged = suppress
    cfg.top_band_from_upper_cells = top_band
    return cfg


def _run(images, masks, cfg, policy="whole_partition_mv"):
    rgb_w, d_w, rgb_s, d_s = images
    return policies.run_perception_epoch(
        wrist=policies.ViewInput("wrist", CAM_W, rgb_w, d_w, T_W, 10**9, 10**9),
        side=policies.ViewInput("tripod", CAM_S, rgb_s, d_s, T_S, 10**9, 10**9),
        config=cfg, policy=policy, dwell=DWELL, external_masks=masks,
    )


def _centre_errors_mm(obs, blocks):
    from scipy.optimize import linear_sum_assignment
    if not obs.poses:
        return []
    truth = np.array([[b["cx"], b["cy"]] for b in blocks])
    pred = np.array([p.xy_m for p in obs.poses])
    dist = np.linalg.norm(pred[:, None] - truth[None], axis=2)
    r, c = linear_sum_assignment(dist)
    return sorted(1000.0 * dist[r, c])


PAIR = [_blk(0.0, -0.0125), _blk(0.0, 0.0125)]
PAIR_ENDS = [{"x": -0.0375, "y": 0.0, "z0": 0.0, "z1": H}, {"x": 0.0375, "y": 0.0, "z0": 0.0, "z1": H}]
STRIPE_ON_A = [{"p0": (0.0, -0.0245), "p1": (0.0, -0.0005), "z": H, "half_w_m": 0.0009}]


@pytest.fixture(scope="module")
def seamless_pair():
    return render_synthetic_scene(CAM_W, T_W, CAM_S, T_S, PAIR, STRIPE_ON_A, PAIR_ENDS, False)


def test_pixel_suppression_deletes_true_single_with_false_parts():
    """Confirmed counterexample: a correct whole-block mask plus two false parts is suppressed in 2D."""
    images = render_synthetic_scene(CAM_W, T_W, CAM_S, T_S, [_blk(0.0, 0.0)], None, None, False)
    b = _blk(0.0, 0.0)
    whole, left, right = _top_mask(b, images[1]), _top_mask(b, images[1], 0.0, 0.5), _top_mask(b, images[1], 0.5)
    kept = suppress_merged_masks([whole, left, right])
    assert len(kept) == 2 and not any(k is whole for k in kept)


def test_false_parts_next_to_neighbour_search_recovers_every_order(seamless_pair):
    """Legacy suppression removes the valid block mask in every order; search keeps both blocks."""
    d = seamless_pair[1]
    a, b = PAIR
    named = {"A": _top_mask(a, d), "A_l": _top_mask(a, d, 0.0, 0.5), "A_r": _top_mask(a, d, 0.5), "B": _top_mask(b, d)}
    legacy, search = _cfg("greedy", True), _cfg("search")
    for order in itertools.permutations(named):
        masks = [named[k] for k in order]
        assert len(_run(seamless_pair, masks, legacy).poses) == 0
        obs = _run(seamless_pair, masks, search)
        errs = _centre_errors_mm(obs, PAIR)
        assert len(obs.poses) == 2 and max(errs) < 1.0, (order, errs)


def test_merged_duplicate_order_independent_with_search(seamless_pair):
    d = seamless_pair[1]
    a, b = (_top_mask(x, d) for x in PAIR)
    outcomes_raw, outcomes_search = set(), set()
    for masks in itertools.permutations([a, b, a | b]):
        outcomes_raw.add(len(_run(seamless_pair, list(masks), _cfg("greedy", False)).poses))
        outcomes_search.add(len(_run(seamless_pair, list(masks), _cfg("search")).poses))
    assert outcomes_raw == {0, 2}          # legacy greedy without suppression is order dependent
    assert outcomes_search == {2}


def test_partial_false_part_offset_pose_rejected():
    """Mask covering 85 % of block A (sliver missing): greedy accepts a ~5.7 mm offset pose; search does not."""
    images = render_synthetic_scene(CAM_W, T_W, CAM_S, T_S, PAIR, [], PAIR_ENDS, False)
    d = images[1]
    masks = [_top_mask(PAIR[0], d, 0.0, 0.85), _top_mask(PAIR[1], d)]
    greedy_errs = _centre_errors_mm(_run(images, masks, _cfg("greedy", True)), PAIR)
    search_errs = _centre_errors_mm(_run(images, masks, _cfg("search")), PAIR)
    assert max(greedy_errs) > 5.0
    assert len(search_errs) == 2 and max(search_errs) < 1.0


def test_missing_child_not_silently_omitted():
    """Masks for 2 of 3 flush blocks: greedy accepts 2 and drops the third without any ambiguity."""
    tri = [_blk(0.0, -0.025), _blk(0.0, 0.0), _blk(0.0, 0.025)]
    ends = [{"x": x, "y": y, "z0": 0.0, "z1": H} for x in (-0.0375, 0.0375) for y in (-0.0125, 0.0125)]
    images = render_synthetic_scene(CAM_W, T_W, CAM_S, T_S, tri, [], ends, False)
    masks = [_top_mask(tri[0], images[1]), _top_mask(tri[1], images[1])]
    legacy = _run(images, masks, _cfg("greedy", True))
    assert len(legacy.poses) == 2 and not legacy.ambiguous_clusters
    fixed = _run(images, masks, _cfg("search"))
    assert len(fixed.poses) == 3 or fixed.ambiguous_clusters
    assert max(_centre_errors_mm(fixed, tri)) < 1.0


@pytest.mark.parametrize("scene", ["stack", "bridge"])
def test_small_top_layer_not_dropped(scene):
    """A single raised block covers < 2 % of the ROI; the legacy all-cell percentile never opened its band."""
    if scene == "stack":
        blocks, top = [_blk(0.0, 0.0), _blk(0.0, 0.0, z0=H)], 1
    else:
        blocks, top = [_blk(0.0, -0.025), _blk(0.0, 0.025), _blk(0.0, 0.0, 90.0, z0=H)], 2
    images = render_synthetic_scene(CAM_W, T_W, CAM_S, T_S, blocks, None, None, False)
    for policy in ("merged_corroborated", "whole_partition_mv"):
        assert len(_run(images, None, _cfg(top_band=False), policy).poses) == 0
        obs = _run(images, None, _cfg(top_band=True), policy)
        assert len(obs.poses) == 1
        p = obs.poses[0]
        assert np.hypot(p.xy_m[0] - blocks[top]["cx"], p.xy_m[1] - blocks[top]["cy"]) < 0.002
        assert abs(p.z_top_m - blocks[top]["z1"]) < 0.002


# ------------------------------------------------------------------ partial occlusion and end stagger

_OCC = {"cx": 0.0, "cy": 0.04, "z0": 0.035, "z1": 0.065, "L": 0.18, "W": 0.066, "yaw_rad": 0.0, "color": (49, 62, 75)}
_PAIR_SEAM = [{"p0": (-0.037, 0.0), "p1": (0.037, 0.0), "z": H, "half_w_m": 0.0009}]


def _occ_cfg(on: bool):
    cfg = _cfg()
    cfg.enable_occlusion_truncated_partition = on
    return cfg


def test_partial_occlusion_fully_visible_block_recovered():
    """A raised occluder hides block B and leaves a 7 mm sliver flush with fully visible block A."""
    images = render_synthetic_scene(CAM_W, T_W, CAM_S, T_S, PAIR + [_OCC], _PAIR_SEAM, PAIR_ENDS, False)
    for policy in ("whole_partition_mv", "merged_corroborated"):
        assert len(_run(images, None, _occ_cfg(False), policy).poses) == 0
        obs = _run(images, None, _occ_cfg(True), policy)
        assert len(obs.poses) == 1, policy
        p = obs.poses[0]
        assert np.hypot(p.xy_m[0] - PAIR[0]["cx"], p.xy_m[1] - PAIR[0]["cy"]) < 0.001
        assert obs.pose_sources == ["split:occlusion_truncated"]
        assert "occluded_partial_remainder" in [a.reason for a in obs.ambiguous_clusters]


def test_truncated_neighbour_without_occluder_still_abstains():
    """Negative control: block + flush 8 mm non-block slab and NO occluder must not be carved."""
    blocks = [_blk(0.0, -0.0125), {"cx": 0.0, "cy": 0.004, "z0": 0.0, "z1": H, "L": 0.075, "W": 0.008, "yaw_rad": 0.0}]
    images = render_synthetic_scene(CAM_W, T_W, CAM_S, T_S, blocks, _PAIR_SEAM, PAIR_ENDS, False)
    for policy in ("whole_partition_mv", "merged_corroborated"):
        obs = _run(images, None, _occ_cfg(True), policy)
        assert len(obs.poses) == 0 and obs.ambiguous_clusters


@pytest.mark.parametrize("yaw_deg", [0.0, 30.0])
def test_three_mm_end_stagger_not_a_dead_zone(yaw_deg):
    a = np.radians(yaw_deg)
    u, v = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
    s = 0.003
    blocks = [_blk(*(-0.0125 * v + s * u), yaw_deg=yaw_deg), _blk(*(0.0125 * v), yaw_deg=yaw_deg)]
    seams = [{"p0": tuple((-0.037 + s) * u), "p1": tuple(0.037 * u), "z": H, "half_w_m": 0.0009}]
    ends = [{"x": float(p[0]), "y": float(p[1]), "z0": 0.0, "z1": H} for p in ((-0.0375 + s) * u, 0.0375 * u)]
    images = render_synthetic_scene(CAM_W, T_W, CAM_S, T_S, blocks, seams, ends, False)
    obs = _run(images, None, _cfg(), "merged_corroborated")
    errs = _centre_errors_mm(obs, blocks)
    assert len(obs.poses) == 2 and max(errs) < 1.0

"""Synthetic and integration checks for v2:  python3 -m pytest -q -p no:cacheprovider test_wp3_heightmap_v2.py
Covers the 18 Nov audit gaps: missing colour, empty input, 75 x 25 x 15 mm prints (side-by-side,
end-to-end, rotated, no-seam), a stripe on a single block, and the custom-bag CLI path."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import wp3_heightmap as hm

G = hm.Grid(-0.15, -0.15, 150, 150, 0.002)
PROPS = hm.Params()
JENGA = replace(hm.Params(), block_l_m=0.075, block_w_m=0.025, block_h_m=0.015)


def scene(blocks, h, seam_dark=0, stripe=None, noise=0.0004, seed=0, step=0.0012):
    """blocks: (cx, cy, L, W, yaw_deg). Each top gets a 1 mm dark border of strength seam_dark, so a
    joint between touching tops shows as a dark line; free edges are removed by the region erosion."""
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.arange(-0.15, 0.15, step), np.arange(-0.15, 0.15, step))
    z = np.zeros_like(xs); col = np.full(xs.shape + (3,), 210, np.int32)
    for cx, cy, L, W, yaw in blocks:
        c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
        u = (xs - cx) * c + (ys - cy) * s; v = -(xs - cx) * s + (ys - cy) * c
        m = (abs(u) <= L / 2) & (abs(v) <= W / 2)
        z[m] = h; col[m] = (180, 150, 110)
        edge = m & ((abs(u) > L / 2 - 0.001) | (abs(v) > W / 2 - 0.001))
        col[edge] -= seam_dark
        if stripe is not None and stripe(u, v, m).any():
            col[stripe(u, v, m)] -= 70
    pts = np.c_[xs.ravel(), ys.ravel(), z.ravel() + rng.normal(0, noise, xs.size)]
    return pts, np.clip(col, 0, 255).astype(np.uint8).reshape(-1, 3)


def run(blocks, p, h, **kw):
    H, C, V = hm.heightmap(*scene(blocks, h, **kw), G)
    return hm.detect(H, C, G, p, "H2", V=V)[0]


# ---------------- audit regressions
def test_missing_colour_columns_do_not_create_seams():
    pts, rgb = scene([(-0.045, 0, .045, .045, 0), (0, 0, .045, .045, 0), (0.045, 0, .045, .045, 0)], 0.038, noise=0)
    H, C, V = hm.heightmap(pts, rgb, G)
    for x in (-0.0225, 0.0225):                                   # the audit's probe: blank two cell columns
        j = int((x - G.x0) / G.cell); m = np.isfinite(H[:, j]) & (H[:, j] > 0.03)
        H[m, j] = np.nan; C[m, j] = 0; V[m, j] = False
    r = hm.detect(H, C, G, PROPS, "H2", V=V)[0]
    assert [x.status for x in r] == ["ambiguous_merged"] and hm.instances(r) == []


def test_empty_cloud_returns_nothing():
    H, C, V = hm.heightmap(np.empty((0, 3)), np.empty((0, 3), np.uint8), G)
    assert not V.any() and hm.detect(H, C, G, PROPS, "H2", V=V)[0] == []


# ---------------- props (as v1)
def test_props_flush_row_with_seams_splits_three():
    r = run([(-0.045, 0, .045, .045, 0), (0, 0, .045, .045, 0), (0.045, 0, .045, .045, 0)], PROPS, 0.038, seam_dark=80)
    assert [x.status for x in r] == ["split"] * 3


def test_props_flush_row_without_seams_stays_ambiguous():
    r = run([(-0.045, 0, .045, .045, 0), (0, 0, .045, .045, 0), (0.045, 0, .045, .045, 0)], PROPS, 0.038)
    assert [(x.status, x.n_est) for x in r] == [("ambiguous_merged", 3)]


# ---------------- printed Jenga 75 x 25 x 15 mm
def test_jenga_single_and_rotated_yaw():
    r = run([(0, 0, .075, .025, 30)], JENGA, 0.015)
    assert [x.status for x in r] == ["single"]
    assert min(abs(r[0].yaw_deg - 30) % 180, 180 - abs(r[0].yaw_deg - 30) % 180) < 3


def test_jenga_side_by_side_layer_splits_three_with_seams():
    r = run([(0, -0.025, .075, .025, 0), (0, 0, .075, .025, 0), (0, 0.025, .075, .025, 0)], JENGA, 0.015, seam_dark=80)
    assert [x.status for x in r] == ["split"] * 3
    assert sorted(round(x.xy_m[1], 3) for x in r) == pytest.approx([-0.025, 0.0, 0.025], abs=0.004)
    assert all(min(x.yaw_deg % 180, 180 - x.yaw_deg % 180) < 5 for x in r)    # long axis along x


def test_jenga_end_to_end_splits_two_with_seam():
    r = run([(-0.0375, 0, .075, .025, 0), (0.0375, 0, .075, .025, 0)], JENGA, 0.015, seam_dark=80)
    assert [x.status for x in r] == ["split"] * 2


def test_jenga_side_by_side_without_seams_stays_ambiguous():
    r = run([(0, -0.025, .075, .025, 0), (0, 0, .075, .025, 0), (0, 0.025, .075, .025, 0)], JENGA, 0.015)
    assert [(x.status, x.n_est) for x in r] == [("ambiguous_merged", 3)] and hm.instances(r) == []


def test_jenga_stripe_on_single_block_is_not_split():
    r = run([(0, 0, .075, .025, 0)], JENGA, 0.015, stripe=lambda u, v, m: m & (abs(u) < 0.0012))
    assert [x.status for x in r] == ["single"]


# ---------------- custom-bag CLI path (the v1 KeyError)
SUBSET = Path(__file__).resolve().parents[1] / "wrist_subsets" / "baseline_wrist.db3"


@pytest.mark.skipif(not SUBSET.exists(), reason="recorded subset not present")
def test_custom_bag_cli_runs(tmp_path):
    import evaluate_bag
    s = evaluate_bag.main([str(SUBSET), "--out", str(tmp_path), "--frame", "plane", "--max-frames", "5",
                           "--block-mm", "75", "25", "15"])
    assert s["frames_total"] == 5 and (tmp_path / "frames.json").exists()

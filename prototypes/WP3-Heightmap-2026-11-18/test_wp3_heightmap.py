"""Synthetic checks for wp3_heightmap (no recorded data needed): python3 -m pytest -q test_wp3_heightmap.py"""
import numpy as np
import wp3_heightmap as hm

P = hm.Params()
G = hm.Grid(-0.10, -0.10, 100, 100, 0.002)


def scene(blocks, seam_dark=0, gap_depth=0.0, noise=0.0005, seed=0):
    """blocks: list of (cx, cy) for 45 x 45 x 38 mm tops along x. Returns a dense point cloud + colours."""
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.arange(-0.1, 0.1, 0.0012), np.arange(-0.1, 0.1, 0.0012))
    z = np.zeros_like(xs); col = np.full(xs.shape + (3,), 200, np.uint8)
    for cx, cy in blocks:
        m = (abs(xs - cx) <= 0.0225) & (abs(ys - cy) <= 0.0225)
        z[m] = 0.038; col[m] = (190, 160, 120)
    for (a, _), (b, _) in zip(blocks[:-1], blocks[1:]):
        if abs((b - a) - 0.045) < 1e-6:                                   # flush joint
            j = abs(xs - (a + b) / 2) < 0.0012
            onrow = j & (abs(ys - blocks[0][1]) <= 0.0225)
            col[onrow] = np.clip(col[onrow].astype(int) - seam_dark, 0, 255)
            z[onrow] -= gap_depth
    pts = np.c_[xs.ravel(), ys.ravel(), z.ravel() + rng.normal(0, noise, xs.size)]
    return pts, col.reshape(-1, 3)


def run(blocks, **kw):
    H, C = hm.heightmap(*scene(blocks, **kw), G)
    return hm.detect(H, C, G, P, "H2")[0]


def test_single_block_is_single():
    r = run([(0.0, 0.0)])
    assert [x.status for x in r] == ["single"]
    assert abs(r[0].xy_m[0]) < 0.003 and abs(r[0].xy_m[1]) < 0.003


def test_separated_blocks_not_merged():
    assert [x.status for x in run([(-0.05, 0.0), (0.03, 0.0)])] == ["single", "single"]


def test_flush_row_without_evidence_stays_ambiguous():
    r = run([(-0.045, 0.0), (0.0, 0.0), (0.045, 0.0)])
    assert len(r) == 1 and r[0].status == "ambiguous_merged" and r[0].n_est == 3
    assert hm.instances(r) == []                  # nothing published as a measured block


def test_flush_row_rgb_seam_splits_into_three():
    r = run([(-0.045, 0.0), (0.0, 0.0), (0.045, 0.0)], seam_dark=80)
    assert [x.status for x in r] == ["split"] * 3 and {x.source for x in r} == {"rgb"}
    xs = sorted(x.xy_m[0] for x in r)
    assert np.allclose(xs, [-0.045, 0.0, 0.045], atol=0.004)


def test_flush_row_depth_gap_splits():
    r = run([(-0.0225, 0.0), (0.0225, 0.0)], gap_depth=0.006)
    assert [x.status for x in r] == ["split"] * 2 and r[0].source == "depth"


def test_texture_alone_does_not_split():
    pts, col = scene([(-0.045, 0.0), (0.0, 0.0), (0.045, 0.0)])
    rng = np.random.default_rng(3)
    col = np.clip(col.astype(int) - rng.integers(0, 40, col.shape[:1])[:, None], 0, 255).astype(np.uint8)
    H, C = hm.heightmap(pts, col, G)
    r = hm.detect(H, C, G, P, "H2")[0]
    assert [x.status for x in r] == ["ambiguous_merged"]

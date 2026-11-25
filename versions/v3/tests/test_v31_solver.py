"""V3.1 solver soundness and adversarial regressions (synthetic, deterministic seeds)."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))

from scenes import box, cameras, render  # noqa: E402
from wp3_live import ViewInput, preset_config, run_perception_epoch  # noqa: E402
from wp3_live import tiling as T  # noqa: E402
from wp3_live.dwell_gate import DwellStatus  # noqa: E402
from wp3_live.v3_regions import robot_links_have_hand, robot_unknown_cells  # noqa: E402

DW = DwellStatus(True, True, True, True, 0.0, 0.1, 0.2, 80.0, "ok")
CW, TW, CS, TS = cameras()


def _run(objs, seed=3, solver="prove", **cfg_kw):
    rng = np.random.default_rng(seed)
    rw, dw = render(CW, TW, objs, rng, noise_m=0.00045)
    rs, ds = render(CS, TS, objs, rng, noise_m=0.0008)
    cfg = preset_config("printed_jenga")
    cfg.enable_v3_regions = True
    cfg.v3_solver = solver
    cfg.v3_frame_budget_ms = cfg.v3_region_budget_ms = 1e9
    for k, v in cfg_kw.items():
        setattr(cfg, k, v)
    return run_perception_epoch(wrist=ViewInput("wrist", CW, rw, dw, TW, 1, 1), side=ViewInput("tripod", CS, rs, ds, TS, 1, 1),
                                config=cfg, policy="merged_corroborated", dwell=DW)


def _errs(res, objs):
    out = []
    for f in res.poses:
        d = [np.hypot(f.xy_m[0] - b["cx"], f.xy_m[1] - b["cy"]) for b in objs if b["kind"] == "block"]
        out.append(min(d) * 1000 if d else np.inf)
    return out


def test_same_is_symmetric_on_the_known_witness():
    m = object.__new__(T.RegionModel)
    m.p = T.TilingParams()
    q = T.Placement(0.0, 0.0, 0.0, 1.0)
    r = T.Placement(0.0079, -0.0036, float(np.radians(7.0)), 1.0)
    assert bool(m._same_in_frame_of(q, r)) != bool(m._same_in_frame_of(r, q))   # old predicate disagreed
    assert not m.same(q, r) and not m.same(r, q)


@pytest.mark.parametrize("solver", ["prove", "adaptive"])
def test_sound_solvers_resolve_L_layout(solver):
    objs = [box(0, 0, 0, 0, 0.015), box(0.05, 0.025, 90, 0, 0.015)]
    r = _run(objs, solver=solver)
    assert len(r.poses) == 2 and max(_errs(r, objs)) < 1.0


def test_adaptive_never_publishes_more_than_prove():
    scenes = [
        [box(0, -0.0125, 0, 0, 0.015, seam=True), box(0, 0.0125, 0, 0, 0.015, seam=True)],
        [box(0, y, 0, 0, 0.015) for y in (-0.025, 0.0, 0.025)],
        [box(-0.0375, 0, 0, 0, 0.015), box(0.0375, 0, 0, 0, 0.015)],
        [box(0, 0, 0, 0, 0.015), box(0.05, 0.025, 90, 0, 0.015)],
    ]
    for objs in scenes:
        a, p = _run(objs, solver="adaptive"), _run(objs, solver="prove")
        assert len(a.poses) <= len(p.poses)
        assert all(e < 5.0 for e in _errs(a, objs))


def test_striped_slab_is_not_split():
    """75 x 50 slab with periodic dark lines (one exactly where a seam would be): never two blocks."""
    slab = box(0, 0, 0, 0, 0.015, 0.075, 0.05, (190, 165, 125), kind="clutter",
               stripes_y=[-0.02, -0.01, 0.0, 0.01, 0.02])
    assert len(_run([slab]).poses) == 0


@pytest.mark.xfail(strict=True, reason=(
    "KNOWN, PRE-EXISTING (V2 front end, V3 on or off): the visible half of a 75 x 50 slab whose other half is "
    "under an occluder is published as a 'single' block. From the cameras it is identical to two flush blocks "
    "with one hidden (same output). Defence must be a declared lab-object set (no block-like clutter) in the "
    "FMEA, or side-view evidence under the occluder. Strict: flips to a failure if behaviour changes."))
def test_half_occluded_slab_is_not_published():
    slab = box(0, 0, 0, 0, 0.015, 0.075, 0.05, (190, 165, 125), kind="clutter")
    # covers exactly the slab's +y half (y in [0, 50] mm); 85 x 50 mm is not block-shaped itself
    occ = box(0, 0.025, 0, 0.045, 0.06, 0.085, 0.05, (60, 60, 70), kind="arm")
    assert len(_run([slab, occ]).poses) == 0


@pytest.mark.xfail(strict=True, reason=(
    "KNOWN, PRE-EXISTING (V2 'single' path): a block-sized object floating at a layer-multiple height (here "
    "85 x 28 mm, 45-60 mm up, no robot TF) is published as a stacked block. V3's unsupported_layer check only "
    "runs in the region stage. Mitigations: robot self-filter with TF (the arm), lab procedure for hands/tools."))
def test_floating_block_sized_occluder_not_published():
    occ = box(0, 0.0125, 0, 0.045, 0.06, 0.085, 0.028, (60, 60, 70), kind="arm")
    assert all(p.z_top_m < 0.03 for p in _run([occ]).poses)


def test_end_to_end_plank_is_not_split():
    """A 150 x 25 x 15 plank is withheld unless a seam cue exists (same rule as 75 x 50 slabs)."""
    plank = box(0, 0, 0, 0, 0.015, 0.150, 0.025, (190, 165, 125), kind="clutter")
    assert len(_run([plank]).poses) == 0


def test_partial_robot_tf_does_not_qualify_upper_layers():
    assert not robot_links_have_hand({"link_base": ([0, 0, 0], [0, 0, 0, 1]), "link1": ([0, 0, 0.1], [0, 0, 0, 1])})
    assert robot_links_have_hand({"link6": ([0.3, 0, 0.3], [0, 0, 0, 1])})


class _Grid:
    """Minimal grid stand-in: 2 mm cells over +-0.16 m."""
    cell_m, nx, ny, x0, y0 = 0.002, 160, 160, -0.16, -0.16

    def cell_xy(self, rr, cc):
        return self.x0 + (cc + 0.5) * self.cell_m, self.y0 + (rr + 0.5) * self.cell_m


def test_wrist_camera_housing_is_unknown_space():
    """The housing capsule (link6 -> 6.5 cm sideways, 7 cm along the flange axis) must mark cells the palm/finger
    capsules do not, matching what filter_robot_links removes from the point cloud."""
    # flange axis pointing down (180 deg about x), link6 low over the table so the housing reaches the ROI
    links = {"link6": ([0.0, 0.0, 0.10], [1.0, 0.0, 0.0, 0.0]), "link_tcp": ([0.0, 0.0, 0.02], [1.0, 0.0, 0.0, 0.0])}
    unk = robot_unknown_cells(_Grid(), links, z_max_m=0.30)
    g = _Grid()
    rr, cc = np.mgrid[0:g.ny, 0:g.nx]
    x, y = g.cell_xy(rr, cc)
    # beside the hand and outside the palm (55 mm) and finger (20 mm) capsules around the flange axis
    housing_only = (np.hypot(x + 0.065, y) < 0.03) & (np.hypot(x, y) > 0.06)
    assert unk[housing_only].any()

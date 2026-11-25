"""V3 region-stage regressions (synthetic scenes from bench/scenes.py; deterministic seeds)."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))

from scenes import cameras, family, render, box, FUZZ  # noqa: E402
from wp3_live import ViewInput, preset_config, run_perception_epoch  # noqa: E402
from wp3_live.dwell_gate import DwellStatus  # noqa: E402
from wp3_live import tiling as T  # noqa: E402

DW = DwellStatus(True, True, True, True, 0.0, 0.1, 0.2, 80.0, "ok")
CW, TW, CS, TS = cameras()


def _run(objs, seed=3, **cfg_kw):
    rng = np.random.default_rng(seed)
    rw, dw = render(CW, TW, objs, rng, noise_m=0.00045)
    rs, ds = render(CS, TS, objs, rng, noise_m=0.0008)
    cfg = preset_config("printed_jenga")
    for k, v in cfg_kw.items():
        setattr(cfg, k, v)
    return run_perception_epoch(wrist=ViewInput("wrist", CW, rw, dw, TW, 1, 1), side=ViewInput("tripod", CS, rs, ds, TS, 1, 1),
                                config=cfg, policy="merged_corroborated", dwell=DW)


def _errs(res, objs):
    out = []
    for f in res.poses:
        d = [np.hypot(f.xy_m[0] - b["cx"], f.xy_m[1] - b["cy"]) for b in objs if b["kind"] == "block"]
        out.append(min(d) * 1000)
    return out


def test_seamed_flush_pair_resolved_by_v3_not_by_v2():
    objs = [box(0, -0.0125, 20, 0, 0.015, seam=True), box(0, 0.0125, 20, 0, 0.015, seam=True)]
    objs = [dict(o, cx=o["cx"] * np.cos(np.radians(20)) - o["cy"] * np.sin(np.radians(20)),
                 cy=o["cx"] * np.sin(np.radians(20)) + o["cy"] * np.cos(np.radians(20))) for o in objs]
    assert len(_run(objs, enable_v3_regions=False).poses) == 0
    r = _run(objs)
    assert len(r.poses) == 2 and max(_errs(r, objs)) < 2.0


def test_block_shaped_slab_never_published():
    objs = [box(0, 0, 10, 0, 0.015, 0.075, 0.05, (150, 150, 160), kind="clutter")]
    assert len(_run(objs).poses) == 0


def test_slab_guard_sees_timed_out_neighbour(monkeypatch):
    """Regression: one block STABLE, its neighbour TIMEOUT -> guard must still withhold (found under CPU load)."""
    orig = T.RegionModel.solve_prove

    def half_timeout(self, shortlist, deadline):
        seed, verdicts, d = orig(self, shortlist, deadline)
        return seed, [verdicts[0]] + [T.TIMEOUT] * (len(verdicts) - 1), d

    monkeypatch.setattr(T.RegionModel, "solve_prove", half_timeout)
    objs = [box(0, 0, 0, 0, 0.015, 0.075, 0.05, (150, 150, 160), kind="clutter")]
    assert len(_run(objs).poses) == 0


def test_square_triple_stays_ambiguous():
    objs = [box(0, y, 0, 0, 0.015, seam=False) for y in (-0.025, 0.0, 0.025)]
    r = _run(objs)
    assert len(r.poses) == 0 and len(r.ambiguous_clusters) >= 1


def test_L_layout_resolved_sub_mm():
    objs = [box(0, 0, 0, 0, 0.015), box(0.05, 0.025, 90, 0, 0.015)]
    r = _run(objs)
    assert len(r.poses) == 2 and max(_errs(r, objs)) < 1.0


def test_mostly_hidden_block_not_published():
    objs = [box(0, -0.0125, 0, 0, 0.015, seam=True), box(0, 0.0125, 0, 0, 0.015, seam=True),
            box(-0.0375 + 0.85 * 0.0375, 0.025, 0, 0.045, 0.06, 0.85 * 0.075, 0.05, (60, 60, 70), kind="arm")]
    r = _run(objs)
    errs = _errs(r, objs)
    assert all(e < 5.0 for e in errs)
    assert len(r.poses) <= 1


def test_v3_off_is_tuned_v2():
    objs = [box(0, 0, 30, 0, 0.015)]
    a, b = _run(objs, enable_v3_regions=False), _run(objs)
    assert [p.xy_m for p in a.poses] == [p.xy_m for p in b.poses]


FAR = {"link6": ([1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]), "link_tcp": ([1.0, 1.0, 0.8], [0.0, 0.0, 0.0, 1.0])}


def test_floating_occluder_on_layer_multiple_not_published_as_stack():
    """Regression: a solid non-robot occluder (e.g. a hand) whose top lands on 7 x 15 mm over a flush pair, with
    robot TF present but parked elsewhere, must not be explained as stacked blocks (found by sched_replay)."""
    objs = [box(0, -0.0125, 15, 0, 0.015), box(0, 0.0125, 15, 0, 0.015),
            box(0.0, 0.0, 0, 0.045, 0.06, 0.07, 0.06, (60, 60, 70), kind="arm")]
    rng = np.random.default_rng(4)
    rw, dw = render(CW, TW, objs, rng, noise_m=0.00045)
    rs, ds = render(CS, TS, objs, rng, noise_m=0.0008)
    r = run_perception_epoch(wrist=ViewInput("wrist", CW, rw, dw, TW, 1, 1), side=ViewInput("tripod", CS, rs, ds, TS, 1, 1),
                             config=preset_config("printed_jenga"), policy="merged_corroborated", dwell=DW, robot_links=FAR)
    assert all(p.z_top_m < 0.03 for p in r.poses)

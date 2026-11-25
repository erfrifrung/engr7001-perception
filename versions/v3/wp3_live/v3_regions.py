"""V3 stage: explain components that the V2 path left unresolved, inside the real epoch call path.

Only components the V2 path could not turn into poses reach this stage, so common frames keep the V2 cost.
Regions are processed largest-first under a per-region and a per-frame wall-clock budget.  A skipped or
timed-out region keeps its ambiguity cluster (never a guess); other regions are unaffected.
"""
from __future__ import annotations

from dataclasses import replace
import time
from typing import Any

import cv2
import numpy as np

from .config import BlockSpec, RuntimeConfig
from .heightmap_core import HeightmapGrid, HeightmapResult
from .proposals import BlockFit
from .tiling import UNK, TilingParams, ViewDepth, build_tristate, explain_region


def robot_unknown_cells(grid: HeightmapGrid, robot_links: dict | None, *, tcp_offset_z_m: float = 0.242,
                        z_max_m: float = 0.12, link_radius_m: float = 0.072, finger_radius_m: float = 0.020) -> np.ndarray | None:
    """Cells under the projected robot capsules (same capsule model as ``filter_robot_links``), as UNKNOWN.

    Uses whatever link transforms the caller supplies: timestamp-matched TF in replay, or the live lookup.
    """
    if not robot_links:
        return None
    from scipy.spatial.transform import Rotation

    def pos(n: str):
        return np.asarray(robot_links[n][0], dtype=np.float64) if n in robot_links else None

    caps = []
    for a, b in [("link_base", "link1"), ("link1", "link2"), ("link2", "link3"), ("link3", "link4"),
                 ("link4", "link5"), ("link5", "link6")]:
        pa, pb = pos(a), pos(b)
        if pa is not None and pb is not None:
            caps.append((pa, pb, link_radius_m))
    p6 = pos("link6")
    if p6 is not None:
        R6 = Rotation.from_quat(robot_links["link6"][1]).as_matrix()
        ptcp = pos("link_tcp")
        ptcp = ptcp if ptcp is not None else p6 + R6 @ np.array([0.0, 0.0, tcp_offset_z_m])
        palm = p6 + (ptcp - p6) * 0.65
        caps += [(p6, palm, 0.055), (palm, ptcp, finger_radius_m)]
    if not caps:
        return None
    rr, cc = np.mgrid[0:grid.ny, 0:grid.nx]
    x, y = grid.cell_xy(rr, cc)
    out = np.zeros((grid.ny, grid.nx), dtype=bool)
    for a, b, r in caps:
        if min(a[2], b[2]) - r > z_max_m:
            continue
        ab = b[:2] - a[:2]
        L2 = float(ab @ ab)
        t = np.clip(((x - a[0]) * ab[0] + (y - a[1]) * ab[1]) / L2, 0.0, 1.0) if L2 > 1e-12 else np.zeros_like(x)
        d = np.hypot(x - (a[0] + t * ab[0]), y - (a[1] + t * ab[1]))
        out |= d <= r
    return out


def run_v3_regions(
    *,
    unresolved: list[tuple[dict, Any]],
    hm: HeightmapResult,
    grid: HeightmapGrid,
    block: BlockSpec,
    config: RuntimeConfig,
    views: list[ViewDepth],
    ambiguous: list,
    confident_with_src: list[tuple[BlockFit, str]],
    label_map: np.ndarray,
    robot_unknown: np.ndarray | None,
    make_cluster,
    t_frame0: float,
    cue_fn=None,
    region_cache=None,
) -> dict[str, Any]:
    """Mutates ``confident_with_src``, ``ambiguous`` and ``label_map``; returns a diagnostics dict."""
    stats: dict[str, Any] = {"regions": 0, "resolved_blocks": 0, "skipped_deadline": 0, "verdicts": {}, "ms": []}
    if not unresolved:
        return stats
    params = TilingParams(solver=config.v3_solver, region_budget_ms=config.v3_region_budget_ms,
                          max_nodes=config.v3_max_nodes)
    frame_deadline = t_frame0 + config.v3_frame_budget_ms / 1000.0
    h_max = hm.h_max
    # group unresolved components per band through nearby UNKNOWN cells
    by_band: dict[tuple[float, float], list[tuple[dict, Any]]] = {}
    for comp, amb in unresolved:
        by_band.setdefault(tuple(round(v, 5) for v in comp["band_m"]), []).append((comp, amb))
    k_join = int(np.ceil(0.04 / grid.cell_m)) * 2 + 1
    groups: list[list[tuple[dict, Any]]] = []
    for band, items in by_band.items():
        st = build_tristate(h_max, band, robot_unknown)
        anym = np.zeros(h_max.shape, bool)
        for comp, _ in items:
            anym |= comp["mask"]
        near = cv2.dilate(anym.astype(np.uint8), np.ones((k_join, k_join), np.uint8)).astype(bool)
        conn = anym | ((st == UNK) & near)
        n_lab, lab = cv2.connectedComponents(conn.astype(np.uint8), connectivity=8)
        bucket: dict[int, list] = {}
        for comp, amb in items:
            r, c = np.argwhere(comp["mask"])[0]
            bucket.setdefault(int(lab[r, c]), []).append((comp, amb))
        groups.extend(bucket.values())
    groups.sort(key=lambda g: -sum(int(c["area_cells"]) for c, _ in g))
    tf_ok = robot_unknown is not None or not config.enable_robot_self_filter
    for g in groups:
        ambs = [a for _, a in g if a is not None]
        if not tf_ok and g[0][0]["band_m"][0] > 1.2 * block.height_m:
            # no robot TF this tick: a robot link whose top lands on k * h is indistinguishable from a stack by
            # depth alone (measured: palm at 7 x 15 mm), so stacked layers keep the V2 outcome
            stats["skipped_no_robot_tf"] = stats.get("skipped_no_robot_tf", 0) + 1
            _keep_or_add(g, ambiguous, make_cluster, "v3_no_robot_tf_upper_layer", config)
            continue
        if time.perf_counter() > frame_deadline:
            stats["skipped_deadline"] += 1
            _keep_or_add(g, ambiguous, make_cluster, "v3_frame_deadline", config)
            continue
        stats["regions"] += 1
        res = explain_region(
            grid=grid, h_filled=hm.h_filled, h_max=h_max, comp_masks=[c["mask"] for c, _ in g],
            band=g[0][0]["band_m"], pts_tbl=hm.pts_tbl, block=block, params=params,
            extra_unknown=robot_unknown, views=views if config.v3_view_carving else None,
            deadline_s=frame_deadline, cue_fn=cue_fn, slab_guard=config.v3_slab_guard,
            region_cache=region_cache,
        )
        stats["ms"].append(res.diag.get("total_ms"))
        for v in res.verdicts:
            stats["verdicts"][v] = stats["verdicts"].get(v, 0) + 1
        if res.diag.get("cache_hit"):
            stats["cache_hits"] = stats.get("cache_hits", 0) + 1
        if not res.fits:
            if res.diag.get("slab_guard", {}).get("withheld"):
                for a in ambs:
                    a.reason = f"{a.reason}|v3_uncued_rectangle"
            _keep_or_add(g, ambiguous, make_cluster, f"v3_{res.reason}", config)
            continue
        for a in ambs:
            if a in ambiguous:
                ambiguous.remove(a)
        for f in res.fits:
            confident_with_src.append((f, f"split:v3_{config.v3_solver}"))
            stats["resolved_blocks"] += 1
            if len(f.points_xyz):
                ri, rj = grid.world_to_ij(f.points_xyz[:, :2])
                ok = (ri >= 0) & (ri < grid.ny) & (rj >= 0) & (rj < grid.nx)
                label_map[ri[ok], rj[ok]] = len(confident_with_src)
        for xy in res.unresolved_xy:
            ambiguous.append(make_cluster(xy, g[0][0], f"v3_{res.reason}"))
    return stats


def _keep_or_add(group, ambiguous, make_cluster, reason: str, config: RuntimeConfig) -> None:
    for comp, amb in group:
        if amb is not None:
            continue
        if config.v3_publish_unexplained:
            pts = comp["pts_tbl"]
            ambiguous.append(make_cluster((float(pts[:, 0].mean()), float(pts[:, 1].mean())), comp,
                                          "unexplained_occupancy" if reason.endswith("deadline") is False else reason))

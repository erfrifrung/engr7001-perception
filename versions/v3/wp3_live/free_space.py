"""V3.2 observed-free evidence: is a tracked block's footprint *seen* empty in this epoch's heightmap?

A block that is merely not detected (hidden under the arm, inside an ambiguous cluster, out of view) must stay a
collision obstacle. It may only be removed when its footprint is observed: enough cells carry real depth, and nearly
all of them are clearly below the block's top. Unobserved cells (no depth, robot self-filtered, outside the ROI) and
anything at or above the block's top count against freeness.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import BlockSpec
from .heightmap_core import HeightmapGrid


@dataclass(frozen=True)
class FreeSpaceMap:
    grid: HeightmapGrid
    h_max: np.ndarray            # (ny, nx) float32 table-frame top height, NaN = no observation
    stamp_ns: int
    edge_margin_m: float = 0.004
    min_observed_frac: float = 0.60
    min_free_frac: float = 0.90

    def footprint_free(self, xy_m: tuple[float, float], yaw_rad: float, z_top_m: float, block: BlockSpec) -> bool:
        """True only when the block's footprint (shrunk by edge_margin_m) is observed and clearly below its top."""
        if not np.isfinite(z_top_m):
            return False
        half_l = 0.5 * block.major_m - self.edge_margin_m
        half_w = 0.5 * block.minor_m - self.edge_margin_m
        if half_l <= 0.0 or half_w <= 0.0:
            return False
        step = self.grid.cell_m
        u = np.arange(-half_l, half_l + 1e-9, step)
        v = np.arange(-half_w, half_w + 1e-9, step)
        uu, vv = np.meshgrid(u, v)
        c, s = float(np.cos(yaw_rad)), float(np.sin(yaw_rad))
        xy = np.stack([xy_m[0] + c * uu.ravel() - s * vv.ravel(), xy_m[1] + s * uu.ravel() + c * vv.ravel()], axis=1)
        rows, cols = self.grid.world_to_ij(xy)
        inside = (rows >= 0) & (rows < self.grid.ny) & (cols >= 0) & (cols < self.grid.nx)
        n = len(xy)
        h = np.full(n, np.nan, dtype=np.float64)
        h[inside] = self.h_max[rows[inside], cols[inside]]
        observed = np.isfinite(h)
        n_obs = int(np.count_nonzero(observed))
        if n_obs < self.min_observed_frac * n:
            return False
        free = observed & (h < float(z_top_m) - 0.5 * float(block.height_m))
        return int(np.count_nonzero(free)) >= self.min_free_frac * n_obs

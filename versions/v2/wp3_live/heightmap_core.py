"""O(N) float32 orthographic tabletop heightmap with real-colour validity and multi-layer bands."""
from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np

from .fast_ops import heightmap_max_z_fast

_MORPH_CROSS_3X3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))


@dataclass(frozen=True)
class HeightmapGrid:
    x0: float
    y0: float
    nx: int
    ny: int
    cell_m: float

    @classmethod
    def from_roi(
        cls,
        roi_min_xy: tuple[float, float],
        roi_max_xy: tuple[float, float],
        cell_m: float = 0.002,
    ) -> "HeightmapGrid":
        nx = max(4, int(np.ceil((roi_max_xy[0] - roi_min_xy[0]) / cell_m)))
        ny = max(4, int(np.ceil((roi_max_xy[1] - roi_min_xy[1]) / cell_m)))
        return cls(x0=float(roi_min_xy[0]), y0=float(roi_min_xy[1]), nx=nx, ny=ny, cell_m=float(cell_m))

    def cell_xy(self, rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = self.x0 + (cols.astype(np.float64) + 0.5) * self.cell_m
        y = self.y0 + (rows.astype(np.float64) + 0.5) * self.cell_m
        return x, y

    def world_to_ij(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cols = np.floor((xy[:, 0] - self.x0) / self.cell_m).astype(np.int32)
        rows = np.floor((xy[:, 1] - self.y0) / self.cell_m).astype(np.int32)
        return rows, cols


@dataclass
class HeightmapResult:
    grid: HeightmapGrid
    h_max: np.ndarray        # (ny, nx) float32, NaN where empty
    h_filled: np.ndarray     # (ny, nx) float32 with strict >=3-neighbour pinhole fill
    color_rgb: np.ndarray    # (ny, nx, 3) uint8
    color_valid: np.ndarray  # (ny, nx) bool, True only where a real colour sample fell
    counts: np.ndarray       # (ny, nx) int32
    pts_cell_flat: np.ndarray
    pts_tbl: np.ndarray
    pts_rgb: np.ndarray


def build_heightmap_fast(
    pts_tbl: np.ndarray,
    rgb_pts: np.ndarray | None,
    grid: HeightmapGrid,
    zlim: tuple[float, float] = (-0.03, 0.30),
) -> HeightmapResult:
    """O(N) max-Z heightmap per XY cell + top colour without black-hole artifacts."""
    ny, nx = grid.ny, grid.nx
    h_max = np.empty((ny, nx), dtype=np.float32)
    counts = np.empty((ny, nx), dtype=np.int32)
    color_rgb = np.zeros((ny, nx, 3), dtype=np.uint8)
    color_valid = np.zeros((ny, nx), dtype=bool)

    if len(pts_tbl) == 0:
        h_nan = np.full((ny, nx), np.nan, dtype=np.float32)
        counts.fill(0)
        return HeightmapResult(
            grid, h_nan, h_nan.copy(), color_rgb, color_valid, counts,
            np.zeros(0, dtype=np.int32), pts_tbl, np.zeros((0, 3), dtype=np.uint8)
        )

    pts = np.asarray(pts_tbl, dtype=np.float32)
    rgb = np.asarray(rgb_pts, dtype=np.uint8) if rgb_pts is not None else np.zeros((len(pts), 3), dtype=np.uint8)
    z_col = pts[:, 2]
    fin_z = np.isfinite(pts).all(axis=1) & (z_col >= zlim[0]) & (z_col <= zlim[1])
    if not bool(fin_z.all()):
        pts = pts[fin_z]
        rgb = rgb[fin_z]

    j = np.floor((pts[:, 0] - grid.x0) / grid.cell_m).astype(np.int32)
    i = np.floor((pts[:, 1] - grid.y0) / grid.cell_m).astype(np.int32)
    inside = (i >= 0) & (i < ny) & (j >= 0) & (j < nx)
    if not bool(inside.all()):
        pts = pts[inside]
        rgb = rgb[inside]
        i = i[inside]
        j = j[inside]

    if len(pts) == 0:
        h_nan = np.full((ny, nx), np.nan, dtype=np.float32)
        counts.fill(0)
        return HeightmapResult(
            grid, h_nan, h_nan.copy(), color_rgb, color_valid, counts,
            np.zeros(0, dtype=np.int32), pts, rgb
        )

    flat = (i * nx + j).astype(np.int32)
    h_flat = h_max.reshape(-1)
    counts_flat = counts.reshape(-1)
    z = pts[:, 2]
    heightmap_max_z_fast(flat, z, h_flat, counts_flat)

    if rgb_pts is not None:
        is_top = z >= (h_flat[flat] - np.float32(1e-5))
        top_flat = flat[is_top]
        top_rgb = rgb[is_top]
        col_flat = color_rgb.reshape(-1, 3)
        val_flat = color_valid.reshape(-1)
        col_flat[top_flat] = top_rgb
        val_flat[top_flat] = True

    h_out = np.where(counts > 0, h_max, np.float32(np.nan))
    h_filled = fill_isolated_pinholes(h_out)
    return HeightmapResult(grid, h_out, h_filled, color_rgb, color_valid, counts, flat, pts, rgb)


def fill_isolated_pinholes(H: np.ndarray) -> np.ndarray:
    """Fill strictly interior 1-cell quantization pinholes (>= 3 cardinal valid neighbours)."""
    fin = np.isfinite(H)
    finite_u8 = fin.astype(np.uint8)
    pad_f = np.pad(finite_u8, 1, mode="constant")
    cardinal = pad_f[:-2, 1:-1] + pad_f[2:, 1:-1] + pad_f[1:-1, :-2] + pad_f[1:-1, 2:]
    Hf = np.where(fin, H, np.float32(-10.0)).astype(np.float32, copy=False)
    dil = cv2.dilate(Hf, _MORPH_CROSS_3X3)
    hole = (~fin) & (cardinal >= 3) & (dil > np.float32(-9.0))
    out = H.copy()
    out[hole] = dil[hole]
    return out


def extract_height_band_components(
    hm: HeightmapResult,
    block_h_m: float,
    band_lo_factor: float = 0.65,
    band_hi_factor: float = 1.45,
    min_area_cells: int = 8,
    multi_layer: bool = True,
) -> list[dict]:
    """Extract connected regions on layer 1 OR active top tower layer without erasing high layers."""
    H = hm.h_filled
    fin = np.isfinite(H)
    if not np.any(fin):
        return []
    z_vals = H[fin]
    bands = [(band_lo_factor * block_h_m, band_hi_factor * block_h_m)]
    if multi_layer:
        k98 = max(0, min(len(z_vals) - 1, int(round(0.98 * (len(z_vals) - 1)))))
        z_hi = float(np.partition(z_vals, k98)[k98])
        if z_hi > band_hi_factor * block_h_m:
            bands.append((max(band_hi_factor * block_h_m, z_hi - 0.45 * block_h_m), z_hi + 0.35 * block_h_m))

    regions: list[dict] = []
    seen_mask = np.zeros(H.shape, dtype=bool)
    z_all = hm.pts_tbl[:, 2]
    for lo_m, hi_m in bands:
        mask = fin & (~seen_mask) & (H >= lo_m) & (H <= hi_m)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, _MORPH_CROSS_3X3)
        n_comp, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        cell_lut = labels.reshape(-1)
        pt_cell_lbls = cell_lut[hm.pts_cell_flat]
        z_in_band = (z_all >= lo_m - 0.003) & (z_all <= hi_m + 0.003)
        for k in range(1, n_comp):
            area = int(stats[k, cv2.CC_STAT_AREA])
            if area < min_area_cells:
                continue
            comp_mask = labels == k
            seen_mask |= comp_mask
            pt_sel = (pt_cell_lbls == k) & z_in_band
            pts_k = hm.pts_tbl[pt_sel]
            if len(pts_k) < 12:
                continue
            regions.append({
                "mask": comp_mask,
                "pts_tbl": pts_k,
                "band_m": (lo_m, hi_m),
                "area_cells": area,
            })
    return regions

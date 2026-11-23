"""WP3 tabletop heightmap comparator (offline, recorded data only).

Implements, on identical RGB-D frames:
  P0  image-space height pilot (copied logic of WP3-One-Camera-2026-11-18/analyse_wrist_capture.py:
      fixed fractional crop, per-frame RANSAC plane, height band, 3x3 open, connected components)
  P1  the same pilot restricted to the pixels whose table-frame XY lies in the physical tracker ROI
  H1  orthographic tabletop heightmap (Song et al. 2026, Eq. 3: max z per XY cell) in the
      table frame, bounded by the same physical ROI, height band + open + connected components
  H2  H1 + seam evidence (local height range and RGB dark-line evidence inside a region) and a
      measured block-footprint prior. Regions that are too large and have no seam are published
      as ONE ambiguous region with an estimated count, not split speculatively.

Song et al. leave the XY cell size and the meaning of the morphology tuple {3, 2, 18} unstated;
the choices made here are listed in PARAMS and in the report. Nothing here talks to ROS or a robot.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import cv2
import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "WP3-Offline-Analysis-2026-11-12" / "tools"   # bagio, projection
sys.path.insert(0, str(TOOLS))


@dataclass
class Params:
    cell_m: float = 0.002            # Song: unstated. ~1.4-1.7 mm/pixel at 437-524 mm, so 2 mm is near native
    block_l_m: float = 0.045         # measured recorded props (BLOCK_DIMENSIONS.md), not the 47 mm config
    block_w_m: float = 0.045
    block_h_m: float = 0.038
    band_lo: float = 0.68            # same band factors as the pilot, so the band is not a variable
    band_hi: float = 1.40
    min_area_frac: float = 0.35      # region area / single-block footprint below which a region is dropped
    shape_tol: tuple = (0.6, 1.35)   # accepted min/max side ratio to the measured footprint for one block
    seam_step_m: float = 0.0025      # same 2.5 mm step as the existing seam tracker
    seam_blackhat: float = 22.0      # grey levels; RGB dark-line strength inside a region
    seam_erode_cells: int = 2        # ignore the outer boundary of a region when searching for seams
    split_min_frac: float = 0.40     # each split part must hold this fraction of a footprint


PARAMS = Params()

# ---------------------------------------------------------------- geometry

def ray_grid(info):
    h, w = info.height, info.width
    pix = np.indices((h, w))[::-1].transpose(1, 2, 0).astype(np.float32)
    K = np.asarray(info.k, float).reshape(3, 3)
    xy = cv2.undistortPoints(pix.reshape(-1, 1, 2), K, np.asarray(info.d, float)).reshape(h, w, 2)
    return np.concatenate((xy, np.ones_like(xy[:, :, :1])), axis=2)


def fit_plane(points, rng_triples, tol=0.004, min_support=0.55):
    """RANSAC plane, identical rule to analyse_wrist_capture.fit_plane. Returns (n, d) with n.p + d = height."""
    if len(points) < 120:
        return None
    a, b, c = points[rng_triples % len(points)].transpose(1, 0, 2)
    n = np.cross(b - a, c - a)
    L = np.linalg.norm(n, axis=1)
    ok = L > 1e-5
    n = n[ok] / L[ok, None]
    d = -np.einsum("ij,ij->i", n, a[ok])
    if not len(n):
        return None
    dist = np.abs(points @ n.T + d)
    best = int(np.argmax(np.sum(dist < tol, axis=0)))
    inl = points[dist[:, best] < tol]
    if len(inl) / len(points) < min_support:
        return None
    cen = inl.mean(axis=0)
    _, _, ax = np.linalg.svd(inl - cen, full_matrices=False)
    nn = ax[-1]
    if nn[2] > 0:
        nn = -nn
    dd = float(-cen @ nn)
    if np.mean(np.abs(points @ nn + dd) < tol) < min_support:
        return None
    return nn, dd


def plane_frame(n, d):
    """T_frame_cam for a table frame whose z is the plane normal (towards the camera), origin under
    the optical axis, x = camera x projected into the plane. Yaw/origin are arbitrary but metric."""
    z = n / np.linalg.norm(n)
    x = np.array([1.0, 0, 0]) - z * z[0]
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])               # rows: frame axes expressed in camera coordinates
    origin_cam = -d * z                    # foot of the perpendicular from the camera centre
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ origin_cam
    return T


# ---------------------------------------------------------------- heightmap

@dataclass
class Grid:
    x0: float
    y0: float
    nx: int
    ny: int
    cell: float

    def ij(self, xy):
        return ((xy[:, 1] - self.y0) / self.cell).astype(int), ((xy[:, 0] - self.x0) / self.cell).astype(int)

    def xy(self, i, j):
        return self.x0 + (np.asarray(j) + 0.5) * self.cell, self.y0 + (np.asarray(i) + 0.5) * self.cell


def heightmap(pts, rgb, grid: Grid, zlim=(-0.03, 0.12)):
    """Max z per cell (Song Eq. 3) and the colour of that top point. Empty cells are NaN."""
    keep = np.isfinite(pts).all(1) & (pts[:, 2] > zlim[0]) & (pts[:, 2] < zlim[1])
    pts, rgb = pts[keep], rgb[keep]
    i, j = grid.ij(pts[:, :2])
    inside = (i >= 0) & (i < grid.ny) & (j >= 0) & (j < grid.nx)
    i, j, z, col = i[inside], j[inside], pts[inside, 2], rgb[inside]
    flat = i * grid.nx + j
    order = np.lexsort((z, flat))          # by cell, then height; the last per cell is the max
    flat, z, col = flat[order], z[order], col[order]
    last = np.r_[flat[1:] != flat[:-1], True]
    H = np.full(grid.ny * grid.nx, np.nan, np.float32)
    C = np.zeros((grid.ny * grid.nx, 3), np.uint8)
    H[flat[last]] = z[last]
    C[flat[last]] = col[last]
    return H.reshape(grid.ny, grid.nx), C.reshape(grid.ny, grid.nx, 3)


def _fill(H):
    """Fill isolated empty cells from the 3x3 neighbourhood max (keeps large holes empty)."""
    Hf = np.where(np.isfinite(H), H, -1.0).astype(np.float32)
    dil = cv2.dilate(Hf, np.ones((3, 3), np.uint8))
    out = H.copy()
    hole = ~np.isfinite(H) & (dil > -1)
    out[hole] = dil[hole]
    return out


@dataclass
class Region:
    xy_m: tuple
    dims_mm: tuple
    yaw_deg: float
    top_mm: float
    area_ratio: float
    n_est: int
    status: str                 # single | split | ambiguous_merged | unexpected_shape | region (H1: no shape logic)
    source: str = ""
    mask_index: int = -1


def _shape(mask, grid, H):
    ii, jj = np.nonzero(mask)
    x, y = grid.xy(ii, jj)
    pts = np.c_[x, y].astype(np.float32)
    (cx, cy), (w, h), ang = cv2.minAreaRect(pts)
    # centroid from cell centres (more stable than the rect centre on ragged stereo edges)
    return (float(x.mean()), float(y.mean())), tuple(sorted((w * 1000, h * 1000))), float(ang), \
        float(np.median(H[mask]) * 1000)


def detect(H, C, grid: Grid, p: Params = PARAMS, variant="H2"):
    """Return (regions, label_image). variant H1 = band + components; H2 adds seam/shape logic."""
    Hf = _fill(H)
    lo, hi = p.band_lo * p.block_h_m, p.band_hi * p.block_h_m
    band = (np.nan_to_num(Hf, nan=-1) >= lo) & (np.nan_to_num(Hf, nan=-1) <= hi)
    band = cv2.morphologyEx(band.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab = cv2.connectedComponents(band)
    a1 = p.block_l_m * p.block_w_m / grid.cell ** 2           # footprint in cells
    regions, out_lab = [], np.zeros_like(lab)
    gray = cv2.cvtColor(C, cv2.COLOR_RGB2GRAY) if C is not None else None
    for k in range(1, n):
        m = lab == k
        area = m.sum()
        if area < p.min_area_frac * a1:
            continue
        xy, dims, yaw, top = _shape(m, grid, Hf)
        ratio = area / a1
        if variant == "H1":
            regions.append(Region(xy, dims, yaw, top, float(ratio), 1, "region"))
            out_lab[m] = len(regions)
            continue
        single_ok = (p.shape_tol[0] * p.block_w_m * 1000 <= dims[0] <= p.shape_tol[1] * p.block_w_m * 1000 and
                     p.shape_tol[0] * p.block_l_m * 1000 <= dims[1] <= p.shape_tol[1] * p.block_l_m * 1000)
        n_est = _count_estimate(dims, ratio, p)
        if single_ok and n_est == 1:
            regions.append(Region(xy, dims, yaw, top, float(ratio), 1, "single"))
            out_lab[m] = len(regions)
            continue
        # ---- seam search: a cut must run across the whole region (straight line), be supported by
        # depth (gap/step) or RGB (dark line) evidence, and leave parts that each match the measured footprint
        parts = _seam_split(m, Hf, gray, grid, p)
        if parts:
            for s_mask, src in parts:
                sxy, sd, syaw, stop = _shape(s_mask, grid, Hf)
                regions.append(Region(sxy, sd, syaw, stop, float(s_mask.sum() / a1), 1, "split", src))
                out_lab[s_mask] = len(regions)
        else:
            status = "ambiguous_merged" if n_est >= 2 else "unexpected_shape"
            regions.append(Region(xy, dims, yaw, top, float(ratio), n_est, status))
            out_lab[m] = len(regions)
    return regions, out_lab


def _count_estimate(dims, ratio, p):
    """How many measured footprints the region could hold (a prior, never a measurement)."""
    w, l = sorted((p.block_w_m * 1000, p.block_l_m * 1000))
    short, long_ = dims
    if p.shape_tol[0] * w <= short <= p.shape_tol[1] * w:          # one block wide: count along the row
        return max(1, int(round(long_ / l)))
    if p.shape_tol[0] * l <= short <= p.shape_tol[1] * l and w < l:  # a layer of side-by-side blocks
        return max(1, int(round(short / w)))
    return max(1, int(round(ratio / 0.85)))


def _profile_peaks(prof, valid, min_prominence, half_window=6):
    """Narrow line evidence: a local maximum that stands min_prominence above the median of its
    +/-half_window neighbourhood (removes slow texture/illumination trends along the region)."""
    peaks = []
    for k in range(1, len(prof) - 1):
        if not valid[k] or prof[k] < prof[k - 1] or prof[k] < prof[k + 1]:
            continue
        lo, hi = max(0, k - half_window), min(len(prof), k + half_window + 1)
        if prof[k] - np.median(prof[lo:hi]) >= min_prominence:
            peaks.append(k)
    return peaks


def _seam_split(m, Hf, gray, grid, p):
    ii, jj = np.nonzero(m)
    P = np.c_[jj, ii].astype(float) * grid.cell
    c = P.mean(0)
    _, _, V = np.linalg.svd(P - c, full_matrices=False)
    inner = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    Hm = np.where(m, Hf, np.nan)
    top = float(np.nanmedian(Hm))
    drop = np.clip(top - np.nan_to_num(Hm, nan=top), 0, None)            # depth gap evidence (m)
    if gray is not None:
        g = np.where(m, gray, int(np.median(gray[m]))).astype(np.uint8)
        bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))).astype(float)
    else:
        bh = np.zeros_like(drop)
    w, l = sorted((p.block_w_m, p.block_l_m))
    for axis in (0, 1):
        u = (P - c) @ V[axis]
        bins = np.round((u - u.min()) / grid.cell).astype(int)
        nb = bins.max() + 1
        inn = inner[ii, jj]
        cnt = np.bincount(bins[inn], minlength=nb).astype(float)
        valid = cnt >= 3
        with np.errstate(invalid="ignore", divide="ignore"):
            prof_c = np.bincount(bins[inn], bh[ii, jj][inn], nb) / cnt
            prof_d = np.bincount(bins[inn], drop[ii, jj][inn], nb) / cnt
        prof_c, prof_d = np.nan_to_num(prof_c), np.nan_to_num(prof_d)
        edge = int(round(0.4 * w / grid.cell))                         # no cut closer than 0.4 W to an end
        valid[:edge] = False; valid[-edge:] = False
        for src, prof, thr in (("depth", prof_d, p.seam_step_m * 0.5), ("rgb", prof_c, p.seam_blackhat * 0.5)):
            cuts = _profile_peaks(prof, valid, thr)
            if not cuts:
                continue
            # keep the strongest cuts that leave parts of a plausible footprint
            cuts = sorted(cuts, key=lambda k: -prof[k])
            chosen = []
            for k in cuts:
                trial = sorted(chosen + [k])
                edges_ = [0] + trial + [nb - 1]
                lens = np.diff(edges_) * grid.cell
                if all(lens >= p.shape_tol[0] * w):
                    chosen = trial
            if not chosen:
                continue
            edges_ = [-1] + chosen + [nb]
            parts, ok = [], True
            for a, b in zip(edges_[:-1], edges_[1:]):
                sel = (bins > a + 1) & (bins < b - 1) if a >= 0 else (bins < b - 1)
                if b == nb:
                    sel = bins > a + 1
                pm = np.zeros_like(m)
                pm[ii[sel], jj[sel]] = True
                if pm.sum() < p.split_min_frac * (w * l / grid.cell ** 2):
                    ok = False; break
                d = sorted(cv2.minAreaRect(np.c_[jj[sel], ii[sel]].astype(np.float32))[1])
                d = [x * grid.cell for x in d]
                if not (p.shape_tol[0] * w <= d[0] <= p.shape_tol[1] * w and d[1] <= p.shape_tol[1] * l):
                    ok = False; break
                parts.append((pm, src))
            if ok and len(parts) >= 2:
                return parts
    return None


def instances(regions):
    """Regions that are published as one measured block each."""
    return [r for r in regions if r.status in ("single", "split", "region")]

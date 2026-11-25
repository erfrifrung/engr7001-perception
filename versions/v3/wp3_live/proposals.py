"""Multi-hypothesis zero-width partition proposals, native-RGB seam engine, and generic scalar watershed."""
from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
from typing import Any

import cv2
import numpy as np

from .config import BlockSpec, PolicyThresholds
from .fast_ops import (
    bilinear_sample_fast,
    fast_median_1d,
    fast_nanmedian_3rows,
    fast_nanmedian_cols,
    fast_nanmedian_rows,
    fast_percentile_pair_1d,
    fast_percentiles_2col,
    fast_rolling_median_1d,
    meyer_watershed_2d_fast,
)

_CUT_OFFSETS_F32 = np.array([-0.007, -0.005, -0.003, -0.0015, 0.0, 0.0015, 0.003, 0.005, 0.007], dtype=np.float32)
_STATIONS_UNIT_F32 = np.linspace(0.0, 1.0, 21, dtype=np.float32)
_EVEN_UNIT_F32 = _STATIONS_UNIT_F32[::2]
_ODD_UNIT_F32 = _STATIONS_UNIT_F32[1::2]
_STRIPE_STATIONS_UNIT_F32 = np.linspace(0.0, 1.0, 15, dtype=np.float32)
_ZERO_TWO_PI_HALF = float(np.pi / 2.0)


@dataclass
class ImageProjectionContext:
    """Lazy colour/gray/HSV image + camera-to-table projection context (reused from heightmap_seam_pipeline)."""

    bgr_img: np.ndarray
    rgb_img: np.ndarray
    K: np.ndarray
    D: np.ndarray
    T_table_cam: np.ndarray
    T_cam_table: np.ndarray
    rvec_ct: np.ndarray
    tvec_ct: np.ndarray
    _gray: np.ndarray | None = field(default=None, repr=False)

    @property
    def gray_img(self) -> np.ndarray:
        if self._gray is None:
            self._gray = cv2.cvtColor(self.rgb_img, cv2.COLOR_RGB2GRAY)
        return self._gray

    @classmethod
    def from_rgb(
        cls,
        rgb_img: np.ndarray,
        K: np.ndarray,
        D: np.ndarray,
        T_table_cam: np.ndarray,
        inv_rvec_tvec: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> "ImageProjectionContext":
        rgb_u8 = np.ascontiguousarray(rgb_img, dtype=np.uint8)
        bgr_u8 = rgb_u8[:, :, ::-1]
        T_tc = np.asarray(T_table_cam, dtype=np.float64).reshape(4, 4)
        if inv_rvec_tvec is not None:
            T_ct, rvec_ct, tvec_ct = inv_rvec_tvec
        else:
            T_ct = np.linalg.inv(T_tc)
            rvec_ct, _ = cv2.Rodrigues(T_ct[:3, :3])
            tvec_ct = T_ct[:3, 3].reshape(3, 1)
        return cls(
            bgr_img=bgr_u8,
            rgb_img=rgb_u8,
            K=np.asarray(K, dtype=np.float64).reshape(3, 3),
            D=np.asarray(D, dtype=np.float64).ravel(),
            T_table_cam=T_tc,
            T_cam_table=T_ct,
            rvec_ct=rvec_ct,
            tvec_ct=tvec_ct,
        )


@dataclass
class BlockFit:
    xy_m: tuple[float, float]
    z_center_m: float
    z_top_m: float
    yaw_rad: float
    yaw_deg_mod: float
    dims_mm: tuple[float, float]   # (major, minor)
    fill: float
    size_err: float
    n_points: int
    points_xyz: np.ndarray = field(repr=False)
    cell_mask: np.ndarray | None = field(default=None, repr=False)
    covariance_xy_m2: np.ndarray | None = None


@dataclass
class CutEvidence:
    cut_val_m: float
    direction: tuple[float, float]
    perp_dir: tuple[float, float]
    z_top_m: float
    p_lo_m: float
    p_hi_m: float
    depth_step_mm: float
    face_step_mm: float
    depth_step_ok: bool
    depth_trough_mm: float
    depth_trough_ok: bool
    rgb_trough: float
    rgb_grad: float
    rgb_color_step: float
    along_cut_continuity: float
    shoulder_support_frac: float
    stripe_alias_peaks: int
    periodic_texture_alias: bool
    rgb_ok: bool
    best_offset_m: float
    side_status: str = "UNAVAILABLE"
    side_visibility_r: float = 0.0
    side_edge_score: float = 0.0
    side_depth_step_mm: float = 0.0
    motion_corroborated: bool = False


@dataclass
class PartitionHypothesis:
    axis_deg: float
    k_blocks: int
    cuts: list[CutEvidence]
    child_fits: list[BlockFit]
    score: float
    mean_rgb_continuity: float
    has_validated_depth_step: bool
    has_depth_trough: bool
    has_strong_rgb: bool
    has_side_corroboration: bool
    has_side_contradiction: bool
    has_periodic_alias: bool


def fold_yaw_rad(yaw_rad: float, ref_yaw_rad: float, square_symmetry: bool) -> float:
    if square_symmetry:
        return ref_yaw_rad + ((yaw_rad - ref_yaw_rad + np.pi / 4.0) % (np.pi / 2.0) - np.pi / 4.0)
    return ref_yaw_rad + ((yaw_rad - ref_yaw_rad + np.pi / 2.0) % np.pi - np.pi / 2.0)


def fit_cuboid_top(
    points_xyz: np.ndarray,
    block: BlockSpec,
    row_yaw_rad: float = 0.0,
    fill_cell_m: float = 0.005,
    extra_yaws_rad: tuple[float, ...] = (),
) -> tuple[BlockFit | None, dict[str, Any]]:
    """Constrained cuboid top fitter checking extent, fill, and symmetry (handles near-square shapes)."""
    if len(points_xyz) < 12:
        return None, {"reason": "too_few_points", "points": int(len(points_xyz))}
    xy_f32 = np.ascontiguousarray(points_xyz[:, :2], dtype=np.float32)
    mean_xy = xy_f32.mean(axis=0, dtype=np.float64)
    centered = xy_f32.astype(np.float64) - mean_xy
    cov = (centered.T @ centered) / max(1, len(xy_f32) - 1)
    if cov.shape != (2, 2) or not np.isfinite(cov).all():
        return None, {"reason": "degenerate_cov", "points": int(len(points_xyz))}
    _, vecs = np.linalg.eigh(cov)
    principal = float(np.arctan2(vecs[1, -1], vecs[0, -1]))
    nom = np.array([block.major_m, block.minor_m], dtype=np.float32)
    centered_f32 = centered.astype(np.float32)

    raw_cands = (principal, principal + _ZERO_TWO_PI_HALF, row_yaw_rad, row_yaw_rad + _ZERO_TWO_PI_HALF, *extra_yaws_rad)
    candidates: list[float] = []
    half_pi = _ZERO_TWO_PI_HALF
    for y in raw_cands:
        if not any(abs((y - y0 + 0.5 * half_pi) % half_pi - 0.5 * half_pi) < 1e-3 for y0 in candidates):
            candidates.append(float(y))

    best = None
    best_diag: dict[str, Any] | None = None
    inv_cell = np.float32(1.0 / fill_cell_m)
    for yaw in candidates:
        c, s = np.float32(np.cos(yaw)), np.float32(np.sin(yaw))
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        loc = centered_f32 @ rot
        lo, hi = fast_percentiles_2col(loc, 4.0, 96.0)
        span_raw = hi - lo
        if span_raw[1] > span_raw[0]:
            span_eval = span_raw[::-1] + np.float32(block.edge_loss_m)
            yaw_use = yaw + _ZERO_TWO_PI_HALF
            cols0 = max(1, int(np.ceil(float(span_raw[1]) / fill_cell_m)))
            cols1 = max(1, int(np.ceil(float(span_raw[0]) / fill_cell_m)))
            b0 = np.clip(((loc[:, 1] - lo[1]) * inv_cell).astype(np.int32), 0, cols0 - 1)
            b1 = np.clip(((loc[:, 0] - lo[0]) * inv_cell).astype(np.int32), 0, cols1 - 1)
        else:
            span_eval = span_raw + np.float32(block.edge_loss_m)
            yaw_use = yaw
            cols0 = max(1, int(np.ceil(float(span_raw[0]) / fill_cell_m)))
            cols1 = max(1, int(np.ceil(float(span_raw[1]) / fill_cell_m)))
            b0 = np.clip(((loc[:, 0] - lo[0]) * inv_cell).astype(np.int32), 0, cols0 - 1)
            b1 = np.clip(((loc[:, 1] - lo[1]) * inv_cell).astype(np.int32), 0, cols1 - 1)

        err = np.abs(span_eval - nom) / nom
        n_cells = cols0 * cols1
        flat_b = b0 * cols1 + b1
        if n_cells <= 8192:
            occ_bm = np.zeros(n_cells, dtype=np.bool_)
            occ_bm[flat_b] = True
            occ = int(np.count_nonzero(occ_bm))
        else:
            occ = int(np.unique(flat_b).size)
        fill = occ / max(1, n_cells)
        size_err = float(err.sum())
        diag = {
            "dims_mm": (round(1000.0 * float(span_eval[0]), 1), round(1000.0 * float(span_eval[1]), 1)),
            "fill": round(float(fill), 3),
            "size_err": size_err,
            "points": int(len(points_xyz)),
        }
        if best_diag is None or size_err < best_diag["size_err"]:
            best_diag = diag
        ok_size = bool(np.all(err <= block.size_tol))
        if ok_size and fill >= block.min_fill:
            cen_xy = mean_xy + rot.astype(np.float64) @ ((lo + hi).astype(np.float64) * 0.5)
            if best is None or size_err < best[0]:
                best = (size_err, cen_xy, yaw_use, diag, fill)

    if best is None:
        assert best_diag is not None
        best_diag["reason"] = (
            "sparse_top_surface"
            if best_diag["size_err"] <= 2.0 * block.size_tol and best_diag["fill"] < block.min_fill
            else "footprint_not_one_block"
        )
        return None, best_diag

    size_err, cen_xy, yaw_use, diag, fill = best
    yaw_out = fold_yaw_rad(yaw_use, row_yaw_rad, block.is_square)
    yaw_deg = float(np.degrees(yaw_out) % block.yaw_symmetry_deg)
    z_top = fast_median_1d(points_xyz[:, 2])
    z_center = z_top - block.height_m / 2.0
    fit = BlockFit(
        xy_m=(float(cen_xy[0]), float(cen_xy[1])),
        z_center_m=z_center,
        z_top_m=z_top,
        yaw_rad=float(yaw_out),
        yaw_deg_mod=yaw_deg,
        dims_mm=diag["dims_mm"],
        fill=float(fill),
        size_err=float(size_err),
        n_points=int(len(points_xyz)),
        points_xyz=points_xyz,
        covariance_xy_m2=cov / max(1, len(points_xyz)),
    )
    return fit, diag


def _bilinear_sample(img: np.ndarray, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear image sampling with strict bounds checking (batched cv2.remap + float32 gather)."""
    return bilinear_sample_fast(img, uv)


def _bilinear_sample_valid_depth(
    depth_m: np.ndarray,
    uv: np.ndarray,
    *,
    min_d: float = 0.15,
    max_d: float = 1.80,
    max_jump_m: float = 0.04,
    require_all_4: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Valid-only bilinear depth sampler: NEVER mixes invalid (0/NaN) returns or step discontinuities."""
    h, w = depth_m.shape[:2]
    x = uv[:, 0]
    y = uv[:, 1]
    inside = (x >= 0.0) & (x <= w - 1.001) & (y >= 0.0) & (y <= h - 1.001)
    out = np.full(len(uv), np.nan, dtype=np.float32)
    ok = np.zeros(len(uv), dtype=bool)
    if not np.any(inside):
        return out, ok

    idx = np.flatnonzero(inside)
    xi = x[idx]
    yi = y[idx]
    x0 = np.floor(xi).astype(np.int32)
    y0 = np.floor(yi).astype(np.int32)
    dx = (xi - x0).astype(np.float32)
    dy = (yi - y0).astype(np.float32)

    c00 = depth_m[y0, x0].astype(np.float32)
    c10 = depth_m[y0, x0 + 1].astype(np.float32)
    c01 = depth_m[y0 + 1, x0].astype(np.float32)
    c11 = depth_m[y0 + 1, x0 + 1].astype(np.float32)
    corners = np.stack((c00, c10, c01, c11), axis=1)
    weights = np.stack(((1.0 - dx) * (1.0 - dy), dx * (1.0 - dy), (1.0 - dx) * dy, dx * dy), axis=1)

    c_ok = np.isfinite(corners) & (corners >= min_d) & (corners <= max_d)
    w_sum = np.sum(np.where(c_ok, weights, 0.0), axis=1)
    big_val = 1e6
    c_min = np.min(np.where(c_ok, corners, big_val), axis=1)
    c_max = np.max(np.where(c_ok, corners, -big_val), axis=1)
    spread = c_max - c_min

    min_corners = 4 if require_all_4 else 1
    good_cnt = np.sum(c_ok, axis=1) >= min_corners
    good = good_cnt & (w_sum >= (0.99 if require_all_4 else 0.50)) & (spread >= 0.0) & (spread <= max_jump_m)
    if np.any(good):
        gi = idx[good]
        val_sum = np.sum(np.where(c_ok[good], corners[good] * weights[good], 0.0), axis=1)
        out[gi] = val_sum / w_sum[good]
        ok[gi] = True
    return out, ok


def _count_periodic_dark_peaks(
    points_xy: np.ndarray,
    z_top: float,
    d: np.ndarray,
    perp_d: np.ndarray,
    lo_a: float,
    hi_a: float,
    p_lo: float,
    p_hi: float,
    img_ctx: ImageProjectionContext,
    blackhat_thresh: float = 10.0,
) -> int:
    """Count parallel dark stripe peaks across the region axis (catches >=3-stripe periodic textures)."""
    n_along = max(15, int(round((hi_a - lo_a) / 0.0015)))
    along_vals = np.linspace(lo_a + 0.004, hi_a - 0.004, n_along, dtype=np.float32)
    p0 = float(p_lo + 0.004)
    p_span = float((p_hi - 0.004) - p0)
    stations = p0 + p_span * _STRIPE_STATIONS_UNIT_F32
    d32 = np.asarray(d, dtype=np.float32)
    pd32 = np.asarray(perp_d, dtype=np.float32)
    xy_a = along_vals[:, None, None] * d32[None, None, :]
    xy_s = stations[None, :, None] * pd32[None, None, :]
    n_st = len(stations)
    pts3 = np.empty((n_along, n_st, 3), dtype=np.float32)
    pts3[:, :, :2] = xy_a + xy_s
    pts3[:, :, 2] = np.float32(z_top)
    uv, _ = cv2.projectPoints(pts3.reshape(-1, 3), img_ctx.rvec_ct, img_ctx.tvec_ct, img_ctx.K, img_ctx.D)
    uv = uv.reshape(-1, 2)
    gvals, inside = _bilinear_sample(img_ctx.gray_img, uv)
    ggrid = np.where(inside, gvals, np.float32(np.nan)).reshape(n_along, n_st)
    valid_a = np.sum(np.isfinite(ggrid), axis=1) >= 5
    if np.count_nonzero(valid_a) < 9:
        return 0
    prof = fast_nanmedian_rows(ggrid)
    half = max(3, int(round(0.006 / ((hi_a - lo_a) / max(1, n_along)))))
    med_wings, _ = fast_rolling_median_1d(prof, half, min_count=4)
    local_min = np.zeros(n_along, dtype=bool)
    local_min[1:-1] = valid_a[1:-1] & (prof[1:-1] <= prof[:-2]) & (prof[1:-1] <= prof[2:])
    in_p = local_min & np.isfinite(med_wings) & ((med_wings - prof) >= np.float32(blackhat_thresh))
    peaks = 0
    last_k = -100
    min_sep = max(2, int(round(0.004 / ((hi_a - lo_a) / max(1, n_along)))))
    for k in np.flatnonzero(in_p):
        if k - last_k >= min_sep:
            peaks += 1
            last_k = int(k)
    return peaks


def _evaluate_depth_cues_only(
    heights: np.ndarray,
    dist: np.ndarray,
    th: PolicyThresholds,
) -> tuple[float, float, float, bool, bool]:
    """Compute depth step and depth trough cues along cut normal using fast_median_1d."""
    near_left = (dist < -th.flank_inner_m) & (dist >= -th.flank_outer_m)
    near_right = (dist > th.flank_inner_m) & (dist <= th.flank_outer_m)
    on_seam = np.abs(dist) <= 0.0022
    depth_step_mm = 0.0
    face_step_mm = 0.0
    depth_trough_mm = 0.0
    depth_step_ok = False
    depth_trough_ok = False
    if min(int(np.count_nonzero(near_left)), int(np.count_nonzero(near_right))) >= th.flank_min_pts:
        hl = fast_median_1d(heights[near_left])
        hr = fast_median_1d(heights[near_right])
        depth_step_m = abs(hl - hr)
        depth_step_mm = depth_step_m * 1000.0
        left_all = dist < 0.0
        right_all = ~left_all
        if np.any(left_all) and np.any(right_all):
            fl = fast_median_1d(heights[left_all])
            fr = fast_median_1d(heights[right_all])
            face_step_m = abs(fl - fr)
            face_step_mm = face_step_m * 1000.0
            same_sign = (hl - hr) * (fl - fr) > 0.0
            depth_step_ok = bool(
                same_sign and depth_step_m >= th.depth_step_min_m and face_step_m >= th.face_step_min_m
            )
        if np.count_nonzero(on_seam) >= 3:
            depth_trough_m = min(hl, hr) - fast_median_1d(heights[on_seam])
            depth_trough_mm = depth_trough_m * 1000.0
            depth_trough_ok = bool(depth_trough_m >= th.depth_trough_min_m)
    return depth_step_mm, face_step_mm, depth_trough_mm, depth_step_ok, depth_trough_ok


def _sample_rgb_grids_batch(
    cut_vals: list[float],
    d: np.ndarray,
    perp_d: np.ndarray,
    z_top: float,
    offsets: np.ndarray,
    stations: np.ndarray,
    img_ctx: ImageProjectionContext,
) -> tuple[np.ndarray, np.ndarray]:
    """One batched projectPoints + bilinear sample for M cuts x len(offsets) x len(stations)."""
    n_cuts = len(cut_vals)
    n_off = len(offsets)
    n_st = len(stations)
    d32 = np.asarray(d, dtype=np.float32)
    pd32 = np.asarray(perp_d, dtype=np.float32)
    cv_arr = np.asarray(cut_vals, dtype=np.float32)
    # Broadcast shape (M, n_off, n_st, 2): (cut_val + off)*d + st*perp_d
    along_pos = (cv_arr[:, None] + offsets[None, :])[:, :, None, None] * d32[None, None, None, :]
    perp_pos = stations[None, None, :, None] * pd32[None, None, None, :]
    pts3 = np.empty((n_cuts, n_off, n_st, 3), dtype=np.float32)
    pts3[:, :, :, :2] = along_pos + perp_pos
    pts3[:, :, :, 2] = np.float32(z_top)
    uv_all, _ = cv2.projectPoints(pts3.reshape(-1, 3), img_ctx.rvec_ct, img_ctx.tvec_ct, img_ctx.K, img_ctx.D)
    uv_all = uv_all.reshape(-1, 2)
    g_all, in_all = _bilinear_sample(img_ctx.gray_img, uv_all)
    bgr_all, _ = _bilinear_sample(img_ctx.bgr_img, uv_all)
    g_grids = np.where(in_all, g_all, np.float32(np.nan)).reshape(n_cuts, n_off, n_st)
    b_grids = np.where(in_all[:, None], bgr_all, np.float32(np.nan)).reshape(n_cuts, n_off, n_st, 3)
    return g_grids, b_grids


def _evaluate_cut_cues_with_rgb(
    points_xyz: np.ndarray,
    along: np.ndarray,
    perp: np.ndarray,
    cut_val: float,
    direction: np.ndarray,
    perp_dir: np.ndarray,
    img_ctx: ImageProjectionContext | None,
    th: PolicyThresholds,
    stripe_peaks_count: int = 0,
    precomputed_rgb: tuple[np.ndarray, np.ndarray, float, float, float, float, np.ndarray] | None = None,
    precomputed_depth: tuple[float, float, float, bool, bool] | None = None,
) -> CutEvidence:
    """Measure validated flank depth step, depth trough, and per-station native-RGB seam continuity."""
    heights = points_xyz[:, 2]
    dist = along - cut_val
    if precomputed_depth is not None:
        depth_step_mm, face_step_mm, depth_trough_mm, depth_step_ok, depth_trough_ok = precomputed_depth
    else:
        depth_step_mm, face_step_mm, depth_trough_mm, depth_step_ok, depth_trough_ok = _evaluate_depth_cues_only(
            heights, dist, th,
        )

    if precomputed_rgb is not None:
        gray_grid, bgr_grid, p_lo, p_hi, margin, z_top, offsets = precomputed_rgb
        s0 = float(p_lo + margin)
        s_span = float((p_hi - margin) - s0)
        stations = s0 + s_span * _STATIONS_UNIT_F32
    else:
        p_lo, p_hi = fast_percentile_pair_1d(perp, 6.0, 94.0)
        margin = min(0.003, max(0.0, (p_hi - p_lo) * 0.12))
        z_top = fast_median_1d(heights)
        offsets = _CUT_OFFSETS_F32
        s0 = float(p_lo + margin)
        s_span = float((p_hi - margin) - s0)
        stations = s0 + s_span * _STATIONS_UNIT_F32
        gray_grid = np.empty(0, dtype=np.float32)
        bgr_grid = np.empty(0, dtype=np.float32)

    rgb_trough = 0.0
    rgb_grad = 0.0
    rgb_color_step = 0.0
    along_continuity = 0.0
    shoulder_support = 0.0
    best_offset_m = 0.0
    rgb_ok = False

    if img_ctx is not None and len(points_xyz) >= 12 and (p_hi - p_lo) > 0.010:
        p_l = np.sort(perp[(dist < -0.002) & (dist >= -0.010)])
        p_r = np.sort(perp[(dist > 0.002) & (dist <= 0.010)])
        if len(p_l) > 0 and len(p_r) > 0:
            has_l = (np.searchsorted(p_l, stations + 0.0035, side="right") - np.searchsorted(p_l, stations - 0.0035, side="left")) > 0
            has_r = (np.searchsorted(p_r, stations + 0.0035, side="right") - np.searchsorted(p_r, stations - 0.0035, side="left")) > 0
            shoulder_support = float(np.mean(has_l & has_r))

        if precomputed_rgb is None:
            g_grids, b_grids = _sample_rgb_grids_batch([float(cut_val)], direction, perp_dir, z_top, offsets, stations, img_ctx)
            gray_grid = g_grids[0]
            bgr_grid = b_grids[0]

        valid_per_off = np.sum(np.isfinite(gray_grid), axis=1) >= 8
        if np.all(valid_per_off):
            prof = fast_nanmedian_rows(gray_grid)
            left_sh = fast_median_1d(prof[:3])
            right_sh = fast_median_1d(prof[-3:])
            seam_idx = 3 + int(np.argmin(prof[3:6]))
            seam_val = float(prof[seam_idx])
            best_offset_m = float(offsets[seam_idx])
            rgb_trough = min(left_sh, right_sh) - seam_val
            rgb_grad = max(abs(left_sh - seam_val), abs(right_sh - seam_val))
            bgr_left = fast_nanmedian_cols(bgr_grid[:3].reshape(-1, 3))
            bgr_right = fast_nanmedian_cols(bgr_grid[-3:].reshape(-1, 3))
            rgb_color_step = float(np.linalg.norm(bgr_left - bgr_right))

            st_left = fast_nanmedian_3rows(gray_grid[:3, :])
            st_right = fast_nanmedian_3rows(gray_grid[-3:, :])
            st_centre = gray_grid[seam_idx, :]
            st_contrast = np.maximum(np.minimum(st_left, st_right) - st_centre, np.abs(st_right - st_left))
            valid_st = np.isfinite(st_contrast)
            if np.count_nonzero(valid_st) >= 8:
                along_continuity = float(np.mean(st_contrast[valid_st] >= 8.0))
            rgb_ok = bool(
                (
                    (rgb_trough >= th.rgb_trough_min and rgb_grad >= th.rgb_grad_min)
                    or (rgb_color_step >= th.rgb_color_step_min)
                )
                and along_continuity >= th.rgb_continuity_min
                and shoulder_support >= th.shoulder_support_min
            )

    return CutEvidence(
        cut_val_m=float(cut_val),
        direction=(float(direction[0]), float(direction[1])),
        perp_dir=(float(perp_dir[0]), float(perp_dir[1])),
        z_top_m=z_top,
        p_lo_m=p_lo + margin,
        p_hi_m=p_hi - margin,
        depth_step_mm=round(depth_step_mm, 2),
        face_step_mm=round(face_step_mm, 2),
        depth_step_ok=depth_step_ok,
        depth_trough_mm=round(depth_trough_mm, 2),
        depth_trough_ok=depth_trough_ok,
        rgb_trough=round(rgb_trough, 2),
        rgb_grad=round(rgb_grad, 2),
        rgb_color_step=round(rgb_color_step, 2),
        along_cut_continuity=round(along_continuity, 3),
        shoulder_support_frac=round(shoulder_support, 3),
        stripe_alias_peaks=int(stripe_peaks_count),
        periodic_texture_alias=bool(stripe_peaks_count > th.stripe_max_peaks),
        rgb_ok=rgb_ok,
        best_offset_m=best_offset_m,
    )


def evaluate_cut_cues(
    points_xyz: np.ndarray,
    along: np.ndarray,
    perp: np.ndarray,
    cut_val: float,
    direction: np.ndarray,
    perp_dir: np.ndarray,
    img_ctx: ImageProjectionContext | None,
    th: PolicyThresholds,
    stripe_peaks_count: int = 0,
) -> CutEvidence:
    """Measure validated flank depth step, depth trough, and per-station native-RGB seam continuity."""
    return _evaluate_cut_cues_with_rgb(
        points_xyz, along, perp, cut_val, direction, perp_dir, img_ctx, th, stripe_peaks_count
    )


def _make_partition_candidate(
    axis_deg: float,
    k_blocks: int,
    cue_list: list[CutEvidence],
    child_fits: list[BlockFit],
    total_size_err: float,
    side_verifier: Any,
    cuts_m: list[float],
    block: BlockSpec,
) -> PartitionHypothesis:
    if side_verifier is not None:
        for idx_c, cue in enumerate(cue_list):
            side_ev = side_verifier.verify_cut(cue, cuts_m[idx_c], block.height_m)
            cue.side_status = side_ev.status
            cue.side_visibility_r = side_ev.visibility_ratio_r
            cue.side_edge_score = side_ev.end_face_edge_score
            cue.side_depth_step_mm = side_ev.end_face_depth_step_mm

    mean_cont = float(np.mean([c.along_cut_continuity for c in cue_list])) if cue_list else 0.0
    cue_bonus = sum(
        0.10 * min(c.depth_step_mm, 6.0) / 6.0
        + 0.08 * min(c.rgb_trough, 25.0) / 25.0 * c.along_cut_continuity
        + (0.12 if c.side_status == "CORROBORATED" else (-0.25 if c.side_status == "CONTRADICTED" else 0.0))
        for c in cue_list
    )
    score = cue_bonus - total_size_err
    return PartitionHypothesis(
        axis_deg=round(axis_deg, 1),
        k_blocks=k_blocks,
        cuts=cue_list,
        child_fits=child_fits,
        score=float(score),
        mean_rgb_continuity=mean_cont,
        has_validated_depth_step=all(c.depth_step_ok for c in cue_list),
        has_depth_trough=all(c.depth_trough_ok for c in cue_list),
        has_strong_rgb=all(c.rgb_ok for c in cue_list),
        has_side_corroboration=all((c.side_status == "CORROBORATED" or c.motion_corroborated or c.depth_step_ok) for c in cue_list),
        has_side_contradiction=any(c.side_status == "CONTRADICTED" for c in cue_list),
        has_periodic_alias=any(c.periodic_texture_alias for c in cue_list),
    )


def _coarse_rgb_cut_plausible(
    gc: np.ndarray,
    bc: np.ndarray,
    th: PolicyThresholds,
) -> bool:
    """Fast coarse-station plausibility gate (conservatively scaled to never reject native hits)."""
    if not bool(np.all(np.sum(np.isfinite(gc), axis=1) >= 4)):
        return False
    prof_c = fast_nanmedian_rows(gc)
    l_c = fast_median_1d(prof_c[:3])
    r_c = fast_median_1d(prof_c[-3:])
    s_idx = 3 + int(np.argmin(prof_c[3:6]))
    s_val = float(prof_c[s_idx])
    trough_c = min(l_c, r_c) - s_val
    grad_c = max(abs(l_c - s_val), abs(r_c - s_val))
    bl_c = fast_nanmedian_cols(bc[:3].reshape(-1, 3))
    br_c = fast_nanmedian_cols(bc[-3:].reshape(-1, 3))
    cstep_c = float(np.linalg.norm(bl_c - br_c))
    if not ((trough_c >= 0.40 * th.rgb_trough_min and grad_c >= 0.40 * th.rgb_grad_min) or cstep_c >= 0.50 * th.rgb_color_step_min):
        return False
    stl_c = fast_nanmedian_3rows(gc[:3, :])
    str_c = fast_nanmedian_3rows(gc[-3:, :])
    stc_c = np.maximum(np.minimum(stl_c, str_c) - gc[s_idx, :], np.abs(str_c - stl_c))
    vst_c = np.isfinite(stc_c)
    if np.count_nonzero(vst_c) < 4:
        return False
    return float(np.mean(stc_c[vst_c] >= 5.0)) >= 0.25


def evaluate_warm_start_hypothesis(
    pts_xyz: np.ndarray,
    img_ctx: ImageProjectionContext | None,
    block: BlockSpec,
    th: PolicyThresholds,
    *,
    direction: tuple[float, float],
    perp_dir: tuple[float, float],
    axis_deg: float,
    k_blocks: int,
    refined_cuts_m: tuple[float, ...],
    side_verifier: Any = None,
) -> PartitionHypothesis | None:
    """Verify last frame's winning cuts first at native resolution before running full search."""
    if len(pts_xyz) < 20 or len(refined_cuts_m) != (k_blocks - 1):
        return None
    xy = np.ascontiguousarray(pts_xyz[:, :2], dtype=np.float32)
    d32 = np.asarray(direction, dtype=np.float32)
    pd32 = np.asarray(perp_dir, dtype=np.float32)
    along = xy @ d32
    perp = xy @ pd32
    lo_a, hi_a = fast_percentile_pair_1d(along, 4.0, 96.0)
    p_lo, p_hi = fast_percentile_pair_1d(perp, 6.0, 94.0)
    if any(c <= lo_a + 0.012 or c >= hi_a - 0.012 for c in refined_cuts_m):
        return None
    if list(refined_cuts_m) != sorted(refined_cuts_m):
        return None

    z_top = fast_median_1d(pts_xyz[:, 2])
    margin = min(0.003, max(0.0, (p_hi - p_lo) * 0.12))
    offsets = _CUT_OFFSETS_F32
    s0 = float(p_lo + margin)
    s_span = float((p_hi - margin) - s0)
    stations = s0 + s_span * _STATIONS_UNIT_F32

    stripe_peaks = 0
    pre_rgb: dict[int, tuple] = {}
    if img_ctx is not None:
        stripe_peaks = _count_periodic_dark_peaks(xy, z_top, d32, pd32, lo_a, hi_a, p_lo, p_hi, img_ctx)
        if (p_hi - p_lo) > 0.010:
            g_grids, b_grids = _sample_rgb_grids_batch(list(refined_cuts_m), d32, pd32, z_top, offsets, stations, img_ctx)
            for i_c in range(len(refined_cuts_m)):
                pre_rgb[i_c] = (g_grids[i_c], b_grids[i_c], p_lo, p_hi, margin, z_top, offsets)

    cue_list: list[CutEvidence] = []
    for i_c, cv_m in enumerate(refined_cuts_m):
        cue = _evaluate_cut_cues_with_rgb(
            pts_xyz, along, perp, float(cv_m), d32, pd32, img_ctx, th, stripe_peaks, precomputed_rgb=pre_rgb.get(i_c),
        )
        if not (cue.depth_step_ok or cue.depth_trough_ok or cue.rgb_ok):
            return None
        cue_list.append(cue)

    bounds = [-np.inf, *refined_cuts_m, np.inf]
    child_fits: list[BlockFit] = []
    total_size_err = 0.0
    row_yaw = float(np.radians(axis_deg))
    for i in range(k_blocks):
        sub_mask = (along >= bounds[i]) & (along < bounds[i + 1])
        fit_i, _ = fit_cuboid_top(pts_xyz[sub_mask], block, row_yaw_rad=row_yaw)
        if fit_i is None:
            return None
        child_fits.append(fit_i)
        total_size_err += fit_i.size_err

    return _make_partition_candidate(
        axis_deg, k_blocks, cue_list, child_fits, total_size_err, side_verifier, list(refined_cuts_m), block,
    )


def enumerate_partition_hypotheses(
    pts_xyz: np.ndarray,
    img_ctx: ImageProjectionContext | None,
    block: BlockSpec,
    th: PolicyThresholds,
    row_yaw_rad: float = 0.0,
    side_verifier: Any = None,
) -> list[PartitionHypothesis]:
    """Generate and score zero-width K=2 and K=3 partition hypotheses with coarse-to-fine batched RGB."""
    xy = np.ascontiguousarray(pts_xyz[:, :2], dtype=np.float32)
    if len(xy) < 20:
        return []
    c_xy = xy.mean(axis=0, dtype=np.float64)
    centered = xy.astype(np.float64) - c_xy
    cov = (centered.T @ centered) / max(1, len(xy) - 1)
    _, vecs = np.linalg.eigh(cov)
    rect = cv2.minAreaRect(xy)
    box = cv2.boxPoints(rect)
    # Edge-aligned minAreaRect directions take precedence in the 5 deg de-duplication below: a small end
    # stagger tilts the PCA axis by ~4 deg, which previously evicted the exact edge direction (3 mm dead zone).
    if getattr(th, "edge_first_directions", True):
        raw_dirs = [box[1] - box[0], box[2] - box[1], vecs[:, -1], vecs[:, 0]]
    else:  # exact V2 order (PCA first)
        raw_dirs = [vecs[:, -1], vecs[:, 0], box[1] - box[0], box[2] - box[1]]
    if row_yaw_rad is not None:
        raw_dirs.extend([
            np.array([np.cos(row_yaw_rad), np.sin(row_yaw_rad)]),
            np.array([-np.sin(row_yaw_rad), np.cos(row_yaw_rad)]),
        ])

    directions: list[tuple[np.ndarray, float]] = []
    for v in raw_dirs:
        nv = float(np.linalg.norm(v))
        if nv < 1e-6:
            continue
        u = (v / nv).astype(np.float32)
        ang = float(np.degrees(np.arctan2(float(u[1]), float(u[0]))) % 180.0)
        if not any(min(abs(ang - a0), 180.0 - abs(ang - a0)) < 5.0 for _, a0 in directions):
            directions.append((u, ang))

    unit_sizes = sorted({block.major_m, block.minor_m})
    hypotheses: list[PartitionHypothesis] = []
    heights = pts_xyz[:, 2]
    z_top = fast_median_1d(heights)
    offsets = _CUT_OFFSETS_F32

    for d, axis_deg in directions:
        perp_d = np.array([-d[1], d[0]], dtype=np.float32)
        along = xy @ d
        perp = xy @ perp_d
        lo_a, hi_a = fast_percentile_pair_1d(along, 4.0, 96.0)
        p_lo, p_hi = fast_percentile_pair_1d(perp, 6.0, 94.0)
        span = (hi_a - lo_a) + block.edge_loss_m

        valid_ks = [
            k_b for k_b in (2, 3)
            if any(abs(span - k_b * u) <= block.size_tol * (k_b * u) for u in unit_sizes)
        ]
        if not valid_ks:
            continue

        fracs_2 = (0.42, 0.46, 0.50, 0.54, 0.58)
        fracs_3_lo = (0.31, 0.333, 0.36)
        fracs_3_hi = (0.64, 0.667, 0.69)
        unique_fracs = sorted(set((fracs_2 if 2 in valid_ks else ()) + (fracs_3_lo + fracs_3_hi if 3 in valid_ks else ())))
        unique_vals = [lo_a + f * (hi_a - lo_a) for f in unique_fracs]
        frac_to_idx = {f: idx for idx, f in enumerate(unique_fracs)}

        stripe_peaks = 0
        margin = min(0.003, max(0.0, (p_hi - p_lo) * 0.12))
        s0 = float(p_lo + margin)
        s_span = float((p_hi - margin) - s0)

        # Evaluate cheap depth cues first
        depth_by_idx: dict[int, tuple[float, float, float, bool, bool]] = {
            idx_c: _evaluate_depth_cues_only(heights, along - cv_m, th)
            for idx_c, cv_m in enumerate(unique_vals)
        }

        # Coarse-to-fine batched RGB sampling across candidate cuts:
        # Stage A samples 11 even stations; Stage B fills 10 odd stations ONLY for surviving cuts!
        pre_rgb_by_idx: dict[int, tuple] = {}
        active_idxs: list[int] = list(range(len(unique_vals)))
        if img_ctx is not None:
            stripe_peaks = _count_periodic_dark_peaks(xy, z_top, d, perp_d, lo_a, hi_a, p_lo, p_hi, img_ctx)
            if (p_hi - p_lo) > 0.010 and unique_vals:
                st_even = s0 + s_span * _EVEN_UNIT_F32
                gc_all, bc_all = _sample_rgb_grids_batch(unique_vals, d, perp_d, z_top, offsets, st_even, img_ctx)
                keep_idxs = []
                for idx_c in range(len(unique_vals)):
                    d_ok = depth_by_idx[idx_c][3] or depth_by_idx[idx_c][4]
                    if d_ok or _coarse_rgb_cut_plausible(gc_all[idx_c], bc_all[idx_c], th):
                        keep_idxs.append(idx_c)
                active_idxs = keep_idxs
                if keep_idxs:
                    st_odd = s0 + s_span * _ODD_UNIT_F32
                    go_all, bo_all = _sample_rgb_grids_batch(
                        [unique_vals[i] for i in keep_idxs], d, perp_d, z_top, offsets, st_odd, img_ctx,
                    )
                    n_off = len(offsets)
                    for pos_k, idx_c in enumerate(keep_idxs):
                        g_grid = np.empty((n_off, 21), dtype=np.float32)
                        b_grid = np.empty((n_off, 21, 3), dtype=np.float32)
                        g_grid[:, 0::2] = gc_all[idx_c]
                        g_grid[:, 1::2] = go_all[pos_k]
                        b_grid[:, 0::2] = bc_all[idx_c]
                        b_grid[:, 1::2] = bo_all[pos_k]
                        pre_rgb_by_idx[idx_c] = (g_grid, b_grid, p_lo, p_hi, margin, z_top, offsets)

        # Score surviving unique cuts at full native resolution and cache
        cut_cache: dict[int, tuple[CutEvidence, float] | None] = {idx_c: None for idx_c in range(len(unique_vals))}
        for idx_c in active_idxs:
            cv_m = unique_vals[idx_c]
            cue = _evaluate_cut_cues_with_rgb(
                pts_xyz, along, perp, cv_m, d, perp_d, img_ctx, th, stripe_peaks,
                precomputed_rgb=pre_rgb_by_idx.get(idx_c),
                precomputed_depth=depth_by_idx[idx_c],
            )
            if cue.depth_step_ok or cue.depth_trough_ok or cue.rgb_ok:
                ref_c = cv_m + (cue.best_offset_m if cue.rgb_ok else 0.0)
                cut_cache[idx_c] = (cue, ref_c)

        for k_blocks in valid_ks:
            if k_blocks == 2:
                tuple_idxs = [(frac_to_idx[f],) for f in fracs_2 if cut_cache.get(frac_to_idx[f]) is not None]
            else:
                valid_lo = [frac_to_idx[f1] for f1 in fracs_3_lo if cut_cache.get(frac_to_idx[f1]) is not None]
                valid_hi = [frac_to_idx[f2] for f2 in fracs_3_hi if cut_cache.get(frac_to_idx[f2]) is not None]
                tuple_idxs = [(i1, i2) for i1 in valid_lo for i2 in valid_hi]
            if not tuple_idxs:
                continue

            # Deduplicate snapped refined-cut tuples and pre-rank top 4 before cuboid fitting
            seen_ref_keys: set[tuple[int, ...]] = set()
            ranked_tuples: list[tuple[float, list[CutEvidence], list[float]]] = []
            for ti in tuple_idxs:
                cues = [cut_cache[i][0] for i in ti]  # type: ignore[index]
                refs = [cut_cache[i][1] for i in ti]  # type: ignore[index]
                rkey = tuple(int(round(r / 0.0015)) for r in refs)
                if rkey in seen_ref_keys:
                    continue
                seen_ref_keys.add(rkey)
                pre_q = sum(
                    (0.20 if c.depth_step_ok else 0.0)
                    + (0.05 if c.depth_trough_ok else 0.0)
                    + 0.08 * min(c.rgb_trough, 25.0) / 25.0 * c.along_cut_continuity
                    for c in cues
                )
                ranked_tuples.append((pre_q, cues, refs))
            ranked_tuples.sort(key=lambda x: -x[0])

            best_overall: PartitionHypothesis | None = None
            best_strict: PartitionHypothesis | None = None
            row_yaw_axis = float(np.radians(axis_deg))
            for _, tpl_cues, refined_cuts in ranked_tuples[:4]:
                # Clone CutEvidence instances so side_status writes stay isolated per tuple
                cue_list = [CutEvidence(**vars(c)) for c in tpl_cues]
                bounds = [-np.inf, *refined_cuts, np.inf]
                child_fits: list[BlockFit] = []
                total_size_err = 0.0
                valid_split = True
                for i in range(k_blocks):
                    sub_mask = (along >= bounds[i]) & (along < bounds[i + 1])
                    fit_i, _ = fit_cuboid_top(
                        pts_xyz[sub_mask], block, row_yaw_rad=row_yaw_axis,
                    )
                    if fit_i is None:
                        valid_split = False
                        break
                    child_fits.append(fit_i)
                    total_size_err += fit_i.size_err
                if not valid_split:
                    continue

                cand = _make_partition_candidate(
                    axis_deg, k_blocks, cue_list, child_fits, total_size_err, side_verifier, refined_cuts, block,
                )
                if best_overall is None or cand.score > best_overall.score:
                    best_overall = cand
                is_strict = cand.has_validated_depth_step or (
                    cand.has_strong_rgb and cand.has_side_corroboration and not cand.has_side_contradiction and not cand.has_periodic_alias
                )
                if is_strict and (best_strict is None or cand.score > best_strict.score):
                    best_strict = cand

            if best_overall is not None:
                hypotheses.append(best_overall)
            if best_strict is not None and best_strict is not best_overall:
                hypotheses.append(best_strict)

    hypotheses.sort(key=lambda h: -h.score)
    return hypotheses


def suppress_merged_masks(masks: list[np.ndarray]) -> list[np.ndarray]:
    """Drop masks explained by two or more nearly disjoint instance masks.

    Accept equally shaped boolean masks in image pixels or in projected depth
    samples. Keep the original order of surviving masks.
    """
    areas = [int(np.count_nonzero(mask)) for mask in masks]
    largest_first = sorted(range(len(masks)), key=areas.__getitem__, reverse=True)
    suppressed: set[int] = set()
    for i in largest_first:
        if areas[i] == 0:
            continue
        covered = np.zeros_like(masks[i], dtype=bool)
        children = 0
        for j in largest_first:
            if not 0.15 * areas[i] <= areas[j] <= 0.8 * areas[i]:
                continue
            inside = masks[i] & masks[j]
            if np.count_nonzero(inside) < 0.9 * areas[j]:
                continue
            if np.count_nonzero(covered & inside) > 0.15 * areas[j]:
                continue
            covered |= inside
            children += 1
        if children >= 2 and np.count_nonzero(covered) >= 0.8 * areas[i]:
            suppressed.add(i)
    return [mask for i, mask in enumerate(masks) if i not in suppressed]


MASK_GROUPING_MODES = ("greedy", "search")


def _near_points(cells: np.ndarray, sel: np.ndarray, halo_cells: int) -> np.ndarray:
    """Points whose 1 mm XY cell lies within ``halo_cells`` of any selected point's cell."""
    near = np.zeros(len(cells), dtype=bool)
    if not np.any(sel):
        return near
    shape = (int(cells[:, 1].max()) + 1, int(cells[:, 0].max()) + 1)
    occ = np.zeros(shape, dtype=np.uint8)
    occ[cells[sel, 1], cells[sel, 0]] = 1
    k = 2 * halo_cells + 1
    occ = cv2.dilate(occ, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return occ[cells[:, 1], cells[:, 0]] > 0


def _search_mask_groups(
    pts_xyz: np.ndarray,
    memberships: list[np.ndarray],
    block: BlockSpec,
    row_yaw_rad: float,
    *,
    min_points: int = 16,
    max_overlap_frac: float = 0.15,
    max_unexplained_frac: float = 0.10,
    residual_halo_m: float = 0.003,
    max_valid_masks: int = 12,
    max_alternatives: int = 3,
    diag: dict[str, Any] | None = None,
) -> list[list[np.ndarray]]:
    """Order-independent search for geometrically valid groups of external masks.

    A mask is eligible only if its projected depth support fits ONE cuboid on its own, so merged
    duplicates (two blocks) and false parts (fragments of one block) are excluded by geometry rather
    than by mask-overlap heuristics. Eligible masks are combined into nearly disjoint sets; points
    claimed by two members are dropped from both. Support not explained by the set, after removing a
    thin halo around accepted groups, must either fit one further cuboid (a block whose mask is
    missing) or stay below ``max_unexplained_frac`` of the smallest group; otherwise the set is rejected so that occupied
    volume is never silently left out of an accepted partition. Returns boolean point-membership
    groups for up to ``max_alternatives`` distinct partitions tied on the primary ranking keys.
    """
    n = len(pts_xyz)
    d = diag if diag is not None else {}
    # Deterministic canonical order and exact-duplicate removal (input order must not matter).
    uniq: dict[bytes, np.ndarray] = {}
    for m in memberships:
        uniq.setdefault(np.packbits(m).tobytes(), m)
    keys = sorted(uniq, key=lambda k: (-int(np.count_nonzero(uniq[k])), k))
    masks = [uniq[k] for k in keys]
    d["unique_masks"] = len(masks)

    fit_cache: dict[bytes, BlockFit | None] = {}

    def fit_of(sel: np.ndarray) -> BlockFit | None:
        key = np.packbits(sel).tobytes()
        if key not in fit_cache:
            fit_cache[key] = (
                fit_cuboid_top(pts_xyz[sel], block, row_yaw_rad=row_yaw_rad)[0]
                if int(np.count_nonzero(sel)) >= min_points else None
            )
        return fit_cache[key]

    valid = [i for i, m in enumerate(masks) if fit_of(m) is not None]
    d["single_cuboid_masks"] = len(valid)
    if len(valid) > max_valid_masks:
        d["truncated_valid_masks"] = len(valid) - max_valid_masks
        valid = valid[:max_valid_masks]
    if not valid:
        return []
    areas = {i: int(np.count_nonzero(masks[i])) for i in valid}
    compatible = {
        (i, j): int(np.count_nonzero(masks[i] & masks[j])) <= max_overlap_frac * min(areas[i], areas[j])
        for a, i in enumerate(valid) for j in valid[a + 1:]
    }

    cell_m = 0.001
    cells = np.floor((pts_xyz[:, :2] - pts_xyz[:, :2].min(axis=0)) / cell_m).astype(np.int32)
    halo_cells = max(1, int(round(residual_halo_m / cell_m)))
    candidates: dict[bytes, tuple[tuple, list[np.ndarray]]] = {}

    def consider(subset: tuple[int, ...]) -> None:
        stack = np.stack([masks[i] for i in subset])
        count = stack.sum(axis=0)
        contested = count >= 2
        groups = [masks[i] & ~contested for i in subset]
        if any(fit_of(g) is None for g in groups):
            return
        union = count >= 1
        # Support hugging accepted groups (mask boundary erosion) counts as explained; far residual must be
        # one further cuboid (missing mask) or smaller than a fraction of ONE block's support.
        explained = union | _near_points(cells, union, halo_cells)
        far = ~explained
        n_resid_groups = 0
        n_far = int(np.count_nonzero(far))
        if n_far >= min_points:
            # Fit the missing block from ALL unclaimed support (halo included) to avoid a halo-trim bias.
            resid = next((r for r in (~union, far) if fit_of(r) is not None), None)
            if resid is not None:
                groups.append(resid)
                explained = explained | resid
                n_resid_groups = 1
            elif n_far > max_unexplained_frac * min(int(np.count_nonzero(g)) for g in groups):
                return
        if len(groups) < 2:
            return
        coverage = float(np.count_nonzero(explained)) / max(1, n)
        size_err = float(sum(fit_of(g).size_err for g in groups))  # type: ignore[union-attr]
        sig = b"".join(sorted(np.packbits(g).tobytes() for g in groups))
        rank = (-round(coverage * 20.0) / 20.0, n_resid_groups, round(size_err, 4), sig)
        if sig not in candidates or rank < candidates[sig][0]:
            candidates[sig] = (rank, groups)

    def dfs(start: int, chosen: tuple[int, ...]) -> None:
        if chosen:
            consider(chosen)
        for a in range(start, len(valid)):
            i = valid[a]
            if all(compatible[(min(i, j), max(i, j))] for j in chosen):
                dfs(a + 1, chosen + (i,))

    dfs(0, ())
    d["partitions_considered"] = len(candidates)
    if not candidates:
        return []
    ranked = sorted(candidates.values(), key=lambda item: item[0])
    best_key = ranked[0][0][:2]
    tied = [groups for rank, groups in ranked if rank[:2] == best_key][:max_alternatives]
    d["returned_partitions"] = len(tied)
    d["best_coverage"] = -best_key[0]
    d["best_residual_groups"] = best_key[1]
    return tied


def propose_hypotheses_from_2d_masks(
    pts_xyz: np.ndarray,
    masks_u8: list[np.ndarray],
    img_ctx: ImageProjectionContext | None,
    block: BlockSpec,
    th: PolicyThresholds,
    row_yaw_rad: float = 0.0,
    side_verifier: Any = None,
    *,
    suppress_merged: bool = True,
    grouping: str = "greedy",
    diag_out: list[dict[str, Any]] | None = None,
) -> list[PartitionHypothesis]:
    """Convert external 2D instance masks into metric cuboid fits + boundary CutEvidence.

    ``grouping="greedy"`` is the legacy first-come point claiming (optionally after merged-mask
    suppression); ``grouping="search"`` uses the order-independent geometric search above and
    ignores ``suppress_merged`` because merged masks fail the single-cuboid test by construction.
    """
    if grouping not in MASK_GROUPING_MODES:
        raise ValueError(f"unknown mask grouping {grouping!r}")
    if img_ctx is None or not masks_u8 or len(pts_xyz) < 20:
        return []
    H, W = img_ctx.gray_img.shape[:2]
    uv, _ = cv2.projectPoints(np.asarray(pts_xyz, dtype=np.float64), img_ctx.rvec_ct, img_ctx.tvec_ct, img_ctx.K, img_ctx.D)
    uv = np.round(uv.reshape(-1, 2)).astype(np.int32)
    in_b = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)

    memberships: list[np.ndarray] = []
    for mask in masks_u8:
        mask = np.asarray(mask)
        if mask.shape != (H, W):
            continue
        hits = np.zeros(len(pts_xyz), dtype=bool)
        hits[in_b] = mask[uv[in_b, 1], uv[in_b, 0]] > 0
        if np.count_nonzero(hits) >= 16:
            memberships.append(hits)

    diag: dict[str, Any] = {"grouping": grouping, "memberships": len(memberships), "component_points": int(len(pts_xyz))}
    if grouping == "search":
        partitions = _search_mask_groups(pts_xyz, memberships, block, row_yaw_rad, diag=diag)
    else:
        if suppress_merged:
            memberships = suppress_merged_masks(memberships)
            diag["after_suppression"] = len(memberships)
        greedy: list[np.ndarray] = []
        claimed = np.zeros(len(pts_xyz), dtype=bool)
        for membership in memberships:
            hits = membership & ~claimed
            if np.count_nonzero(hits) >= 16:
                claimed |= hits
                greedy.append(hits)
        diag["unclaimed_frac"] = round(1.0 - float(np.count_nonzero(claimed)) / max(1, len(pts_xyz)), 3)
        partitions = [greedy]
    if diag_out is not None:
        diag_out.append(diag)

    out: list[PartitionHypothesis] = []
    for groups in partitions:
        hyp = _partition_from_point_groups(pts_xyz, [pts_xyz[g] for g in groups], img_ctx, block, th, row_yaw_rad, side_verifier)
        if hyp is not None:
            out.append(hyp)
    diag["hypotheses"] = len(out)
    return out


def _partition_from_point_groups(
    pts_xyz: np.ndarray,
    groups: list[np.ndarray],
    img_ctx: ImageProjectionContext,
    block: BlockSpec,
    th: PolicyThresholds,
    row_yaw_rad: float,
    side_verifier: Any,
) -> PartitionHypothesis | None:
    if len(groups) < 2:
        return None

    child_fits: list[BlockFit] = []
    total_size_err = 0.0
    for gpts in groups:
        fit_i, _ = fit_cuboid_top(gpts, block, row_yaw_rad=row_yaw_rad)
        if fit_i is None:
            return None
        child_fits.append(fit_i)
        total_size_err += fit_i.size_err

    # Estimate separating direction between sorted child centroids
    ref_yaw = float(child_fits[0].yaw_rad)
    c0 = np.array(child_fits[0].xy_m)
    c1 = np.array(child_fits[1].xy_m)
    dv = c1 - c0
    nv = float(np.linalg.norm(dv))
    if nv < 1e-4:
        return None
    d = dv / nv
    perp_d = np.array([-d[1], d[0]], dtype=np.float64)
    along = pts_xyz[:, :2] @ d
    perp = pts_xyz[:, :2] @ perp_d
    axis_deg = float(np.degrees(np.arctan2(d[1], d[0])) % 180.0)
    ordered = sorted(child_fits, key=lambda f: float(np.dot(f.xy_m, d)))
    cuts_m = [0.5 * (float(np.dot(a.xy_m, d)) + float(np.dot(b.xy_m, d))) for a, b in zip(ordered[:-1], ordered[1:])]
    cue_list = [evaluate_cut_cues(pts_xyz, along, perp, c_m, d, perp_d, img_ctx, th, 0) for c_m in cuts_m]
    return _make_partition_candidate(axis_deg, len(ordered), cue_list, ordered, total_size_err, side_verifier, cuts_m, block)


# ------------------------------------------------------------------ Generic scalar-elevation watershed (Meyer)

def _meyer_watershed_2d(elevation: np.ndarray, markers: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Exact 4-connected priority-flood scalar-elevation watershed (Numba-optional + clean heapq)."""
    return meyer_watershed_2d_fast(elevation, markers, mask)


_HAS_SKIMAGE = importlib.util.find_spec("skimage") is not None


def segment_scalar_watershed(
    h_filled: np.ndarray,
    color_rgb: np.ndarray,
    color_valid: np.ndarray,
    fg_mask: np.ndarray,
    step_barrier_m: float = 0.0025,
) -> tuple[np.ndarray, int]:
    """Generic class-agnostic scalar-elevation watershed control (NO lane/block-width priors!).

    Splits objects separated by valid depth discontinuities, and deterministically keeps a flat
    coplanar flush rectangle as ONE unresolved seed (0% hallucinated flush splits).
    """
    fg = fg_mask.astype(bool)
    if not np.any(fg):
        return np.zeros(fg.shape, dtype=np.int32), 0

    h_safe = np.where(np.isfinite(h_filled), h_filled, 0.0).astype(np.float32)
    gx = np.abs(cv2.Sobel(h_safe, cv2.CV_32F, 1, 0, ksize=3)) / 4.0
    gy = np.abs(cv2.Sobel(h_safe, cv2.CV_32F, 0, 1, ksize=3)) / 4.0
    depth_grad = np.hypot(gx, gy)

    # Interior markers strictly from depth-step-eroded foreground (never split coplanar tops on RGB edges)
    barrier = (depth_grad >= step_barrier_m) & fg
    seed_mask = (fg & (~cv2.dilate(barrier.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool))).astype(np.uint8)
    seed_mask = cv2.morphologyEx(seed_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n_seeds, markers = cv2.connectedComponents(seed_mask, connectivity=4)
    if n_seeds <= 1:
        # Fallback to plain connected components on fg
        n_c, lab = cv2.connectedComponents(fg.astype(np.uint8), connectivity=8)
        return lab.astype(np.int32), max(0, n_c - 1)
    if n_seeds == 2:
        # 1 single interior seed -> entire foreground component stays 1 region
        lab = np.where(fg, 1, 0).astype(np.int32)
        return lab, 1

    gray = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    cgx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)) / 255.0
    cgy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)) / 255.0
    rgb_grad = np.where(color_valid, np.hypot(cgx, cgy), 0.0)
    elev = (depth_grad * 1000.0 + 0.5 * rgb_grad).astype(np.float32)

    if _HAS_SKIMAGE:
        from skimage.segmentation import watershed as sk_watershed
        labels = sk_watershed(elev, markers=markers, mask=fg, connectivity=1, compactness=0.0).astype(np.int32)
    else:
        labels = _meyer_watershed_2d(elev, markers, fg)
    return labels, n_seeds - 1


# ------------------------------------------------------------------ Optional Stage 2 Mask Challenger Interface

class ExternalMaskChallenger:
    """Clean interface for frozen-prompt offline/AMD Stage 2 instance mask challenger (e.g. SAM 2 Hiera-T)."""

    def __init__(self, weights_path: str | None = None) -> None:
        self.weights_path = weights_path
        self.has_ort = importlib.util.find_spec("onnxruntime") is not None
        self.has_sam2 = importlib.util.find_spec("sam2") is not None
        self.available = bool(weights_path and (self.has_ort or self.has_sam2))
        if not self.has_sam2 and not self.has_ort:
            self.unavailable_reason = "sam2 and onnxruntime packages not installed in this environment"
        elif not weights_path:
            self.unavailable_reason = "no frozen SAM 2 checkpoint/ONNX path supplied"
        else:
            self.unavailable_reason = ""

    def propose_masks_from_roi_boxes(self, rgb_img: np.ndarray, roi_boxes_xyxy: list[tuple[int, int, int, int]]) -> dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.unavailable_reason,
            "masks": [],
        }

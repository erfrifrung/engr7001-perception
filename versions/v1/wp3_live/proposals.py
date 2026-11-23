"""Multi-hypothesis zero-width partition proposals, native-RGB seam engine, and generic scalar watershed."""
from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import importlib.util
from typing import Any

import cv2
import numpy as np

from .config import BlockSpec, PolicyThresholds


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
    def from_rgb(cls, rgb_img: np.ndarray, K: np.ndarray, D: np.ndarray, T_table_cam: np.ndarray) -> "ImageProjectionContext":
        rgb_u8 = np.ascontiguousarray(rgb_img, dtype=np.uint8)
        bgr_u8 = rgb_u8[:, :, ::-1]
        T_tc = np.asarray(T_table_cam, dtype=np.float64).reshape(4, 4)
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
    xy = np.asarray(points_xyz[:, :2], dtype=np.float64)
    mean_xy = xy.mean(axis=0)
    cov = np.cov((xy - mean_xy).T)
    if cov.shape != (2, 2) or not np.isfinite(cov).all():
        return None, {"reason": "degenerate_cov", "points": int(len(points_xyz))}
    _, vecs = np.linalg.eigh(cov)
    principal = float(np.arctan2(vecs[1, -1], vecs[0, -1]))
    nom = np.array([block.major_m, block.minor_m], dtype=np.float64)

    candidates = [principal, principal + np.pi / 2.0, row_yaw_rad, row_yaw_rad + np.pi / 2.0, *extra_yaws_rad]
    best = None
    best_diag: dict[str, Any] | None = None
    for yaw in candidates:
        c, s = float(np.cos(yaw)), float(np.sin(yaw))
        rot = np.array([[c, -s], [s, c]], dtype=np.float64)
        loc = (xy - mean_xy) @ rot
        lo, hi = np.percentile(loc, [4.0, 96.0], axis=0)
        span_raw = hi - lo
        if span_raw[1] > span_raw[0]:
            span_eval = span_raw[::-1] + block.edge_loss_m
            yaw_use = yaw + np.pi / 2.0
            cols = np.maximum(1, np.ceil(span_raw[::-1] / fill_cell_m).astype(int))
            loc_bins = np.column_stack((loc[:, 1] - lo[1], loc[:, 0] - lo[0]))
        else:
            span_eval = span_raw + block.edge_loss_m
            yaw_use = yaw
            cols = np.maximum(1, np.ceil(span_raw / fill_cell_m).astype(int))
            loc_bins = loc - lo

        err = np.abs(span_eval - nom) / nom
        bins = np.clip((loc_bins / fill_cell_m).astype(int), 0, cols - 1)
        occ = int(np.unique(bins[:, 0] * cols[1] + bins[:, 1]).size)
        fill = occ / max(1, int(cols[0] * cols[1]))
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
            cen_xy = mean_xy + rot @ ((lo + hi) / 2.0)
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
    z_top = float(np.median(points_xyz[:, 2]))
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
    """Bilinear image sampling with strict bounds checking (from heightmap_seam_pipeline.py:537)."""
    h, w = img.shape[:2]
    x = uv[:, 0]
    y = uv[:, 1]
    inside = (x >= 0.0) & (x <= w - 1.001) & (y >= 0.0) & (y <= h - 1.001)
    if not np.any(inside):
        return np.zeros((len(uv),) + img.shape[2:], dtype=np.float32), inside
    xi = np.clip(x, 0.0, w - 1.001)
    yi = np.clip(y, 0.0, h - 1.001)
    x0 = np.floor(xi).astype(np.int32)
    y0 = np.floor(yi).astype(np.int32)
    dx = (xi - x0)[:, None] if img.ndim == 3 else (xi - x0)
    dy = (yi - y0)[:, None] if img.ndim == 3 else (yi - y0)
    v00 = img[y0, x0].astype(np.float32)
    v10 = img[y0, x0 + 1].astype(np.float32)
    v01 = img[y0 + 1, x0].astype(np.float32)
    v11 = img[y0 + 1, x0 + 1].astype(np.float32)
    out = (1.0 - dy) * ((1.0 - dx) * v00 + dx * v10) + dy * ((1.0 - dx) * v01 + dx * v11)
    return out, inside


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
    along_vals = np.linspace(lo_a + 0.004, hi_a - 0.004, n_along)
    stations = np.linspace(p_lo + 0.004, p_hi - 0.004, 15)
    aa, ss = np.meshgrid(along_vals, stations, indexing="ij")
    xy_grid = aa[..., None] * d[None, None, :] + ss[..., None] * perp_d[None, None, :]
    pts3 = np.concatenate((xy_grid, np.full((len(along_vals), len(stations), 1), z_top)), axis=2).reshape(-1, 3)
    uv, _ = cv2.projectPoints(pts3, img_ctx.rvec_ct, img_ctx.tvec_ct, img_ctx.K, img_ctx.D)
    uv = uv.reshape(-1, 2)
    gvals, inside = _bilinear_sample(img_ctx.gray_img, uv)
    ggrid = np.where(inside, gvals, np.nan).reshape(len(along_vals), len(stations))
    valid_a = np.sum(np.isfinite(ggrid), axis=1) >= 5
    if np.count_nonzero(valid_a) < 9:
        return 0
    prof = np.nanmedian(ggrid, axis=1)
    # Local dark valley prominence relative to +-6 mm trend
    peaks = 0
    half = max(3, int(round(0.006 / ((hi_a - lo_a) / max(1, n_along)))))
    in_p = np.zeros(len(prof), dtype=bool)
    for k in range(1, len(prof) - 1):
        if not valid_a[k] or not (prof[k] <= prof[k - 1] and prof[k] <= prof[k + 1]):
            continue
        w0, w1 = max(0, k - half), min(len(prof), k + half + 1)
        wing = prof[w0:w1]
        wing = wing[np.isfinite(wing)]
        if len(wing) >= 4 and (float(np.median(wing)) - float(prof[k])) >= blackhat_thresh:
            in_p[k] = True
    # Merge adjacent valley indices within 4 mm
    last_k = -100
    min_sep = max(2, int(round(0.004 / ((hi_a - lo_a) / max(1, n_along)))))
    for k in np.flatnonzero(in_p):
        if k - last_k >= min_sep:
            peaks += 1
            last_k = int(k)
    return peaks


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
    heights = points_xyz[:, 2]
    dist = along - cut_val

    # 1. Validated depth step MEASURED ON FLANKS AWAY FROM THE LINE ([flank_inner_m, flank_outer_m])
    near_left = (dist < -th.flank_inner_m) & (dist >= -th.flank_outer_m)
    near_right = (dist > th.flank_inner_m) & (dist <= th.flank_outer_m)
    on_seam = np.abs(dist) <= 0.0022
    depth_step_mm = 0.0
    face_step_mm = 0.0
    depth_trough_mm = 0.0
    depth_step_ok = False
    depth_trough_ok = False
    if min(int(near_left.sum()), int(near_right.sum())) >= th.flank_min_pts:
        hl = float(np.median(heights[near_left]))
        hr = float(np.median(heights[near_right]))
        depth_step_m = abs(hl - hr)
        depth_step_mm = depth_step_m * 1000.0
        left_all = dist < 0.0
        right_all = dist >= 0.0
        if np.any(left_all) and np.any(right_all):
            fl = float(np.median(heights[left_all]))
            fr = float(np.median(heights[right_all]))
            face_step_m = abs(fl - fr)
            face_step_mm = face_step_m * 1000.0
            same_sign = (hl - hr) * (fl - fr) > 0.0
            depth_step_ok = bool(
                same_sign and depth_step_m >= th.depth_step_min_m and face_step_m >= th.face_step_min_m
            )
        if np.count_nonzero(on_seam) >= 3:
            depth_trough_m = min(hl, hr) - float(np.median(heights[on_seam]))
            depth_trough_mm = depth_trough_m * 1000.0
            depth_trough_ok = bool(depth_trough_m >= th.depth_trough_min_m)

    # 2. Native-RGB dark crevice + colour step + along-cut station continuity + both-shoulder depth support
    p_lo, p_hi = float(np.percentile(perp, 6.0)), float(np.percentile(perp, 94.0))
    margin = min(0.003, max(0.0, (p_hi - p_lo) * 0.12))
    z_top = float(np.median(points_xyz[:, 2]))

    rgb_trough = 0.0
    rgb_grad = 0.0
    rgb_color_step = 0.0
    along_continuity = 0.0
    shoulder_support = 0.0
    best_offset_m = 0.0
    rgb_ok = False

    if img_ctx is not None and len(points_xyz) >= 12 and (p_hi - p_lo) > 0.010:
        stations = np.linspace(p_lo + margin, p_hi - margin, 21)
        # Depth support on both flanks along stations (vectorised searchsorted)
        p_l = np.sort(perp[(dist < -0.002) & (dist >= -0.010)])
        p_r = np.sort(perp[(dist > 0.002) & (dist <= 0.010)])
        if len(p_l) > 0 and len(p_r) > 0:
            has_l = (np.searchsorted(p_l, stations + 0.0035, side="right") - np.searchsorted(p_l, stations - 0.0035, side="left")) > 0
            has_r = (np.searchsorted(p_r, stations + 0.0035, side="right") - np.searchsorted(p_r, stations - 0.0035, side="left")) > 0
            shoulder_support = float(np.mean(has_l & has_r))

        offsets = np.array([-0.007, -0.005, -0.003, -0.0015, 0.0, 0.0015, 0.003, 0.005, 0.007])
        oo, ss = np.meshgrid(offsets, stations, indexing="ij")
        xy_grid = (
            (cut_val + oo[..., None]) * direction[None, None, :]
            + ss[..., None] * perp_dir[None, None, :]
        )
        pts3 = np.concatenate((xy_grid, np.full((len(offsets), len(stations), 1), z_top)), axis=2).reshape(-1, 3)
        uv, _ = cv2.projectPoints(pts3, img_ctx.rvec_ct, img_ctx.tvec_ct, img_ctx.K, img_ctx.D)
        uv = uv.reshape(-1, 2)
        gray_vals, inside = _bilinear_sample(img_ctx.gray_img, uv)
        bgr_vals, _ = _bilinear_sample(img_ctx.bgr_img, uv)
        gray_grid = np.where(inside, gray_vals, np.nan).reshape(len(offsets), len(stations))
        bgr_grid = np.where(inside[:, None], bgr_vals, np.nan).reshape(len(offsets), len(stations), 3)

        valid_per_off = np.sum(np.isfinite(gray_grid), axis=1) >= 8
        if np.all(valid_per_off):
            prof = np.nanmedian(gray_grid, axis=1)
            left_sh = float(np.median(prof[:3]))
            right_sh = float(np.median(prof[-3:]))
            seam_idx = 3 + int(np.argmin(prof[3:6]))
            seam_val = float(prof[seam_idx])
            best_offset_m = float(offsets[seam_idx])
            rgb_trough = min(left_sh, right_sh) - seam_val
            rgb_grad = max(abs(left_sh - seam_val), abs(right_sh - seam_val))
            bgr_left = np.nanmedian(bgr_grid[:3].reshape(-1, 3), axis=0)
            bgr_right = np.nanmedian(bgr_grid[-3:].reshape(-1, 3), axis=0)
            rgb_color_step = float(np.linalg.norm(bgr_left - bgr_right))

            # Per-station along-cut continuity (inspired by tabletop.py:117-125)
            st_left = np.nanmedian(gray_grid[:3, :], axis=0)
            st_right = np.nanmedian(gray_grid[-3:, :], axis=0)
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


def enumerate_partition_hypotheses(
    pts_xyz: np.ndarray,
    img_ctx: ImageProjectionContext | None,
    block: BlockSpec,
    th: PolicyThresholds,
    row_yaw_rad: float = 0.0,
    side_verifier: Any = None,
) -> list[PartitionHypothesis]:
    """Generate and score zero-width K=2 and K=3 partition hypotheses across candidate axes."""
    xy = np.asarray(pts_xyz[:, :2], dtype=np.float64)
    if len(xy) < 20:
        return []
    c_xy = xy.mean(axis=0)
    _, vecs = np.linalg.eigh(np.cov((xy - c_xy).T))
    rect = cv2.minAreaRect(xy.astype(np.float32))
    box = cv2.boxPoints(rect)
    raw_dirs = [vecs[:, -1], vecs[:, 0], box[1] - box[0], box[2] - box[1]]
    if row_yaw_rad is not None:
        raw_dirs.extend([
            np.array([np.cos(row_yaw_rad), np.sin(row_yaw_rad)]),
            np.array([-np.sin(row_yaw_rad), np.cos(row_yaw_rad)]),
        ])

    # Deduplicate axes modulo 180 deg within 5 deg (preserves both 0 & 90 deg on 75x75 squares!)
    directions: list[tuple[np.ndarray, float]] = []
    for v in raw_dirs:
        nv = float(np.linalg.norm(v))
        if nv < 1e-6:
            continue
        u = v / nv
        ang = float(np.degrees(np.arctan2(u[1], u[0])) % 180.0)
        if not any(min(abs(ang - a0), 180.0 - abs(ang - a0)) < 5.0 for _, a0 in directions):
            directions.append((u, ang))

    unit_sizes = sorted({block.major_m, block.minor_m})
    hypotheses: list[PartitionHypothesis] = []
    z_top = float(np.median(pts_xyz[:, 2]))

    for d, axis_deg in directions:
        perp_d = np.array([-d[1], d[0]], dtype=np.float64)
        along = xy @ d
        perp = xy @ perp_d
        lo_a, hi_a = float(np.percentile(along, 4.0)), float(np.percentile(along, 96.0))
        p_lo, p_hi = float(np.percentile(perp, 6.0)), float(np.percentile(perp, 94.0))
        span = (hi_a - lo_a) + block.edge_loss_m

        stripe_peaks = 0
        if img_ctx is not None:
            stripe_peaks = _count_periodic_dark_peaks(xy, z_top, d, perp_d, lo_a, hi_a, p_lo, p_hi, img_ctx)

        for k_blocks in (2, 3):
            expected_spans = [k_blocks * u for u in unit_sizes]
            if not any(abs(span - exp) <= block.size_tol * exp for exp in expected_spans):
                continue

            if k_blocks == 2:
                cut_tuples = [(lo_a + f * (hi_a - lo_a),) for f in (0.42, 0.46, 0.50, 0.54, 0.58)]
            else:
                cut_tuples = [
                    (lo_a + f1 * (hi_a - lo_a), lo_a + f2 * (hi_a - lo_a))
                    for f1 in (0.31, 0.333, 0.36)
                    for f2 in (0.64, 0.667, 0.69)
                ]

            best_for_axis_k: PartitionHypothesis | None = None
            for cuts in cut_tuples:
                cue_list: list[CutEvidence] = []
                refined_cuts: list[float] = []
                any_cue = True
                for c_val in cuts:
                    cue = evaluate_cut_cues(pts_xyz, along, perp, c_val, d, perp_d, img_ctx, th, stripe_peaks)
                    if not (cue.depth_step_ok or cue.depth_trough_ok or cue.rgb_ok):
                        any_cue = False
                        break
                    cue_list.append(cue)
                    refined_cuts.append(c_val + (cue.best_offset_m if cue.rgb_ok else 0.0))
                if not any_cue:
                    continue

                # ZERO-WIDTH cut assignment (neither block loses area from a 1-cell erosion!)
                bounds = [-np.inf, *refined_cuts, np.inf]
                child_fits: list[BlockFit] = []
                total_size_err = 0.0
                valid_split = True
                for i in range(k_blocks):
                    sub_mask = (along >= bounds[i]) & (along < bounds[i + 1])
                    fit_i, _ = fit_cuboid_top(
                        pts_xyz[sub_mask], block, row_yaw_rad=float(np.radians(axis_deg)),
                    )
                    if fit_i is None:
                        valid_split = False
                        break
                    child_fits.append(fit_i)
                    total_size_err += fit_i.size_err
                if not valid_split:
                    continue

                # Run side-view verification ONLY on geometrically valid candidate cuts
                if side_verifier is not None:
                    for idx_c, cue in enumerate(cue_list):
                        side_ev = side_verifier.verify_cut(cue, refined_cuts[idx_c], block.height_m)
                        cue.side_status = side_ev.status
                        cue.side_visibility_r = side_ev.visibility_ratio_r
                        cue.side_edge_score = side_ev.end_face_edge_score
                        cue.side_depth_step_mm = side_ev.end_face_depth_step_mm

                mean_cont = float(np.mean([c.along_cut_continuity for c in cue_list]))
                cue_bonus = sum(
                    0.10 * min(c.depth_step_mm, 6.0) / 6.0
                    + 0.08 * min(c.rgb_trough, 25.0) / 25.0 * c.along_cut_continuity
                    + (0.12 if c.side_status == "CORROBORATED" else (-0.25 if c.side_status == "CONTRADICTED" else 0.0))
                    for c in cue_list
                )
                score = cue_bonus - total_size_err
                cand = PartitionHypothesis(
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
                if best_for_axis_k is None or cand.score > best_for_axis_k.score:
                    best_for_axis_k = cand
            if best_for_axis_k is not None:
                hypotheses.append(best_for_axis_k)

    hypotheses.sort(key=lambda h: -h.score)
    return hypotheses


# ------------------------------------------------------------------ Generic scalar-elevation watershed (Meyer)

def _meyer_watershed_2d(elevation: np.ndarray, markers: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Exact 4-connected priority-flood scalar-elevation watershed (Accepts float32 directly)."""
    H, W = elevation.shape
    labels = markers.copy().astype(np.int32)
    pq: list[tuple[float, int, int, int]] = []
    seq = 0
    seeded_r, seeded_c = np.nonzero(labels > 0)
    for r, c in zip(seeded_r.tolist(), seeded_c.tolist()):
        if not mask[r, c]:
            continue
        lbl = int(labels[r, c])
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and labels[nr, nc] == 0:
                heapq.heappush(pq, (float(elevation[nr, nc]), seq, nr, nc))
                labels[nr, nc] = -lbl  # queued sentinel carrying seed label? use parent assignment below
                seq += 1
    # Clean re-implementation with explicit visited flags (zero label collision)
    labels = markers.copy().astype(np.int32)
    in_q = labels > 0
    pq = []
    for r, c in zip(seeded_r.tolist(), seeded_c.tolist()):
        if mask[r, c]:
            heapq.heappush(pq, (float(elevation[r, c]), 0, r, c, int(labels[r, c])))
    step = 1
    while pq:
        _, _, r, c, lbl = heapq.heappop(pq)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not in_q[nr, nc]:
                in_q[nr, nc] = True
                labels[nr, nc] = lbl
                heapq.heappush(pq, (float(elevation[nr, nc]), step, nr, nc, lbl))
                step += 1
    return np.where(mask & (labels > 0), labels, 0)


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

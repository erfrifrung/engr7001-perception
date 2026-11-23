"""Fixed-side facade detector, cross-view corroboration/contradiction & cropped §7a silhouette verifier."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import threading
import cv2
import numpy as np

from .config import BlockSpec, PolicyThresholds
from .fast_ops import fast_median_1d, fast_nanmedian_rows
from .geometry_cache import CameraModel
from .proposals import (
    BlockFit,
    CutEvidence,
    ImageProjectionContext,
    _bilinear_sample,
    _bilinear_sample_valid_depth,
)

_SIDE_OFF_D_F32 = np.array([-0.010, -0.006, -0.003, 0.0, 0.003, 0.006, 0.010], dtype=np.float32)
_UNIT_Z_FAC_F32 = np.linspace(0.0, 1.0, 7, dtype=np.float32)
_UNIT_3X3_A_FLAT, _UNIT_3X3_B_FLAT = [
    arr.ravel().astype(np.float32)
    for arr in np.meshgrid(np.linspace(-1.0, 1.0, 3, dtype=np.float32), np.linspace(-1.0, 1.0, 3, dtype=np.float32))
]
_SLOT_TT_FLAT, _SLOT_ZZ_FLAT = [
    arr.ravel().astype(np.float64)
    for arr in np.meshgrid(np.linspace(-0.4, 0.4, 3), np.linspace(-0.15, 0.35, 3))
]
_SLOT_RAY_LOCK = threading.Lock()
_SLOT_RAY_CACHE: dict[tuple, dict] = {}


@dataclass
class SideViewData:
    """Calibrated fixed-side observation bundle for one epoch."""

    cam: CameraModel
    T_table_cam: np.ndarray
    rgb_img: np.ndarray
    depth_m: np.ndarray          # float32 image (H, W) in metres
    stamp_ns: int
    img_ctx: ImageProjectionContext
    is_fresh: bool = True
    is_frozen: bool = False


@dataclass(frozen=True)
class SideCutVerification:
    status: str                  # CORROBORATED | CONTRADICTED | SIDE_UNSEEN | INCONCLUSIVE | UNAVAILABLE
    visibility_ratio_r: float
    end_face_edge_score: float
    end_face_depth_step_mm: float
    projected_px_span: float
    reason: str


@dataclass(frozen=True)
class SlotOccupancy:
    layer: int
    slot: int
    state: str                   # OCCUPIED | OBSERVED_EMPTY | DISPLACED | UNKNOWN
    visibility_ratio_r: float
    median_depth_delta_mm: float | None
    expected_range_m: float


class SideViewVerifier:
    """Projects top-view cut boundaries & tower slots into the calibrated fixed-side camera."""

    def __init__(self, side_data: SideViewData | None, th: PolicyThresholds) -> None:
        self.side = side_data
        self.th = th

    def _project_points(self, pts_table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self.side is not None
        ctx = self.side.img_ctx
        pts = np.asarray(pts_table, dtype=np.float64).reshape(-1, 3)
        cam_pts = pts @ ctx.T_cam_table[:3, :3].T + ctx.T_cam_table[:3, 3]
        uv, _ = cv2.projectPoints(pts, ctx.rvec_ct, ctx.tvec_ct, ctx.K, ctx.D)
        return uv.reshape(-1, 2), cam_pts[:, 2]

    def verify_cut(self, cue: CutEvidence, cut_val_m: float, block_h_m: float) -> SideCutVerification:
        """Check whether the fixed side view sees the cut's exposed end face and corroborates or contradicts."""
        if self.side is None or not self.side.is_fresh or self.side.is_frozen:
            return SideCutVerification("UNAVAILABLE", 0.0, 0.0, 0.0, 0.0, "side_stream_unavailable")

        d = np.array(cue.direction, dtype=np.float64)
        p = np.array(cue.perp_dir, dtype=np.float64)
        cam_pos_tbl = self.side.T_table_cam[:3, 3]

            # Project end-face break directly using outer block face offset!
            # Remember p_lo_m and p_hi_m had a 12% interior margin subtracted in evaluate_cut_cues,
            # so for a 75 mm block p_hi_m - p_lo_m ~ 52 mm; recover true half-span (~span/2) + 0.5 mm!
        half_perp = 0.5 * (cue.p_hi_m - cue.p_lo_m) / 0.76
        mid_perp = 0.5 * (cue.p_hi_m + cue.p_lo_m)
        ends = [
            (mid_perp - half_perp, -p),
            (mid_perp + half_perp, p),
        ]
        best_vis = None
        for p_end, face_normal_2d in ends:
            top_pt = cut_val_m * d + p_end * p
            to_cam_2d = cam_pos_tbl[:2] - top_pt
            norm_to_cam = float(np.linalg.norm(to_cam_2d))
            if norm_to_cam < 1e-4:
                continue
            facing = float(np.dot(face_normal_2d, to_cam_2d / norm_to_cam))
            if facing < 0.15:
                continue  # End face faces away from the fixed side camera

            # Sample a vertical column on the end face at offsets along d around cut_val_m
            z0_s = max(0.002, cue.z_top_m - 0.85 * block_h_m)
            z1_s = cue.z_top_m - 0.002
            z_samples = z0_s + (z1_s - z0_s) * _UNIT_Z_FAC_F32
            off_d = _SIDE_OFF_D_F32
            oo, zz = np.meshgrid(off_d, z_samples, indexing="ij")
            pts3 = np.stack([
                (cut_val_m + oo) * d[0] + p_end * p[0],
                (cut_val_m + oo) * d[1] + p_end * p[1],
                zz,
            ], axis=-1).reshape(-1, 3)
            uv, exp_z = self._project_points(pts3)
            H_img, W_img = self.side.depth_m.shape
            in_bounds = (
                (exp_z > 0.15)
                & (uv[:, 0] >= 2) & (uv[:, 0] <= W_img - 3)
                & (uv[:, 1] >= 2) & (uv[:, 1] <= H_img - 3)
            )
            if float(np.mean(in_bounds)) < 0.70:
                continue

            # F03 modal/amodal visibility ratio r_i against fixed-side valid-only depth
            d_meas, valid_d = _bilinear_sample_valid_depth(
                self.side.depth_m, uv, max_jump_m=self.th.max_depth_interp_jump_m,
            )
            valid_d = valid_d & in_bounds
            if np.any(valid_d):
                unoccluded = (d_meas[valid_d] >= exp_z[valid_d] - 0.020)
                vis_r = float(np.mean(unoccluded))
            else:
                vis_r = 0.0

            # Measure projected 15 mm face height span in pixels
            pt_top_bot = np.array([
                [cut_val_m * d[0] + p_end * p[0], cut_val_m * d[1] + p_end * p[1], cue.z_top_m],
                [cut_val_m * d[0] + p_end * p[0], cut_val_m * d[1] + p_end * p[1], max(0.0, cue.z_top_m - block_h_m)],
            ])
            uv_tb, _ = self._project_points(pt_top_bot)
            px_span = float(np.linalg.norm(uv_tb[0] - uv_tb[1]))

            if vis_r < self.th.side_vis_min_r:
                cand = SideCutVerification("SIDE_UNSEEN", round(vis_r, 3), 0.0, 0.0, round(px_span, 2), "end_face_occluded")
                if best_vis is None:
                    best_vis = cand
                continue

            # Sample RGB dark/edge gradient across the vertical end-face break (minimum of middle 3 offsets allows sub-pixel alignment)
            g_vals, g_in = _bilinear_sample(self.side.img_ctx.gray_img, uv)
            g_grid = np.where(g_in, g_vals, np.float32(np.nan)).reshape(len(off_d), len(z_samples))
            if np.sum(np.isfinite(g_grid)) < 15:
                continue
            d_grid = np.where(valid_d, d_meas, np.float32(np.nan)).reshape(len(off_d), len(z_samples))
            mid_z_rows = slice(1, len(z_samples) - 1)
            g_face = g_grid[:, mid_z_rows]
            if np.sum(np.isfinite(g_face)) < 10:
                continue
            prof_end = fast_nanmedian_rows(g_face)
            sh_l = float(prof_end[1]) if np.isfinite(prof_end[1]) else float(prof_end[0])
            sh_r = float(prof_end[-2]) if np.isfinite(prof_end[-2]) else float(prof_end[-1])
            val_c = float(np.nanmin(prof_end[2:5]))
            edge_score = float(min(sh_l, sh_r) - val_c)

            # Check actual depth proximity inside end face (must hit the vertical face, not air!)
            d_mid = d_grid[:, mid_z_rows].ravel()
            ez_mid = exp_z.reshape(len(off_d), len(z_samples))[:, mid_z_rows].ravel()
            ok_mid = np.isfinite(d_mid)
            face_hit = fast_median_1d(np.abs(d_mid[ok_mid] - ez_mid[ok_mid])) <= 0.015 if np.any(ok_mid) else False

            # Compare depth step on left vs right flanking shoulders (require >= 4 valid samples per flank!)
            d_l_blk = d_grid[:2, mid_z_rows].ravel()
            d_r_blk = d_grid[-2:, mid_z_rows].ravel()
            ok_l = np.isfinite(d_l_blk)
            ok_r = np.isfinite(d_r_blk)
            dl = d_l_blk[ok_l]
            dr = d_r_blk[ok_r]
            ez_l = exp_z.reshape(len(off_d), len(z_samples))[:2, mid_z_rows].ravel()[ok_l]
            ez_r = exp_z.reshape(len(off_d), len(z_samples))[-2:, mid_z_rows].ravel()[ok_r]
            d_step_mm = (
                abs(fast_median_1d(dl - ez_l) - fast_median_1d(dr - ez_r)) * 1000.0
                if (len(dl) >= 4 and len(dr) >= 4)
                else 0.0
            )

            corroborated = bool(face_hit and ((edge_score >= self.th.side_rgb_grad_min) or (d_step_mm >= self.th.side_depth_step_min_m * 1000.0)))
            if corroborated:
                return SideCutVerification(
                    "CORROBORATED", round(vis_r, 3), round(edge_score, 2), round(d_step_mm, 2), round(px_span, 2),
                    "visible_end_face_break"
                )
            elif self.th.allow_appearance_contradiction and vis_r >= 0.90 and px_span >= 14.0 and edge_score < 3.0 and d_step_mm < 1.2:
                best_vis = SideCutVerification(
                    "CONTRADICTED", round(vis_r, 3), round(edge_score, 2), round(d_step_mm, 2), round(px_span, 2),
                    "smooth_continuous_end_face_contradicts_split"
                )
            else:
                best_vis = SideCutVerification(
                    "INCONCLUSIVE", round(vis_r, 3), round(edge_score, 2), round(d_step_mm, 2), round(px_span, 2),
                    "smooth_or_subpixel_end_cue"
                )

        if best_vis is not None:
            return best_vis
        return SideCutVerification("SIDE_UNSEEN", 0.0, 0.0, 0.0, 0.0, "no_end_face_visible_to_side")


class TwoSideCutVerifier:
    """Combine two fixed-view cut checks without counting a seam twice."""

    _STATUS_PRIORITY = {
        "UNAVAILABLE": 0, "SIDE_UNSEEN": 1, "INCONCLUSIVE": 2,
        "CORROBORATED": 3, "CONTRADICTED": 4,
    }

    def __init__(self, side: SideViewVerifier, third: SideViewVerifier) -> None:
        self.side = side
        self.third = third
        self.third_verdicts: dict[str, int] = {}

    def verify_cut(self, cue: CutEvidence, cut_val_m: float, block_h_m: float) -> SideCutVerification:
        side_result = self.side.verify_cut(cue, cut_val_m, block_h_m)
        third_result = self.third.verify_cut(cue, cut_val_m, block_h_m)
        status = third_result.status
        self.third_verdicts[status] = self.third_verdicts.get(status, 0) + 1
        # Visible contradiction wins over corroboration from the other view.
        if self._STATUS_PRIORITY[status] > self._STATUS_PRIORITY[side_result.status]:
            return third_result
        return side_result


def _ray_box_slab(o_loc: np.ndarray, u_loc: np.ndarray, half_ext: np.ndarray) -> tuple[float, float] | None:
    """Exact 3D slab ray-AABB intersection [t_enter, t_exit] in block-local coordinates."""
    t_min = -1e9
    t_max = 1e9
    for ax in range(3):
        if abs(u_loc[ax]) < 1e-8:
            if o_loc[ax] < -half_ext[ax] or o_loc[ax] > half_ext[ax]:
                return None
        else:
            inv_d = 1.0 / u_loc[ax]
            t1 = (-half_ext[ax] - o_loc[ax]) * inv_d
            t2 = (half_ext[ax] - o_loc[ax]) * inv_d
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
            if t_min > t_max:
                return None
    if t_max <= 0.0:
        return None
    return max(0.0, float(t_min)), float(t_max)


def _build_slot_ray_cache(
    side: SideViewData,
    block: BlockSpec,
    layers: int,
    tower_origin_xy: tuple[float, float],
    tower_yaw_rad: float,
) -> dict[str, Any]:
    """Precompute static side-camera slot projections and ray-AABB intersections once per pose/calib."""
    ctx = side.img_ctx
    cam_pos = side.T_table_cam[:3, 3]
    H_img, W_img = side.depth_m.shape
    c0, s0 = float(np.cos(tower_yaw_rad)), float(np.sin(tower_yaw_rad))
    R_tw = np.array([[c0, -s0], [s0, c0]], dtype=np.float64)
    half_ext = np.array([block.length_m / 2.0, block.width_m / 2.0, block.height_m / 2.0], dtype=np.float64) + 0.0005
    z_grid_base = _SLOT_ZZ_FLAT * block.height_m

    slot_specs: list[dict[str, Any]] = []
    uv_chunks: list[np.ndarray] = []
    cursor = 0
    for layer in range(1, layers + 1):
        odd = (layer % 2 == 1)
        z_mid = (layer - 0.5) * block.height_m
        yaw_blk = tower_yaw_rad + (0.0 if odd else np.pi / 2.0)
        ux = np.array([np.cos(yaw_blk), np.sin(yaw_blk), 0.0])
        uy = np.array([-np.sin(yaw_blk), np.cos(yaw_blk), 0.0])
        uz = np.array([0.0, 0.0, 1.0])
        R_blk = np.stack((ux, uy, uz), axis=0)

        for slot in (-1, 0, 1):
            local_xy = np.array([0.0, slot * block.width_m]) if odd else np.array([slot * block.width_m, 0.0])
            cen_xy = np.asarray(tower_origin_xy) + R_tw @ local_xy
            cen = np.array([cen_xy[0], cen_xy[1], z_mid])
            face_cands = [
                (ux, block.length_m / 2.0, uy, block.width_m / 2.0),
                (-ux, block.length_m / 2.0, uy, block.width_m / 2.0),
                (uy, block.width_m / 2.0, ux, block.length_m / 2.0),
                (-uy, block.width_m / 2.0, ux, block.length_m / 2.0),
            ]
            to_cam = cam_pos - cen
            to_cam_unit = to_cam / max(1e-6, float(np.linalg.norm(to_cam)))
            cands = face_cands[:2] if slot == 0 else face_cands
            fnorm, foff, tang, h_tang = max(cands, key=lambda f: float(np.dot(f[0], to_cam_unit)))
            if float(np.dot(fnorm, to_cam_unit)) < 0.25:
                slot_specs.append({"kind": "unknown", "layer": layer, "slot": slot, "exp_range": float(np.linalg.norm(to_cam))})
                continue

            pts3 = (
                cen[None, :]
                + foff * fnorm[None, :]
                + (_SLOT_TT_FLAT * h_tang)[:, None] * tang[None, :]
                + z_grid_base[:, None] * uz[None, :]
            )
            exp_z = (pts3 @ ctx.T_cam_table[:3, :3].T + ctx.T_cam_table[:3, 3])[:, 2]
            uv, _ = cv2.projectPoints(pts3, ctx.rvec_ct, ctx.tvec_ct, ctx.K, ctx.D)
            uv = uv.reshape(-1, 2)
            in_b = (exp_z > 0.15) & (uv[:, 0] >= 1) & (uv[:, 0] <= W_img - 2) & (uv[:, 1] >= 1) & (uv[:, 1] <= H_img - 2)
            if not np.any(in_b):
                slot_specs.append({"kind": "unknown", "layer": layer, "slot": slot, "exp_range": float(np.mean(exp_z))})
                continue

            o_loc = R_blk @ (cam_pos - cen)
            slab_ok = np.zeros(len(pts3), dtype=bool)
            t_in = np.zeros(len(pts3), dtype=np.float64)
            t_out = np.zeros(len(pts3), dtype=np.float64)
            ray_scale = np.zeros(len(pts3), dtype=np.float64)
            for k in np.flatnonzero(in_b):
                ray_tbl = pts3[k] - cam_pos
                ray_len = float(np.linalg.norm(ray_tbl))
                if ray_len < 1e-4 or exp_z[k] < 1e-4:
                    continue
                slab = _ray_box_slab(o_loc, R_blk @ (ray_tbl / ray_len), half_ext)
                if slab is None:
                    continue
                slab_ok[k] = True
                t_in[k], t_out[k] = slab
                ray_scale[k] = ray_len / float(exp_z[k])

            if int(np.count_nonzero(slab_ok)) < 4:
                slot_specs.append({"kind": "unknown", "layer": layer, "slot": slot, "exp_range": float(np.mean(exp_z))})
                continue

            n_r = len(pts3)
            uv_chunks.append(uv.astype(np.float32))
            slot_specs.append({
                "kind": "rays",
                "layer": layer,
                "slot": slot,
                "s0": cursor,
                "s1": cursor + n_r,
                "slab_ok": slab_ok,
                "t_in": t_in,
                "t_out": t_out,
                "ray_scale": ray_scale,
                "mean_exp_z": float(np.mean(exp_z)),
            })
            cursor += n_r

    all_uv = np.vstack(uv_chunks) if uv_chunks else np.zeros((0, 2), dtype=np.float32)
    return {"specs": slot_specs, "all_uv": all_uv}


def detect_side_facade_slots(
    side: SideViewData | None,
    block: BlockSpec,
    th: PolicyThresholds,
    *,
    layers: int = 3,
    tower_origin_xy: tuple[float, float] = (0.0, 0.0),
    tower_yaw_rad: float = 0.0,
) -> list[SlotOccupancy]:
    """Detect lower-layer tower slot occupancy/recessed/empty/unknown via cached 3D ray-AABB clearing."""
    if side is None or not side.is_fresh or side.is_frozen:
        return [
            SlotOccupancy(L, s, "UNKNOWN", 0.0, None, 0.0)
            for L in range(1, layers + 1)
            for s in (-1, 0, 1)
        ]

    H_img, W_img = side.depth_m.shape
    cache_key = (
        side.cam.camera_id,
        H_img,
        W_img,
        tuple(np.round(side.cam.K_f64.ravel(), 4).tolist()),
        tuple(np.round(np.asarray(side.T_table_cam, dtype=np.float64).ravel(), 6).tolist()),
        round(block.length_m, 5),
        round(block.width_m, 5),
        round(block.height_m, 5),
        int(layers),
        tuple(round(float(x), 5) for x in tower_origin_xy),
        round(float(tower_yaw_rad), 6),
    )
    with _SLOT_RAY_LOCK:
        cached = _SLOT_RAY_CACHE.get(cache_key)
    if cached is None:
        cached = _build_slot_ray_cache(side, block, layers, tower_origin_xy, tower_yaw_rad)
        with _SLOT_RAY_LOCK:
            if len(_SLOT_RAY_CACHE) > 32:
                _SLOT_RAY_CACHE.clear()
            _SLOT_RAY_CACHE[cache_key] = cached

    all_uv = cached["all_uv"]
    if len(all_uv) > 0:
        d_meas_all, d_in_all = _bilinear_sample_valid_depth(
            side.depth_m, all_uv, max_jump_m=th.max_depth_interp_jump_m,
        )
    else:
        d_meas_all = np.zeros(0, dtype=np.float32)
        d_in_all = np.zeros(0, dtype=bool)

    out: list[SlotOccupancy] = []
    clear_tol = min(th.slot_clear_tol_m, 0.003)
    for spec in cached["specs"]:
        if spec["kind"] == "unknown":
            out.append(SlotOccupancy(spec["layer"], spec["slot"], "UNKNOWN", 0.0, None, spec["exp_range"]))
            continue

        s0, s1 = spec["s0"], spec["s1"]
        valid = spec["slab_ok"] & d_in_all[s0:s1]
        if int(np.count_nonzero(valid)) < 4:
            out.append(SlotOccupancy(spec["layer"], spec["slot"], "UNKNOWN", 0.0, None, spec["mean_exp_z"]))
            continue

        t_enter = spec["t_in"][valid]
        t_exit = spec["t_out"][valid]
        t_meas = d_meas_all[s0:s1][valid].astype(np.float64) * spec["ray_scale"][valid]
        vis_r = float(np.mean(t_meas >= t_enter - 0.015))
        d_front = t_meas - t_enter
        med_delta_m = fast_median_1d(d_front)
        med_delta_mm = round(med_delta_m * 1000.0, 2)

        if vis_r < 0.60:
            state = "UNKNOWN"
        else:
            clear_frac = float(np.mean(t_meas >= t_exit - clear_tol))
            inside_frac = float(np.mean((t_meas >= t_enter + th.recessed_slot_min_m) & (t_meas < t_exit - clear_tol)))
            if clear_frac >= 0.65 and vis_r >= th.empty_slot_vis_min_r:
                state = "OBSERVED_EMPTY"
            elif inside_frac >= 0.55:
                state = "RECESSED"
            elif abs(med_delta_m) <= th.occupied_tol_m:
                state = "OCCUPIED"
            elif med_delta_m <= -th.displaced_protrusion_m:
                state = "DISPLACED"
            else:
                state = "UNKNOWN"
        out.append(SlotOccupancy(spec["layer"], spec["slot"], state, round(vis_r, 3), med_delta_mm, round(float(np.mean(t_enter)), 3)))
    return out


def compute_target_side_visibility(
    fit: BlockFit,
    side: SideViewData | None,
    block: BlockSpec,
    th: PolicyThresholds,
) -> bool:
    """Target-level F03 check: True ONLY when THIS specific block is unoccluded in a fresh side frame."""
    if side is None or not side.is_fresh or side.is_frozen:
        return False
    H_img, W_img = side.depth_m.shape
    ctx = side.img_ctx
    c, s = float(np.cos(fit.yaw_rad)), float(np.sin(fit.yaw_rad))
    ux = np.array([c, s, 0.0], dtype=np.float32)
    uy = np.array([-s, c, 0.0], dtype=np.float32)
    hl = max(0.004, block.major_m * 0.25)
    hw = max(0.004, block.minor_m * 0.22)
    cen_top = np.array([fit.xy_m[0], fit.xy_m[1], fit.z_top_m], dtype=np.float32)
    pts3 = cen_top[None, :] + (_UNIT_3X3_A_FLAT * hl)[:, None] * ux[None, :] + (_UNIT_3X3_B_FLAT * hw)[:, None] * uy[None, :]
    exp_z = (pts3.astype(np.float64) @ ctx.T_cam_table[:3, :3].T + ctx.T_cam_table[:3, 3])[:, 2]
    uv, _ = cv2.projectPoints(pts3, ctx.rvec_ct, ctx.tvec_ct, ctx.K, ctx.D)
    uv = uv.reshape(-1, 2)
    in_b = (exp_z > 0.15) & (uv[:, 0] >= 2) & (uv[:, 0] <= W_img - 3) & (uv[:, 1] >= 2) & (uv[:, 1] <= H_img - 3)
    if np.count_nonzero(in_b) < 4:
        return False
    d_meas, d_ok = _bilinear_sample_valid_depth(side.depth_m, uv, max_jump_m=th.max_depth_interp_jump_m)
    valid = in_b & d_ok
    if np.count_nonzero(valid) < 4:
        return False
    vis_r = float(np.mean(d_meas[valid] >= exp_z[valid] - 0.020))
    return vis_r >= th.side_vis_min_r


def score_partition_multiview(
    fits: list[BlockFit],
    cuts: list[CutEvidence],
    wrist_ctx: ImageProjectionContext | None,
    wrist_depth_m: np.ndarray | None,
    side: SideViewData | None,
    block: BlockSpec,
    th: PolicyThresholds,
    *,
    is_intact: bool = False,
    third_side: SideViewData | None = None,
) -> dict[str, Any]:
    """Score a complete partition candidate strictly on mutually observable surfaces and cut evidence."""
    if not fits:
        return {"mv_score": -1e6, "visible_faces": 0, "phantom_frac": 1.0, "mean_size_err": 1.0, "cut_support": 0.0}

    mean_size_err = float(np.mean([f.size_err for f in fits]))
    visible_faces = 0
    side_errs: list[float] = []
    phantom_hits = 0
    total_side_samples = 0

    for cur_side in (side, third_side):
        if cur_side is not None and cur_side.is_fresh and not cur_side.is_frozen:
            H_s, W_s = cur_side.depth_m.shape
            cam_pos = cur_side.T_table_cam[:3, 3]
            s_ctx = cur_side.img_ctx
            hl = max(0.004, block.major_m * 0.25)
            hw = max(0.004, block.minor_m * 0.22)
            aa_hl = _UNIT_3X3_A_FLAT * hl
            bb_hw = _UNIT_3X3_B_FLAT * hw
            all_pts3 = []
            ux_list = []
            uy_list = []
            for f in fits:
                c, s = float(np.cos(f.yaw_rad)), float(np.sin(f.yaw_rad))
                ux = np.array([c, s, 0.0], dtype=np.float32)
                uy = np.array([-s, c, 0.0], dtype=np.float32)
                ux_list.append(ux)
                uy_list.append(uy)
                cen = np.array([f.xy_m[0], f.xy_m[1], f.z_top_m], dtype=np.float32)
                all_pts3.append(cen[None, :] + aa_hl[:, None] * ux[None, :] + bb_hw[:, None] * uy[None, :])
            batch_pts3 = np.vstack(all_pts3)
            exp_z_all = (batch_pts3.astype(np.float64) @ s_ctx.T_cam_table[:3, :3].T + s_ctx.T_cam_table[:3, 3])[:, 2]
            uv_all, _ = cv2.projectPoints(batch_pts3, s_ctx.rvec_ct, s_ctx.tvec_ct, s_ctx.K, s_ctx.D)
            uv_all = uv_all.reshape(-1, 2)
            in_b_all = (exp_z_all > 0.15) & (uv_all[:, 0] >= 2) & (uv_all[:, 0] <= W_s - 3) & (uv_all[:, 1] >= 2) & (uv_all[:, 1] <= H_s - 3)
            dm_all, dok_all = _bilinear_sample_valid_depth(cur_side.depth_m, uv_all, max_jump_m=th.max_depth_interp_jump_m)
            val_all = in_b_all & dok_all

            for i_f, f in enumerate(fits):
                sl = slice(i_f * 9, (i_f + 1) * 9)
                val = val_all[sl]
                if np.count_nonzero(val) >= 4:
                    deltas = dm_all[sl][val] - exp_z_all[sl][val]
                    vis_r = float(np.mean(deltas >= -0.020))
                    if vis_r >= th.side_vis_min_r:
                        visible_faces += 1
                        total_side_samples += int(np.count_nonzero(val))
                        phantom_hits += int(np.count_nonzero(deltas > th.verifier_residual_m))
                        side_errs.append(fast_median_1d(np.abs(deltas)))
                to_cam = cam_pos[:2] - np.array([f.xy_m[0], f.xy_m[1]])
                nrm_len = float(np.linalg.norm(to_cam))
                if nrm_len > 1e-4:
                    tc_u = to_cam / nrm_len
                    if max(abs(float(np.dot(ux_list[i_f][:2], tc_u))), abs(float(np.dot(uy_list[i_f][:2], tc_u)))) >= 0.25:
                        visible_faces += 1

    wrist_err = 0.0
    if wrist_ctx is not None and wrist_depth_m is not None:
        H_w, W_w = wrist_depth_m.shape
        all_top = np.array([[f.xy_m[0], f.xy_m[1], f.z_top_m] for f in fits], dtype=np.float64)
        exp_zw = (all_top @ wrist_ctx.T_cam_table[:3, :3].T + wrist_ctx.T_cam_table[:3, 3])[:, 2]
        uv_w, _ = cv2.projectPoints(all_top, wrist_ctx.rvec_ct, wrist_ctx.tvec_ct, wrist_ctx.K, wrist_ctx.D)
        uv_w = uv_w.reshape(-1, 2)
        in_bw = (exp_zw > 0.15) & (uv_w[:, 0] >= 1) & (uv_w[:, 0] <= W_w - 2) & (uv_w[:, 1] >= 1) & (uv_w[:, 1] <= H_w - 2)
        dm_w, dok_w = _bilinear_sample_valid_depth(wrist_depth_m, uv_w, max_jump_m=th.max_depth_interp_jump_m)
        val_w = in_bw & dok_w
        if np.any(val_w):
            wrist_err = fast_median_1d(np.abs(dm_w[val_w] - exp_zw[val_w]))

    phantom_frac = phantom_hits / max(1, total_side_samples)
    side_rmse_mm = (float(np.mean(side_errs)) * 1000.0) if side_errs else 0.0

    cut_support = 0.0
    if not is_intact and cuts:
        per_cut = []
        for c in cuts:
            s_val = 0.0
            if c.depth_step_ok:
                s_val += 0.45 + 0.15 * min(c.depth_step_mm, 6.0) / 6.0
            elif c.depth_trough_ok:
                s_val += 0.12
            if c.side_status == "CORROBORATED":
                s_val += 0.40
            elif c.side_status == "CONTRADICTED":
                s_val -= 0.50
            if c.rgb_ok:
                s_val += 0.28 * c.along_cut_continuity
            per_cut.append(s_val)
        cut_support = float(np.mean(per_cut))

    depth_consistency = -0.15 * wrist_err * 100.0 - 0.02 * min(side_rmse_mm, 15.0) - 0.80 * phantom_frac
    mv_score = cut_support + depth_consistency - mean_size_err + 0.02 * min(visible_faces, 6)
    return {
        "mv_score": round(float(mv_score), 4),
        "visible_faces": int(visible_faces),
        "phantom_frac": round(float(phantom_frac), 3),
        "mean_size_err": round(float(mean_size_err), 4),
        "cut_support": round(float(cut_support), 4),
    }


class CroppedSilhouetteVerifier:
    """Switchable §7a cropped CPU second-view silhouette/depth verifier with veto-only authority.

    Follows Astra's sampling rule (EVALUATION_AND_DECISIONS.md:36): evaluates difficult confident
    splits PLUS a frozen deterministic random subset of FRESH non-splits (never already-AMBIGUOUS).
    """

    def __init__(self, th: PolicyThresholds, seed: int = 20261119) -> None:
        self.th = th
        self.seed = seed

    def _should_sample(self, index: int, was_split: bool, epoch_id: int) -> bool:
        if was_split:
            return True
        digest = hashlib.blake2b(f"{self.seed}:{epoch_id}:{index}".encode(), digest_size=4).digest()
        val = int.from_bytes(digest, "little") / float(2**32)
        return val < self.th.verifier_sample_frac

    def verify_confident_outputs(
        self,
        fits: list[tuple[BlockFit, str]],
        side: SideViewData | None,
        block: BlockSpec,
        epoch_id: int = 0,
    ) -> tuple[list[tuple[BlockFit, str]], list[tuple[BlockFit, str]], dict[str, int]]:
        """Return (kept_confident, downgraded_to_ambiguous, stats). Strictly veto-only."""
        if side is None or not side.is_fresh or side.is_frozen:
            return fits, [], {"checked": 0, "vetoed": 0, "skipped_unseen": len(fits)}

        kept: list[tuple[BlockFit, str]] = []
        vetoed: list[tuple[BlockFit, str]] = []
        checked = 0
        skipped_unseen = 0
        H_img, W_img = side.depth_m.shape
        ctx = side.img_ctx
        hl = max(0.004, block.major_m * 0.25)
        hw = max(0.004, block.minor_m * 0.22)
        aa_hl = _UNIT_3X3_A_FLAT * hl
        bb_hw = _UNIT_3X3_B_FLAT * hw

        for idx, (fit, src) in enumerate(fits):
            was_split = src.startswith("split")
            if not self._should_sample(idx, was_split, epoch_id):
                kept.append((fit, src))
                continue

            c, s = float(np.cos(fit.yaw_rad)), float(np.sin(fit.yaw_rad))
            ux = np.array([c, s, 0.0], dtype=np.float32)
            uy = np.array([-s, c, 0.0], dtype=np.float32)
            cen_top = np.array([fit.xy_m[0], fit.xy_m[1], fit.z_top_m], dtype=np.float32)
            pts3 = cen_top[None, :] + aa_hl[:, None] * ux[None, :] + bb_hw[:, None] * uy[None, :]
            exp_z = (pts3.astype(np.float64) @ ctx.T_cam_table[:3, :3].T + ctx.T_cam_table[:3, 3])[:, 2]
            uv, _ = cv2.projectPoints(pts3, ctx.rvec_ct, ctx.tvec_ct, ctx.K, ctx.D)
            uv = uv.reshape(-1, 2)
            in_b = (exp_z > 0.15) & (uv[:, 0] >= 2) & (uv[:, 0] <= W_img - 3) & (uv[:, 1] >= 2) & (uv[:, 1] <= H_img - 3)
            if not np.any(in_b):
                skipped_unseen += 1
                kept.append((fit, src))
                continue

            ui = np.round(uv[:, 0]).astype(np.int32)
            vi = np.round(uv[:, 1]).astype(np.int32)
            deltas_list: list[float] = []
            for k in np.flatnonzero(in_b):
                patch = side.depth_m[vi[k] - 1 : vi[k] + 2, ui[k] - 1 : ui[k] + 2]
                vp = np.isfinite(patch) & (patch > 0.15) & (patch < 1.80)
                if np.any(vp):
                    deltas_list.append(float(np.min(patch[vp]) - exp_z[k]))
            if len(deltas_list) < 4:
                skipped_unseen += 1
                kept.append((fit, src))
                continue

            checked += 1
            deltas = np.asarray(deltas_list, dtype=np.float32)
            vis_r = float(np.mean(deltas >= -0.020))
            if vis_r < self.th.side_vis_min_r:
                skipped_unseen += 1
                kept.append((fit, src))
                continue

            med_behind = fast_median_1d(deltas)
            if med_behind > self.th.verifier_residual_m:
                vetoed.append((fit, f"vetoed:{src}:phantom_behind_{round(med_behind*1000,1)}mm"))
            else:
                kept.append((fit, src))

        return kept, vetoed, {"checked": checked, "vetoed": len(vetoed), "skipped_unseen": skipped_unseen}


def check_local_homography_step(
    wrist_ctx: ImageProjectionContext,
    side_ctx: ImageProjectionContext,
    cen_xy: tuple[float, float],
    z_top_m: float,
    step_test_mm: float = 2.0,
) -> dict[str, float]:
    """Bounded F27/F14 outer-edge-anchored cross-view homography step sensitivity helper.

    Computes local cross-view pixel shift induced by a >=2.0 mm step (explicitly NOT 0.5 mm bevels!).
    """
    n1 = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    cam1_tbl = wrist_ctx.T_table_cam
    cam2_tbl = side_ctx.T_table_cam
    # Express top plane in wrist camera coords
    n_c1 = cam1_tbl[:3, :3].T @ n1
    p0_tbl = np.array([cen_xy[0], cen_xy[1], z_top_m])
    p0_c1 = wrist_ctx.T_cam_table[:3, :3] @ p0_tbl + wrist_ctx.T_cam_table[:3, 3]
    d_c1 = abs(float(np.dot(n_c1, p0_c1)))
    if d_c1 < 0.15:
        return {"induced_parallax_px": 0.0}
    # Relative transform cam2 <- cam1
    T_21 = side_ctx.T_cam_table @ cam1_tbl
    R, t = T_21[:3, :3], T_21[:3, 3:4]
    H_top = side_ctx.K @ (R + (t @ n_c1.reshape(1, 3)) / d_c1) @ np.linalg.inv(wrist_ctx.K)
    H_step = side_ctx.K @ (R + (t @ n_c1.reshape(1, 3)) / max(0.15, d_c1 - step_test_mm * 0.001)) @ np.linalg.inv(wrist_ctx.K)
    uv1 = (wrist_ctx.K @ (p0_c1 / p0_c1[2]))[:2]
    p_uv1 = np.array([uv1[0], uv1[1], 1.0])
    q0 = H_top @ p_uv1; q0 = q0[:2] / q0[2]
    q1 = H_step @ p_uv1; q1 = q1[:2] / q1[2]
    return {"induced_parallax_px": round(float(np.linalg.norm(q1 - q0)), 2)}

"""Fixed-side facade detector, cross-view corroboration/contradiction & cropped §7a silhouette verifier."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import cv2
import numpy as np

from .config import BlockSpec, PolicyThresholds
from .geometry_cache import CameraModel
from .proposals import BlockFit, CutEvidence, ImageProjectionContext, _bilinear_sample


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
            z_samples = np.linspace(max(0.002, cue.z_top_m - 0.85 * block_h_m), cue.z_top_m - 0.002, 7)
            off_d = np.array([-0.010, -0.006, -0.003, 0.0, 0.003, 0.006, 0.010])
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

            # F03 modal/amodal visibility ratio r_i against fixed-side depth image
            d_meas, d_in = _bilinear_sample(self.side.depth_m, uv)
            valid_d = d_in & in_bounds & (d_meas > 0.15) & (d_meas < 1.80)
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
            g_grid = np.where(g_in, g_vals, np.nan).reshape(len(off_d), len(z_samples))
            if np.sum(np.isfinite(g_grid)) < 15:
                continue
            d_grid = np.where(valid_d, d_meas, np.nan).reshape(len(off_d), len(z_samples))
            mid_z_rows = slice(1, len(z_samples) - 1)
            g_face = g_grid[:, mid_z_rows]
            if np.sum(np.isfinite(g_face)) < 10:
                continue
            prof_end = np.nanmedian(g_face, axis=1)
            sh_l = float(prof_end[1]) if np.isfinite(prof_end[1]) else float(prof_end[0])
            sh_r = float(prof_end[-2]) if np.isfinite(prof_end[-2]) else float(prof_end[-1])
            val_c = float(np.nanmin(prof_end[2:5]))
            edge_score = float(min(sh_l, sh_r) - val_c)

            # Check actual depth proximity inside end face (must hit the vertical face, not air!)
            d_mid = d_grid[:, mid_z_rows].ravel()
            ez_mid = exp_z.reshape(len(off_d), len(z_samples))[:, mid_z_rows].ravel()
            ok_mid = np.isfinite(d_mid)
            face_hit = float(np.median(np.abs(d_mid[ok_mid] - ez_mid[ok_mid]))) <= 0.015 if np.any(ok_mid) else False

            # Compare depth step on left vs right flanking shoulders (wait: along oblique face, natural slope
            # between -6 mm and +6 mm along d is cos/sin(angle)*12 mm ~ 6-10 mm! Measure RESIDUAL from linear plane!)
            dl = d_grid[:2, mid_z_rows].ravel(); dl = dl[np.isfinite(dl)]
            dr = d_grid[-2:, mid_z_rows].ravel(); dr = dr[np.isfinite(dr)]
            ez_l = exp_z.reshape(len(off_d), len(z_samples))[:2, mid_z_rows].ravel()[np.isfinite(d_grid[:2, mid_z_rows].ravel())]
            ez_r = exp_z.reshape(len(off_d), len(z_samples))[-2:, mid_z_rows].ravel()[np.isfinite(d_grid[-2:, mid_z_rows].ravel())]
            d_step_mm = float(abs(np.median(dl - ez_l) - np.median(dr - ez_r)) * 1000.0) if (len(dl) >= 3 and len(dr) >= 3) else 0.0

            corroborated = bool(face_hit and ((edge_score >= self.th.side_rgb_grad_min) or (d_step_mm >= self.th.side_depth_step_min_m * 1000.0)))
            if corroborated:
                return SideCutVerification(
                    "CORROBORATED", round(vis_r, 3), round(edge_score, 2), round(d_step_mm, 2), round(px_span, 2),
                    "visible_end_face_break"
                )
            elif vis_r >= 0.85 and px_span >= 5.0 and edge_score < 4.0 and d_step_mm < 1.2:
                best_vis = SideCutVerification(
                    "CONTRADICTED", round(vis_r, 3), round(edge_score, 2), round(d_step_mm, 2), round(px_span, 2),
                    "smooth_continuous_end_face_contradicts_split"
                )
            else:
                best_vis = SideCutVerification(
                    "INCONCLUSIVE", round(vis_r, 3), round(edge_score, 2), round(d_step_mm, 2), round(px_span, 2),
                    "weak_or_subpixel_end_cue"
                )

        if best_vis is not None:
            return best_vis
        return SideCutVerification("SIDE_UNSEEN", 0.0, 0.0, 0.0, 0.0, "no_end_face_visible_to_side")


def detect_side_facade_slots(
    side: SideViewData | None,
    block: BlockSpec,
    th: PolicyThresholds,
    *,
    layers: int = 3,
    tower_origin_xy: tuple[float, float] = (0.0, 0.0),
    tower_yaw_rad: float = 0.0,
) -> list[SlotOccupancy]:
    """Detect lower-layer tower slot occupancy/emptiness/unknown from fixed-side depth + F03 r_i."""
    if side is None or not side.is_fresh or side.is_frozen:
        return [
            SlotOccupancy(L, s, "UNKNOWN", 0.0, None, 0.0)
            for L in range(1, layers + 1)
            for s in (-1, 0, 1)
        ]

    ctx = side.img_ctx
    cam_pos = side.T_table_cam[:3, 3]
    H_img, W_img = side.depth_m.shape
    c0, s0 = float(np.cos(tower_yaw_rad)), float(np.sin(tower_yaw_rad))
    R_tw = np.array([[c0, -s0], [s0, c0]], dtype=np.float64)

    out: list[SlotOccupancy] = []
    for layer in range(1, layers + 1):
        odd = (layer % 2 == 1)
        z_mid = (layer - 0.5) * block.height_m
        yaw_blk = tower_yaw_rad + (0.0 if odd else np.pi / 2.0)
        ux = np.array([np.cos(yaw_blk), np.sin(yaw_blk), 0.0])
        uy = np.array([-np.sin(yaw_blk), np.cos(yaw_blk), 0.0])
        uz = np.array([0.0, 0.0, 1.0])

        for slot in (-1, 0, 1):
            local_xy = np.array([0.0, slot * block.width_m]) if odd else np.array([slot * block.width_m, 0.0])
            cen_xy = np.asarray(tower_origin_xy) + R_tw @ local_xy
            cen = np.array([cen_xy[0], cen_xy[1], z_mid])

            # Pick the most camera-facing vertical face (+-ux at L/2 or +-uy at W/2)
            face_cands = [
                (ux, block.length_m / 2.0, uy, block.width_m / 2.0),
                (-ux, block.length_m / 2.0, uy, block.width_m / 2.0),
                (uy, block.width_m / 2.0, ux, block.length_m / 2.0),
                (-uy, block.width_m / 2.0, ux, block.length_m / 2.0),
            ]
            # For an interior slot (slot == 0 in a 3-slot row), the tangential sides (+-uy for odd) are
            # blocked by neighbours (+1 / -1), so check its exposed END faces (+-ux) for OBSERVED_EMPTY!
            to_cam = cam_pos - cen
            to_cam_unit = to_cam / max(1e-6, float(np.linalg.norm(to_cam)))
            if slot == 0:
                end_cands = face_cands[:2]  # ends +-ux are exposed at tower face even for centre slot!
                best_face = max(end_cands, key=lambda f: float(np.dot(f[0], to_cam_unit)))
                fnorm, foff, tang, h_tang = best_face
                # At 55 deg elevation (~55 deg grazing angle down into the tunnel), the camera ray enters the empty
                # layer-1 slot at z=7.5 mm and hits the TABLE inside the hollow tunnel ~10-25 mm deeper than the front face!
                slot_pen_thresh_m = 0.008
            else:
                best_face = max(face_cands, key=lambda f: float(np.dot(f[0], to_cam_unit)))
                fnorm, foff, tang, h_tang = best_face
                slot_pen_thresh_m = th.empty_slot_penetration_m
            if float(np.dot(fnorm, to_cam_unit)) < 0.25:
                out.append(SlotOccupancy(layer, slot, "UNKNOWN", 0.0, None, float(np.linalg.norm(to_cam))))
                continue

            # Sample 3x3 interior points on the visible slot face (avoid outer 4 mm chamfer/edge)
            tt, zz = np.meshgrid(np.linspace(-0.4 * h_tang, 0.4 * h_tang, 3), np.linspace(-0.25 * block.height_m, 0.25 * block.height_m, 3))
            pts3 = cen[None, :] + foff * fnorm[None, :] + tt.ravel()[:, None] * tang[None, :] + zz.ravel()[:, None] * uz[None, :]
            cam_pts = pts3 @ ctx.T_cam_table[:3, :3].T + ctx.T_cam_table[:3, 3]
            exp_z = cam_pts[:, 2]
            uv, _ = cv2.projectPoints(pts3, ctx.rvec_ct, ctx.tvec_ct, ctx.K, ctx.D)
            uv = uv.reshape(-1, 2)
            in_b = (exp_z > 0.15) & (uv[:, 0] >= 1) & (uv[:, 0] <= W_img - 2) & (uv[:, 1] >= 1) & (uv[:, 1] <= H_img - 2)
            if not np.any(in_b):
                out.append(SlotOccupancy(layer, slot, "UNKNOWN", 0.0, None, float(np.mean(exp_z))))
                continue
            d_meas, d_in = _bilinear_sample(side.depth_m, uv)
            valid = in_b & d_in & (d_meas > 0.15) & (d_meas < 1.80)
            if np.count_nonzero(valid) < 4:
                out.append(SlotOccupancy(layer, slot, "UNKNOWN", 0.0, None, float(np.mean(exp_z))))
                continue

            deltas = d_meas[valid] - exp_z[valid]  # >0 means measured ray goes PAST slot front (empty!)
            vis_r = float(np.mean(d_meas[valid] >= exp_z[valid] - 0.015))
            med_delta_m = float(np.median(deltas))
            med_delta_mm = round(med_delta_m * 1000.0, 2)
            if vis_r < 0.60:
                state = "UNKNOWN"  # occluded by nearer block or hand
            elif med_delta_m >= slot_pen_thresh_m and vis_r >= th.empty_slot_vis_min_r:
                state = "OBSERVED_EMPTY"
            elif abs(med_delta_m) <= th.occupied_tol_m:
                state = "OCCUPIED"
            elif med_delta_m <= -th.displaced_protrusion_m:
                state = "DISPLACED"
            else:
                state = "UNKNOWN"
            out.append(SlotOccupancy(layer, slot, state, round(vis_r, 3), med_delta_mm, round(float(np.mean(exp_z)), 3)))
    return out


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

        for idx, (fit, src) in enumerate(fits):
            was_split = src.startswith("split")
            if not self._should_sample(idx, was_split, epoch_id):
                kept.append((fit, src))
                continue

            # Sample 3x3 inner-core points well inside the top face so ~5 mm cross-camera extrinsic shifts
            # never fall off the block rim onto the table behind; absorb +-1 px (~3 mm) local shift via min-delta.
            c, s = float(np.cos(fit.yaw_rad)), float(np.sin(fit.yaw_rad))
            ux = np.array([c, s, 0.0])
            uy = np.array([-s, c, 0.0])
            hl = max(0.004, block.major_m * 0.25)
            hw = max(0.004, block.minor_m * 0.22)
            aa, bb = np.meshgrid(np.linspace(-hl, hl, 3), np.linspace(-hw, hw, 3))
            cen_top = np.array([fit.xy_m[0], fit.xy_m[1], fit.z_top_m])
            pts3 = cen_top[None, :] + aa.ravel()[:, None] * ux[None, :] + bb.ravel()[:, None] * uy[None, :]
            cam_pts = pts3 @ ctx.T_cam_table[:3, :3].T + ctx.T_cam_table[:3, 3]
            exp_z = cam_pts[:, 2]
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
                    # Closest solid surface in +-1 px tolerance window relative to expected depth
                    deltas_list.append(float(np.min(patch[vp]) - exp_z[k]))
            if len(deltas_list) < 4:
                skipped_unseen += 1
                kept.append((fit, src))
                continue

            checked += 1
            deltas = np.asarray(deltas_list, dtype=np.float64)
            # Occluded if foreground is much closer than cuboid; phantom/contradicted if rays pass far behind cuboid
            vis_r = float(np.mean(deltas >= -0.020))
            if vis_r < self.th.side_vis_min_r:
                skipped_unseen += 1
                kept.append((fit, src))
                continue

            med_behind = float(np.median(deltas))
            if med_behind > self.th.verifier_residual_m:
                # Contradicted (measured surface sits strictly behind proposed cuboid -> phantom!)
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

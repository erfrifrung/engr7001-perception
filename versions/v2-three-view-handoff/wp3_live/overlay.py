"""Annotated live/replay OpenCV HUD overlay showing resolved poses, ambiguous clusters, cuts, and timings."""
from __future__ import annotations

import cv2
import numpy as np

from .config import BlockSpec
from .fusion_tracker import TrackedBlock
from .policies import EpochResult
from .proposals import BlockFit, ImageProjectionContext


def _project_box_outline(
    ctx: ImageProjectionContext,
    fit: BlockFit,
    block: BlockSpec,
) -> np.ndarray | None:
    hl, hw = block.major_m / 2.0, block.minor_m / 2.0
    c, s = float(np.cos(fit.yaw_rad)), float(np.sin(fit.yaw_rad))
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    local_corners = np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]], dtype=np.float64)
    xy = np.asarray(fit.xy_m) + local_corners @ R.T
    pts3 = np.column_stack((xy, np.full(4, fit.z_top_m, dtype=np.float64)))
    uv, _ = cv2.projectPoints(pts3, ctx.rvec_ct, ctx.tvec_ct, ctx.K, ctx.D)
    return np.round(uv.reshape(-1, 2)).astype(np.int32)


def _draw_on_view(
    bgr: np.ndarray,
    ctx: ImageProjectionContext | None,
    epoch: EpochResult,
    tracks: list[TrackedBlock],
    block: BlockSpec,
    title: str,
) -> np.ndarray:
    canvas = bgr.copy()
    H, W = canvas.shape[:2]
    if ctx is not None:
        # 1. Draw resolved block footprints (or framewise poses if tracks empty)
        draw_items = (
            [(tr.block_name, tr.fit, tr.state, tr.source) for tr in tracks]
            if tracks
            else [(f"P{i+1}", p, "MEASURED", s) for i, (p, s) in enumerate(zip(epoch.poses, epoch.pose_sources))]
        )
        for name, fit, state, src in draw_items:
            poly = _project_box_outline(ctx, fit, block)
            if poly is None:
                continue
            col = (50, 220, 90) if state == "MEASURED" else ((0, 165, 255) if state == "DISPLACED" else (0, 220, 220))
            cv2.polylines(canvas, [poly], True, col, 2, cv2.LINE_AA)
            cx, cy = poly.mean(axis=0).astype(int)
            cv2.putText(canvas, f"{name} {fit.yaw_deg_mod:.0f}d", (cx - 28, cy - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, f"{name} {fit.yaw_deg_mod:.0f}d", (cx - 28, cy - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)

        # 2. Draw AMBIGUOUS envelopes cleanly in yellow/amber
        for idx, amb in enumerate(epoch.ambiguous_clusters, start=1):
            p3 = np.array([[amb.xy_m[0], amb.xy_m[1], amb.top_z_m]], dtype=np.float64)
            uv, _ = cv2.projectPoints(p3, ctx.rvec_ct, ctx.tvec_ct, ctx.K, ctx.D)
            u, v = int(round(uv[0, 0, 0])), int(round(uv[0, 0, 1]))
            if 0 <= u < W and 0 <= v < H:
                cv2.rectangle(canvas, (u - 26, v - 18), (u + 26, v + 18), (0, 215, 255), 2)
                lbl = f"AMB~{amb.n_est} ({amb.reason[:18]})"
                cv2.putText(canvas, lbl, (max(4, u - 55), max(16, v - 22)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 215, 255), 1, cv2.LINE_AA)

    # 3. Clean compact HUD header bar
    cv2.rectangle(canvas, (0, 0), (W, 44), (20, 24, 28), -1)
    ms_tot = epoch.stage_timings_ms.get("total_ms", 0.0)
    hud1 = f"{title} | policy={epoch.policy} | mode={epoch.fallback_mode} | {ms_tot:.1f} ms"
    hud2 = f"resolved={len(epoch.poses)} amb={len(epoch.ambiguous_clusters)} crop={epoch.evidence_log.get('crop_frac', 1.0):.2f} slots={len(epoch.slot_occupancies)}"
    cv2.putText(canvas, hud1, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 245, 250), 1, cv2.LINE_AA)
    cv2.putText(canvas, hud2, (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 225, 200), 1, cv2.LINE_AA)
    return canvas


def render_overlay(
    wrist_rgb: np.ndarray | None,
    side_rgb: np.ndarray | None,
    epoch: EpochResult,
    tracks: list[TrackedBlock],
    block: BlockSpec,
    third_rgb: np.ndarray | None = None,
) -> np.ndarray:
    """Render annotated BGR8 image (dual or 3-panel side-by-side when multiple cameras exist, else single view)."""
    panels: list[np.ndarray] = []
    if wrist_rgb is not None:
        wrist_bgr = wrist_rgb[:, :, ::-1]
        panels.append(_draw_on_view(wrist_bgr, epoch.primary_img_ctx if epoch.primary_role == "wrist" else None,
                                    epoch, tracks, block, f"WRIST ({'PRIMARY' if epoch.primary_role == 'wrist' else 'OFFLINE'})"))
    if side_rgb is not None:
        side_bgr = side_rgb[:, :, ::-1]
        side_ctx = epoch.side_img_ctx if epoch.side_img_ctx is not None else (epoch.primary_img_ctx if epoch.primary_role != "wrist" else None)
        panels.append(_draw_on_view(side_bgr, side_ctx, epoch, tracks, block,
                                    f"SIDE ({'VERIFIER' if epoch.primary_role == 'wrist' else 'PRIMARY'})"))
    if third_rgb is not None:
        third_bgr = third_rgb[:, :, ::-1]
        third_status = epoch.evidence_log.get("third_cut_status", "VERIFIER")
        panels.append(_draw_on_view(third_bgr, getattr(epoch, "third_img_ctx", None), epoch, tracks, block,
                                    f"THIRD ({str(third_status).upper()})"))
    if not panels:
        blank = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(blank, f"NO VALID CAMERA STREAM ({epoch.fallback_mode})", (40, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2, cv2.LINE_AA)
        return blank
    if len(panels) == 1:
        return panels[0]
    h_target = min(p.shape[0] for p in panels)
    resized = [
        cv2.resize(p, (int(round(p.shape[1] * h_target / p.shape[0])), h_target)) if p.shape[0] != h_target else p
        for p in panels
    ]
    return np.hstack(resized)

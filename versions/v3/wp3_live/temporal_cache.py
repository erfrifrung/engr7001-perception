"""Stationary-frame depth+RGB skip cache and previous-frame partition warm-start cache."""
from __future__ import annotations

from dataclasses import dataclass, replace
import time
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    from .dwell_gate import DwellStatus
    from .policies import EpochResult, ViewInput


@dataclass(frozen=True)
class WarmStartEntry:
    """Cached winning partition cuts from the previous epoch in table-frame space."""

    centroid_xy: tuple[float, float]
    span_mm: tuple[float, float]
    axis_deg: float
    direction: tuple[float, float]
    perp_dir: tuple[float, float]
    k_blocks: int
    refined_cuts_m: tuple[float, ...]
    size_err: float


def _mat_key(mat: np.ndarray | None) -> tuple[float, ...] | None:
    if mat is None:
        return None
    return tuple(np.round(np.asarray(mat, dtype=np.float64).ravel(), 5).tolist())


def _sample_depth_gray_roi(
    depth_raw: np.ndarray,
    rgb_img: np.ndarray | None,
    bbox: tuple[int, int, int, int],
    depth_scale_m: float,
    min_d: float,
    max_d: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    u0, v0, u1, v1 = bbox
    H, W = depth_raw.shape[:2]
    u0c, v0c = max(0, min(W - 1, u0)), max(0, min(H - 1, v0))
    u1c, v1c = max(u0c + 1, min(W, u1)), max(v0c + 1, min(H, v1))
    step = 4 if W >= 960 else 2
    sub_d = depth_raw[v0c:v1c:step, u0c:u1c:step]
    if np.issubdtype(sub_d.dtype, np.floating):
        d_m = sub_d.astype(np.float32, copy=False)
        valid = np.isfinite(d_m) & (d_m > min_d) & (d_m < max_d)
    else:
        d_m = sub_d.astype(np.float32) * np.float32(depth_scale_m)
        valid = (d_m > min_d) & (d_m < max_d)
    thumb = None
    if rgb_img is not None:
        # Fast integer luma proxy (0.25R + 0.5G + 0.25B) on strided ROI avoids full cvtColor allocation
        sub_r = rgb_img[v0c:v1c:step, u0c:u1c:step]
        thumb = (
            (sub_r[:, :, 0].astype(np.uint16) + 2 * sub_r[:, :, 1].astype(np.uint16) + sub_r[:, :, 2].astype(np.uint16)) >> 2
        ).astype(np.int16)
    return d_m, valid, thumb


def _depth_roi_unchanged(
    d_m: np.ndarray,
    valid: np.ndarray,
    prev_d_m: np.ndarray,
    prev_valid: np.ndarray,
    depth_tol_m: float,
    *,
    step_m_override: float | None = None,
) -> bool:
    if d_m.shape != prev_d_m.shape:
        return False
    val_chg = valid != prev_valid
    if float(np.mean(val_chg)) > 0.015:
        return False
    if int(np.count_nonzero(cv2.boxFilter(val_chg.astype(np.float32), -1, (3, 3)) >= 0.60)) > 32:
        return False
    both = valid & prev_valid
    n_both = int(np.count_nonzero(both))
    if n_both < 25:
        return False
    abs_diff = np.abs(d_m - prev_d_m)
    diff = abs_diff[both]
    k95 = min(n_both - 1, max(0, int(0.95 * (n_both - 1))))
    d_p95 = float(np.partition(diff, k95)[k95])
    if d_p95 > depth_tol_m:
        return False
    # Catch single-block disappearance/slide (15 mm height change << 5% global p95 area!)
    step_m = float(step_m_override) if step_m_override is not None else max(0.008, 2.0 * depth_tol_m)
    step_mask = (both & (abs_diff >= step_m)).astype(np.float32)
    if int(np.count_nonzero(cv2.boxFilter(step_mask, -1, (3, 3)) >= 0.60)) > 32:
        return False
    return True


def _gray_roi_unchanged(thumb: np.ndarray | None, prev_thumb: np.ndarray | None) -> bool:
    if thumb is None or prev_thumb is None or thumb.shape != prev_thumb.shape:
        return False
    adiff = np.abs(thumb - prev_thumb)
    if float(np.mean(adiff)) > 6.0:
        return False
    g_chg = (adiff > 16).astype(np.float32)
    if int(np.count_nonzero(cv2.boxFilter(g_chg, -1, (3, 3)) >= 0.60)) > 16:
        return False
    return True


class EpochTemporalCache:
    """Stateful cache for stationary ROI skip + previous-frame partition warm-starting."""

    def __init__(self) -> None:
        self._last_result: EpochResult | None = None
        self._last_policy: str = ""
        self._last_calib_epoch: str = ""
        self._last_fallback_mode: str = ""
        self._last_primary_role: str = ""
        self._last_pose_frame_mode: str = ""
        self._last_T_p_key: tuple[float, ...] | None = None
        self._last_T_s_key: tuple[float, ...] | None = None
        self._last_T_t_key: tuple[float, ...] | None = None
        self._last_p_bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
        self._last_p_depth: np.ndarray | None = None
        self._last_p_valid: np.ndarray | None = None
        self._last_p_gray: np.ndarray | None = None
        self._last_s_bbox: tuple[int, int, int, int] | None = None
        self._last_s_depth: np.ndarray | None = None
        self._last_s_valid: np.ndarray | None = None
        self._last_s_gray: np.ndarray | None = None
        self._last_t_bbox: tuple[int, int, int, int] | None = None
        self._last_t_depth: np.ndarray | None = None
        self._last_t_valid: np.ndarray | None = None
        self._last_t_gray: np.ndarray | None = None
        self._skips_since_full: int = 0
        self._warm_starts: list[WarmStartEntry] = []
        self._ws_sig: tuple[str, str, str, bool] = ("", "", "", False)

    def reset_skip(self) -> None:
        self._last_result = None
        self._skips_since_full = 0

    def invalidate_warm_starts(self) -> None:
        self._warm_starts.clear()

    def lookup_warm_start(
        self,
        policy: str,
        calib_epoch: str,
        fallback_mode: str,
        centroid_xy: tuple[float, float],
        span_mm: tuple[float, float],
        *,
        third_active: bool = False,
    ) -> WarmStartEntry | None:
        if (policy, calib_epoch, fallback_mode, third_active) != self._ws_sig:
            return None
        best_entry = None
        best_dist = 0.012  # 12 mm centroid gate
        for ent in self._warm_starts:
            dist = float(np.hypot(centroid_xy[0] - ent.centroid_xy[0], centroid_xy[1] - ent.centroid_xy[1]))
            if dist <= best_dist and abs(span_mm[0] - ent.span_mm[0]) <= 10.0 and abs(span_mm[1] - ent.span_mm[1]) <= 10.0:
                best_dist = dist
                best_entry = ent
        return best_entry

    def record_warm_starts(
        self,
        policy: str,
        calib_epoch: str,
        fallback_mode: str,
        entries: list[WarmStartEntry],
        *,
        third_active: bool = False,
    ) -> None:
        self._ws_sig = (policy, calib_epoch, fallback_mode, third_active)
        self._warm_starts = list(entries)

    def try_reuse_stationary_epoch(
        self,
        *,
        primary: ViewInput,
        secondary: ViewInput | None,
        policy: str,
        calib_epoch: str,
        fallback_mode: str,
        dwell: DwellStatus | None,
        max_skips: int,
        depth_tol_m: float,
        min_d_p: float,
        max_d_p: float,
        min_d_s: float,
        max_d_s: float,
        has_external_masks: bool,
        third: ViewInput | None = None,
    ) -> EpochResult | None:
        t0 = time.perf_counter()
        if (
            self._last_result is None
            or self._last_result.status != "ok"
            or has_external_masks
            or dwell is None
            or not dwell.stationary
            or self._skips_since_full >= max_skips
            or policy != self._last_policy
            or calib_epoch != self._last_calib_epoch
            or fallback_mode != self._last_fallback_mode
            or primary.role != self._last_primary_role
            or primary.pose_frame_mode != self._last_pose_frame_mode
            or _mat_key(primary.T_table_cam) != self._last_T_p_key
            or self._last_p_depth is None
            or self._last_p_valid is None
        ):
            return None

        d_p, v_p, g_p = _sample_depth_gray_roi(
            primary.depth_raw, primary.rgb_img, self._last_p_bbox, primary.cam.depth_scale_m, min_d_p, max_d_p,
        )
        if not _depth_roi_unchanged(d_p, v_p, self._last_p_depth, self._last_p_valid, depth_tol_m):
            return None
        if not _gray_roi_unchanged(g_p, self._last_p_gray):
            return None

        if secondary is not None:
            if (
                self._last_s_bbox is None
                or self._last_s_depth is None
                or self._last_s_valid is None
                or _mat_key(secondary.T_table_cam) != self._last_T_s_key
            ):
                return None
            d_s, v_s, g_s = _sample_depth_gray_roi(
                secondary.depth_raw, secondary.rgb_img, self._last_s_bbox, secondary.cam.depth_scale_m, min_d_s, max_d_s,
            )
            z_med = float(np.median(d_s[v_s])) if np.any(v_s) else 1.0
            side_tol_m = max(depth_tol_m + 0.001, min(0.030, 0.024 * (z_med * z_med)))
            side_step_m = max(0.025, 2.2 * side_tol_m)
            if not _depth_roi_unchanged(
                d_s, v_s, self._last_s_depth, self._last_s_valid, side_tol_m, step_m_override=side_step_m,
            ):
                return None
            if secondary.rgb_img is not None and self._last_s_gray is not None:
                if not _gray_roi_unchanged(g_s, self._last_s_gray):
                    return None

        if third is not None:
            if (
                self._last_t_bbox is None
                or self._last_t_depth is None
                or self._last_t_valid is None
                or _mat_key(third.T_table_cam) != self._last_T_t_key
            ):
                return None
            d_t, v_t, g_t = _sample_depth_gray_roi(
                third.depth_raw, third.rgb_img, self._last_t_bbox, third.cam.depth_scale_m, min_d_s, max_d_s,
            )
            z_med_t = float(np.median(d_t[v_t])) if np.any(v_t) else 1.0
            third_tol_m = max(depth_tol_m + 0.001, min(0.030, 0.024 * (z_med_t * z_med_t)))
            third_step_m = max(0.025, 2.2 * third_tol_m)
            if not _depth_roi_unchanged(
                d_t, v_t, self._last_t_depth, self._last_t_valid, third_tol_m, step_m_override=third_step_m,
            ):
                return None
            if third.rgb_img is not None and self._last_t_gray is not None:
                if not _gray_roi_unchanged(g_t, self._last_t_gray):
                    return None
        elif self._last_t_bbox is not None:
            return None

        self._skips_since_full += 1
        dt_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        prev = self._last_result
        ev_log = dict(prev.evidence_log)
        ev_log["temporal_cache_hit"] = True
        ev_log["skip_streak"] = self._skips_since_full
        return replace(
            prev,
            stamp_ns=int(primary.stamp_ns),
            dwell=dwell,
            stage_timings_ms={
                "deproject_roi_ms": 0.0,
                "plane_ms": 0.0,
                "robot_filter_ms": 0.0,
                "side_facade_ms": 0.0,
                "heightmap_ms": 0.0,
                "proposals_and_rgb_ms": 0.0,
                "verifier_ms": 0.0,
                "temporal_cache_ms": dt_ms,
                "total_ms": dt_ms,
            },
            evidence_log=ev_log,
        )

    def store_epoch(
        self,
        *,
        result: EpochResult,
        primary: ViewInput,
        secondary: ViewInput | None,
        policy: str,
        calib_epoch: str,
        p_bbox: tuple[int, int, int, int],
        s_bbox: tuple[int, int, int, int] | None,
        min_d_p: float,
        max_d_p: float,
        min_d_s: float,
        max_d_s: float,
        third: ViewInput | None = None,
        t_bbox: tuple[int, int, int, int] | None = None,
    ) -> None:
        self._skips_since_full = 0
        self._last_result = result
        self._last_policy = policy
        self._last_calib_epoch = calib_epoch
        self._last_fallback_mode = result.fallback_mode
        self._last_primary_role = primary.role
        self._last_pose_frame_mode = primary.pose_frame_mode
        self._last_T_p_key = _mat_key(primary.T_table_cam)
        self._last_p_bbox = p_bbox
        d_p, v_p, g_p = _sample_depth_gray_roi(
            primary.depth_raw, primary.rgb_img, p_bbox, primary.cam.depth_scale_m, min_d_p, max_d_p,
        )
        self._last_p_depth = d_p
        self._last_p_valid = v_p
        self._last_p_gray = g_p

        if secondary is not None and s_bbox is not None:
            self._last_T_s_key = _mat_key(secondary.T_table_cam)
            self._last_s_bbox = s_bbox
            d_s, v_s, g_s = _sample_depth_gray_roi(
                secondary.depth_raw, secondary.rgb_img, s_bbox, secondary.cam.depth_scale_m, min_d_s, max_d_s,
            )
            self._last_s_depth = d_s
            self._last_s_valid = v_s
            self._last_s_gray = g_s
        else:
            self._last_T_s_key = None
            self._last_s_bbox = None
            self._last_s_depth = None
            self._last_s_valid = None
            self._last_s_gray = None

        if third is not None and t_bbox is not None:
            self._last_T_t_key = _mat_key(third.T_table_cam)
            self._last_t_bbox = t_bbox
            d_t, v_t, g_t = _sample_depth_gray_roi(
                third.depth_raw, third.rgb_img, t_bbox, third.cam.depth_scale_m, min_d_s, max_d_s,
            )
            self._last_t_depth = d_t
            self._last_t_valid = v_t
            self._last_t_gray = g_t
        else:
            self._last_T_t_key = None
            self._last_t_bbox = None
            self._last_t_depth = None
            self._last_t_valid = None
            self._last_t_gray = None

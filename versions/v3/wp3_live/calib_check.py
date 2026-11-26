"""V3.2 calibration checks (fault #10, observed live on 26 Nov: every camera had moved 99-274 mm while the static TF
publisher still served the 19 Nov calibration, and WP3 fused the views anyway).

* ``transform_delta``: translation / rotation difference between two 4x4 transforms.
* ``TablePlaneCheck``: per camera, does the table plane the depth image actually sees agree with the plane its
  camera->table transform predicts? Catches any tilt, height or distance error. Blind to a pure rotation about the
  table normal or a slide along the table (the plane is unchanged): the session-start board check
  (scripts/check_calibration_drift.py) covers those.
* ``CalibrationMonitor``: rate-limited checks with hysteresis; a camera in ``mismatch`` must not be fused.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .geometry_cache import fit_table_plane_fast


def transform_delta(T_a: np.ndarray, T_b: np.ndarray) -> tuple[float, float]:
    """(|dt| mm, rotation angle deg) between two 4x4 rigid transforms."""
    A = np.asarray(T_a, dtype=np.float64).reshape(4, 4)
    B = np.asarray(T_b, dtype=np.float64).reshape(4, 4)
    dt_mm = float(np.linalg.norm(A[:3, 3] - B[:3, 3]) * 1000.0)
    R = A[:3, :3].T @ B[:3, :3]
    c = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    return dt_mm, float(np.degrees(np.arccos(c)))


def compare_board_fits(reference: dict[str, Any], current: dict[str, Any], max_mm: float = 5.0,
                       max_deg: float = 1.0) -> dict[str, Any]:
    """Compare two register_board.py fits ({"transforms": {link: 4x4}}): the one being published vs a fresh one.

    Per camera link: drift of jenga_table->link. Pairwise: drift of link->link (moves of the rig relative to itself;
    a pure board move shifts every camera but leaves these unchanged). Any link over max_mm / max_deg fails.
    """
    ref = {k: np.asarray(v, dtype=np.float64) for k, v in reference.get("transforms", {}).items()}
    cur = {k: np.asarray(v, dtype=np.float64) for k, v in current.get("transforms", {}).items()}
    links = sorted(set(ref) & set(cur))
    per_link, pairwise, failed = {}, {}, []
    for k in links:
        dt, dr = transform_delta(ref[k], cur[k])
        ok = dt <= max_mm and dr <= max_deg
        per_link[k] = {"dt_mm": round(dt, 2), "dr_deg": round(dr, 3), "ok": ok}
        if not ok:
            failed.append(k)
    for i, a in enumerate(links):
        for b in links[i + 1:]:
            rel_ref = np.linalg.inv(ref[a]) @ ref[b]
            rel_cur = np.linalg.inv(cur[a]) @ cur[b]
            dt, dr = transform_delta(rel_ref, rel_cur)
            pairwise[f"{a}->{b}"] = {"dt_mm": round(dt, 2), "dr_deg": round(dr, 3)}
    missing = sorted(set(ref) ^ set(cur))
    return {"passed": bool(links) and not failed and not missing, "links": per_link, "pairwise": pairwise,
            "failed": failed, "missing": missing, "max_mm": max_mm, "max_deg": max_deg}


_RNG = np.random.default_rng(7)
_TRIPLES = _RNG.integers(0, 1 << 30, size=(160, 3))


@dataclass
class PlaneCheckResult:
    status: str                      # "ok" | "suspect" (reported, still fused) | "mismatch" | "unverified"
    angle_deg: float = float("nan")
    offset_mm: float = float("nan")
    support: float = float("nan")
    n_points: int = 0
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        r = lambda v: None if not np.isfinite(v) else round(float(v), 2)  # noqa: E731
        return {"status": self.status, "plane_angle_deg": r(self.angle_deg), "plane_offset_mm": r(self.offset_mm),
                "plane_support": r(self.support), "n_points": self.n_points, "reason": self.reason}


@dataclass
class TablePlaneCheck:
    """Thresholds come from real data, not only synthetic tests. On the 12 Nov bag the tripod's depth plane is tilted
    4.6-4.9 deg against its board calibration (yet puts the table at +1 mm inside the workspace): the oblique-camera
    depth bias also seen on 26 Nov (board-normal error -0.6 .. +9.2 mm across the board). That must not stop the view
    being fused, so 2.5-6 deg is only "suspect"; the 26 Nov moves gave 11 deg and more. The offset is measured at the
    workspace reference point, not at the camera, where a small tilt is multiplied by the viewing distance."""

    max_angle_deg: float = 6.0       # reject above this
    warn_angle_deg: float = 2.5      # report "suspect" above this
    max_offset_mm: float = 15.0      # table height error at the workspace reference point: reject above this
    warn_offset_mm: float = 6.0
    slab_m: float = 0.080            # points predicted within +-80 mm of the table take part in the fit
    stride: int = 8
    min_points: int = 400
    min_support: float = 0.30

    def check(self, depth_raw: np.ndarray, rays_f32: np.ndarray, depth_scale_m: float, T_table_cam: np.ndarray,
              min_depth_m: float = 0.15, max_depth_m: float = 2.0,
              ref_xyz_table: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> PlaneCheckResult:
        d = np.asarray(depth_raw)[:: self.stride, :: self.stride]
        rays = rays_f32[:: self.stride, :: self.stride].reshape(-1, 3).astype(np.float64)
        z = d.reshape(-1).astype(np.float64)
        if np.issubdtype(d.dtype, np.integer):
            z = z * float(depth_scale_m)
        ok = np.isfinite(z) & (z >= min_depth_m) & (z <= max_depth_m)
        pts = rays[ok] * z[ok, None]
        T = np.asarray(T_table_cam, dtype=np.float64).reshape(4, 4)
        n_pred = T[2, :3].copy()                  # table "up" in the camera frame
        d_pred = float(T[2, 3])                   # camera height above the table
        h = pts @ n_pred + d_pred
        pts = pts[np.abs(h) <= self.slab_m]
        if len(pts) < self.min_points:
            return PlaneCheckResult("unverified", n_points=int(len(pts)), reason="too_few_points_near_predicted_table")
        fit = fit_table_plane_fast(pts, _TRIPLES, tol=0.004, min_support=self.min_support)
        if fit is None:
            return PlaneCheckResult("unverified", n_points=int(len(pts)), reason="no_dominant_plane")
        n_fit, d_fit, sup, _ = fit
        n_fit = np.asarray(n_fit, dtype=np.float64)
        if float(n_fit @ n_pred) < 0.0:          # same orientation (towards the camera)
            n_fit, d_fit = -n_fit, -float(d_fit)
        ang = float(np.degrees(np.arccos(np.clip(float(n_fit @ n_pred) / np.linalg.norm(n_fit), -1.0, 1.0))))
        # table height error at the workspace reference point (the calibration's table point there, in the camera
        # frame, against the plane the depth image actually shows)
        R, t = T[:3, :3], T[:3, 3]
        ref_cam = R.T @ (np.asarray(ref_xyz_table, dtype=np.float64) - t)
        off = abs(float(n_fit @ ref_cam) / float(np.linalg.norm(n_fit)) + float(d_fit)) * 1000.0
        if ang > self.max_angle_deg or off > self.max_offset_mm:
            status, reason = "mismatch", "plane_disagrees_with_calibration"
        elif ang > self.warn_angle_deg or off > self.warn_offset_mm:
            status, reason = "suspect", "plane_tilt_or_offset_above_warning"
        else:
            status, reason = "ok", ""
        return PlaneCheckResult(status, ang, off, float(sup), int(len(pts)), reason)


@dataclass
class _RoleState:
    status: str = "unverified"
    bad_run: int = 0
    good_run: int = 0
    last_check_s: float = -np.inf
    last: PlaneCheckResult | None = None


@dataclass
class CalibrationMonitor:
    """Rate-limited table-plane checks with hysteresis. ``mismatch`` after ``bad_to_trip`` consecutive failures,
    back to ``ok`` only after ``good_to_clear`` consecutive passes. ``unverified`` checks change nothing."""

    period_s: float = 1.0
    bad_to_trip: int = 2
    good_to_clear: int = 3
    plane: TablePlaneCheck = field(default_factory=TablePlaneCheck)
    _roles: dict[str, _RoleState] = field(default_factory=dict)

    def reset(self) -> None:
        self._roles.clear()

    def status(self, role: str) -> str:
        return self._roles.get(role, _RoleState()).status

    def observe(self, role: str, now_s: float, depth_raw, rays_f32, depth_scale_m: float, T_table_cam,
                ref_xyz_table: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> str:
        """Returns "ok", "suspect" (fused, reported), "mismatch" (do not fuse) or "unverified"."""
        st = self._roles.setdefault(role, _RoleState())
        if T_table_cam is None or (now_s - st.last_check_s) < self.period_s:
            return st.status
        st.last_check_s = now_s
        res = self.plane.check(depth_raw, rays_f32, depth_scale_m, T_table_cam, ref_xyz_table=ref_xyz_table)
        st.last = res
        if res.status == "mismatch":
            st.bad_run, st.good_run = st.bad_run + 1, 0
            if st.bad_run >= self.bad_to_trip:
                st.status = "mismatch"
        elif res.status in ("ok", "suspect"):
            st.good_run, st.bad_run = st.good_run + 1, 0
            if st.status != "mismatch" or st.good_run >= self.good_to_clear:
                st.status = res.status
        return st.status

    def diagnostics(self, role: str) -> dict[str, Any]:
        st = self._roles.get(role)
        if st is None:
            return {"status": "unverified"}
        out = {"status": st.status}
        if st.last is not None:
            out["last_check"] = st.last.as_dict()
        return out

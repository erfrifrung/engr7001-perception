"""FK-first wrist dwell qualification, debiased IMU corroboration, and SO(3)-log time sync."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(frozen=True)
class DwellStatus:
    stationary: bool
    fk_qualified: bool
    imu_corroborated: bool
    sharpness_ok: bool
    max_joint_speed_rad_s: float | None
    imu_rms_deg_s: float | None
    imu_peak_deg_s: float | None
    laplacian_var: float | None
    reason: str


class WristDwellGate:
    """Exposure-bracketing FK stillness (primary) + debiased IMU RMS + sharpness (secondary)."""

    def __init__(
        self,
        *,
        window_s: float = 0.150,
        max_joint_speed_rad_s: float = 0.010,
        max_imu_rms_deg_s: float = 1.0,
        max_imu_peak_deg_s: float = 3.0,
        min_laplacian_var: float = 18.0,
        require_fk: bool = True,
    ) -> None:
        self.window_s = float(window_s)
        self.max_joint_speed = float(max_joint_speed_rad_s)
        self.max_imu_rms = float(max_imu_rms_deg_s)
        self.max_imu_peak = float(max_imu_peak_deg_s)
        self.min_laplacian_var = float(min_laplacian_var)
        self.require_fk = bool(require_fk)
        self._joints: deque[tuple[float, np.ndarray]] = deque(maxlen=256)
        self._gyro: deque[tuple[float, np.ndarray]] = deque(maxlen=512)
        self._bias_samples: list[np.ndarray] = []
        self._gyro_bias: np.ndarray = np.zeros(3, dtype=np.float64)

    def push_joint_state(self, stamp_s: float, positions_rad: np.ndarray, velocities_rad_s: np.ndarray | None = None) -> None:
        q = np.asarray(positions_rad, dtype=np.float64).ravel()
        self._joints.append((float(stamp_s), q))

    def push_imu_gyro(self, stamp_s: float, omega_rad_s: np.ndarray) -> None:
        w = np.asarray(omega_rad_s, dtype=np.float64).ravel()
        self._gyro.append((float(stamp_s), w))
        # Estimate initial DC bias from low-rate samples
        if len(self._bias_samples) < 200 and np.linalg.norm(w) < np.deg2rad(2.0):
            self._bias_samples.append(w)
            if len(self._bias_samples) >= 20:
                self._gyro_bias = np.median(self._bias_samples, axis=0)

    def evaluate(self, frame_stamp_s: float, gray_roi: np.ndarray | None = None) -> DwellStatus:
        t0 = frame_stamp_s - self.window_s
        # 1. FK / joint-state primary check over [t0, frame_stamp_s]
        js = [(t, q) for (t, q) in self._joints if t0 <= t <= frame_stamp_s + 0.03]
        fk_qualified = False
        max_qd: float | None = None
        if len(js) >= 2:
            speeds = []
            for (ta, qa), (tb, qb) in zip(js[:-1], js[1:]):
                dt = tb - ta
                if dt > 1e-4 and qa.shape == qb.shape:
                    speeds.append(float(np.max(np.abs(qb - qa)) / dt))
            if speeds:
                max_qd = max(speeds)
                fk_qualified = max_qd <= self.max_joint_speed

        # 2. Secondary debiased IMU RMS & peak
        gs = [np.rad2deg(w - self._gyro_bias) for (t, w) in self._gyro if t0 <= t <= frame_stamp_s + 0.03]
        imu_rms: float | None = None
        imu_peak: float | None = None
        imu_ok = True
        if len(gs) >= 3:
            mags = np.linalg.norm(np.asarray(gs), axis=1)
            imu_rms = float(np.sqrt(np.mean(mags ** 2)))
            imu_peak = float(np.max(mags))
            imu_ok = (imu_rms <= self.max_imu_rms) and (imu_peak <= self.max_imu_peak)

        # 3. Secondary image sharpness check (central 50% crop with CV_32F for ~4x speedup)
        lap_var: float | None = None
        sharp_ok = True
        if gray_roi is not None and gray_roi.size >= 100:
            h, w = gray_roi.shape[:2]
            roi = gray_roi[h // 4 : (3 * h) // 4, w // 4 : (3 * w) // 4] if (h >= 120 and w >= 120) else gray_roi
            lap_var = float(cv2.Laplacian(roi, cv2.CV_32F).var())
            sharp_ok = lap_var >= self.min_laplacian_var

        if self.require_fk:
            if not fk_qualified:
                reason = "fk_missing_or_moving"
                stationary = False
            elif not imu_ok:
                reason = "imu_vibration_exceeded"
                stationary = False
            elif not sharp_ok:
                reason = "motion_blur_low_sharpness"
                stationary = False
            else:
                reason = "fk_and_imu_verified"
                stationary = True
        else:
            # Stationary pre-registered session mode (no FK topic present)
            stationary = imu_ok and sharp_ok
            reason = "stationary_session_imu_ok" if stationary else "imu_or_blur_rejected"

        return DwellStatus(
            stationary=stationary,
            fk_qualified=fk_qualified,
            imu_corroborated=imu_ok,
            sharpness_ok=sharp_ok,
            max_joint_speed_rad_s=max_qd,
            imu_rms_deg_s=imu_rms,
            imu_peak_deg_s=imu_peak,
            laplacian_var=lap_var,
            reason=reason,
        )


def estimate_so3_clock_offset(
    gyro_t: np.ndarray,
    gyro_omega: np.ndarray,
    fk_t: np.ndarray,
    fk_R: np.ndarray,
    search_range_s: float = 0.25,
    step_s: float = 0.002,
) -> tuple[float, float]:
    """Cross-correlate angular-velocity magnitudes (Kalibr/SO(3)-log) to estimate clock offset."""
    from scipy.spatial.transform import Rotation

    if len(fk_t) < 5 or len(gyro_t) < 10:
        return 0.0, 0.0
    dt = np.diff(fk_t)
    fk_mid_t = 0.5 * (fk_t[:-1] + fk_t[1:])
    fk_speed = np.empty(len(dt), dtype=np.float64)
    for i in range(len(dt)):
        if dt[i] <= 1e-4:
            fk_speed[i] = 0.0
        else:
            dR = fk_R[i].T @ fk_R[i + 1]
            fk_speed[i] = float(np.linalg.norm(Rotation.from_matrix(dR).as_rotvec()) / dt[i])
    gyro_speed = np.linalg.norm(gyro_omega, axis=1)
    best_off, best_corr = 0.0, -1.0
    offsets = np.arange(-search_range_s, search_range_s + 0.5 * step_s, step_s)
    fk_z = fk_speed - fk_speed.mean()
    fk_std = float(np.linalg.norm(fk_z))
    if fk_std < 1e-6:
        return 0.0, 0.0
    for off in offsets:
        g_interp = np.interp(fk_mid_t, gyro_t + off, gyro_speed)
        g_z = g_interp - g_interp.mean()
        denom = fk_std * float(np.linalg.norm(g_z))
        if denom > 1e-9:
            corr = float(np.dot(fk_z, g_z) / denom)
            if corr > best_corr:
                best_corr = corr
                best_off = float(off)
    return best_off, best_corr

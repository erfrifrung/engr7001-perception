"""Cached camera ray grids, 2D pre-deprojection workspace ROI cropping, TF caching & fast plane fit."""
from __future__ import annotations

from dataclasses import dataclass, field
import cv2
import numpy as np


def _point_to_segment_dist(pts: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    ab_len2 = float(np.dot(ab, ab))
    if ab_len2 < 1e-12:
        return np.linalg.norm(pts - a, axis=1)
    t = np.clip(((pts - a) @ ab) / ab_len2, 0.0, 1.0)
    proj = a[None, :] + t[:, None] * ab[None, :]
    return np.linalg.norm(pts - proj, axis=1)


@dataclass
class CameraModel:
    """Intrinsics, optical-frame checks, depth scale, and precomputed float32 ray grid."""

    camera_id: str
    width: int
    height: int
    K: np.ndarray
    D: np.ndarray
    frame_id: str = ""
    depth_scale_m: float = 0.001
    rays_f32: np.ndarray = field(init=False, repr=False)
    K_f64: np.ndarray = field(init=False, repr=False)
    D_f64: np.ndarray = field(init=False, repr=False)
    optical_frame_ok: bool = field(init=False)

    def __post_init__(self) -> None:
        self.K_f64 = np.asarray(self.K, dtype=np.float64).reshape(3, 3)
        self.D_f64 = np.asarray(self.D, dtype=np.float64).ravel()
        if self.D_f64.size == 0:
            self.D_f64 = np.zeros(5, dtype=np.float64)
        self.optical_frame_ok = (not self.frame_id) or self.frame_id.endswith("_optical_frame")
        if self.depth_scale_m <= 0 or not np.isfinite(self.depth_scale_m):
            raise ValueError(f"Invalid depth_scale_m={self.depth_scale_m}")
        pix = np.indices((self.height, self.width), dtype=np.float32)[::-1].transpose(1, 2, 0)
        xy = cv2.undistortPoints(pix.reshape(-1, 1, 2), self.K_f64, self.D_f64).reshape(
            self.height, self.width, 2
        ).astype(np.float32)
        ones = np.ones((self.height, self.width, 1), dtype=np.float32)
        self.rays_f32 = np.ascontiguousarray(np.concatenate((xy, ones), axis=2))


class CameraModelCache:
    """Cache CameraModel objects keyed by (camera_id, w, h, K, D, frame_id, depth_scale)."""

    def __init__(self) -> None:
        self._cache: dict[tuple, CameraModel] = {}

    def get(
        self,
        camera_id: str,
        width: int,
        height: int,
        K: np.ndarray,
        D: np.ndarray,
        frame_id: str = "",
        depth_scale_m: float = 0.001,
    ) -> CameraModel:
        k_tuple = tuple(np.round(np.asarray(K, dtype=float).ravel(), 6))
        d_tuple = tuple(np.round(np.asarray(D, dtype=float).ravel(), 6))
        key = (camera_id, int(width), int(height), k_tuple, d_tuple, frame_id, round(float(depth_scale_m), 7))
        model = self._cache.get(key)
        if model is None:
            model = CameraModel(camera_id, int(width), int(height), K, D, frame_id, float(depth_scale_m))
            self._cache[key] = model
        return model


class StaticTFCache:
    """Epoch-keyed 4x4 transform cache; invalidates automatically when calib_epoch changes."""

    def __init__(self) -> None:
        self._epoch: str | None = None
        self._mats: dict[tuple[str, str], np.ndarray] = {}

    def set_epoch(self, epoch: str) -> None:
        if epoch != self._epoch:
            self._epoch = epoch
            self._mats.clear()

    def get(self, epoch: str, parent: str, child: str) -> np.ndarray | None:
        self.set_epoch(epoch)
        return self._mats.get((parent, child))

    def put(self, epoch: str, parent: str, child: str, mat: np.ndarray) -> np.ndarray:
        self.set_epoch(epoch)
        m = np.asarray(mat, dtype=np.float64).reshape(4, 4)
        self._mats[(parent, child)] = m
        return m


def project_workspace_roi_2d(
    cam: CameraModel,
    T_table_cam: np.ndarray,
    roi_xyz_min: tuple[float, float, float],
    roi_xyz_max: tuple[float, float, float],
    pad_px: int = 14,
) -> tuple[int, int, int, int] | None:
    """Project the 8 corners of the workspace box (including max tower height) into clamped 2D bounds."""
    x0, y0, z0 = roi_xyz_min
    x1, y1, z1 = roi_xyz_max
    corners = np.array(
        [
            [x, y, z]
            for x in (x0, x1)
            for y in (y0, y1)
            for z in (z0, z1)
        ],
        dtype=np.float64,
    )
    T_cam_table = np.linalg.inv(np.asarray(T_table_cam, dtype=np.float64))
    cam_pts = corners @ T_cam_table[:3, :3].T + T_cam_table[:3, 3]
    in_front = cam_pts[:, 2] > 0.05
    if not np.any(in_front):
        return None
    if not np.all(in_front):
        # If any corner sits behind near plane, fall back to full image safely
        return (0, 0, cam.width, cam.height)
    rvec, _ = cv2.Rodrigues(T_cam_table[:3, :3])
    tvec = T_cam_table[:3, 3].reshape(3, 1)
    uv, _ = cv2.projectPoints(corners, rvec, tvec, cam.K_f64, cam.D_f64)
    uv = uv.reshape(-1, 2)
    u0 = max(0, int(np.floor(uv[:, 0].min())) - pad_px)
    v0 = max(0, int(np.floor(uv[:, 1].min())) - pad_px)
    u1 = min(cam.width, int(np.ceil(uv[:, 0].max())) + pad_px + 1)
    v1 = min(cam.height, int(np.ceil(uv[:, 1].max())) + pad_px + 1)
    if u1 - u0 < 16 or v1 - v0 < 16:
        return None
    return (u0, v0, u1, v1)


def deproject_roi_points(
    depth_raw: np.ndarray,
    rgb_img: np.ndarray,
    cam: CameraModel,
    T_table_cam: np.ndarray | None,
    roi_xyz_min: tuple[float, float, float],
    roi_xyz_max: tuple[float, float, float],
    *,
    min_depth_m: float = 0.16,
    max_depth_m: float = 1.60,
    pad_px: int = 14,
    use_roi_precrop: bool = True,
    extra_frac_crop: tuple[float, float, float, float] | None = None,
    depth_stride: int = 0,
) -> dict:
    """Deproject ONLY the workspace 2D ROI pixels into camera & table float32 coordinates."""
    H, W = depth_raw.shape[:2]
    if H != cam.height or W != cam.width:
        raise ValueError(f"Depth shape {(W, H)} != CameraModel {(cam.width, cam.height)}")

    stride = int(depth_stride) if depth_stride > 0 else (2 if W >= 960 else 1)

    if use_roi_precrop and T_table_cam is not None:
        bbox = project_workspace_roi_2d(cam, T_table_cam, roi_xyz_min, roi_xyz_max, pad_px=pad_px)
        if bbox is None:
            return {
                "bbox_2d": (0, 0, 0, 0),
                "pts_cam_f32": np.zeros((0, 3), dtype=np.float32),
                "pts_tbl_f32": np.zeros((0, 3), dtype=np.float32),
                "rgb_u8": np.zeros((0, 3), dtype=np.uint8),
                "crop_frac": 0.0,
            }
        u0, v0, u1, v1 = bbox
    elif extra_frac_crop is not None:
        fx0, fy0, fx1, fy1 = extra_frac_crop
        u0, v0 = max(0, round(fx0 * W)), max(0, round(fy0 * H))
        u1, v1 = min(W, round(fx1 * W)), min(H, round(fy1 * H))
    else:
        u0, v0, u1, v1 = 0, 0, W, H

    su0 = (u0 // stride) * stride
    sv0 = (v0 // stride) * stride
    sub_d = depth_raw[sv0:v1:stride, su0:u1:stride]
    if np.issubdtype(sub_d.dtype, np.floating):
        d_m = sub_d.astype(np.float32, copy=False)
    else:
        d_m = sub_d.astype(np.float32) * np.float32(cam.depth_scale_m)

    valid = np.isfinite(d_m) & (d_m > min_depth_m) & (d_m < max_depth_m)
    if not np.any(valid):
        return {
            "bbox_2d": (u0, v0, u1, v1),
            "pts_cam_f32": np.zeros((0, 3), dtype=np.float32),
            "pts_tbl_f32": np.zeros((0, 3), dtype=np.float32),
            "rgb_u8": np.zeros((0, 3), dtype=np.uint8),
            "crop_frac": float((u1 - u0) * (v1 - v0)) / max(1, W * H),
        }

    rays_sub = cam.rays_f32[sv0:v1:stride, su0:u1:stride]
    pts_cam = rays_sub[valid] * d_m[valid, None]
    rgb_sub = rgb_img[sv0:v1:stride, su0:u1:stride]
    rgb_pts = np.ascontiguousarray(rgb_sub[valid], dtype=np.uint8)

    if T_table_cam is not None:
        R = np.asarray(T_table_cam[:3, :3], dtype=np.float32)
        t = np.asarray(T_table_cam[:3, 3], dtype=np.float32)
        pts_tbl = pts_cam @ R.T + t
        rmin = np.asarray(roi_xyz_min, dtype=np.float32)
        rmax = np.asarray(roi_xyz_max, dtype=np.float32)
        in_box = (
            (pts_tbl[:, 0] >= rmin[0]) & (pts_tbl[:, 0] <= rmax[0])
            & (pts_tbl[:, 1] >= rmin[1]) & (pts_tbl[:, 1] <= rmax[1])
            & (pts_tbl[:, 2] >= rmin[2]) & (pts_tbl[:, 2] <= rmax[2])
        )
        pts_cam = pts_cam[in_box]
        pts_tbl = pts_tbl[in_box]
        rgb_pts = rgb_pts[in_box]
    else:
        pts_tbl = pts_cam

    return {
        "bbox_2d": (u0, v0, u1, v1),
        "pts_cam_f32": np.ascontiguousarray(pts_cam, dtype=np.float32),
        "pts_tbl_f32": np.ascontiguousarray(pts_tbl, dtype=np.float32),
        "rgb_u8": rgb_pts,
        "crop_frac": float((u1 - u0) * (v1 - v0)) / max(1, W * H),
    }


def fit_table_plane_fast(
    points: np.ndarray,
    rng_triples: np.ndarray,
    tol: float = 0.004,
    min_support: float = 0.55,
) -> tuple[np.ndarray, float, float, float] | None:
    """Fast RANSAC + SVD plane fit in camera frame (normal[2] < 0 toward camera)."""
    if len(points) < 120:
        return None
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) > 4000:
        step = max(1, len(pts) // 2500)
        sample = pts[::step]
    else:
        sample = pts
    a, b, c = sample[rng_triples % len(sample)].transpose(1, 0, 2)
    normals = np.cross(b - a, c - a)
    lengths = np.linalg.norm(normals, axis=1)
    ok = lengths > 1e-5
    if not np.any(ok):
        return None
    normals = normals[ok] / lengths[ok, None]
    offsets = -np.einsum("ij,ij->i", normals, a[ok])
    dist = np.abs(sample @ normals.T + offsets)
    best = int(np.argmax(np.sum(dist < tol, axis=0)))
    inl = sample[dist[:, best] < tol]
    if len(inl) / len(sample) < min_support:
        return None
    cen = inl.mean(axis=0)
    _, _, axes = np.linalg.svd(inl - cen, full_matrices=False)
    n = axes[-1]
    if n[2] > 0:
        n = -n
    d = float(-cen @ n)
    res = np.abs(sample @ n + d)
    mask = res < tol
    sup = float(np.mean(mask))
    if sup < min_support:
        return None
    return n, d, sup, float(1000.0 * np.median(res[mask]))


def plane_to_local_table_tf(normal_cam: np.ndarray, offset_cam: float) -> np.ndarray:
    """Metric Z-up local tabletop frame from camera-frame plane normal/offset."""
    z_up = np.asarray(normal_cam, dtype=np.float64)
    z_up /= np.linalg.norm(z_up)
    x_hint = np.array([1.0, 0.0, 0.0])
    x_axis = x_hint - float(x_hint @ z_up) * z_up
    if np.linalg.norm(x_axis) < 1e-6:
        x_hint = np.array([0.0, 1.0, 0.0])
        x_axis = x_hint - float(x_hint @ z_up) * z_up
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_up, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    t_hit = -offset_cam / normal_cam[2]
    p0_cam = np.array([0.0, 0.0, t_hit])
    R_tc = np.vstack((x_axis, y_axis, z_up))
    t_tc = -R_tc @ p0_cam
    T_tc = np.eye(4, dtype=np.float64)
    T_tc[:3, :3] = R_tc
    T_tc[:3, 3] = t_tc
    return T_tc


def check_frozen_plane(
    points_cam: np.ndarray,
    ref_normal: np.ndarray,
    ref_offset: float,
    tol: float = 0.004,
    min_support_frac: float = 0.20,
) -> tuple[bool, float, float]:
    """Validate frozen table plane on perimeter/table points (works on tall towers!)."""
    if len(points_cam) < 80:
        return False, 0.0, float("nan")
    dist = np.abs(points_cam @ ref_normal + ref_offset)
    sup = float(np.mean(dist < tol))
    med_mm = float(1000.0 * np.median(dist[dist < tol])) if sup > 0 else float("nan")
    return sup >= min_support_frac, sup, med_mm


def filter_robot_links(
    points: np.ndarray,
    link_transforms: dict[str, tuple[list[float], list[float]]],
    rgb: np.ndarray | None = None,
    *,
    tcp_offset_z_m: float = 0.242,
    link_radius_m: float = 0.072,
    finger_radius_m: float = 0.020,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    """Narrower link + 0.242 m TCP + 20 mm finger filter that avoids erasing target blocks."""
    if not len(points) or not link_transforms:
        return points, rgb, 0
    from scipy.spatial.transform import Rotation

    pts = np.asarray(points, dtype=np.float64)
    keep = np.ones(len(pts), dtype=bool)

    def pos(name: str) -> np.ndarray | None:
        return np.asarray(link_transforms[name][0], dtype=np.float64) if name in link_transforms else None

    def rot(name: str) -> np.ndarray:
        return Rotation.from_quat(link_transforms[name][1]).as_matrix() if name in link_transforms else np.eye(3)

    pairs = [("link_base", "link1"), ("link1", "link2"), ("link2", "link3"), ("link3", "link4"), ("link4", "link5"), ("link5", "link6")]
    capsules = []
    for a_name, b_name in pairs:
        pa, pb = pos(a_name), pos(b_name)
        if pa is not None and pb is not None:
            capsules.append((pa, pb, link_radius_m))
    p6 = pos("link6")
    ptcp = pos("link_tcp")
    if p6 is not None:
        R6 = rot("link6")
        p_tcp_eff = ptcp if ptcp is not None else (p6 + R6 @ np.array([0.0, 0.0, tcp_offset_z_m]))
        p_wrist_cam = p6 + R6 @ np.array([-0.065, 0.0, 0.070])
        # Gripper body upper half + narrow finger tips so table blocks remain intact on approach
        p_palm = p6 + (p_tcp_eff - p6) * 0.65
        capsules.append((p6, p_palm, 0.055))
        capsules.append((p_palm, p_tcp_eff, finger_radius_m))
        capsules.append((p6, p_wrist_cam, 0.045))

    for a, b, rad in capsules:
        keep &= (_point_to_segment_dist(pts, a, b) > rad)

    removed = int(np.sum(~keep))
    return pts[keep], (rgb[keep] if rgb is not None else None), removed

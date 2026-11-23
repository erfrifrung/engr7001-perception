"""Project jenga_table points into the recorded RGB images using /tf_static and CameraInfo.

The movement bag has no CameraInfo; the baseline (same session, same static TF tree for
the camera links - checked in diagnosis/projection_check.json) supplies the intrinsics.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent))
from bagio import Bag  # noqa: E402


def static_tf(bag: Bag) -> dict:
    tf = {}
    for _, m in bag.messages("/tf_static"):
        for t in m.transforms:
            tr, q = t.transform.translation, t.transform.rotation
            T = np.eye(4)
            T[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            T[:3, 3] = [tr.x, tr.y, tr.z]
            tf[(t.header.frame_id, t.child_frame_id)] = T
    return tf


def chain(tf, parent, child):
    """T_parent_child by walking the static tree from child up to parent."""
    up = {c: p for (p, c) in tf}
    T, node = np.eye(4), child
    path = []
    while node != parent:
        if node not in up:
            raise KeyError(f"No static path {parent} -> {child}")
        path.append((up[node], node)); node = up[node]
    for p, c in reversed(path):
        T = T @ tf[(p, c)]
    return T


class Camera:
    def __init__(self, role: str, tf_bag="movement", info_bag="baseline"):
        tf = static_tf(Bag(tf_bag))
        self.T_table_cam = chain(tf, "jenga_table", f"{role}_color_optical_frame")
        b = Bag(info_bag)
        _, info = next(b.messages(f"/perception/{role}/color/camera_info"))
        self.K = np.array(info.k).reshape(3, 3)
        self.D = np.array(info.d)
        self.size = (info.width, info.height)
        self.role = role

    def project(self, xyz):
        xyz = np.atleast_2d(np.asarray(xyz, float))
        T = np.linalg.inv(self.T_table_cam)
        pc = xyz @ T[:3, :3].T + T[:3, 3]
        uv = pc[:, :2] / pc[:, 2:3]
        return uv @ self.K[:2, :2].T + self.K[:2, 2], pc[:, 2]

    def backproject_to_plane(self, uv, z=0.0):
        """Pixel -> jenga_table point on the horizontal plane z (e.g. block top 0.038)."""
        uv = np.atleast_2d(np.asarray(uv, float))
        rays = np.c_[(uv - self.K[:2, 2]) / np.diag(self.K)[:2], np.ones(len(uv))]
        R, t = self.T_table_cam[:3, :3], self.T_table_cam[:3, 3]
        d = rays @ R.T
        s = (z - t[2]) / d[:, 2]
        return t + d * s[:, None]

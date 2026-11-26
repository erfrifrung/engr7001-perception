"""ROS-free maths and bookkeeping for the WP3 perception -> digital twin link.

Kept separate from the ROS nodes so it can be unit-tested anywhere:  python3 test_wp3_offline.py

  fit_table_transform(table_pts, world_pts, planar=True)
      Finds the pose of WP3's `jenga_table` frame in the twin's `world` frame from 3+ points that are
      known in both frames (e.g. checkerboard corners touched with the fingertip in teach_keyboard.py `v`).

  BlockTracker
      Turns a stream of per-block observations into planning-scene actions (add / move / stale / remove)
      with the same movement tolerances and 180-degree yaw symmetry as perception_bridge.py.
"""
import math

import numpy as np


# ----------------------------------------------------------------------------- rotations
def quat_from_matrix(R):
    """3x3 rotation matrix -> (x, y, z, w)."""
    R = np.asarray(R, dtype=float)
    t = np.trace(R)
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    q /= np.linalg.norm(q)
    if q[3] < 0.0:
        q = -q
    return tuple(float(v) for v in q)


def matrix_from_quat(q):
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rotz_quat(yaw_rad):
    """Quaternion (x, y, z, w) for pure yaw rotation in radians."""
    half = 0.5 * float(yaw_rad)
    return (0.0, 0.0, math.sin(half), math.cos(half))


def yaw_of_quat(q):
    x, y, z, w = q
    return math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def yaw_diff_deg(a, b, symmetry=180.0):
    """Smallest yaw difference, treating a block turned by `symmetry` degrees as the same block."""
    half = symmetry / 2.0
    return abs(((a - b + half) % symmetry) - half)


# ----------------------------------------------------------------------------- table -> world fit
def fit_table_transform(table_pts, world_pts, planar=True):
    """Rigid transform with world_p = R @ table_p + t, fitted by least squares (Kabsch).

    planar=True  : rotation about z only (both frames have z up, the checkerboard lies on the table).
                   Needs >= 2 points; 3-4 spread-out points recommended. More robust to fingertip-height noise.
    planar=False : full 3-D rotation. Needs >= 3 non-collinear points.

    Returns dict(R, t, quat (x,y,z,w), yaw_deg, rms_m, max_m, residuals_m).
    """
    A = np.asarray(table_pts, dtype=float)
    B = np.asarray(world_pts, dtype=float)
    if A.shape != B.shape or A.ndim != 2 or A.shape[1] != 3:
        raise ValueError('need two N x 3 arrays of matching points')
    n = A.shape[0]
    if n < (2 if planar else 3):
        raise ValueError(f'need at least {2 if planar else 3} points, got {n}')
    ca, cb = A.mean(axis=0), B.mean(axis=0)
    a, b = A - ca, B - cb
    if planar:
        # best yaw for the xy parts; z is a pure offset
        num = np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
        den = np.sum(a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1])
        if abs(num) < 1e-12 and abs(den) < 1e-12:
            raise ValueError('points are all in the same place: spread them out over the board')
        th = math.atan2(num, den)
        c, s = math.cos(th), math.sin(th)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    else:
        if np.linalg.matrix_rank(a, tol=1e-6) < 2:
            raise ValueError('points are collinear: pick corners that form a triangle')
        H = a.T @ b
        U, _, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        D = np.diag([1.0, 1.0, d])
        R = Vt.T @ D @ U.T
    t = cb - R @ ca
    res = np.linalg.norm((R @ A.T).T + t - B, axis=1)
    q = quat_from_matrix(R)
    return {
        'R': R, 't': t, 'quat': q, 'yaw_deg': yaw_of_quat(q),
        'rms_m': float(math.sqrt(np.mean(res ** 2))), 'max_m': float(res.max()),
        'residuals_m': [float(r) for r in res],
    }


# ----------------------------------------------------------------------------- block bookkeeping
class BlockTracker:
    """Decides what to do in the planning scene for each observed block.

    observe(key, dims, xyz, quat, now) for every block in a message, then actions(now) returns a list of
      ('add'|'move', key, dims, xyz, quat)   block is new or moved beyond the tolerances
      ('stale', key)                          not seen for stale_s: keep last pose, show it as old
      ('remove', key)                         not seen for remove_s (only if remove_s is set) or deleted
    Tolerances match perception_bridge.py (3 mm, 2 deg, 180-deg yaw symmetry). Note these are an update
    dead-band, not an accuracy requirement: the scene can lag WP3's latest pose by up to move_tol / yaw_tol.
    """

    def __init__(self, move_tol=0.003, yaw_tol=2.0, dim_tol=0.003, stale_s=2.0, remove_s=None):
        self.move_tol, self.yaw_tol, self.dim_tol = move_tol, yaw_tol, dim_tol
        self.stale_s, self.remove_s = stale_s, remove_s
        self.latest = {}      # key -> (dims, xyz, quat, t_seen)
        self.applied = {}     # key -> (dims, xyz, quat)
        self.stale = set()
        self.deleted = set()

    def observe(self, key, dims, xyz, quat, now):
        self.latest[key] = (tuple(dims), tuple(xyz), tuple(quat), now)
        self.deleted.discard(key)

    def delete(self, key):
        self.latest.pop(key, None)
        if key in self.applied:
            self.deleted.add(key)

    def delete_all(self):
        for k in list(self.latest):
            self.delete(k)
        self.deleted |= set(self.applied)

    def retain_only(self, keys):
        """Snapshot semantics: every known block not in `keys` was removed upstream (e.g. a retired WP3 track)."""
        keep = set(keys)
        for k in [k for k in set(self.latest) | set(self.applied) if k not in keep]:
            self.delete(k)

    def snapshot(self):
        """Copy of the bookkeeping, so a planning-scene update MoveIt rejects can be retried instead of lost."""
        return dict(self.latest), dict(self.applied), set(self.stale), set(self.deleted)

    def restore(self, snap):
        self.latest, self.applied, self.stale, self.deleted = dict(snap[0]), dict(snap[1]), set(snap[2]), set(snap[3])

    def _changed(self, key, dims, xyz, quat):
        old = self.applied.get(key)
        if old is None:
            return True
        odims, oxyz, oquat = old
        if max(abs(a - b) for a, b in zip(dims, odims)) > self.dim_tol:
            return True
        if math.dist(xyz, oxyz) > self.move_tol:
            return True
        return yaw_diff_deg(yaw_of_quat(quat), yaw_of_quat(oquat)) > self.yaw_tol

    def actions(self, now, skip=(), hold_removals=()):
        """Planning-scene actions since the last call. Keys in `skip` (objects attached to the gripper) are left
        untouched and stay pending, so a removal or move that happens while the twin owns them is not lost.
        Keys in `hold_removals` (just released by the gripper) keep a pending removal for now, so the camera can
        re-report the block where it was dropped before it is taken out of the scene."""
        skip = set(skip)
        hold = skip | set(hold_removals)
        out = []
        for key in sorted(self.deleted - hold):
            out.append(('remove', key))
            self.applied.pop(key, None)
            self.stale.discard(key)
        self.deleted &= hold
        for key, (dims, xyz, quat, seen) in sorted(self.latest.items()):
            if key in skip:
                continue
            age = now - seen
            if self.remove_s is not None and age > self.remove_s:
                if key in self.applied:
                    out.append(('remove', key))
                    self.applied.pop(key, None)
                self.stale.discard(key)
                del self.latest[key]
                continue
            if age > self.stale_s:
                if key in self.applied and key not in self.stale:
                    out.append(('stale', key))
                    self.stale.add(key)
                continue
            was_stale = key in self.stale
            if was_stale or self._changed(key, dims, xyz, quat):
                out.append(('add' if key not in self.applied else 'move', key, dims, xyz, quat))
                self.applied[key] = (dims, xyz, quat)
                self.stale.discard(key)
        return out


def board_corner_table_xyz(index_or_col, row=None, pitch_m=0.020, cols=8, rows=6):
    """Return WP3 jenga_table (x, y, z) in metres for an inner corner of the 8x6 checkerboard.

    Matches register_board.py UNIT_GRID:
      origin = board centre; +x = along the 8-corner direction (col 0..7); +y = up the 6-corner direction (row 0..5).
      Corner 0 is (col=0, row=0) -> ((-3.5)*pitch, (+2.5)*pitch, 0).
      Corner 7 is (col=7, row=0) -> ((+3.5)*pitch, (+2.5)*pitch, 0).
      Corner 40 is (col=0, row=5) -> ((-3.5)*pitch, (-2.5)*pitch, 0).
      Corner 47 is (col=7, row=5) -> ((+3.5)*pitch, (-2.5)*pitch, 0).
    """
    if row is None:
        idx = int(index_or_col)
        if not (0 <= idx < cols * rows):
            raise ValueError(f'corner index must be 0..{cols * rows - 1}, got {idx}')
        col, row = idx % cols, idx // cols
    else:
        col, row = int(index_or_col), int(row)
        if not (0 <= col < cols and 0 <= row < rows):
            raise ValueError(f'(col, row) must be in 0..{cols - 1} x 0..{rows - 1}, got ({col}, {row})')
    return (
        (col - (cols - 1) / 2.0) * float(pitch_m),
        ((rows - 1) / 2.0 - row) * float(pitch_m),
        0.0,
    )


def scene_id(ns, marker_id):
    """MoveIt object id for a WP3 marker, e.g. ('wp3_blocks', 3) -> 'wp3_blocks_3'."""
    safe = ''.join(ch if ch.isalnum() or ch == '_' else '_' for ch in (ns or 'block'))
    prefix = '' if safe.startswith('wp3_') else 'wp3_'
    return f'{prefix}{safe}_{int(marker_id)}'

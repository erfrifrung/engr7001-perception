"""Offline checks for the WP3 link, no ROS or robot needed:  python3 test_wp3_offline.py"""
import math

import numpy as np

from wp3_link import (BlockTracker, board_corner_table_xyz, fit_table_transform, matrix_from_quat,
                      quat_from_matrix, scene_id, yaw_diff_deg, yaw_of_quat)


def check(c, m):
    print(('PASS ' if c else 'FAIL ') + m)
    assert c


def rotz(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


# --- quaternion round trip
for deg in (0, 37, 90, 179, -120):
    R = rotz(deg)
    check(np.allclose(matrix_from_quat(quat_from_matrix(R)), R, atol=1e-9), f'quat round trip yaw {deg}')
    check(abs(yaw_of_quat(quat_from_matrix(R)) - ((deg + 180) % 360 - 180)) < 1e-6, f'yaw read back {deg}')
Rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.0]])
check(np.allclose(matrix_from_quat(quat_from_matrix(Rx)), Rx), 'quat round trip about x')

# --- planar fit: checkerboard corners (25 mm squares) in jenga_table, true pose 12 deg + offset
sq = 0.025
board = np.array([[0, 0, 0], [6 * sq, 0, 0], [6 * sq, 4 * sq, 0], [0, 4 * sq, 0]], dtype=float)
R_true, t_true = rotz(12.0), np.array([0.31, -0.08, -0.005])
world = (R_true @ board.T).T + t_true
fit = fit_table_transform(board, world, planar=True)
check(abs(fit['yaw_deg'] - 12.0) < 1e-6 and np.allclose(fit['t'], t_true), 'planar fit exact data')
check(fit['rms_m'] < 1e-9, 'planar fit zero residual')

# fingertip touch noise ~1.5 mm: fit stays within a few mm / under a degree
rng = np.random.default_rng(0)
noisy = world + rng.normal(0, 0.0015, world.shape)
fit = fit_table_transform(board, noisy, planar=True)
check(abs(fit['yaw_deg'] - 12.0) < 1.5 and np.linalg.norm(fit['t'] - t_true) < 0.004, 'planar fit noisy data')
check(0.0005 < fit['rms_m'] < 0.004, f"noisy rms reported ({fit['rms_m'] * 1000:.2f} mm)")

# full 3-D fit with a small tilt
tilt = np.array([[1, 0, 0], [0, math.cos(0.05), -math.sin(0.05)], [0, math.sin(0.05), math.cos(0.05)]])
R3 = R_true @ tilt
board3 = np.vstack([board, [[3 * sq, 2 * sq, 0.04]]])           # one raised point (e.g. on a block)
world3 = (R3 @ board3.T).T + t_true
fit3 = fit_table_transform(board3, world3, planar=False)
check(np.allclose(fit3['R'], R3, atol=1e-9) and np.allclose(fit3['t'], t_true), '3-D fit exact data')
check(np.isclose(np.linalg.det(fit3['R']), 1.0), '3-D fit is a rotation, not a reflection')

for bad, msg in ((board[:1], 'one point refused'),):
    try:
        fit_table_transform(bad, bad)
        check(False, msg)
    except ValueError:
        check(True, msg)
try:
    line = np.array([[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0]], dtype=float)
    fit_table_transform(line, line, planar=False)
    check(False, 'collinear points refused for 3-D fit')
except ValueError:
    check(True, 'collinear points refused for 3-D fit')

# --- yaw symmetry
check(yaw_diff_deg(1.0, 179.0) < 2.5, '180-deg symmetric yaw')
check(abs(yaw_diff_deg(0.0, 90.0) - 90.0) < 1e-9, '90 deg stays 90')

# --- tracker
q0 = quat_from_matrix(rotz(0))
q180 = quat_from_matrix(rotz(180))
tr = BlockTracker(stale_s=2.0, remove_s=None)
d = (0.075, 0.025, 0.015)
tr.observe('a', d, (0.3, 0.0, 0.0075), q0, 0.0)
tr.observe('b', d, (0.3, 0.05, 0.0075), q0, 0.0)
acts = tr.actions(0.0)
check([a[0] for a in acts] == ['add', 'add'], 'two new blocks added')
tr.observe('a', d, (0.301, 0.0, 0.0075), q180, 0.5)              # 1 mm + flipped 180: same block pose
check(tr.actions(0.5) == [], 'sub-tolerance move and 180 flip ignored')
tr.observe('a', d, (0.31, 0.0, 0.0075), q0, 1.0)
acts = tr.actions(1.0)
check(len(acts) == 1 and acts[0][0] == 'move' and acts[0][1] == 'a', '1 cm move updates the block')
acts = tr.actions(2.6)                                            # b last seen at t=0
check(acts == [('stale', 'b')], 'unseen block marked stale, not removed')
check(tr.actions(2.7) == [], 'stale reported once')
tr.observe('b', d, (0.3, 0.05, 0.0075), q0, 3.0)
acts = tr.actions(3.0)
check(len(acts) == 1 and acts[0][:2] == ('move', 'b'), 'stale block refreshed when seen again')
tr.delete('a')
check(tr.actions(3.1) == [('remove', 'a')], 'DELETE marker removes the block')

tr2 = BlockTracker(stale_s=1.0, remove_s=5.0)
tr2.observe('c', d, (0.2, 0, 0), q0, 0.0)
tr2.actions(0.0)
check(tr2.actions(1.5) == [('stale', 'c')], 'stale before remove')
check(tr2.actions(5.5) == [('remove', 'c')], 'removed after remove_s')
check(tr2.actions(6.0) == [], 'nothing left')

tr3 = BlockTracker()
tr3.observe('x', d, (0, 0, 0), q0, 0.0)
tr3.observe('y', d, (0.1, 0, 0), q0, 0.0)
tr3.actions(0.0)
tr3.delete_all()
check(sorted(tr3.actions(0.1)) == [('remove', 'x'), ('remove', 'y')], 'DELETEALL removes everything')

check(scene_id('wp3_blocks', 3) == 'wp3_blocks_3' and scene_id('blocks', 3) == 'wp3_blocks_3' and scene_id('', 0) == 'wp3_block_0', 'scene ids are safe')
check(np.allclose(board_corner_table_xyz(0, pitch_m=0.020), (-0.070, 0.050, 0.0)), 'corner 0 matches register_board.py UNIT_GRID')
check(np.allclose(board_corner_table_xyz(7, pitch_m=0.020), (0.070, 0.050, 0.0)), 'corner 7 matches register_board.py UNIT_GRID')
check(np.allclose(board_corner_table_xyz(40, pitch_m=0.020), (-0.070, -0.050, 0.0)), 'corner 40 matches register_board.py UNIT_GRID')
check(np.allclose(board_corner_table_xyz(47, pitch_m=0.020), (0.070, -0.050, 0.0)), 'corner 47 matches register_board.py UNIT_GRID')
check(np.allclose(board_corner_table_xyz(7, 5, pitch_m=0.020), board_corner_table_xyz(47, pitch_m=0.020)), 'col/row matches flat index 47')
print('all WP3 offline checks passed')

#!/usr/bin/env python3
"""Measure where WP3's table frame (`jenga_table`, origin at the 8x6 checkerboard centre) sits in the twin's `world` frame.

WP3 publishes blocks in `jenga_table`. MoveIt works in `world` (robot base at the origin). This fits the
fixed transform between them from points you know in both frames, then wp3_table_tf.py publishes it on TF.

1. Leave the 8x6 inner-corner checkerboard exactly where WP3 calibrated it with register_board.py.
   Check the saved overlay PNGs (`wrist_board.png`, `tripod_board.png`, `third_board.png`) which label
   the four outer inner-corners: 0 (top-left), 7 (top-right), 40 (bottom-left), and 47 (bottom-right).
2. Real arm, slowest speed, supervisor present: python3 teach_keyboard.py, jog the fingertip onto each
   labelled inner corner until it just touches the paper, press `v`, note the printed x y z.
3. Write them in a CSV, either by corner index (4 numbers per line: corner_idx, world_x, world_y, world_z)
   or full 6-column coordinates (table_x, table_y, table_z, world_x, world_y, world_z in metres):
        # corner_idx, world_x, world_y, world_z   (uses --pitch-mm, default 20.0 mm for 8x6 grid)
        0,   0.2412, -0.0421, -0.0049
        7,   0.3810, -0.0418, -0.0051
        40,  0.2415, -0.1420, -0.0048
        47,  0.3812, -0.1419, -0.0050
4. python3 fit_table_tf.py points.csv --pitch-mm 20  # planar fit (yaw + offset), writes wp3_table_tf.yaml
   python3 fit_table_tf.py points.csv --full-3d      # full rotation, needs 3+ non-collinear points

Accept the result if RMS < 3 mm and no point is off by more than 5 mm; otherwise re-touch the worst point.
"""
import argparse
import datetime
import sys

import numpy as np

from wp3_link import board_corner_table_xyz, fit_table_transform

OUT = 'wp3_table_tf.yaml'


def read_points(path, pitch_m=0.020):
    table_rows, world_rows = [], []
    with open(path) as f:
        for line in f:
            line = line.split('#', 1)[0].strip()
            if not line:
                continue
            vals = [float(v) for v in line.replace(',', ' ').split()]
            if len(vals) == 4:
                tx, ty, tz = board_corner_table_xyz(int(round(vals[0])), pitch_m=pitch_m)
                table_rows.append([tx, ty, tz])
                world_rows.append(vals[1:4])
            elif len(vals) == 6:
                table_rows.append(vals[:3])
                world_rows.append(vals[3:])
            else:
                sys.exit(f'bad line (need 4 numbers [corner_idx, wx, wy, wz] or 6 numbers [tx, ty, tz, wx, wy, wz]): {line}')
    return np.array(table_rows, dtype=float), np.array(world_rows, dtype=float)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('csv')
    ap.add_argument('--pitch-mm', type=float, default=20.0, help='checkerboard square pitch in mm (default: 20.0)')
    ap.add_argument('--full-3d', action='store_true', help='fit a full 3-D rotation instead of yaw only')
    ap.add_argument('--parent', default='world')
    ap.add_argument('--child', default='jenga_table')
    ap.add_argument('--out', default=OUT)
    args = ap.parse_args()

    table, world = read_points(args.csv, pitch_m=args.pitch_mm * 0.001)
    fit = fit_table_transform(table, world, planar=not args.full_3d)
    t, q = fit['t'], fit['quat']
    print(f'{len(table)} points, {"3-D" if args.full_3d else "planar"} fit')
    for i, r in enumerate(fit['residuals_m']):
        flag = '  <-- re-touch this one' if r > 0.005 else ''
        print(f'  point {i + 1}: residual {r * 1000:5.1f} mm{flag}')
    print(f'RMS {fit["rms_m"] * 1000:.1f} mm, max {fit["max_m"] * 1000:.1f} mm')
    print(f'{args.child} in {args.parent}: xyz = ({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}) m, yaw = {fit["yaw_deg"]:.2f} deg')
    ok = fit['rms_m'] < 0.003 and fit['max_m'] < 0.005
    print('ACCEPT' if ok else 'CHECK: RMS >= 3 mm or a point > 5 mm off. Re-touch and refit before trusting it.')

    with open(args.out, 'w') as f:
        f.write(f'# written by fit_table_tf.py {datetime.datetime.now().isoformat(timespec="seconds")}\n')
        f.write(f'# {len(table)} points, rms {fit["rms_m"] * 1000:.2f} mm, max {fit["max_m"] * 1000:.2f} mm, '
                f'{"3-D" if args.full_3d else "planar"} fit, {"ACCEPTED" if ok else "NOT ACCEPTED"}\n')
        f.write(f'parent: {args.parent}\nchild: {args.child}\n')
        f.write(f'translation: [{t[0]:.6f}, {t[1]:.6f}, {t[2]:.6f}]\n')
        f.write(f'rotation_xyzw: [{q[0]:.8f}, {q[1]:.8f}, {q[2]:.8f}, {q[3]:.8f}]\n')
        f.write(f'yaw_deg: {fit["yaw_deg"]:.4f}\nrms_mm: {fit["rms_m"] * 1000:.3f}\n')
    print(f'wrote {args.out}  ->  publish it with: python3 wp3_table_tf.py')


if __name__ == '__main__':
    main()

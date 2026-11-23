#!/usr/bin/env python3
"""Publish the fixed world -> jenga_table transform from wp3_table_tf.yaml (made by fit_table_tf.py).

  python3 wp3_table_tf.py                         use wp3_table_tf.yaml
  python3 wp3_table_tf.py --file other.yaml
  python3 wp3_table_tf.py --xyz 0.30 -0.10 -0.005 --yaw 0     quick manual guess, before measuring

Leave it running. It is a latched static transform, so RViz and the bridge get it whenever they start.
Before running: make sure nothing else already publishes a parent for `jenga_table`
(`ros2 run tf2_tools view_frames` on the WP3 side) -- a frame can only have one parent.
"""
import argparse
import math
import sys

import rclpy
import yaml
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', default='wp3_table_tf.yaml')
    ap.add_argument('--xyz', type=float, nargs=3, help='manual translation (m), overrides the file')
    ap.add_argument('--yaw', type=float, default=0.0, help='manual yaw (deg), with --xyz')
    ap.add_argument('--parent', default=None)
    ap.add_argument('--child', default=None)
    args = ap.parse_args()

    if args.xyz:
        h = math.radians(args.yaw) / 2.0
        cfg = {'parent': 'world', 'child': 'jenga_table', 'translation': args.xyz,
               'rotation_xyzw': [0.0, 0.0, math.sin(h), math.cos(h)]}
        print('MANUAL transform (not measured) -- use only to get the display going')
    else:
        try:
            with open(args.file) as f:
                cfg = yaml.safe_load(f)
        except FileNotFoundError:
            sys.exit(f'{args.file} not found: run fit_table_tf.py first, or pass --xyz/--yaw')
    parent = args.parent or cfg.get('parent', 'world')
    child = args.child or cfg.get('child', 'jenga_table')

    rclpy.init()
    node = Node('wp3_table_tf')
    tf = TransformStamped()
    tf.header.stamp = node.get_clock().now().to_msg()
    tf.header.frame_id, tf.child_frame_id = parent, child
    tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = map(float, cfg['translation'])
    (tf.transform.rotation.x, tf.transform.rotation.y,
     tf.transform.rotation.z, tf.transform.rotation.w) = map(float, cfg['rotation_xyzw'])
    StaticTransformBroadcaster(node).sendTransform(tf)
    node.get_logger().info(f'static TF {parent} -> {child}: t={list(cfg["translation"])} q={list(cfg["rotation_xyzw"])}')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

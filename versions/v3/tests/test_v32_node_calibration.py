"""V3.2 live node: a changed /tf_static is picked up without a restart (fault #10).

V3.1 cached the fixed-camera transforms for the node lifetime, so after a recalibration WP3 kept using the old one
until someone restarted it (26 Nov: the old calibration was 99-274 mm off).
"""
from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np
import pytest

IMPL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(IMPL_ROOT))
sys.path.insert(0, str(IMPL_ROOT / "ros2_ws" / "src" / "wp3_live_perception"))

rclpy = pytest.importorskip("rclpy")
from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from rclpy.parameter import Parameter  # noqa: E402
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy  # noqa: E402
from geometry_msgs.msg import TransformStamped  # noqa: E402
from tf2_msgs.msg import TFMessage  # noqa: E402

from wp3_live.proposals import BlockFit  # noqa: E402
from wp3_live_perception.live_node import WP3LivePerceptionNode  # noqa: E402

def _tf(frame: str, x: float) -> TFMessage:
    tr = TransformStamped()
    tr.header.frame_id = "jenga_table"
    tr.child_frame_id = frame
    tr.transform.translation.x, tr.transform.translation.y, tr.transform.translation.z = x, -0.4, 0.5
    tr.transform.rotation.w = 1.0
    return TFMessage(transforms=[tr])


def _run(refresh_s: float, frame: str):
    # one child frame per test: a latched /tf_static sample from an earlier test in this process can still arrive
    rclpy.init()
    node = driver = None
    try:
        driver = rclpy.create_node("wp3_tf_driver")
        node = WP3LivePerceptionNode(parameter_overrides=[
            Parameter("static_tf_refresh_s", Parameter.Type.DOUBLE, refresh_s),
        ])
        node.cfg.calib_epoch = "calib_A"
        node._calib_base = "calib_A"
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        pub = driver.create_publisher(TFMessage, "/tf_static", qos)
        ex = SingleThreadedExecutor()
        ex.add_node(driver)
        ex.add_node(node)

        def spin(t):
            end = time.monotonic() + t
            while time.monotonic() < end:
                ex.spin_once(timeout_sec=0.01)

        def lookup_until(x_expected, t):
            end = time.monotonic() + t
            T = None
            while time.monotonic() < end:
                T = node._lookup_mat("jenga_table", frame, static=True)
                if T is not None and abs(T[0, 3] - x_expected) < 1e-6:
                    return T
                ex.spin_once(timeout_sec=0.02)
            return T

        pub.publish(_tf(frame, 0.10))
        T0 = lookup_until(0.10, 2.0)
        assert T0 is not None and abs(T0[0, 3] - 0.10) < 1e-6
        # one held track from the old calibration
        f = BlockFit(xy_m=(0.0, 0.0), z_center_m=0.0075, z_top_m=0.015, yaw_rad=0.0, yaw_deg_mod=0.0,
                     dims_mm=(75.0, 25.0), fill=1.0, size_err=0.0, n_points=500, points_xyz=np.zeros((0, 3)),
                     covariance_xy_m2=None)
        node.tracker.update([f], ["test"], 1, node._now_s(), side_fresh_global=True)
        assert len(node.tracker._tracks) == 1

        pub.publish(_tf(frame, 0.13))              # recalibrated: the tripod moved 30 mm
        spin(0.2)
        return node, lookup_until(0.13, 2.5)
    except Exception:
        if node is not None:
            node.destroy_node()
        if driver is not None:
            driver.destroy_node()
        rclpy.try_shutdown()
        raise


def _close(node):
    node.destroy_node()
    rclpy.try_shutdown()


def test_recalibration_is_picked_up_and_starts_a_new_epoch():
    node, T = _run(refresh_s=0.5, frame="tripod_a_color_optical_frame")
    try:
        assert T is not None and abs(T[0, 3] - 0.13) < 1e-6
        assert node.cfg.calib_epoch == "calib_A#r1"
        assert len(node.tracker._tracks) == 0      # tracks from the old geometry dropped
    finally:
        _close(node)


def test_v31_behaviour_keeps_the_old_transform_for_the_node_lifetime():
    node, T = _run(refresh_s=0.0, frame="tripod_b_color_optical_frame")
    try:
        assert T is not None and abs(T[0, 3] - 0.10) < 1e-6
        assert node.cfg.calib_epoch == "calib_A"
    finally:
        _close(node)

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


def test_parked_wrist_static_tf_recalibrates_until_joint_states_arrives():
    from sensor_msgs.msg import CameraInfo, Image, JointState

    rclpy.init()
    node = driver = None
    try:
        driver = rclpy.create_node("wp3_wrist_tf_driver")
        node = WP3LivePerceptionNode(parameter_overrides=[
            Parameter("static_tf_refresh_s", Parameter.Type.DOUBLE, 0.3),
            Parameter("event_driven_wakeup", Parameter.Type.BOOL, False),
        ])
        node.cfg.calib_epoch = "calib_A"
        node._calib_base = "calib_A"
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        pub = driver.create_publisher(TFMessage, "/tf_static", qos)
        ex = SingleThreadedExecutor()
        ex.add_node(driver)
        ex.add_node(node)

        frame = "wrist_c_color_optical_frame"
        cinfo = CameraInfo()
        cinfo.width, cinfo.height = 64, 48
        cinfo.k = [60.0, 0.0, 32.0, 0.0, 60.0, 24.0, 0.0, 0.0, 1.0]
        cinfo.header.frame_id = frame
        node._on_cinfo("wrist", cinfo)

        def push_pair():
            now_msg = node.get_clock().now().to_msg()
            rgb = Image(width=64, height=48, encoding="rgb8", step=64 * 3, data=bytes(64 * 48 * 3))
            dep = Image(width=64, height=48, encoding="16UC1", step=64 * 2, data=bytes(64 * 48 * 2))
            rgb.header.stamp = dep.header.stamp = now_msg
            node._on_img("wrist", "rgb", rgb)
            node._on_img("wrist", "depth", dep)

        def select_until(x_expected, timeout_s=2.0):
            end = time.monotonic() + timeout_s
            while time.monotonic() < end:
                push_pair()
                ex.spin_once(timeout_sec=0.02)
                sel, _ = node._select_pair("wrist", node._now_s())
                if sel is not None and sel["T_tc"] is not None and abs(sel["T_tc"][0, 3] - x_expected) < 1e-6:
                    return sel
            return None

        pub.publish(_tf(frame, 0.10))
        assert select_until(0.10) is not None
        f = BlockFit(xy_m=(0.0, 0.0), z_center_m=0.0075, z_top_m=0.015, yaw_rad=0.0, yaw_deg_mod=0.0,
                     dims_mm=(75.0, 25.0), fill=1.0, size_err=0.0, n_points=500, points_xyz=np.zeros((0, 3)),
                     covariance_xy_m2=None)
        node.tracker.update([f], ["test"], 1, node._now_s(), side_fresh_global=True)
        assert len(node.tracker._tracks) == 1

        # Parked wrist (no /joint_states): /tf_static change starts #r1 and clears old tracks
        pub.publish(_tf(frame, 0.14))
        time.sleep(0.35)
        assert select_until(0.14) is not None
        assert node.cfg.calib_epoch == "calib_A#r1"
        assert len(node.tracker._tracks) == 0

        # Once /joint_states arrives, wrist switches to dynamic TF lookup and leaves calib_epoch unchanged
        node._on_joint_states(JointState())
        node.tracker.update([f], ["test"], 2, node._now_s(), side_fresh_global=True)
        pub.publish(_tf(frame, 0.18))
        time.sleep(0.35)
        assert select_until(0.18) is not None
        assert node.cfg.calib_epoch == "calib_A#r1"
        assert len(node.tracker._tracks) == 1
    finally:
        if node is not None:
            node.destroy_node()
        if driver is not None:
            driver.destroy_node()
        rclpy.try_shutdown()

"""Live ROS 2 Jazzy display-only node loopback & zero-MoveIt verification test."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time
import numpy as np

IMPL_ROOT = Path(__file__).resolve().parents[1]
if str(IMPL_ROOT) not in sys.path:
    sys.path.insert(0, str(IMPL_ROOT))
sys.path.insert(0, str(IMPL_ROOT / "scripts"))

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Header, String
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import MarkerArray
from scipy.spatial.transform import Rotation

from wp3_live.geometry_cache import CameraModelCache  # noqa: E402
from wp3_live_perception.live_node import WP3LivePerceptionNode  # noqa: E402
from run_printed_jenga_suite import make_synthetic_cameras, render_synthetic_scene  # noqa: E402


def test_ros2_display_node_loopback_dropout_and_zero_moveit(tmp_path):
    rclpy.init()
    node = None
    driver = None
    try:
        driver = rclpy.create_node("wp3_test_harness")
        cfg_path = str(IMPL_ROOT / "ros2_ws/src/wp3_live_perception/config/printed_jenga_live.json")
        node = WP3LivePerceptionNode(
            parameter_overrides=[
                Parameter("config", Parameter.Type.STRING, cfg_path),
                Parameter("policy", Parameter.Type.STRING, "merged_corroborated"),
                Parameter("save_dir", Parameter.Type.STRING, str(tmp_path / "live_run")),
                Parameter("rate_hz", Parameter.Type.DOUBLE, 20.0),
            ]
        )
        node.cfg.calib_epoch = "loopback_calib"
        node.watchdog.max_age_s = 0.5
        node.watchdog.freeze_timeout_s = 0.55

        # Verify ZERO MoveIt services or clients exist on node!
        srvs = node.get_service_names_and_types_by_node(node.get_name(), node.get_namespace())
        assert not [s for s in srvs if "planning_scene" in s[0]]

        # Publishers mimicking RealSense drivers (RELIABLE images + VOLATILE CameraInfo!)
        latched_tf_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL, history=HistoryPolicy.KEEP_LAST, depth=1)
        rel_vol = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=1)
        pub_tf_s = driver.create_publisher(TFMessage, "/tf_static", latched_tf_qos)
        pub_js = driver.create_publisher(JointState, "/joint_states", 10)
        pubs = {}
        for role, ns in (("wrist", "/perception/wrist"), ("side", "/perception/tripod")):
            pubs[role] = (
                driver.create_publisher(Image, f"{ns}/color/image_raw", rel_vol),
                driver.create_publisher(Image, f"{ns}/aligned_depth_to_color/image_raw", rel_vol),
                driver.create_publisher(CameraInfo, f"{ns}/color/camera_info", rel_vol),
            )

        received_scenes = []
        received_overlays = []
        received_markers = []
        driver.create_subscription(String, "/wp3_live/scene_observation", lambda m: received_scenes.append(json.loads(m.data)), 10)
        driver.create_subscription(Image, "/wp3_live/overlay", lambda m: received_overlays.append(m), 10)
        driver.create_subscription(MarkerArray, "/wp3_live/markers", lambda m: received_markers.append(m), 10)

        # Wait up to 2.0 s for full DDS pub/sub discovery (before publishing any sensor frames)
        exec_ = SingleThreadedExecutor()
        exec_.add_node(driver)
        exec_.add_node(node)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            exec_.spin_once(timeout_sec=0.02)
            all_pubs = [pub for group in pubs.values() for pub in group] + [node.pub_scene]
            if all(p.get_subscription_count() >= 1 for p in all_pubs):
                break
        received_scenes.clear()
        received_overlays.clear()
        received_markers.clear()

        cache = CameraModelCache()
        cam_w, Tw, cam_s, Ts = make_synthetic_cameras(cache)
        rw, dw, rs, ds = render_synthetic_scene(
            cam_w, Tw, cam_s, Ts,
            [
                {"cx": 0.0, "cy": -0.0125, "z0": 0.0, "z1": 0.015, "L": 0.075, "W": 0.025, "yaw_rad": 0.0},
                {"cx": 0.0, "cy": 0.0125, "z0": 0.0, "z1": 0.015, "L": 0.075, "W": 0.025, "yaw_rad": 0.0},
            ],
            seams_top=[{"p0": (-0.037, 0.0), "p1": (0.037, 0.0), "z": 0.015}],
            end_breaks=[{"x": -0.0375, "y": 0.0, "z0": 0.0, "z1": 0.015}, {"x": 0.0375, "y": 0.0, "z0": 0.0, "z1": 0.015}],
        )

        # Publish /tf_static
        tfs = []
        for child, T in (("wrist_color_optical_frame", Tw), ("tripod_color_optical_frame", Ts)):
            tr = TransformStamped()
            tr.header.frame_id = "jenga_table"
            tr.child_frame_id = child
            tr.transform.translation.x, tr.transform.translation.y, tr.transform.translation.z = [float(v) for v in T[:3, 3]]
            q = Rotation.from_matrix(T[:3, :3]).as_quat()
            tr.transform.rotation.x, tr.transform.rotation.y, tr.transform.rotation.z, tr.transform.rotation.w = [float(v) for v in q]
            tfs.append(tr)
        pub_tf_s.publish(TFMessage(transforms=tfs))

        def make_cinfo(cam, stamp):
            ci = CameraInfo()
            ci.header.frame_id = cam.frame_id
            ci.header.stamp = stamp
            ci.width, ci.height = cam.width, cam.height
            ci.k = cam.K_f64.ravel().tolist()
            ci.d = cam.D_f64.tolist()
            return ci

        def spin_for(dur_s: float) -> None:
            t_end = time.monotonic() + dur_s
            while time.monotonic() < t_end:
                exec_.spin_once(timeout_sec=0.01)

        bridge = CvBridge()
        spin_for(0.06)  # Allow /tf_static callback to reach node's tf_buffer
        # Spin 6 dual-camera ticks with stationary joint_states
        for k in range(6):
            now_msg = node.get_clock().now().to_msg()
            pub_js.publish(JointState(header=Header(stamp=now_msg), position=[0.0] * 6))
            for role, cam, rgb, dep in (("wrist", cam_w, rw, dw), ("side", cam_s, rs, ds)):
                pr, pd, pci = pubs[role]
                m_rgb = bridge.cv2_to_imgmsg(rgb, encoding="rgb8"); m_rgb.header.stamp = now_msg; m_rgb.header.frame_id = cam.frame_id
                m_dep = bridge.cv2_to_imgmsg(dep, encoding="16UC1"); m_dep.header.stamp = now_msg; m_dep.header.frame_id = cam.frame_id
                pci.publish(make_cinfo(cam, now_msg)); pr.publish(m_rgb); pd.publish(m_dep)
            spin_for(0.08)

        assert any(s["fallback_mode"] == "dual_view" and len(s["tracks"]) == 2 for s in received_scenes), f"Got scenes: {received_scenes}"
        assert len(received_overlays) >= 1 and len(received_markers) >= 1
        assert (tmp_path / "live_run" / "live_predictions.jsonl").is_file()

        # Freeze side camera (> 0.65 s without new unique frames) -> MUST degrade cleanly to wrist_only!
        spin_for(0.06)  # Drain any in-flight dual_view publish before sampling baseline
        time.sleep(0.65)
        js1_msg = node.get_clock().now().to_msg()
        pub_js.publish(JointState(header=Header(stamp=js1_msg), position=[0.0] * 6))
        spin_for(0.03)
        now_msg = node.get_clock().now().to_msg()
        now_msg_ns = int(now_msg.sec) * 1_000_000_000 + int(now_msg.nanosec)
        pub_js.publish(JointState(header=Header(stamp=now_msg), position=[0.0] * 6))
        m_rgb = bridge.cv2_to_imgmsg(rw, encoding="rgb8"); m_rgb.header.stamp = now_msg; m_rgb.header.frame_id = cam_w.frame_id
        m_dep = bridge.cv2_to_imgmsg(dw, encoding="16UC1"); m_dep.header.stamp = now_msg; m_dep.header.frame_id = cam_w.frame_id
        pubs["wrist"][2].publish(make_cinfo(cam_w, now_msg)); pubs["wrist"][0].publish(m_rgb); pubs["wrist"][1].publish(m_dep)
        wait_end = time.monotonic() + 1.0
        while time.monotonic() < wait_end and not any(s.get("observation_stamp_ns", 0) >= now_msg_ns for s in received_scenes):
            exec_.spin_once(timeout_sec=0.02)
        spin_for(0.03)
        assert any(s.get("observation_stamp_ns", 0) >= now_msg_ns for s in received_scenes), f"Did not receive post-freeze tick"
        assert received_scenes[-1]["fallback_mode"] == "wrist_only", f"Tail: {received_scenes[-2:]}"
    finally:
        if node is not None:
            node.destroy_node()
        if driver is not None:
            driver.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

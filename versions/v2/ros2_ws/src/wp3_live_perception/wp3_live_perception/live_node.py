"""Display-only ROS 2 Jazzy node wrapping the shared wp3_live core (ZERO MoveIt/robot-actuation calls)."""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
from typing import Any

import cv2
import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from scipy.spatial.transform import Rotation

from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from sensor_msgs.msg import CameraInfo, Image, Imu, JointState
from std_msgs.msg import Header, String
from tf2_ros import Buffer, TransformListener
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray


def _ensure_wp3_live_on_path() -> None:
    """Locate the WP3 live snapshot root so wp3_live works with both symlink and installed trees."""
    for p in Path(__file__).resolve().parents:
        if (p / "wp3_live" / "__init__.py").is_file():
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
            return


_ensure_wp3_live_on_path()
from wp3_live import (  # noqa: E402
    EpochTemporalCache,
    ViewInput,
    configure_single_thread_libs,
    load_runtime_config,
    run_perception_epoch,
)
from wp3_live.contract import build_atomic_scene_payload  # noqa: E402
from wp3_live.dwell_gate import WristDwellGate  # noqa: E402
from wp3_live.fusion_tracker import StreamWatchdog, WP3TemporalTracker  # noqa: E402
from wp3_live.geometry_cache import CameraModelCache, StaticTFCache  # noqa: E402
from wp3_live.gpu_mask_worker import AsyncMaskWorker, SAM2MaskChallenger  # noqa: E402
from wp3_live.io_worker import AsyncIOArtifactWriter  # noqa: E402
from wp3_live.overlay import render_overlay  # noqa: E402

configure_single_thread_libs()


ROBOT_LINKS = ("link_base", "link1", "link2", "link3", "link4", "link5", "link6", "link_tcp")


def _stamp_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


class WP3LivePerceptionNode(Node):
    """Display-only perception node with bounded RGB/depth pairing buffers."""

    def __init__(self, *, parameter_overrides=None) -> None:
        self._io_writer = None
        self._mask_worker = None
        self._gc_frozen = False
        self._gc_we_disabled = False
        super().__init__("wp3_live_node", parameter_overrides=parameter_overrides or [])
        self.declare_parameter("config", "")
        self.declare_parameter("policy", "merged_corroborated")
        self.declare_parameter("wrist_ns", "/perception/wrist")
        self.declare_parameter("side_ns", "/perception/tripod")
        self.declare_parameter("use_side_verifier", True)
        self.declare_parameter("publish_overlay", True)
        self.declare_parameter("save_dir", "")
        self.declare_parameter("rate_hz", 15.0)
        self.declare_parameter("enable_gpu_mask_worker", False)
        self.declare_parameter("sam2_weights_path", "")
        self.declare_parameter("enable_temporal_skip", True)

        cfg_path = str(self.get_parameter("config").value)
        self.policy = str(self.get_parameter("policy").value)
        self.wrist_ns = str(self.get_parameter("wrist_ns").value).rstrip("/")
        self.side_ns = str(self.get_parameter("side_ns").value).rstrip("/")
        use_ver = bool(self.get_parameter("use_side_verifier").value)
        self.pub_overlay_flag = bool(self.get_parameter("publish_overlay").value)
        save_dir_str = str(self.get_parameter("save_dir").value)
        rate_hz = max(1.0, float(self.get_parameter("rate_hz").value))
        use_gpu_worker = bool(self.get_parameter("enable_gpu_mask_worker").value)
        sam2_weights = str(self.get_parameter("sam2_weights_path").value)
        use_temporal_skip = bool(self.get_parameter("enable_temporal_skip").value)

        self.cfg = load_runtime_config(
            cfg_path,
            overrides={"use_side_verifier": use_ver, "enable_temporal_skip": use_temporal_skip},
        )
        self.code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
        if use_gpu_worker:
            ch = SAM2MaskChallenger(sam2_weights or None)
            if ch.available:
                self._mask_worker = AsyncMaskWorker(ch)

        self.save_dir = Path(save_dir_str) if save_dir_str else None
        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            (self.save_dir / "overlays").mkdir(exist_ok=True)
            self._io_writer = AsyncIOArtifactWriter(self.save_dir / "live_predictions.jsonl")

        self.bridge = CvBridge()
        self.cam_cache = CameraModelCache()
        self.tf_cache = StaticTFCache()
        self.temporal_cache = EpochTemporalCache()
        self.dwell_gate = WristDwellGate(require_fk=self.cfg.require_fk_for_moving_wrist)
        self.watchdog = StreamWatchdog(
            max_age_s=self.cfg.thresholds.max_age_s,
            freeze_timeout_s=self.cfg.thresholds.freeze_timeout_s,
        )
        self.tracker = WP3TemporalTracker(self.cfg.block, self.cfg.thresholds)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # BEST_EFFORT + VOLATILE connects cleanly to both
        # RELIABLE/BEST_EFFORT and VOLATILE/TRANSIENT_LOCAL publishers (fixes CameraInfo mismatch!)
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=4,
        )

        # Keep a few headers so DDS callback order cannot discard a complete exposure.
        self._latest_lock = threading.Lock()
        self._latest: dict[str, dict[str, Any]] = {
            "wrist": {"rgb": {}, "depth": {}, "cinfo": None},
            "side": {"rgb": {}, "depth": {}, "cinfo": None},
        }
        self._last_processed_pair = (-1, -1)
        self._last_health_sig = ("", "")
        self._last_pub_wall_s = -1.0
        self._cached_robot_links: tuple[float, dict] | None = None
        self._epoch_id = 0

        self._sensor_cb_group = ReentrantCallbackGroup()
        self._tick_cb_group = MutuallyExclusiveCallbackGroup()

        for role, ns in (("wrist", self.wrist_ns), ("side", self.side_ns)):
            self.create_subscription(
                Image, f"{ns}/color/image_raw", lambda m, r=role: self._on_img(r, "rgb", m),
                sensor_qos, callback_group=self._sensor_cb_group,
            )
            self.create_subscription(
                Image, f"{ns}/aligned_depth_to_color/image_raw", lambda m, r=role: self._on_img(r, "depth", m),
                sensor_qos, callback_group=self._sensor_cb_group,
            )
            self.create_subscription(
                CameraInfo, f"{ns}/color/camera_info", lambda m, r=role: self._on_cinfo(r, m),
                sensor_qos, callback_group=self._sensor_cb_group,
            )

        self.create_subscription(JointState, "/joint_states", self._on_joint_states, 10, callback_group=self._sensor_cb_group)
        self.create_subscription(Imu, f"{self.wrist_ns}/gyro/sample", self._on_gyro, 10, callback_group=self._sensor_cb_group)

        # Isolated display-only outputs (/wp3_live/* — ZERO MoveIt clients!)
        self.pub_scene = self.create_publisher(String, "/wp3_live/scene_observation", 10)
        self.pub_health = self.create_publisher(String, "/wp3_live/health", 10)
        self.pub_detections = self.create_publisher(Detection3DArray, "/wp3_live/detections3d", 10)
        self.pub_markers = self.create_publisher(MarkerArray, "/wp3_live/markers", 10)
        self.pub_overlay = self.create_publisher(Image, "/wp3_live/overlay", 1)

        self.create_timer(1.0 / rate_hz, self._tick, callback_group=self._tick_cb_group)
        self.get_logger().info(
            f"wp3_live_node started (policy={self.policy}, wrist={self.wrist_ns}, side={self.side_ns}, display-only)"
        )

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_img(self, role: str, kind: str, msg: Image) -> None:
        rx_ns = self.get_clock().now().nanoseconds
        stamp_ns = _stamp_ns(msg.header.stamp)
        with self._latest_lock:
            images = self._latest[role][kind]
            images[stamp_ns] = (rx_ns, msg)
            if len(images) > 6:
                del images[next(iter(images))]

    def _on_cinfo(self, role: str, msg: CameraInfo) -> None:
        with self._latest_lock:
            self._latest[role]["cinfo"] = msg

    def _on_joint_states(self, msg: JointState) -> None:
        s = _stamp_ns(msg.header.stamp) * 1e-9
        self.dwell_gate.push_joint_state(s if s > 0 else self._now_s(), np.asarray(msg.position, dtype=np.float64))

    def _on_gyro(self, msg: Imu) -> None:
        s = _stamp_ns(msg.header.stamp) * 1e-9
        av = msg.angular_velocity
        self.dwell_gate.push_imu_gyro(s if s > 0 else self._now_s(), np.array([av.x, av.y, av.z], dtype=np.float64))

    def _lookup_mat(self, target: str, source: str, stamp=None, static: bool = False, timeout_s: float = 0.0) -> np.ndarray | None:
        if static:
            cached = self.tf_cache.get(self.cfg.calib_epoch, target, source)
            if cached is not None:
                return cached
        try:
            q_time = Time() if (static or stamp is None) else Time.from_msg(stamp)
            tr = self.tf_buffer.lookup_transform(target, source, q_time, timeout=Duration(seconds=timeout_s)).transform
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = Rotation.from_quat([tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w]).as_matrix()
            T[:3, 3] = [tr.translation.x, tr.translation.y, tr.translation.z]
            if static:
                self.tf_cache.put(self.cfg.calib_epoch, target, source, T)
            return T
        except Exception:
            return None

    def _lookup_robot_links(self, frame_id: str, now_s: float, *, stationary: bool = False) -> dict:
        if stationary and self._cached_robot_links is not None and (now_s - self._cached_robot_links[0]) <= 0.5:
            return self._cached_robot_links[1]
        out = {}
        for link in ROBOT_LINKS:
            T = self._lookup_mat(frame_id, link, None, static=False, timeout_s=0.0)
            if T is not None:
                q = Rotation.from_matrix(T[:3, :3]).as_quat().tolist()
                out[link] = (T[:3, 3].tolist(), q)
        if stationary and out:
            self._cached_robot_links = (now_s, out)
        elif not stationary:
            self._cached_robot_links = None
        return out

    def _build_view(self, role: str, now_s: float) -> tuple[ViewInput | None, dict[str, Any]]:
        with self._latest_lock:
            buf = self._latest[role]
            rgb, depth, cinfo = buf["rgb"], buf["depth"], buf["cinfo"]
            if not rgb or not depth or cinfo is None:
                return None, {"status": "missing_stream", "has_rgb": bool(rgb), "has_depth": bool(depth), "has_info": cinfo is not None}
            common = rgb.keys() & depth.keys()
            latest_rgb = max(rgb)
            latest_depth = max(depth)
            latest_common = max(common, default=None)
            max_skew_ns = self.cfg.thresholds.max_pair_skew_s * 1e9
            if latest_common is not None and min(latest_rgb, latest_depth) - latest_common <= max_skew_ns:
                h_rgb = h_dep = latest_common
            else:
                # Preserve tolerance for inputs whose RGB and depth stamps differ.
                h_rgb, h_dep = min(
                    ((r, d) for r in rgb for d in depth),
                    key=lambda p: (abs(p[0] - p[1]) > max_skew_ns, -min(p), abs(p[0] - p[1])),
                )
            rgb_pair, dep_pair = rgb[h_rgb], depth[h_dep]

        rx_rgb_ns, msg_rgb = rgb_pair
        _, msg_dep = dep_pair
        skew_s = abs(h_rgb - h_dep) * 1e-9
        health = self.watchdog.update(role, h_rgb, rx_rgb_ns * 1e-9, now_s, memoryview(msg_rgb.data))
        if skew_s > self.cfg.thresholds.max_pair_skew_s:
            return None, {"status": "rgb_depth_skew", "skew_s": round(skew_s, 4)}

        cam = self.cam_cache.get(
            role,
            cinfo.width,
            cinfo.height,
            np.asarray(cinfo.k, dtype=np.float64),
            np.asarray(cinfo.d, dtype=np.float64),
            cinfo.header.frame_id or f"{role}_color_optical_frame",
            self.cfg.wrist.depth_scale_m if role == "wrist" else self.cfg.side.depth_scale_m,
        )
        # Dynamic TF lookup for wrist if non-static, cached TF for tripod
        T_tc = self._lookup_mat(
            self.cfg.frame_id,
            cam.frame_id,
            msg_rgb.header.stamp if role == "wrist" else None,
            static=(role != "wrist"),
            timeout_s=0.005 if role == "wrist" else 0.0,
        )
        rgb_np = self.bridge.imgmsg_to_cv2(msg_rgb, desired_encoding="rgb8")
        dep_np = self.bridge.imgmsg_to_cv2(msg_dep, desired_encoding="passthrough")
        vi = ViewInput(
            role="wrist" if role == "wrist" else "tripod",
            cam=cam,
            rgb_img=np.ascontiguousarray(rgb_np, dtype=np.uint8),
            depth_raw=np.ascontiguousarray(dep_np),
            T_table_cam=T_tc,
            stamp_ns=h_rgb,
            rx_ns=rx_rgb_ns,
            is_fresh=not health.stale,
            is_frozen=health.frozen,
            pose_frame_mode="table" if T_tc is not None else "plane",
        )
        return vi, {
            "status": "frozen" if health.frozen else ("stale" if health.stale else "ok"),
            "stamp_ns": h_rgb,
            "age_ms": round((now_s - h_rgb * 1e-9) * 1000.0, 2),
            "rx_minus_hdr_ms": round((rx_rgb_ns - h_rgb) * 1e-6, 2),
            "tf_valid": T_tc is not None,
        }

    def _tick(self) -> None:
        now_ns = self.get_clock().now().nanoseconds
        now_s = now_ns * 1e-9
        wrist_vi, wrist_h = self._build_view("wrist", now_s)
        side_vi, side_h = self._build_view("side", now_s)

        pair_sig = (
            wrist_vi.stamp_ns if wrist_vi is not None else -1,
            side_vi.stamp_ns if side_vi is not None else -1,
        )
        health_sig = (str(wrist_h.get("status", "")), str(side_h.get("status", "")))
        if (
            pair_sig != (-1, -1)
            and pair_sig == self._last_processed_pair
            and health_sig == self._last_health_sig
            and (now_s - self._last_pub_wall_s) < 0.25
        ):
            # Skip redundant compute only while frame stamps AND health state remain unchanged (< 0.25 s heartbeat)
            return
        self._last_processed_pair = pair_sig
        self._last_health_sig = health_sig
        self._last_pub_wall_s = now_s
        self._epoch_id += 1

        dwell = None
        if wrist_vi is not None:
            gray_w = cv2.cvtColor(wrist_vi.rgb_img, cv2.COLOR_RGB2GRAY)
            dwell = self.dwell_gate.evaluate(wrist_vi.stamp_ns * 1e-9, gray_w)

        ext_masks = None
        if self._mask_worker is not None:
            m_res = self._mask_worker.poll_matching(self._epoch_id, self.cfg.calib_epoch, "global")
            if m_res is not None:
                ext_masks = m_res.masks_u8

        robot_links = (
            self._lookup_robot_links(
                self.cfg.frame_id,
                now_s,
                stationary=bool(dwell is not None and dwell.stationary),
            )
            if self.cfg.enable_robot_self_filter
            else None
        )
        epoch = run_perception_epoch(
            wrist=wrist_vi,
            side=side_vi,
            config=self.cfg,
            policy=self.policy,
            dwell=dwell,
            robot_links=robot_links,
            epoch_id=self._epoch_id,
            external_masks=ext_masks,
            temporal_cache=self.temporal_cache,
        )
        if self._mask_worker is not None and wrist_vi is not None and epoch.ambiguous_clusters:
            bbox = tuple(epoch.evidence_log.get("bbox_2d", (0, 0, wrist_vi.cam.width, wrist_vi.cam.height)))
            if len(bbox) == 4 and (bbox[2] - bbox[0]) >= 16 and (bbox[3] - bbox[1]) >= 16:
                self._mask_worker.submit(self._epoch_id, wrist_vi.stamp_ns, self.cfg.calib_epoch, "global", wrist_vi.rgb_img, bbox)

        side_fresh_ok = side_vi is not None and side_vi.is_fresh and not side_vi.is_frozen and side_vi.T_table_cam is not None
        tracks = self.tracker.update(
            epoch.poses,
            epoch.pose_sources,
            epoch.stamp_ns,
            now_s,
            side_fresh_global=side_fresh_ok,
            pose_side_visible=epoch.pose_side_visible,
        )

        cam_health = {"wrist": wrist_h, "side": side_h}
        payload = build_atomic_scene_payload(
            epoch,
            tracks,
            self.cfg,
            epoch_id=self._epoch_id,
            publish_stamp_ns=now_ns,
            code_hash=self.code_hash,
            camera_health=cam_health,
        )
        self.pub_scene.publish(String(data=json.dumps(payload)))
        self.pub_health.publish(String(data=json.dumps({
            "epoch_id": self._epoch_id,
            "fallback_mode": epoch.fallback_mode,
            "camera_health": cam_health,
            "stage_timings_ms": epoch.stage_timings_ms,
        })))

        header = Header(frame_id=self.cfg.frame_id, stamp=self.get_clock().now().to_msg())
        self._publish_detections(header, tracks)
        self._publish_markers(header, tracks, epoch)

        if self.pub_overlay_flag:
            bgr = render_overlay(
                wrist_vi.rgb_img if wrist_vi else None,
                side_vi.rgb_img if side_vi else None,
                epoch,
                tracks,
                self.cfg.block,
            )
            self.pub_overlay.publish(self.bridge.cv2_to_imgmsg(bgr, encoding="bgr8", header=header))
            if self._io_writer is not None and self.save_dir is not None and (self._epoch_id % 15 == 1 or self._epoch_id <= 3):
                self._io_writer.write_png(self.save_dir / "overlays" / f"overlay_{self._epoch_id:05d}.png", bgr)

        if self._io_writer is not None:
            self._io_writer.write_jsonl(json.dumps(payload))

        if self._epoch_id >= 5:
            if not self._gc_frozen:
                gc.collect()
                gc.freeze()
                gc.disable()
                self._gc_frozen = True
                self._gc_we_disabled = True
            elif self._epoch_id % 15 == 0:
                gc.collect(0)

    def _publish_detections(self, header: Header, tracks) -> None:
        arr = Detection3DArray(header=header)
        for tr in tracks:
            if tr.state not in ("MEASURED", "DISPLACED"):
                continue
            d = Detection3D(header=header, id=tr.block_name)
            d.bbox.center.position.x = float(tr.fit.xy_m[0])
            d.bbox.center.position.y = float(tr.fit.xy_m[1])
            d.bbox.center.position.z = float(tr.fit.z_center_m)
            d.bbox.center.orientation.z = float(np.sin(tr.fit.yaw_rad / 2.0))
            d.bbox.center.orientation.w = float(np.cos(tr.fit.yaw_rad / 2.0))
            d.bbox.size.x = float(self.cfg.block.major_m)
            d.bbox.size.y = float(self.cfg.block.minor_m)
            d.bbox.size.z = float(self.cfg.block.height_m)
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = f"jenga_block:{tr.source}"
            hyp.hypothesis.score = 1.0 if tr.confirmed else 0.5
            hyp.pose.pose = d.bbox.center
            d.results = [hyp]
            arr.detections.append(d)
        self.pub_detections.publish(arr)

    def _publish_markers(self, header: Header, tracks, epoch) -> None:
        arr = MarkerArray()
        clr = Marker(header=header, action=Marker.DELETEALL)
        arr.markers.append(clr)
        cols = {
            "MEASURED": (0.20, 0.85, 0.40, 0.82),
            "DISPLACED": (1.00, 0.55, 0.10, 0.90),
            "OCCLUDED": (0.98, 0.80, 0.25, 0.35),
            "STALE": (0.70, 0.70, 0.70, 0.25),
            "UNKNOWN": (0.90, 0.25, 0.25, 0.25),
        }
        for tr in tracks:
            m = Marker(header=header, ns="wp3_blocks", id=tr.track_id, type=Marker.CUBE, action=Marker.ADD)
            m.pose.position.x = float(tr.fit.xy_m[0])
            m.pose.position.y = float(tr.fit.xy_m[1])
            m.pose.position.z = float(tr.fit.z_center_m)
            m.pose.orientation.z = float(np.sin(tr.fit.yaw_rad / 2.0))
            m.pose.orientation.w = float(np.cos(tr.fit.yaw_rad / 2.0))
            m.scale.x, m.scale.y, m.scale.z = self.cfg.block.major_m, self.cfg.block.minor_m, self.cfg.block.height_m
            r, g, b, a = cols.get(tr.state, (0.5, 0.5, 0.5, 0.5))
            m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, a
            arr.markers.append(m)

            t = Marker(header=header, ns="wp3_labels", id=tr.track_id, type=Marker.TEXT_VIEW_FACING, action=Marker.ADD)
            t.pose.position.x = float(tr.fit.xy_m[0])
            t.pose.position.y = float(tr.fit.xy_m[1])
            t.pose.position.z = float(tr.fit.z_top_m + 0.03)
            t.scale.z = 0.018
            t.color.r = t.color.g = t.color.b = t.color.a = 1.0
            t.text = f"{tr.block_name}/{tr.state}"
            arr.markers.append(t)

        for idx, amb in enumerate(epoch.ambiguous_clusters, start=1):
            am = Marker(header=header, ns="wp3_ambiguous", id=1000 + idx, type=Marker.CUBE, action=Marker.ADD)
            am.pose.position.x, am.pose.position.y = float(amb.xy_m[0]), float(amb.xy_m[1])
            am.pose.position.z = float(amb.top_z_m - self.cfg.block.height_m / 2.0)
            am.pose.orientation.w = 1.0
            am.scale.x = max(0.03, (amb.span_mm[0] or 75.0) / 1000.0)
            am.scale.y = max(0.03, (amb.span_mm[1] or 50.0) / 1000.0)
            am.scale.z = self.cfg.block.height_m + 0.004
            am.color.r, am.color.g, am.color.b, am.color.a = 1.0, 0.85, 0.0, 0.45
            arr.markers.append(am)
        self.pub_markers.publish(arr)

    def destroy_node(self) -> bool:
        if getattr(self, "_gc_we_disabled", False):
            gc.enable()
            self._gc_we_disabled = False
        if getattr(self, "_mask_worker", None) is not None:
            self._mask_worker.stop()
            self._mask_worker = None
        if getattr(self, "_io_writer", None) is not None:
            self._io_writer.stop()
            self._io_writer = None
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WP3LivePerceptionNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

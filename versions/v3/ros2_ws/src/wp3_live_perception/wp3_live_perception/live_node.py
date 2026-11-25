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
    """Locate WP3-Live-Implementation root so wp3_live works with both symlink and installed trees."""
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


def _package_code_hash() -> str:
    """Hash this node plus every wp3_live module, so V2/V3/V3.1 runs are distinguishable in logs."""
    import wp3_live

    h = hashlib.sha256()
    files = [Path(__file__).resolve()] + sorted(Path(wp3_live.__file__).resolve().parent.glob("*.py"))
    for p in files:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


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
        self.declare_parameter("third_ns", "")
        self.declare_parameter("use_side_verifier", True)
        self.declare_parameter("publish_overlay", True)
        self.declare_parameter("save_dir", "")
        self.declare_parameter("rate_hz", 15.0)
        self.declare_parameter("enable_gpu_mask_worker", False)
        self.declare_parameter("sam2_weights_path", "")
        # V3.1: frame skipping re-publishes cached geometry, so it is opt-in.
        self.declare_parameter("enable_temporal_skip", False)
        # V3.1: the live default is the V2 path. The ROS parameter wins over the JSON profile for this switch.
        self.declare_parameter("enable_v3_regions", False)
        # Empty / negative = keep the JSON profile (or dataclass default) value.
        self.declare_parameter("v3_solver", "")
        self.declare_parameter("v3_frame_budget_ms", -1.0)
        self.declare_parameter("persistent_joint_occlusion", True)
        self.declare_parameter("reanchor_after", 2)
        # Robot TF must be within this of the camera frame stamp to be used (never unbounded "latest").
        self.declare_parameter("robot_tf_max_age_s", 0.050)

        cfg_path = str(self.get_parameter("config").value)
        self.policy = str(self.get_parameter("policy").value)
        self.wrist_ns = str(self.get_parameter("wrist_ns").value).rstrip("/")
        self.side_ns = str(self.get_parameter("side_ns").value).rstrip("/")
        self.third_ns = str(self.get_parameter("third_ns").value).rstrip("/")
        if self.third_ns and self.third_ns in (self.wrist_ns, self.side_ns):
            raise ValueError("third_ns must identify a distinct fixed camera")
        self._camera_namespaces = {"wrist": self.wrist_ns, "side": self.side_ns}
        if self.third_ns:
            self._camera_namespaces["third"] = self.third_ns
        use_ver = bool(self.get_parameter("use_side_verifier").value)
        self.pub_overlay_flag = bool(self.get_parameter("publish_overlay").value)
        save_dir_str = str(self.get_parameter("save_dir").value)
        rate_hz = max(1.0, float(self.get_parameter("rate_hz").value))
        use_gpu_worker = bool(self.get_parameter("enable_gpu_mask_worker").value)
        sam2_weights = str(self.get_parameter("sam2_weights_path").value)
        use_temporal_skip = bool(self.get_parameter("enable_temporal_skip").value)

        overrides: dict[str, Any] = {
            "use_side_verifier": use_ver,
            "enable_temporal_skip": use_temporal_skip,
            "enable_v3_regions": bool(self.get_parameter("enable_v3_regions").value),
            "persistent_joint_occlusion": bool(self.get_parameter("persistent_joint_occlusion").value),
            "reanchor_after": int(self.get_parameter("reanchor_after").value),
        }
        v3_solver = str(self.get_parameter("v3_solver").value).strip()
        if v3_solver:
            if v3_solver not in ("prove", "enumerate"):
                raise ValueError(f"v3_solver must be 'prove' or 'enumerate', got {v3_solver!r}")
            overrides["v3_solver"] = v3_solver
        v3_budget = float(self.get_parameter("v3_frame_budget_ms").value)
        if v3_budget > 0.0:
            overrides["v3_frame_budget_ms"] = v3_budget
        self.robot_tf_max_age_s = max(0.0, float(self.get_parameter("robot_tf_max_age_s").value))
        self.cfg = load_runtime_config(cfg_path, overrides=overrides)
        self.code_hash = _package_code_hash()
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
        self.tracker = WP3TemporalTracker(
            self.cfg.block,
            self.cfg.thresholds,
            persistent_joint_occlusion=self.cfg.persistent_joint_occlusion,
            reanchor_after=self.cfg.reanchor_after,
        )
        self.get_logger().info(
            f"config: enable_v3_regions={self.cfg.enable_v3_regions} v3_solver={self.cfg.v3_solver} "
            f"v3_frame_budget_ms={self.cfg.v3_frame_budget_ms} persistent_joint_occlusion={self.cfg.persistent_joint_occlusion} "
            f"reanchor_after={self.cfg.reanchor_after} enable_temporal_skip={self.cfg.enable_temporal_skip} "
            f"require_fk_for_moving_wrist={self.cfg.require_fk_for_moving_wrist} code_hash={self.code_hash}"
        )
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
            role: {"rgb": {}, "depth": {}, "cinfo": None} for role in self._camera_namespaces
        }
        self._last_processed_pair = ()
        self._last_health_sig = ()
        self._last_pub_wall_s = -1.0
        self._cached_robot_links: tuple[float, dict] | None = None
        self._epoch_id = 0

        self._sensor_cb_group = ReentrantCallbackGroup()
        self._tick_cb_group = MutuallyExclusiveCallbackGroup()

        for role, ns in self._camera_namespaces.items():
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
            f"wp3_live_node started (policy={self.policy}, wrist={self.wrist_ns}, side={self.side_ns}, third={self.third_ns or 'disabled'}, display-only)"
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

    def _lookup_tf_stamped(self, target: str, source: str, q_time: Time, timeout_s: float):
        """(4x4, header stamp ns) or (None, None). Never raises."""
        try:
            ts = self.tf_buffer.lookup_transform(target, source, q_time, timeout=Duration(seconds=timeout_s))
        except Exception:
            return None, None
        tr = ts.transform
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = Rotation.from_quat([tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w]).as_matrix()
        T[:3, 3] = [tr.translation.x, tr.translation.y, tr.translation.z]
        return T, _stamp_ns(ts.header.stamp)

    def _lookup_robot_links(self, frame_id: str, now_s: float, frame_stamp_ns: int | None, *,
                            stationary: bool = False) -> dict:
        """Robot link poses at the camera exposure time.

        The arm is looked up at ``frame_stamp_ns`` (short wait for /tf to catch up). If that extrapolates, the
        latest TF is used only when its own stamp is within ``robot_tf_max_age_s`` of the frame. Otherwise no
        links are returned, so the V3 stage treats upper layers as unqualified rather than using a stale arm pose.
        """
        if stationary and self._cached_robot_links is not None and (now_s - self._cached_robot_links[0]) <= 0.5:
            return self._cached_robot_links[1]
        out: dict = {}
        q_time = Time(nanoseconds=int(frame_stamp_ns)) if frame_stamp_ns else Time()
        T6, st6 = self._lookup_tf_stamped(frame_id, "link6", q_time, 0.015 if frame_stamp_ns else 0.0)
        if T6 is None and frame_stamp_ns:
            T6, st6 = self._lookup_tf_stamped(frame_id, "link6", Time(), 0.0)
            if T6 is None or st6 is None or abs(st6 - int(frame_stamp_ns)) * 1e-9 > self.robot_tf_max_age_s:
                T6 = None
            else:
                q_time = Time(nanoseconds=int(st6))
        if T6 is None:
            if not stationary:
                self._cached_robot_links = None
            return out
        for link in ROBOT_LINKS:
            T, _ = (T6, st6) if link == "link6" else self._lookup_tf_stamped(frame_id, link, q_time, 0.0)
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

        cam_profile = self.cfg.wrist if role == "wrist" else (self.cfg.third if role == "third" else self.cfg.side)
        cam = self.cam_cache.get(
            role,
            cinfo.width,
            cinfo.height,
            np.asarray(cinfo.k, dtype=np.float64),
            np.asarray(cinfo.d, dtype=np.float64),
            cinfo.header.frame_id or f"{role}_color_optical_frame",
            cam_profile.depth_scale_m,
        )
        # Dynamic TF lookup for wrist if non-static, cached TF for tripod
        T_tc = self._lookup_mat(
            self.cfg.frame_id,
            cam.frame_id,
            msg_rgb.header.stamp if role == "wrist" else None,
            static=(role != "wrist"),
            timeout_s=0.005 if role == "wrist" else 0.0,
        )
        if role == "third" and T_tc is None:
            return None, {"status": "missing_table_tf", "tf_valid": False}
        rgb_np = self.bridge.imgmsg_to_cv2(msg_rgb, desired_encoding="rgb8")
        dep_np = self.bridge.imgmsg_to_cv2(msg_dep, desired_encoding="passthrough")
        if rgb_np.shape[:2] != (cam.height, cam.width) or dep_np.shape[:2] != rgb_np.shape[:2]:
            # Wrong profile or unaligned depth would silently mis-project every pixel: refuse the view.
            return None, {"status": "camera_info_shape_mismatch", "tf_valid": T_tc is not None}
        vi = ViewInput(
            role="tripod" if role == "side" else role,
            cam=cam,
            rgb_img=np.ascontiguousarray(rgb_np, dtype=np.uint8),
            depth_raw=np.ascontiguousarray(dep_np),
            T_table_cam=T_tc,
            stamp_ns=h_rgb,
            rx_ns=rx_rgb_ns,
            is_fresh=not health.stale,
            is_frozen=health.frozen,
            # V3.1: never fall back to a camera-local plane frame live. Without T_table_cam the epoch reports
            # status "uncalibrated" and publishes no poses under frame_id "jenga_table".
            pose_frame_mode="table",
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

        third_vi = None
        cam_health = {"wrist": wrist_h, "side": side_h}
        views = [wrist_vi, side_vi]
        if self.third_ns:
            third_vi, third_h = self._build_view("third", now_s)
            cam_health["third"] = third_h
            views.append(third_vi)
        pair_sig = tuple(view.stamp_ns if view is not None else -1 for view in views)
        health_sig = tuple(str(health.get("status", "")) for health in cam_health.values())
        if (
            any(stamp >= 0 for stamp in pair_sig)
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

        frame_stamp_ns = wrist_vi.stamp_ns if wrist_vi is not None else (side_vi.stamp_ns if side_vi is not None else None)
        robot_links = (
            self._lookup_robot_links(
                self.cfg.frame_id,
                now_s,
                frame_stamp_ns,
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
            third=third_vi,
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

        if self.third_ns:
            cam_health["third"]["cut_status"] = epoch.evidence_log.get("third_cut_status", "unavailable")
            cam_health["third"]["cut_verdicts"] = epoch.evidence_log.get("third_cut_verdicts", {})
        # V3.1: stamp publication after compute, so publish - observation shows the real processing delay
        # (tick start is kept separately as tick_start_ns in the health message).
        publish_ns = self.get_clock().now().nanoseconds
        payload = build_atomic_scene_payload(
            epoch,
            tracks,
            self.cfg,
            epoch_id=self._epoch_id,
            publish_stamp_ns=publish_ns,
            code_hash=self.code_hash,
            camera_health=cam_health,
        )
        self.pub_scene.publish(String(data=json.dumps(payload)))
        self.pub_health.publish(String(data=json.dumps({
            "epoch_id": self._epoch_id,
            "tick_start_ns": now_ns,
            "publish_stamp_ns": publish_ns,
            "status": epoch.status,
            "dwell_reason": epoch.dwell.reason if epoch.dwell else None,
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
                third_rgb=third_vi.rgb_img if third_vi else None,
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

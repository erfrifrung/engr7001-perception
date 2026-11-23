"""Read-only dual-backend (rosbag2_py + rosbags fallback) SQLite3 bag reader excluding PointCloud2."""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


STALE_PAIR_NS = 2_000_000_000


def _quat_to_mat(tr, q) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rotation.from_quat([float(q.x), float(q.y), float(q.z), float(q.w)]).as_matrix()
    T[:3, 3] = [float(tr.x), float(tr.y), float(tr.z)]
    return T


def _decode_image_msg(msg) -> np.ndarray:
    enc = msg.encoding
    dtype = np.uint16 if enc in ("16UC1", "mono16") else np.uint8
    data = np.frombuffer(bytes(msg.data), dtype=dtype)
    ch = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1, "16UC1": 1, "mono16": 1}[enc]
    if ch == 1:
        arr = data.reshape(msg.height, -1)[:, : msg.width]
        return np.ascontiguousarray(arr)
    arr = data.reshape(msg.height, -1)[:, : msg.width * ch].reshape(msg.height, msg.width, ch)
    if enc == "bgr8":
        arr = arr[:, :, ::-1]
    elif enc == "bgra8":
        arr = arr[:, :, [2, 1, 0]]
    elif enc == "rgba8":
        arr = arr[:, :, :3]
    return np.ascontiguousarray(arr)


def chain_tf(tf_map: dict[tuple[str, str], np.ndarray], parent: str, child: str) -> np.ndarray:
    up = {c: p for (p, c) in tf_map}
    T = np.eye(4, dtype=np.float64)
    node = child
    path = []
    while node != parent:
        if node not in up:
            raise KeyError(f"No static TF path {parent} -> {child}")
        path.append((up[node], node))
        node = up[node]
    for p, c in reversed(path):
        T = T @ tf_map[(p, c)]
    return T


@dataclass
class CameraStreamData:
    role: str
    namespace: str
    info: dict[str, Any] | None
    pairs: list[dict[str, Any]]     # sorted by rx_ns: {rx_ns, header_ns, rgb, depth_u16}
    dropped: dict[str, int]


@dataclass
class BagContents:
    bag_path: Path
    t0_ns: int
    tf_static: dict[tuple[str, str], np.ndarray]
    cameras: dict[str, CameraStreamData]
    imu_gyro: list[tuple[float, np.ndarray]]
    joint_states: list[tuple[float, np.ndarray]]


def _resolve_db3(bag_path: str | Path) -> Path:
    p = Path(bag_path)
    if p.is_dir():
        dbs = sorted(p.glob("*.db3"))
        if not dbs:
            raise FileNotFoundError(f"No .db3 file found in {p}")
        return dbs[0]
    if not p.is_file():
        raise FileNotFoundError(f"Bag not found: {p}")
    return p


def load_bag_fast(
    bag_path: str | Path,
    camera_specs: list[tuple[str, str]],
    *,
    use_aligned_depth: bool = True,
) -> BagContents:
    """Single-pass read-only bag load filtering strictly to needed Image/CameraInfo/TF/JointState/IMU topics."""
    db3 = _resolve_db3(bag_path)
    tf_static: dict[tuple[str, str], np.ndarray] = {}
    imu_gyro: list[tuple[float, np.ndarray]] = []
    joint_states: list[tuple[float, np.ndarray]] = []

    wanted_topics: set[str] = {"/tf_static", "/joint_states"}
    topic_role_map: dict[str, tuple[str, str]] = {}
    for role, ns in camera_specs:
        depth_topic = f"{ns}/aligned_depth_to_color/image_raw" if use_aligned_depth else f"{ns}/depth/image_rect_raw"
        dinfo_topic = f"{ns}/aligned_depth_to_color/camera_info" if use_aligned_depth else f"{ns}/depth/camera_info"
        c_topic = f"{ns}/color/image_raw"
        cinfo_topic = f"{ns}/color/camera_info"
        gyro_topic = f"{ns}/gyro/sample"
        for t, kind in ((c_topic, "rgb"), (depth_topic, "depth"), (cinfo_topic, "cinfo"), (dinfo_topic, "dinfo"), (gyro_topic, "gyro")):
            wanted_topics.add(t)
            topic_role_map[t] = (role, kind)

    raw_rgb: dict[str, dict[int, tuple[int, np.ndarray]]] = {r: {} for r, _ in camera_specs}
    raw_depth: dict[str, dict[int, tuple[int, np.ndarray]]] = {r: {} for r, _ in camera_specs}
    infos: dict[str, dict[str, Any]] = {}
    dropped_counts: dict[str, dict[str, int]] = {
        r: {"colour_stale": 0, "depth_stale": 0, "unpaired": 0} for r, _ in camera_specs
    }

    t0_ns: int | None = None
    has_rosbag2_py = importlib.util.find_spec("rosbag2_py") is not None and importlib.util.find_spec("rclpy") is not None

    if has_rosbag2_py:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message

        reader = rosbag2_py.SequentialReader()
        storage_opts = rosbag2_py.StorageOptions(uri=str(db3.parent if db3.name.endswith("_0.db3") and (db3.parent / "metadata.yaml").is_file() else db3), storage_id="sqlite3")
        conv_opts = rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")
        try:
            reader.open(storage_opts, conv_opts)
        except Exception:
            reader = rosbag2_py.SequentialReader()
            reader.open(rosbag2_py.StorageOptions(uri=str(db3), storage_id="sqlite3"), conv_opts)

        meta = reader.get_metadata()
        # t0_ns matches first message timestamp across bag
        import sqlite3
        with sqlite3.connect(f"file:{db3}?mode=ro&immutable=1", uri=True) as conn:
            t0_ns = int(conn.execute("select min(timestamp) from messages").fetchone()[0])

        topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
        active_topics = [t for t in wanted_topics if t in topic_types]
        if active_topics:
            reader.set_filter(rosbag2_py.StorageFilter(topics=active_topics))
        msg_cls_cache = {t: get_message(topic_types[t]) for t in active_topics}

        while reader.has_next():
            topic, data, rx_ns = reader.read_next()
            msg = deserialize_message(data, msg_cls_cache[topic])
            _consume_msg(topic, int(rx_ns), msg, tf_static, imu_gyro, joint_states, topic_role_map, raw_rgb, raw_depth, infos, dropped_counts)
    else:
        import sqlite3
        from rosbags.typesys import Stores, get_typestore

        typestore = get_typestore(Stores.ROS2_JAZZY)
        conn = sqlite3.connect(f"file:{db3}?mode=ro&immutable=1", uri=True)
        t0_ns = int(conn.execute("select min(timestamp) from messages").fetchone()[0])
        topics_db = {n: (tid, typ) for tid, n, typ in conn.execute("select id,name,type from topics")}
        active_ids = [topics_db[t][0] for t in wanted_topics if t in topics_db]
        id_to_meta = {tid: (n, typ) for n, (tid, typ) in topics_db.items() if n in wanted_topics}
        if active_ids:
            placeholders = ",".join("?" for _ in active_ids)
            for tid, ts, raw in conn.execute(
                f"select topic_id, timestamp, data from messages where topic_id in ({placeholders}) order by timestamp",
                active_ids,
            ):
                topic, typ = id_to_meta[tid]
                msg = typestore.deserialize_cdr(raw, typ)
                _consume_msg(topic, int(ts), msg, tf_static, imu_gyro, joint_states, topic_role_map, raw_rgb, raw_depth, infos, dropped_counts)
        conn.close()

    assert t0_ns is not None
    cameras: dict[str, CameraStreamData] = {}
    for role, ns in camera_specs:
        c_dict = raw_rgb[role]
        d_dict = raw_depth[role]
        pairs = []
        for h_ns, (rx_ns, rgb_img) in sorted(c_dict.items(), key=lambda kv: kv[1][0]):
            if h_ns not in d_dict:
                dropped_counts[role]["unpaired"] += 1
                continue
            _, d_img = d_dict[h_ns]
            pairs.append({"rx_ns": rx_ns, "header_ns": h_ns, "rgb": rgb_img, "depth_u16": d_img})
        cameras[role] = CameraStreamData(role=role, namespace=ns, info=infos.get(role), pairs=pairs, dropped=dropped_counts[role])

    return BagContents(
        bag_path=db3,
        t0_ns=t0_ns,
        tf_static=tf_static,
        cameras=cameras,
        imu_gyro=imu_gyro,
        joint_states=joint_states,
    )


def _consume_msg(
    topic: str,
    rx_ns: int,
    msg: Any,
    tf_static: dict,
    imu_gyro: list,
    joint_states: list,
    topic_role_map: dict,
    raw_rgb: dict,
    raw_depth: dict,
    infos: dict,
    dropped: dict,
) -> None:
    if topic == "/tf_static":
        for tr in msg.transforms:
            tf_static[(tr.header.frame_id, tr.child_frame_id)] = _quat_to_mat(tr.transform.translation, tr.transform.rotation)
        return
    if topic == "/joint_states":
        h_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        joint_states.append((h_ns * 1e-9, np.asarray(msg.position, dtype=np.float64)))
        return
    if topic not in topic_role_map:
        return
    role, kind = topic_role_map[topic]
    if kind in ("cinfo", "dinfo"):
        if role not in infos or kind == "cinfo":
            infos[role] = {
                "width": int(msg.width),
                "height": int(msg.height),
                "K": np.asarray(msg.k, dtype=np.float64).reshape(3, 3),
                "D": np.asarray(msg.d, dtype=np.float64).ravel(),
                "frame_id": str(msg.header.frame_id),
            }
        return
    if kind == "gyro":
        h_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        av = msg.angular_velocity
        imu_gyro.append((h_ns * 1e-9, np.array([av.x, av.y, av.z], dtype=np.float64)))
        return

    h_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
    if rx_ns - h_ns > STALE_PAIR_NS:
        dropped[role]["colour_stale" if kind == "rgb" else "depth_stale"] += 1
        return
    img = _decode_image_msg(msg)
    if kind == "rgb":
        raw_rgb[role][h_ns] = (rx_ns, img)
    elif kind == "depth":
        raw_depth[role][h_ns] = (rx_ns, img)

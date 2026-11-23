"""Read-only access to ROS 2 Jazzy SQLite bags without a ROS installation.

The database is opened with SQLite's immutable read-only URI, so the original
recording is never modified. Messages are decoded with the rosbags Jazzy type store.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
from rosbags.typesys import Stores, get_typestore

TYPESTORE = get_typestore(Stores.ROS2_JAZZY)
ROOT = Path(__file__).resolve().parents[2] / "WP3-Lab-Recordings" / "Recordings"
BAGS = {
    "baseline": ROOT / "three_cam_demo_bag" / "three_cam_demo_bag_0.db3",
    "movement": ROOT / "three_cam_demo_bag_move_20261112_162844"
    / "three_cam_demo_bag_move_20261112_162844_0.db3",
    "one_minute": ROOT / "one_minute_take_20261112_163920" / "bag" / "bag_0.db3",
}


class Bag:
    def __init__(self, name_or_path: str | Path):
        path = Path(BAGS.get(str(name_or_path), Path(name_or_path)))
        if path.is_dir():
            dbs = sorted(path.glob("*.db3"))
            if not dbs:
                raise FileNotFoundError(f"No .db3 in {path} (record with --storage sqlite3)")
            path = dbs[0]
        self.path = path
        self.db = sqlite3.connect(f"file:{self.path}?mode=ro&immutable=1", uri=True)
        self.topics = {n: (i, t) for i, n, t in self.db.execute("select id,name,type from topics")}

    def has(self, topic: str) -> bool:
        return topic in self.topics

    def count(self, topic: str) -> int:
        return self.db.execute("select count(*) from messages where topic_id=?", (self.topics[topic][0],)).fetchone()[0]

    def stamps(self, topic: str) -> np.ndarray:
        rows = self.db.execute("select timestamp from messages where topic_id=? order by timestamp", (self.topics[topic][0],))
        return np.array([r[0] for r in rows], dtype=np.int64)

    def messages(self, topic: str, t0: int | None = None, t1: int | None = None, raw: bool = False):
        """Yield (bag_receive_time_ns, message) in bag order."""
        tid, typ = self.topics[topic]
        q, a = "select timestamp,data from messages where topic_id=?", [tid]
        if t0 is not None:
            q += " and timestamp>=?"; a.append(int(t0))
        if t1 is not None:
            q += " and timestamp<=?"; a.append(int(t1))
        for ts, data in self.db.execute(q + " order by timestamp", a):
            yield ts, (data if raw else TYPESTORE.deserialize_cdr(data, typ))

    def nearest(self, topic: str, t_ns: int):
        """Message whose bag time is nearest to t_ns."""
        tid, typ = self.topics[topic]
        row = self.db.execute(
            "select timestamp,data from messages where topic_id=? order by abs(timestamp-?) limit 1", (tid, int(t_ns))
        ).fetchone()
        return row[0], TYPESTORE.deserialize_cdr(row[1], typ)

    def header_index(self, topic: str) -> np.ndarray:
        """(bag_ns, header_stamp_ns) for every message, parsed from the raw CDR header (fast)."""
        import struct
        key = ("_hidx", topic)
        if not hasattr(self, "_cache"):
            self._cache = {}
        if key not in self._cache:
            rows = self.db.execute("select timestamp, substr(data,1,12) from messages where topic_id=? order by timestamp",
                                   (self.topics[topic][0],))
            out = []
            for ts, head in rows:
                sec, nsec = struct.unpack_from("<iI", head, 4)
                out.append((ts, sec * 1_000_000_000 + nsec))
            self._cache[key] = np.array(out, dtype=np.int64)
        return self._cache[key]

    def by_header_stamp(self, topic: str, stamp_ns: int):
        """Message whose header stamp is nearest to stamp_ns -> (bag_ns, header_ns, msg)."""
        idx = self.header_index(topic)
        k = int(np.argmin(np.abs(idx[:, 1] - int(stamp_ns))))
        bag_ns = int(idx[k, 0])
        tid, typ = self.topics[topic]
        data = self.db.execute("select data from messages where topic_id=? and timestamp=?", (tid, bag_ns)).fetchone()[0]
        return bag_ns, int(idx[k, 1]), TYPESTORE.deserialize_cdr(data, typ)

    def json_messages(self, topic: str):
        for ts, m in self.messages(topic):
            yield ts, json.loads(m.data)


def stamp_s(header) -> float:
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def cloud_xyz(msg) -> np.ndarray:
    """XYZ float32 from a PointCloud2 (handles padded xyz/xyzrgb layouts)."""
    fields = {f.name: f for f in msg.fields}
    step = msg.point_step
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(-1, step)
    out = np.empty((len(buf), 3), dtype=np.float32)
    for k, n in enumerate("xyz"):
        o = fields[n].offset
        out[:, k] = buf[:, o:o + 4].copy().view(np.float32)[:, 0]
    return out


def cloud_rgb(msg) -> np.ndarray | None:
    fields = {f.name: f for f in msg.fields}
    if "rgb" not in fields:
        return None
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(-1, msg.point_step)
    o = fields["rgb"].offset
    return buf[:, [o + 2, o + 1, o]].copy()  # BGRA little-endian packed -> RGB


def image(msg) -> np.ndarray:
    data = np.frombuffer(bytes(msg.data), dtype=np.uint16 if msg.encoding in ("16UC1", "mono16") else np.uint8)
    ch = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1, "16UC1": 1, "mono16": 1}[msg.encoding]
    arr = data.reshape(msg.height, -1)[:, : msg.width * ch].reshape(msg.height, msg.width, ch) if ch > 1 else \
        data.reshape(msg.height, -1)[:, : msg.width]
    return arr


def bgr(msg) -> np.ndarray:
    arr = image(msg)
    if msg.encoding == "rgb8":
        return arr[:, :, ::-1].copy()
    if msg.encoding == "rgba8":
        return arr[:, :, [2, 1, 0]].copy()
    if msg.encoding == "bgra8":
        return arr[:, :, :3].copy()
    return arr


def first_message_ns(name_or_path) -> int:
    """Label/video time base: bag time of the first message of any topic."""
    return Bag(name_or_path).db.execute("select min(timestamp) from messages").fetchone()[0]

"""Read a saved wrist ROS 2 SQLite bag and measure RGB-D and IMU agreement."""
from __future__ import annotations

import argparse
from contextlib import closing
import json
from pathlib import Path
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OUTPUT_ROOT / "WP3-Offline-Analysis-2026-11-12" / "tools"))
from bagio import Bag, bgr, image
from projection import chain, static_tf

DEFAULT_CAPTURE = OUTPUT_ROOT / "WP3-Lab-Recordings" / "2026-11-17-wrist-imu" / "bag" / "bag_0.db3"
DEFAULT_OUTPUT = Path(__file__).resolve().parent
PLANE_TOLERANCE_M = 0.004
MIN_PLANE_SUPPORT = 0.55
MOTION_WINDOW_NS = 180_000_000
STILL_GYRO_DEG_S = 5.0
STILL_ACCEL_TOLERANCE_M_S2 = 0.3
EXAMPLE_TIMES_S = (5, 28, 34, 40, 52, 66)


def stamp_ns(msg):
    stamp = msg.header.stamp
    return stamp.sec * 1_000_000_000 + stamp.nanosec


def measure_stream(bag, name):
    index = bag.header_index(name)
    times = index[:, 1]
    arrival = index[:, 0]
    if len(times) < 2:
        raise ValueError(f"{name} needs two or more messages")
    intervals = np.diff(times) / 1e6
    return {
        "messages": len(times),
        "frequency_hz": float((len(times) - 1) * 1e9 / (times[-1] - times[0])),
        "median_header_gap_ms": float(np.median(intervals)),
        "p95_header_gap_ms": float(np.percentile(intervals, 95)),
        "max_header_gap_ms": float(intervals.max()),
        "nonincreasing_header_steps": int(np.sum(intervals <= 0)),
        "median_bag_minus_header_ms": float(np.median(arrival - times) / 1e6),
        "p95_bag_minus_header_ms": float(np.percentile(arrival - times, 95) / 1e6),
    }


def imu_samples(bag, topic, field):
    values = []
    headers = []
    frame = None
    for _, msg in bag.messages(topic):
        sample = getattr(msg, field)
        headers.append(stamp_ns(msg))
        values.append((sample.x, sample.y, sample.z))
        frame = msg.header.frame_id
    order = np.argsort(headers, kind="stable")
    return np.asarray(headers)[order], np.asarray(values)[order], frame


def rotation_to_colour(graph, colour_frame, source_frame):
    parents = {child: parent for parent, child in graph}
    common_root = colour_frame
    while common_root in parents:
        common_root = parents[common_root]
    colour_in_root = chain(graph, common_root, colour_frame)
    source_in_root = chain(graph, common_root, source_frame)
    return colour_in_root[:3, :3].T @ source_in_root[:3, :3]


def ray_grid(info):
    pixels = np.indices((info.height, info.width))[::-1].transpose(1, 2, 0).astype(np.float32)
    intrinsics = np.asarray(info.k).reshape(3, 3)
    xy = cv2.undistortPoints(
        pixels.reshape(-1, 1, 2), intrinsics, np.asarray(info.d)
    ).reshape(info.height, info.width, 2)
    return np.concatenate((xy, np.ones_like(xy[:, :, :1])), axis=2), intrinsics


def crop_slices(roi, width, height):
    x0, y0, x1, y1 = roi
    return slice(round(y0 * height), round(y1 * height)), slice(round(x0 * width), round(x1 * width))


def fit_plane(points, triples):
    if len(points) < 120:
        return None
    a, b, c = points[triples % len(points)].transpose(1, 0, 2)
    normals = np.cross(b - a, c - a)
    lengths = np.linalg.norm(normals, axis=1)
    usable = lengths > 1e-5
    normals = normals[usable] / lengths[usable, None]
    offsets = -np.einsum("ij,ij->i", normals, a[usable])
    if not len(normals):
        return None
    distances = np.abs(points @ normals.T + offsets)
    counts = np.sum(distances < PLANE_TOLERANCE_M, axis=0)
    best = int(np.argmax(counts))
    inliers = points[distances[:, best] < PLANE_TOLERANCE_M]
    if len(inliers) / len(points) < MIN_PLANE_SUPPORT:
        return None
    centre = inliers.mean(axis=0)
    _, _, axes = np.linalg.svd(inliers - centre, full_matrices=False)
    normal = axes[-1]
    if normal[2] > 0:
        normal = -normal
    offset = float(-centre @ normal)
    residuals = np.abs(points @ normal + offset)
    mask = residuals < PLANE_TOLERANCE_M
    if np.mean(mask) < MIN_PLANE_SUPPORT:
        return None
    return normal, offset, float(np.mean(mask)), float(1000 * np.median(residuals[mask]))


def height_components(height_m, valid, roi, nominal_height_m):
    """Return connected regions near the specified object-top height in image space."""
    mask = np.zeros_like(valid, dtype=np.uint8)
    lower, upper = nominal_height_m * 0.68, nominal_height_m * 1.40
    mask[roi] = (valid & (height_m >= lower) & (height_m <= upper))[roi]
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    count, labels, boxes, _ = cv2.connectedComponentsWithStats(mask)
    return [(boxes[k, :4], float(np.median(height_m[labels == k]) * 1000))
            for k in range(1, count) if boxes[k, cv2.CC_STAT_AREA] >= 65]


def draw_example(color, components, time_s):
    view = bgr(color)
    for (x, y, w, h), height in components:
        cv2.rectangle(view, (x, y), (x + w, y + h), (30, 220, 255), 2)
        cv2.putText(view, f"{height:.0f} mm", (x, max(y - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (30, 220, 255), 1, cv2.LINE_AA)
    return time_s, cv2.cvtColor(view, cv2.COLOR_BGR2RGB), len(components)


def plot_results(output, rows, gyro_times, gyro_speed, start_ns, examples):
    colour = "#106d80"
    time_s = np.array([r["time_s"] for r in rows])
    fit_distance = [r.get("plane_distance_mm", np.nan) for r in rows]
    tilt = [r.get("table_gravity_angle_deg", np.nan) for r in rows]
    have_gyro = len(gyro_times) > 0
    height = 7.6 if have_gyro else 3.2
    fig, axes = plt.subplots(3 if have_gyro else 1, 1, figsize=(10, height),
                             sharex=True, squeeze=False, constrained_layout=True)
    axes = axes[:, 0]
    title = "Wrist camera | RGB-D and IMU check" if have_gyro else "Wrist camera | RGB-D check"
    fig.suptitle(title, fontsize=16, fontweight="bold")
    plane_axis = axes[1] if have_gyro else axes[0]
    plane_axis.plot(time_s, fit_distance, ".", color=colour, markersize=3)
    plane_axis.set_ylabel("Camera to plane\n(mm)")
    if have_gyro:
        axes[0].plot((gyro_times - start_ns) / 1e9, gyro_speed, color="#bc5507", linewidth=0.8)
        axes[0].axhline(STILL_GYRO_DEG_S, color="#606c78", linestyle=":", linewidth=1)
        axes[0].set_ylabel("Angular speed\n(deg/s)")
        axes[2].plot(time_s, tilt, ".", color="#2d5c3e", markersize=3)
        axes[2].set_ylabel("Table normal vs\nIMU gravity (deg)")
    axes[-1].set_xlabel("Seconds since first colour frame (camera header time)")
    for ax in axes:
        ax.grid(axis="y", alpha=0.23)
        ax.spines[["right", "top"]].set_visible(False)
    fig.savefig(output / "wrist_capture_summary.png", dpi=155)
    plt.close(fig)

    if examples:
        fig, axes = plt.subplots((len(examples) + 2) // 3, 3, figsize=(12, 5.4),
                                 squeeze=False, constrained_layout=True)
        fig.suptitle("Wrist RGB-D | connected regions near the expected top height",
                     fontsize=15, fontweight="bold")
        for ax, (t, frame, count) in zip(axes.flat, examples):
            ax.imshow(frame)
            unit = "region" if count == 1 else "regions"
            ax.set_title(f"{t:.1f} s  •  {count} candidate {unit}", fontsize=11)
        for ax in axes.flat:
            ax.axis("off")
        fig.savefig(output / "wrist_height_examples.png", dpi=150)
        plt.close(fig)


def analyse(bag, roi, block_height_m, example_times, output):
    prefix = "/perception/wrist"
    topics = {key: f"{prefix}/{suffix}" for key, suffix in {
        "rgb": "color/image_raw", "depth": "aligned_depth_to_color/image_raw",
        "aligned_info": "aligned_depth_to_color/camera_info", "colour_info": "color/camera_info",
        "gyro": "gyro/sample",
        "accel": "accel/sample",
    }.items()}
    streams = {key: measure_stream(bag, topics[key]) for key in ("rgb", "depth")}
    info_topic = next(
        (topics[key] for key in ("aligned_info", "colour_info")
         if bag.has(topics[key]) and bag.count(topics[key])), None)
    if info_topic is None:
        raise ValueError("The bag needs wrist aligned-depth or colour CameraInfo")
    _, info = next(bag.messages(info_topic))
    rays, intrinsics = ray_grid(info)
    roi = crop_slices(roi, info.width, info.height)
    depth_index = bag.header_index(topics["depth"])
    depth_by_stamp = {int(h): int(t) for t, h in depth_index}

    transforms = static_tf(bag) if bag.has("/tf_static") else {}
    g_ns, gyro_speed = np.array([], dtype=np.int64), np.array([])
    accel_rotation = None
    if all(bag.has(topics[key]) and bag.count(topics[key]) >= 2 for key in ("gyro", "accel")):
        streams.update({key: measure_stream(bag, topics[key]) for key in ("gyro", "accel")})
        g_ns, gyro, _ = imu_samples(bag, topics["gyro"], "angular_velocity")
        a_ns, accel, accel_frame = imu_samples(bag, topics["accel"], "linear_acceleration")
        gyro_speed = np.rad2deg(np.linalg.norm(gyro, axis=1))
        try:
            accel_rotation = rotation_to_colour(transforms, info.header.frame_id, accel_frame)
        except KeyError:
            pass

    triples = np.random.default_rng(18).integers(0, 2048, size=(120, 3))
    rows = []
    examples = []
    example_target = iter(example_times)
    next_example = next(example_target, None)
    start = int(bag.header_index(topics["rgb"])[0, 1])
    for receipt_time, color in bag.messages(topics["rgb"]):
        h = stamp_ns(color)
        t = (h - start) / 1e9
        row = {"time_s": t}
        rows.append(row)
        depth_receipt = depth_by_stamp.get(h)
        if depth_receipt is None:
            continue
        _, depth_msg = next(bag.messages(topics["depth"], depth_receipt, depth_receipt))
        if (depth_msg.encoding != "16UC1" or (depth_msg.height, depth_msg.width) != rays.shape[:2]
                or depth_msg.header.frame_id != info.header.frame_id):
            raise ValueError("Expected 16UC1 aligned depth matching the CameraInfo shape and optical frame")
        depth = image(depth_msg).astype(np.float32) * 0.001
        valid = (depth > 0.2) & (depth < 1.0)
        row["depth_valid_roi_pct"] = float(100 * np.mean(valid[roi]))
        row["depth_minus_rgb_bag_ms"] = (depth_receipt - receipt_time) / 1e6
        points = rays * depth[:, :, None]
        subpoints, subvalid = points[roi][::3, ::3], valid[roi][::3, ::3]
        fit = fit_plane(subpoints[subvalid], triples)
        if fit is None:
            continue
        normal, offset, fraction, residual = fit
        row.update(plane_distance_mm=offset * 1000, plane_support_pct=fraction * 100,
                   plane_inlier_residual_mm=residual)

        if accel_rotation is not None:
            ends = [h - MOTION_WINDOW_NS, h + MOTION_WINDOW_NS]
            g0, g1 = np.searchsorted(g_ns, ends)
            a0, a1 = np.searchsorted(a_ns, ends)
            if g1 > g0 and a1 > a0:
                angular_speed = float(np.percentile(gyro_speed[g0:g1], 95))
                gravity = accel_rotation @ np.median(accel[a0:a1], axis=0)
                norm_g = np.linalg.norm(gravity)
                row["gyro_window_p95_deg_s"] = angular_speed
                if angular_speed < STILL_GYRO_DEG_S and abs(norm_g - 9.81) < STILL_ACCEL_TOLERANCE_M_S2:
                    cosine = np.clip(abs(float(normal @ gravity)) / norm_g, 0, 1)
                    row["table_gravity_angle_deg"] = float(np.rad2deg(np.arccos(cosine)))
        height = points @ normal + offset
        components = height_components(height, valid, roi, block_height_m)
        row["height_candidate_regions"] = len(components)
        if next_example is not None and t >= next_example:
            examples.append(draw_example(color, components, t))
            next_example = next(example_target, None)

    def summary(key):
        values = [r[key] for r in rows if key in r]
        if not values:
            return {"frames": 0}
        low, med, high = map(float, np.percentile(values, [5, 50, 95]))
        return {"frames": len(values), "p05": low, "median": med, "p95": high,
                "min": min(values), "max": max(values)}

    topic_counts = {name: bag.count(name) for name in bag.topics}
    result = {
        "source": str(bag.path),
        "camera_info": {"topic": info_topic, "width": info.width, "height": info.height, "fx_px": float(intrinsics[0, 0]),
                        "fy_px": float(intrinsics[1, 1]), "distortion_model": info.distortion_model},
        "streams": streams,
        "bag_topic_counts": topic_counts,
        "static_transform_frames": [list(pair) for pair in transforms],
        "rgb_depth_exact_header_pairs": sum("depth_valid_roi_pct" in row for row in rows),
        "rgb_header_span_s": rows[-1]["time_s"],
        "gyro_peak_deg_s": float(np.max(gyro_speed)) if len(gyro_speed) else None,
        "roi_xyxy_px": [roi[1].start, roi[0].start, roi[1].stop, roi[0].stop],
        "pilot_nominal_block_height_mm": 1000 * block_height_m,
        "metrics": {key: summary(key) for key in (
            "depth_valid_roi_pct", "depth_minus_rgb_bag_ms", "plane_distance_mm", "plane_support_pct",
            "plane_inlier_residual_mm", "table_gravity_angle_deg", "gyro_window_p95_deg_s",
            "height_candidate_regions")},
        "height_examples": [{"time_s": t, "candidate_regions": count} for t, _, count in examples],
        "frames": rows,
    }
    plot_results(output, rows, g_ns, gyro_speed, start, examples)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db3", nargs="?", type=Path, default=DEFAULT_CAPTURE,
                        help="a completed ROS 2 SQLite bag file with wrist RGB-D streams")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--roi", type=float, nargs=4, default=(0.24, 0.08, 0.89, 0.77),
                        help="x0 y0 x1 y1 as fractions of the image")
    parser.add_argument("--block-height-mm", type=float, default=38.0,
                        help="nominal object-top height for the height-only candidate count")
    parser.add_argument("--examples-s", type=float, nargs="+", default=EXAMPLE_TIMES_S,
                        help="up to six example times in seconds since the first colour frame")
    args = parser.parse_args()
    x0, y0, x1, y1 = args.roi
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1 and args.block_height_mm > 0):
        parser.error("Use an ordered normalised ROI and a positive block height")
    if len(args.examples_s) > 6 or any(t < 0 for t in args.examples_s):
        parser.error("Use up to six nonnegative example times")
    if args.db3.suffix != ".db3" or not args.db3.is_file():
        parser.error("Provide the path to a completed .db3 file")
    args.output.mkdir(exist_ok=True, parents=True)
    cv2.setNumThreads(2)
    bag = Bag(args.db3.resolve())
    with closing(bag.db):
        result = analyse(bag, args.roi, args.block_height_mm * 0.001, sorted(args.examples_s), args.output)
    dest = args.output / "wrist_capture_metrics.json"
    dest.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(dest)
    print(json.dumps({key: value for key, value in result.items() if key != "frames"}, indent=2))


if __name__ == "__main__":
    main()

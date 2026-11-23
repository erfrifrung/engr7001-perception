"""Fit a provisional static board frame from synchronized live RGB-D."""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener


ROLES = ("wrist", "tripod", "third")
UNIT_GRID = np.array([[(x - 3.5), (2.5 - y), 0]
                      for y in range(6) for x in range(8)])
VALIDATE = np.array([i % 2 == (i // 8) % 2 for i in range(48)])


def stamp(msg):
    return msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec


def projection(points, camera_from_board, k, d):
    rvec = cv2.Rodrigues(camera_from_board[:3, :3])[0]
    return cv2.projectPoints(points, rvec, camera_from_board[:3, 3], k, d)[0].reshape(-1, 2)


def matrix(transform):
    tr, q = transform.translation, transform.rotation
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    result[:3, 3] = [tr.x, tr.y, tr.z]
    return result


def fit(role, samples, model, seed, optical_from_link, out, grid, flip=False):
    k, d = model
    seed_pixels = projection(grid, np.linalg.inv(seed), k, d) if seed is not None else None
    frames = []
    for rgb, depth in samples:
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        ok, pixels = cv2.findChessboardCornersSB(gray, (8, 6), cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not ok:
            continue
        pixels = pixels.reshape(-1, 2)
        orientation_ref = seed_pixels if seed_pixels is not None else (frames[0][0] if frames else pixels)
        if np.mean((pixels[::-1] - orientation_ref)**2) < np.mean((pixels - orientation_ref)**2):
            pixels = pixels[::-1].copy()
        # Keep the unflipped order for comparing subsequent exposures.
        frames.append((pixels, rgb, depth))
    if flip:
        frames = [(pixels[::-1].copy(), rgb, depth) for pixels, rgb, depth in frames]
    if len(frames) < 4:
        raise RuntimeError(f"{role}: only {len(frames)}/{len(samples)} complete boards")

    # Fit one exposure on alternating corners. Hold out other corners and exposures.
    ref, rgb, _ = frames[0]
    used = ~VALIDATE
    ok, rvec, tvec = cv2.solvePnP(grid[used], ref[used], k, d)
    if not ok:
        raise RuntimeError(f"{role}: board pose fit failed")
    cam_from_board = np.eye(4)
    cam_from_board[:3, :3] = cv2.Rodrigues(rvec)[0]
    cam_from_board[:3, 3] = tvec.ravel()
    board_from_cam = np.linalg.inv(cam_from_board)
    pixels_fit = projection(grid, cam_from_board, k, d)
    holdout_errors = np.concatenate([
        np.linalg.norm(corners[VALIDATE] - pixels_fit[VALIDATE], axis=1)
        for corners, _, _ in frames[1:]
    ])
    stability = np.concatenate([np.linalg.norm(corners - ref, axis=1) for corners, _, _ in frames[1:]])

    # Independent depth pixels inside the board. The board's metric scale uses nominal pitch.
    mask = np.zeros(rgb.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, np.rint(ref[[9, 14, 38, 33]]).astype(np.int32), 1)
    v, u = np.nonzero(mask)
    xy = cv2.undistortPoints(np.column_stack((u, v)).astype(float).reshape(-1, 1, 2), k, d).reshape(-1, 2)
    rays = np.column_stack((xy, np.ones(len(u))))
    depth = np.stack([depth[v, u] for _, _, depth in frames[1:]])
    projected_height_mm = depth * (rays @ board_from_cam[2, :3]) + 1000 * board_from_cam[2, 3]
    height_pct = np.percentile(projected_height_mm[depth > 0], [5, 50, 95])
    overlay = rgb.copy()
    for idx, pt in enumerate(ref):
        cv2.circle(overlay, tuple(np.rint(pt).astype(int)), 3, (0, 185, 255) if VALIDATE[idx] else (0, 240, 0), -1)
    for idx in (0, 7, 40, 47):
        x, y = np.rint(ref[idx]).astype(int)
        cv2.putText(overlay, str(idx), (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 60, 255), 1)
    cv2.imwrite(str(out / f"{role}_board.png"), overlay)
    info = dict(
        board_frames=len(frames),
        heldout_corners_per_frame=int(VALIDATE.sum()),
        heldout_reprojection_rms_px=float(np.sqrt(np.mean(holdout_errors**2))),
        heldout_reprojection_p95_px=float(np.percentile(holdout_errors, 95)),
        inter_frame_corner_drift_p95_px=float(np.percentile(stability, 95)),
        board_depth_nonzero_pct=float(100 * np.mean(depth > 0)),
        board_normal_depth_error_mm_p05_p50_p95=height_pct.tolist(),
        board_from_optical=board_from_cam.tolist(),
        corner_order_uses_reference=seed is not None,
        operator_reversed_order=flip,
    )
    return info, board_from_cam @ optical_from_link


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path, help="New directory for the fit and corner overlays")
    parser.add_argument("reference", type=Path, help="Saved board orientation reference")
    parser.add_argument("--pitch-mm", type=float, required=True, help="Physically checked square pitch")
    parser.add_argument("--roles", nargs="+", choices=ROLES, default=ROLES)
    parser.add_argument("--flip", action="append", choices=ROLES, default=[],
                        help="Reverse this camera's corner order after checking corner 0 in its overlay")
    args = parser.parse_args()
    if not np.isfinite(args.pitch_mm) or args.pitch_mm <= 0:
        parser.error("--pitch-mm must be positive and finite")
    if len(set(args.roles)) != len(args.roles) or set(args.flip) - set(args.roles):
        parser.error("Roles must be unique and --flip must name a selected role")
    out, roles = args.out, args.roles
    out.mkdir(parents=True, exist_ok=False)
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    seeds = reference.get("diagnostics", reference.get("poses", {}))
    grid = UNIT_GRID * (args.pitch_mm * .001)
    cv2.setNumThreads(1)
    rclpy.init()
    node = rclpy.create_node("wp3_stationary_board_registration")
    buffer = Buffer()
    listener = TransformListener(buffer, node)
    messages = {role: {} for role in roles}
    info = {}
    samples = {role: [] for role in roles}
    last = {role: 0 for role in roles}
    bridge = CvBridge()

    def receive(msg, role, kind):
        if stamp(msg) - last[role] < 350_000_000:
            return
        cache = messages[role]
        pair = cache.setdefault(stamp(msg), {})
        pair[kind] = msg
        if len(pair) == 2 and len(samples[role]) < 7:
            last[role] = stamp(msg)
            rgb = bridge.imgmsg_to_cv2(pair["rgb"], desired_encoding="bgr8")
            depth = bridge.imgmsg_to_cv2(pair["depth"], desired_encoding="passthrough")
            if depth.dtype == np.uint16 and rgb.shape[:2] == depth.shape:
                samples[role].append((rgb, depth))
            cache.clear()
        while len(cache) > 15:
            cache.pop(next(iter(cache)))

    try:
        for role in roles:
            ns = f"/perception/{role}"
            for kind, topic in (("rgb", "color/image_raw"), ("depth", "aligned_depth_to_color/image_raw")):
                node.create_subscription(Image, f"{ns}/{topic}", lambda msg, r=role, k=kind: receive(msg, r, k),
                                         qos_profile_sensor_data)
            node.create_subscription(CameraInfo, f"{ns}/color/camera_info",
                                     lambda msg, r=role: info.update({r: msg}), qos_profile_sensor_data)

        deadline = time.monotonic() + 22
        while time.monotonic() < deadline and (len(info) < len(roles) or min(map(len, samples.values())) < 7):
            rclpy.spin_once(node, timeout_sec=.15)
        result = dict(parent="jenga_table", pitch_m=args.pitch_mm * .001, transforms={},
                      unoriented_transforms={}, diagnostics={}, errors={})
        for role in roles:
            try:
                if role not in info:
                    raise RuntimeError("No camera info received")
                camera = info[role]
                optical = camera.header.frame_id
                link = f"{role}_link"
                model = np.asarray(camera.k).reshape(3, 3), np.asarray(camera.d)
                if not optical or any(rgb.shape[:2] != (camera.height, camera.width) for rgb, _ in samples[role]):
                    raise RuntimeError("CameraInfo frame or image size does not match RGB-D")
                tr = buffer.lookup_transform(optical, link, Time()).transform
                seed = np.asarray(seeds[role]["board_from_optical"]) if role in seeds else None
                diagnostic, board_from_link = fit(role, samples[role], model, seed, matrix(tr), out, grid, role in args.flip)
                target = "transforms" if seed is not None else "unoriented_transforms"
                result[target][link] = board_from_link.tolist()
                result["diagnostics"][role] = diagnostic
            except (RuntimeError, ValueError, cv2.error, TransformException) as exc:
                result["errors"][role] = str(exc)
        qualified = not result["errors"] and not result["unoriented_transforms"] and all(
            item["heldout_reprojection_p95_px"] < 1.5 and item["inter_frame_corner_drift_p95_px"] < 1.5
            and item["board_depth_nonzero_pct"] > 90
            and abs(item["board_normal_depth_error_mm_p05_p50_p95"][1]) < 8
            for item in result["diagnostics"].values()
        )
        result["passed_stationary_smoke_gate"] = qualified
        (out / "current_board_tf.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result["diagnostics"], indent=2), flush=True)
        if not qualified:
            raise SystemExit("Current RGB-D board checks exceeded the stationary smoke thresholds")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

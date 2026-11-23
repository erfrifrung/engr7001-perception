"""Publish the camera-link edges of a checked stationary checkerboard fit."""

import argparse
import json
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.executors import ExternalShutdownException
from scipy.spatial.transform import Rotation
from tf2_ros import StaticTransformBroadcaster


def load_board_fit(path: Path) -> dict[str, np.ndarray]:
    fit = json.loads(path.read_text(encoding="utf-8"))
    if not fit.get("passed_stationary_smoke_gate") or fit.get("parent") != "jenga_table":
        raise ValueError("The fit must pass register_board.py and use jenga_table")
    links = fit.get("transforms", {})
    expected_links = {"wrist_link", "tripod_link", "third_link"}
    if not links or set(links) - expected_links:
        raise ValueError("Expected camera-link transforms from register_board.py")
    matrices = {link: np.asarray(value, dtype=float) for link, value in links.items()}
    for link, transform in matrices.items():
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError(f"{link}: expected a finite 4 by 4 transform")
        rotation = transform[:3, :3]
        if (
            not np.allclose(transform[3], [0, 0, 0, 1])
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1.0)
        ):
            raise ValueError(f"{link}: expected a rigid transform")
    return matrices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit", type=Path)
    args = parser.parse_args()
    try:
        matrices = load_board_fit(args.fit)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    rclpy.init()
    node = rclpy.create_node("wp3_hazem_static_board")
    try:
        broadcaster = StaticTransformBroadcaster(node)
        edges = []
        for link, transform in matrices.items():
            msg = TransformStamped()
            msg.header.frame_id = "jenga_table"
            msg.header.stamp = node.get_clock().now().to_msg()
            msg.child_frame_id = link
            pos, rot = msg.transform.translation, msg.transform.rotation
            pos.x, pos.y, pos.z = map(float, transform[:3, 3])
            rot.x, rot.y, rot.z, rot.w = map(float, Rotation.from_matrix(transform[:3, :3]).as_quat())
            edges.append(msg)
        broadcaster.sendTransform(edges)
        node.get_logger().info(f"Stationary TF from {args.fit} for {', '.join(matrices)}")
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

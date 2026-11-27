#!/usr/bin/env python3
"""WP3 multi-block perception -> MoveIt planning scene.

Successor to perception_bridge.py for more than one block. It listens to the markers WP3's live node
already publishes, puts every block (CUBE marker) into MoveIt as its own collision object with its own
measured size and full pose, and keeps blocks the cameras lose as "last known" (greyed out) instead of
deleting them. perception_bridge.py is left as it was for the single-block demo.

  topic   /wp3_live/markers       visualization_msgs/msg/MarkerArray   (frame: jenga_table, from WP3 v2)
  needs   TF world -> jenga_table  (python3 wp3_table_tf.py, after fit_table_tf.py)
  writes  collision objects 'wp3_<ns>_<id>' in the planning scene, frame `world`

  python3 wp3_scene_bridge.py                          all CUBE markers are blocks
  python3 wp3_scene_bridge.py --ns tracks               only CUBE markers in namespace 'tracks'
  python3 wp3_scene_bridge.py --remove-after 10         drop blocks unseen for 10 s (default: keep them)
  python3 wp3_scene_bridge.py --remove-demo-block       delete the fixed 'jenga_block' from build_scene.py

Non-CUBE markers (outlines of ambiguous clusters, text, arrows) are left for RViz to draw and are NOT
added to the planning scene. Whether WP5 may plan against stale or ambiguous blocks is a WP3/WP5
interface decision; this bridge only shows them.
"""
import argparse
import json
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, ObjectColor, PlanningScene, PlanningSceneComponents
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.time import Time
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import ColorRGBA, String
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

import build_scene as bs
from wp3_link import BlockTracker, matrix_from_quat, quat_from_matrix, rotz_quat, scene_id

TOPIC = '/wp3_live/markers'
SCENE_TOPIC = '/wp3_live/scene_observation'
DEFAULT_NS = ('wp3_blocks', 'blocks', 'tracks')
LIVE = (0.95, 0.55, 0.10, 1.0)      # orange: currently seen by WP3
STALE = (0.55, 0.55, 0.55, 0.35)    # faded grey: last known pose, not currently seen
FINGER_LINKS = ['left_outer_knuckle', 'left_inner_knuckle', 'left_finger',
                'right_outer_knuckle', 'right_inner_knuckle', 'right_finger', 'link_tcp']


class WP3SceneBridge(Node):
    def __init__(self, args):
        super().__init__('wp3_scene_bridge')
        self.args = args
        self.frame = bs.FRAME
        self.tracker = BlockTracker(stale_s=args.stale_after, remove_s=args.remove_after)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.get_scene = self.create_client(GetPlanningScene, '/get_planning_scene')
        self.apply_scene = self.create_client(ApplyPlanningScene, '/apply_planning_scene')
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        if args.use_scene_json:
            self.create_subscription(String, args.scene_topic, self.on_scene_json, qos)
            self.get_logger().info(f'listening on {args.scene_topic} (JSON contract); blocks go into frame "{self.frame}"')
        else:
            self.create_subscription(MarkerArray, args.topic, self.on_markers, qos)
            self.get_logger().info(f'listening on {args.topic} (ns={args.ns}); blocks go into frame "{self.frame}"')
            # Markers carry no status. WP3's health topic says whether the cameras are delivering; while they are
            # not, an empty marker snapshot means "cannot see", not "blocks left", so removals are held.
            self.create_subscription(String, args.health_topic, self.on_health, qos)
        self._health = None            # (monotonic time, ok) from /wp3_live/health; None = never received (mock)
        self._calib_epoch = None
        self.last_warn = {}
        self.pending_demo_removal = args.remove_demo_block
        self._was_attached = set()     # objects in the gripper at the previous flush
        self._released_at = {}         # key -> time it left the gripper (removal grace period)

    # ------------------------------------------------------------------ helpers
    def warn(self, key, text, every=5.0):
        if time.time() - self.last_warn.get(key, 0.0) > every:
            self.get_logger().warning(text)
            self.last_warn[key] = time.time()

    def to_world(self, frame_id, pose):
        if frame_id == self.frame:
            T = np.eye(4)
        else:
            try:
                tf = self.tf_buffer.lookup_transform(self.frame, frame_id, Time())
            except TransformException as e:
                self.warn('tf', f'no TF {self.frame} -> {frame_id} yet ({e}). Is wp3_table_tf.py running?')
                return None
            tr, rq = tf.transform.translation, tf.transform.rotation
            T = np.eye(4)
            T[:3, :3] = matrix_from_quat((rq.x, rq.y, rq.z, rq.w))
            T[:3, 3] = (tr.x, tr.y, tr.z)
        p, o = pose.position, pose.orientation
        R = T[:3, :3] @ matrix_from_quat((o.x, o.y, o.z, o.w))
        xyz = T[:3, :3] @ np.array([p.x, p.y, p.z]) + T[:3, 3]
        return tuple(float(v) for v in xyz), quat_from_matrix(R)

    # ------------------------------------------------------------------ input
    def on_health(self, msg):
        try:
            h = json.loads(msg.data)
        except Exception:
            return
        ep = h.get('calibration_epoch')
        if ep:
            prev_ep = getattr(self, '_calib_epoch', None)
            self._calib_epoch = ep
            if prev_ep is not None and ep != prev_ep:
                self.tracker.delete_all()
                if hasattr(self, 'get_logger'):
                    self.get_logger().info(f'calibration_epoch changed ({prev_ep} -> {ep}): clearing old-calibration poses')
        ok = h.get('status', 'ok') == 'ok' and h.get('fallback_mode') not in ('no_valid_stream', None)
        self._health = (time.monotonic(), ok)

    def wp3_can_see(self):
        """True if a snapshot may remove blocks: no health topic at all (mock / older WP3), or a recent OK one."""
        if self._health is None:
            return True
        t, ok = self._health
        return ok and (time.monotonic() - t) <= self.args.health_timeout

    def on_markers(self, msg):
        now = time.monotonic()
        # WP3 publishes DELETEALL followed by every current track: the array is a full snapshot. Blocks that
        # are no longer listed (retired tracks) must leave the planning scene, or they stay as phantom obstacles.
        snapshot = any(m.action == Marker.DELETEALL for m in msg.markers)
        listed = set()
        for m in msg.markers:
            if m.action == Marker.DELETEALL:
                continue
            if self.args.ns and m.ns not in self.args.ns:
                continue
            key = scene_id(m.ns, m.id)
            if m.action != Marker.DELETE and m.type == Marker.CUBE:
                listed.add(key)
            if m.action == Marker.DELETE:
                self.tracker.delete(key)
                continue
            if m.type != Marker.CUBE:
                continue                     # outlines, labels, arrows: RViz only
            dims = (abs(m.scale.x), abs(m.scale.y), abs(m.scale.z))
            if min(dims) <= 0.0 or max(dims) > self.args.max_size:
                self.warn('size_' + key, f'{key}: implausible size {dims} m, ignored')
                continue
            w = self.to_world(m.header.frame_id, m.pose)
            if w is None:
                continue                     # no TF yet: still listed, so the last known pose is kept
            self.tracker.observe(key, dims, w[0], w[1], now)
        if snapshot:
            if self.wp3_can_see():
                self.tracker.retain_only(listed)
            else:
                self.warn('blind', 'WP3 reports no valid camera stream (or its health went silent): '
                                   'keeping last known blocks instead of removing them')

    def on_scene_json(self, msg):
        now = time.monotonic()
        try:
            payload = json.loads(msg.data)
        except Exception as e:
            self.warn('json', f'invalid JSON on {self.args.scene_topic}: {e}')
            return
        status = payload.get('status', 'ok')
        if status != 'ok':
            # uncalibrated / no_valid_stream / no_plane / camera_moved / roi_empty: nothing in this message is
            # usable, and it is not evidence that blocks left. Keep the last known scene; it ages to STALE.
            self.warn('status', f'WP3 status "{status}" (calibration_epoch={payload.get("calibration_epoch")}): '
                                f'scene not updated')
            return
        ep = payload.get('calibration_epoch')
        if ep:
            prev_ep = getattr(self, '_calib_epoch', None)
            self._calib_epoch = ep
            if prev_ep is not None and ep != prev_ep:
                self.tracker.delete_all()
                if hasattr(self, 'get_logger'):
                    self.get_logger().info(f'calibration_epoch changed ({prev_ep} -> {ep}): clearing old-calibration poses')
        frame_id = payload.get('frame_id', 'jenga_table')
        # Every live track (MEASURED, DISPLACED, OCCLUDED, STALE) is listed in each atomic snapshot. A track that
        # is missing has been retired by WP3, so its collision object is removed.
        self.tracker.retain_only(scene_id('wp3_blocks', tr.get('track_id', 0)) for tr in payload.get('tracks', ()))
        for tr in payload.get('tracks', ()):
            state = tr.get('state', '')
            if state not in ('MEASURED', 'DISPLACED'):
                continue                     # held / stale tracks keep their last pose (not refreshed)
            # Collision safety: identity-ambiguous or not-yet-eligible blocks are still physically there, so they
            # stay obstacles by default. --commit-eligible-only is for a WP5 *target* view, not the collision world.
            if self.args.commit_eligible_only and not tr.get('snapshot_commit_eligible', False):
                continue
            tid = tr.get('track_id', 0)
            key = scene_id('wp3_blocks', tid)
            pose_tbl = tr.get('pose_table_m')
            if not pose_tbl or len(pose_tbl) != 3:
                continue
            dims_mm = tr.get('dims_mm', (75.0, 25.0))
            height_m = 2.0 * max(0.005, float(tr.get('z_top_table_rel_m', 0.015)) - float(pose_tbl[2]))
            dims = (max(dims_mm[0], dims_mm[1]) / 1000.0, min(dims_mm[0], dims_mm[1]) / 1000.0, height_m)
            p = Pose()
            p.position.x, p.position.y, p.position.z = map(float, pose_tbl)
            qx, qy, qz, qw = rotz_quat(float(tr.get('yaw_rad', 0.0)))
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = qx, qy, qz, qw
            w = self.to_world(frame_id, p)
            if w is None:
                continue
            self.tracker.observe(key, dims, w[0], w[1], now)

    # ------------------------------------------------------------------ output
    def call(self, client, request):
        while not client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info(f'waiting for {client.srv_name} (is move_group up?)')
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def attached_ids(self):
        req = GetPlanningScene.Request()
        req.components.components = PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS
        scene = self.call(self.get_scene, req).scene
        return {a.object.id for a in scene.robot_state.attached_collision_objects}

    def colour(self, scene, key, rgba):
        r, g, b, a = rgba
        scene.object_colors.append(ObjectColor(id=key, color=ColorRGBA(r=r, g=g, b=b, a=a)))

    def flush(self):
        scene = PlanningScene(is_diff=True)
        if self.pending_demo_removal:
            scene.world.collision_objects.append(
                CollisionObject(id='jenga_block', operation=CollisionObject.REMOVE))
            self.pending_demo_removal = False
        pending = bool(self.tracker.deleted) or bool(self.tracker.latest)
        # In the gripper: the twin owns it until release. Ask MoveIt first so the tracker leaves those blocks
        # pending instead of recording an update that was never applied.
        attached = self.attached_ids() if pending else set()
        now = time.monotonic()
        # A block that was just released is often not re-reported yet (WP3 does not see a block in the gripper,
        # the mock stops publishing it). Hold any removal that was waiting for it for --release-grace seconds,
        # so it is not deleted from the scene and then re-added a moment later.
        for k in self._was_attached - attached:
            self._released_at[k] = now
        self._was_attached = set(attached)
        self._released_at = {k: t for k, t in self._released_at.items() if now - t < self.args.release_grace}
        before = self.tracker.snapshot()
        actions = self.tracker.actions(now, skip=attached, hold_removals=set(self._released_at))
        if not actions and not scene.world.collision_objects:
            return
        log = []
        new_keys = []
        for act in actions:
            kind, key = act[0], act[1]
            if kind in ('add', 'move'):
                _, _, dims, xyz, quat = act
                obj = CollisionObject(id=key, operation=CollisionObject.ADD)
                obj.header.frame_id = self.frame
                obj.primitives.append(SolidPrimitive(type=SolidPrimitive.BOX, dimensions=list(dims)))
                p = Pose()
                p.position.x, p.position.y, p.position.z = xyz
                p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = quat
                obj.primitive_poses.append(p)
                scene.world.collision_objects.append(obj)
                self.colour(scene, key, LIVE)
                if kind == 'add':
                    new_keys.append(key)
                log.append(f'{kind} {key} ({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f})')
            elif kind == 'stale':
                self.colour(scene, key, STALE)
                log.append(f'stale {key}')
            elif kind == 'remove':
                scene.world.collision_objects.append(CollisionObject(id=key, operation=CollisionObject.REMOVE))
                log.append(f'remove {key}')
        if new_keys and not self.args.no_finger_contact:
            # v4: the fingers may close around a perceived block (the gripper body and the arm still may not hit it)
            req = GetPlanningScene.Request()
            req.components.components = PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
            acm = self.call(self.get_scene, req).scene.allowed_collision_matrix
            scene.allowed_collision_matrix = bs.set_acm_entries(acm, {k: FINGER_LINKS for k in new_keys})
        if scene.world.collision_objects or scene.object_colors:
            if self.call(self.apply_scene, ApplyPlanningScene.Request(scene=scene)).success:
                for line in log:
                    self.get_logger().info(line)
            else:
                self.tracker.restore(before)  # retry on the next flush instead of silently dropping obstacles
                self.get_logger().error('apply_planning_scene failed; will retry')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--topic', default=TOPIC)
    ap.add_argument('--scene-topic', default=SCENE_TOPIC)
    ap.add_argument('--use-scene-json', action='store_true', help='consume /wp3_live/scene_observation JSON directly')
    ap.add_argument('--commit-eligible-only', action='store_true',
                    help='with --use-scene-json, only insert snapshot_commit_eligible blocks. Leave OFF for the '
                         'collision world: an identity-ambiguous block is still a physical obstacle. Use it only '
                         'for a separate WP5 target view.')
    ap.add_argument('--ns', nargs='*', default=list(DEFAULT_NS), help='only these marker namespaces are blocks (excludes wp3_ambiguous by default)')
    ap.add_argument('--stale-after', type=float, default=2.0, help='s unseen before a block is greyed out')
    ap.add_argument('--remove-after', type=float, default=None, help='s unseen before a block is removed')
    ap.add_argument('--no-finger-contact', action='store_true', help='v4: do not let the fingers touch perceived blocks')
    ap.add_argument('--health-topic', default='/wp3_live/health')
    ap.add_argument('--health-timeout', type=float, default=1.0,
                    help='s; marker snapshots only remove blocks while WP3 health is OK and newer than this')
    ap.add_argument('--release-grace', type=float, default=1.0,
                    help='s after the gripper releases a block during which its pending removal is held')
    ap.add_argument('--max-size', type=float, default=0.30, help='m, larger CUBE markers are ignored')
    ap.add_argument('--rate', type=float, default=5.0, help='Hz, planning-scene update rate')
    ap.add_argument('--remove-demo-block', action='store_true')
    args = ap.parse_args()
    rclpy.init()
    node = WP3SceneBridge(args)
    period = 1.0 / args.rate
    try:
        next_flush = time.monotonic()
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            if time.monotonic() >= next_flush:
                node.flush()
                next_flush = time.monotonic() + period
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

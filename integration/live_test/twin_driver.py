#!/usr/bin/env python3
"""Scripted WP3-side driver for a live MoveIt (fake hardware) + wp3_scene_bridge.py test.

  python3 twin_driver.py markers OUT.json    marker-mode scenario (what twin.sh / start_perception.sh use)
  python3 twin_driver.py json OUT.json       /wp3_live/scene_observation scenario (--use-scene-json)
  python3 twin_driver.py watch OUT.json SECS watch the planning scene only (full-pipeline replay)

Table frame for every scenario: world -> jenga_table = (0.30, 0.00, -0.008), yaw 0 (start_perception.sh default).
Attach / release use exactly the calls of the twin's hold node (etee_hold.py try_grasp / release).
"""
import json
import sys
import time

import rclpy
from geometry_msgs.msg import Pose
from moveit_msgs.msg import AttachedCollisionObject, CollisionObject, PlanningScene, PlanningSceneComponents
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
from rclpy.node import Node
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

TABLE = (0.30, 0.00, -0.008)
SIZE = (0.075, 0.025, 0.015)
ATTACH_LINK = 'link_tcp'


class Driver(Node):
    def __init__(self):
        super().__init__('wp3_test_driver')
        self.pub_m = self.create_publisher(MarkerArray, '/wp3_live/markers', 5)
        self.pub_j = self.create_publisher(String, '/wp3_live/scene_observation', 5)
        self.pub_h = self.create_publisher(String, '/wp3_live/health', 5)
        self.get_scene = self.create_client(GetPlanningScene, '/get_planning_scene')
        self.apply_scene = self.create_client(ApplyPlanningScene, '/apply_planning_scene')
        self.log = []

    def call(self, client, req):
        client.wait_for_service(timeout_sec=10.0)
        fut = client.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
        return fut.result()

    def scene(self):
        req = GetPlanningScene.Request()
        req.components.components = (PlanningSceneComponents.WORLD_OBJECT_NAMES | PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
                                     | PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS)
        s = self.call(self.get_scene, req).scene
        world = {}
        for o in s.world.collision_objects:
            if not o.id.startswith('wp3_'):
                continue
            p = o.primitive_poses[0].position if o.primitive_poses else o.pose.position
            world[o.id] = (round(o.pose.position.x + p.x, 4), round(o.pose.position.y + p.y, 4), round(o.pose.position.z + p.z, 4))
        attached = sorted(a.object.id for a in s.robot_state.attached_collision_objects)
        return world, attached

    def acm_allows(self, oid, link):
        req = GetPlanningScene.Request()
        req.components.components = PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
        acm = self.call(self.get_scene, req).scene.allowed_collision_matrix
        if oid not in acm.entry_names or link not in acm.entry_names:
            return False
        return bool(acm.entry_values[acm.entry_names.index(oid)].enabled[acm.entry_names.index(link)])

    # ---------------------------------------------------------------- publishers
    def markers(self, blocks):
        arr = MarkerArray()
        arr.markers.append(Marker(action=Marker.DELETEALL))
        for bid, x, y in blocks:
            m = Marker()
            m.header.frame_id = 'jenga_table'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns, m.id, m.type, m.action = 'wp3_blocks', bid, Marker.CUBE, Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, SIZE[2] / 2
            m.pose.orientation.w = 1.0
            m.scale.x, m.scale.y, m.scale.z = SIZE
            arr.markers.append(m)
        self.pub_m.publish(arr)

    def json_msg(self, blocks, status='ok'):
        tracks = []
        for bid, x, y in blocks:
            tracks.append(dict(track_id=bid, block_id=f'B{bid:02d}', state='MEASURED', pose_table_m=[x, y, SIZE[2] / 2],
                               z_top_table_rel_m=SIZE[2], yaw_rad=0.0, dims_mm=[75.0, 25.0], snapshot_commit_eligible=True,
                               identity_ambiguous=False, observation_stamp_ns=self.get_clock().now().nanoseconds))
        payload = dict(schema_version='wp3_scene_observation_v2.1', frame_id='jenga_table', status=status,
                       calibration_epoch='live_test', tracks=tracks if status == 'ok' else [], ambiguous_clusters=[])
        self.pub_j.publish(String(data=json.dumps(payload)))

    def run_for(self, secs, fn, sample=None):
        """Publish fn() at 5 Hz for secs; optionally sample the scene every 0.2 s."""
        t_end = time.monotonic() + secs
        while time.monotonic() < t_end:
            fn()
            if sample is not None:
                sample.append((round(time.monotonic(), 2), self.scene()))
            rclpy.spin_once(self, timeout_sec=0.2)

    # ---------------------------------------------------------------- twin-side actions (as etee_hold.py)
    def attach(self, oid):
        aco = AttachedCollisionObject()
        aco.link_name = ATTACH_LINK
        aco.object = CollisionObject(id=oid, operation=CollisionObject.ADD)
        aco.object.header.frame_id = 'world'
        scene = PlanningScene(is_diff=True)
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects.append(aco)
        return self.call(self.apply_scene, ApplyPlanningScene.Request(scene=scene)).success

    def release(self, oid, drop_world_xyz):
        aco = AttachedCollisionObject()
        aco.link_name = ATTACH_LINK
        aco.object = CollisionObject(id=oid, operation=CollisionObject.REMOVE)
        scene = PlanningScene(is_diff=True)
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects.append(aco)
        ok = self.call(self.apply_scene, ApplyPlanningScene.Request(scene=scene)).success
        obj = CollisionObject(id=oid, operation=CollisionObject.ADD)
        obj.header.frame_id = 'world'
        obj.pose.orientation.w = 1.0
        obj.primitives.append(SolidPrimitive(type=SolidPrimitive.BOX, dimensions=list(SIZE)))
        p = Pose(); p.position.x, p.position.y, p.position.z = drop_world_xyz; p.orientation.w = 1.0
        obj.primitive_poses.append(p)
        scene = PlanningScene(is_diff=True)
        scene.world.collision_objects.append(obj)
        return ok and self.call(self.apply_scene, ApplyPlanningScene.Request(scene=scene)).success

    def cleanup(self):
        world, attached = self.scene()
        scene = PlanningScene(is_diff=True)
        scene.robot_state.is_diff = True
        for oid in attached:
            aco = AttachedCollisionObject(link_name=ATTACH_LINK, object=CollisionObject(id=oid, operation=CollisionObject.REMOVE))
            scene.robot_state.attached_collision_objects.append(aco)
        self.call(self.apply_scene, ApplyPlanningScene.Request(scene=scene))
        world, _ = self.scene()
        scene = PlanningScene(is_diff=True)
        for oid in world:
            scene.world.collision_objects.append(CollisionObject(id=oid, operation=CollisionObject.REMOVE))
        self.call(self.apply_scene, ApplyPlanningScene.Request(scene=scene))


def tw(x, y):
    """jenga_table -> world for the test table frame."""
    return (round(x + TABLE[0], 4), round(y + TABLE[1], 4), round(SIZE[2] / 2 + TABLE[2], 4))


def near(a, b, tol=0.002):
    return a is not None and all(abs(u - v) <= tol for u, v in zip(a, b))


def scenario_markers(d):
    R = {}
    B1, B2, B3 = (1, -0.04, 0.03), (2, 0.03, -0.02), (3, 0.08, 0.05)
    d.run_for(2.0, lambda: d.markers([B1, B2, B3]))
    w, _ = d.scene()
    R['1_add_three'] = dict(ok=all(near(w.get(f'wp3_blocks_{i}'), tw(x, y)) for i, x, y in (B1, B2, B3)), world=w)
    R['1b_finger_contact_acm'] = dict(ok=d.acm_allows('wp3_blocks_1', 'left_finger'))
    B1m = (1, -0.02, 0.03)
    d.run_for(1.5, lambda: d.markers([B1m, B2, B3]))
    w, _ = d.scene()
    R['2_move_20mm'] = dict(ok=near(w.get('wp3_blocks_1'), tw(-0.02, 0.03)), got=w.get('wp3_blocks_1'))
    d.run_for(2.0, lambda: d.markers([B1m, B2]))          # track 3 retired upstream
    w, _ = d.scene()
    R['3_retired_track_removed'] = dict(ok='wp3_blocks_3' not in w, world=sorted(w))
    # grasp block 2 (twin), WP3 stops reporting it (it is in the gripper) and retires the track
    R['4a_attach'] = dict(ok=d.attach('wp3_blocks_2'))
    d.run_for(2.0, lambda: d.markers([B1m]))
    w, att = d.scene()
    R['4b_held_block_untouched'] = dict(ok=att == ['wp3_blocks_2'] and 'wp3_blocks_2' not in w, attached=att, world=sorted(w))
    drop = (0.05, 0.06)
    R['4c_release'] = dict(ok=d.release('wp3_blocks_2', tw(*drop)))
    samples = []
    d.run_for(0.6, lambda: d.markers([B1m]), sample=samples)              # camera has not re-reported it yet
    d.run_for(1.4, lambda: d.markers([B1m, (2, drop[0], drop[1])]), sample=samples)   # re-reported at the drop pose
    gone = [t for t, (w_, _) in samples if 'wp3_blocks_2' not in w_]
    w, att = d.scene()
    R['4d_released_block_never_vanished'] = dict(ok=not gone, vanished_at=gone[:5])
    R['4e_released_block_at_drop_pose'] = dict(ok=near(w.get('wp3_blocks_2'), tw(*drop)), got=w.get('wp3_blocks_2'))
    # real WP3: the held block's track retires and the dropped block comes back with a NEW track id
    R['5a_attach_again'] = dict(ok=d.attach('wp3_blocks_2'))
    d.run_for(2.0, lambda: d.markers([B1m]))
    drop2 = (-0.06, -0.05)
    d.release('wp3_blocks_2', tw(*drop2))
    d.run_for(3.0, lambda: d.markers([B1m, (7, drop2[0], drop2[1])]))     # new id 7 at the drop pose
    w, att = d.scene()
    dup = [k for k, v in w.items() if near(v, tw(*drop2), 0.01)]
    R['5b_no_duplicate_after_new_track_id'] = dict(ok=dup == ['wp3_blocks_7'], at_drop=dup, world=sorted(w))
    # cameras lost: WP3 health says no_valid_stream and its marker snapshots become empty after retirement
    def health(mode):
        d.pub_h.publish(String(data=json.dumps({'status': 'ok' if mode != 'no_valid_stream' else 'no_valid_stream',
                                                'fallback_mode': mode})))
    d.run_for(1.0, lambda: (health('dual_view'), d.markers([B1m, (7, drop2[0], drop2[1])])))
    d.run_for(2.5, lambda: (health('no_valid_stream'), d.markers([])))
    w, _ = d.scene()
    R['6a_blind_keeps_obstacles'] = dict(ok={'wp3_blocks_1', 'wp3_blocks_7'} <= set(w), world=sorted(w))
    d.run_for(2.0, lambda: (health('dual_view'), d.markers([])))
    w, _ = d.scene()
    R['6b_seeing_again_removes_them'] = dict(ok=not ({'wp3_blocks_1', 'wp3_blocks_7'} & set(w)), world=sorted(w))
    return R


def scenario_json(d):
    R = {}
    B1, B2 = (11, -0.04, 0.03), (12, 0.03, -0.02)
    d.run_for(2.0, lambda: d.json_msg([B1, B2]))
    w, _ = d.scene()
    R['j1_add_two'] = dict(ok=near(w.get('wp3_blocks_11'), tw(-0.04, 0.03)) and near(w.get('wp3_blocks_12'), tw(0.03, -0.02)), world=w)
    d.run_for(2.0, lambda: d.json_msg([], status='uncalibrated'))
    w, _ = d.scene()
    R['j2_bad_status_changes_nothing'] = dict(ok={'wp3_blocks_11', 'wp3_blocks_12'} <= set(w), world=sorted(w))
    d.run_for(2.0, lambda: d.json_msg([B1]))
    w, _ = d.scene()
    R['j3_retired_track_removed'] = dict(ok='wp3_blocks_12' not in w and 'wp3_blocks_11' in w, world=sorted(w))
    return R


def scenario_mock(d, drop_after, t_start_wall):
    """Hazem's mock_wp3_publisher.py --drop-after N, started by the harness at wall time t_start_wall: all 3
    blocks must appear, and block 3 must leave once the mock stops publishing it (snapshot semantics)."""
    R = {}
    t = lambda: time.time() - t_start_wall
    first3 = None
    while t() < drop_after:
        w, _ = d.scene()
        if {'wp3_blocks_1', 'wp3_blocks_2', 'wp3_blocks_3'} <= set(w):
            first3 = round(t(), 2)
            break
        rclpy.spin_once(d, timeout_sec=0.2)
    R['m1_three_mock_blocks_present'] = dict(ok=first3 is not None, seen_at_s=first3)
    gone = None
    while t() < drop_after + 6.0:
        w, _ = d.scene()
        if 'wp3_blocks_3' not in w and {'wp3_blocks_1', 'wp3_blocks_2'} <= set(w):
            gone = round(t(), 2)
            break
        rclpy.spin_once(d, timeout_sec=0.2)
    R['m2_dropped_block_removed_others_kept'] = dict(ok=gone is not None and gone >= drop_after, removed_at_s=gone,
                                                     drop_after_s=drop_after)
    return R


def watch(d, secs):
    t0 = time.monotonic(); hist = []
    while time.monotonic() - t0 < secs:
        w, att = d.scene()
        hist.append(dict(t=round(time.monotonic() - t0, 1), n=len(w), ids=sorted(w)))
        rclpy.spin_once(d, timeout_sec=0.5)
    return dict(history=hist, max_objects=max((h['n'] for h in hist), default=0), final=hist[-1] if hist else None)


def main():
    mode, out = sys.argv[1], sys.argv[2]
    rclpy.init()
    d = Driver()
    for _ in range(50):
        if d.get_scene.wait_for_service(timeout_sec=0.2):
            break
    time.sleep(1.0)
    if mode == 'watch':
        res = watch(d, float(sys.argv[3]))
    elif mode == 'mock':
        res = scenario_mock(d, float(sys.argv[3]), float(sys.argv[4]))
    else:
        d.cleanup()
        res = scenario_markers(d) if mode == 'markers' else scenario_json(d)
        d.cleanup()
    with open(out, 'w') as f:
        json.dump(res, f, indent=1)
    if mode != 'watch':
        for k, v in res.items():
            print(f"  {'PASS' if v['ok'] else 'FAIL'}  {k}  {'' if v['ok'] else {kk: vv for kk, vv in v.items() if kk != 'ok'}}")
    else:
        print(f"  max wp3 objects seen: {res['max_objects']}; final: {res['final']}")
    d.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

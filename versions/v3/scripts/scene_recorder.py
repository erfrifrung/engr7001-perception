#!/usr/bin/env python3
"""Record /wp3_live/scene_observation (+ health) to JSONL for offline analysis of a bag replay.

  python3 scene_recorder.py OUT.jsonl SECONDS
Each line: epoch_id, status, fallback_mode, tracks by state, MEASURED poses, ambiguous count, stage timings,
processing delay (publish - observation, same clock as the node: sim time under bag replay).
"""
import json
import sys
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def main():
    out, secs = sys.argv[1], float(sys.argv[2])
    rclpy.init()
    node = Node('wp3_scene_recorder')
    fh = open(out, 'w')
    n = [0]

    def on_scene(msg):
        p = json.loads(msg.data)
        tracks = p.get('tracks', [])
        rec = dict(
            epoch=p.get('epoch_id'), status=p.get('status'), mode=p.get('fallback_mode'),
            n_measured=sum(t['state'] == 'MEASURED' for t in tracks),
            n_displaced=sum(t['state'] == 'DISPLACED' for t in tracks),
            n_held=sum(t['state'] in ('OCCLUDED', 'STALE') for t in tracks),
            n_commit=sum(bool(t.get('snapshot_commit_eligible')) for t in tracks),
            n_amb=len(p.get('ambiguous_clusters', [])),
            measured=[dict(id=t['track_id'], xy=t['pose_table_m'][:2], yaw=t.get('yaw_rad'), src=t.get('evidence_source'))
                      for t in tracks if t['state'] in ('MEASURED', 'DISPLACED')],
            timings=p.get('stage_timings_ms', {}),
            proc_ms=round((p.get('publish_stamp_ns', 0) - p.get('observation_stamp_ns', 0)) / 1e6, 2)
            if p.get('observation_stamp_ns') else None,
            v3=[s.get('v3') for s in p.get('evidence', {}).get('split_logs', []) if isinstance(s, dict) and 'v3' in s],
        )
        fh.write(json.dumps(rec) + '\n')
        n[0] += 1

    node.create_subscription(String, '/wp3_live/scene_observation', on_scene, 50)
    t0 = time.monotonic()
    while time.monotonic() - t0 < secs:
        rclpy.spin_once(node, timeout_sec=0.1)
    fh.close()
    print(f'recorded {n[0]} scene messages')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

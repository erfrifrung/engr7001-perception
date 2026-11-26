# WP3 → WP2 / WP5 interface specification V1 (draft for review, task 6103)

Owner: WP3 (Shadi). Consumers: WP2 twin bridge (Hazem), WP5 controller and fixtures (Ant). Status: **draft until signed below.**
Schema `wp3_scene_observation_v2.1` is frozen by this document. Everything under "v2.2 (proposed)" is additive and optional.

## 1. Transport and timing

| Item | Value |
|---|---|
| Topic | `/wp3_live/scene_observation`, `std_msgs/String`, one JSON object per message (atomic snapshot) |
| Display topics | `/wp3_live/markers` (`MarkerArray`, starts with `DELETEALL`), `/wp3_live/health`, `/wp3_live/overlay`. Display only. |
| Rate | Target 15 Hz. Heartbeat at least every 0.25 s while camera stamps are unchanged. |
| ROS domain / discovery | Domain 67, `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`. **M1: WP3, the twin bridge and `move_group` run on the same lab desktop.** Any cross-host setup needs a separate agreement: discovery and a clock-sync check. |
| QoS | Reliable, keep-last depth 10 (publisher). Subscribers use depth ≥ 5. |
| Clock | All stamps are ROS time in ns from the WP3 host. On the same host: `age_ms = now − observation_stamp_ns`. |

## 2. Message (top level)

| Field | Type | Meaning | Consumer |
|---|---|---|---|
| `schema_version` | str | `"wp3_scene_observation_v2.1"` | all |
| `epoch_id` | int | Increments per published epoch | all |
| `frame_id` | str | `"jenga_table"` | all |
| `observation_stamp_ns` | int | Camera exposure stamp of the primary view | WP5 (age) |
| `publish_stamp_ns` | int | Taken **after** compute (V3.1); `publish − observation` = processing delay | WP2/WP5 diagnostics |
| `status` | str | `ok` or one of `uncalibrated`, `calibration_mismatch` (V3.2: every available camera failed its table-plane check), `no_valid_stream`, `no_plane`, `camera_moved`, `roi_empty`. **Only `ok` carries usable geometry.** A non-`ok` message is not evidence that blocks left. | WP2 bridge, WP5 |
| `fallback_mode` | str | `dual_view`, `wrist_only`, `side_only`, `wrist_unqualified`, `no_valid_stream` | WP5 (confidence), logs |
| `calibration_epoch` | str | ID of the calibration in use. A change invalidates old poses. V3.2: when `/tf_static` changes under a running node the epoch gains a `#rN` suffix and every track is re-detected (new ids). | WP2, WP5 |
| `config_hash`, `code_hash` | str | Provenance (code hash covers the node and all `wp3_live` modules) | logs |
| `dwell` | obj | `stationary`, `fk_qualified`, `imu_corroborated`, `reason` | WP5 |
| `tracks` | list | Every live track (see 3), including held ones | WP2, WP5 |
| `ambiguous_clusters` | list | Occupied regions WP3 will not split into blocks (see 4) | WP2 display, WP5 keep-out |
| `slot_occupancies` | list | Tower slot states | WP5, twin |
| `stage_timings_ms`, `evidence`, `camera_health` | obj | Diagnostics, not contract. V3.2: `camera_health.<role>.calibration` = `{status: ok \| mismatch \| unverified, last_check: {plane_angle_deg, plane_offset_mm, …}}`; a `mismatch` camera was not fused | logs |

## 3. Tracks

| Field | Meaning |
|---|---|
| `track_id` (int), `block_id` (`B07`) | Identity. Never reused. |
| `state` | `MEASURED` (seen now), `DISPLACED` (seen now, moved ≥ 8 mm from anchor), `OCCLUDED` (not seen, held ≤ 1.8 s), `STALE` (not seen, held). A track absent from the message has been **retired**. V3.1: retired 3.0 s after it was last seen. **V3.2:** retired only when 3 consecutive epochs show its footprint empty (depth observed, clearly below its top), or after 60 s unseen; a block hidden under the arm or inside an ambiguous region stays `STALE`. |
| (V3.2) duplicates | Two tracks on one layer closer than 0.4 × block width cannot be two blocks: they are merged into the older id, which must re-confirm before it is `snapshot_commit_eligible` again. |
| `pose_table_m` | `[x, y, z_centre]` in `jenga_table`, metres |
| `yaw_rad` | Rotation about +z of `jenga_table` of the **`dims_mm[0]` axis**. For blocks `dims_mm[0]` is the long side. 180° symmetric (`yaw_symmetry_deg`). |
| `dims_mm` | `[along yaw, across yaw]`, mm. Measured when both opposite edges are visible, otherwise nominal 75 × 25. |
| `z_top_table_rel_m` | Top-face height |
| `validity_mask` | Per component; `roll`/`pitch` are always false (planar blocks) |
| `observation_stamp_ns` | Stamp of the last measurement. **Not** refreshed while held. |
| `identity_ambiguous`, `ambiguous_peer_ids` | Identity not verified after joint occlusion or a near-tie |
| `snapshot_commit_eligible` | WP3's recommendation that the pose may be used as a target: confirmed, not displaced, not identity-ambiguous, fresh side view |
| `pose_base_m` | `null` in v2.1 by agreement: the single source of `world ← jenga_table` is the TF published by the twin (`wp3_table_tf.py`) |

## 4. Ambiguous clusters (v2.1)

`cluster_id` (`AMB_nn`) is a per-message index and is **not stable**. `envelope_xy_m` is the centre; `span_mm` is the extent (may be 0 when unknown); `top_z_m`; `n_est`; `reason`. There is no yaw and no bottom height. **M1: display only; not inserted into the MoveIt collision world.**

## 5. Consumer rules (proposed, to sign)

1. **Collision world (WP2 bridge):**
   - Insert every track in state `MEASURED`/`DISPLACED` as an obstacle, **including** identity-ambiguous and non-eligible ones (they are physically there).
   - Held tracks keep their last pose. Retired tracks are removed (snapshot semantics).
   - Ignore non-`ok` messages.
   - In marker mode, removals are held while `/wp3_live/health` reports `no_valid_stream` or goes silent (a blind WP3 is not evidence that blocks left).
   - Blocks in the gripper are left to the twin. A just-released block's pending removal is held for 1 s.

   *Implemented on Hazem's twin v5 in `integration/twin_bridge_v31.patch`; live A/B against fake-hardware MoveIt in `integration/live_test/` (original v5 bridge fails 3/13 checks, patched passes 13/13).*
2. **Targets (WP5):** a pose may be used as a grasp/fixture target only if `status == "ok"`, `state == "MEASURED"`, `snapshot_commit_eligible`, not `identity_ambiguous`, and `age_ms ≤ A` (A agreed in section 7).
3. **Tolerance tiers (provisional, WP5 to confirm the grasp mode):** extract `|Δz| ≤ 2 mm`, lateral `≤ 3 mm`, yaw `≤ 2°`; place XY `≤ 3 mm`, yaw `≤ 2°`; free-block lengthwise pick `≤ 2.5 mm / 2°`.
4. **Dead-band:** the bridge suppresses scene updates below 3 mm / 2°, so MoveIt can lag WP3 by that much. WP5 must read target poses from this topic, not from the MoveIt scene, or the dead-band must be reduced (decision in section 7).

## 6. v2.2 (proposed, additive, after Baseline)

- `candidates[]`: best-guess poses for blocks WP3 will not certify (`snapshot_commit_eligible = false` always). Needed for the "flagged vs treated as certain" experiment. Never a target.
- `ambiguous_clusters[].stable_id`, `yaw_rad`, `z_bottom_m`, `robot_overlap_frac`, so clusters can be keep-out slabs one layer thick (15 mm) instead of table-to-top boxes.
- `geometry_stamp_ns` (separates reused geometry from fresh observation if frame skipping is ever enabled).

## 7. Decisions to record at the review

| # | Decision | Options | Agreed |
|---|---|---|---|
| D1 | Twin + bridge host for M1 | same lab desktop (proposed) / cross-host | |
| D2 | Freshness bound A for WP5 targets | 150 ms proposed. **Note:** the recorded live sample already shows `age_ms ≈ 134` and ~85 ms receive lag (`display_replay_scene_observations.json`), so margin is small. Measure on Thursday. | |
| D3 | Grasp mode WP5 will use | push-then-grasp across width / across height / lengthwise free-block pick | |
| D4 | Dead-band | WP5 reads poses from this topic / reduce bridge dead-band to 1 mm / 0.5° | |
| D5 | Ambiguous clusters in MoveIt at Baseline | layer slabs (v2.2 fields) / display only | |
| D6 | Lab object declaration | no block-like clutter (75 × 50, 150 × 25 objects) in the workspace during tests: needed because a half-occluded 75 × 50 slab is indistinguishable from two flush blocks (`tests/test_v31_solver.py`) | |

Signed: WP3 ______  WP2 ______  WP5 ______  Date ______

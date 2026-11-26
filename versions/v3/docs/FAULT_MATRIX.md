# WP3 fault-injection matrix (V3.1, 26 Nov)

Each fault has the required behaviour, the test that shows it, where it ran, and its status. Use it for the testing row (predictions vs results) and as the source for the FMEA rows.

- **Test locations:** "live" = fake-hardware MoveIt with Hazem's twin v5 (`integration/live_test/`); "unit" = `versions/v3/tests/`; "evidence" = branch `v3.1-evidence` (`tests/test_v31_hazards.py`, `docs/EVIDENCE_26NOV.md`).
- **Collision vs target:** a block that is still physically there stays a **collision obstacle** even when it is not a valid **target**. Target validity (`snapshot_commit_eligible`, not identity-ambiguous, fresh `observation_stamp_ns`) is checked by WP5.

| # | Fault | Required behaviour | Evidence | Status |
|---|---|---|---|---|
| 1 | Normal scene updates | blocks added and moved in the planning scene at the published pose | live checks 1, 2, j1; mock m1 | **Pass** |
| 2 | Track retired upstream | its collision object is removed | live 3, j3; mock m2 (original v5 bridge **fails**: phantom obstacle) | **Pass** (patched) |
| 3 | Same block re-detected with a new track id | no duplicate obstacle | live 5b (original v5 **fails**: 2 obstacles at the drop spot) | **Pass** (patched) |
| 4 | Block attached to / released by the gripper | twin owns it while held; no deletion and re-adding at release; pending removal kept | live 4a–4e; unit `test_block_tracker_attached_and_rejected_updates_are_not_lost`, `test_block_tracker_holds_removal_just_after_release` | **Pass** |
| 5 | All cameras lost (WP3 reports `no_valid_stream`) | obstacles kept (greyed), not removed | live 6a (marker mode, health gate); j2 (JSON: non-`ok` messages ignored) | **Pass** |
| 6 | Recovery with an empty, healthy snapshot | obsolete obstacles removed | live 6b (original v5 **fails**) | **Pass** (patched) |
| 7 | WP3 process crashes (no messages at all) | obstacles kept; targets invalid by age | bridge: health silent > 1 s ⇒ removals held; no new markers ⇒ nothing removed; WP5 age check | **Partly covered**: code path not exercised by a live kill test |
| 8 | One camera stream freezes / drops | degrade dual → single → `no_valid_stream`, heartbeats continue | unit `test_ros2_display_node` (side freeze → `wrist_only`; full loss reported ≤ 1.2 s; 8/8 runs) | **Pass** |
| 9 | Health message delayed / out of order | stale health must not authorise removals | bridge treats health older than 1 s as blind | **Not tested**. Recommended fix: twin uses the JSON topic (status, epoch and tracks in one atomic message) |
| 10 | Recalibration while running | old-calibration poses rejected | **Limitation**: the JSON bridge ignores `calibration_epoch`; the node caches tripod/third transforms for its lifetime (xfail `test_bridge_drops_held_poses_from_a_previous_calibration`, `test_fixed_camera_transform_is_cached_for_the_node_lifetime`) | **Known limitation** → procedure: restart WP3 + bridge after any recalibration |
| 11 | Robot TF missing or late | no stale arm pose used; upper layers not qualified | node: TF at frame stamp, fallback only within 50 ms; unit `test_partial_robot_tf_does_not_qualify_upper_layers` | **Pass** (unit); live-node path not fault-injected |
| 12 | Block hidden > 3 s with healthy cameras (e.g. under the arm) | stays an obstacle until its space is seen empty | **Reproduced**: OCCLUDED 0.0 s, STALE 1.8 s, retired 3.0 s, removed from MoveIt 3.0 s (xfail `test_hidden_block_stays_an_obstacle_until_observed_empty`) | **Known hazard** → FMEA; post-M1 fix: retire only on observed-free |
| 13 | Slow push below the tracking thresholds | no re-anchor (commit-eligible) while moving; scene follows cumulative motion | unit `test_steady_slide_never_reanchors_mid_motion`; bridge compares against the last applied pose (cumulative) | **Pass** (tracker); bridge by design, no dedicated test |
| 14 | MoveIt rejects a planning-scene update | retried, state not lost | unit snapshot/restore test | **Pass** (unit); not forced live |
| 15 | Camera without a table transform | `status: uncalibrated`, no `jenga_table` poses | unit `test_uncalibrated_primary_publishes_no_table_poses` | **Pass** |
| 16 | Twin started with live WP3 but no measured table frame | refuse, or explicit `--manual-table` with a warning | `start_perception.sh` branches (exit 2 without yaml) | **Pass** |
| 17 | Block-like clutter (half-occluded 75 × 50 slab; block-sized floating object) | never published as a block | **Limitation** (xfail `tests/test_v31_solver.py`) | **Known hazard** → lab rule: no block-like clutter (spec D6) |

## FMEA rows to add or update (severity / likelihood / detection, 1–10; draft ratings to review)

| Failure mode | Effect | S | L | D | Mitigation now | Planned fix |
|---|---|---|---|---|---|---|
| Hidden block retired after 3 s (healthy cameras) | obstacle missing; arm may plan into it | 8 | 5 | 6 | operator procedure: do not plan near blocks hidden > 3 s; twin greys stale blocks first | retire only on observed-free space (Baseline) |
| Recalibration without restarting WP3 / bridge | tripod/third transforms and held poses from the old calibration | 7 | 4 | 5 | restart rule in the runbook; `calibration_epoch` in every message | invalidate on epoch / TF change |
| Half-occluded slab or floating block-sized object | false block pose | 7 | 2 | 6 | no block-like clutter in the workspace (D6) | side-view evidence under occluders |
| Twin on the manual table frame | all blocks offset | 7 | 4 | 3 | live mode refuses without the measured frame | — |
| Phantom / duplicate obstacles (unpatched v5 bridge) | spurious collisions, planner lock-out | 5 | 8 | 4 | patched bridge (live 13/13) | — |

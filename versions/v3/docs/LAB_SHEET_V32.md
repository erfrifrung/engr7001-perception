# WP3 V3.2 live test sheet

Baseline `v3.2-candidate-2` on the Ryzen 5 5600 lab PC. Keep `v3.1-m1-candidate-3` available for demo fallback. Run one perception node at a time on ROS domain 67. Record the ref, camera roles, calibration epoch, ROS profile and solver settings with each run.

## 0. Warm up on the 26 November recording

Use the three-camera, three-cube recording and keep live publishers off during replay. From `versions/v3`, start the node in one terminal and replay the bag in a second. Source `scripts/lab_env.sh` in both terminals.

```bash
source scripts/lab_env.sh
source ros2_ws/install/setup.bash
ros2 run wp3_live_perception wp3_live_node --ros-args \
  -p use_sim_time:=true \
  -p config:=ros2_ws/src/wp3_live_perception/config/printed_jenga_live.json \
  -p third_ns:=/perception/third
```

```bash
source scripts/lab_env.sh
ros2 bag play '<bag-path>' --clock
```

In a third sourced terminal, run `ros2 topic echo /wp3_live/health`. Once the health check has enough frames, inspect `camera_health.<role>.calibration.status` for `wrist`, `side` and `third`. A camera may report `ok` or `suspect` during the warm-up. Flag persistent `mismatch` or missing calibration. Use `/wp3_live/scene_observation` to check track IDs. Stop the playback and the warm-up node before launching live perception.

## 1. Check live calibration and joint states

Use the [twin setup guide](TWIN_INTEGRATION_SETUP.md) for the camera fit and robot touch-off. Leave one board TF publisher running. Run `ros2 topic hz /joint_states` and confirm the expected real-arm feed (target at least 30 Hz in the lab setup).

```bash
source scripts/lab_env.sh
cal="$HOME/wp3_board_$(date +%Y%m%d_%H%M%S)"
published_fit='<published-fit.json>'
python3 scripts/register_board.py "$cal" "$published_fit" --pitch-mm 20
python3 scripts/check_calibration_drift.py "$published_fit" "$cal/current_board_tf.json"
```

Exit 0 is a pass. Record the drift check output and note which fit is being published. Measure the checkerboard thickness with calipers. If the camera calibration uses the board surface and the robot touch-off uses the bare table, account for the thickness so the two fits share the same height reference.

## 2. Launch the live node

```bash
bash scripts/launch_live.sh --policy merged_corroborated \
  --third-ns /perception/third --save-dir "$cal/live_v32"
```

Check the startup log for `tracker_merge_overlapping=True`, `static_tf_refresh_s=1.0`, `calibration_check=True`, `retire_on_observed_free=True` and `event_wakeup=True`.

For a separate V3 region-stage test on touching printed blocks, use `--v3 on --v3-solver enumerate --v3-budget-ms 45` or `--v3 on --v3-solver prove --v3-budget-ms 120`. Keep these solver trials separate from the threading comparison.

## 3. Compare scheduling and view threading

Use the same scene, camera inputs, solver and logging settings for each 60 s run. Restart the node between runs. Copy the default profile and set `"enable_parallel_views": false` in the copy. Give each run a separate save directory.

| Run | Change from default | Publish rate (Hz) | Latency p50 (ms) | Latency p95 (ms) | Node CPU (%) |
|---|---|---|---|---|---|
| Baseline | Defaults | | | | |
| Timer wakeup | Add `--event-wakeup off` | | | | |
| Serial views | Pass `--config <copied-profile.json>` with parallel views disabled and default wakeup | | | | |

Measure publication rate on `/wp3_live/scene_observation`. For latency, subtract the observation stamp from the publish stamp for each valid snapshot. Use the same process CPU sampling method across runs.

## 4. Run the live checks

| Test | Procedure and pass condition |
|---|---|
| Duplicate tracks | Keep one printed block or cube at roughly 45° for 3 min. The distinct track count stays in single digits and `snapshot_commit_eligible` is true on at least 90% of settled frames. |
| Removal at all four corners | Place and remove a 15 mm block at each workspace corner. Record time to retire each track. Expect roughly 0.2–1 s, watching for slower clearance at far corners where the oblique depth bias can approach the 7.5 mm free-space threshold. |
| Hidden block | Keep a confirmed block hidden for at least 10 s with healthy camera streams. Its track becomes `STALE` and its obstacle stays in MoveIt. Greying indicates camera loss. Reveal the block and check that it returns to measured. Repeat with the block removed while hidden. With the footprint visible and empty, check that it retires after three consecutive empty observations. |
| Camera bump | With the robot parked and motion paused, move one fixed camera enough to exceed the table-plane check (> 6° plane tilt or > 15 mm offset at the workspace centre). Expect `mismatch` after about 2 s (1 Hz, two consecutive failures), with other healthy views continuing. Recalibrate to the camera's new position and publish the updated fit. Confirm that a changed fit creates a new `#rN` epoch, resets the tracks and clears the bridge's old-epoch obstacles. Restore the original camera position and re-register before the next test. |
| 15-minute soak | Record RSS at start and end (`ps -o rss= -p <pid>`), max track ID, publish-minus-observation p50/p95 latency and percentage of dual-view snapshots. |
| Printed-block reference recording | Record a 90 s bag at `<bag-path>` with a single block, separated pair, touching pair and flush triple. Include `/tf`, `/tf_static`, `/joint_states` and each camera's RGB, aligned depth and camera info topics. Keep the associated `current_board_tf.json` with the local recording. |

## Diagnostic switches

Record any switch used and restore defaults after collecting the comparison.

| Symptom or comparison | Setting |
|---|---|
| Oblique depth bias trips the table-plane check | `--calibration-check off` |
| Repeated TF drift warnings or track resets | `--static-tf-refresh-s 0` |
| Removed track persists when empty space is hard to observe | `--retire-on-observed-free off` |
| Two nearby tracks merge unexpectedly | `--merge-overlapping off` |
| Compare timer wakeup against event wakeup | `--event-wakeup off` |
| Compare serial and parallel camera processing | Set `"enable_parallel_views": false` in a copy of the JSON and pass it with `--config`. The launcher has no dedicated flag for this setting. |

# ENGR7001 robotic perception

This archive tracks the development of tabletop block pose estimation, segmentation and multi-object tracking for an intent-aware shared-manipulation project. The ROS 2 perception nodes publish observations and RViz displays.

| Snapshot | Main change |
|---|---|
| [Early prototypes](prototypes/) | Wrist-only RGB-D segmentation and the tabletop heightmap comparison with and without RGB seam evidence. |
| [V1](versions/v1/) | Consolidated heightmap, RGB seam cues, side-view verification and a shared live/replay detector with configurable policies. |
| [V2](versions/v2/) | Whole-partition scoring, improved cross-camera depth and skew handling, temporal caching, and explicit tracking and ambiguity states. |
| [V2 three-view candidate](versions/v2-three-view/) | Option to corroborate seams with a third fixed view, plus checkerboard registration tools. |
| [V2 three-view handoff iteration](versions/v2-three-view-handoff/) | Follow-on snapshot from the expanded handoff, with changes to multi-view cut verification, replay, caching, diagnostics and tests. |
| [WP3 twin integration](integration/) | RViz configuration and the discrete WP3 additions for frame fitting and a trial multi-block receiver. |

## Replay evaluation (V2)

Run the [replay comparison](versions/v2/scripts/compare_bag.py) from the repository root with a SQLite rosbag2 recording. Use `--config` for the block geometry and camera settings of your capture and `--labels` for a CSV of timed ground-truth block positions. See the [setup guide](SETUP.md#run-the-included-synthetic-comparison) for a configuration example.

```bash
bash versions/v2/scripts/compare_bag.sh /path/to/recording --out results/replay \
  --views both --labels /path/to/labels.csv
```

V2 runs the same replay epochs through six policies. `depth_only` and `generic_watershed` are segmentation baselines. `permissive_rgb_seam` and `merged_wrist_rgb` test seam-based splits with different thresholds. `merged_corroborated` checks RGB seams against side-view evidence, and `whole_partition_mv` scores competing block layouts across views. Use `--policies` to select a subset.

The runner prints and saves `decision_table.md`, with per-run frame coverage and p50/p95/p99 processing time. With labels, it reports 15 mm XY matches, flags frames where one prediction region covers multiple labelled blocks, and counts false-confident frames and unmatched detections. XY error is summarised over matched predictions. `comparison_report.json` contains additional metrics, including ambiguity counts and yaw error where yaw labels are available. Per-frame outputs are saved in `predictions/`.

Example from 12 rendered frames of one stationary touching pair, yielding 24 labelled block observations. The timings exclude two warm-up frames and reflect the local run.

| Run | OK / total | 15 mm membership matches | Merge frames | False-confident frames | XY error p50/p95 (mm) | Runtime p50/p95/p99 (ms) |
|---|---:|---:|---:|---:|---:|---:|
| `both__depth_only` | 12/12 | 0/24 | 0 | 0% | — | 41.06 / 51.79 / 55.98 |
| `both__merged_corroborated` | 12/12 | 24/24 | 0 | 0% | 0.22 / 0.29 | 60.24 / 88.14 / 91.87 |
| `both__whole_partition_mv` | 12/12 | 24/24 | 0 | 0% | 0.22 / 0.29 | 60.99 / 102.65 / 106.62 |

The [synthetic suite](versions/v2/scripts/run_printed_jenga_suite.py) generates inputs and a separate summary table without a recording. The [stereo depth benchmark](versions/v2/scripts/bench_stereo_depth.py) reports OpenCV StereoSGBM coverage and runtime. Pass `--native-depth` with a 16-bit D435 frame from the same capture aligned to the left IR grid to report agreement over valid pixels in both depth maps. Use the camera's measured baseline, focal length and depth scale.

## Perception safety check

Run `python3 versions/v2/scripts/verify_zero_moveit.py` from the repository root. This static check scans the perception snapshot for MoveIt and robot-control imports and for planning-scene service and trajectory-action references. Its scope is the display-only snapshot. The optional planning-scene receiver is described in the [integration setup](SETUP.md#wp3-and-the-digital-twin).

## Development evidence

- On one recorded row of three touching wooden blocks, the RGB seam detector returned the correct count in **135/141 frames versus 0/141 with depth alone**, flagging the remaining six as ambiguous.
- In a stationary **720p** lab trial while recording all three streams, Fast DDS shared memory increased live **two-view** scene updates from **2.42 to 13.16 Hz** (rounded to 2.4 and 13.2 Hz on the CV).
- These development runs used recorded wooden props and a stationary rig. Independent physical accuracy on 3D-printed blocks, comparative accuracy of the third view, and acceptance in the full twin or planning workflow remain open.

Start with the [setup and replay guide](SETUP.md). See the [source and sharing inventory](INVENTORY.md) for attribution and the [local verification](VERIFICATION.md) for build and test results.

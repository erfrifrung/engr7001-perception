# Evidence, 26 Nov (branch v3.1-evidence). All timings: WSL i7-1185G7 laptop, NOT the Ryzen

## 1. Same bag, V2 vs V3, into MoveIt (`scripts/evidence_pipeline_ab.sh`)
Run on `three_cam_demo_bag` (45 mm wooden props, no ground truth), fake-hardware MoveIt with the patched v5 bridge. Five runs in rotated order, exactly one perception node per run.

- **Output was identical across all modes:** 3 blocks, 0 ambiguous clusters, 3 track ids, per-track xy spread 0.20–0.29 mm, 3 MoveIt objects.
- **V3 never ran.** There were no V3 verdicts and no `v3_regions_ms`: the V2 front end resolved the separated props, so this bag does not exercise V3. The V2/V3 difference can only be measured on touching or flush scenes (printed blocks on Thursday).
- **Per-frame processing:** 450–620 ms p50 per run with V3 on or off. That's dominated by contention on the laptop (bag playback of two 720p streams, MoveIt and the recorder all on one machine), not by V3. **It is not a lab-desktop prediction.** The stage split was consistent, though: `proposals_and_rgb` ≈ 60–70 % of each frame, `heightmap` and `deproject` ≈ 10 % each.
- **Two measurement artefacts, found and fixed:**
  - Reading the bag cold from `/mnt/c` inflated the first run.
  - Killing a `ros2 run` wrapper leaves the node alive, so earlier modes kept publishing (the message count rose 45 → 575). The script now launches the node executable directly and asserts that one node is running.

## 2. Healthy-camera prolonged occlusion (`tests/test_v31_hazards.py`)
- **Timeline** (block hidden at t = 0, dual-view healthy): OCCLUDED at 0.0 s, STALE at 1.8 s, retired at 3.0 s, **removed from the MoveIt collision world at 3.0 s**, although the space was never seen empty.
- **Status:** strict xfail. FMEA row added (`FAULT_MATRIX.md` on branch v3.1). No architecture change before M1.

## 3. Calibration-epoch change
- **Bridge:** the JSON bridge ignores `calibration_epoch`, so held tracks keep old-calibration poses (strict xfail).
- **Node:** it caches tripod / third transforms per `calib_epoch`, which is fixed for its lifetime, so re-registering the board while it runs has no effect.
- **Procedure until fixed:** restart WP3 and the bridge after any recalibration.

## 4. Budget sweep: completed on Ryzen 5 5600 (see Section 6)
`bash bench/budget_sweep.sh` (prove/enumerate × 30/45/66/120/250 ms/unlimited, 2 rounds, pinned, plus a 60 000-node cap). `bench/sweep_analyze.py` separates timeouts and counterexamples, reports correct and wrong poses and latency, and lists the blocks still unresolved at an unlimited budget. `run_scenes.py` now logs the V3 stage's per-region outcomes. See Section 6 below for the measured Ryzen 5 5600 results.

## 5. V4 implications so far
- On real frames the **front end** dominates (proposals and RGB seam checks, then heightmap and deprojection), not the solver. Selective computation (skip unchanged regions) and moving the RGB seam scoring off the Python path are the first V4 targets.
- The solver matters only on touching or flush scenes; quantify it with the sweep.
- Measure only on an uncontended host with one node, the bag pre-read, rotated order and repeated runs.

## 6. Target-host results: Ryzen 5 5600 (`engr7001-MS-7C96`, 26 Nov)

All measurements in this section ran on the lab desktop **AMD Ryzen 5 5600** (`engr7001-MS-7C96`, 6C/12T, Ubuntu 24.04.5 LTS, ROS 2 Jazzy, Python 3.12.3, NumPy 1.26.4, OpenCV 4.6.0, SciPy 1.11.4, CPU governor `performance`, SMT `on`, `OMP/OPENBLAS/MKL/WP3_CV2_THREADS=1`, pinned with `taskset -c 2` for benchmarks). GPU: Radeon RX 7600/7650 (`1002:7480`, no `/opt/rocm`, `torch=false`).

> **Background load provenance:** Three live RealSense camera nodes (`wrist`, `tripod`, `third`, ~15–18% CPU each), `live_view.py` (~10% CPU), and `wp3-static-board-tf.py` were running on the shared desktop during all runs (`uptime` load 0.53 / 0.66 / 0.71). No MoveIt or robot motion was launched.

### 6.1 Build and test verification
- `colcon build --symlink-install` (`v3.1`, commit `5709862`): succeeded (`1.53 s`).
- `python3 scripts/verify_zero_moveit.py`: `ZERO_MOVEIT_AUDIT: PASS`.
- `python3 -m pytest -q tests` (`v3.1`, with ROS 2 Jazzy sourced): **`62 passed, 2 xfailed`** (`53.34 s`).
- `python3 -m pytest -q tests/test_v31_hazards.py` (`v3.1-evidence`, commit `cc48278`): **`2 passed, 2 xfailed`** (`0.44 s`).

### 6.2 Synthetic held-out benchmark (`v3.1`, `5709862`, 168 scenes / 312 blocks, 3 rounds rotated, `taskset -c 2`)
| Method | Published / Correct (of 312) | Wrong (5 mm / 15 mm) | Cov5 | p50 / p95 / p99 / max ms (3 rounds, excl. warmup) | > 66.7 ms frac |
|---|---:|---:|---:|---:|---:|
| `tuned_mc` (V2 path) | 86 / 86 | 0 / 0 | 0.276 | 33.0 / 41.6 / 47.5 / 51.6 | 0.000 |
| `v3enum_mc` | 175 / 175 | 0 / 0 | 0.561 | 46.0 / 72.5 / 121.6 / 126.2 | 0.069 |
| `v3prove_mc` | 190 / 190 | 0 / 0 | 0.609 | 68.6 / 114.9 / 138.1 / 172.8 | 0.541 |
| `v3adapt_mc` | 190 / 190 | 0 / 0 | 0.609 | 69.2 / 125.5 / 142.2 / 153.1 | 0.545 |

- Matches `THURSDAY_PREDICTIONS.md` §4 predictions (`tuned_mc` predicted 20–35 / 30–50 ms → measured 33.0 / 41.6 ms; `v3prove_mc` predicted 60–90 / 110–150 ms → measured 68.6 / 114.9 ms), with **0 wrong poses** across all methods.
- On one core, `v3prove_mc` (`68.6 ms` p50) sits just above the 66.7 ms (15 Hz) frame period (`54.1%` over period), while `v3enum_mc` (`46.0 ms` p50) mostly fits (`6.9%` over period) at 15 fewer resolved blocks.

### 6.3 Solver budget sweep on Ryzen 5 5600 (`v3.1-evidence`, `cc48278`, 168 scenes / 312 blocks, 2 rounds, `taskset -c 2`)
| solver | budget ms | node cap | correct (mean of rounds) | wrong | coverage | stable / counterex. / timeout verdicts | regions skipped by deadline | frames that hit a budget | p50 / p95 / p99 / max ms |
|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| enum | 30 | 6000 | 86.5 (86–87) | 0 | 0.278 | 0 / 0 / 0 | 120 | 108 | 34 / 43 / 47 / 48 |
| enum | 45 | 6000 | 169.0 (168–170) | 0 | 0.542 | 137 / 0 / 0 | 16 | 11 | 46 / 64 / 97 / 110 |
| enum | 66 | 6000 | 175.0 (175) | 0 | 0.561 | 165 / 0 / 0 | 1 | 1 | 46 / 69 / 102 / 108 |
| enum | 120 | 6000 | 175.0 (175) | 0 | 0.561 | 168 / 9 / 0 | 0 | 0 | 46 / 73 / 121 / 127 |
| enum | 250 | 6000 | 175.0 (175) | 0 | 0.561 | 168 / 12 / 0 | 0 | 0 | 46 / 73 / 154 / 221 |
| enum | inf | 6000 | 175.0 (175) | 0 | 0.561 | 168 / 12 / 0 | 0 | 0 | 46 / 73 / 151 / 221 |
| prove | 30 | 6000 | 86.0 (86) | 0 | 0.276 | 0 / 0 / 0 | 120 | 108 | 35 / 45 / 49 / 53 |
| prove | 45 | 6000 | 87.0 (87) | 0 | 0.279 | 1 / 0 / 150 | 23 | 88 | 46 / 56 / 71 / 81 |
| prove | 66 | 6000 | 131.5 (124–139) | 0 | 0.421 | 54 / 12 / 150 | 8 | 82 | 67 / 85 / 89 / 93 |
| prove | 120 | 6000 | 190.0 (190) | 0 | 0.609 | 173 / 42 / 4 | 4 | 6 | 72 / 125 / 140 / 146 |
| prove | 250 | 6000 | 190.0 (190) | 0 | 0.609 | 173 / 43 / 3 | 0 | 1 | 71 / 122 / 141 / 276 |
| prove | inf | 6000 | 190.0 (190) | 0 | 0.609 | 175 / 44 / 0 | 0 | 0 | 70 / 118 / 143 / 842 |
| prove | inf | 60000 | 190.0 (190) | 0 | 0.609 | 175 / 44 / 0 | 0 | 0 | 70 / 115 / 142 / 852 |

Unresolved ground-truth blocks at unlimited wall-clock budget, by family (last round of each run):
| family | `enum\|inf\|n6000` | `prove\|inf\|n60000` | `prove\|inf\|n6000` |
|---|---:|---:|---:|
| `L_T` | 2 | 0 | 0 |
| `bridge` | 21 | 19 | 19 |
| `end_to_end` | 10 | 10 | 10 |
| `flush_pair` | 16 | 16 | 16 |
| `flush_triple` | 36 | 36 | 36 |
| `holes_pair` | 8 | 8 | 8 |
| `mixed_yaw` | 8 | 0 | 0 |
| `occluded_pair` | 22 | 21 | 21 |
| `stack2` | 12 | 12 | 12 |
| `stagger` | 2 | 0 | 0 |

- **Key conclusions (Ryzen 5 5600):**
  - **0 blocks lost to the 6 000-node cap:** `prove|inf|n60000` is identical to `prove|inf|n6000` (190/312 = 0.609, identical unresolved counts per family). All remaining 122/312 unresolved blocks are genuinely ambiguous geometry (`flush_triple`, `occluded_pair`, `bridge`, `flush_pair`, `stack2`, `end_to_end`, `holes_pair`).
  - **Budget knee:** `prove` at 45 ms gains almost nothing over V2 (87 vs 86 blocks, 150 timeout verdicts) and at 66 ms reaches 131.5 (0.421); it needs ≥ 120 ms to reach its 190-block plateau (0.609). `enum` already achieves 169.0 (0.542) at 45 ms (`46 / 64 ms` p50/p95) and reaches its 175.0 plateau (0.561) by 66 ms.

### 6.4 Real bag replay (`v3.1`, `5709862`, `CONFIG=historical_props_12nov.json`, `taskset -c 2`)
All four bags were replayed with the 45 mm wooden-prop profile (`historical_props_12nov.json`); `block-moves` used `--tf-json .../recalibrate-three-20261119/three_camera_tf.json` (no `/tf_static` in bag):
- `three_cam_demo_bag` (142 epochs): V2 `both__merged_corroborated` `8.33 / 9.38 / 14.51 ms` (p50/p95/p99), `both__whole_partition_mv` `9.73 / 10.66 / 10.94 ms`. With V3 on, `merged_corroborated` activated V3 on `14/142` frames (`prove_45ms` `8.36 / 83.16 / 114.24 ms`; `prove_120ms` `8.39 / 189.47 / 331.87 ms`), while `whole_partition_mv` resolved all 3 blocks in V2 (`v3_nz=0/142`).
- `block-moves` (438 epochs): V2 `both__merged_corroborated` `8.38 / 17.90 / 22.25 ms`, `both__whole_partition_mv` `10.45 / 16.96 / 21.76 ms`. V3 activated on `137/438` frames under `merged_corroborated` (`prove_45ms` `9.03 / 55.30 / 128.53 ms`) and `47/438` under `whole_partition_mv` (`prove_45ms` `10.58 / 50.89 / 132.99 ms`).
- `wood-05-three-touching` (129 epochs): V2 `both__merged_corroborated` `11.63 / 13.66 / 14.46 ms`, `v3_nz=0/129`, `poses=1/1/1` (only 1 block fell inside `historical_props_12nov.json`'s Nov 12 ROI).
- `holdout-rehearsal` (159 epochs): `5.36 / 6.67 / 7.17 ms`, `poses=0/0/0` under `historical_props_12nov.json` ROI.
- CPU usage on one pinned core: `79–85%` (V2) and `90–99%` (V3 modes); max RSS `0.9–1.08 GB` (`three_cam_demo_bag`), `2.1–2.3 GB` (`block-moves`), `2.76 GB` (`holdout-rehearsal`), `4.77–4.90 GB` (`wood-05` 1080p RGB).

### 6.5 Passive live camera checks (`v3.1`, `ROS_DOMAIN_ID=67`, no robot/MoveIt)
- Pre-checks: `/joint_states` **absent**; `/perception/wrist` `global_time_enabled=True`; live camera publish rates ~`9.7 Hz` (`wrist`), ~`12.1 Hz` (`tripod`).
- **Default M1 V2 (`require_fk_for_moving_wrist=true`, `enable_v3_regions=false`):** logged `enable_v3_regions=False`, `dwell_reason=fk_missing_or_moving` on `89/92` ok frames (`side_only` 41, `wrist_unqualified` 51, `dual_view=0/92` = `0.0%`), `9.94 Hz`, stage `total_ms` `42.80 / 132.17 ms` (p50/p95), `publish - obs` `176.85 / 342.09 ms`, `rx_minus_hdr_ms` p50 `85.35 ms` (wrist) / `84.25 ms` (side).
- **Parked-arm V2 (`require_fk_for_moving_wrist=false`, `third_ns=/perception/third`):** `dwell_reason=stationary_session_imu_ok` (`118/121`), `dual_view=46/121` (`38.0%`), `8.09 Hz`, stage `total_ms` `55.70 / 232.14 ms`, `publish - obs` `260.19 / 438.96 ms`, `rx_minus_hdr_ms` p50 `85.99 ms` (wrist) / `84.42 ms` (side).
- **Parked-arm V3.1 (`prove`, `45 ms`, `third_ns=/perception/third`):** `dwell_reason=stationary_session_imu_ok` (`90/91`), `dual_view=37/91` (`40.7%`), `7.70 Hz`, stage `total_ms` `63.45 / 169.14 ms`, `v3_regions_ms` (`n=80`) `2.24 / 9.17 ms`, `publish - obs` `258.99 / 429.62 ms`, `rx_minus_hdr_ms` p50 `85.84 ms` (wrist) / `83.99 ms` (side).
- Camera-to-driver receive lag (`rx_minus_hdr_ms` ~85 ms p50) plus live camera/display contention (`~10–12 Hz` camera streams, `live_view.py` running) pushed `publish - obs` above the 150 ms WP5 freshness bound across all live runs.

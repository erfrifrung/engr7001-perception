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

## 5. V4 implications so fa
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

## 7. Round 2 live diagnostics, `live_node.py` optimisation, and MoveIt fault tests (Ryzen 5 5600 + WSL)

### 7.1 Transport, USB topology, and stale `/tf_static` diagnosis
- **USB & FastDDS SHM:** All 3 RealSense cameras share Bus 004 (`28:00.3` Matisse xHCI, 5000M); Bus 002 (`02:00.0`, 10000M) is empty. All 3 camera nodes use `fastdds_shm_profile.xml`, while pre-existing `live_view.py` runs without SHM. Disabling SHM (Cond F) stalled `wrist/color` and dropped `tripod/color` to `5.77 Hz` (`aligned_depth` `13.09 Hz`). Turning `align_depth=false` cut camera driver CPU ~3× (`15.7–17.7% -> 5.4–6.2%`) and saved ~8 ms.
- **Test 1 (Stale `/tf_static` diff, Fault #10 observed live):** Comparing the broadcast 19 Nov `/tf_static` against a fresh 26 Nov checkerboard fit (`current_board_tf.json`) showed `wrist_link` shifted `98.9 mm` (`[+72.2, +10.6, -66.8] mm`, `3.30°`), `tripod_link` `273.9 mm` (`38.11°`), `third_link` `149.1 mm` (`25.12°`), and pairwise camera-to-camera TFs `193.4–641.8 mm` (`22.86–63.17°`). Robot arm IP was unreachable (`FAILED` ARP) and `/joint_states` was absent.

### 7.2 Depth accuracy vs resolution/FPS (Test 3, `register_board.py --pitch-mm 20.0`)
| Mode | Gate | `wrist` RMS / nz / bias (min/med/max mm) | `tripod` RMS / nz / bias (min/med/max mm) | `third` RMS / nz / bias (min/med/max mm) |
|---|---|---|---|---|
| `720p15` (`1280x720x15`) | **PASS** | `0.192 px` / `99.98%` / `-1.88 / +1.08 / +4.62` | `0.200 px` / `96.62%` / `-0.60 / +4.77 / +9.21` | `0.166 px` / `90.96%` / `+0.88 / +6.76 / +11.76` |
| `480d15` (`720p` RGB / `480p` depth) | **PASS** | `0.190 px` / `100.00%` / `-0.26 / +1.82 / +4.57` | `0.197 px` / `99.75%` / `+2.01 / +4.78 / +7.16` | `0.159 px` / `99.02%` / `+4.20 / +7.69 / +11.31` |
| `480p30` (`848x480x30`) | **FAIL** | median bias `+2.52` (`0.122–0.139 px`, `97.85–100% nz`) | median bias `+2.91` | median bias **`+9.09 >= 8 mm`** |

Decision: keep `1280x720x15` as default and keep `third` excluded from V3 carving.

### 7.3 15 Hz vs 30 Hz ABAB + per-stage latency breakdown (Tests 2 & 4, untweaked `live_node.py`, `publish_overlay=false`)
| Run | `pub_hz` | `hdr->rx` p50/p95 | `rx->tick` p50/p95 | `compute` p50/p95 | `overhead` p50/p95 | `TOTAL` p50/p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|
| `A1_15hz_t15` | `12.58 Hz` | `85.5 / 136.7` | `10.8 / 22.9` | `16.7 / 104.1` | `3.1 / 66.4` | `118.6 / 248.9` |
| `A1_15hz_t30` (`rate_hz=30` on 15 Hz) | `14.82 Hz` | — | `25.1 / 114.8` | — | — | `155.9 / 304.6` |
| `B1_30hz_t30` (`480p30`, cams `29.99 Hz`) | `9.04 Hz` | `43.6 / 52.0` | `20.0 / 37.1` | `64.1 / 207.1` | — | `135.3 / 339.5` |
| `A2_15hz_t15` | `13.25 Hz` | `85.1 / 102.2` | `12.1 / 51.5` | `27.4 / 112.4` | `4.1 / 44.4` | `137.2 / 233.9` |
| `B2_30hz_t30` | `9.63 Hz` | `43.5 / 54.4` | `19.1 / 34.7` | `65.2 / 230.7` | — | `136.5 / 318.1` |

`848x480x30` held `29.99 Hz` (`38 ms` topic delay) across all 3 cameras on Bus 004 and halved `hdr->rx` (`85 -> 43.5 ms`), but 270 msg/s Python callbacks saturated the 2-thread executor GIL and inflated `compute` p50 (`17–27 -> 64–65 ms`).

### 7.4 `live_node.py` tweak, 180 s soak, golden bag, and MoveIt faults (`v3.1`, `68ec2da`)
- **Code changes & regression suite:** `live_node.py` splits `_build_view` into `_select_pair` + `_decode_view` (signature-gated copy skip via `pair_sig`/`health_sig`), uses zero-copy `np.frombuffer` for unpadded `rgb8`/`16UC1`, and skips `render_overlay()` when `pub_overlay.get_subscription_count() == 0` and not saving PNG. Local checks: `verify_zero_moveit.py` `PASS`, `pytest -q tests` **`62 passed, 2 xfailed`** (`86.27s`).
- **Live Ryzen A/B (`720p15`, `current_board_tf.json`, 3 wooden cubes):**
  - `publish_overlay=false`: `ok=238/262`, **`13.78 Hz`**, `dual_view=238/238 (100%)`, `TOTAL=121.6 / 232.7 ms` (`overhead=4.3 / 43.9 ms`).
  - `publish_overlay=true` (0 subs, PNG save): `ok=198/216`, **`12.67 Hz`**, `dual_view=177/198 (89.4%)`, `TOTAL=128.3 / 241.6 ms`; `overlay_00211.png` confirmed `resolved=3 amb=0` across wrist, side, and third views.
- **Test 5 (180 s soak at `720p15`):** `ok=2068/2091`, `11.63 Hz`, `dual_view=2053/2068 (99.3%)`, `TOTAL=171.1 / 307.2 ms` (drift `167.1/302.4 -> 174.3/318.4 ms`); axis-aligned middle cube `track 3` (`(-9.0, 131.5) mm`, `n=2068/2068`) held `std_xy=(0.09, 0.12) mm`, `std_z=0.13 mm`, `std_yaw=0.02°`, whereas two ~45°-rotated corner cubes churned track IDs `[1..3203]` (`eq3=1/2068`). Process RSS rose `535.6 -> 575.2 MB` over 180 s.
- **Test 9 (Fresh golden bag):** Recorded a 16.3 s (`3.0 GiB`, 2767 msgs) MCAP bag (`/tf_static` + 3 cameras' color/aligned_depth/camera_info) with `current_board_tf.json` on the lab external drive (not committed).
- **WSL MoveIt Faults #7 & #9 (`integration/live_test/`):** `run_twin_live_test.sh` updated to launch `wp3_live_node` directly (`orig 10/13, patched 13/13, json 3/3, mock 2/2`). Live MoveIt Fault #7 (`SIGKILL` publisher keeps `['wp3_blocks_1', 'wp3_blocks_2']` after 3.5 s) and Fault #9 (1.5 s silent health blocks empty `DELETEALL` snapshot, fresh `ok` health removes `['wp3_blocks_11', 'wp3_blocks_12']` -> `[]`) both passed.

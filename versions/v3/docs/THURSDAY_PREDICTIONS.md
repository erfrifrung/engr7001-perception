# Thursday 26 Nov lab session: predictions written before testing

Rubric: "Every result needs a requirement before it and a prediction beside it" (Project Guide p. 5). Fill in the **Measured** column on the day and explain every miss in the decision log. A miss with a good explanation is worth more than a hit.

Setup: arm parked; V3.1 branch, `--v3 off` (V2 path) unless stated; `pointcloud.enable:=false` (already the default in `scripts/start_camera.sh`); 1280 × 720 @ 15 Hz.

## 0. Pre-checks (5 min, before any measurement)

| Check | Command / how | Expect |
|---|---|---|
| Joint states present | `ros2 topic hz /joint_states` | ≥ 30 Hz. **If absent, the wrist is treated as moving and the node falls back to `side_only`** (this explains the 94/138 `side_only` epochs in `cpu_check.json`). Then either start the xArm read-only driver or launch with a profile that sets `require_fk_for_moving_wrist: false` (IMU + sharpness gate, parked arm only). |
| RealSense global time | `ros2 node list`, then `ros2 param list /perception/wrist \| grep -i global_time` and `ros2 param get` on what it lists | Enabled (librealsense default). It is **not** an `rs_launch.py` argument, so `start_camera.sh` can't set it; the exact parameter name depends on the wrapper version. |
| Node config | first log line of `wp3_live_node` | `enable_v3_regions=False ... require_fk_for_moving_wrist=...` |
| `/wp3_live/health` | `ros2 topic echo /wp3_live/health --once` | `status: ok`, `dwell_reason` not `fk_missing_or_moving` |

## 1. Calibration verification (task 3332)

| Quantity | Prediction | Reasoning | Measured |
|---|---|---|---|
| Board reprojection RMS per camera | ≤ 0.3 px | previous fits 0.06–0.19 px (`three_camera_tf.json`) | **Ryzen 5 5600 (`720p15`, `register_board.py --pitch-mm 20.0`)**: `wrist` 0.192 px, `tripod` 0.200 px, `third` 0.166 px (all ≤ 0.3 px; `480p30`: 0.122–0.139 px) |
| 39 mm matte step, wrist and tripod, 4 positions | error ≤ ±2 mm at every position | previous checks within ~1 mm on good patches | **Not run** (no Jenga/step artifact on hand; board-normal median bias at `720p15`: `wrist` +1.08 mm, `tripod` +4.77 mm, max +9.21 mm) |
| 39 mm step, third camera | ≥ 1 of 4 positions off by > 3 mm | earlier note: 33.5 mm on patch 1 vs 38.7–41.0 mm elsewhere (`Hazem_START_HERE.md`) | **Not run** (board-normal median bias at `720p15`: `third` +6.76 mm, max +11.76 mm; `480p30` +9.09 mm **fails** `< 8 mm` smoke gate) |
| Decision | third camera: seam evidence only, **no V3 carving**, unless all 4 positions are within ±2 mm | | Confirmed: keep `1280x720x15` default; `third` stays excluded from V3 carving |

## 2. Table → robot base (task 3331 / 3341)

| Quantity | Prediction | Reasoning | Measured |
|---|---|---|---|
| Fit RMS (`fit_table_transform`, planar, ≥ 4 points) | 1.5–3 mm | fingertip contact ~1–2 mm plus manual-teach repeatability; ground truth from a 3D-printed or paper target | **Blocked / not run**: arm unreachable (`FAILED` ARP), `/joint_states` absent |
| Independent check point error | ≤ 3 mm | | **Blocked / not run** |
| M1 acceptance (provisional) | RMS ≤ 3 mm, stated with its uncertainty | | **Open** until `/joint_states` and touch-off run on lab arm |

## 3. Live V2 path, parked arm, printed blocks

| Quantity | Prediction | Reasoning | Measured |
|---|---|---|---|
| Scene rate | 12–15 Hz | 13.16 Hz two-view while recording three streams; 11.5 Hz in `cpu_check.json` | **Ryzen 5 5600 (`require_fk_for_moving_wrist: false`, 3 wooden cubes, `720p15`, `live_view.py` active)**: untweaked `12.58–13.25 Hz` (`publish_overlay=false`); **tweaked `live_node.py`**: **13.78 Hz** (`publish_overlay=false`), **12.67 Hz** (`true`, 0 subs), **11.63 Hz** (180 s soak). At `480p30`: `9.04–9.63 Hz` (Python GIL saturation) |
| `dual_view` fraction | ≥ 80 % **with** `/joint_states`; ≈ 10 % without | dwell gate (section 0) | With `require_fk_for_moving_wrist: false` (`/joint_states` absent) and fresh `current_board_tf.json` on tweaked `live_node.py`: **100%** (`238/238`, `ov=false`), **89.4%** (`177/198`, `ov=true`), **99.3%** (`2053/2068`, 180 s soak). (Pre-fix stale 19 Nov `/tf_static` masked TF drift behind 79–99% pairing) |
| `publish − observation` (processing), p50 / p95 | 30–45 ms / ≤ 100 ms | V2 path; `cpu_check.json` 30.7 / 91.2 ms | Untweaked `720p15`: `compute` `16.7–27.4 / 104.1–112.4 ms`, `rx->tick` `10.8–12.1 ms`, `overhead` `3.1–4.1 ms`; at `480p30`: `compute` `64.1–65.2 / 207.1–230.7 ms`. Tweaked `720p15` end-to-end (`TOTAL`): **121.6 / 232.7 ms** (`ov=false`), **128.3 / 241.6 ms** (`ov=true`) |
| Capture → bridge age, p50 | 110–150 ms | recorded sample: `age_ms` 134 with 85 ms receive lag (`display_replay_scene_observations.json`) | `hdr->rx` `85.1–85.5 ms` (`720p15`) / `43.5–43.6 ms` (`480p30`); tweaked `live_node.py` capture→publish p50: **121.6 ms** (`ov=false`), **128.3 ms** (`ov=true`), **171.1 ms** (180 s soak) |
| WP5 freshness bound 150 ms met at p95 | **Possibly not**: margin is ~15 ms | if missed, the cause is likely USB/driver receive lag, not compute; record `rx_minus_hdr_ms` | **Missed at p95** (`232.7 / 241.6 ms` tweaked `720p15`; `307.2 ms` 180 s soak): ~85 ms camera/driver buffer delay at 15 Hz + compute/tail spikes. `480p30` halved `hdr->rx` (`43.5 ms`) but flooded Python GIL callbacks (`TOTAL` p50 `135.3–136.5 ms`, p95 `318.1–339.5 ms`) |
| Single block, separated pair, seamed touching pair | correct count, 0 wrong poses | | **Printed blocks not tested** (none on hand). With **3 wooden cubes** + `current_board_tf.json`: `resolved=3, amb=0` (`overlay_00211.png`); 180 s soak held axis-aligned center cube (`std_xy=(0.09, 0.12) mm`, `std_z=0.13 mm`, `std_yaw=0.02°`, `n=2068/2068`) while two 45°-rotated corner cubes churned track IDs `[1..3203]` |
| Plain flush pair | ambiguous cluster, 0 poses | no seam cue | **Not tested live** (no Jenga blocks on hand; synthetic `v3prove_mc` left `16` `flush_pair` and `36` `flush_triple` unresolved with `0` wrong) |

## 4. Ryzen replay benchmark (`bash scripts/ryzen_replay_bench.sh <bags>`)

Bags: `three_cam_demo_bag`, `wood-05-three-touching`, `block-moves`, `holdout-rehearsal` (45 mm wooden props: timing and false-split evidence only).

| Quantity | Prediction | Reasoning | Measured |
|---|---|---|---|
| Synthetic held-out, `tuned_mc` p50 / p95 (one pinned core) | 20–35 / 30–50 ms | M4 18.9 / 24.0 ms; Ryzen slower per core in Python (projection, not measured) | **Ryzen 5 5600** (3 rounds, 168 scenes / 312 blocks, `taskset -c 2`): **33.0 / 41.6 ms** (`max 51.6 ms`), `86/312` (`cov5=0.276`) |
| `v3prove_mc` p50 / p95 | 60–90 / 110–150 ms | M4 46.5 / 69.8 ms × 1.3–1.9 | **Ryzen 5 5600**: **68.6 / 114.9 ms** (`max 172.8 ms`, `54.1% > 66.7 ms`), `190/312` (`cov5=0.609`) |
| `v3adapt_mc` vs `v3prove_mc` | same or fewer published blocks (by construction); latency within ±20 % | sound adaptive solver | **Ryzen 5 5600**: identical `190/312` (`cov5=0.609`), **69.2 / 125.5 ms** (vs `68.6 / 114.9 ms`, within ±20%) |
| Wrong poses (5 mm / 5° / 3 mm Z), all methods | 0 | synthetic held-out history | **0** across all 4 methods (`3` rounds each) and all `25` budget-sweep runs |
| Bags, V2 vs V3.1 adaptive at 45 ms | 0 false splits on wooden props; V3 budget exits on ≤ 10 % of frames | | With `historical_props_12nov.json`: V2 p50 `5.36–11.63 ms`; V3 (`prove_45ms`) ran on `14/142` (`8.36 / 83.16 ms`) on `three_cam_demo_bag` and `137/438` (`9.03 / 55.30 ms`) on `block-moves`, `0/129` on `wood-05-three-touching`, `0/159` on `holdout-rehearsal`; synthetic sweep `enum` `45 ms` reached `169.0/312` vs `175.0/312` (`≥66 ms`), while `prove` `45 ms` hit `150` timeouts (`87.0/312`) |
| CPU % of one core, V2 path replay | 80–100 % (single-threaded Python) | | Pinned single-core (`taskset -c 2`) V2 replay p50 `5.36–11.63 ms`; live 3 camera drivers used `15.7–17.7%` CPU each with `align_depth=true` vs `5.4–6.2%` with `false` |

## 5. What would change the plan

- `dual_view` stays low **with** joint states present → camera sync or USB is the problem: order the sync cable / USB card now.
- Capture → bridge p95 > 150 ms → WP5's bound must be renegotiated or the receive lag fixed. Compute alone won't fix it.
- Any wrong pose on printed blocks → stop, keep that frame, add it as a regression test before anything else.

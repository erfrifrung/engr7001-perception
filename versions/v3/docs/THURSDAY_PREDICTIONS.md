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
| Board reprojection RMS per camera | ≤ 0.3 px | previous fits 0.06–0.19 px (`three_camera_tf.json`) | |
| 39 mm matte step, wrist and tripod, 4 positions | error ≤ ±2 mm at every position | previous checks within ~1 mm on good patches | |
| 39 mm step, third camera | ≥ 1 of 4 positions off by > 3 mm | earlier note: 33.5 mm on patch 1 vs 38.7–41.0 mm elsewhere (`Hazem_START_HERE.md`) | |
| Decision | third camera: seam evidence only, **no V3 carving**, unless all 4 positions are within ±2 mm | | |

## 2. Table → robot base (task 3331 / 3341)

| Quantity | Prediction | Reasoning | Measured |
|---|---|---|---|
| Fit RMS (`fit_table_transform`, planar, ≥ 4 points) | 1.5–3 mm | fingertip contact ~1–2 mm plus manual-teach repeatability; ground truth from a 3D-printed or paper target | |
| Independent check point error | ≤ 3 mm | | |
| M1 acceptance (provisional) | RMS ≤ 3 mm, stated with its uncertainty | | |

## 3. Live V2 path, parked arm, printed blocks

| Quantity | Prediction | Reasoning | Measured |
|---|---|---|---|
| Scene rate | 12–15 Hz | 13.16 Hz two-view while recording three streams; 11.5 Hz in `cpu_check.json` | |
| `dual_view` fraction | ≥ 80 % **with** `/joint_states`; ≈ 10 % without | dwell gate (section 0) | |
| `publish − observation` (processing), p50 / p95 | 30–45 ms / ≤ 100 ms | V2 path; `cpu_check.json` 30.7 / 91.2 ms | |
| Capture → bridge age, p50 | 110–150 ms | recorded sample: `age_ms` 134 with 85 ms receive lag (`display_replay_scene_observations.json`) | |
| WP5 freshness bound 150 ms met at p95 | **Possibly not**: margin is ~15 ms | if missed, the cause is likely USB/driver receive lag, not compute; record `rx_minus_hdr_ms` | |
| Single block, separated pair, seamed touching pair | correct count, 0 wrong poses | | |
| Plain flush pair | ambiguous cluster, 0 poses | no seam cue | |

## 4. Ryzen replay benchmark (`bash scripts/ryzen_replay_bench.sh <bags>`)

Bags: `three_cam_demo_bag`, `wood-05-three-touching`, `block-moves`, `holdout-rehearsal` (45 mm wooden props: timing and false-split evidence only).

| Quantity | Prediction | Reasoning | Measured |
|---|---|---|---|
| Synthetic held-out, `tuned_mc` p50 / p95 (one pinned core) | 20–35 / 30–50 ms | M4 18.9 / 24.0 ms; Ryzen slower per core in Python (projection, not measured) | |
| `v3prove_mc` p50 / p95 | 60–90 / 110–150 ms | M4 46.5 / 69.8 ms × 1.3–1.9 | |
| `v3adapt_mc` vs `v3prove_mc` | same or fewer published blocks (by construction); latency within ±20 % | sound adaptive solver | |
| Wrong poses (5 mm / 5° / 3 mm Z), all methods | 0 | synthetic held-out history | |
| Bags, V2 vs V3.1 adaptive at 45 ms | 0 false splits on wooden props; V3 budget exits on ≤ 10 % of frames | | |
| CPU % of one core, V2 path replay | 80–100 % (single-threaded Python) | | |

## 5. What would change the plan

- `dual_view` stays low **with** joint states present → camera sync or USB is the problem: order the sync cable / USB card now.
- Capture → bridge p95 > 150 ms → WP5's bound must be renegotiated or the receive lag fixed. Compute alone won't fix it.
- Any wrong pose on printed blocks → stop, keep that frame, add it as a regression test before anything else.

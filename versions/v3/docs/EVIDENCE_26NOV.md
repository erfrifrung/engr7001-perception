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

## 4. Budget sweep: NOT RUN (usage limit)
Ready to run: `bash bench/budget_sweep.sh` (prove/enumerate × 30/45/66/120/250 ms/unlimited, 2 rounds, pinned, plus a 60 000-node cap). `bench/sweep_analyze.py` separates timeouts and counterexamples, reports correct and wrong poses and latency, and lists the blocks still unresolved at an unlimited budget. `run_scenes.py` now logs the V3 stage's per-region outcomes.

## 5. V4 implications so far
- On real frames the **front end** dominates (proposals and RGB seam checks, then heightmap and deprojection), not the solver. Selective computation (skip unchanged regions) and moving the RGB seam scoring off the Python path are the first V4 targets.
- The solver matters only on touching or flush scenes; quantify it with the sweep.
- Measure only on an uncontended host with one node, the bag pre-read, rotated order and repeated runs.

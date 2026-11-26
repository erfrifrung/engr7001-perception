# WP3 perception V3 / V3.1 / V3.2

V3 = the V2 three-view candidate + the M4 fixes + the V3 region stage (`wp3_live/tiling.py`, `wp3_live/v3_regions.py`), CPU fast paths, the V3 tracker options and an optional async mask scheduler (`wp3_live/mask_scheduler.py`). The scene contract is unchanged (`wp3_scene_observation_v2.1`). The original V3 report (M4, synthetic) is not in this repository.

## V3.2 (branch `v3.2`) — the three safety findings from the 26 Nov lab session

Same perception and solver as V3.1 (no change to `tiling.py`, `v3_regions.py`, proposals or the scene schema). Every change has a V3.1 switch, and a test that reproduces the V3.1 failure first.

| 26 Nov finding | V3.2 change | Switch (V3.1 value) | Evidence |
|---|---|---|---|
| **Duplicate-track spiral**: a 180 s soak produced 3203 track ids for 3 objects. One split or missed association left two tracks at one spot; the near-tie veto then rejected every later detection there, and each one spawned another track. | Tracks on one layer closer than 0.4 × block width cannot be two blocks (two blocks' inscribed circles would overlap): merged into the older id, which must re-confirm before it is a target again. | `tracker_merge_overlapping` (false) | `tests/test_v32_tracker.py`: V3.1 mode reproduces > 30 ids for one block; V3.2 keeps id 1. Touching blocks (25 mm) and stacked blocks are never merged. |
| **Fault #12**: a block hidden > 3 s with healthy cameras (e.g. under the arm) was retired and removed from MoveIt. | A track that is not detected is retired only when 3 consecutive epochs show its footprint **empty** (depth observed, clearly below its top: `wp3_live/free_space.py`), or after 60 s unseen. A removed block now clears in ~0.2 s instead of 3 s. | `retire_on_observed_free` (false), `hidden_retire_max_s`, `free_confirm_frames` | `tests/test_v32_tracker.py`: hidden 20 s under an occluder + 5 s without depth → still an obstacle; seen empty → retired after 3 epochs. |
| **Fault #10, observed live**: every camera had moved 99–274 mm since the published calibration; WP3 cached the fixed transforms for its lifetime and kept fusing. | (1) the node re-reads `/tf_static` every second: a change starts a new `calibration_epoch` (`…#r1`) and drops old tracks; (2) per-camera table-plane check (> 6° or > 15 mm at the workspace centre, twice in a row ⇒ that camera is not fused; none left ⇒ `status: calibration_mismatch`; 2.5–6° is `suspect`, reported and still fused); (3) `scripts/check_calibration_drift.py` for the session-start board check (catches the slides / pans the plane check cannot see). | ROS params `static_tf_refresh_s` (≤ 0), `calibration_check` (false) | `tests/test_v32_calibration.py`, `tests/test_v32_node_calibration.py` (live node picks up a new `/tf_static` without a restart). |

**Real-data check (12 Nov 3-camera bag → node → patched bridge → fake MoveIt, laptop WSL, no ground truth):**
- V3.2 and V3.1-tracker modes both give 3 track ids, 3 MoveIt objects and 0.2–0.3 mm median per-track spread (no regression; this bag never triggered the spiral).
- **The first plane-check thresholds were wrong on real data.** They were 2.5°, with the offset measured at the camera. The bag's tripod depth plane is 4.7° off its board calibration (oblique depth bias; the table is +3 mm at the workspace), so the check rejected it: dual view fell from 12/47 to 2/54 epochs.
- After the fix (6° reject, 2.5° `suspect`, offset at the workspace centre): dual view 9/25 (V2 path) and 15/26 (V3) with the check on, vs 15/35 with it off. Tripod `suspect`, wrist `ok`. Regression test: `test_12nov_bag_tripod_depth_tilt_is_not_rejected`.

Not changed in V3.2 (still open): oblique-camera depth bias (+4.8 / +6.8 mm), tail latency (p95 ≈ 230–300 ms), the half-occluded-slab hazard (D6), Hazem's bridge ignoring `calibration_epoch` (restart the bridge after a recalibration).

## V3.1 (branch `v3.1`) — what changed and how to run it

**Live default = the V2 path.** The V3 region stage is opt-in:

```bash
bash scripts/launch_live.sh --policy merged_corroborated --config <profile.json>          # M1 default (V2 path)
bash scripts/launch_live.sh --v3 on --v3-solver adaptive --v3-budget-ms 45 ...            # evaluate V3
```

| Area | Change | Evidence |
|---|---|---|
| Config | `enable_v3_regions` defaults to False; all `v3_*` keys and the tracker options load from JSON; the node exposes them as ROS parameters | `tests/test_v3_regions.py::test_json_profile_can_switch_v3` |
| Frame authority | no camera→table transform ⇒ `status: "uncalibrated"`, no poses; V3 carving only in the qualified table frame | `test_uncalibrated_primary_publishes_no_table_poses` |
| Timing honesty | `publish_stamp_ns` after compute; robot TF at the frame stamp (bounded fallback) | |
| Tracker | layer gate in association; slide-safe re-anchoring; retired bookkeeping dropped | `tests/test_v31_tracker.py` |
| Solver | symmetric `same()`; deadline-checked placement generation (truncation ⇒ TIMEOUT); no repeated dense search; optional sound `adaptive` solver (`prove` stays default: adaptive was slower and published less on the held-out split) | `tests/test_v31_solver.py`, held-out table below |
| Tried and reverted (measured) | yaw from FREE-bordered sides only (L layout 1.30 mm vs < 1.0 mm); measured footprint dims (read 30.5 mm widths for 25 mm blocks); per-pool search structures (prove 134 vs 169 published) | notes in `wp3_live/tiling.py` |
| Seam cue | periodic-stripe alias check restored (was disabled by a hard-coded 0) | striped-slab ablation |
| Robot space | wrist-camera housing is UNKNOWN; upper layers need `link6` TF | |
| Twin bridge (on Hazem's twin v5) | snapshot semantics (retired tracks removed); non-`ok` messages ignored; removals held while WP3 health says it is blind; blocks in the gripper left to the twin, with a 1 s grace after release; rejected updates retried; v5's finger-contact collision entries kept | `tests/test_v3_contract_twin.py`; live A/B against fake MoveIt in `../../integration/live_test/` (original v5: 10/13, patched: 13/13); patch `../../integration/twin_bridge_v31.patch` |
| Scoring | Z gated; yaw / Z percentiles reported | `bench/score.py` |

**Known, pre-existing hazards** (strict xfail tests, so any change is noticed): the visible half of a half-occluded 75 × 50 slab is published as a block, and a block-sized object floating at a layer-multiple height is published as a stacked block (V2 single-block path). Mitigation: declared lab object set and robot self-filter with TF. See `docs/INTERFACE_SPEC_V1.md` D6.

**Front-end change vs V2:** `PolicyThresholds.edge_first_directions` (default True). On the 12 Nov wooden-prop bag it changes yaw by up to 8.2° on one epoch relative to V2; with it False the output reproduces V2's golden exactly. Settle it with printed-block ground truth.

Docs: `docs/INTERFACE_SPEC_V1.md` (for task 6103), `docs/THURSDAY_PREDICTIONS.md`, `docs/CHAMFER_PROPOSAL.md`.

## Tests and benchmarks

```bash
python3 -m pytest -q tests                       # with Jazzy sourced and ros2_ws built; WP3_REQUIRE_GOLDEN_BAG=1 to require the bag
python3 scripts/verify_zero_moveit.py
bash scripts/ryzen_replay_bench.sh <bag> ...     # lab Ryzen: held-out synthetic + bag replays, pinned core
```

`scripts/amd_replay_bench.sh` / `amd_hw_check.sh` expect the original V3 bundle layout and are kept for reference; use `ryzen_replay_bench.sh` in this repository.

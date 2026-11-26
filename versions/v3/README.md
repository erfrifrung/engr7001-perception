# WP3 perception V3 / V3.1

V3 = the V2 three-view candidate + the M4 fixes + the V3 region stage (`wp3_live/tiling.py`, `wp3_live/v3_regions.py`), CPU fast paths, the V3 tracker options and an optional async mask scheduler (`wp3_live/mask_scheduler.py`). The scene contract is unchanged (`wp3_scene_observation_v2.1`). The original V3 report (M4, synthetic) is not in this repository.

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

# WP3 Live v2 Three View Candidate

See [Hazem's handoff guide](../START_HERE.md) for SSH access, calibration, launch, RViz and verification. The original live v2 remains a separate reference in Shadi's workspace.


## V3 (this directory)

V3 = this V2 three-view candidate + the previous M4 fixes + the V3 region stage (`wp3_live/tiling.py`, `wp3_live/v3_regions.py`), CPU fast paths, the V3 tracker options and an optional async mask scheduler (`wp3_live/mask_scheduler.py`). Every V3 feature has a `RuntimeConfig` switch; `enable_v3_regions=False` gives tuned V2. The scene contract is unchanged (`wp3_scene_observation_v2.1`). Results, decisions and reproduction commands: `../results/V3_REPORT.md`. Target-machine entry points: `scripts/amd_hw_check.sh`, `scripts/amd_replay_bench.sh`. Benchmarks: `bench/`.

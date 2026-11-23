# Local verification

Checked on 23 November 2026 in WSL Ubuntu 24.04 with ROS 2 Jazzy. Each ROS package was built in its own overlay, and display loopbacks ran on local-only ROS domains.

| Snapshot | colcon build | Static perception-only audit | Non-ROS tests |
|---|---|---|---|
| V1 | Passed | Passed | 5 passed |
| V2 | Passed | Passed | 13 passed, 1 skipped |
| V2 three-view candidate | Passed | Passed | 15 passed, 1 skipped |
| V2 three-view handoff | Passed | Passed | 16 passed, 1 skipped |

All 49 non-ROS tests passed. Three recorded-bag tests skipped because their private fixtures are held separately. The WP3 twin-integration math and bookkeeping checks passed without ROS. The wrist-only prototype previously passed 3 tests and the two heightmap snapshots passed 6 and 9 tests, respectively, with one separate historical heightmap replay test skipped.

## ROS display loopback

- **V1 and V2.** Both passed at portfolio export. In a later two-run check from freshly built overlays, V1 failed twice and V2 passed once. The failed runs missed the initial two-track dual-view scene.
- **V2 three-view candidate.** Two of three retest runs passed both cases. In the other run the third-camera case received no initial scene. The export also recorded a dropout case that stayed in `dual_view` when it expected `wrist_only`.
- **V2 three-view handoff.** Both cases passed in one post-merge run. Later runs reproduced failures in initial dual-view detection, wrist RGB-D timestamp pairing and fallback. Some dropout cases reported `wrist_unqualified` when the test expected `wrist_only`.

Live fallback still needs bench verification. The MoveIt receiver and physical hardware were outside these checks.

## Replay and synthetic checks

- The nine-scene synthetic benchmark reproduced 8/9 correct outcomes for `permissive_rgb_seam` (one false-confident outcome) and 8/9 for `whole_partition_mv` (zero false-confident outcomes).
- A labelled replay of the opening 3.2 seconds of the 12 November stationary recording exercised both V1 controls and both heightmap baselines. The `historical_props_12nov` profile used saved transforms, 18 wrist-led epochs and four exactly paired tripod RGB-D observations. Scoring used 51 labelled block opportunities at a 15 mm match threshold, with reference positions derived from camera data. The short segment verifies the comparator paths. A separate measurement would be needed to establish independent pose accuracy.
- The README demo uses the recorded 19 November three-touching-block scene. V2 in wrist-plus-tripod mode held B01–B03 as measured across 60 consecutive replay epochs and reported zero ambiguous clusters. The selected interval shows stationary tracking.

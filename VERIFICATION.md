# Local verification

Portfolio export checked on 23 November 2026 using WSL Ubuntu 24.04 and ROS 2 Jazzy. Each ROS package was built and tested in its own overlay on an isolated local ROS domain.

| Snapshot | colcon build | Static perception-only audit | pytest |
|---|---|---|---|
| V1 | Passed | Passed | 6 passed |
| V2 | Passed | Passed | 14 passed, 1 skipped |
| V2 three-view candidate | Passed | Passed | 16 passed, 1 skipped, 1 failed |
| V2 three-view handoff | Passed | Passed | 17 passed, 1 skipped, 1 failed |

The recorded-bag tests skipped because their private fixtures are held separately. In the candidate’s three-camera ROS loopback, the last reported mode was `dual_view` when the test expected `wrist_only` after the dropout. The handoff’s two-camera ROS loopback failed to observe the expected initial two-track dual-view scene in the final run. A preceding run of that same test instead reached the dropout and reported `wrist_unqualified` where it expected `wrist_only`. These archived tests need follow-up before relying on either candidate for live fallback.

The wrist-only prototype passed 3 tests and the two heightmap snapshots passed 6 and 9 tests, respectively. One historical heightmap replay test skipped. The separate WP3 twin-integration math and bookkeeping checks passed without ROS. The MoveIt receiver and physical hardware were outside this verification.

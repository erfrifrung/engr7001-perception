# ENGR7001 robotic perception

It tracks the development of tabletop block pose estimation, segmentation and multi-object tracking for an intent-aware shared-manipulation project. The ROS 2 perception nodes publish observations and RViz displays.

| Snapshot | Main change |
|---|---|
| [Early prototypes](prototypes/) | Wrist-only RGB-D segmentation and the tabletop heightmap comparison with and without RGB seam evidence. |
| [V1](versions/v1/) | Consolidated heightmap, RGB seam cues, side-view verification and a shared live/replay detector with configurable policies. |
| [V2](versions/v2/) | Whole-partition scoring, improved cross-camera depth and skew handling, temporal caching, and explicit tracking and ambiguity states. |
| [V2 three-view candidate](versions/v2-three-view/) | Option to corroborate seams with a third fixed view, plus checkerboard registration tools. |
| [V2 three-view handoff iteration](versions/v2-three-view-handoff/) | Follow-on snapshot from the expanded handoff, with changes to multi-view cut verification, replay, caching, diagnostics and tests. |
| [WP3 twin integration](integration/) | RViz configuration and the discrete WP3 additions for frame fitting and a trial multi-block receiver. |

## Development evidence

- On one recorded row of three touching wooden blocks, the RGB seam detector returned the correct count in **135/141 frames versus 0/141 with depth alone**, flagging the remaining six as ambiguous.
- In a stationary **720p** lab trial while recording all three streams, Fast DDS shared memory increased live **two-view** scene updates from **2.42 to 13.16 Hz** (rounded to 2.4 and 13.2 Hz on the CV).
- These development runs used recorded wooden props and a stationary rig. Independent physical accuracy on 3D-printed blocks, comparative accuracy of the third view, and acceptance in the full twin or planning workflow remain open.

Start with the [setup and replay guide](SETUP.md). See the [source and sharing inventory](INVENTORY.md) for attribution and the [local verification](VERIFICATION.md) for build and test results.

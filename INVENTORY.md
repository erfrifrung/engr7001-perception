# Source and sharing inventory

## Included

| Content | Source and scope |
|---|---|
| V1 and V2 | The original separate `WP3-Live-Implementation` and `WP3-Live-Implementation-v2` development folders. Includes source, ROS 2 overlays, algorithm presets, launch and replay scripts and tests. |
| Both three-view snapshots | Shadi’s initial WP3 handoff and the expanded handoff copy. Archived separately because the expanded copy changes three-view verification, overlay, replay and tests. |
| Wrist-only and heightmap prototypes | Selected offline source and tests from the 18 November development work and read-only bag/projection helpers from the 12 November analysis. Historical paths and relative references in the prototype code describe the original workspace; the associated recordings and labels stay with that workspace. |
| Recorded replay GIF | Cropped wrist and tripod views with the V2 perception and tracking overlay, generated from a short stationary lab segment. The source bag and full capture stay private. |
| Stationary baseline overview image | Three-camera stationary baseline still showing wrist, tripod and third RGB views, published WP3 `jenga_table` poses (`B01`–`B03`), and a separate illustrative WP2 xArm 6 reference view. |
| Twin-facing additions | Shadi’s handoff identifies the standalone `fit_table_tf.py`, `wp3_table_tf.py`, `wp3_link.py`, `wp3_scene_bridge.py` and `test_wp3_offline.py` as WP3 additions to Hazem’s twin. Includes the WP3 RViz display configuration. |

V1 draws on the earlier WP3 heightmap and RGB seam experiments, as well as geometry from the tabletop perception prototype. The heightmap experiments credit the Song et al. orthographic heightmap approach in the source. Policy and comparator names preserve the distinctions in the original research snapshots. Original package maintainer and license metadata are retained. The portfolio does not add licensing terms for other contributors’ work.

## Original upstream

- **Ant Royster, team repository and WP5 planning.** The public [upstream project](https://github.com/antroyster/MEng-Project-7-Shared-Autonomy-) hosts the digital twin archives. The reviewed original single-pose bridge is in the [pinned v1 archive](https://github.com/antroyster/MEng-Project-7-Shared-Autonomy-/blob/dd2ed06bf3f4675d3c0dfaa67e046c2a7478d168/digital_twin_v1.zip). The relevant repository trees and metadata did not show a redistribution license.
- **Hazem, WP2 digital twin.** His bridge, scene builders, robot launch and controls and the expanded twin bundle are upstream/team material. The handoff’s edit to the existing `build_scene.py` stays with that copy. Only the separate WP3 additions are included here.

## Held locally or kept with upstream

- Drive contains the private 19 November lab recordings and runs, plus shared project schedules and other team documents. Raw bags, calibration captures and transforms, camera serials, lab connection scripts, logs, screen recordings and team documents stay out of the archive.
- Bundled manufacturer meshes, vendor dependencies, model weights, private archives, generated ROS workspaces and third-party papers stay with their original sources.
- The local historical bag path is replaced by a repo-relative fallback in the regression test, and the recorded half of the golden fixture is withheld. The synthetic golden data is included. The 720p15 RealSense launcher takes a local serial argument instead of embedding the lab’s device mapping.
- The pre-V1 combined lab bundle, live robot setup, and mixed-provenance experiments stay local. Their relevant chain is represented by the earlier offline prototypes and separate WP3 implementation snapshots.

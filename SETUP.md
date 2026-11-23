# Setup and replay

The live overlays target Ubuntu 24.04 and ROS 2 Jazzy. The detector uses Python, NumPy, SciPy and OpenCV. The ROS package uses ROS image, geometry, vision, marker and TF messages and cv_bridge. The RealSense camera drivers and RViz are separate dependencies.

## Run the included synthetic comparison

With Python, NumPy, SciPy, OpenCV and pytest installed, work from one version directory at a time. Each snapshot uses the same Python module name.

```bash
cd versions/v2
python3 -m pytest -q tests --ignore=tests/test_ros2_display_node.py
python3 scripts/verify_zero_moveit.py
python3 scripts/run_printed_jenga_suite.py --out results/synthetic
```

The synthetic suite renders its inputs. The recorded-bag parity check needs the original approved lab recording and its expected-output fixture. Use a locally authorised SQLite rosbag2 recording for comparisons with your own data, and choose the right configuration and frame for that capture.

```bash
bash scripts/compare_bag.sh /path/to/approved/bag_directory \
  --out results/replay --config ros2_ws/src/wp3_live_perception/config/historical_props_12nov.json
```

## View a live desktop session

Start the ROS 2 Jazzy environment and the RealSense publishers independently. Use aligned depth, RGB and camera info in the wrist and fixed-side namespaces, with a calibrated TF tree for the current camera positions. Use one publisher for each static transform. Choose an isolated ROS domain and the same discovery settings for your source, node and viewer.

```bash
source /opt/ros/jazzy/setup.bash
bash scripts/launch_live.sh --help
bash scripts/launch_live.sh --wrist-ns /perception/wrist --side-ns /perception/tripod
```

The launcher builds its local colcon overlay on first use. In RViz, set the fixed frame to the frame in the messages, normally `jenga_table`, and display `/wp3_live/markers`. An [RViz configuration](integration/wp3_smoke.rviz) also shows the camera overlay. For the three-view snapshots, use the optional `--third-ns /perception/third` flag after independently checking that camera’s pose and timestamps. The checkerboard registration scripts require a local orientation reference and a physically checked board pitch.

The three-view snapshots also include the lab’s 720p15 RealSense launcher with the serial supplied at runtime. From either three-view directory, set the serial locally and use the corresponding role in each camera terminal. Use the same Fast DDS environment in the detector and viewer terminals.

```bash
bash scripts/start_camera.sh wrist "$WRIST_SERIAL"
# In the detector terminal:
source scripts/lab_env.sh
bash scripts/launch_live.sh --third-ns /perception/third
```

For an isolated ROS loopback test, source Jazzy and build the version’s overlay before running pytest.

```bash
source /opt/ros/jazzy/setup.bash
(cd ros2_ws && colcon build --symlink-install)
source ros2_ws/install/setup.bash
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export ROS_DOMAIN_ID=191
python3 -m pytest -q tests
```

## WP3 and the digital twin

Hazem owns the WP2 twin and synchronisation. Ant Royster owns WP5 planning and target eligibility. The original twin bridge accepts one `geometry_msgs/PoseStamped` on `/perception/block_pose`. The WP3 ROS node publishes an RViz `MarkerArray` on `/wp3_live/markers` and a structured observation on `/wp3_live/scene_observation`. The latter carries per-instance IDs and validity, with timestamp and frame information. See the [upstream project](https://github.com/antroyster/MEng-Project-7-Shared-Autonomy-) and [WP3 publisher contract](versions/v2/wp3_live/contract.py).

Start with the RViz marker display. Before interpreting observations in the twin’s `world` frame, verify the table-to-world transform and the frame convention. The optional [WP3 receiver](integration/wp3_scene_bridge.py) mutates the MoveIt planning scene and requires an authorised copy of Hazem’s `build_scene.py` and the MoveIt messages. Keep receiver trials in isolated simulation until the team validates frame alignment, replay/reset behaviour and downstream handling of stale or ambiguous observations. An expired track alone cannot establish vacancy.

The offline transform and bookkeeping checks run without ROS.

```bash
python3 integration/test_wp3_offline.py
```

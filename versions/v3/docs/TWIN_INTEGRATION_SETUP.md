# Run WP3 V3.2 with the WP2 digital twin

This setup runs the RealSense cameras, the WP3 node and Hazem's WP2 digital twin on one ROS 2 Jazzy machine. The bridge patch in `integration/twin_bridge_v31.patch` passes the WP3 scene to MoveIt. It is already applied in the V3.2 lab twin package.

## One-time setup

Install rosbridge and check which xArm workspace is already built. The twin launcher sources `~/xarm_ws/install/setup.bash`. If the workspace is built at `~/xarm6_ws` and `~/xarm_ws` does not exist, run `ln -s "$HOME/xarm6_ws" "$HOME/xarm_ws"`.

```bash
sudo apt install -y ros-jazzy-rosbridge-suite
git clone --branch v3.2 https://github.com/erfrifrung/engr7001-perception.git "$HOME/engr7001-perception"
source /opt/ros/jazzy/setup.bash
cd "$HOME/engr7001-perception/versions/v3/ros2_ws"
colcon build --symlink-install
```

Unzip the V3.2 twin package into a separate directory. If you are using an unpatched copy of the matching twin instead, apply `integration/twin_bridge_v31.patch` from the root of that copy. Use your lab's robot connection settings in the local twin package.

## A. Calibrate the cameras to the board

Park the wrist and leave the board fixed once registration starts. From `versions/v3`, start each camera in its own terminal. Replace the serial placeholders locally.

```bash
bash scripts/start_camera.sh wrist '<wrist-serial>'
bash scripts/start_camera.sh tripod '<tripod-serial>'
bash scripts/start_camera.sh third '<third-serial>'
```

In a separate terminal under `versions/v3`, set `published_fit` to the saved orientation reference for the board fit currently in use. Check the board's actual square pitch before fitting. The lab board uses 20 mm.

```bash
source scripts/lab_env.sh
cal="$HOME/wp3_board_$(date +%Y%m%d_%H%M%S)"
published_fit='<published-fit.json>'
python3 scripts/register_board.py "$cal" "$published_fit" --pitch-mm 20
python3 scripts/check_calibration_drift.py "$published_fit" "$cal/current_board_tf.json"
```

Exit 0 means the existing calibration passes the drift check. Exit 1 means the fit drifted. If you are adopting the new fit or starting the first publisher, stop any previous board TF publisher and leave the following command running. Use the same value of `cal` in the later terminals.

```bash
python3 scripts/publish_board_tf.py "$cal/current_board_tf.json"
```

## B. Calibrate the board to the robot

Keep the board in place. With the supervisor present and the E-stop in reach, go to the twin's `ros` directory and run `bash start_real.sh`. In a second terminal in that directory, use `python3 teach_keyboard.py` at the lowest jog speed to touch board corners **0, 7, 40 and 47**. At each corner, press `v` and record the printed fingertip coordinates in the robot's `world` frame in metres. Write one numeric row per corner to `points.csv` in the order `corner_idx,world_x,world_y,world_z`, with no header.

```bash
python3 fit_table_tf.py points.csv --pitch-mm 20
```

The fit writes `ros/wp3_table_tf.yaml` when run from the twin's `ros` directory. Accept the fit when the RMS residual is below **3 mm** and the maximum residual is below **5 mm**. Stop `start_real.sh` after the touch-off. The twin refuses to start while the previous MoveIt instance is running.

## Run and check

From `versions/v3`, launch WP3 in the first terminal. Set `cal` to the calibration directory from step A.

```bash
bash scripts/launch_live.sh --policy merged_corroborated \
  --third-ns /perception/third --save-dir "$cal/live_v32"
```

From the twin's root in the second terminal, start the real-arm twin with the arm parked, workspace clear, supervisor present and E-stop in reach.

```bash
WP3_ROBOT_IP='<arm-ip>' bash twin.sh --wp3 --real
```

Set `WP3_ROBOT_IP` to the arm's IPv4 address in your local shell. The launcher asks you to type `YES` and defaults to 30 mm/s. Add `--rviz` if you need RViz. The twin starts the arm driver, MoveIt, the planning scene, the measured table frame, the patched bridge and the [GUI](http://localhost:8000). In live marker mode, the bridge reads `/wp3_live/markers` and `/wp3_live/health`. Check block placement in the GUI before issuing any motion commands.

```bash
source scripts/lab_env.sh
ros2 topic echo /wp3_live/health --once
ros2 topic hz /joint_states
tail -f ~/twin_logs/perception.log
```

## Common faults

- Live wrist observations require the real arm's `/joint_states`. The simulated joint states would place the wrist camera at the wrong pose.
- Both sides use `ROS_DOMAIN_ID=67` and localhost discovery in this setup. Source `scripts/lab_env.sh` in monitoring terminals.
- `--wp3` requires `ros/wp3_table_tf.yaml` in this setup. Keep `--manual-table` for simulation. Use the measured frame for real blocks.
- If the board moved, repeat the touch-off and restart `twin.sh` so it loads the updated table frame. With a changed camera calibration, the V3.2 node refreshes `/tf_static`, starts a new `calibration_epoch` and resets its tracks. The V3.2 bridge clears the obstacles from the old epoch.

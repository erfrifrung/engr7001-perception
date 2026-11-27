# Lab runbook: V3.1 + Hazem's twin v5 on the lab desktop (Thursday / M1)

Everything here runs on **one machine** (the lab desktop), ROS domain 67, localhost discovery: WP3 (`scripts/lab_env.sh`) and the twin (`twin.sh`) both default to that.

## 0. Install (once, about 10 min)

From the USB stick (`WP3_V3.1_lab_package/`):

```bash
# WP3 code: clone the bundle (keeps history and the v3.1-m1-candidate tag)
git clone -b v3.1 engr7001-perception_v3.1.bundle ~/engr7001-perception
cd ~/engr7001-perception/versions/v3
source /opt/ros/jazzy/setup.bash && (cd ros2_ws && colcon build --symlink-install)
source ros2_ws/install/setup.bash
python3 scripts/verify_zero_moveit.py
python3 -m pytest -q tests          # expect: all pass, 2 xfailed (documented V2 hazards)

# Twin: Hazem's digital_twin_v5 with the WP3 bridge patch applied (only ros/wp3_scene_bridge.py, ros/wp3_link.py change)
unzip digital_twin_v5_wp3_v31.zip -d ~/          # or apply WP3_BRIDGE_V31.patch to Hazem's own copy: git apply
```

## 1. Cameras (one terminal each)

```bash
cd ~/engr7001-perception/versions/v3
bash scripts/start_camera.sh wrist  <WRIST_SERIAL>
bash scripts/start_camera.sh tripod <TRIPOD_SERIAL>
bash scripts/start_camera.sh third  <THIRD_SERIAL>     # optional
```

Pre-checks (see `THURSDAY_PREDICTIONS.md` section 0):
- `ros2 topic hz /joint_states` must show the xArm driver publishing. Without it, the wrist counts as moving and the node degrades to `side_only`.
- `ros2 topic hz /perception/wrist/color/image_raw` should be ~15 Hz.

## 2. Calibration (task 3332) and table → robot (task 3331)

**Every session starts with a drift check** (V3.2; on 26 Nov every camera had moved 99–274 mm since the published calibration and nothing noticed):

```bash
source scripts/lab_env.sh
cal="$HOME/wp3_board_$(date +%Y%m%d_%H%M%S)"
python3 scripts/register_board.py "$cal" <orientation_reference.json> --pitch-mm <measured pitch>
python3 scripts/check_calibration_drift.py <fit being published>.json "$cal/current_board_tf.json"
#   exit 0 PASS: keep the running publisher.
#   exit 1 DRIFT: publish the fresh fit and redo the touch-off:
python3 scripts/publish_board_tf.py "$cal/current_board_tf.json"      # leave running (stop the old publisher)
```

The drift check prints each camera's move and the camera-to-camera changes: if every camera "moved" by the same amount but the pairwise changes are ~0, the **board** moved, not the rig.

Touch-off: record ≥ 4 board corners in `world` with the fingertip, then run Hazem's `ros/fit_table_tf.py points.csv` → `ros/wp3_table_tf.yaml`.

**Table frame.**
- *Unpatched v5:* `start_perception.sh` always used a manual guess (`0.30 0.00 -0.008`) and ignored the yaml.
- *Patched twin:* `start_perception.sh` / `twin.sh --wp3` use `ros/wp3_table_tf.yaml` whenever it exists. **Live WP3 refuses to start without it** unless you pass `--manual-table` (`twin.sh --wp3 --manual-table`), which prints a warning. Simulation/mock without the yaml prints `SIMULATION - table frame not calibrated`.

**After any recalibration** (a camera moved, the board re-registered):
- **WP3 V3.2** re-reads `/tf_static` every second (`static_tf_refresh_s`). A changed camera transform starts a new `calibration_epoch` (`<epoch>#r1`, `#r2`…), drops every track from the old calibration and re-detects. The node log says `calibration changed: … moved X mm / Y deg`.
- **Restart the bridge** anyway: Hazem's bridge still ignores `calibration_epoch`, so obstacles it already placed stay where the old calibration put them until WP3's next snapshot replaces them.
- WP3 V3.1 (`v3.1-m1-candidate-*`) must be restarted too: it cached the fixed transforms for its whole lifetime.

**While running**, V3.2 checks each camera's view of the table plane against its calibration once a second (`camera_health.<role>.calibration`):
- `ok`; `suspect` (> 2.5° or > 6 mm at the workspace centre): **still fused**, reported. The tripod on the 12 Nov bag is `suspect` (4.7° depth tilt from the oblique-camera depth bias, table +3 mm at the workspace).
- `mismatch` (> 6° or > 15 mm at the workspace centre, twice in a row): **not fused**. If no camera is left the scene `status` is `calibration_mismatch` (obstacles kept, no targets).
- The check cannot see a camera that only slid along the table or rotated about the vertical; the session-start drift check covers that.

## 3. WP3 live node (M1 default: V2 path)

```bash
source scripts/lab_env.sh
bash scripts/launch_live.sh --policy merged_corroborated --config <printed-block profile.json> --save-dir "$cal/live"
# first log line must read: enable_v3_regions=False ... require_fk_for_moving_wrist=...
ros2 topic echo /wp3_live/health --once      # status ok, dwell_reason not fk_missing_or_moving
```

To evaluate V3 live (not the M1 default; based on the 26 Nov Ryzen budget sweep, use `enumerate` at `45 ms` or `prove` at `120 ms` — `prove` at `45 ms` times out on 150/168 regions):
```bash
bash scripts/launch_live.sh --policy merged_corroborated --third-ns /perception/third --v3 on --v3-solver enumerate --v3-budget-ms 45 --save-dir "$cal/live_v3"
```
Useful V3.2 runtime switches on `launch_live.sh` (all default to safe V3.2 settings; no rebuild required): `--calibration-check on|off`, `--static-tf-refresh-s SEC`, `--retire-on-observed-free on|off`, `--merge-overlapping on|off`, `--publish-overlay on|off`, `--event-wakeup on|off`.

## 4. Twin with live WP3 (Hazem)

```bash
cd ~/digital_twin_v5_wp3_v31
bash twin.sh --wp3          # simulated arm + scene + live WP3 perception + GUI (needs ros/wp3_table_tf.yaml)
bash twin.sh --wp3 --manual-table   # only if the touch-off hasn't been done yet: positions NOT calibrated
# real arm (supervised only): bash twin.sh --real --wp3
```

Before the real session, the bridge can be checked against fake-hardware MoveIt on any Jazzy machine with the xArm workspace. Allow about 4 minutes; it never touches the real arm:

```bash
TWIN=~/digital_twin_v5 bash ~/engr7001-perception/integration/live_test/run_twin_live_test.sh   # TWIN = UNPATCHED v5
```

Expect the original v5 bridge to fail checks 3, 5b and 6b (phantom/duplicate obstacles) and the patched one to pass 13/13, plus 3/3 JSON checks and 2/2 checks with Hazem's mock publisher.

Check in the twin GUI (or RViz with `--rviz`):
- every block WP3 shows appears once;
- moving a block moves it;
- removing a block removes it within ~3 s (track retirement);
- a block picked and dropped by the twin is not duplicated.

## 5. Replay benchmark (background, any time)

```bash
cd ~/engr7001-perception/versions/v3
CONFIG=ros2_ws/src/wp3_live_perception/config/historical_props_12nov.json \
  bash scripts/ryzen_replay_bench.sh <bag dirs ...>
```

## 6. Evidence to save for Friday

- calibration directory (overlays, residual table);
- `wp3_table_tf.yaml` + fit RMS;
- `live/` (JSONL with stamps);
- `ros2 bag record /wp3_live/scene_observation /wp3_live/health /tf /tf_static` for 2 min;
- twin GUI or RViz screenshot;
- `results/ryzen_*`.

Fill in the Measured column of `THURSDAY_PREDICTIONS.md` the same day.

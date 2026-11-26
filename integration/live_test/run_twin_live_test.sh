#!/usr/bin/env bash
# Live A/B test of the WP3 -> MoveIt bridge against Hazem's twin (fake-hardware MoveIt; never the real arm).
#
#   TWIN=~/digital_twin_v5 bash integration/live_test/run_twin_live_test.sh            # bridge scenarios only
#   TWIN=... BAG=<rosbag dir> WP3=<repo>/versions/v3 bash integration/live_test/run_twin_live_test.sh
#
# TWIN    : an UNPATCHED digital_twin_v5 directory (the script copies it twice: original and patched)
# XARM_WS : xArm ROS 2 workspace (default ~/xarm_ws, falls back to ~/xarm6_ws)
# BAG     : optional recorded bag for the full pipeline (bag -> wp3_live_node -> markers -> bridge -> MoveIt)
# WP3     : versions/v3 checkout with ros2_ws built (needed with BAG)
# Results: PASS/FAIL per check on stdout; logs in $OUT (default /tmp/wp3_twin_live_test).
set -o pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTEG="$(cd "$HERE/.." && pwd)"
: "${TWIN:?set TWIN to an unpatched digital_twin_v5 directory}"
XARM_WS=${XARM_WS:-$HOME/xarm_ws}; [ -d "$XARM_WS/install" ] || XARM_WS=$HOME/xarm6_ws
OUT=${OUT:-/tmp/wp3_twin_live_test}; T=$OUT/twins
rm -rf "$OUT"; mkdir -p "$T"
cp -r "$TWIN" "$T/orig"; cp -r "$TWIN" "$T/patched"
(cd "$T/patched" && git apply "$INTEG/twin_bridge_v31.patch") || { echo "patch does not apply to $TWIN"; exit 1; }
source /opt/ros/jazzy/setup.bash; source "$XARM_WS/install/setup.bash"
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID_TEST:-$((120 + RANDOM % 60))} ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
echo "domain $ROS_DOMAIN_ID (isolated from the lab's 67)"
pids=()
stop_all() { kill "${pids[@]}" 2>/dev/null; sleep 1; }
trap stop_all EXIT
ros2 launch xarm_moveit_config xarm6_moveit_fake.launch.py add_gripper:=true show_rviz:=false > "$OUT/arm.log" 2>&1 & pids+=($!)
ros2 daemon stop >/dev/null 2>&1
ready=0
for i in $(seq 1 45); do
  sleep 2; out=$(timeout 15 ros2 service list --no-daemon 2>&1)
  grep -qx "/get_planning_scene" <<< "$out" && { ready=1; break; }
done
[ $ready = 1 ] || { echo "MoveIt did not come up: $OUT/arm.log"; exit 1; }
sleep 4
cd "$T/patched/ros"
echo YES | timeout 90 python3 go_home.py > "$OUT/go_home.log" 2>&1; echo "go_home exit=$?"
timeout 60 python3 build_scene.py > "$OUT/scene.log" 2>&1; echo "build_scene exit=$?"
python3 wp3_table_tf.py --xyz 0.30 0.00 -0.008 --yaw 0 > "$OUT/tf.log" 2>&1 & pids+=($!)
sleep 2
for v in orig patched; do
  echo "== marker scenario, $v bridge"
  (cd "$T/$v/ros" && exec python3 wp3_scene_bridge.py --remove-demo-block) > "$OUT/bridge_markers_$v.log" 2>&1 & BP=$!
  sleep 3; python3 "$HERE/twin_driver.py" markers "$OUT/markers_$v.json"; kill $BP; wait $BP 2>/dev/null
done
echo "== JSON scenario, patched bridge"
(cd "$T/patched/ros" && exec python3 wp3_scene_bridge.py --use-scene-json) > "$OUT/bridge_json.log" 2>&1 & BP=$!
sleep 3; python3 "$HERE/twin_driver.py" json "$OUT/json_patched.json"; kill $BP; wait $BP 2>/dev/null
if [ -n "${BAG:-}" ] && [ -n "${WP3:-}" ]; then
  source "$WP3/ros2_ws/install/setup.bash"; export PYTHONPATH="$WP3:${PYTHONPATH:-}"
  TOPICS="/tf_static /perception/wrist/color/image_raw /perception/wrist/aligned_depth_to_color/image_raw /perception/wrist/color/camera_info /perception/tripod/color/image_raw /perception/tripod/aligned_depth_to_color/image_raw /perception/tripod/color/camera_info"
  echo "== full pipeline (patched bridge): $BAG"
  ros2 run wp3_live_perception wp3_live_node --ros-args -p use_sim_time:=true \
    -p config:="$WP3/ros2_ws/src/wp3_live_perception/config/historical_props_12nov.json" \
    -p policy:=whole_partition_mv -p publish_overlay:=false > "$OUT/live_node.log" 2>&1 & LP=$!
  (cd "$T/patched/ros" && exec python3 wp3_scene_bridge.py --remove-demo-block) > "$OUT/bridge_pipeline.log" 2>&1 & BP=$!
  sleep 4
  timeout 40 ros2 bag play "$BAG" --clock --topics $TOPICS > "$OUT/bag.log" 2>&1 & PP=$!
  python3 "$HERE/twin_driver.py" watch "$OUT/pipeline.json" 35
  kill $PP $BP $LP 2>/dev/null; wait $PP $BP $LP 2>/dev/null
  echo "  bridge: $(grep -c ' add ' "$OUT/bridge_pipeline.log") adds, $(grep -c ' remove ' "$OUT/bridge_pipeline.log") removes"
fi
echo "logs: $OUT"

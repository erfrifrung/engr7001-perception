#!/usr/bin/env bash
# Evidence: the same recorded bag through bag -> wp3_live_node -> markers -> patched twin bridge -> MoveIt,
# once per perception mode (V2 path, V3 prove @120 ms, V3 prove @45 ms). Fake-hardware MoveIt only.
#
#   TWIN=<unpatched digital_twin_v5> BAG=<bag dir> bash scripts/evidence_pipeline_ab.sh
# Env: XARM_WS (~/xarm_ws or ~/xarm6_ws), OUT (/tmp/wp3_pipeline_ab), CONFIG (wooden-prop profile by default).
# The bag has no ground truth: this measures published vs ambiguous, stability, processing delay, stage timings
# and planning-scene behaviour, NOT accuracy or coverage.
set -o pipefail
V3="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INTEG="$(cd "$V3/../../integration" && pwd)"
: "${TWIN:?set TWIN}"; : "${BAG:?set BAG}"
XARM_WS=${XARM_WS:-$HOME/xarm_ws}; [ -d "$XARM_WS/install" ] || XARM_WS=$HOME/xarm6_ws
OUT=${OUT:-/tmp/wp3_pipeline_ab}; rm -rf "$OUT"; mkdir -p "$OUT"
CONFIG=${CONFIG:-$V3/ros2_ws/src/wp3_live_perception/config/historical_props_12nov.json}
cp -r "$TWIN" "$OUT/twin"; (cd "$OUT/twin" && git apply "$INTEG/twin_bridge_v31.patch") || exit 1
source /opt/ros/jazzy/setup.bash; source "$XARM_WS/install/setup.bash"; source "$V3/ros2_ws/install/setup.bash"
export PYTHONPATH="$V3:${PYTHONPATH:-}"
NODE_EXE="$V3/ros2_ws/install/wp3_live_perception/lib/wp3_live_perception/wp3_live_node"
[ -x "$NODE_EXE" ] || { echo "build ros2_ws first ($NODE_EXE missing)"; exit 1; }
if pgrep -f "lib/wp3_live_perception/wp3_live_node" >/dev/null; then echo "a wp3_live_node is already running: stop it first"; exit 1; fi
export ROS_DOMAIN_ID=$((120 + RANDOM % 60)) ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
pids=(); trap 'kill "${pids[@]}" 2>/dev/null; sleep 1' EXIT
ros2 launch xarm_moveit_config xarm6_moveit_fake.launch.py add_gripper:=true show_rviz:=false > "$OUT/arm.log" 2>&1 & pids+=($!)
ros2 daemon stop >/dev/null 2>&1
for i in $(seq 1 45); do sleep 2; out=$(timeout 15 ros2 service list --no-daemon 2>&1); grep -qx /get_planning_scene <<< "$out" && break; done
sleep 4
cd "$OUT/twin/ros"
echo YES | timeout 90 python3 go_home.py > "$OUT/go_home.log" 2>&1; timeout 60 python3 build_scene.py > "$OUT/scene.log" 2>&1
python3 wp3_table_tf.py --xyz 0.30 0.00 -0.008 --yaw 0 > "$OUT/tf.log" 2>&1 & pids+=($!)
TOPICS="/tf_static /perception/wrist/color/image_raw /perception/wrist/aligned_depth_to_color/image_raw /perception/wrist/color/camera_info /perception/tripod/color/image_raw /perception/tripod/aligned_depth_to_color/image_raw /perception/tripod/color/camera_info"
declare -A MODE_ARGS=(
  [v2]="-p enable_v3_regions:=false"
  [v3prove120]="-p enable_v3_regions:=true -p v3_solver:=prove -p v3_frame_budget_ms:=120.0"
  [v3prove45]="-p enable_v3_regions:=true -p v3_solver:=prove -p v3_frame_budget_ms:=45.0"
  [v2b]="-p enable_v3_regions:=false"
  [v3prove120b]="-p enable_v3_regions:=true -p v3_solver:=prove -p v3_frame_budget_ms:=120.0"
)
# Pre-read the bag so the first mode does not pay the disk / page-cache warm-up (seen: 950 ms -> 250 ms per
# frame for the same work when the bag was read from /mnt/c cold). Copy bags to a native filesystem for timing.
cat "$BAG"/*.db3 > /dev/null
MODES=${MODES:-v2 v3prove120 v3prove45}
for mode in $MODES; do
  echo "== $mode"
  python3 "$INTEG/live_test/twin_driver.py" watch /dev/null 1 > /dev/null 2>&1
  python3 - <<'PY'
import rclpy
from moveit_msgs.msg import CollisionObject, PlanningScene, PlanningSceneComponents
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
rclpy.init(); n = rclpy.create_node("wp3_ab_reset")
g = n.create_client(GetPlanningScene, "/get_planning_scene"); a = n.create_client(ApplyPlanningScene, "/apply_planning_scene")
g.wait_for_service(timeout_sec=10); a.wait_for_service(timeout_sec=10)
r = GetPlanningScene.Request(); r.components.components = PlanningSceneComponents.WORLD_OBJECT_NAMES
f = g.call_async(r); rclpy.spin_until_future_complete(n, f, timeout_sec=5)
s = PlanningScene(is_diff=True)
for o in f.result().scene.world.collision_objects:
    if o.id.startswith("wp3_"):
        s.world.collision_objects.append(CollisionObject(id=o.id, operation=CollisionObject.REMOVE))
f = a.call_async(ApplyPlanningScene.Request(scene=s)); rclpy.spin_until_future_complete(n, f, timeout_sec=5)
n.destroy_node(); rclpy.shutdown()
PY
  # Run the node executable directly: killing a `ros2 run` wrapper leaves the node alive (measured), which made
  # earlier modes keep publishing during later ones.
  "$NODE_EXE" --ros-args -r __node:=wp3_live_node_$mode -p use_sim_time:=true -p config:="$CONFIG" \
      -p policy:=whole_partition_mv -p publish_overlay:=false ${MODE_ARGS[$mode]} > "$OUT/node_$mode.log" 2>&1 & LP=$!
  (cd "$OUT/twin/ros" && exec python3 wp3_scene_bridge.py --remove-demo-block) > "$OUT/bridge_$mode.log" 2>&1 & BP=$!
  sleep 4
  n_nodes=$(pgrep -fc "lib/wp3_live_perception/wp3_live_node")
  echo "  wp3_live_node processes running: $n_nodes (must be 1)"
  [ "$n_nodes" = 1 ] || { echo "  ABORT: stale perception nodes running"; exit 1; }
  python3 "$V3/scripts/scene_recorder.py" "$OUT/scenes_$mode.jsonl" 34 > "$OUT/rec_$mode.log" 2>&1 & RP=$!
  python3 "$INTEG/live_test/twin_driver.py" watch "$OUT/scene_$mode.json" 34 > /dev/null 2>&1 & WP=$!
  sleep 1
  timeout 40 ros2 bag play "$BAG" --clock --topics $TOPICS > "$OUT/bag_$mode.log" 2>&1
  wait $RP $WP
  kill $BP $LP 2>/dev/null; wait $BP $LP 2>/dev/null
  sleep 1
  grep -m1 "config:" "$OUT/node_$mode.log" | sed 's/^.*config:/  node config:/'
  cat "$OUT/rec_$mode.log"
done
python3 "$V3/scripts/evidence_pipeline_summary.py" "$OUT" $MODES | tee "$OUT/summary.md"

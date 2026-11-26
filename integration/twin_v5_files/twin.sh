#!/usr/bin/env bash
# ONE command for the whole digital twin, with the GUI as the only window (no RViz):
#
#   bash twin.sh                 simulation, 3 mock blocks, top speed 50 mm/s
#   bash twin.sh --speed 0.08    a different top speed (m/s, at most 0.15)
#   bash twin.sh --no-mock       simulation without the mock blocks
#   bash twin.sh --wp3           blocks from the live WP3 perception node (needs ros/wp3_table_tf.yaml)
#   bash twin.sh --wp3 --manual-table   live WP3 with the manual, UNCALIBRATED table guess (explicit opt-in)
#   WP3_ROBOT_IP must be set for the real arm.
#   bash twin.sh --real          the REAL arm (asks YES first; top speed 30 mm/s unless --speed is given)
#   bash twin.sh --rviz          also show RViz (for debugging)
#
# It starts, in order: the arm (simulated or real) with MoveIt, the scene, the Etee bridge, the hold node,
# perception, rosbridge and the GUI server, then opens http://localhost:8000 in Windows.
# Every part writes a log in ~/twin_logs/. Ctrl+C stops everything.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS="$HERE/ros"
LOGS=~/twin_logs
REAL=0; SPEED=""; MOCK=1; WP3=0; RVIZ=0; MANUAL_TABLE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --real) REAL=1 ;;
    --speed) SPEED="$2"; shift ;;
    --no-mock) MOCK=0 ;;
    --wp3) WP3=1; MOCK=0 ;;
    --manual-table) MANUAL_TABLE=1 ;;
    --rviz) RVIZ=1 ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "unknown option $1 (try --help)"; exit 1 ;;
  esac
  shift
done
[ -z "$SPEED" ] && { [ $REAL = 1 ] && SPEED=0.03 || SPEED=0.05; }

source /opt/ros/jazzy/setup.bash
source ~/xarm_ws/install/setup.bash
export ROS_DOMAIN_ID=67 ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
mkdir -p "$LOGS"
say() { printf '\033[1;36m[twin]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[twin] %s\033[0m\n' "$*"; exit 1; }

# ---------------------------------------------------------------- checks
if [ $WP3 = 1 ] && [ ! -f "$ROS/wp3_table_tf.yaml" ] && [ $MANUAL_TABLE = 0 ]; then
  die "--wp3 needs the measured table frame ros/wp3_table_tf.yaml (fit_table_tf.py after the touch-off). Add --manual-table to use the uncalibrated guess."
fi
ros2 node list 2>/dev/null | grep -q move_group && die "MoveIt is already running (an old start_sim.sh / start_real.sh?). Stop it first."
for port in 9090 8000; do
  ss -ltn 2>/dev/null | grep -q ":$port " && die "Port $port is in use (an old rosbridge or GUI server?). Stop it first."
done
ros2 pkg prefix rosbridge_server >/dev/null 2>&1 || die "rosbridge is missing: sudo apt install -y ros-jazzy-rosbridge-suite"
if [ $REAL = 1 ]; then
  arm_ip="${WP3_ROBOT_IP:-}"
  [[ "$arm_ip" =~ ^[0-9]{1,3}(\.[0-9]{1,3}){3}$ ]] || die "Set WP3_ROBOT_IP to the arm IPv4 address."
  ping -c 1 -W 2 "$arm_ip" >/dev/null 2>&1 || die "Cannot reach the arm. Check the cable, power, and local network settings."
  echo "REAL ARM at up to $(python3 -c "print(round($SPEED*1000))") mm/s. Supervisor present, E-stop in reach, workspace clear?"
  read -r -p "Type YES to start: " ok; [ "$ok" = "YES" ] || exit 0
fi

# everything started below belongs to this script's process group: Ctrl+C stops all of it
trap 'trap - EXIT INT TERM; say "stopping everything ..."; kill 0 2>/dev/null; wait 2>/dev/null; say "stopped"' EXIT INT TERM

# ---------------------------------------------------------------- 1. arm + MoveIt, RViz hidden
if [ $REAL = 1 ]; then
  LAUNCH=(xarm_moveit_config xarm6_moveit_realmove.launch.py robot_ip:="$arm_ip" add_gripper:=true)
else
  LAUNCH=(xarm_moveit_config xarm6_moveit_fake.launch.py add_gripper:=true)
fi
RVIZ_ARG=""
if [ $RVIZ = 0 ]; then
  # the xArm launch files read "show_rviz" inside an included file, so it is not in --show-args: look for it there
  SHARE=$(ros2 pkg prefix xarm_moveit_config 2>/dev/null)/share/xarm_moveit_config/launch
  if grep -rqs "show_rviz" "$SHARE"; then
    RVIZ_ARG="show_rviz:=false"
  else
    say "this xArm version has no switch to hide RViz: it will open, just minimise it"
  fi
fi
say "starting the $([ $REAL = 1 ] && echo REAL || echo simulated) arm and MoveIt ${RVIZ_ARG:+($RVIZ_ARG)}"
ros2 launch "${LAUNCH[@]}" $RVIZ_ARG > "$LOGS/arm.log" 2>&1 &
ARM_PID=$!
ready=0
for i in $(seq 1 90); do
  kill -0 $ARM_PID 2>/dev/null || die "the arm launch stopped: see $LOGS/arm.log"
  ros2 service list 2>/dev/null | grep -q "^/get_planning_scene$" && ros2 topic list 2>/dev/null | grep -q "^/joint_states$" && { ready=1; break; }
  sleep 1
done
[ $ready = 1 ] || die "MoveIt did not come up in 90 s: see $LOGS/arm.log"
sleep 4
cd "$ROS"
if [ $REAL = 0 ]; then
  say "moving the simulated arm to the real home pose"
  echo YES | timeout 60 python3 go_home.py > "$LOGS/go_home.log" 2>&1 || die "moving home failed or took over 60 s: see $LOGS/go_home.log and $LOGS/arm.log"
fi
say "building the scene (table, base plate, fence, power box)"
kill -0 $ARM_PID 2>/dev/null || die "the arm launch stopped: see $LOGS/arm.log"
timeout 60 python3 build_scene.py > "$LOGS/scene.log" 2>&1 || die "build_scene failed: see $LOGS/scene.log"

# ---------------------------------------------------------------- 2. inputs, hold node, perception
say "Etee bridge (port 5005) for the Windows reader"
python3 etee_bridge.py > "$LOGS/etee_bridge.log" 2>&1 &
say "hold node, top speed $(python3 -c "print(round($SPEED*1000))") mm/s"
echo YES | python3 etee_hold.py --speed "$SPEED" > "$LOGS/etee_hold.log" 2>&1 &
HOLD_PID=$!
if [ $MOCK = 1 ]; then
  say "perception: 3 MOCK blocks"
  bash start_perception.sh > "$LOGS/perception.log" 2>&1 &
elif [ $WP3 = 1 ]; then
  say "perception: live WP3 node"
  bash start_perception.sh --live $([ $MANUAL_TABLE = 1 ] && echo --manual-table) > "$LOGS/perception.log" 2>&1 &
fi

# ---------------------------------------------------------------- 3. GUI
say "rosbridge (port 9090) and the GUI server (port 8000)"
ros2 launch rosbridge_server rosbridge_websocket_launch.xml > "$LOGS/rosbridge.log" 2>&1 &
cd "$HERE/gui"
python3 twin_gui_server.py > "$LOGS/gui.log" 2>&1 &
for i in $(seq 1 20); do curl -s -o /dev/null http://localhost:8000/ && break; sleep 0.5; done
sleep 2
kill -0 $HOLD_PID 2>/dev/null || die "the hold node stopped: see $LOGS/etee_hold.log"
say "ready: http://localhost:8000"
command -v cmd.exe >/dev/null 2>&1 && cmd.exe /c start http://localhost:8000 >/dev/null 2>&1
say "logs: $LOGS   (live hold-node output: tail -f $LOGS/etee_hold.log)"
say "Ctrl+C here stops everything."
while kill -0 $ARM_PID 2>/dev/null; do sleep 2; done
die "the arm launch stopped: see $LOGS/arm.log"

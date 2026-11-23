#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    cat <<'HELP'
Usage: bash versions/v1/scripts/launch_live.sh [options]

Launch the live WP3 perception display. Start your camera publishers first.

Options:
  --policy NAME     Policy (default: merged_corroborated; also merged_wrist_rgb, depth_only, astra_permissive, generic_watershed)
  --config PATH     Config JSON (default: printed_jenga_live.json)
  --save-dir PATH   Directory to save live overlays, JSONL predictions & timings
  --wrist-ns NS     Wrist camera namespace (default: /perception/wrist)
  --side-ns NS      Fixed side camera namespace (default: /perception/tripod)
  --rate-hz HZ      Loop rate Hz (default: 15.0)
Stop with Ctrl+C (SIGINT); Wait 1 s for JSONL flush.
HELP
    exit 0
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
IMPL_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
WS_SETUP="$IMPL_ROOT/ros2_ws/install/setup.bash"
POLICY="merged_corroborated"
CONFIG="$IMPL_ROOT/ros2_ws/src/wp3_live_perception/config/printed_jenga_live.json"
SAVE_DIR="$IMPL_ROOT/results/live_$(date +%Y%m%d_%H%M%S)"
WRIST_NS="/perception/wrist"
SIDE_NS="/perception/tripod"
RATE_HZ="15.0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --policy) POLICY="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --save-dir) SAVE_DIR="$2"; shift 2 ;;
        --wrist-ns) WRIST_NS="$2"; shift 2 ;;
        --side-ns) SIDE_NS="$2"; shift 2 ;;
        --rate-hz) RATE_HZ="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-67}"
export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-LOCALHOST}"
set +u
source /opt/ros/jazzy/setup.bash
if [[ ! -f "$WS_SETUP" ]]; then
    echo "Building isolated overlay first..." >&2
    (cd "$IMPL_ROOT/ros2_ws" && colcon build --symlink-install)
fi
source "$WS_SETUP"
set -u
export PYTHONPATH="$IMPL_ROOT:${PYTHONPATH:-}"
mkdir -p "$SAVE_DIR"

echo "Launching wp3_live_node (policy=$POLICY, config=$CONFIG, save_dir=$SAVE_DIR)"
echo "Stop cleanly with Ctrl+C."
ros2 launch wp3_live_perception wp3_live.launch.py \
    policy:="$POLICY" \
    config:="$CONFIG" \
    save_dir:="$SAVE_DIR" \
    wrist_ns:="$WRIST_NS" \
    side_ns:="$SIDE_NS" \
    rate_hz:="$RATE_HZ" \
    2>&1 | tee "$SAVE_DIR/live_node.log"

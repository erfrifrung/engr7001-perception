#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    cat <<'HELP'
Usage: bash WP3-Live-Implementation-v2/scripts/launch_live.sh [options]

THE ONE COMMAND to launch the chosen live WP3 display-only perception method.
Requires existing camera publishers (does NOT start or stop cameras or MoveIt).

Options:
  --policy NAME     Policy (default: merged_corroborated; also whole_partition_mv, merged_wrist_rgb, depth_only, astra_permissive, generic_watershed)
  --config PATH     Config JSON (default: printed_jenga_live.json)
  --save-dir PATH   Directory to save live overlays, JSONL predictions & timings
  --wrist-ns NS     Wrist camera namespace (default: /perception/wrist)
  --side-ns NS      Fixed side camera namespace (default: /perception/tripod)
  --third-ns NS     Additional qualified fixed view for seam checks (e.g. /perception/third)
  --rate-hz HZ      Loop rate Hz (default: 15.0)
  --v3 on|off       V3 region stage (default: off = tuned V2 path, the M1 live default)
  --v3-solver S     prove | enumerate (only with --v3 on; default: config value)
  --v3-budget-ms MS V3 per-frame budget (default: config value)
  --temporal-skip on|off  Re-publish cached geometry on stationary frames (default: off)
  --publish-overlay on|off     Publish/save overlay images (default: on)
  --calibration-check on|off   V3.2 per-camera table-plane check (default: on)
  --static-tf-refresh-s SEC    V3.2 /tf_static refresh period s, 0=lifetime cache (default: 1.0)
  --retire-on-observed-free on|off  V3.2 retire unseen blocks only on empty heightmap (default: config)
  --merge-overlapping on|off   V3.2 merge same-layer duplicate tracks (default: config)
  --event-wakeup on|off        Trigger compute immediately on synced frame-pair arrival (default: on)
Stop with Ctrl+C (SIGINT); Wait 1 s for JSONL flush.
HELP
    exit 0
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export WP3_CV2_THREADS=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
IMPL_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
WS_SETUP="$IMPL_ROOT/ros2_ws/install/setup.bash"
POLICY="merged_corroborated"
CONFIG="$IMPL_ROOT/ros2_ws/src/wp3_live_perception/config/printed_jenga_live.json"
SAVE_DIR="$IMPL_ROOT/results/live_$(date +%Y%m%d_%H%M%S)"
WRIST_NS="/perception/wrist"
SIDE_NS="/perception/tripod"
THIRD_NS=""
RATE_HZ="15.0"
V3="false"
V3_SOLVER=""
V3_BUDGET="-1.0"
TSKIP="false"
PUB_OVERLAY="true"
CAL_CHECK="true"
TF_REFRESH_S="1.0"
RETIRE_FREE=""
MERGE_OVLP=""
EVENT_WAKEUP="true"

onoff() { case "$1" in on|true|1) echo true ;; off|false|0) echo false ;; *) echo "expected on|off, got $1" >&2; exit 2 ;; esac; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --policy) POLICY="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --save-dir) SAVE_DIR="$2"; shift 2 ;;
        --wrist-ns) WRIST_NS="$2"; shift 2 ;;
        --side-ns) SIDE_NS="$2"; shift 2 ;;
        --third-ns) THIRD_NS="$2"; shift 2 ;;
        --rate-hz) RATE_HZ="$2"; shift 2 ;;
        --v3) V3="$(onoff "$2")"; shift 2 ;;
        --v3-solver) V3_SOLVER="$2"; shift 2 ;;
        --v3-budget-ms) V3_BUDGET="$2"; shift 2 ;;
        --temporal-skip) TSKIP="$(onoff "$2")"; shift 2 ;;
        --publish-overlay) PUB_OVERLAY="$(onoff "$2")"; shift 2 ;;
        --calibration-check) CAL_CHECK="$(onoff "$2")"; shift 2 ;;
        --static-tf-refresh-s) TF_REFRESH_S="$2"; shift 2 ;;
        --retire-on-observed-free) RETIRE_FREE="$(onoff "$2")"; shift 2 ;;
        --merge-overlapping) MERGE_OVLP="$(onoff "$2")"; shift 2 ;;
        --event-wakeup) EVENT_WAKEUP="$(onoff "$2")"; shift 2 ;;
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

echo "Launching wp3_live_node (policy=$POLICY, config=$CONFIG, v3=$V3, temporal_skip=$TSKIP, cal_check=$CAL_CHECK, wakeup=$EVENT_WAKEUP, save_dir=$SAVE_DIR)"
echo "Stop cleanly with Ctrl+C."
ros2 launch wp3_live_perception wp3_live.launch.py \
    policy:="$POLICY" \
    config:="$CONFIG" \
    save_dir:="$SAVE_DIR" \
    wrist_ns:="$WRIST_NS" \
    side_ns:="$SIDE_NS" \
    third_ns:="$THIRD_NS" \
    rate_hz:="$RATE_HZ" \
    enable_v3_regions:="$V3" \
    v3_solver:="$V3_SOLVER" \
    v3_frame_budget_ms:="$V3_BUDGET" \
    enable_temporal_skip:="$TSKIP" \
    publish_overlay:="$PUB_OVERLAY" \
    calibration_check:="$CAL_CHECK" \
    static_tf_refresh_s:="$TF_REFRESH_S" \
    tracker_merge_overlapping:="$MERGE_OVLP" \
    retire_on_observed_free:="$RETIRE_FREE" \
    event_driven_wakeup:="$EVENT_WAKEUP" \
    2>&1 | tee "$SAVE_DIR/live_node.log"

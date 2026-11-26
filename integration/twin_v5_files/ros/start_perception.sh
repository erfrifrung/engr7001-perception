#!/usr/bin/env bash
# v4: perception into the twin in ONE window: table frame + WP3 scene bridge (+ mock blocks unless --live).
#   bash start_perception.sh                         simulation: 3 fake (mock) blocks, clearly labelled as mock
#   bash start_perception.sh --wood                  mock with the 45 mm wooden props
#   bash start_perception.sh --live                  real WP3 node publishing /wp3_live/markers (no mock)
#   bash start_perception.sh --live --manual-table   real WP3 node with the MANUAL table guess (not calibrated)
# Table frame (WP3 V3.1 change):
#   - wp3_table_tf.yaml (written by fit_table_tf.py after the touch-off) is used whenever it exists.
#   - --live without the yaml refuses to start unless --manual-table is given: real blocks placed in an
#     unmeasured table frame look plausible in the twin but can be centimetres off.
#   - simulation / mock without the yaml uses TABLE_XYZ / TABLE_YAW and says so loudly.
TABLE_XYZ="${TABLE_XYZ:-0.30 0.00 -0.008}"
TABLE_YAW="${TABLE_YAW:-0}"
cd "$(dirname "$0")"
LIVE=0; MANUAL=0; MOCK_ARGS=()
for a in "$@"; do
  case "$a" in
    --live) LIVE=1 ;;
    --manual-table) MANUAL=1 ;;
    *) MOCK_ARGS+=("$a") ;;
  esac
done
if [[ -f wp3_table_tf.yaml && $MANUAL == 0 ]]; then
  TF_ARGS=(--file wp3_table_tf.yaml)
  echo "table frame: MEASURED (wp3_table_tf.yaml)"
elif [[ $LIVE == 1 && $MANUAL == 0 ]]; then
  echo "ERROR: live WP3 needs the measured table frame (wp3_table_tf.yaml from fit_table_tf.py)." >&2
  echo "       Run the touch-off fit first, or pass --manual-table to use the uncalibrated guess ($TABLE_XYZ, ${TABLE_YAW} deg)." >&2
  exit 2
else
  TF_ARGS=(--xyz $TABLE_XYZ --yaw $TABLE_YAW)
  if [[ $LIVE == 1 ]]; then
    echo "WARNING: LIVE WP3 with a MANUAL, UNCALIBRATED table frame ($TABLE_XYZ, ${TABLE_YAW} deg). Do not trust block positions."
  else
    echo "SIMULATION - table frame not calibrated (manual $TABLE_XYZ, ${TABLE_YAW} deg)"
  fi
fi
pids=()
cleanup() { kill "${pids[@]}" 2>/dev/null; wait 2>/dev/null; echo "perception stopped"; }
trap cleanup EXIT INT TERM
python3 wp3_table_tf.py "${TF_ARGS[@]}" & pids+=($!)
sleep 1
python3 wp3_scene_bridge.py --remove-demo-block & pids+=($!)
if [[ $LIVE == 0 ]]; then
  sleep 2
  python3 mock_wp3_publisher.py "${MOCK_ARGS[@]}" & pids+=($!)
fi
echo "perception running (Ctrl+C stops all of it)"
wait

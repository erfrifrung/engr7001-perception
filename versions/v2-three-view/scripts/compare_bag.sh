#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h || $# -lt 2 ]]; then
    cat <<'HELP'
Usage: bash versions/v2-three-view/scripts/compare_bag.sh BAG_DB3 --out OUT_DIR [compare_bag.py options...]

Compare WP3 perception policies on the same frames and calibration from a
SQLite rosbag2 recording.
HELP
    [[ ${1:-} == --help || ${1:-} == -h ]] && exit 0 || exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
IMPL_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export WP3_CV2_THREADS=1
export PYTHONDONTWRITEBYTECODE=1

set +u
source /opt/ros/jazzy/setup.bash 2>/dev/null || true
set -u
exec python3 "$IMPL_ROOT/scripts/compare_bag.py" "$@"

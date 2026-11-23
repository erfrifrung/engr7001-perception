#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == --help || $# -lt 2 ]]; then
    cat <<'HELP'
Usage: bash WP3-Live-Implementation-v2/scripts/compare_bag.sh BAG_DB3 --out OUT_DIR [compare_bag.py options...]

THE ONE COMMAND to benchmark all WP3 perception policies & historical v2 controls
on identical scheduled epochs and calibration from one SQLite3 bag.
HELP
    [[ ${1:-} == --help ]] && exit 0 || exit 2
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

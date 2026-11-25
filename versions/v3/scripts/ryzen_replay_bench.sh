#!/usr/bin/env bash
# WP3 V3.1 replay benchmark for the lab Ryzen 5 5600 (CPU path). Offline: no robot, no cameras, no ROS graph.
#
#   bash scripts/ryzen_replay_bench.sh [BAG_DIR_OR_DB3 ...]
#
# Run from versions/v3. Env: PY (python3), PIN ("taskset -c 2"; "" disables), ROUNDS (3), BUDGETS ("45 120"),
# SKIP_SYNTH=1 (bags only), MAX_FRAMES (cap per bag, for a smoke test), OUT (results/ryzen_<host>_<time>),
# CONFIG (replay profile; for the 45 mm wooden-prop bags use
#   ros2_ws/src/wp3_live_perception/config/historical_props_12nov.json, default is printed_jenga_live.json).
# Writes everything under OUT; prints the tables.
# What it measures (label every number with this host, never mix with M4 / WSL numbers):
#   1. host + provenance (CPU model, governor, SMT, Python/NumPy/OpenCV, git revision, config hash)
#   2. synthetic held-out split (168 scenes): tuned V2 vs V3.1 prove / adaptive / enumerate, pinned to one core,
#      rotated method order per round, warm-up excluded; coverage + wrong poses + p50/p95/p99 + budget exits
#   3. each bag given: compare_bag.py with V3 off (V2 path) vs V3.1 adaptive / prove at each budget, with
#      /usr/bin/time -v (CPU %, max RSS). Bags are wooden 45 mm props unless stated: timing and false-split
#      evidence only, not printed-block accuracy.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
PIN=${PIN-taskset -c 2}
ROUNDS=${ROUNDS:-3}
BUDGETS=${BUDGETS:-45 120}
OUT=${OUT:-results/ryzen_$(hostname)_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 WP3_CV2_THREADS=1
if [[ -n "$PIN" ]] && ! command -v taskset >/dev/null; then echo "taskset missing: running unpinned"; PIN=""; fi

# 1. host + provenance
{
  echo "date: $(date -Is)"; echo "host: $(hostname)"
  lscpu | grep -E 'Model name|^CPU\(s\)|Thread\(s\) per core|MHz' || true
  echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo n/a)"
  echo "smt: $(cat /sys/devices/system/cpu/smt/control 2>/dev/null || echo n/a)"
  echo "pin: ${PIN:-none}"
  $PY -c "import sys, numpy, cv2, scipy; print('python', sys.version.split()[0], 'numpy', numpy.__version__, 'cv2', cv2.__version__, 'scipy', scipy.__version__)"
  echo "git: $(git rev-parse --short HEAD 2>/dev/null || echo n/a) $(git status --porcelain 2>/dev/null | wc -l) uncommitted"
  $PY -c "from wp3_live import preset_config; from wp3_live.config import config_hash; print('config_hash(printed_jenga)', config_hash(preset_config('printed_jenga')))"
} | tee "$OUT/host.txt"

# 2. synthetic held-out split
if [[ "${SKIP_SYNTH:-0}" != "1" ]]; then
  SC=results/scenes_heldout
  [[ -f $SC/index.json ]] || (cd bench && $PY scenes.py --out "../$SC" --split heldout --per-family 12)
  METHODS=(tuned_mc v3prove_mc v3adapt_mc v3enum_mc)
  for ((r = 0; r < ROUNDS; r++)); do
    n=${#METHODS[@]}
    for ((i = 0; i < n; i++)); do
      m=${METHODS[$(( (i + r) % n ))]}           # rotate order so no method always runs first/hot
      echo "round $r: $m"
      $PIN $PY bench/run_scenes.py --root . --scenes "$SC" --method "$m" --out "$OUT/${m}_r$r.jsonl"
    done
  done
  (cd bench && $PY score.py --scenes "../$SC" --runs ../"$OUT"/*_r0.jsonl --out "../$OUT/heldout_score.json") | tee "$OUT/heldout_score.txt"
  $PY bench/latency_summary.py --runs "$OUT"/*_r*.jsonl --warmup 3 --period-ms 66.7 --out "$OUT/heldout_latency.json" | tee "$OUT/heldout_latency.txt"
fi

# 3. bags
for BAG in "$@"; do
  name=$(basename "${BAG%/}"); name=${name%.db3}
  run_bag() { # tag, extra compare_bag args...
    local tag=$1; shift
    echo "bag $name: $tag"
    /usr/bin/time -v -o "$OUT/bag_${name}_${tag}_time.txt" $PIN $PY scripts/compare_bag.py "$BAG" \
      --out "$OUT/bag_${name}_${tag}" --policies merged_corroborated,whole_partition_mv --views both,three \
      --warmup-frames 5 ${MAX_FRAMES:+--max-frames "$MAX_FRAMES"} ${CONFIG:+--config "$CONFIG"} "$@" \
      > "$OUT/bag_${name}_${tag}.log" 2>&1 \
      || echo "  FAILED (see $OUT/bag_${name}_${tag}.log)"
    grep -E 'Percent of CPU|Maximum resident' "$OUT/bag_${name}_${tag}_time.txt" 2>/dev/null | sed 's/^/  /'
  }
  run_bag v2 --v3 off
  for b in $BUDGETS; do
    run_bag "adaptive_${b}ms" --v3 on --v3-solver adaptive --v3-frame-budget-ms "$b"
    run_bag "prove_${b}ms" --v3 on --v3-solver prove --v3-frame-budget-ms "$b"
  done
done

echo
echo "Results in $OUT. Report each number with this host. Compare p95 with 66.7 ms (15 Hz) and the M-STAT budget"
echo "agreed with WP5; budget exits cost coverage (abstention), never correctness."

#!/usr/bin/env bash
# V3 solver-budget sweep on the held-out synthetic split (168 scenes, 312 blocks).
#   bash bench/budget_sweep.sh [SCENES_DIR] [OUT_DIR]      (run from versions/v3)
# prove and enumerate at frame budgets 30/45/66/120/250 ms and unlimited (region budget = frame budget,
# node cap 6000), ROUNDS rounds (default 2), pinned to one core (PIN, default "taskset -c 2"); plus
# prove unlimited with a 10x node cap (60000), one round. Label every number with the host.
set -o pipefail
cd "$(dirname "$0")/.."
SC=${1:-results/scenes_heldout}; OUT=${2:-results/budget_sweep}
PIN=${PIN-taskset -c 2}; ROUNDS=${ROUNDS:-2}; PY=${PY:-python3}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$OUT"
[ -f "$SC/index.json" ] || (cd bench && $PY scenes.py --out "../$SC" --split heldout --per-family 12)
{ lscpu | grep 'Model name'; echo "pin: ${PIN:-none}"; git rev-parse --short HEAD 2>/dev/null; } > "$OUT/host.txt"
BUDGETS=(30 45 66 120 250 inf)
for ((r = 0; r < ROUNDS; r++)); do
  for solver in prove enum; do
    for b in "${BUDGETS[@]}"; do
      t0=$(date +%s)
      $PIN $PY bench/run_scenes.py --root . --scenes "$SC" --method "v3${solver}_mc" \
        --frame-budget-ms "$b" --region-budget-ms "$b" --out "$OUT/sweep_${solver}_${b}_r$r.jsonl"
      echo "round $r ${solver} ${b} ms: $(( $(date +%s) - t0 )) s"
    done
  done
done
t0=$(date +%s)
$PIN $PY bench/run_scenes.py --root . --scenes "$SC" --method v3prove_mc --frame-budget-ms inf --region-budget-ms inf \
  --max-nodes 60000 --out "$OUT/sweep_prove_inf_n60000_r0.jsonl"
echo "prove inf, 60000 nodes: $(( $(date +%s) - t0 )) s"
(cd bench && $PY sweep_analyze.py --scenes "../$SC" --runs ../"$OUT"/sweep_*.jsonl --out "../$OUT/sweep.json" --md "../$OUT/sweep.md")

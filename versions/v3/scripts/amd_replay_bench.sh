#!/usr/bin/env bash
# WP3 V3 — replay benchmark for the Ryzen 5 5600 (+ optional Radeon) desktop. Offline; no robot, no ROS needed.
# Reproduces the M4 numbers in results/V3_REPORT.md on the target CPU so the frame budget can be set from data.
# Usage (from the bundle root):
#   PY=python3 bash v3/scripts/amd_replay_bench.sh            # CPU path (default deployment)
#   PY=python3 GPU_KERNELS=1 bash v3/scripts/amd_replay_bench.sh   # + torch-ROCm kernel parity/timing if available
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-python3}
OUT=${OUT:-results/amd_bench_$(hostname)_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 WP3_CV2_THREADS=1
bash v3/scripts/amd_hw_check.sh > "$OUT/hw_check.txt" 2>&1 || true
# 1. unit + regression tests and the no-motion audit
(cd v3 && $PY -m pytest -q tests --ignore=tests/test_ros2_display_node.py -p no:cacheprovider) | tail -2 | tee "$OUT/tests.txt"
(cd v3 && $PY scripts/verify_zero_moveit.py) | tee "$OUT/zero_moveit.txt"
# 2. scenes (regenerated deterministically if absent: seeds 1000+ tune, 5000+ held-out)
[ -f results/v3_bench/scenes_heldout/index.json ] || $PY v3/bench/scenes.py --out results/v3_bench/scenes_heldout --split heldout --per-family 12
[ -f results/v3_bench/sequences/sequences.json ] || $PY v3/bench/sequences.py results/v3_bench/sequences
# 3. single-frame timing, one process at a time, rotated order (pin to one core to mimic ROS sharing)
PIN=${PIN:-taskset -c 2}
for rnd in 0 1 2; do
  for m in tuned_mc v3prove_mc v3enum_mc; do
    root=v3
    $PIN $PY v3/bench/run_scenes.py --root $root --scenes results/v3_bench/scenes_heldout --method $m --out "$OUT/${m}_r$rnd.jsonl"
  done
  $PIN $PY v3/bench/run_scenes.py --root v2-baseline --scenes results/v3_bench/scenes_heldout --method v2base_mc --out "$OUT/v2base_mc_r$rnd.jsonl"
done
$PY v3/bench/score.py --scenes results/v3_bench/scenes_heldout --runs "$OUT"/*_r0.jsonl --out "$OUT/score.json" | tee "$OUT/score.txt"
# 4. sequences (tracker + contract + WP5 rule), steady state incl. region cache
for m in tuned v3 v3_skip; do $PIN $PY v3/bench/run_sequences.py --root v3 --seqdir results/v3_bench/sequences --method $m --out "$OUT/seq_$m.json" | tee -a "$OUT/seq.txt"; done
# 5. optional GPU kernels (torch ROCm). Reports device, parity and time incl. host<->device transfer.
if [ "${GPU_KERNELS:-0}" = "1" ]; then
  $PY results/m4_gpu/scripts/gpu_kernels.py --backend torch-cpu --out "$OUT/kernels_torch_cpu.json" || true
  $PY - <<'PY' && $PY results/m4_gpu/scripts/gpu_kernels.py --backend torch-cuda --out "$OUT/kernels_torch_gpu.json" || echo "no ROCm torch device"
import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)
PY
fi
echo "results in $OUT  (compare p95 against the 66.7 ms frame period at 15 fps; see results/V3_REPORT.md section 6)"

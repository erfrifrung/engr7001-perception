#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == --help || $# -eq 0 ]]; then
    cat <<'HELP'
Usage:
  1. Profile live ROS node with py-spy flamegraph:
     bash scripts/profile_amd.sh --pid <PID> [--duration 15] [--out flame.svg]

  2. Profile bag replay on AMD desktop (cProfile + line_profiler if installed + p50/p95/p99 excl. warmup):
     bash scripts/profile_amd.sh --bag <BAG_DB3> [--config <JSON>] [--policy whole_partition_mv] [--frames 60] [--warmup 5]
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

PID=""
BAG=""
CFG="$IMPL_ROOT/ros2_ws/src/wp3_live_perception/config/printed_jenga_live.json"
POLICY="whole_partition_mv"
FRAMES="60"
WARMUP="5"
DURATION="15"
OUT_SVG="$IMPL_ROOT/results/amd_flame.svg"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pid) PID="$2"; shift 2 ;;
        --bag) BAG="$2"; shift 2 ;;
        --config) CFG="$2"; shift 2 ;;
        --policy) POLICY="$2"; shift 2 ;;
        --frames) FRAMES="$2"; shift 2 ;;
        --warmup) WARMUP="$2"; shift 2 ;;
        --duration) DURATION="$2"; shift 2 ;;
        --out) OUT_SVG="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ -n "$PID" ]]; then
    if ! command -v py-spy >/dev/null 2>&1; then
        echo "py-spy not found. Install on AMD desktop: pip install py-spy" >&2
        exit 1
    fi
    mkdir -p "$(dirname "$OUT_SVG")"
    echo "Recording py-spy flamegraph for PID=$PID (${DURATION}s) -> $OUT_SVG"
    exec py-spy record -o "$OUT_SVG" --duration "$DURATION" --rate 200 --pid "$PID"
fi

if [[ -z "$BAG" ]]; then
    echo "Provide either --pid <PID> or --bag <BAG_DB3>" >&2
    exit 2
fi

set +u
source /opt/ros/jazzy/setup.bash 2>/dev/null || true
set -u
export PYTHONPATH="$IMPL_ROOT:${PYTHONPATH:-}"

python3 - "$BAG" "$CFG" "$POLICY" "$FRAMES" "$WARMUP" <<'PY'
import cProfile, importlib.util, pstats, sys
import numpy as np

bag_path, cfg_path, policy, n_frames_s, warmup_s = sys.argv[1:6]
n_frames, warmup = int(n_frames_s), int(warmup_s)

from wp3_live import ViewInput, configure_single_thread_libs, load_runtime_config, run_perception_epoch
from wp3_live.bag_reader import chain_tf, load_bag_fast
from wp3_live.geometry_cache import CameraModelCache

configure_single_thread_libs()
cfg = load_runtime_config(cfg_path)
bag = load_bag_fast(bag_path, [("wrist", cfg.wrist.namespace), ("side", cfg.side.namespace)], use_aligned_depth=True)
cc = CameraModelCache()
iw, isd = bag.cameras["wrist"].info, bag.cameras["side"].info
cw = cc.get("wrist", iw["width"], iw["height"], iw["K"], iw["D"], iw["frame_id"], cfg.wrist.depth_scale_m) if iw else None
cs = cc.get("tripod", isd["width"], isd["height"], isd["K"], isd["D"], isd["frame_id"], cfg.side.depth_scale_m) if isd else None
Tw = chain_tf(bag.tf_static, cfg.frame_id, cw.frame_id) if cw else None
Ts = None
if cs:
    try:
        Ts = chain_tf(bag.tf_static, cfg.frame_id, cs.frame_id)
    except KeyError:
        Ts = None

wp_list = bag.cameras["wrist"].pairs[:n_frames]
sp_list = bag.cameras["side"].pairs[:n_frames]

pr = cProfile.Profile()
pr.enable()
times = []
for i, wp in enumerate(wp_list):
    sp = sp_list[i] if i < len(sp_list) else None
    w_vi = ViewInput("wrist", cw, wp["rgb"], wp["depth_u16"], Tw, wp["header_ns"], wp["rx_ns"]) if cw else None
    s_vi = ViewInput("tripod", cs, sp["rgb"], sp["depth_u16"], Ts, sp["header_ns"], sp["rx_ns"]) if (cs and sp and Ts is not None) else None
    res = run_perception_epoch(wrist=w_vi, side=s_vi, config=cfg, policy=policy, epoch_id=i + 1)
    if res.status == "ok":
        times.append(res.stage_timings_ms)
pr.disable()

print(f"=== cProfile cumtime (n_ok={len(times)}, warmup_excl={warmup}) ===")
pstats.Stats(pr).sort_stats("cumtime").print_stats(20)
print("=== cProfile tottime ===")
pstats.Stats(pr).sort_stats("tottime").print_stats(15)

eval_t = times[warmup:] if len(times) > warmup + 5 else times
if eval_t:
    tot = [t["total_ms"] for t in eval_t]
    p50, p95, p99 = np.percentile(tot, [50, 95, 99])
    print(f"Total ms p50/p95/p99 (excl {len(times) - len(eval_t)} warmup): {p50:.2f} / {p95:.2f} / {p99:.2f}")

if importlib.util.find_spec("line_profiler") is not None:
    from line_profiler import LineProfiler
    from wp3_live.proposals import _evaluate_cut_cues_with_rgb, enumerate_partition_hypotheses, fit_cuboid_top
    lp = LineProfiler(_evaluate_cut_cues_with_rgb, enumerate_partition_hypotheses, fit_cuboid_top)
    lp_fn = lp(run_perception_epoch)
    for i, wp in enumerate(wp_list[:20]):
        sp = sp_list[i] if i < len(sp_list) else None
        w_vi = ViewInput("wrist", cw, wp["rgb"], wp["depth_u16"], Tw, wp["header_ns"], wp["rx_ns"]) if cw else None
        s_vi = ViewInput("tripod", cs, sp["rgb"], sp["depth_u16"], Ts, sp["header_ns"], sp["rx_ns"]) if (cs and sp and Ts is not None) else None
        lp_fn(wrist=w_vi, side=s_vi, config=cfg, policy=policy, epoch_id=i + 1)
    lp.print_stats()
else:
    print("(Optional line_profiler not installed; install via pip install line_profiler for per-line timings.)")
PY

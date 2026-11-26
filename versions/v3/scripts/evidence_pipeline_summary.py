"""Summarise evidence_pipeline_ab.sh output: one row per perception mode (same bag, same twin).

python3 evidence_pipeline_summary.py OUT_DIR mode1 mode2 ...
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np

STAGES = ("deproject_roi_ms", "plane_ms", "robot_filter_ms", "heightmap_ms", "side_facade_ms",
          "proposals_and_rgb_ms", "verifier_ms", "v3_regions_ms", "total_ms")


def pct(a, q):
    return round(float(np.percentile(a, q)), 1) if len(a) else None


def main():
    out = Path(sys.argv[1])
    modes = sys.argv[2:]
    rows, stage_rows = [], []
    for m in modes:
        recs = [json.loads(l) for l in open(out / f"scenes_{m}.jsonl")]
        ok = [r for r in recs if r["status"] == "ok"]
        modes_c = Counter(r["mode"] for r in recs)
        n_meas = [r["n_measured"] + r["n_displaced"] for r in ok]
        n_amb = [r["n_amb"] for r in ok]
        proc = [r["proc_ms"] for r in ok if r["proc_ms"] is not None and r["proc_ms"] >= 0]
        # stability: per track id, spread of the published xy over the run
        xy = defaultdict(list)
        for r in ok:
            for t in r["measured"]:
                xy[t["id"]].append(t["xy"])
        spread = [1000 * float(np.linalg.norm(np.std(np.asarray(v), axis=0))) for v in xy.values() if len(v) >= 5]
        v3v = Counter()
        for r in ok:
            for s in r.get("v3") or []:
                for k, v in (s.get("verdicts") or {}).items():
                    v3v[k] += v
        srcs = Counter(t.get("src") for r in ok for t in r["measured"])
        scene = json.loads((out / f"scene_{m}.json").read_text())
        ids = set()
        for h in scene["history"]:
            ids |= set(h["ids"])
        rows.append(f"| {m} | {len(recs)} | {dict(modes_c)} | {pct(n_meas, 50)} / {max(n_meas) if n_meas else 0} | "
                    f"{pct(n_amb, 50)} / {max(n_amb) if n_amb else 0} | {len(xy)} | "
                    f"{(round(float(np.median(spread)), 2) if spread else None)} | {pct(proc, 50)} / {pct(proc, 95)} | "
                    f"{dict(v3v) or '-'} | {dict(srcs)} | {scene['max_objects']} / {len(ids)} |")
        st = {s: [r["timings"].get(s) for r in ok if r["timings"].get(s) is not None] for s in STAGES}
        stage_rows.append(f"| {m} | " + " | ".join(f"{pct(v, 50)} / {pct(v, 95)}" if v else "-" for v in st.values()) + " |")
    print("| mode | scene msgs | fallback modes | blocks published p50 / max | ambiguous p50 / max | track ids | "
          "median per-track xy spread (mm) | processing delay p50 / p95 (ms) | V3 verdicts | pose sources | "
          "MoveIt objects max at once / distinct ids |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    print("\n".join(rows))
    print()
    print("Stage timings p50 / p95 (ms):")
    print("| mode | " + " | ".join(STAGES) + " |")
    print("|---|" + "---|" * len(STAGES))
    print("\n".join(stage_rows))


if __name__ == "__main__":
    main()

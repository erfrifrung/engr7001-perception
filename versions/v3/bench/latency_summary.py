"""Latency and V3 budget-exit summary for run_scenes.py JSONL files (V3.1).

Warm-up is excluded per file (first --warmup scenes), each file is one (method, round). Reports p50/p95/p99/max,
the fraction of frames over a frame period, and the fraction of frames where V3 hit a budget (timeout/deadline
ambiguity reasons), i.e. where coverage was given up for time.

Usage: python latency_summary.py --runs OUT/*.jsonl --period-ms 66.7 --out OUT/latency.json
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

BUDGET_REASONS = ("timeout", "deadline")


def summarize(files, warmup: int, period_ms: float) -> dict:
    lat = defaultdict(list)
    budget = defaultdict(int)
    frames = defaultdict(int)
    for f in files:
        rows = [json.loads(l) for l in open(f)]
        for r in rows[warmup:]:
            m = r["method"]
            lat[m].append(float(r["timings"].get("total_ms", r["wall_ms"])))
            frames[m] += 1
            if any(any(k in str(a.get("reason", "")) for k in BUDGET_REASONS) for a in r.get("amb", [])):
                budget[m] += 1
    out = {}
    for m, L in lat.items():
        a = np.asarray(L)
        out[m] = dict(n=int(len(a)), p50=round(float(np.percentile(a, 50)), 2), p95=round(float(np.percentile(a, 95)), 2),
                      p99=round(float(np.percentile(a, 99)), 2), max=round(float(a.max()), 2),
                      over_period_frac=round(float(np.mean(a > period_ms)), 4),
                      v3_budget_exit_frac=round(budget[m] / max(1, frames[m]), 4))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--period-ms", type=float, default=66.7)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = summarize(a.runs, a.warmup, a.period_ms)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"{'method':16s} {'n':>5s} {'p50':>7s} {'p95':>7s} {'p99':>7s} {'max':>7s} {'>period':>8s} {'budget':>7s}")
    for m, r in sorted(res.items()):
        print(f"{m:16s} {r['n']:5d} {r['p50']:7.1f} {r['p95']:7.1f} {r['p99']:7.1f} {r['max']:7.1f} "
              f"{r['over_period_frac']:8.3f} {r['v3_budget_exit_frac']:7.3f}")

"""Analyse the V3 budget sweep (run_scenes.py JSONL with v3 stats) on the held-out split.

python3 sweep_analyze.py --scenes DIR --runs OUT/sweep_*.jsonl --out OUT/sweep.json --md OUT/sweep.md
Per (solver, budget, round): correct / wrong (5 mm, 5 deg, 3 mm Z), coverage, per-candidate-block verdicts
(stable / counterexample / timeout), regions skipped by the frame deadline, latency p50/p95/p99/max (warm-up
excluded). Then, for the unlimited-budget runs, which ground-truth blocks remain unresolved, by family:
"fundamentally ambiguous within the model" = unresolved with no wall-clock limit (and with a 10x node cap).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score as S  # noqa: E402


def tight_matches(poses, blocks):
    """Indices of GT blocks matched by a pose passing the tight gate, and count of wrong (published, not tight)."""
    if not poses or not blocks:
        return set(), len(poses)
    P = np.array([[*p["xy"], p["z"]] for p in poses])
    G = np.array([[b["cx"], b["cy"], 0.5 * (b["z0"] + b["z1"])] for b in blocks])
    D = np.linalg.norm(P[:, None] - G[None], axis=2)
    r, c = linear_sum_assignment(np.where(D <= S.LEG, D, 10.0))
    ok = set()
    for i, j in zip(r, c):
        if D[i, j] > S.LEG:
            continue
        p, b = poses[i], blocks[j]
        e = float(np.hypot(p["xy"][0] - b["cx"], p["xy"][1] - b["cy"]))
        ze = abs(float(p["z"]) - 0.5 * (b["z0"] + b["z1"]))
        if e <= S.TIGHT and S.yaw_err(p["yaw"], b["yaw"]) <= S.YAWT and ze <= S.ZT:
            ok.add(j)
    return ok, len(poses) - len(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", required=True)
    ap.add_argument("--warmup", type=int, default=3)
    a = ap.parse_args()
    index = json.loads((Path(a.scenes) / "index.json").read_text())
    by_file = {s["file"]: s for s in index}
    n_gt = sum(len(s["blocks"]) for s in index)
    rows = {}
    unresolved = {}
    for f in sorted(a.runs):
        tag = Path(f).stem                      # sweep_<solver>_<budget>[_n<nodes>]_r<round>
        m = re.match(r"sweep_(\w+?)_(\w+?)(?:_n(\d+))?_r(\d+)$", tag)
        if not m:
            continue
        solver, budget, nodes, rnd = m.group(1), m.group(2), m.group(3) or "6000", int(m.group(4))
        recs = [json.loads(l) for l in open(f)]
        ok_total = wrong_total = 0
        verd = defaultdict(int)
        skipped = 0
        frames_budget = 0
        lat = []
        un = defaultdict(int)
        for k, r in enumerate(recs):
            sc = by_file[r["file"]]
            ok, wrong = tight_matches(r["poses"], sc["blocks"])
            ok_total += len(ok)
            wrong_total += wrong
            for j, b in enumerate(sc["blocks"]):
                if j not in ok:
                    un[sc["family"]] += 1
            v3 = r.get("v3") or {}
            for kk, vv in (v3.get("verdicts") or {}).items():
                verd[kk] += vv
            skipped += int(v3.get("skipped_deadline", 0))
            frames_budget += int(bool(v3.get("skipped_deadline")) or bool((v3.get("verdicts") or {}).get("timeout")))
            if k >= a.warmup:
                lat.append(float(r["timings"].get("total_ms", r["wall_ms"])))
        L = np.asarray(lat)
        key = f"{solver}|{budget}|n{nodes}"
        rows.setdefault(key, []).append(dict(
            solver=solver, budget=budget, nodes=int(nodes), round=rnd, correct=ok_total, wrong=wrong_total,
            coverage=round(ok_total / n_gt, 3), stable=verd.get("stable", 0), counterexample=verd.get("counterexample", 0),
            timeout=verd.get("timeout", 0), regions_skipped_deadline=skipped, frames_hit_budget=frames_budget,
            p50=round(float(np.percentile(L, 50)), 1), p95=round(float(np.percentile(L, 95)), 1),
            p99=round(float(np.percentile(L, 99)), 1), max=round(float(L.max()), 1)))
        if budget == "inf":
            unresolved[key] = dict(un)
    order = {"30": 0, "45": 1, "66": 2, "120": 3, "250": 4, "inf": 5}
    keys = sorted(rows, key=lambda k: (k.split("|")[0], order.get(k.split("|")[1], 9), k.split("|")[2]))
    res = dict(n_gt_blocks=n_gt, n_scenes=len(index), runs={k: rows[k] for k in keys}, unresolved_at_inf=unresolved)
    Path(a.out).write_text(json.dumps(res, indent=1))
    md = ["| solver | budget ms | node cap | correct (mean of rounds) | wrong | coverage | stable / counterex. / timeout verdicts | regions skipped by deadline | frames that hit a budget | p50 / p95 / p99 / max ms |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for k in keys:
        rs = rows[k]
        mean = lambda f: np.mean([r[f] for r in rs])
        rng = lambda f: (f"{min(r[f] for r in rs)}–{max(r[f] for r in rs)}" if len(rs) > 1 and min(r[f] for r in rs) != max(r[f] for r in rs) else f"{rs[0][f]}")
        md.append(f"| {rs[0]['solver']} | {rs[0]['budget']} | {rs[0]['nodes']} | {mean('correct'):.1f} ({rng('correct')}) | {rng('wrong')} | "
                  f"{mean('coverage'):.3f} | {mean('stable'):.0f} / {mean('counterexample'):.0f} / {mean('timeout'):.0f} | "
                  f"{mean('regions_skipped_deadline'):.0f} | {mean('frames_hit_budget'):.0f} | "
                  f"{mean('p50'):.0f} / {mean('p95'):.0f} / {mean('p99'):.0f} / {mean('max'):.0f} |")
    md.append("")
    md.append("Unresolved ground-truth blocks at unlimited wall-clock budget, by family (last round of each run):")
    fams = sorted({f for u in unresolved.values() for f in u})
    md.append("| family | " + " | ".join(unresolved) + " |")
    md.append("|---|" + "---|" * len(unresolved))
    for fam in fams:
        md.append(f"| {fam} | " + " | ".join(str(u.get(fam, 0)) for u in unresolved.values()) + " |")
    Path(a.md).write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()

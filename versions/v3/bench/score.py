"""Score JSONL outputs of run_scenes.py against scene ground truth (one-to-one Hungarian on 3D centre).

Gates: legacy 15 mm (continuity) and provisional 5 mm / 5 deg (symmetry-aware) + |dz| <= 3 mm + fitted size
within +-7.5 mm / +-5 mm.  The provisional gate is NOT a production threshold: WP5 (fixture owner) must confirm it.
V3.1: Z is gated and yaw / Z error percentiles are reported (they were collected but dropped from the summary).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

LEG, TIGHT, YAWT, ZT = 0.015, 0.005, 5.0, 0.003


def yaw_err(a, b):
    d = np.degrees(a - b) % 180.0
    return float(min(d, 180.0 - d))


def inside(xy, b, m=0.0):
    d = np.asarray(xy) - [b["cx"], b["cy"]]
    c, s = np.cos(b["yaw"]), np.sin(b["yaw"])
    return abs(d[0] * c + d[1] * s) <= b["lx"] / 2 + m and abs(-d[0] * s + d[1] * c) <= b["wy"] / 2 + m


def score_scene(poses, blocks):
    n_p, n_g = len(poses), len(blocks)
    out = dict(n_pred=n_p, n_gt=n_g, matched15=0, tight=0, wrong_tight=0, wrong15=0, false_split=0,
               false_merge=0, errs_mm=[], yaw_errs=[], z_errs_mm=[], dims_errs_mm=[])
    pairs = []
    if n_p and n_g:
        P = np.array([[*p["xy"], p["z"]] for p in poses])
        G = np.array([[b["cx"], b["cy"], 0.5 * (b["z0"] + b["z1"])] for b in blocks])
        D = np.linalg.norm(P[:, None] - G[None], axis=2)
        r, c = linear_sum_assignment(np.where(D <= LEG, D, 10.0))
        pairs = [(i, j) for i, j in zip(r, c) if D[i, j] <= LEG]
    tight_p = set()
    for i, j in pairs:
        p, b = poses[i], blocks[j]
        e = float(np.hypot(p["xy"][0] - b["cx"], p["xy"][1] - b["cy"]))
        ye = yaw_err(p["yaw"], b["yaw"])
        ze = abs(float(p["z"]) - 0.5 * (b["z0"] + b["z1"]))
        dims = sorted(p["dims"], reverse=True)
        ok = e <= TIGHT and ye <= YAWT and ze <= ZT and abs(dims[0] - 75) <= 7.5 and abs(dims[1] - 25) <= 5
        out["errs_mm"].append(round(e * 1000, 3))
        out["yaw_errs"].append(round(ye, 3))
        out["z_errs_mm"].append(round(ze * 1000, 3))
        out["dims_errs_mm"].append(round(max(abs(dims[0] - 75), abs(dims[1] - 25)), 2))
        if ok:
            tight_p.add(i)
    out["matched15"] = len(pairs)
    out["tight"] = len(tight_p)
    out["wrong_tight"] = n_p - len(tight_p)
    out["wrong15"] = n_p - len(pairs)
    for b in blocks:
        zc = 0.5 * (b["z0"] + b["z1"])
        if sum(inside(p["xy"], b) and abs(p["z"] - zc) < 0.006 for p in poses) >= 2:
            out["false_split"] += 1
    for p in poses:
        near = [b for b in blocks if inside(p["xy"], b, 0.003) and abs(p["z"] - 0.5 * (b["z0"] + b["z1"])) < 0.006]
        if len(near) >= 2 and not any(inside(p["xy"], b, -0.003) for b in near):
            out["false_merge"] += 1
    return out


def summarize(rows, index):
    by_file = {s["file"]: s for s in index}
    agg = defaultdict(lambda: defaultdict(float))
    fam = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    lat = defaultdict(list)
    errs = defaultdict(list)
    yerrs = defaultdict(list)
    zerrs = defaultdict(list)
    for r in rows:
        if r.get("rep", 0) != 0:
            lat[r["method"]].append(r["timings"].get("total_ms", r["wall_ms"]))
            continue
        sc = by_file[r["file"]]
        s = score_scene(r["poses"], sc["blocks"])
        m = r["method"]
        lat[m].append(r["timings"].get("total_ms", r["wall_ms"]))
        errs[m].extend(s["errs_mm"])
        yerrs[m].extend(s["yaw_errs"])
        zerrs[m].extend(s["z_errs_mm"])
        for k in ("n_pred", "n_gt", "matched15", "tight", "wrong_tight", "wrong15", "false_split", "false_merge"):
            agg[m][k] += s[k]
            fam[m][sc["family"]][k] += s[k]
        agg[m]["scenes"] += 1
        agg[m]["abstained_scenes"] += int(len(r["amb"]) > 0)
        fam[m][sc["family"]]["scenes"] += 1
    out = {}
    for m, a in agg.items():
        L = np.array(lat[m])
        out[m] = dict(
            scenes=int(a["scenes"]), gt_blocks=int(a["n_gt"]), published=int(a["n_pred"]),
            correct_tight=int(a["tight"]), wrong_tight=int(a["wrong_tight"]), wrong_legacy15=int(a["wrong15"]),
            matched_legacy15=int(a["matched15"]), coverage_tight=round(a["tight"] / max(1, a["n_gt"]), 3),
            false_splits=int(a["false_split"]), false_merges=int(a["false_merge"]),
            scenes_with_ambiguity=int(a["abstained_scenes"]),
            centre_err_mm_p50=round(float(np.median(errs[m])), 3) if errs[m] else None,
            centre_err_mm_p95=round(float(np.percentile(errs[m], 95)), 3) if errs[m] else None,
            yaw_err_deg_p50=round(float(np.median(yerrs[m])), 3) if yerrs[m] else None,
            yaw_err_deg_p95=round(float(np.percentile(yerrs[m], 95)), 3) if yerrs[m] else None,
            yaw_err_deg_max=round(float(np.max(yerrs[m])), 3) if yerrs[m] else None,
            z_err_mm_p50=round(float(np.median(zerrs[m])), 3) if zerrs[m] else None,
            z_err_mm_p95=round(float(np.percentile(zerrs[m], 95)), 3) if zerrs[m] else None,
            latency_ms=dict(n=len(L), p50=round(float(np.percentile(L, 50)), 2), p95=round(float(np.percentile(L, 95)), 2),
                            p99=round(float(np.percentile(L, 99)), 2), max=round(float(L.max()), 2)),
            per_family={f: {k: int(v) for k, v in d.items()} for f, d in sorted(fam[m].items())},
        )
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    index = json.loads((Path(a.scenes) / "index.json").read_text())
    rows = [json.loads(l) for f in a.runs for l in open(f)]
    res = summarize(rows, index)
    Path(a.out).write_text(json.dumps(res, indent=1))
    hdr = f"{'method':22s} {'pub':>4s} {'ok5':>4s} {'wrong5':>6s} {'wr15':>4s} {'cov5':>5s} {'split':>5s} {'merge':>5s} {'amb':>4s} {'p50':>6s} {'p95':>6s} {'p99':>6s} {'max':>6s}"
    print(hdr)
    for m, r in res.items():
        l = r["latency_ms"]
        print(f"{m:22s} {r['published']:4d} {r['correct_tight']:4d} {r['wrong_tight']:6d} {r['wrong_legacy15']:4d} {r['coverage_tight']:5.2f} {r['false_splits']:5d} {r['false_merges']:5d} {r['scenes_with_ambiguity']:4d} {l['p50']:6.1f} {l['p95']:6.1f} {l['p99']:6.1f} {l['max']:6.1f}")

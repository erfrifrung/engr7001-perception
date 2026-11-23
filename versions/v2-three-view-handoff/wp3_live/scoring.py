"""Strict physical-membership, false-confidence, EMPTY-negative & legacy centre-distance scorer."""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from .policies import EpochResult


def clopper_pearson_upper_95(failures: int, n_trials: int) -> float | None:
    """Exact one-sided 95% upper confidence bound on true failure probability."""
    if n_trials <= 0:
        return None
    from scipy.stats import beta
    if failures >= n_trials:
        return 1.0
    return float(beta.ppf(0.95, failures + 1, n_trials - failures))


def load_ground_truth_labels(path: str | Path, bag_filter: str | None = None) -> list[dict[str, Any]]:
    """Load frozen 12 Nov labels.csv OR extended printed-Jenga holdout labels."""
    rows: list[dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if bag_filter and r.get("bag") and r.get("bag") != bag_filter:
                continue
            x_s = (r.get("x_m") or "").strip()
            y_s = (r.get("y_m") or "").strip()
            z_s = (r.get("z_m") or "").strip()
            yaw_s = (r.get("yaw_deg") or "").strip()
            aid = r.get("annotation_id", r.get("block_id", ""))
            in_roi = str(r.get("in_roi", "yes")).strip().lower()
            scorable = str(r.get("scorable", "True")).strip().lower() != "false"
            is_empty = aid.upper() == "EMPTY" or in_roi == "no_blocks"
            rows.append({
                "t0": float(r["t_start_s"]),
                "t1": float(r["t_end_s"]),
                "id": aid,
                "xy": (float(x_s), float(y_s)) if x_s and y_s else None,
                "z_m": float(z_s) if z_s else None,
                "yaw_deg": float(yaw_s) if yaw_s else None,
                "scorable": scorable,
                "is_empty": is_empty,
                "stratum": r.get("stratum", r.get("split", "default")),
            })
    return rows


def sym_yaw_err_deg(a_deg: float, b_deg: float, sym_deg: float) -> float:
    d = (a_deg - b_deg) % sym_deg
    return float(min(d, sym_deg - d))


def score_epoch_against_labels(
    t_rel_s: float,
    epoch: EpochResult,
    labels: list[dict[str, Any]],
    sym_deg: float,
    gates_m: tuple[float, float] = (0.015, 0.030),
    exclude_m: float = 0.040,
) -> dict[str, Any] | None:
    """Score one frame with strict physical membership + EMPTY negative support + legacy centre hits."""
    act = [l for l in labels if l["t0"] <= t_rel_s <= l["t1"]]
    if not act:
        return None
    is_empty_frame = any(l["is_empty"] for l in act)
    blocks = [l for l in act if l["scorable"] and l["xy"] is not None]
    if not blocks and not is_empty_frame:
        return None

    excl = [l["xy"] for l in act if not l["scorable"] and l["xy"] is not None]
    inst_xy = [p.xy_m for p in epoch.poses]
    inst_yaw = [p.yaw_deg_mod for p in epoch.poses]
    grid = epoch.hm_grid
    lmap = epoch.label_map

    def lookup_region(xy: tuple[float, float]) -> int:
        if grid is None or lmap is None:
            return 0
        rows, cols = grid.world_to_ij(np.array([[xy[0], xy[1]]], dtype=np.float64))
        i, j = int(rows[0]), int(cols[0])
        if 0 <= i < grid.ny and 0 <= j < grid.nx:
            patch = lmap[max(0, i - 2): i + 3, max(0, j - 2): j + 3]
            pos = patch[patch > 0]
            if pos.size:
                return int(np.bincount(pos).argmax())
        return 0

    if is_empty_frame:
        n_fp = len(inst_xy)
        return {
            "t": round(t_rel_s, 3),
            "empty_frame": True,
            "n_labels": 0,
            "n_out": len(inst_xy),
            "n_amb": len(epoch.ambiguous_clusters),
            "count_correct": int(n_fp == 0),
            "empty_fp": n_fp,
            "false_confident": int(n_fp > 0),
            "merged_label_pairs": 0,
            "membership_hits_15": 0,
            "legacy_hit_15": 0,
            "legacy_hit_30": 0,
            "legacy_extra_15": n_fp,
            "all_matched_15_no_extra": int(n_fp == 0),
            "membership_all_ok": int(n_fp == 0),
            "err_mm_15": [],
            "yaw_err_deg_15": [],
        }

    # Map each GT label to predicted instance ID in label_map (detecting 144 mm merged regions!)
    gt_region_ids = [lookup_region(b["xy"]) for b in blocks]
    region_counts: dict[int, int] = {}
    for rid in gt_region_ids:
        if rid > 0:
            region_counts[rid] = region_counts.get(rid, 0) + 1
    merged_pairs = sum(max(0, c - 1) for c in region_counts.values())

    out: dict[str, Any] = {
        "t": round(t_rel_s, 3),
        "empty_frame": False,
        "n_labels": len(blocks),
        "n_out": len(inst_xy),
        "n_amb": len(epoch.ambiguous_clusters),
        "count_correct": int(len(inst_xy) == len(blocks)),
        "empty_fp": 0,
        "merged_label_pairs": merged_pairs,
    }

    for g in gates_m:
        hits = 0
        memb_hits = 0
        errs: list[float] = []
        yerrs: list[float] = []
        used: set[int] = set()
        if inst_xy:
            D = np.array([[math.hypot(ox - b["xy"][0], oy - b["xy"][1]) for (ox, oy) in inst_xy] for b in blocks])
            rr, cc = linear_sum_assignment(np.where(D <= g, D, 1e3))
            for r_, c_ in zip(rr, cc):
                if D[r_, c_] <= g:
                    hits += 1
                    used.add(int(c_))
                    errs.append(float(D[r_, c_] * 1000.0))
                    rid = gt_region_ids[r_]
                    # Strict physical membership rejects a centroid if its underlying region covers >=2 blocks
                    if rid == 0 or region_counts.get(rid, 0) <= 1:
                        memb_hits += 1
                    if blocks[r_]["yaw_deg"] is not None:
                        yerrs.append(sym_yaw_err_deg(inst_yaw[c_], blocks[r_]["yaw_deg"], sym_deg))
        extra = sum(
            1 for k, (ox, oy) in enumerate(inst_xy)
            if k not in used and not any(math.hypot(ox - ex[0], oy - ex[1]) <= exclude_m for ex in excl)
        )
        tag = int(round(g * 1000))
        out[f"legacy_hit_{tag}"] = hits
        out[f"membership_hits_{tag}"] = memb_hits
        out[f"legacy_extra_{tag}"] = extra
        out[f"err_mm_{tag}"] = [round(v, 2) for v in errs]
        out[f"yaw_err_deg_{tag}"] = [round(v, 2) for v in yerrs]

    out["all_matched_15_no_extra"] = int(out["legacy_hit_15"] == len(blocks) and out["legacy_extra_15"] == 0)
    out["membership_all_ok"] = int(
        out["membership_hits_15"] == len(blocks) and out["legacy_extra_15"] == 0 and merged_pairs == 0
    )
    # False-confident frame: emitted an extra, a merged confident prediction, or wrong count without abstention
    out["false_confident"] = int(
        out["legacy_extra_15"] > 0 or merged_pairs > 0 or (len(inst_xy) > 0 and len(epoch.ambiguous_clusters) == 0 and out["membership_hits_15"] < len(inst_xy))
    )
    return out


def aggregate_method_summary(
    frames_records: list[dict[str, Any]],
    total_raw_frames: int,
    dropped: dict[str, int],
    warmup_frames: int = 0,
) -> dict[str, Any]:
    """Compute exact per-stage p50/p95/p99 ms (excluding warm-up) and coverage/risk gates."""
    ok_recs = [r for r in frames_records if r.get("status") == "ok"]
    use_warm = max(0, int(warmup_frames))
    t_recs = ok_recs[use_warm:] if (use_warm > 0 and len(ok_recs) > use_warm + 5) else ok_recs
    warm_excluded = len(ok_recs) - len(t_recs)

    summary: dict[str, Any] = {
        "frames_total": total_raw_frames,
        "frames_ok": len(ok_recs),
        "warmup_excluded": warm_excluded,
        "dropped": dropped,
        "frames_with_ambiguous": sum(1 for r in ok_recs if r.get("n_amb", 0) > 0),
        "timings_ms": {},
    }
    if t_recs:
        all_keys = sorted({k for r in t_recs for k in r.get("stage_timings_ms", {})})
        for k in all_keys:
            vals = [float(r["stage_timings_ms"][k]) for r in t_recs if k in r.get("stage_timings_ms", {})]
            if vals:
                p50, p95, p99 = np.percentile(vals, [50.0, 95.0, 99.0])
                summary["timings_ms"][k] = {
                    "p50": round(float(p50), 2),
                    "p95": round(float(p95), 2),
                    "p99": round(float(p99), 2),
                }

    sc = [r["score"] for r in ok_recs if r.get("score") is not None]
    if sc:
        non_empty = [s for s in sc if not s["empty_frame"]]
        empty = [s for s in sc if s["empty_frame"]]
        opps = sum(s["n_labels"] for s in non_empty)
        memb_hits_15 = sum(s["membership_hits_15"] for s in non_empty)
        leg_hits_15 = sum(s["legacy_hit_15"] for s in non_empty)
        leg_hits_30 = sum(s["legacy_hit_30"] for s in non_empty)
        extras_15 = sum(s["legacy_extra_15"] for s in sc)
        errs = [e for s in non_empty for e in s["err_mm_15"]]
        yerrs = [y for s in non_empty for y in s["yaw_err_deg_15"]]
        fc_frames = sum(s["false_confident"] for s in sc)
        summary["scored"] = {
            "scored_frames": len(sc),
            "empty_frames": len(empty),
            "empty_fp_total": sum(s["empty_fp"] for s in empty),
            "opportunities": opps,
            "raw_count_correct": sum(s["count_correct"] for s in sc),
            "membership_hits_15": memb_hits_15,
            "membership_all_ok": sum(s["membership_all_ok"] for s in non_empty),
            "membership_recall_pct": round(100.0 * memb_hits_15 / max(1, opps), 1),
            "legacy_hit_15": leg_hits_15,
            "legacy_hit_30": leg_hits_30,
            "legacy_all_matched_15_no_extra": sum(s["all_matched_15_no_extra"] for s in non_empty),
            "extra_15": extras_15,
            "frames_with_merge": sum(1 for s in non_empty if s["merged_label_pairs"] > 0),
            "false_confident_frames": fc_frames,
            "false_confident_rate_pct": round(100.0 * fc_frames / max(1, len(sc)), 1),
            "xy_err_mm_15": {
                "median": round(float(np.median(errs)), 2),
                "p95": round(float(np.percentile(errs, 95)), 2),
            } if errs else None,
            "yaw_err_deg_15": {
                "median": round(float(np.median(yerrs)), 2),
                "p95": round(float(np.percentile(yerrs, 95)), 2),
            } if yerrs else None,
        }
    return summary

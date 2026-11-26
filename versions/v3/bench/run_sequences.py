"""Replay rendered sequences through epoch -> tracker -> v2.1 contract -> offline WP5 fixture rule.

WP5 rule (provisional, owned by WP5): update a fixture from a track iff state in {MEASURED, DISPLACED},
snapshot_commit_eligible, not identity_ambiguous, and publish_stamp - observation_stamp <= 150 ms.
An update is *incorrect* if the track pose is > 5 mm / 5 deg from the true pose of the nearest block at the
observation frame.  A *confident identity error* is a fresh, non-ambiguous track whose underlying object
differs from the object it was first associated with.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--seqdir", required=True)
ap.add_argument("--method", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()
sys.path.insert(0, str(Path(args.root).resolve()))

from wp3_live import ViewInput, preset_config, run_perception_epoch  # noqa: E402
from wp3_live.contract import build_atomic_scene_payload  # noqa: E402
from wp3_live.dwell_gate import DwellStatus  # noqa: E402
from wp3_live.fusion_tracker import WP3TemporalTracker  # noqa: E402
try:                                    # V3.2 trees only (--root may point at an older tree)
    from wp3_live.fusion_tracker import tracker_from_config  # noqa: E402
except ImportError:
    tracker_from_config = None
from wp3_live.geometry_cache import CameraModelCache  # noqa: E402
from wp3_live.temporal_cache import EpochTemporalCache  # noqa: E402

DW = DwellStatus(True, True, True, True, 0.0, 0.1, 0.2, 80.0, "ok")
cache = CameraModelCache()
meta = json.loads((Path(args.seqdir) / "sequences.json").read_text())
m = args.method


def make():
    cfg = preset_config("printed_jenga")
    cfg.calib_epoch = "synthetic"
    v3 = m.startswith("v3")
    if hasattr(cfg, "enable_v3_regions"):
        cfg.enable_v3_regions = v3
    if "skip" in m:
        cfg.enable_temporal_skip = True
    if v3 and "noregcache" in m:
        cfg.v3_region_cache = False
    if v3 and "v3fix" in m and hasattr(cfg, "v3_skip_pose_check"):
        cfg.v3_skip_pose_check = True
    kw = {}
    if v3 and "v2tracker" not in m:
        kw = dict(persistent_joint_occlusion=True, reanchor_after=2)
    if "v32" in m:                       # V3.2 tracker: duplicate merge + observed-free retirement (config defaults)
        if tracker_from_config is None:
            raise SystemExit(f"method {m} needs a V3.2 tree (--root)")
        return cfg, tracker_from_config(cfg)
    tr = WP3TemporalTracker(cfg.block, cfg.thresholds, **kw)
    return cfg, tr


def yaw_err(a, b):
    d = np.degrees(a - b) % 180.0
    return min(d, 180.0 - d)


results = {}
for seq, frames in meta.items():
    cfg, tracker = make()
    tc = EpochTemporalCache()
    id_map: dict[int, int] = {}
    last_now = -1.0
    rows = []
    for k, fr in enumerate(frames):
        z = np.load(Path(args.seqdir) / fr["file"])
        cw = cache.get("wrist", 1280, 720, z["Kw"], np.zeros(5), "w", 0.001)
        cs = cache.get("tripod", 640, 480, z["Ks"], np.zeros(5), "s", 0.001)
        links = fr["links"]
        if "urdflag" in m and k >= 2:
            links = frames[k - 2]["links"]           # latest-TF lookup lagging the image by 2 frames
        if "nourdf" in m:
            links = None
        st = fr["stamp_ns"]
        t0 = time.perf_counter()
        ep = run_perception_epoch(
            wrist=ViewInput("wrist", cw, z["rgb_w"], z["d_w"], z["Tw"], st, st),
            side=ViewInput("tripod", cs, z["rgb_s"], z["d_s"], z["Ts"], st, st),
            config=cfg, policy="merged_corroborated", dwell=DW, epoch_id=k + 1, temporal_cache=tc,
            robot_links=links,
        )
        proc_ms = (time.perf_counter() - t0) * 1000.0
        # monotonic replay clock: stamp + processing, never earlier than the previous publish (a backwards
        # clock makes the V2 tracker reset as if the bag had been seeked)
        now_s = max(last_now + 1e-6, st * 1e-9 + proc_ms / 1000.0)
        last_now = now_s
        fs_kw = {"free_space": ep.free_space} if getattr(tracker, "retire_on_observed_free", False) else {}
        tracks = tracker.update(ep.poses, ep.pose_sources, ep.stamp_ns, now_s, side_fresh_global=True,
                                pose_side_visible=ep.pose_side_visible, **fs_kw)
        payload = build_atomic_scene_payload(ep, tracks, cfg, epoch_id=k + 1, publish_stamp_ns=int(now_s * 1e9),
                                             code_hash="bench")
        json.dumps(payload)  # contract must stay JSON-serialisable
        gt = fr["blocks"]
        frame_by_stamp = {f["stamp_ns"]: f for f in frames}
        out_tr = []
        for t in payload["tracks"]:
            age_ms = (payload["publish_stamp_ns"] - t["observation_stamp_ns"]) * 1e-6
            obs_frame = frame_by_stamp.get(t["observation_stamp_ns"], fr)
            g = obs_frame["blocks"]
            d = [np.hypot(t["pose_table_m"][0] - b["cx"], t["pose_table_m"][1] - b["cy"]) for b in g]
            j = int(np.argmin(d)) if d else -1
            err = d[j] * 1000 if d else 1e9
            yerr = yaw_err(t["yaw_rad"], g[j]["yaw"]) if d else 180.0
            # truth at publish time (the block may have moved since the observation)
            dn = [np.hypot(t["pose_table_m"][0] - b["cx"], t["pose_table_m"][1] - b["cy"]) for b in gt]
            jn = int(np.argmin(dn)) if dn else -1
            err_now = dn[jn] * 1000 if dn else 1e9
            fresh_ok = t["state"] in ("MEASURED", "DISPLACED")
            update = fresh_ok and t["snapshot_commit_eligible"] and not t["identity_ambiguous"] and age_ms <= 150.0
            oid = gt[jn]["oid"] if (dn and err_now < 15) else None
            if fresh_ok and oid is not None and t["track_id"] not in id_map:
                id_map[t["track_id"]] = oid
            id_err = bool(fresh_ok and not t["identity_ambiguous"] and oid is not None
                          and id_map.get(t["track_id"]) not in (None, oid))
            out_tr.append(dict(id=t["track_id"], state=t["state"], age_ms=round(age_ms, 1), err_mm=round(err, 2),
                               err_now_mm=round(err_now, 2), yaw_err=round(yerr, 2), update=update,
                               wrong_update=bool(update and (err_now > 5.0 or yerr > 5.0)),
                               id_ambiguous=t["identity_ambiguous"], confident_id_error=id_err,
                               src=t["evidence_source"]))
        rows.append(dict(frame=k, arm=fr["arm_present"], proc_ms=round(proc_ms, 2), n_poses=len(ep.poses),
                         n_amb=len(ep.ambiguous_clusters), cache_hit=bool(ep.evidence_log.get("temporal_cache_hit")),
                         v3=[s.get("v3") for s in ep.evidence_log.get("split_logs", []) if "v3" in s],
                         tracks=out_tr, n_gt=len(gt)))
    # summary
    ups = [t for r in rows for t in r["tracks"] if t["update"]]
    n_blocks = len(frames[0]["blocks"])
    last_arm = max([r["frame"] for r in rows if r["arm"]], default=None)
    rec = None
    if last_arm is not None:
        for r in rows:
            if r["frame"] > last_arm and sum(t["update"] and not t["wrong_update"] for t in r["tracks"]) >= n_blocks:
                rec = round((r["frame"] - last_arm) * 66.67, 1)
                break
    good_frames = sum(1 for r in rows if sum(t["update"] and not t["wrong_update"] for t in r["tracks"]) >= n_blocks)
    results[seq] = dict(
        frames=len(rows), fixture_updates=len(ups), incorrect_updates=sum(t["wrong_update"] for t in ups),
        max_err_now_of_updates_mm=round(max([t["err_now_mm"] for t in ups], default=0.0), 2),
        frames_all_blocks_updatable=good_frames, confident_id_errors=sum(t["confident_id_error"] for r in rows for t in r["tracks"]),
        ms_to_full_recovery_after_arm=rec,
        proc_ms=dict(p50=round(float(np.percentile([r["proc_ms"] for r in rows], 50)), 2),
                     p95=round(float(np.percentile([r["proc_ms"] for r in rows], 95)), 2),
                     max=round(max(r["proc_ms"] for r in rows), 2)),
        temporal_cache_hits=sum(r["cache_hit"] for r in rows),
        rows=rows,
    )
    print(f"{m:24s} {seq:24s} upd={len(ups):3d} wrong={results[seq]['incorrect_updates']:2d} "
          f"maxerr={results[seq]['max_err_now_of_updates_mm']:6.2f} allok_frames={good_frames:2d}/{len(rows)} "
          f"idErr={results[seq]['confident_id_errors']:2d} recov={rec} p50={results[seq]['proc_ms']['p50']} "
          f"p95={results[seq]['proc_ms']['p95']} hits={results[seq]['temporal_cache_hits']}")
Path(args.out).write_text(json.dumps(results, indent=1))

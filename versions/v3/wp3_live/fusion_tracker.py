"""Late view association, MDS track confirmation, 1.8 s occlusion hold, and frozen-stream watchdog."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import BlockSpec, PolicyThresholds
from .proposals import BlockFit


@dataclass
class StreamHealth:
    role: str
    last_stamp_ns: int = -1
    last_rx_s: float = -1.0
    last_frame_sig: str = ""
    same_sig_since_s: float = -1.0
    frozen: bool = False
    stale: bool = False
    dropped_stale: int = 0
    dropped_unpaired: int = 0


class StreamWatchdog:
    """Detect missing, stale, future-stamped, and frozen-frame streams separately per camera."""

    def __init__(self, max_age_s: float = 0.35, freeze_timeout_s: float = 1.2) -> None:
        self.max_age_s = float(max_age_s)
        self.freeze_timeout_s = float(freeze_timeout_s)
        self.streams: dict[str, StreamHealth] = {}

    def update(self, role: str, stamp_ns: int, rx_s: float, now_s: float, img_bytes: bytes | memoryview) -> StreamHealth:
        st = self.streams.setdefault(role, StreamHealth(role=role))
        age_s = now_s - (stamp_ns * 1e-9)
        # Lightweight 4 KB sparse checksum detects identical frozen publisher frames in < 0.05 ms
        mv = memoryview(img_bytes).cast("B")
        stride = max(1, len(mv) // 1024)
        sample = bytes(mv[::stride])
        sig = hashlib.blake2b(sample, digest_size=8).hexdigest()

        if stamp_ns != st.last_stamp_ns or sig != st.last_frame_sig:
            st.last_stamp_ns = stamp_ns
            st.last_frame_sig = sig
            st.same_sig_since_s = now_s
            st.frozen = False
        else:
            if st.same_sig_since_s >= 0 and (now_s - st.same_sig_since_s) >= self.freeze_timeout_s:
                st.frozen = True

        st.last_rx_s = now_s
        # Fix node.py:151 bug: allow slight clock skew [-5 ms, max_age_s]
        st.stale = not (-0.005 <= age_s <= self.max_age_s)
        if st.stale:
            st.dropped_stale += 1
        return st

    def check_timeout(self, role: str, now_s: float) -> StreamHealth:
        st = self.streams.setdefault(role, StreamHealth(role=role))
        if st.last_rx_s < 0 or (now_s - st.last_rx_s) > self.max_age_s:
            st.stale = True
        return st


def symmetry_yaw_diff_rad(a_rad: float, b_rad: float, is_square: bool) -> float:
    period = np.pi / 2.0 if is_square else np.pi
    d = abs((a_rad - b_rad + period / 2.0) % period - period / 2.0)
    return float(d)


def associate_late_hypotheses(
    primary_fits: list[BlockFit],
    secondary_fits: list[BlockFit],
    block: BlockSpec,
    th: PolicyThresholds,
) -> list[tuple[int, int]]:
    """1-to-1 Hungarian late hypothesis match requiring >=2x uniqueness margin & symmetry check."""
    if not primary_fits or not secondary_fits:
        return []
    n_p, n_s = len(primary_fits), len(secondary_fits)
    cost = np.full((n_p, n_s), np.inf, dtype=np.float64)
    yaw_lim = float(np.radians(th.assoc_gate_yaw_deg))
    for i, pf in enumerate(primary_fits):
        for j, sf in enumerate(secondary_fits):
            dist = float(np.hypot(pf.xy_m[0] - sf.xy_m[0], pf.xy_m[1] - sf.xy_m[1]))
            dyaw = symmetry_yaw_diff_rad(pf.yaw_rad, sf.yaw_rad, block.is_square)
            if dist <= th.assoc_gate_xy_m and dyaw <= yaw_lim:
                cost[i, j] = dist

    # Reject any row/column where 2nd best finite match is not >= uniqueness_margin * best
    for i in range(n_p):
        fin = np.sort(cost[i][np.isfinite(cost[i])])
        if len(fin) >= 2 and fin[1] < th.assoc_uniqueness_margin * max(fin[0], 1e-4):
            cost[i, :] = np.inf
    for j in range(n_s):
        fin = np.sort(cost[:, j][np.isfinite(cost[:, j])])
        if len(fin) >= 2 and fin[1] < th.assoc_uniqueness_margin * max(fin[0], 1e-4):
            cost[:, j] = np.inf

    finite = np.isfinite(cost)
    if not np.any(finite):
        return []
    big = (min(n_p, n_s) + 1) * (1.0 + float(cost[finite].max()))
    rr, cc = linear_sum_assignment(np.where(finite, cost, big))
    return [(int(r), int(c)) for r, c in zip(rr, cc) if finite[r, c]]


@dataclass
class TrackedBlock:
    track_id: int
    block_name: str
    fit: BlockFit
    source: str
    state: str                   # MEASURED | DISPLACED | OCCLUDED | STALE | UNKNOWN
    mds_score: int               # F38 signed agreement score (+1/-1)
    confirmed: bool
    observation_stamp_ns: int    # NEVER refreshed during occlusion/hold!
    last_seen_s: float
    anchor_xy_m: tuple[float, float]
    target_side_fresh: bool
    snapshot_commit_eligible: bool
    identity_ambiguous: bool = False
    ambiguous_peer_ids: tuple[int, ...] = ()
    stable_count: int = 0                # V3: consecutive fresh, unambiguous, consistent observations
    stable_ref_xy_m: tuple[float, float] | None = None  # V3.1: pose where the current stable run started


class WP3TemporalTracker:
    """Identity-preserving tracker with F38 MDS confirmation, 1.8 s hold, and target-level shift check."""

    def __init__(self, block: BlockSpec, th: PolicyThresholds, *, persistent_joint_occlusion: bool = False,
                 reanchor_after: int = 0) -> None:
        self.block = block
        self.th = th
        self._tracks: dict[int, TrackedBlock] = {}
        self._next_id: int = 1
        self._last_now_s: float = -np.inf
        # V3 (D4): identities of identical blocks hidden at the same time stay unverified until retirement;
        # DISPLACED tracks re-anchor after N consistent fresh observations.  Both off = V2 behaviour.
        self.persistent_joint_occlusion = bool(persistent_joint_occlusion)
        self.reanchor_after = int(reanchor_after)
        self._co_hidden: dict[int, set[int]] = {}
        self._unverified: dict[int, set[int]] = {}

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
        self._last_now_s = -np.inf
        self._co_hidden.clear()
        self._unverified.clear()

    def update(
        self,
        fits: list[BlockFit],
        sources: list[str],
        stamp_ns: int,
        now_s: float,
        *,
        side_fresh_global: bool = False,
        pose_side_visible: list[bool] | None = None,
        target_shift_gate_m: float = 0.008,
    ) -> list[TrackedBlock]:
        """Update tracks; expiry/retirement NEVER infer OBSERVED_EMPTY."""
        if not np.isfinite(now_s):
            raise ValueError("Tracker clock must be finite")
        if now_s < self._last_now_s:
            # Handle bag seek backwards cleanly
            self.reset()
        self._last_now_s = now_s

        active_ids = [
            tid for tid, tr in self._tracks.items()
            if (now_s - tr.last_seen_s) <= self.th.retire_s
        ]
        # Drop permanently retired IDs (never reuse ID!)
        self._tracks = {tid: self._tracks[tid] for tid in active_ids}
        # V3.1: drop bookkeeping keyed by retired tracks. References *to* a retired peer are kept on purpose: a
        # block co-hidden with a block that never came back may still be that block, so it stays unverified.
        live = set(active_ids)
        for book in (self._co_hidden, self._unverified):
            for tid in [t for t in book if t not in live]:
                del book[tid]

        n_t = len(active_ids)
        n_d = len(fits)
        matches: dict[int, int] = {}
        used_det: set[int] = set()
        ambig_peers_by_tid: dict[int, tuple[int, ...]] = {}

        if n_t > 0 and n_d > 0:
            cost = np.full((n_t, n_d), np.inf, dtype=np.float64)
            yaw_lim = float(np.radians(self.th.assoc_gate_yaw_deg))
            # V3.1: a block one or more layers away is a different block, even at the same (x, y, yaw)
            # (e.g. the block revealed below a lifted one). Gate on top height when both are finite.
            z_lim = 0.6 * float(self.block.height_m)
            for r, tid in enumerate(active_ids):
                tr = self._tracks[tid]
                for c, f in enumerate(fits):
                    dist = float(np.hypot(f.xy_m[0] - tr.fit.xy_m[0], f.xy_m[1] - tr.fit.xy_m[1]))
                    dyaw = symmetry_yaw_diff_rad(f.yaw_rad, tr.fit.yaw_rad, self.block.is_square)
                    dz = abs(float(f.z_top_m) - float(tr.fit.z_top_m))
                    z_ok = (not np.isfinite(dz)) or dz <= z_lim
                    if dist <= 0.025 and dyaw <= yaw_lim and z_ok:
                        cost[r, c] = dist
            # Detect near-ties after occlusion or between adjacent tracks before pruning hard swaps
            near_rows: set[int] = set()
            near_cols: dict[int, list[int]] = {}
            for r in range(n_t):
                fin = np.sort(cost[r][np.isfinite(cost[r])])
                if len(fin) >= 2 and fin[1] - fin[0] <= 0.005:
                    near_rows.add(r)
                if len(fin) >= 2 and fin[1] - fin[0] <= 0.002:
                    cost[r, :] = np.inf
            for c in range(n_d):
                col_fin = np.flatnonzero(np.isfinite(cost[:, c]))
                if len(col_fin) >= 2:
                    vals = np.sort(cost[col_fin, c])
                    if vals[1] - vals[0] <= 0.006:
                        near_cols[c] = [active_ids[int(r)] for r in col_fin]
                    if vals[1] - vals[0] <= 0.002:
                        cost[:, c] = np.inf
            finite = np.isfinite(cost)
            if np.any(finite):
                big = (min(n_t, n_d) + 1) * (1.0 + float(cost[finite].max()))
                rr, cc = linear_sum_assignment(np.where(finite, cost, big))
                for r, c in zip(rr, cc):
                    if finite[r, c]:
                        tid = active_ids[int(r)]
                        matches[tid] = int(c)
                        used_det.add(int(c))
                        peers: set[int] = set()
                        if int(r) in near_rows or int(c) in near_cols:
                            peers.update(p for p in near_cols.get(int(c), ()) if p != tid)
                            for orow, otid in enumerate(active_ids):
                                if orow != int(r) and np.isfinite(cost[orow, int(c)]):
                                    peers.add(otid)
                        if self._tracks[tid].state == "OCCLUDED":
                            for otid, otr in self._tracks.items():
                                if otid != tid and np.hypot(otr.fit.xy_m[0] - fits[int(c)].xy_m[0], otr.fit.xy_m[1] - fits[int(c)].xy_m[1]) <= 0.032:
                                    peers.add(otid)
                        if peers:
                            ambig_peers_by_tid[tid] = tuple(sorted(peers))

        if self.persistent_joint_occlusion:
            hidden_now = {tid for tid in active_ids if tid not in matches}
            for tid in hidden_now:
                self._co_hidden.setdefault(tid, set()).update(hidden_now - {tid})
            for tid in active_ids:
                if tid in matches and self._co_hidden.get(tid):
                    peers_h = self._co_hidden.pop(tid)
                    self._unverified.setdefault(tid, set()).update(peers_h)
                    for p_ in peers_h:
                        self._unverified.setdefault(p_, set()).add(tid)
                elif tid in matches:
                    self._co_hidden.pop(tid, None)

        # Update matched & occluded tracks (NEVER refresh observation_stamp_ns on miss!)
        for tid in active_ids:
            tr = self._tracks[tid]
            if tid in matches:
                det_i = matches[tid]
                new_fit = fits[det_i]
                new_src = sources[det_i]
                mds = min(self.th.mds_confirm_frames + 2, tr.mds_score + 1)
                confirmed = mds >= self.th.mds_confirm_frames
                shift_m = float(np.hypot(new_fit.xy_m[0] - tr.anchor_xy_m[0], new_fit.xy_m[1] - tr.anchor_xy_m[1]))
                shifted = shift_m >= target_shift_gate_m
                peers_tup = ambig_peers_by_tid.get(tid, ())
                if self.persistent_joint_occlusion and self._unverified.get(tid):
                    peers_tup = tuple(sorted(set(peers_tup) | self._unverified[tid]))
                id_ambig = bool(peers_tup)
                anchor = tr.anchor_xy_m
                stable_count = 0
                stable_ref = None
                if self.reanchor_after > 0:
                    # V3.1: measure consistency from where the stable run started, not frame to frame, so a block
                    # sliding steadily at < 2 mm/frame cannot re-anchor (and become commit-eligible) mid-motion.
                    ref = tr.stable_ref_xy_m if (tr.stable_count > 0 and tr.stable_ref_xy_m is not None) else tr.fit.xy_m
                    consistent = float(np.hypot(new_fit.xy_m[0] - ref[0], new_fit.xy_m[1] - ref[1])) <= 0.002
                    stable_count = (tr.stable_count + 1) if (consistent and not id_ambig and tr.state in ("MEASURED", "DISPLACED")) else 0
                    stable_ref = ref if stable_count > 0 else None
                    if shifted and stable_count >= self.reanchor_after:
                        anchor, shifted, stable_count, stable_ref = new_fit.xy_m, False, 0, None
                t_side = bool(pose_side_visible[det_i]) if pose_side_visible is not None else bool(side_fresh_global)
                state = "DISPLACED" if shifted else "MEASURED"
                snap_ok = bool(
                    confirmed and (not shifted) and (not id_ambig) and t_side and (stamp_ns >= tr.observation_stamp_ns)
                )
                self._tracks[tid] = TrackedBlock(
                    track_id=tid,
                    block_name=f"B{tid:02d}",
                    fit=new_fit,
                    source=new_src,
                    state=state,
                    mds_score=mds,
                    confirmed=confirmed,
                    observation_stamp_ns=int(stamp_ns),
                    last_seen_s=now_s,
                    anchor_xy_m=anchor,
                    target_side_fresh=t_side,
                    snapshot_commit_eligible=snap_ok,
                    identity_ambiguous=id_ambig,
                    ambiguous_peer_ids=peers_tup,
                    stable_count=stable_count,
                    stable_ref_xy_m=stable_ref,
                )
            else:
                miss_age = now_s - tr.last_seen_s
                mds = max(-1, tr.mds_score - 1)
                if miss_age <= self.th.occlusion_hold_s:
                    state = "OCCLUDED"
                else:
                    state = "STALE"
                self._tracks[tid] = replace(
                    tr,
                    state=state,
                    mds_score=mds,
                    # Critically preserve unmodified tr.observation_stamp_ns!
                    target_side_fresh=False,
                    snapshot_commit_eligible=False,
                    identity_ambiguous=False,
                    ambiguous_peer_ids=(),
                )

        # Spawn new tracks for unmatched detections
        for c, (fit, src) in enumerate(zip(fits, sources)):
            if c in used_det:
                continue
            tid = self._next_id
            self._next_id += 1
            mds = 1
            confirmed = mds >= self.th.mds_confirm_frames
            t_side = bool(pose_side_visible[c]) if pose_side_visible is not None else bool(side_fresh_global)
            self._tracks[tid] = TrackedBlock(
                track_id=tid,
                block_name=f"B{tid:02d}",
                fit=fit,
                source=src,
                state="MEASURED",
                mds_score=mds,
                confirmed=confirmed,
                observation_stamp_ns=int(stamp_ns),
                last_seen_s=now_s,
                anchor_xy_m=fit.xy_m,
                target_side_fresh=t_side,
                snapshot_commit_eligible=bool(confirmed and t_side),
                identity_ambiguous=False,
                ambiguous_peer_ids=(),
            )

        return [self._tracks[k] for k in sorted(self._tracks)]

"""V3 known-dimension explanation of unresolved height-band regions.

A region (one or more height-band components of one layer, joined through unknown space) is described by a
tri-state cell map: OCCUPIED (measured top inside the layer band), FREE (measured lower surface, or a fixed camera
sees through the volume), UNKNOWN (no depth, something above the band such as the arm, projected robot).  Measured
depth is never replaced.

An *explanation* is a set of non-overlapping 75 x 25 mm placements that covers every OCCUPIED cell, contains no FREE
cell and survives the fixed-view ray test.  Two solvers decide which blocks may be published:

* ``enumerate`` (approach A): enumerate all minimum-count exact covers over a lattice restricted to a few dominant
  angles; publish blocks present in every enumerated explanation.
* ``prove`` (approach B, default): find one plausible explanation, then for each of its blocks search for a
  *counterexample* -- any admissible explanation (any block count, dense angle set, worse score allowed) that does
  not contain that block.  STABLE = the bounded search exhausted without one (contradiction within the model);
  COUNTEREXAMPLE = found one; TIMEOUT = budget hit.  Only STABLE blocks are published.

Both are bounded; a budget exit always means abstention for the affected blocks, never a guess.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Callable

import cv2
import numpy as np

from .config import BlockSpec
from .heightmap_core import HeightmapGrid
from .proposals import BlockFit

OCC, FREE, UNK = 1, 2, 3
STABLE, COUNTEREXAMPLE, TIMEOUT = "stable", "counterexample", "timeout"


@dataclass(frozen=True)
class TilingParams:
    solver: str = "prove"              # "prove" (B) | "enumerate" (A, STABLE not a proof) | "adaptive" (A then B, sound)
    bin_m: float = 0.0025              # placement lattice step in the rotated frame
    tol_bins: int = 1                  # free/overlap shrink and coverage expand tolerance (bins)
    max_free_cells: int = 2            # FREE cells tolerated inside the shrunk rectangle
    min_obs_frac: float = 0.15         # placement must contain >= this fraction of its area as OCCUPIED
    max_blocks: int = 8
    dense_angle_step_deg: float = 3.0  # B: full sweep of [0, 90) (+ refined dominant angles)
    max_angles: int = 3                # A: dominant-angle shortlist size
    count_slack: int = 0               # A: near-optimal = min count + slack
    local_proof: bool = False          # B: local relaxation pre-check before the dense search (measured slower)
    max_nodes: int = 6000              # per search call (deterministic primary budget)
    max_explanations: int = 96         # A
    region_budget_ms: float = 250.0    # wall-clock safety net per region; the node cap normally binds first
    invariance_xy_m: float = 0.0076     # structural sameness: 3 lattice bins; position is guarded by pinning
    invariance_yaw_deg: float = 8.0     # structural: the +-2.5 mm slack admits ~6 deg rotations of the same block
    height_gate_m: float = 0.0035      # median top within this of k * block height
    window_margin_m: float = 0.080
    max_window_cells: int = 60000
    edge_bias_m: float = 0.0
    min_observed_frac: float = 0.6     # a published block must have >= this fraction of its top observed
    max_unsupported_frac: float = 0.3  # layer >= 2: max fraction of occupied cells with measured free space below
    ray_margin_m: float = 0.008        # fixed-view see-through margin
    ray_min_violations: int = 4
    ray_violation_frac: float = 0.30


@dataclass
class Placement:
    cx: float
    cy: float
    yaw: float                       # long axis, table frame (rad)
    score: float
    cov_bits: int = 0                # OCCUPIED cells covered (expanded rectangle)
    int_bits: int = 0                # window cells inside the shrunk rectangle (overlap test)
    angle_key: float = 0.0


@dataclass
class RegionResult:
    fits: list[BlockFit]
    verdicts: list[str]                      # per candidate block of the seed explanation
    unresolved_xy: list[tuple[float, float]]
    reason: str
    diag: dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------------------------------- helpers

def build_tristate(h_filled: np.ndarray, band: tuple[float, float], extra_unknown: np.ndarray | None = None,
                   extra_free: np.ndarray | None = None) -> np.ndarray:
    lo, hi = band
    fin = np.isfinite(h_filled)
    st = np.full(h_filled.shape, UNK, dtype=np.uint8)
    st[fin & (h_filled < lo)] = FREE
    st[fin & (h_filled >= lo) & (h_filled <= hi)] = OCC
    if extra_free is not None:
        st[(st == UNK) & extra_free] = FREE
    if extra_unknown is not None:
        st[extra_unknown] = UNK
    return st


def _bits(indices: np.ndarray) -> int:
    if len(indices) == 0:
        return 0
    arr = np.zeros(int(indices.max()) + 1, dtype=bool)
    arr[indices] = True
    return int.from_bytes(np.packbits(arr, bitorder="little").tobytes(), "little")


def _popcount(x: int) -> int:
    return x.bit_count()


def _rect_sum(ii: np.ndarray, i0: np.ndarray, j0: np.ndarray, a: int, b: int) -> np.ndarray:
    n0, n1 = ii.shape[0] - 1, ii.shape[1] - 1
    i1, j1 = np.clip(i0 + a, 0, n0), np.clip(j0 + b, 0, n1)
    i0c, j0c = np.clip(i0, 0, n0), np.clip(j0, 0, n1)
    return ii[i1, j1] - ii[i0c, j1] - ii[i1, j0c] + ii[i0c, j0c]


def refine_angle_deg(boundary_xy: np.ndarray, theta0_deg: float, span_deg: float = 4.0) -> float:
    """Rectilinear-boundary angle: maximise projection-histogram peakiness of boundary points (0.5 mm bins)."""
    if len(boundary_xy) < 20:
        return theta0_deg
    best, best_s = theta0_deg, -1.0
    for th in np.arange(theta0_deg - span_deg, theta0_deg + span_deg + 1e-9, 0.25):
        r = np.radians(th)
        u = boundary_xy @ np.array([np.cos(r), np.sin(r)])
        v = boundary_xy @ np.array([-np.sin(r), np.cos(r)])
        hu = np.bincount(np.floor((u - u.min()) / 0.0005).astype(np.int64))
        hv = np.bincount(np.floor((v - v.min()) / 0.0005).astype(np.int64))
        s = float((hu.astype(np.float64) ** 2).sum() + (hv.astype(np.float64) ** 2).sum())
        if s > best_s:
            best_s, best = s, float(th)
    return best % 90.0


def dominant_angles_deg(occ_mask: np.ndarray, boundary_xy: np.ndarray, k: int,
                        extra_deg: tuple[float, ...] = ()) -> list[float]:
    """Shortlist of boundary orientations mod 90 deg, each refined on raw boundary points."""
    m = cv2.GaussianBlur(occ_mask.astype(np.float32), (5, 5), 1.0)
    gx = cv2.Sobel(m, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(m, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    sel = mag > 0.05
    raw: list[float] = []
    if np.count_nonzero(sel) >= 8:
        # grid rows are +y and cols +x, so atan2(gy, gx) is already in table orientation
        ang = np.degrees(np.arctan2(gy[sel], gx[sel])) % 90.0
        hist, _ = np.histogram(ang, bins=45, range=(0.0, 90.0), weights=mag[sel])
        hs = np.convolve(np.concatenate([hist[-1:], hist, hist[:1]]), [1, 2, 1], mode="valid")
        for idx in np.argsort(-hs):
            if hs[idx] < 0.3 * hs.max() or len(raw) >= k:
                break
            raw.append((idx + 0.5) * 2.0)
    raw.extend(a % 90.0 for a in extra_deg)
    out: list[float] = []
    for a in raw:
        r = refine_angle_deg(boundary_xy, a)
        if all(min(abs(r - b), 90 - abs(r - b)) >= 2.0 for b in out):
            out.append(r)
    return out


# ----------------------------------------------------------------------------------------------- view rays

@dataclass
class ViewDepth:
    """One calibrated depth view for signed-depth see-through tests (measured depth only, never inpainted)."""

    name: str
    K: np.ndarray
    T_cam_table: np.ndarray       # 4x4, table -> camera
    depth_m: np.ndarray           # float32 metres, 0/NaN invalid
    margin_m: float = 0.006
    min_d: float = 0.15
    max_d: float = 1.8

    def see_through(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(valid, through) per table-frame point: through = the measured surface is behind the point."""
        R, t = self.T_cam_table[:3, :3], self.T_cam_table[:3, 3]
        pc = pts @ R.T + t
        z = pc[:, 2]
        h, w = self.depth_m.shape
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.round(self.K[0, 0] * pc[:, 0] / z + self.K[0, 2]).astype(np.int64)
            v = np.round(self.K[1, 1] * pc[:, 1] / z + self.K[1, 2]).astype(np.int64)
        inb = (z > self.min_d) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        meas = np.full(len(pts), np.nan, np.float32)
        meas[inb] = self.depth_m[v[inb], u[inb]]
        valid = inb & np.isfinite(meas) & (meas > self.min_d) & (meas < self.max_d)
        through = valid & (meas > z + self.margin_m)
        return valid, through


def carve_free(xs: np.ndarray, ys: np.ndarray, z: float, views: list[ViewDepth]) -> np.ndarray:
    """Cells whose layer mid-height point is seen through by at least one view (measured free space)."""
    free = np.zeros(len(xs), dtype=bool)
    if not views or len(xs) == 0:
        return free
    pts = np.stack([xs, ys, np.full(len(xs), z)], axis=1)
    for vw in views:
        _, thr = vw.see_through(pts)
        free |= thr
    return free


# ----------------------------------------------------------------------------------------------- region model

class RegionModel:
    """Tri-state window, admissible placement generation and the two solvers."""

    def __init__(self, block: BlockSpec, p: TilingParams, state: np.ndarray, must: np.ndarray,
                 xw: np.ndarray, yw: np.ndarray, cell_m: float, layer: int,
                 ray_check: Callable[[np.ndarray, np.ndarray, np.ndarray, float, float], np.ndarray] | None = None) -> None:
        self.block, self.p = block, p
        self.state = state.ravel()
        self.must = must.ravel()
        self.x, self.y = xw.ravel(), yw.ravel()
        self.cell_m = cell_m
        self.layer = layer
        self.ray_check = ray_check
        self.n_must = int(np.count_nonzero(self.must))
        self.must_idx = np.flatnonzero(self.must)
        self.int_idx = np.flatnonzero(self.state != FREE)   # overlap universe (FREE cells never interior)
        self.full = (1 << self.n_must) - 1
        self.stats: dict[str, Any] = {"placements": 0, "ray_rejected": 0, "nodes": 0, "angles": 0}
        self.last_truncated = False
        self._cache: dict[tuple[float, bool], tuple[list[Placement], np.ndarray, np.ndarray]] = {}
        self._mats: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    _ANGLE_CHUNK = 6

    def placements(self, angles_deg: list[float], nms: bool = False, deadline: float | None = None) -> list[Placement]:
        """Placements for the given angles. V3.1: generated in chunks with a deadline check between them (this
        used to run unbounded, e.g. 218 ms on one flush triple). ``self.last_truncated`` is True when the
        deadline cut the set short; callers must then treat the search as a TIMEOUT, never as a proof."""
        self.last_truncated = False
        todo = [a for a in angles_deg if round(a, 3) not in self._cache]
        for i in range(0, len(todo), self._ANGLE_CHUNK):
            if deadline is not None and time.perf_counter() > deadline:
                self.last_truncated = True
                break
            self._cache.update(self._placements_batch(todo[i:i + self._ANGLE_CHUNK]))
        out: list[Placement] = []
        covs, inns = [], []
        for a in angles_deg:
            hit = self._cache.get(round(a, 3))
            if hit is None:
                continue
            pl, cv_, in_ = hit
            if pl:
                out.extend(pl)
                covs.append(cv_)
                inns.append(in_)
        if out:
            self._mats[id(out)] = (np.concatenate(covs, 0), np.concatenate(inns, 0))
        return out

    def placements_local(self, angles_deg: list[float], restrict: np.ndarray) -> list[Placement]:
        """Uncached dense placements touching the given must cells (local relaxation for stability proofs)."""
        out, covs, inns = [], [], []
        for pl, cv_, in_ in self._placements_batch(angles_deg, restrict=restrict).values():
            if pl:
                out.extend(pl); covs.append(cv_); inns.append(in_)
        if out:
            self._mats[id(out)] = (np.concatenate(covs, 0), np.concatenate(inns, 0))
        return out

    def subset(self, places: list[Placement], keep: np.ndarray) -> list[Placement]:
        C, I = self._mats[id(places)]
        sub = [q for q, k in zip(places, keep) if k]
        self._mats[id(sub)] = (C[keep], I[keep])
        return sub

    def _placements_batch(self, angles_deg: list[float], restrict: np.ndarray | None = None
                          ) -> dict[float, tuple[list[Placement], np.ndarray, np.ndarray]]:
        """All lattice placements for many angles at once (one scatter-add, one cumsum, vectorised rect sums)."""
        p, blk = self.p, self.block
        nA = len(angles_deg)
        self.stats["angles"] += nA
        th = np.radians(np.asarray(angles_deg, dtype=np.float64))[:, None]
        c, s = np.cos(th), np.sin(th)
        u = self.x[None] * c + self.y[None] * s                     # (A, N)
        v = -self.x[None] * s + self.y[None] * c
        b = p.bin_m
        u0 = u.min(axis=1, keepdims=True) - b
        v0 = v.min(axis=1, keepdims=True) - b
        iu = np.floor((u - u0) / b).astype(np.int32)
        iv = np.floor((v - v0) / b).astype(np.int32)
        nu, nv = int(iu.max()) + 2, int(iv.max()) + 2
        base = (np.arange(nA, dtype=np.int64)[:, None] * nu + iu) * nv + iv
        tot = nA * nu * nv
        cnt_o = np.bincount(base[:, self.must].ravel(), minlength=tot).reshape(nA, nu, nv)
        cnt_f = np.bincount(base[:, self.state == FREE].ravel(), minlength=tot).reshape(nA, nu, nv)
        ii_o = np.zeros((nA, nu + 1, nv + 1)); ii_o[:, 1:, 1:] = cnt_o.cumsum(1).cumsum(2)
        ii_f = np.zeros((nA, nu + 1, nv + 1)); ii_f[:, 1:, 1:] = cnt_f.cumsum(1).cumsum(2)
        if restrict is not None:      # keep only placements whose coverage rectangle touches a restricted must cell
            rmask = np.zeros(len(self.must), bool)
            rmask[self.must_idx[restrict]] = True
            cnt_r = np.bincount(base[:, rmask].ravel(), minlength=tot).reshape(nA, nu, nv)
            ii_r = np.zeros((nA, nu + 1, nv + 1)); ii_r[:, 1:, 1:] = cnt_r.cumsum(1).cumsum(2)
        La, Wa = int(round(blk.major_m / b)), int(round(blk.minor_m / b))
        area_cells = blk.major_m * blk.minor_m / (self.cell_m ** 2)
        t = p.tol_bins
        mu, mv = iu[:, self.must], iv[:, self.must]

        rows = []   # (angle_idx, i0, j0, A, B, score)
        for long_u in (True, False):
            A, B = (La, Wa) if long_u else (Wa, La)
            I = np.arange(int(mu.min()) - A + 1, int(mu.max()) + 1)
            J = np.arange(int(mv.min()) - B + 1, int(mv.max()) + 1)
            # separable index form: rectangle sums for all (I, J) per angle
            i1, j1 = np.clip(I + A, 0, nu), np.clip(J + B, 0, nv)
            i0c, j0c = np.clip(I, 0, nu), np.clip(J, 0, nv)
            ti0, tj0 = np.clip(I + t, 0, nu), np.clip(J + t, 0, nv)
            ti1, tj1 = np.clip(I + A - t, 0, nu), np.clip(J + B - t, 0, nv)
            def box(ii, a0, a1, b0, b1):
                return (ii[:, a1][:, :, b1] - ii[:, a0][:, :, b1] - ii[:, a1][:, :, b0] + ii[:, a0][:, :, b0])
            f_in = box(ii_f, ti0, ti1, tj0, tj1)
            f_ex = box(ii_f, i0c, i1, j0c, j1)
            o_ex = box(ii_o, i0c, i1, j0c, j1)
            valid = (f_in <= p.max_free_cells) & (o_ex >= p.min_obs_frac * area_cells)
            if restrict is not None:
                ei0, ej0 = np.clip(I - t, 0, nu), np.clip(J - t, 0, nv)
                ei1, ej1 = np.clip(I + A + t, 0, nu), np.clip(J + B + t, 0, nv)
                valid &= box(ii_r, ei0, ei1, ej0, ej1) > 0
            ai, ii_, jj_ = np.nonzero(valid)
            sc = (o_ex - 3.0 * f_ex)[ai, ii_, jj_]
            rows.append(np.stack([ai, I[ii_], J[jj_], np.full(len(ai), A), np.full(len(ai), B), sc], 1))
        R = np.concatenate(rows, 0) if rows else np.zeros((0, 6))
        out: dict[float, tuple[list[Placement], np.ndarray, np.ndarray]] = {
            round(a, 3): ([], np.zeros((0, len(self.must_idx)), bool), np.zeros((0, len(self.int_idx)), bool))
            for a in angles_deg}
        if len(R) == 0:
            return out
        ai = R[:, 0].astype(int)
        i0, j0, A, B = (R[:, k].astype(np.int64) for k in range(1, 5))
        thv = th[ai, 0]
        cu = u0[ai, 0] + (i0 + A / 2.0) * b
        cv_ = v0[ai, 0] + (j0 + B / 2.0) * b
        cxp = cu * np.cos(thv) - cv_ * np.sin(thv)
        cyp = cu * np.sin(thv) + cv_ * np.cos(thv)
        yaw = np.where(A > B, thv, thv + np.pi / 2)
        ok = np.ones(len(R), bool)
        if self.ray_check is not None:
            ok = self.ray_check(cxp, cyp, yaw, blk.major_m, blk.minor_m)
            self.stats["ray_rejected"] += int(np.count_nonzero(~ok))
        keep = np.flatnonzero(ok)
        if len(keep) == 0:
            return out
        # bitsets: coverage over must cells (expanded), interior over non-FREE cells (shrunk)
        mi, mj = iu[ai[keep]][:, self.must_idx], iv[ai[keep]][:, self.must_idx]
        k0, l0 = i0[keep, None], j0[keep, None]
        AA, BB = A[keep, None], B[keep, None]
        cov = (mi >= k0 - t) & (mi < k0 + AA + t) & (mj >= l0 - t) & (mj < l0 + BB + t)
        ui, uj = iu[ai[keep]][:, self.int_idx], iv[ai[keep]][:, self.int_idx]
        inn = (ui >= k0 + t) & (ui < k0 + AA - t) & (uj >= l0 + t) & (uj < l0 + BB - t)
        cov_b = np.packbits(cov, axis=1, bitorder="little")
        int_b = np.packbits(inn, axis=1, bitorder="little")
        per: dict[int, list[int]] = {}
        for r, k in enumerate(keep):
            per.setdefault(int(ai[k]), []).append(r)
        for a_i, rr in per.items():
            rr_arr = np.asarray(rr)
            pl = [Placement(float(cxp[keep[r]]), float(cyp[keep[r]]),
                            float((yaw[keep[r]] + np.pi / 2) % np.pi - np.pi / 2), float(R[keep[r], 5]),
                            int.from_bytes(cov_b[r].tobytes(), "little"), int.from_bytes(int_b[r].tobytes(), "little"),
                            float(angles_deg[a_i])) for r in rr]
            out[round(angles_deg[a_i], 3)] = (pl, cov[rr_arr], inn[rr_arr])
        self.stats["placements"] += len(keep)
        return out

    # ---------------------------------------------------------------- exact cover core
    def _cover(self, places: list[Placement], *, first_only: bool, max_count: int, deadline: float,
               bound_by_best: bool, target: np.ndarray | None = None) -> tuple[list[tuple[int, ...]], bool]:
        """Exact cover DFS. Cells are grouped into classes with identical covering sets (vectorised);
        options are counted with placement bitsets; conflict rows are built lazily from the interior matrix."""
        if not places:
            return [], False
        C, I = self._mats[id(places)]
        tcols = np.arange(C.shape[1]) if target is None else np.asarray(target)
        full = self.full if target is None else _bits(tcols)
        n_target = len(tcols)
        if not np.all(C[:, tcols].any(axis=0)):
            return [], False
        # cell classes: identical columns of C (restricted to the target cells)
        colbytes = np.packbits(C[:, tcols].T, axis=1, bitorder="little")
        uniq, inv = np.unique(colbytes, axis=0, return_inverse=True)
        inv = inv.ravel()
        cls = []
        for ci in range(len(uniq)):
            cells = tcols[np.flatnonzero(inv == ci)]
            cls.append((int.from_bytes(uniq[ci].tobytes(), "little"), _bits(cells)))
        cov_bits = [q.cov_bits for q in places]
        max_cov = int(C[:, tcols].sum(axis=1).max())
        score = np.array([q.score for q in places])
        If = I.astype(np.float32)
        conflict_cache: dict[int, int] = {}

        def conflicts(k: int) -> int:
            r = conflict_cache.get(k)
            if r is None:
                row = (If @ If[k]) > 0.5
                row[k] = True
                r = int.from_bytes(np.packbits(row, bitorder="little").tobytes(), "little")
                conflict_cache[k] = r
            return r

        sols: list[tuple[int, ...]] = []
        best = [max_count]
        nodes = [0]
        timed_out = [False]

        def dfs(covered: int, banned: int, chosen: tuple[int, ...]) -> bool:
            nodes[0] += 1
            if nodes[0] > self.p.max_nodes or (nodes[0] & 63 == 0 and time.perf_counter() > deadline):
                timed_out[0] = True
                return True
            if covered & full == full:
                sols.append(tuple(sorted(chosen)))
                if bound_by_best:
                    best[0] = min(best[0], len(chosen))
                return first_only
            remaining = n_target - (covered & full).bit_count()
            lim = best[0] + (self.p.count_slack if bound_by_best else 0)
            if len(chosen) + (-(-remaining // max_cov)) > lim:
                return False
            best_sig, best_n = 0, None
            for sig, cbits in cls:
                if cbits & ~covered == 0:
                    continue
                o = sig & ~banned
                n = o.bit_count()
                if best_n is None or n < best_n:
                    best_sig, best_n = o, n
                    if n <= 1:
                        break
            if not best_n:
                return False
            opts = []
            o = best_sig
            while o:
                low = o & -o
                opts.append(low.bit_length() - 1)
                o ^= low
            opts.sort(key=lambda k: -score[k])
            for k in opts:
                if dfs(covered | cov_bits[k], banned | conflicts(k), chosen + (k,)):
                    return True
            return False

        dfs(0, 0, ())
        self.stats["nodes"] += nodes[0]
        return sols, timed_out[0]

    # ---------------------------------------------------------------- approach A
    def solve_enumerate(self, angles: list[float], deadline: float) -> tuple[list[Placement], list[str], dict]:
        """Minimum-count exact covers only. Its COUNTEREXAMPLE verdicts are sound (a real alternative cover was
        found); its STABLE verdicts are NOT proofs, because covers with more blocks are never searched (under
        occlusion an extra hidden block can explain the same cells). See ``solve_adaptive``."""
        places = self.placements(angles, deadline=deadline)
        if self.last_truncated:
            return [], [], {"abort": "timeout"}
        sols, to = self._cover(places, first_only=False, max_count=self.p.max_blocks, deadline=deadline,
                               bound_by_best=True)
        if to:
            return [], [], {"abort": "timeout"}
        if not sols:
            return [], [], {"abort": "no_explanation"}
        n_min = min(len(s) for s in sols)
        exps = [[places[k] for k in s] for s in sorted(set(sols)) if len(s) <= n_min + self.p.count_slack]
        if len(exps) > self.p.max_explanations:
            return [], [], {"abort": "too_many_explanations", "explanations": len(exps)}
        seed = max(exps, key=lambda e: sum(q.score for q in e))
        verdicts = [STABLE if all(any(self.same(q, r) for r in e) for e in exps) else COUNTEREXAMPLE for q in seed]
        return seed, verdicts, {"explanations": len(exps), "min_blocks": n_min}

    # ---------------------------------------------------------------- approach B
    def solve_adaptive(self, shortlist: list[float], deadline: float) -> tuple[list[Placement], list[str], dict]:
        """V3.1 sound adaptive solver: minimum-count enumeration first, then counterexample search.

        Only enumeration's COUNTEREXAMPLE verdicts are kept (sound). Every block enumeration calls STABLE is
        still refuted by ``solve_prove``'s search over any block count, so the published set is never larger
        than ``prove`` alone would publish within the same budget. If enumeration aborts (timeout, too many
        explanations, none), the full counterexample search decides.
        """
        seed, verdicts, d = self.solve_enumerate(shortlist, deadline)
        if not seed:
            s2, v2, d2 = self.solve_prove(shortlist, deadline)
            d2["enumerate_abort"] = d.get("abort")
            return s2, v2, d2
        prior = [COUNTEREXAMPLE if v == COUNTEREXAMPLE else None for v in verdicts]
        s2, v2, d2 = self.solve_prove(shortlist, deadline, prior=(seed, prior))
        d2["enumerate_explanations"] = d.get("explanations")
        d2["enumerate_pruned"] = sum(v is not None for v in prior)
        return s2, v2, d2

    def solve_prove(self, shortlist: list[float], deadline: float,
                    prior: tuple[list[Placement], list[str | None]] | None = None) -> tuple[list[Placement], list[str], dict]:
        """Seed from the shortlist lattice; refute each seed block with a counterexample, searching the cheap
        shortlist set first and escalating to the dense angle set only to *confirm* blocks that survive.

        ``prior`` = (seed, verdicts) from a sound pre-pass: only COUNTEREXAMPLE/TIMEOUT entries may be set; None
        entries are refuted here. A truncated placement set (deadline) yields TIMEOUT, never STABLE."""
        seed_places = self.placements(shortlist, deadline=deadline)
        dense_angles = sorted(set([round(a, 3) for a in np.arange(0.0, 90.0, self.p.dense_angle_step_deg)]
                                  + [round(a, 3) for a in shortlist]))
        dense: list[Placement] | None = None
        dense_truncated = False
        if prior is not None:
            seed = list(prior[0])
            verdicts: list[str | None] = [v if v in (COUNTEREXAMPLE, TIMEOUT) else None for v in prior[1]]
        else:
            sols, to = self._cover(seed_places, first_only=True, max_count=self.p.max_blocks, deadline=deadline,
                                   bound_by_best=False)
            if not sols:
                dense = self.placements(dense_angles, deadline=deadline)
                dense_truncated = self.last_truncated
                sols, to = self._cover(dense, first_only=True, max_count=self.p.max_blocks, deadline=deadline,
                                       bound_by_best=False)
                if not sols:
                    return [], [], {"abort": "timeout" if (to or dense_truncated) else "no_explanation"}
                seed = [dense[k] for k in sols[0]]
            else:
                seed = [seed_places[k] for k in sols[0]]
            verdicts = [None] * len(seed)
        # V3.1: when the shortlist-stage pool already WAS the dense set, a refutation that found nothing there
        # is exactly the dense refutation; re-running it (as before) only cost time.
        shortlist_pool_is_dense = dense is not None and not dense_truncated
        escalated = 0

        def refute(i: int, pool: list[Placement]) -> str | None:
            q = seed[i]
            # V3.1 note: reusing per-pool search structures with a banned mask instead of this subset was tried and
            # REVERTED: full-pool cell classes are finer, each DFS node scans more of them, and on the held-out
            # split more searches hit the node cap (prove: 134 vs 169 published, p50 +18 ms on the WSL host).
            allowed = self.subset(pool, np.array([not self.same(r, q) for r in pool]))
            cex, to_ = self._cover(allowed, first_only=True, max_count=self.p.max_blocks, deadline=deadline,
                                   bound_by_best=False)
            if cex:
                ce = [allowed[k] for k in cex[0]]
                self.stats.setdefault("counterexamples", []).append(
                    [(round(r.cx * 1000, 1), round(r.cy * 1000, 1), round(float(np.degrees(r.yaw)), 1)) for r in ce])
                for j, q2 in enumerate(seed):          # seed blocks absent from the counterexample fall too
                    if verdicts[j] is None and not any(self.same(q2, r) for r in ce):
                        verdicts[j] = COUNTEREXAMPLE
                return COUNTEREXAMPLE
            return TIMEOUT if to_ else None

        for stage in ("shortlist", "dense"):
            for i in range(len(seed)):
                if verdicts[i] is not None:
                    continue
                if time.perf_counter() > deadline:
                    verdicts[i] = TIMEOUT
                    continue
                if stage == "shortlist":
                    v = refute(i, seed_places if dense is None else dense)
                    if v is not None:
                        verdicts[i] = v
                    elif shortlist_pool_is_dense:
                        verdicts[i] = STABLE      # the complete dense search already found no counterexample
                else:
                    # sound relaxation: if no non-overlapping alternative set can even cover this block's own
                    # occupied cells, no full counterexample exists -> STABLE without the global dense search
                    q = seed[i]
                    if self.p.local_proof:
                        tgt = np.flatnonzero(np.unpackbits(np.frombuffer(q.cov_bits.to_bytes((self.n_must + 7) // 8, "little"),
                                                                         np.uint8), bitorder="little")[: self.n_must])
                        local = self.placements_local(dense_angles, tgt)
                        if local:
                            local = self.subset(local, np.array([not self.same(r, q) for r in local], bool))
                        lc, lto = (self._cover(local, first_only=True, max_count=self.p.max_blocks, deadline=deadline,
                                               bound_by_best=False, target=tgt) if local else ([], False))
                        if not lc and not lto:
                            verdicts[i] = STABLE
                            self.stats["local_proofs"] = self.stats.get("local_proofs", 0) + 1
                            continue
                    if dense is None:
                        dense = self.placements(dense_angles, deadline=deadline)
                        dense_truncated = self.last_truncated
                    if dense_truncated or not dense:
                        verdicts[i] = TIMEOUT     # an incomplete alternative set proves nothing
                        continue
                    escalated += 1
                    v = refute(i, dense)
                    verdicts[i] = v if v is not None else STABLE
        return seed, [str(v) for v in verdicts], {
            "dense_placements": len(dense) if dense is not None else 0, "escalated": escalated,
            "seed": [(round(q.cx * 1000, 1), round(q.cy * 1000, 1), round(float(np.degrees(q.yaw)), 1)) for q in seed]}

    def _same_in_frame_of(self, a: Placement, b: Placement) -> bool:
        d = abs((a.yaw - b.yaw + np.pi / 2) % np.pi - np.pi / 2)
        if d > np.radians(self.p.invariance_yaw_deg):
            return False
        dx, dy = b.cx - a.cx, b.cy - a.cy
        ca, sa = np.cos(a.yaw), np.sin(a.yaw)
        return abs(dx * ca + dy * sa) <= self.p.invariance_xy_m and abs(-dx * sa + dy * ca) <= self.p.invariance_xy_m

    def same(self, a: Placement, b: Placement) -> bool:
        """Same physical hypothesis at the model's resolution: yaw within the angle step and the centre within
        the lattice slop along each block axis (the joint refinement later decides the published pose).

        V3.1: symmetric. The displacement used to be measured only in ``a``'s frame, so ``same(q, r)`` and
        ``same(r, q)`` could disagree for rotated placements near the 7.6 mm bound (e.g. q=(0,0,0deg),
        r=(7.9,-3.6 mm,7deg)). Pool exclusion and witness checks used opposite argument orders, so a genuine
        counterexample could be excluded as "the same block". Requiring both directions can only make the
        solver abstain more, never certify more.
        """
        return self._same_in_frame_of(a, b) and self._same_in_frame_of(b, a)


# ----------------------------------------------------------------------------------------------- refinement

def _hard_long_sides(q: Placement, hard_xy: np.ndarray | None, block: BlockSpec) -> list[int]:
    """Long sides (-1/+1 along the short axis) bordered by FREE cells along most of their length."""
    if hard_xy is None or len(hard_xy) == 0:
        return []
    L, W = block.major_m, block.minor_m
    ux = np.array([np.cos(q.yaw), np.sin(q.yaw)])
    uy = np.array([-np.sin(q.yaw), np.cos(q.yaw)])
    r = hard_xy - (q.cx, q.cy)
    dt, dn = r @ ux, r @ uy
    need = max(3, int(0.25 * (L / 2) / 0.002))
    out = []
    for sgn in (-1, 1):
        sel = (sgn * dn > W / 2 - 0.002) & (sgn * dn < W / 2 + 0.006) & (np.abs(dt) <= L / 2 - 0.003)
        if int(np.count_nonzero(sel)) >= need:
            out.append(sgn)
    return out


def _refine_yaw(q: Placement, pts_band: np.ndarray, block: BlockSpec, hard_xy: np.ndarray | None = None) -> float:
    """Yaw from the straightest long edge (outermost point per 3 mm bin, line fit); +-3.5 deg clamp.

    V3.1: if any long edge is FREE-bordered, only those edges are used. A long edge shared with a neighbour
    also collects the neighbour's top (the band extends 4 mm past the edge), which biases the line fit."""
    L, W = block.major_m, block.minor_m
    ux = np.array([np.cos(q.yaw), np.sin(q.yaw)])
    uy = np.array([-np.sin(q.yaw), np.cos(q.yaw)])
    rel = pts_band[:, :2] - (q.cx, q.cy)
    a, b = rel @ ux, rel @ uy
    inside = (np.abs(a) <= L / 2 - 0.006) & (np.abs(b) <= W / 2 + 0.004)
    est, wts = [], []
    # V3.1 note: restricting this to FREE-bordered sides only (_hard_long_sides) was tried and REGRESSED the L
    # layout (1.30 mm vs < 1.0 mm): a side can pass the 25 %-free test while still touching the neighbour along
    # part of its length. A proper fix needs refine_group's segment-wise masking; until then both sides are used.
    for sgn in (-1, 1):
        sel = inside & (sgn * b > W / 2 - 0.004)
        if np.count_nonzero(sel) < 15:
            continue
        aa, bb = a[sel], b[sel]
        kb = np.floor((aa + L / 2) / 0.003).astype(int)
        order = np.lexsort((-sgn * bb, kb))
        first = np.ones(len(order), bool)
        first[1:] = kb[order][1:] != kb[order][:-1]
        xs, ys = aa[order][first], bb[order][first]
        if len(xs) >= 8:
            coef, res, *_ = np.polyfit(xs, ys, 1, full=True)
            est.append(float(np.arctan(coef[0])))
            wts.append(1.0 / (1e-8 + float(res[0]) / len(xs) if len(res) else 1e-8))
    if not est:
        return q.yaw
    d = float(np.average(est, weights=wts))
    return q.yaw + float(np.clip(d, -np.radians(3.5), np.radians(3.5)))


def refine_group(placements: list[Placement], pts_band: np.ndarray, hard_xy: np.ndarray, block: BlockSpec,
                 edge_bias_m: float = 0.0, slop_m: float = 0.006, occ_xy: np.ndarray | None = None,
                 cell_m: float = 0.002, min_observed_frac: float = 0.6,
                 trusted: list[bool] | None = None,
                 contacts_out: list | None = None) -> list[tuple[BlockFit, bool, dict]]:
    """Joint sub-cell refinement of stable blocks.

    Equations: each FREE-bordered (hard) edge measured from raw points; each pair of parallel facing edges in
    contact (within ``slop_m``) is constrained to touch.  A block is *pinned* only if both of its axes are
    determined by hard edges, directly or through a chain of contacts.  Unpinned blocks are reported but must not
    be published as poses.
    """
    L, W = block.major_m, block.minor_m
    n = len(placements)
    yaws = [_refine_yaw(q, pts_band, block) for q in placements]
    cen0 = np.array([[q.cx, q.cy] for q in placements], dtype=np.float64)
    # edges: (block, normal(2), half-extent, tangent(2), tangent half-length)
    edges = []
    for k in range(n):
        ux = np.array([np.cos(yaws[k]), np.sin(yaws[k])])
        uy = np.array([-np.sin(yaws[k]), np.cos(yaws[k])])
        for sgn in (-1, 1):
            edges.append((k, sgn * ux, L / 2, uy, W / 2, "a"))
            edges.append((k, sgn * uy, W / 2, ux, L / 2, "b"))
    rows, rhs, wts, info = [], [], [], []
    hard_e: dict[tuple[int, str], list[float]] = {}     # (block, axis) -> measured edge offsets from cen0
    pinned_dirs: list[list[np.ndarray]] = [[] for _ in range(n)]
    hard_pts = hard_xy if len(hard_xy) else np.zeros((0, 2))
    for (k, nrm, h, tng, th, ax) in edges:
        c = cen0[k]
        # hard edge: FREE cells just outside this edge along most of its length
        if len(hard_pts):
            r = hard_pts - c
            dn, dt = r @ nrm, r @ tng
            sel = (dn > h - 0.002) & (dn < h + 0.006) & (np.abs(dt) <= th - 0.003)
            hard = int(np.count_nonzero(sel)) >= max(3, int(0.25 * th / 0.002))
        else:
            hard = False
        if not hard:
            continue
        # measure only along the FREE-bordered segment of this edge (a partly shared edge must not see the
        # neighbour's top), and only points on this side of the edge band
        hard_t = np.sort(dt[sel])
        rp = pts_band[:, :2] - c
        pn, pt = rp @ nrm, rp @ tng
        near = (pn > h - 0.005) & (pn < h + 0.005) & (np.abs(pt) <= th - 0.003)
        if np.any(near):
            k_ = np.searchsorted(hard_t, pt[near])
            gap_l = np.abs(pt[near] - hard_t[np.clip(k_ - 1, 0, len(hard_t) - 1)])
            gap_r = np.abs(hard_t[np.clip(k_, 0, len(hard_t) - 1)] - pt[near])
            near[np.flatnonzero(near)] = np.minimum(gap_l, gap_r) <= 0.0015
        if np.count_nonzero(near) < 6:
            continue
        e = float(np.percentile(pn[near], 98.0)) + edge_bias_m      # measured edge offset from lattice centre
        row = np.zeros(2 * n)
        row[2 * k: 2 * k + 2] = nrm
        rows.append(row); rhs.append(e - h); wts.append(1.0)
        pinned_dirs[k].append(nrm)
        info.append(("hard", k, ax))
        hard_e.setdefault((k, ax), []).append(e)
    contacts = []
    all_contacts = []
    for i in range(len(edges)):
        ki, ni, hi_, ti, thi, _ = edges[i]
        for j in range(i + 1, len(edges)):
            kj, nj, hj, tj, thj, _ = edges[j]
            if ki == kj or float(ni @ nj) > -0.995:
                continue
            gap = float(ni @ (cen0[kj] - cen0[ki])) - hi_ - hj
            if abs(gap) > slop_m:
                continue
            ov = min(float(ti @ cen0[ki]) + thi, float(ti @ cen0[kj]) + thj) - max(float(ti @ cen0[ki]) - thi, float(ti @ cen0[kj]) - thj)
            if ov < 0.006:
                continue
            all_contacts.append((ki, kj, ni))
            if trusted is not None and not (trusted[ki] and trusted[kj]):
                continue
            row = np.zeros(2 * n)
            row[2 * kj: 2 * kj + 2] = ni
            row[2 * ki: 2 * ki + 2] = -ni
            rows.append(row); rhs.append(hi_ + hj - float(ni @ (cen0[kj] - cen0[ki]))); wts.append(0.7)
            contacts.append((ki, kj, ni))
    # propagate pinning through contacts (direction-wise)
    changed = True
    while changed:
        changed = False
        for ki, kj, nrm in contacts:
            for a_, b_ in ((ki, kj), (kj, ki)):
                if any(abs(float(d @ nrm)) > 0.99 for d in pinned_dirs[a_]) and not any(abs(float(d @ nrm)) > 0.99 for d in pinned_dirs[b_]):
                    pinned_dirs[b_].append(nrm)
                    changed = True
    # weak prior keeps the system well posed
    for k in range(2 * n):
        row = np.zeros(2 * n); row[k] = 1.0
        rows.append(row); rhs.append(0.0); wts.append(0.02)
    A = np.array(rows) * np.array(wts)[:, None]
    y = np.array(rhs) * np.array(wts)
    delta = np.linalg.lstsq(A, y, rcond=None)[0].reshape(n, 2)
    delta = np.clip(delta, -slop_m, slop_m)
    out = []
    for k, q in enumerate(placements):
        ux = np.array([np.cos(yaws[k]), np.sin(yaws[k])])
        uy = np.array([-np.sin(yaws[k]), np.cos(yaws[k])])
        pin_a = any(abs(float(d @ ux)) > 0.99 for d in pinned_dirs[k])
        pin_b = any(abs(float(d @ uy)) > 0.99 for d in pinned_dirs[k])
        c2 = cen0[k] + delta[k]
        rel = pts_band[:, :2] - c2
        inside = (np.abs(rel @ ux) <= L / 2 + 0.002) & (np.abs(rel @ uy) <= W / 2 + 0.002)
        pts_in = pts_band[inside]
        z_top = float(np.median(pts_in[:, 2])) if len(pts_in) else float("nan")
        yaw_f = (yaws[k] + np.pi / 2) % np.pi - np.pi / 2
        # V3.1 note: a measured footprint (sum of the two opposite 98th-percentile edge offsets) was tried and
        # REVERTED: on the 168-scene held-out split it read widths of 30.3-30.7 mm for 25 mm blocks (edge
        # outliers add up in the sum while they cancel in the centre). dims stay nominal for V3 poses, so the
        # size gate is not evidence for them; the raw extent is kept in diagnostics for a future robust estimator.
        ea, eb = hard_e.get((k, "a"), []), hard_e.get((k, "b"), [])
        fit = BlockFit(
            xy_m=(float(c2[0]), float(c2[1])), z_center_m=z_top - block.height_m / 2, z_top_m=z_top,
            yaw_rad=float(yaw_f), yaw_deg_mod=float(np.degrees(yaw_f) % block.yaw_symmetry_deg),
            dims_mm=(round(L * 1000, 1), round(W * 1000, 1)),
            fill=float(len(pts_in)) / max(1.0, L * W / 4e-6), size_err=0.0, n_points=int(len(pts_in)),
            points_xyz=pts_in, covariance_xy_m2=None,
        )
        obs = 1.0
        if occ_xy is not None and len(occ_xy):
            ro = occ_xy - c2
            n_in = int(np.count_nonzero((np.abs(ro @ ux) <= L / 2) & (np.abs(ro @ uy) <= W / 2)))
            obs = n_in / (L * W / cell_m ** 2)
        ok = bool(pin_a and pin_b and obs >= min_observed_frac and (trusted is None or trusted[k]))
        out.append((fit, ok, {"pin_long": pin_a, "pin_short": pin_b, "observed_frac": round(obs, 3),
                              "shift_mm": [round(1000 * float(v), 2) for v in delta[k]],
                              "raw_extent_mm": [round(1000 * float(sum(ea)), 1) if len(ea) == 2 else None,
                                                round(1000 * float(sum(eb)), 1) if len(eb) == 2 else None]}))
    # V3.1: contacts go to the caller's list (was a function attribute shared by every caller/thread)
    if contacts_out is not None:
        contacts_out.extend((ki, kj, (float(n_[0]), float(n_[1]))) for ki, kj, n_ in all_contacts)
    return out


def _slab_guard(refined, contacts, pts_band, block: BlockSpec, cue_fn, diag) -> set[int]:
    """Blocks to withhold because their contact-connected cluster forms one rectangle (a block-shaped single
    object would explain it equally well) and the contact seam shows no depth/RGB/side cue."""
    n = len(refined)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for ki, kj, _ in contacts:
        parent[find(ki)] = find(kj)
    clusters: dict[int, list[int]] = {}
    for k in range(n):
        clusters.setdefault(find(k), []).append(k)
    withheld: set[int] = set()
    log = []
    for members in clusters.values():
        if len(members) < 2:
            continue
        fits = [refined[k][0] for k in members]
        yaws = [f.yaw_rad for f in fits]
        par = all(abs((y - yaws[0] + np.pi / 4) % (np.pi / 2) - np.pi / 4) < np.radians(4) for y in yaws)
        corners = []
        for f in fits:
            ux = np.array([np.cos(f.yaw_rad), np.sin(f.yaw_rad)]) * block.major_m / 2
            uy = np.array([-np.sin(f.yaw_rad), np.cos(f.yaw_rad)]) * block.minor_m / 2
            c = np.array(f.xy_m)
            corners += [c + ux + uy, c + ux - uy, c - ux + uy, c - ux - uy]
        (_, _), (rw, rh), _ = cv2.minAreaRect(np.array(corners, np.float32))
        rect = par and abs(rw * rh - len(members) * block.major_m * block.minor_m) <= 0.06 * rw * rh
        if not rect:
            continue
        for ki, kj, nrm in contacts:
            if find(ki) != find(members[0]):
                continue
            ok, kind = (False, "no_cue_fn")
            if cue_fn is not None:
                a, b = refined[ki][0], refined[kj][0]
                nv = np.array(nrm)
                pts = np.vstack([a.points_xyz, b.points_xyz])
                cut = 0.5 * (float(np.dot(a.xy_m, nv)) + float(np.dot(b.xy_m, nv)))
                ok, kind = cue_fn(pts, nv, cut)
            log.append(kind)
            if not ok:
                withheld.update((ki, kj))
    if log:
        diag["slab_guard"] = {"seam_cues": log, "withheld": sorted(withheld)}
    return withheld


class RegionCache:
    """Reuse a region's *structural* verdicts (which blocks are provably stable) while its tri-state occupancy is
    unchanged; poses are always re-measured from the current frame by ``refine_group``.

    A hit requires the same band, a window within 2 cells, must-cell IoU >= ``min_iou`` and <= ``max_state_diff``
    of window cells changing state, compared with the frame on which the proof was made (no drift accumulation).
    Entries expire after ``max_uses`` reuses.  The owner must ``clear()`` on calibration/config change.
    """

    def __init__(self, min_iou: float = 0.97, max_state_diff: float = 0.01, max_uses: int = 30) -> None:
        self.min_iou, self.max_state_diff, self.max_uses = min_iou, max_state_diff, max_uses
        self.entries: list[dict[str, Any]] = []
        self.hits = 0
        self.misses = 0
        self.key: tuple = ()

    def clear(self) -> None:
        self.entries.clear()

    def bind(self, key: tuple) -> None:
        if key != self.key:
            self.entries.clear()
            self.key = key

    def lookup(self, band, win, st, must):
        for e in self.entries:
            if e["band"] != tuple(round(v, 4) for v in band) or e["uses"] >= self.max_uses:
                continue
            if any(abs(a - b) > 2 for a, b in zip(e["win"], win)):
                continue
            if e["st"].shape != st.shape or e["must"].shape != must.shape:
                continue
            inter = np.count_nonzero(e["must"] & must)
            union = np.count_nonzero(e["must"] | must)
            if union == 0 or inter / union < self.min_iou:
                continue
            if np.count_nonzero(e["st"] != st) > self.max_state_diff * st.size:
                continue
            e["uses"] += 1
            self.hits += 1
            return e["seed"], e["verdicts"]
        self.misses += 1
        return None

    def store(self, band, win, st, must, seed, verdicts) -> None:
        self.entries = [e for e in self.entries if any(abs(a - b) > 2 for a, b in zip(e["win"], win))][-15:]
        self.entries.append(dict(band=tuple(round(v, 4) for v in band), win=win, st=st.copy(), must=must.copy(),
                                 seed=seed, verdicts=verdicts, uses=0))


# ----------------------------------------------------------------------------------------------- entry point

def explain_region(
    *,
    grid: HeightmapGrid,
    h_filled: np.ndarray,
    comp_masks: list[np.ndarray],
    band: tuple[float, float],
    pts_tbl: np.ndarray,
    block: BlockSpec,
    params: TilingParams | None = None,
    extra_unknown: np.ndarray | None = None,
    extra_free: np.ndarray | None = None,
    ray_check: Callable | None = None,
    views: list[ViewDepth] | None = None,
    h_max: np.ndarray | None = None,
    prior_angles_deg: tuple[float, ...] = (),
    deadline_s: float | None = None,
    cue_fn: Callable[[np.ndarray, np.ndarray, float], tuple[bool, str]] | None = None,
    slab_guard: bool = True,
    region_cache: "RegionCache | None" = None,
) -> RegionResult:
    p = params or TilingParams()
    t0 = time.perf_counter()
    deadline = min(t0 + p.region_budget_ms / 1000.0, deadline_s if deadline_s is not None else np.inf)
    union = np.zeros(h_filled.shape, dtype=bool)
    for m in comp_masks:
        union |= m
    # unfilled heightmap: shadow / hole cells stay UNKNOWN instead of being filled with a neighbour's height
    state = build_tristate(h_max if h_max is not None else h_filled, band, extra_unknown, extra_free)
    state[(state == OCC) & ~union] = FREE          # other objects of this band are not ours
    rows, cols = np.nonzero(union)
    diag: dict[str, Any] = {}
    if len(rows) == 0:
        return RegionResult([], [], [], "empty", diag)
    marg = int(np.ceil(p.window_margin_m / grid.cell_m))
    r0, r1 = max(0, rows.min() - marg), min(grid.ny, rows.max() + marg + 1)
    c0, c1 = max(0, cols.min() - marg), min(grid.nx, cols.max() + marg + 1)
    st = state[r0:r1, c0:c1]
    must = union[r0:r1, c0:c1] & (st == OCC)
    rr, cc = np.mgrid[r0:r1, c0:c1]
    xw, yw = grid.cell_xy(rr, cc)
    cen = (float(xw[must].mean()), float(yw[must].mean())) if np.any(must) else (0.0, 0.0)
    z_cells = h_filled[union & np.isfinite(h_filled)]
    z_med = float(np.median(z_cells)) if len(z_cells) else float("nan")
    layer = max(1, int(round(z_med / block.height_m))) if np.isfinite(z_med) else 1
    diag.update(z_med_mm=round(z_med * 1000, 2), layer=layer, cells=int(np.count_nonzero(must)))
    if st.size > p.max_window_cells:
        return RegionResult([], [], [cen], "window_too_large", diag)
    if views:
        unk = st == UNK
        if np.any(unk):
            t_c = time.perf_counter()
            st = st.copy()
            fr = carve_free(xw[unk], yw[unk], (layer - 0.5) * block.height_m, views)
            idx = np.flatnonzero(unk.ravel())[fr]
            st.ravel()[idx] = FREE
            must = must & (st == OCC)
            diag["carved_free"] = int(np.count_nonzero(fr))
            diag["unknown_cells"] = int(np.count_nonzero(unk))
            diag["carve_ms"] = round((time.perf_counter() - t_c) * 1000, 3)
        # physical support: a block on layer k >= 2 cannot float. If fixed views see *through* the layer below
        # at most of the occupied cells, the occupied surface is not a stacked block (e.g. a robot palm at 7 x h).
        # Every layer down to the table is checked: a solid occluder (hand, robot link) whose top lands on k * h
        # is solid just below its top but has measured free air further down (measured: palm at 7 x 15 mm).
        if layer >= 2 and np.any(must):
            worst = 0.0
            for j in range(1, layer):
                below = carve_free(xw[must], yw[must], (j - 0.5) * block.height_m, views)
                worst = max(worst, float(np.mean(below)))
            diag["unsupported_frac"] = round(worst, 3)
            if worst > p.max_unsupported_frac:
                return RegionResult([], [], [cen], "unsupported_layer", diag)
    if not np.isfinite(z_med) or abs(z_med - layer * block.height_m) > p.height_gate_m:
        return RegionResult([], [], [cen], "height_not_block_multiple", diag)
    if np.count_nonzero(must) < 8:
        return RegionResult([], [], [cen], "too_few_cells", diag)
    lo, hi = band
    zb = pts_tbl[:, 2]
    pts_band = pts_tbl[(zb >= lo - 0.003) & (zb <= hi + 0.003)]
    # raw boundary points (inside the group, cell adjacent to a non-group cell) for angle refinement
    edge_cells = must & ~cv2.erode(must.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    ri, rj = grid.world_to_ij(pts_band[:, :2])
    okp = (ri >= r0) & (ri < r1) & (rj >= c0) & (rj < c1)
    sel = np.zeros(len(pts_band), bool)
    sel[okp] = edge_cells[ri[okp] - r0, rj[okp] - c0]
    boundary_xy = pts_band[sel, :2].astype(np.float64)
    shortlist = dominant_angles_deg(must, boundary_xy, p.max_angles, prior_angles_deg)
    diag["angles_deg"] = [round(a, 2) for a in shortlist]
    hit = region_cache.lookup(band, (r0, r1, c0, c1), st, must) if region_cache is not None else None
    if hit is not None:
        seed, verdicts = hit
        diag["cache_hit"] = True
    else:
        model = RegionModel(block, p, st, must, xw, yw, grid.cell_m, layer, ray_check)
        if p.solver == "enumerate":
            seed, verdicts, d2 = model.solve_enumerate(shortlist, deadline)
        elif p.solver == "adaptive":
            seed, verdicts, d2 = model.solve_adaptive(shortlist, deadline)
        else:
            seed, verdicts, d2 = model.solve_prove(shortlist, deadline)
        diag.update(d2)
        diag.update(model.stats)
        if region_cache is not None and seed and not any(v == TIMEOUT for v in verdicts):
            region_cache.store(band, (r0, r1, c0, c1), st, must, seed, verdicts)
    ring = cv2.dilate(must.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & (st == FREE)
    hard_xy = np.stack([xw[ring], yw[ring]], axis=1) if np.any(ring) else np.zeros((0, 2))
    # refine the whole seed explanation: only STABLE blocks are trusted for pinning and publication, but the slab
    # guard must see every block of the explanation (a proven block next to a timed-out one is still a seam)
    occ_xy = np.stack([xw[must], yw[must]], axis=1)
    trusted = [v == STABLE for v in verdicts]
    contacts: list = []
    refined = refine_group(seed, pts_band, hard_xy, block, p.edge_bias_m, occ_xy=occ_xy, cell_m=grid.cell_m,
                           min_observed_frac=p.min_observed_frac, trusted=trusted,
                           contacts_out=contacts) if any(trusted) else []
    withheld = _slab_guard(refined, contacts, pts_band, block, cue_fn, diag) if (slab_guard and len(refined) >= 2) else set()
    refined = [(f, pinned and (k not in withheld), d) for k, (f, pinned, d) in enumerate(refined)]
    fits = [f for f, pinned, _ in refined if pinned]
    unresolved = [(q.cx, q.cy) for q, v in zip(seed, verdicts) if v != STABLE]
    unresolved += [f.xy_m for (f, pinned, _), t in zip(refined, trusted) if t and not pinned]
    diag["pins"] = [d for _, _, d in refined]
    if not seed:
        unresolved = [cen]
    diag["verdicts"] = verdicts
    diag["total_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
    if not seed:
        reason = d2.get("abort", "no_explanation")
    elif all(v == STABLE for v in verdicts) and len(fits) == len(seed):
        reason = "all_stable"
    elif any(v == TIMEOUT for v in verdicts):
        reason = "timeout_partial" if fits else "timeout"
    else:
        reason = "partial" if fits else "ambiguous_explanations"
    return RegionResult(fits, verdicts, unresolved, reason, diag)

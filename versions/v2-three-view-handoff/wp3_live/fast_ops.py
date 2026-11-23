"""Fast float32 reductions, batched bilinear sampling, single-thread BLAS/CV discipline & optional Numba hooks."""
from __future__ import annotations

import heapq
import importlib.util
import math
import os
from typing import Any

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

HAS_NUMBA = importlib.util.find_spec("numba") is not None
_THREADS_CONFIGURED = False


def configure_single_thread_libs() -> None:
    """Pin OpenMP/BLAS/OpenCV to 1 thread so multi-view workers never oversubscribe a 6-core CPU."""
    global _THREADS_CONFIGURED
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    if not _THREADS_CONFIGURED:
        cv_threads = int(os.environ.get("WP3_CV2_THREADS", "1"))
        try:
            cv2.setNumThreads(cv_threads)
        except Exception:
            pass
        _THREADS_CONFIGURED = True


def fast_median_1d(arr: np.ndarray) -> float:
    """Fast 1D median using O(n) np.partition without numpy._ureduce overhead."""
    a = np.asarray(arr, dtype=np.float32).ravel()
    n = a.size
    if n == 0:
        return float("nan")
    if n == 1:
        return float(a[0])
    if n == 2:
        return 0.5 * (float(a[0]) + float(a[1]))
    if n == 3:
        x, y, z = float(a[0]), float(a[1]), float(a[2])
        return x + y + z - min(x, y, z) - max(x, y, z)
    mid = n >> 1
    if n & 1:
        return float(np.partition(a, mid)[mid])
    part = np.partition(a, (mid - 1, mid))
    return 0.5 * (float(part[mid - 1]) + float(part[mid]))


def fast_percentiles_2col(
    arr_2col: np.ndarray,
    q_lo: float = 4.0,
    q_hi: float = 96.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Simultaneous [q_lo, q_hi] percentile along axis=0 for (N, 2) finite float32 arrays."""
    a = np.asarray(arr_2col, dtype=np.float32)
    n = a.shape[0]
    if n == 0:
        nan2 = np.full(a.shape[1], np.nan, dtype=np.float32)
        return nan2, nan2
    if n == 1:
        v = a[0].copy()
        return v, v.copy()
    idx_lo = (q_lo / 100.0) * (n - 1)
    idx_hi = (q_hi / 100.0) * (n - 1)
    lo_f = int(math.floor(idx_lo))
    lo_c = int(math.ceil(idx_lo))
    hi_f = int(math.floor(idx_hi))
    hi_c = int(math.ceil(idx_hi))
    kth = sorted({lo_f, lo_c, hi_f, hi_c})
    part = np.partition(a, kth, axis=0)
    w_lo = np.float32(idx_lo - lo_f)
    w_hi = np.float32(idx_hi - hi_f)
    v_lo = part[lo_f] * (np.float32(1.0) - w_lo) + part[lo_c] * w_lo
    v_hi = part[hi_f] * (np.float32(1.0) - w_hi) + part[hi_c] * w_hi
    return v_lo, v_hi


def fast_percentile_pair_1d(arr: np.ndarray, q_lo: float = 6.0, q_hi: float = 94.0) -> tuple[float, float]:
    """Simultaneous [q_lo, q_hi] percentile for a 1D finite array via one np.partition call."""
    a = np.asarray(arr, dtype=np.float32).ravel()
    n = a.size
    if n == 0:
        return float("nan"), float("nan")
    if n == 1:
        v = float(a[0])
        return v, v
    idx_lo = (q_lo / 100.0) * (n - 1)
    idx_hi = (q_hi / 100.0) * (n - 1)
    lo_f, lo_c = int(math.floor(idx_lo)), int(math.ceil(idx_lo))
    hi_f, hi_c = int(math.floor(idx_hi)), int(math.ceil(idx_hi))
    kth = sorted({lo_f, lo_c, hi_f, hi_c})
    part = np.partition(a, kth)
    w_lo = idx_lo - lo_f
    w_hi = idx_hi - hi_f
    return (
        float(part[lo_f]) * (1.0 - w_lo) + float(part[lo_c]) * w_lo,
        float(part[hi_f]) * (1.0 - w_hi) + float(part[hi_c]) * w_hi,
    )


def fast_nanmedian_rows(arr_2d: np.ndarray) -> np.ndarray:
    """Fast NaN-safe float32 row median with ZERO numpy.ma overhead and ZERO all-NaN warnings."""
    a = np.asarray(arr_2d, dtype=np.float32)
    nrows, ncols = a.shape
    if ncols == 0:
        return np.full(nrows, np.nan, dtype=np.float32)
    fin = np.isfinite(a)
    if bool(fin.all()):
        mid = ncols >> 1
        if ncols & 1:
            return np.partition(a, mid, axis=1)[:, mid].astype(np.float32, copy=False)
        part = np.partition(a, (mid - 1, mid), axis=1)
        return (0.5 * (part[:, mid - 1] + part[:, mid])).astype(np.float32, copy=False)

    work = np.where(fin, a, np.float32(np.inf))
    s = np.sort(work, axis=1)
    cnt = np.sum(fin, axis=1)
    has = cnt > 0
    out = np.full(nrows, np.nan, dtype=np.float32)
    if np.any(has):
        hs = s[has]
        hc = cnt[has]
        i1 = np.clip((hc - 1) >> 1, 0, ncols - 1)[:, None]
        i2 = np.clip(hc >> 1, 0, ncols - 1)[:, None]
        v1 = np.take_along_axis(hs, i1, axis=1)[:, 0]
        v2 = np.take_along_axis(hs, i2, axis=1)[:, 0]
        out[has] = 0.5 * (v1 + v2)
    return out


def fast_nanmedian_3rows(arr_3xN: np.ndarray) -> np.ndarray:
    """Exact branchless NaN-safe float32 median along axis=0 for shape (3, N) arrays."""
    a = np.asarray(arr_3xN, dtype=np.float32)
    if a.shape[0] != 3:
        return fast_nanmedian_rows(np.ascontiguousarray(a.T))
    fin = np.isfinite(a)
    if bool(fin.all()):
        x, y, z = a[0], a[1], a[2]
        return (x + y + z - np.minimum(np.minimum(x, y), z) - np.maximum(np.maximum(x, y), z)).astype(np.float32, copy=False)

    cnt = fin.astype(np.int32).sum(axis=0)
    vals = np.where(fin, a, np.float32(0.0))
    s = vals[0] + vals[1] + vals[2]
    out = np.full(a.shape[1], np.nan, dtype=np.float32)
    m3 = cnt == 3
    m2 = cnt == 2
    m1 = cnt == 1
    if np.any(m3):
        a3 = a[:, m3]
        mn = np.minimum(np.minimum(a3[0], a3[1]), a3[2])
        mx = np.maximum(np.maximum(a3[0], a3[1]), a3[2])
        out[m3] = s[m3] - mn - mx
    if np.any(m2):
        out[m2] = 0.5 * s[m2]
    if np.any(m1):
        out[m1] = s[m1]
    return out


def fast_nanmedian_cols(arr_NxC: np.ndarray) -> np.ndarray:
    """Fast NaN-safe float32 median along axis=0 for (N, C) arrays."""
    return fast_nanmedian_rows(np.ascontiguousarray(np.asarray(arr_NxC, dtype=np.float32).T))


def fast_rolling_median_1d(prof: np.ndarray, half: int, min_count: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized rolling NaN-safe median over [max(0, k-half), min(n, k+half+1)] matching loop edges."""
    p = np.asarray(prof, dtype=np.float32).ravel()
    n = p.size
    if n == 0 or half < 0:
        return np.empty(0, dtype=np.float32), np.zeros(0, dtype=np.int32)
    if 2 * half + 1 == 1:
        cnt = np.isfinite(p).astype(np.int32)
        return p.copy(), cnt
    padded = np.empty(n + 2 * half, dtype=np.float32)
    padded[:half] = np.nan
    padded[half : half + n] = p
    padded[half + n :] = np.nan
    win = sliding_window_view(padded, 2 * half + 1)
    med = fast_nanmedian_rows(win)
    cnt = np.sum(np.isfinite(win), axis=1).astype(np.int32)
    if min_count > 1:
        med = np.where(cnt >= min_count, med, np.float32(np.nan))
    return med, cnt


def bilinear_sample_fast(
    img: np.ndarray,
    uv: np.ndarray,
    *,
    use_cv2_remap: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear image sampling with strict bounds check; batched C cv2.remap + float32 fallback."""
    h, w = img.shape[:2]
    uv32 = np.asarray(uv, dtype=np.float32)
    x = uv32[:, 0]
    y = uv32[:, 1]
    inside = (x >= 0.0) & (x <= (w - 1.001)) & (y >= 0.0) & (y <= (h - 1.001))
    n = len(uv32)
    if n == 0 or not np.any(inside):
        return np.zeros((n,) + img.shape[2:], dtype=np.float32), inside

    xi = np.clip(x, 0.0, w - 1.001)
    yi = np.clip(y, 0.0, h - 1.001)

    if use_cv2_remap and n <= 32766 and img.dtype == np.float32:
        # cv2.remap runs native SIMD bilinear interpolation in C on float32 without uint8 truncation
        res = cv2.remap(
            img,
            xi.reshape(n, 1),
            yi.reshape(n, 1),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        out = res.reshape((n,) + img.shape[2:]).astype(np.float32, copy=False)
        return out, inside

    x0 = np.floor(xi).astype(np.int32)
    y0 = np.floor(yi).astype(np.int32)
    dx = (xi - x0)[:, None] if img.ndim == 3 else (xi - x0)
    dy = (yi - y0)[:, None] if img.ndim == 3 else (yi - y0)
    v00 = img[y0, x0].astype(np.float32)
    v10 = img[y0, x0 + 1].astype(np.float32)
    v01 = img[y0 + 1, x0].astype(np.float32)
    v11 = img[y0 + 1, x0 + 1].astype(np.float32)
    out = (np.float32(1.0) - dy) * ((np.float32(1.0) - dx) * v00 + dx * v10) + dy * ((np.float32(1.0) - dx) * v01 + dx * v11)
    return out, inside


def _heightmap_max_z_py(flat: np.ndarray, z: np.ndarray, h_flat: np.ndarray, counts_flat: np.ndarray) -> None:
    h_flat.fill(-np.inf)
    counts_flat.fill(0)
    np.maximum.at(h_flat, flat, z)
    counts_flat[:] = np.bincount(flat, minlength=counts_flat.size).astype(np.int32)


def _meyer_watershed_py(elevation: np.ndarray, markers: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Clean single-pass 4-connected priority-flood Meyer watershed."""
    H, W = elevation.shape
    labels = markers.copy().astype(np.int32)
    in_q = labels > 0
    pq: list[tuple[float, int, int, int, int]] = []
    seeded_r, seeded_c = np.nonzero(in_q)
    for r, c in zip(seeded_r.tolist(), seeded_c.tolist()):
        if mask[r, c]:
            heapq.heappush(pq, (float(elevation[r, c]), 0, r, c, int(labels[r, c])))
    step = 1
    while pq:
        _, _, r, c, lbl = heapq.heappop(pq)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not in_q[nr, nc]:
                in_q[nr, nc] = True
                labels[nr, nc] = lbl
                heapq.heappush(pq, (float(elevation[nr, nc]), step, nr, nc, lbl))
                step += 1
    return np.where(mask & (labels > 0), labels, 0).astype(np.int32)


if HAS_NUMBA:
    from numba import njit  # type: ignore

    @njit(cache=True)
    def _heightmap_max_z_nb(flat: np.ndarray, z: np.ndarray, h_flat: np.ndarray, counts_flat: np.ndarray) -> None:
        h_flat[:] = -np.inf
        counts_flat[:] = 0
        for idx in range(flat.shape[0]):
            f = flat[idx]
            zv = z[idx]
            counts_flat[f] += 1
            if zv > h_flat[f]:
                h_flat[f] = zv

    @njit(cache=True)
    def _meyer_watershed_nb(elevation: np.ndarray, markers: np.ndarray, mask: np.ndarray) -> np.ndarray:
        H, W = elevation.shape
        n_tot = H * W
        labels = markers.copy().astype(np.int32)
        in_q = labels > 0
        heap_elev = np.empty(n_tot + 8, dtype=np.float32)
        heap_step = np.empty(n_tot + 8, dtype=np.int64)
        heap_pos = np.empty(n_tot + 8, dtype=np.int32)
        heap_lbl = np.empty(n_tot + 8, dtype=np.int32)
        h_sz = 0
        step = 1

        def push(ev: float, st: int, pos: int, lb: int) -> None:
            nonlocal h_sz
            i = h_sz
            h_sz += 1
            heap_elev[i] = ev
            heap_step[i] = st
            heap_pos[i] = pos
            heap_lbl[i] = lb
            while i > 0:
                p = (i - 1) >> 1
                pe = heap_elev[p]
                ie = heap_elev[i]
                if ie < pe or (ie == pe and heap_step[i] < heap_step[p]):
                    heap_elev[p], heap_elev[i] = ie, pe
                    heap_step[p], heap_step[i] = heap_step[i], heap_step[p]
                    heap_pos[p], heap_pos[i] = heap_pos[i], heap_pos[p]
                    heap_lbl[p], heap_lbl[i] = heap_lbl[i], heap_lbl[p]
                    i = p
                else:
                    break

        for r in range(H):
            for c in range(W):
                if in_q[r, c] and mask[r, c]:
                    push(elevation[r, c], 0, r * W + c, labels[r, c])

        while h_sz > 0:
            pos = heap_pos[0]
            lbl = heap_lbl[0]
            h_sz -= 1
            if h_sz > 0:
                heap_elev[0] = heap_elev[h_sz]
                heap_step[0] = heap_step[h_sz]
                heap_pos[0] = heap_pos[h_sz]
                heap_lbl[0] = heap_lbl[h_sz]
                i = 0
                while True:
                    l = (i << 1) + 1
                    if l >= h_sz:
                        break
                    r_idx = l + 1
                    best = l
                    if r_idx < h_sz:
                        if heap_elev[r_idx] < heap_elev[l] or (
                            heap_elev[r_idx] == heap_elev[l] and heap_step[r_idx] < heap_step[l]
                        ):
                            best = r_idx
                    if heap_elev[best] < heap_elev[i] or (
                        heap_elev[best] == heap_elev[i] and heap_step[best] < heap_step[i]
                    ):
                        heap_elev[i], heap_elev[best] = heap_elev[best], heap_elev[i]
                        heap_step[i], heap_step[best] = heap_step[best], heap_step[i]
                        heap_pos[i], heap_pos[best] = heap_pos[best], heap_pos[i]
                        heap_lbl[i], heap_lbl[best] = heap_lbl[best], heap_lbl[i]
                        i = best
                    else:
                        break
            r = pos // W
            c = pos - r * W
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not in_q[nr, nc]:
                    in_q[nr, nc] = True
                    labels[nr, nc] = lbl
                    push(elevation[nr, nc], step, nr * W + nc, lbl)
                    step += 1

        out = np.zeros((H, W), dtype=np.int32)
        for r in range(H):
            for c in range(W):
                if mask[r, c] and labels[r, c] > 0:
                    out[r, c] = labels[r, c]
        return out

    def heightmap_max_z_fast(flat: np.ndarray, z: np.ndarray, h_flat: np.ndarray, counts_flat: np.ndarray) -> None:
        _heightmap_max_z_nb(flat, z, h_flat, counts_flat)

    def meyer_watershed_2d_fast(elevation: np.ndarray, markers: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return _meyer_watershed_nb(np.asarray(elevation, dtype=np.float32), markers.astype(np.int32), mask.astype(np.bool_))

else:
    def heightmap_max_z_fast(flat: np.ndarray, z: np.ndarray, h_flat: np.ndarray, counts_flat: np.ndarray) -> None:
        _heightmap_max_z_py(flat, z, h_flat, counts_flat)

    def meyer_watershed_2d_fast(elevation: np.ndarray, markers: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return _meyer_watershed_py(elevation, markers, mask)

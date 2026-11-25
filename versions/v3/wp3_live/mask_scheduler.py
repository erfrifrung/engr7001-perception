"""Bounded latest-only asynchronous mask proposer with explicit validity (V3 D5).

The CPU pipeline never waits.  A job carries (frame id, observation stamp, calibration epoch, config hash,
region id, region revision, deadline).  A result is usable only if all of them still match the caller's current
state *and* the result's observation is younger than ``max_age_ms`` at poll time; anything else is counted and
dropped.  Backend errors, restarts (generation bump) and queue saturation are counted, never raised into the
publish path.  Masks may only act as evidence inside existing geometric checks (V3 slab-guard seam cue).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class MaskJob:
    frame_id: int
    obs_stamp_ns: int
    calib_epoch: str
    config_hash: str
    region_id: str
    region_rev: int
    roi_xyxy: tuple[int, int, int, int]
    submitted_s: float
    generation: int


@dataclass
class MaskResult:
    job: MaskJob
    masks: list[np.ndarray]
    backend: str
    latency_ms: float
    done_s: float


@dataclass
class SchedulerStats:
    submitted: int = 0
    replaced_unstarted: int = 0
    completed: int = 0
    used: int = 0
    rejected: dict[str, int] = field(default_factory=dict)
    backend_errors: int = 0
    restarts: int = 0
    busy_s: float = 0.0

    def reject(self, why: str) -> None:
        self.rejected[why] = self.rejected.get(why, 0) + 1


class LatestOnlyMaskScheduler:
    def __init__(self, backend: Callable[[np.ndarray, tuple[int, int, int, int]], list[np.ndarray]] | None,
                 *, backend_name: str = "none", max_age_ms: float = 600.0) -> None:
        self.backend = backend
        self.backend_name = backend_name
        self.max_age_ms = max_age_ms
        self.stats = SchedulerStats()
        self._lock = threading.Lock()
        self._pending: tuple[MaskJob, np.ndarray] | None = None
        self._results: dict[str, MaskResult] = {}
        self._gen = 0
        self._ev = threading.Event()
        self._stop = False
        self._thr = threading.Thread(target=self._run, name="wp3_v3_mask", daemon=True)
        self._thr.start()

    @property
    def available(self) -> bool:
        return self.backend is not None

    def submit(self, job_fields: dict[str, Any], rgb: np.ndarray) -> bool:
        """Latest-only: an unstarted job is replaced. Copies only the ROI crop (not the full frame)."""
        if self.backend is None:
            self.stats.reject("backend_unavailable")
            return False
        x0, y0, x1, y1 = job_fields["roi_xyxy"]
        crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])
        with self._lock:
            job = MaskJob(submitted_s=time.perf_counter(), generation=self._gen, **job_fields)
            if self._pending is not None:
                self.stats.replaced_unstarted += 1
            self._pending = (job, crop)
            self.stats.submitted += 1
        self._ev.set()
        return True

    def restart(self) -> None:
        """Backend restart: in-flight and stored results from the old generation become invalid."""
        with self._lock:
            self._gen += 1
            self._pending = None
            self._results.clear()
            self.stats.restarts += 1

    def _run(self) -> None:
        while not self._stop:
            self._ev.wait(0.05)
            self._ev.clear()
            with self._lock:
                item, self._pending = self._pending, None
            if item is None:
                continue
            job, crop = item
            t0 = time.perf_counter()
            try:
                masks = self.backend(crop, job.roi_xyxy)
            except Exception:
                self.stats.backend_errors += 1
                continue
            t1 = time.perf_counter()
            self.stats.busy_s += t1 - t0
            with self._lock:
                if job.generation != self._gen:
                    self.stats.reject("generation")
                    continue
                self._results[job.region_id] = MaskResult(job, masks, self.backend_name, (t1 - t0) * 1000.0, t1)
                self.stats.completed += 1

    def poll(self, *, region_id: str, region_rev: int, calib_epoch: str, config_hash: str,
             now_stamp_ns: int) -> MaskResult | None:
        with self._lock:
            r = self._results.get(region_id)
        if r is None:
            return None
        j = r.job
        why = None
        if j.calib_epoch != calib_epoch or j.config_hash != config_hash:
            why = "epoch_mismatch"
        elif j.region_rev != region_rev:
            why = "region_revision"
        elif (now_stamp_ns - j.obs_stamp_ns) * 1e-6 > self.max_age_ms:
            why = "too_old"
        if why is not None:
            self.stats.reject(why)
            with self._lock:
                if self._results.get(region_id) is r:
                    del self._results[region_id]
            return None
        self.stats.used += 1
        return r

    def stop(self) -> None:
        self._stop = True
        self._ev.set()
        self._thr.join(timeout=1.0)


def region_revision(must_mask: np.ndarray, quant_cells: int = 4) -> int:
    """Revision signature of a region's occupied cells (coarse, noise tolerant): changes when occupancy moves."""
    ys, xs = np.nonzero(must_mask)
    if len(xs) == 0:
        return 0
    key = (int(xs.min()) // quant_cells, int(xs.max()) // quant_cells, int(ys.min()) // quant_cells,
           int(ys.max()) // quant_cells, int(len(xs)) // (quant_cells * 25))
    return hash(key) & 0x7FFFFFFF

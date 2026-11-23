"""Bounded asynchronous disk I/O worker so PNG encoding and JSONL flushing never spike live latency."""
from __future__ import annotations

from pathlib import Path
import queue
import threading
from typing import Any

import cv2
import numpy as np


class AsyncIOArtifactWriter:
    """Single background daemon thread draining JSONL log writes and backpressure-bounded PNG saves."""

    def __init__(self, jsonl_path: Path | None, *, max_pending_png: int = 2, flush_every: int = 10) -> None:
        self.jsonl_path = Path(jsonl_path) if jsonl_path is not None else None
        self.max_pending_png = max(1, int(max_pending_png))
        self.flush_every = max(1, int(flush_every))
        self._q: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=256)
        self._lock = threading.Lock()
        self._pending_png = 0
        self.dropped_png = 0
        self.dropped_jsonl = 0
        self._written_jsonl = 0
        self._stopped = False

        self._log_fh = None
        if self.jsonl_path is not None:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_fh = open(self.jsonl_path, "w", encoding="utf-8")

        self._thr = threading.Thread(target=self._run, name="wp3_async_io", daemon=True)
        self._thr.start()

    def write_jsonl(self, line: str) -> bool:
        if self._log_fh is None or self._stopped:
            return False
        try:
            self._q.put_nowait(("jsonl", line))
            return True
        except queue.Full:
            self.dropped_jsonl += 1
            return False

    def write_png(self, path: Path, bgr: np.ndarray) -> bool:
        if self._stopped:
            return False
        with self._lock:
            if self._pending_png >= self.max_pending_png:
                self.dropped_png += 1
                return False
            self._pending_png += 1
        try:
            self._q.put_nowait(("png", (Path(path), np.ascontiguousarray(bgr).copy())))
            return True
        except queue.Full:
            with self._lock:
                self._pending_png = max(0, self._pending_png - 1)
                self.dropped_png += 1
            return False

    def _run(self) -> None:
        pending_lines = 0
        while True:
            kind, payload = self._q.get()
            if kind == "stop":
                break
            if kind == "jsonl":
                if self._log_fh is not None:
                    self._log_fh.write(payload + "\n")
                    self._written_jsonl += 1
                    pending_lines += 1
                    if pending_lines >= self.flush_every or self._q.empty():
                        self._log_fh.flush()
                        pending_lines = 0
            elif kind == "png":
                out_path, bgr = payload
                try:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out_path), bgr)
                finally:
                    with self._lock:
                        self._pending_png = max(0, self._pending_png - 1)

        if self._log_fh is not None:
            try:
                self._log_fh.flush()
                self._log_fh.close()
            finally:
                self._log_fh = None

    def stop(self, timeout_s: float = 2.0) -> None:
        if self._stopped:
            return
        self._stopped = True
        try:
            self._q.put(("stop", None), timeout=0.25)
        except queue.Full:
            pass
        if self._thr.is_alive():
            self._thr.join(timeout=timeout_s)
        if self._log_fh is not None:
            try:
                self._log_fh.flush()
                self._log_fh.close()
            except Exception:
                pass
            self._log_fh = None

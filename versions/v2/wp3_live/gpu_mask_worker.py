"""Asynchronous ROCm/ONNX/callable SAM 2 mask challenger & bounded 1-slot GPU worker."""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import queue
import threading
import time
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class MaskProposalResult:
    epoch_id: int
    observation_stamp_ns: int
    calib_epoch: str
    region_key: str
    masks_u8: list[np.ndarray]
    latency_ms: float
    backend: str


class SAM2MaskChallenger:
    """Frozen-prompt Stage 2 mask proposal worker (PyTorch ROCm SAM 2 / ONNX / callable)."""

    def __init__(
        self,
        weights_path: str | None = None,
        *,
        backend: str = "auto",
        callable_fn: Callable[[np.ndarray, tuple[int, int, int, int]], list[np.ndarray]] | None = None,
    ) -> None:
        self.weights_path = weights_path or ""
        self.has_torch = importlib.util.find_spec("torch") is not None
        self.has_sam2 = importlib.util.find_spec("sam2") is not None
        self.has_ort = importlib.util.find_spec("onnxruntime") is not None
        self.callable_fn = callable_fn
        self._predictor: Any = None
        self._ort_sess: Any = None
        self.backend = "none"
        self.available = False
        self.unavailable_reason = ""

        if callable_fn is not None and backend in ("auto", "callable"):
            self.backend = "callable"
            self.available = True
            return

        if backend in ("auto", "sam2") and self.has_sam2 and self.has_torch and self.weights_path:
            try:
                import torch
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor

                device = "cuda" if torch.cuda.is_available() else "cpu"
                model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
                model = build_sam2(model_cfg, self.weights_path, device=device)
                self._predictor = SAM2ImagePredictor(model)
                self.backend = f"sam2_{device}"
                self.available = True
                return
            except Exception as exc:
                self.unavailable_reason = f"sam2 init failed: {exc}"

        if backend in ("auto", "onnx") and self.has_ort and self.weights_path:
            try:
                import onnxruntime as ort

                provs = ort.get_available_providers()
                chosen = [p for p in ("MIGraphXExecutionProvider", "ROCMExecutionProvider", "CPUExecutionProvider") if p in provs]
                self._ort_sess = ort.InferenceSession(self.weights_path, providers=chosen or ["CPUExecutionProvider"])
                self.backend = f"onnx_{(chosen[0] if chosen else 'CPU')}"
                self.available = True
                return
            except Exception as exc:
                self.unavailable_reason = f"onnxruntime init failed: {exc}"

        if not self.unavailable_reason:
            if not self.has_sam2 and not self.has_ort:
                self.unavailable_reason = "sam2 and onnxruntime packages not installed in this environment"
            elif not self.weights_path:
                self.unavailable_reason = "no frozen SAM 2 checkpoint/ONNX path supplied"

    def propose_masks_from_roi_boxes(
        self,
        rgb_img: np.ndarray,
        roi_boxes_xyxy: list[tuple[int, int, int, int]],
    ) -> dict[str, Any]:
        """Run frozen-box prompt segmentation or return structured unavailable info."""
        if not self.available or not roi_boxes_xyxy:
            return {"available": self.available, "reason": self.unavailable_reason, "masks": [], "backend": self.backend}

        t0 = time.perf_counter()
        masks: list[np.ndarray] = []
        if self.callable_fn is not None:
            for box in roi_boxes_xyxy:
                masks.extend(self.callable_fn(rgb_img, box))
        elif self._predictor is not None:
            self._predictor.set_image(rgb_img)
            for x0, y0, x1, y1 in roi_boxes_xyxy:
                m_arr, scores, _ = self._predictor.predict(box=np.array([x0, y0, x1, y1]), multimask_output=True)
                for m, sc in zip(m_arr, scores):
                    if float(sc) >= 0.70 and int(np.count_nonzero(m)) >= 16:
                        masks.append((m > 0).astype(np.uint8))
        return {
            "available": True,
            "backend": self.backend,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
            "masks": masks,
        }


class AsyncMaskWorker:
    """Bounded 1-slot asynchronous challenger thread never blocking CPU frame publication."""

    def __init__(self, challenger: SAM2MaskChallenger) -> None:
        self.challenger = challenger
        self._q: queue.Queue[tuple[int, int, str, str, np.ndarray, tuple[int, int, int, int]]] = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._results: dict[str, MaskProposalResult] = {}
        self._stop_ev = threading.Event()
        self._thr = threading.Thread(target=self._run, name="wp3_async_mask_worker", daemon=True)
        self._thr.start()

    def submit(
        self,
        epoch_id: int,
        observation_stamp_ns: int,
        calib_epoch: str,
        region_key: str,
        rgb_img: np.ndarray,
        box_xyxy: tuple[int, int, int, int],
    ) -> bool:
        """Enqueue latest hard-crop job, dropping any stale unstarted job on backpressure."""
        if not self.challenger.available:
            return False
        item = (int(epoch_id), int(observation_stamp_ns), str(calib_epoch), str(region_key), rgb_img.copy(), box_xyxy)
        try:
            self._q.put_nowait(item)
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(item)
            except queue.Full:
                return False
        return True

    def _run(self) -> None:
        while not self._stop_ev.is_set():
            try:
                ep, obs_ns, calib, rkey, rgb, box = self._q.get(timeout=0.05)
            except queue.Empty:
                continue
            t0 = time.perf_counter()
            out = self.challenger.propose_masks_from_roi_boxes(rgb, [box])
            res = MaskProposalResult(
                epoch_id=ep,
                observation_stamp_ns=obs_ns,
                calib_epoch=calib,
                region_key=rkey,
                masks_u8=list(out.get("masks", [])),
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 3),
                backend=str(out.get("backend", self.challenger.backend)),
            )
            with self._lock:
                self._results[rkey] = res

    def poll_matching(
        self,
        epoch_id: int,
        calib_epoch: str,
        region_key: str,
        *,
        max_epoch_lag: int = 2,
    ) -> MaskProposalResult | None:
        """Return latest mask result ONLY if its observation epoch and calib_epoch remain current."""
        with self._lock:
            res = self._results.get(region_key)
        if res is None or res.calib_epoch != calib_epoch:
            return None
        if (epoch_id - res.epoch_id) > max_epoch_lag:
            return None
        return res

    def stop(self) -> None:
        self._stop_ev.set()
        if self._thr.is_alive():
            self._thr.join(timeout=0.5)

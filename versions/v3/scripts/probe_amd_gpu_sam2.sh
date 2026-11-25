#!/usr/bin/env bash
set -euo pipefail

echo "=== AMD RX 7650 GRE (gfx1102) GPU / SAM2 / ONNX Probe ==="
VENV_PY="${WP3_SAM2_PYTHON:-}"
PY_BIN="python3"
if [[ -x "$VENV_PY" ]]; then
    PY_BIN="$VENV_PY"
fi
echo "Using Python: $PY_BIN"

"$PY_BIN" - <<'PY'
import importlib.util, time
for pkg in ("torch", "sam2", "onnxruntime", "numpy", "cv2"):
    print(f"  pkg {pkg}: {importlib.util.find_spec(pkg) is not None}")
if importlib.util.find_spec("torch") is not None:
    import torch
    print("  torch:", torch.__version__, "cuda_avail:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("  device[0]:", torch.cuda.get_device_name(0))
        a = torch.randn((1024, 1024), device="cuda")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = (a @ a).sum().item()
        torch.cuda.synchronize()
        print(f"  1024x1024 fp32 matmul ms: {(time.perf_counter()-t0)*1000:.2f}")
if importlib.util.find_spec("onnxruntime") is not None:
    import onnxruntime as ort
    print("  ort providers:", ort.get_available_providers())
PY

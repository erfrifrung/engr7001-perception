#!/usr/bin/env bash
# WP3 V3 — hardware / driver / runtime verification for the lab desktop (Ryzen 5 5600 + Radeon "7650").
# Read-only diagnostics: no installs, no robot, no network calls. Writes a JSON-ish report to stdout and $OUT.
# Usage:  bash scripts/amd_hw_check.sh [OUT=amd_hw_report.txt]
set -uo pipefail
OUT=${OUT:-amd_hw_report.txt}
exec > >(tee "$OUT") 2>&1
echo "=== WP3 V3 AMD hardware check $(date -Is) ==="
echo "--- H0 host"; uname -a; cat /etc/os-release 2>/dev/null | grep -E '^(NAME|VERSION)='
echo "--- H6 CPU"; lscpu 2>/dev/null | grep -E 'Model name|^CPU\(s\)|Thread|MHz' ; nproc
echo "--- H1/H2 GPU PCI (expect 1002:7480 Navi 33 family for RX 7600/7650 GRE; record subsystem + revision)"
lspci -nn 2>/dev/null | grep -Ei 'vga|display|3d' ; lspci -vnn 2>/dev/null | grep -A3 -Ei 'navi|radeon' | grep -Ei 'subsystem|rev' | head -4
echo "--- kernel driver"; lsmod 2>/dev/null | grep -E '^amdgpu' ; cat /sys/module/amdgpu/version 2>/dev/null
for c in /sys/class/drm/card*/device; do
  [ -f "$c/vendor" ] || continue
  echo "$c vendor=$(cat $c/vendor) device=$(cat $c/device) rev=$(cat $c/revision 2>/dev/null) vram_total_B=$(cat $c/mem_info_vram_total 2>/dev/null) gtt_B=$(cat $c/mem_info_gtt_total 2>/dev/null)"
done
echo "--- H3 Vulkan (Mesa RADV expected)"; (vulkaninfo --summary 2>/dev/null || echo "vulkaninfo not installed (apt: vulkan-tools)") | grep -Ei 'deviceName|driverName|driverInfo|apiVersion|conformance' | head -8
echo "--- OpenGL"; (glxinfo -B 2>/dev/null || echo "glxinfo not installed") | grep -Ei 'renderer|version' | head -3
echo "--- H4 ROCm runtime (gfx target must read gfx1102 WITHOUT HSA_OVERRIDE_GFX_VERSION)"
echo "HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-<unset>}"
ls -d /opt/rocm* 2>/dev/null; cat /opt/rocm/.info/version 2>/dev/null
(rocminfo 2>/dev/null || echo "rocminfo not available") | grep -Ei 'Marketing Name|Name: *gfx|Compute Unit|Max Clock' | head -8
(rocm-smi --showproductname --showmeminfo vram 2>/dev/null || amd-smi static 2>/dev/null || echo "rocm-smi/amd-smi not available") | head -20
echo "--- H5 Python frameworks (whatever is installed; nothing is installed by this script)"
PY=${PY:-python3}
"$PY" - <<'PY'
import importlib.util, json, time
rep = {}
for pkg in ("numpy", "cv2", "scipy", "torch", "onnxruntime", "migraphx", "ncnn", "numba"):
    rep[pkg] = importlib.util.find_spec(pkg) is not None
print("packages", json.dumps(rep))
if rep["torch"]:
    import torch
    info = {"torch": torch.__version__, "hip": getattr(torch.version, "hip", None), "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(name=p.name, gcn_arch=getattr(p, "gcnArchName", "?"), vram_GB=round(p.total_memory / 1e9, 2),
                    arch_list=torch.cuda.get_arch_list() if hasattr(torch.cuda, "get_arch_list") else None)
        a = torch.randn(1024, 1024, device="cuda"); torch.cuda.synchronize()
        t0 = time.perf_counter(); b = (a @ a); torch.cuda.synchronize(); info["matmul1024_ms"] = round((time.perf_counter() - t0) * 1e3, 2)
        ref = (a.cpu() @ a.cpu()); info["matmul_max_abs_err_vs_cpu"] = float((b.cpu() - ref).abs().max())
    print("torch", json.dumps(info))
if rep["onnxruntime"]:
    import onnxruntime as ort
    print("onnxruntime", ort.__version__, "providers", ort.get_available_providers())
PY
echo "--- verdict hints"
echo " * gfx1102 + ROCm >= 7.12 (preview) or a production release that lists RX 7600 -> HIP/MIGraphX route may be tested"
echo " * HSA_OVERRIDE_GFX_VERSION needed -> UNSUPPORTED route: keep CPU default, report as unofficial"
echo " * RADV present -> Vulkan route (ncnn) is the vendor-neutral fallback for small CNNs"
echo "=== end ==="

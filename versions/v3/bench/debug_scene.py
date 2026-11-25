"""Run V3 on one saved scene and print region diagnostics (developer tool)."""
import json, sys
from pathlib import Path
import numpy as np
V3 = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(V3))
from wp3_live import ViewInput, preset_config, run_perception_epoch
from wp3_live.dwell_gate import DwellStatus
from wp3_live.geometry_cache import CameraModelCache
scdir, pattern = sys.argv[1], sys.argv[2]
solver = sys.argv[3] if len(sys.argv) > 3 else "prove"
idx = json.loads((Path(scdir) / "index.json").read_text())
cache = CameraModelCache()
for sc in idx:
    if pattern not in sc["file"]:
        continue
    z = np.load(Path(scdir) / sc["file"])
    cw = cache.get("wrist", 1280, 720, z["Kw"], np.zeros(5), "w", 0.001)
    cs = cache.get("tripod", 640, 480, z["Ks"], np.zeros(5), "s", 0.001)
    cfg = preset_config("printed_jenga"); cfg.v3_solver = solver
    r = run_perception_epoch(wrist=ViewInput("wrist", cw, z["rgb_w"], z["d_w"], z["Tw"], 1, 1),
                             side=ViewInput("tripod", cs, z["rgb_s"], z["d_s"], z["Ts"], 1, 1), config=cfg,
                             policy="merged_corroborated", dwell=DwellStatus(True, True, True, True, 0, .1, .2, 80, "ok"))
    v3 = [s["v3"] for s in r.evidence_log["split_logs"] if "v3" in s]
    print(sc["file"], "poses", len(r.poses), r.pose_sources, [a.reason for a in r.ambiguous_clusters], v3, r.stage_timings_ms.get("v3_regions_ms"))

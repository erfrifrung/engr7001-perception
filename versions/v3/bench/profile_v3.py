"""cProfile the real epoch path (V3 prove) on hard scenes (developer tool)."""
import cProfile, pstats, sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wp3_live import ViewInput, preset_config, run_perception_epoch
from wp3_live.dwell_gate import DwellStatus
from wp3_live.geometry_cache import CameraModelCache
c = CameraModelCache(); D = DwellStatus(True, True, True, True, 0, .1, .2, 80, 'ok')
scdir = Path(sys.argv[1]); files = sys.argv[2:]
Z = [np.load(scdir / f) for f in files]
cfg = preset_config('printed_jenga')
def run():
    for z in Z:
        cw = c.get('wrist', 1280, 720, z['Kw'], np.zeros(5), 'w', 0.001); cs = c.get('tripod', 640, 480, z['Ks'], np.zeros(5), 's', 0.001)
        run_perception_epoch(wrist=ViewInput('wrist', cw, z['rgb_w'], z['d_w'], z['Tw'], 1, 1), side=ViewInput('tripod', cs, z['rgb_s'], z['d_s'], z['Ts'], 1, 1), config=cfg, policy='merged_corroborated', dwell=D)
run()
pr = cProfile.Profile(); pr.enable(); run(); pr.disable()
pstats.Stats(pr).sort_stats('tottime').print_stats(22)

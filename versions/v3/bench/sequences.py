"""Render dynamic RGB-D sequences (15 fps) with per-frame ground truth, robot link stamps and object identities."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scenes import H, L, W, box, cameras, render  # noqa: E402

DT_NS = 66_666_667


def _rot_xy(x, y, deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return c * x - s * y, s * x + c * y


def arm_objects(x, y, *, fingers: bool, finger_dy=0.0):
    """Palm box above the table and optionally one finger box reaching down to block height."""
    objs = [box(x, y, 0, 0.045, 0.06, 0.07, 0.06, (60, 60, 70), kind="arm")]
    links = {"link6": ([x, y, 0.25], [0, 0, 0, 1]), "link_tcp": ([x, y + finger_dy, 0.003], [0, 0, 0, 1])}
    if fingers:
        objs.append(box(x, y + finger_dy, 0, 0.003, 0.05, 0.014, 0.014, (70, 70, 80), kind="arm"))
    return objs, links


def seq_defs():
    S = {}
    yaw = 25.0
    pair = []
    for i, dy in enumerate((-W / 2 - 0.0003, W / 2 + 0.0003)):
        x, y = _rot_xy(0.0, dy, yaw)
        pair.append(box(x, y, yaw, 0, H, seam=True, oid=i, seam_w=0.001))

    def frames(n, fn):
        return [fn(k) for k in range(n)]

    S["stationary_seamed_pair"] = frames(24, lambda k: (pair, None))

    def sweep(fing):
        def f(k):
            x = -0.14 + 0.012 * k
            arm, links = arm_objects(x, 0.045, fingers=fing, finger_dy=-0.012)
            return pair + arm, links
        return f
    S["arm_sweep_fingers"] = frames(24, sweep(True))
    S["arm_sweep_overhead"] = frames(24, sweep(False))

    A = box(-0.045, 0.0, 10, 0, H, seam=True, oid=0)
    B0 = box(0.045, 0.0, 10, 0, H, seam=True, oid=1)

    def hidden_move(k):
        B = dict(B0)
        if k >= 9:
            B["cx"] += 0.003; B["yaw"] += np.radians(2.0)
        objs = [A, B]
        links = None
        if 6 <= k <= 12:
            arm, links = arm_objects(B0["cx"], B0["cy"], fingers=False)
            objs += [dict(arm[0], lx=0.11, wy=0.06)]
        return objs, links
    S["hidden_move_3mm"] = frames(24, hidden_move)

    def hidden_swap(k):
        a, b = dict(A), dict(B0)
        if k >= 9:          # identical blocks exchange places while both are hidden
            a["cx"], b["cx"] = B0["cx"], A["cx"]
        objs = [a, b]
        links = None
        if 6 <= k <= 12:
            objs.append(box(0.0, 0.0, 0, 0.045, 0.06, 0.2, 0.07, (60, 60, 70), kind="arm"))
        return objs, links
    S["hidden_swap"] = frames(24, hidden_swap)

    def drift(k):
        b = dict(A)
        b["cx"] += 0.0007 * max(0, k - 4)        # slow push: 0.7 mm per frame from frame 5
        return [b, dict(B0)], None
    S["slow_push_drift"] = frames(24, drift)
    return S


def render_all(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    cw, Tw, cs, Ts = cameras()
    meta = {}
    for name, frames in seq_defs().items():
        rng = np.random.default_rng(abs(hash(name)) % (2 ** 31))
        rec = []
        for k, (objs, links) in enumerate(frames):
            rw, dw = render(cw, Tw, objs, rng, noise_m=0.00045)
            rs, ds = render(cs, Ts, objs, rng, noise_m=0.0008)
            fn = f"{name}_{k:03d}.npz"
            np.savez_compressed(out / fn, rgb_w=rw, d_w=dw, rgb_s=rs, d_s=ds, Tw=Tw, Ts=Ts, Kw=cw.K_f64, Ks=cs.K_f64)
            rec.append(dict(file=fn, stamp_ns=(k + 1) * DT_NS, links=links,
                            blocks=[o for o in objs if o["kind"] == "block"],
                            arm_present=any(o["kind"] == "arm" for o in objs)))
        meta[name] = rec
    (out / "sequences.json").write_text(json.dumps(meta, indent=1))
    return meta


if __name__ == "__main__":
    m = render_all(Path(sys.argv[1]))
    print({k: len(v) for k, v in m.items()})

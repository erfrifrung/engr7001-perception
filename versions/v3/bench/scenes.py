"""Deterministic ray-cast RGB-D scene generator for WP3 V3 benchmarks (synthetic ground truth only).

Objects are oriented boxes (blocks, clutter, arm links). Cameras match V2's synthetic rig (wrist 1280x720 looking
down from 0.45 m, fixed side 640x480).  Fuzz: yaw jitter, sub-cell shifts, thin gaps, block size tolerance, colour,
depth noise + 1 mm quantisation, random holes, edge dropout, calibration jitter of the transform given to the
pipeline (render uses the true transform).
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

V3 = Path(__file__).resolve().parents[1]
if str(V3) not in sys.path:
    sys.path.insert(0, str(V3))

from wp3_live.geometry_cache import CameraModelCache  # noqa: E402

H = 0.015
L, W = 0.075, 0.025


def cameras():
    cache = CameraModelCache()
    Kw = np.array([[918.0, 0, 640.0], [0, 918.0, 360.0], [0, 0, 1]])
    cw = cache.get("wrist", 1280, 720, Kw, np.zeros(5), "wrist_color_optical_frame", 0.001)
    Tw = np.array([[1.0, 0, 0, 0], [0, -1.0, 0, 0], [0, 0, -1.0, 0.45], [0, 0, 0, 1]])
    Ks = np.array([[606.26, 0, 320.0], [0, 606.26, 240.0], [0, 0, 1]])
    cs = cache.get("tripod", 640, 480, Ks, np.zeros(5), "tripod_color_optical_frame", 0.001)
    pos = np.array([-0.38, -0.38, 0.37])
    fwd = -pos / np.linalg.norm(pos)
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
    down = np.cross(fwd, right); down /= np.linalg.norm(down)
    Ts = np.eye(4); Ts[:3, 0], Ts[:3, 1], Ts[:3, 2], Ts[:3, 3] = right, down, fwd, pos
    return cw, Tw, cs, Ts


def box(cx, cy, yaw_deg, z0, hz, lx=L, wy=W, color=(196, 170, 128), kind="block", seam=False, **extra):
    return dict(cx=float(cx), cy=float(cy), yaw=float(np.radians(yaw_deg)), z0=float(z0), z1=float(z0 + hz),
                lx=float(lx), wy=float(wy), color=tuple(int(c) for c in color), kind=kind, seam=seam, **extra)


def render(cam, T_tc, objects, rng, *, noise_m, hole_frac=0.0, edge_drop=0.0, table_rgb=(210, 210, 195),
           return_ids=False):
    Hh, Ww = cam.height, cam.width
    d = (cam.rays_f32.reshape(-1, 3) @ T_tc[:3, :3].T).astype(np.float64)
    o = T_tc[:3, 3]
    t_best = np.where(d[:, 2] < 0, -o[2] / d[:, 2], np.inf)
    rgb = np.empty((len(d), 3), np.float32)
    rgb[:] = table_rgb
    hit_id = np.full(len(d), -1, np.int32)
    for oi, ob in enumerate(objects):
        c, s = np.cos(ob["yaw"]), np.sin(ob["yaw"])
        Rt = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1.0]])
        ol = Rt @ (o - np.array([ob["cx"], ob["cy"], 0.5 * (ob["z0"] + ob["z1"])]))
        dl = d @ Rt.T
        he = np.array([ob["lx"] / 2, ob["wy"] / 2, 0.5 * (ob["z1"] - ob["z0"])])
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (-he[None, :] - ol[None, :]) / dl
            t2 = (he[None, :] - ol[None, :]) / dl
        tn = np.minimum(t1, t2)
        tf = np.maximum(t1, t2)
        tn = np.where(np.isnan(tn), -np.inf, tn)
        tf = np.where(np.isnan(tf), np.inf, tf)
        tnear = tn.max(axis=1)
        tfar = tf.min(axis=1)
        hit = (tnear < tfar) & (tfar > 0) & (tnear > 0.05) & (tnear < t_best)
        if not np.any(hit):
            continue
        t_best[hit] = tnear[hit]
        ax = np.argmax(tn[hit], axis=1)
        hid = np.where(ax == 2, oi, -2 - oi)
        hit_id[hit] = hid
        col = np.array(ob["color"], np.float32)
        shade = np.where(ax == 2, 1.0, 0.82)[:, None]
        cc = col[None, :] * shade
        if ob.get("seam"):
            p = ol[None, :] + dl[hit] * tnear[hit][:, None]
            edge = (np.minimum(he[0] - np.abs(p[:, 0]), he[1] - np.abs(p[:, 1])) < ob.get("seam_w", 0.0007)) & (ax == 2)
            cc[edge] = (60, 55, 50)
        if ob.get("stripe_x") is not None:
            p = ol[None, :] + dl[hit] * tnear[hit][:, None]
            on = (np.abs(p[:, 0] - ob["stripe_x"]) < 0.0008) & (ax == 2)
            cc[on] = (55, 50, 45)
        if ob.get("stripes_y"):   # V3.1 adversary: periodic dark lines along the long axis (FDM-like texture)
            p = ol[None, :] + dl[hit] * tnear[hit][:, None]
            on = np.zeros(len(p), bool)
            for s in ob["stripes_y"]:
                on |= np.abs(p[:, 1] - s) < 0.0008
            cc[on & (ax == 2)] = (55, 50, 45)
        rgb[hit] = cc
    depth = np.where(np.isfinite(t_best), t_best, 0.0).reshape(Hh, Ww)
    depth = depth + rng.normal(0.0, noise_m, depth.shape)
    if edge_drop > 0:
        gy, gx = np.gradient(depth)
        jump = np.hypot(gx, gy) > 0.006
        depth[jump & (rng.random(depth.shape) < edge_drop)] = 0.0
    if hole_frac > 0:
        n = int(hole_frac * 40)
        for _ in range(n):
            u, v = rng.integers(0, Ww), rng.integers(0, Hh)
            ru, rv = rng.integers(4, 18), rng.integers(4, 14)
            yy, xx = np.ogrid[:Hh, :Ww]
            depth[((xx - u) / ru) ** 2 + ((yy - v) / rv) ** 2 <= 1.0] = 0.0
    d16 = np.clip(np.round(depth * 1000.0), 0, 65535).astype(np.uint16)
    img = np.clip(rgb.reshape(Hh, Ww, 3) + rng.normal(0, 2.0, (Hh, Ww, 3)), 0, 255).astype(np.uint8)
    if return_ids:
        return img, d16, hit_id.reshape(Hh, Ww)
    return img, d16


def jitter_T(T, rng, trans_m, rot_deg):
    if trans_m <= 0 and rot_deg <= 0:
        return T.copy()
    from scipy.spatial.transform import Rotation
    dR = Rotation.from_rotvec(np.radians(rot_deg) * rng.normal(size=3) / np.sqrt(3)).as_matrix()
    Tj = T.copy()
    Tj[:3, :3] = dR @ T[:3, :3]
    Tj[:3, 3] = T[:3, 3] + trans_m * rng.normal(size=3) / np.sqrt(3)
    return Tj


# ------------------------------------------------------------------------------------------------ families

def _rot(objs, yaw_deg, dx, dy):
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    out = []
    for b in objs:
        b = dict(b)
        x, y = b["cx"], b["cy"]
        b["cx"], b["cy"] = c * x - s * y + dx, s * x + c * y + dy
        b["yaw"] += np.radians(yaw_deg)
        out.append(b)
    return out


def family(name, rng, fz):
    """Return (objects, expected) where expected['resolvable'] marks blocks a safe system *may* publish and
    expected['must_abstain'] marks scenes whose top view is genuinely ambiguous (no seams)."""
    gap = lambda: float(rng.uniform(0, fz["gap_mm"])) / 1000.0  # noqa: E731
    sz = lambda: (L + rng.normal(0, fz["size_mm"] / 1000), W + rng.normal(0, fz["size_mm"] / 1000))  # noqa: E731
    col = tuple(int(c) for c in np.clip(np.array([196, 170, 128]) + rng.normal(0, fz["col"], 3), 40, 250))
    yaw = float(rng.uniform(-90, 90))
    dx, dy = rng.uniform(-0.02, 0.02, 2)
    seam = bool(rng.random() < fz["seam_p"])
    note = {}

    def B(cx, cy, yd=0.0, z0=0.0, **kw):
        lx, wy = sz()
        return box(cx, cy, yd + rng.normal(0, fz["yaw_deg"]), z0, H, lx, wy, col, seam=seam, **kw)

    if name == "flush_pair":
        g = gap()
        objs = [B(0, -(W + g) / 2), B(0, (W + g) / 2)]
    elif name == "flush_triple":
        objs = [B(0, -W - gap()), B(0, 0), B(0, W + gap())]
        note["square"] = True
    elif name == "end_to_end":
        g = gap()
        objs = [B(-(L + g) / 2, 0), B((L + g) / 2, 0)]
    elif name == "stagger":
        off = float(rng.choice([1, 2, 3, 5, 8, 12, 20])) / 1000
        note["stagger_mm"] = off * 1000
        objs = [B(off, -W / 2 - gap() / 2), B(0, W / 2 + gap() / 2)]
    elif name == "L_T":
        if rng.random() < 0.5:
            objs = [B(0, 0), B(L / 2 + W / 2 + gap(), W, 90)]
        else:
            objs = [B(0, 0), B(0, W / 2 + L / 2 + gap(), 90)]
    elif name == "mixed_yaw":
        a = float(rng.uniform(5, 15))
        note["delta_yaw"] = a
        objs = [B(0, -W / 2), box(0.004, W / 2 + 0.003, a, 0, H, color=col, seam=seam)]
    elif name == "false_seam_single":
        objs = [box(0, 0, 0, 0, H, color=col, stripe_x=float(rng.uniform(-0.02, 0.02)))]
    elif name == "separated":
        objs = [B(0, -0.03), B(0.01, 0.035, float(rng.uniform(-30, 30)))]
    elif name == "stack2":
        objs = [B(0, 0), B(rng.normal(0, 0.002), rng.normal(0, 0.002), float(rng.choice([0, 90])), z0=H)]
        note["upper_index"] = 1
    elif name == "bridge":
        objs = [B(0, -0.025), B(0, 0.025), B(0, 0, 90, z0=H)]
        note["upper_index"] = 2
    elif name == "holes_pair":
        g = gap()
        objs = [B(0, -(W + g) / 2), B(0, (W + g) / 2)]
        note["holes"] = True
    elif name == "clutter_slab":
        # block-footprint slab of a non-block height (8 mm) next to a block, and a 30 mm cube
        objs = [B(0, -0.03), box(0, 0.025, 0, 0, 0.008, L, 2 * W, (150, 150, 160), kind="clutter"),
                box(0.06, 0.06, 20, 0, 0.03, 0.03, 0.03, (120, 120, 200), kind="clutter")]
    elif name == "clutter_block_like":
        # adversarial: a 75 x 50 x 15 mm slab is geometrically identical to a flush pair from the top
        objs = [box(0, 0, 0, 0, H, L, 2 * W, (150, 150, 160), kind="clutter")]
        note["indistinguishable_from_pair"] = True
    elif name == "occluded_pair":
        g = gap()
        objs = [B(0, -(W + g) / 2), B(0, (W + g) / 2)]
        frac = float(rng.uniform(0.3, 0.9))
        note["arm_cover_frac"] = frac
        # arm/gripper box above the +y block, covering part of it
        objs.append(box(-L / 2 + frac * L / 2, W, 0, 0.045, 0.06, frac * L, 0.05, (60, 60, 70), kind="arm"))
    else:
        raise KeyError(name)
    objs = _rot(objs, yaw, dx, dy)
    return objs, note


FAMILIES = ["flush_pair", "flush_triple", "end_to_end", "stagger", "L_T", "mixed_yaw", "false_seam_single",
            "separated", "stack2", "bridge", "holes_pair", "clutter_slab", "clutter_block_like", "occluded_pair"]

FUZZ = {
    "tune": dict(gap_mm=0.3, size_mm=0.1, col=6, yaw_deg=0.3, seam_p=0.3, noise_w=0.00045, noise_s=0.0008,
                 holes=0.0, edge_drop=0.0, calib_mm=0.0, calib_deg=0.0),
    "heldout": dict(gap_mm=1.0, size_mm=0.3, col=15, yaw_deg=0.8, seam_p=0.3, noise_w=0.0006, noise_s=0.001,
                    holes=0.15, edge_drop=0.15, calib_mm=1.0, calib_deg=0.15),
}


def make_scene(split: str, idx: int, fam: str):
    seed = (1000 if split == "tune" else 5000) + idx
    rng = np.random.default_rng(seed)
    fz = FUZZ[split]
    objs, note = family(fam, rng, fz)
    cw, Tw, cs, Ts = cameras()
    holes = fz["holes"] if fam != "holes_pair" else 0.6
    rgb_w, d_w = render(cw, Tw, objs, rng, noise_m=fz["noise_w"], hole_frac=holes, edge_drop=fz["edge_drop"])
    rgb_s, d_s = render(cs, Ts, objs, rng, noise_m=fz["noise_s"], hole_frac=fz["holes"], edge_drop=fz["edge_drop"])
    Tw_g = jitter_T(Tw, rng, fz["calib_mm"] / 1000, fz["calib_deg"])
    Ts_g = jitter_T(Ts, rng, fz["calib_mm"] / 1000, fz["calib_deg"])
    blocks = [b for b in objs if b["kind"] == "block"]
    return dict(seed=seed, split=split, family=fam, note=note, objects=objs, blocks=blocks,
                rgb_w=rgb_w, d_w=d_w, rgb_s=rgb_s, d_s=d_s, Tw=Tw_g, Ts=Ts_g, Kw=cw.K_f64, Ks=cs.K_f64)


def write_split(out_dir: Path, split: str, per_family: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    i = 0
    for fam in FAMILIES:
        for k in range(per_family):
            sc = make_scene(split, i, fam)
            fn = f"{split}_{i:04d}_{fam}.npz"
            np.savez_compressed(out_dir / fn, rgb_w=sc["rgb_w"], d_w=sc["d_w"], rgb_s=sc["rgb_s"], d_s=sc["d_s"],
                                Tw=sc["Tw"], Ts=sc["Ts"], Kw=sc["Kw"], Ks=sc["Ks"])
            index.append(dict(file=fn, seed=sc["seed"], family=fam, note=sc["note"], blocks=sc["blocks"],
                              objects=sc["objects"]))
            i += 1
    (out_dir / "index.json").write_text(json.dumps(index, indent=1))
    return index


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", choices=["tune", "heldout"], required=True)
    ap.add_argument("--per-family", type=int, default=4)
    a = ap.parse_args()
    idx = write_split(Path(a.out), a.split, a.per_family)
    print(len(idx), "scenes written")

"""Printed 75x25x15 mm synthetic benchmark & physical metrology suite (step, flush, stripe, rotated square, lower slots)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np

IMPL_ROOT = Path(__file__).resolve().parents[1]
if str(IMPL_ROOT) not in sys.path:
    sys.path.insert(0, str(IMPL_ROOT))

from wp3_live import ViewInput, configure_single_thread_libs, preset_config, run_perception_epoch  # noqa: E402
from wp3_live.dwell_gate import DwellStatus  # noqa: E402
from wp3_live.geometry_cache import CameraModelCache  # noqa: E402
from wp3_live.overlay import render_overlay  # noqa: E402
from wp3_live.policies import ALL_POLICIES  # noqa: E402
from wp3_live.scoring import sym_yaw_err_deg  # noqa: E402

configure_single_thread_libs()


def make_synthetic_cameras(cache: CameraModelCache):
    # Overhead wrist 1280x720 looking down from z = 0.45 m along -Z_table
    Kw = np.array([[918.0, 0.0, 640.0], [0.0, 918.0, 360.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    Dw = np.zeros(5, dtype=np.float64)
    cam_w = cache.get("wrist", 1280, 720, Kw, Dw, "wrist_color_optical_frame", 0.001)
    # T_table_wrist: cam X=table X, cam Y=-table Y, cam Z=-table Z (z=0.45)
    Tw = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.45],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)

    # Fixed side camera 640x480 @ ~45 deg azimuth, ~55 deg elevation, ~0.65 m range looking at table origin
    Ks = np.array([[606.26, 0.0, 320.0], [0.0, 606.26, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    Ds = np.zeros(5, dtype=np.float64)
    cam_s = cache.get("tripod", 640, 480, Ks, Ds, "tripod_color_optical_frame", 0.001)
    pos_s = np.array([-0.38, -0.38, 0.37], dtype=np.float64)
    fwd = -pos_s / np.linalg.norm(pos_s)
    up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(fwd, up_hint); right /= np.linalg.norm(right)
    down = np.cross(fwd, right); down /= np.linalg.norm(down)
    Ts = np.eye(4, dtype=np.float64)
    Ts[:3, 0] = right; Ts[:3, 1] = down; Ts[:3, 2] = fwd; Ts[:3, 3] = pos_s
    return cam_w, Tw, cam_s, Ts


def render_synthetic_scene(
    cam_w, Tw, cam_s, Ts,
    blocks: list[dict],
    seams_top: list[dict] | None = None,
    end_breaks: list[dict] | None = None,
    occlude_side: bool = False,
):
    """Render deterministic noisy synthetic RGB-D views for printed 75x25x15 mm blocks."""
    seams_top = seams_top or []
    end_breaks = end_breaks or []
    rng = np.random.default_rng(7)

    def render_cam(cam, T_tc, is_side: bool):
        H, W = cam.height, cam.width
        rgb = np.full((H, W, 3), 210, dtype=np.uint8)
        rgb[:, :, 2] = 195  # pale tabletop
        T_ct = np.linalg.inv(T_tc)
        rays_tbl = (cam.rays_f32 @ T_tc[:3, :3].T)
        o_tbl = T_tc[:3, 3]
        # Intersect table z = 0
        s_tbl = (0.0 - o_tbl[2]) / rays_tbl[:, :, 2]
        hit_xy = o_tbl[:2] + rays_tbl[:, :, :2] * s_tbl[:, :, None]
        depth_m = (s_tbl * 1.0).astype(np.float32)  # approx range ~ cam_z for optical ray
        # Convert ray parameter s to optical Z: P_cam_z = s (since rays_f32[:,:,2] == 1!)
        depth_m = s_tbl.astype(np.float32)

        for blk in blocks:
            cx, cy, z0, z1 = blk["cx"], blk["cy"], blk["z0"], blk["z1"]
            hl, hw = blk["L"] / 2.0, blk["W"] / 2.0
            yaw = blk["yaw_rad"]
            col = blk.get("color", (195, 192, 186))
            side_col = tuple(max(40, c - 18) for c in col)
            c, s = float(np.cos(yaw)), float(np.sin(yaw))
            # Top plane z = z1 intersection
            s_top = (z1 - o_tbl[2]) / rays_tbl[:, :, 2]
            px = o_tbl[0] + rays_tbl[:, :, 0] * s_top - cx
            py = o_tbl[1] + rays_tbl[:, :, 1] * s_top - cy
            lx = c * px + s * py
            ly = -s * px + c * py
            on_top = (s_top > 0.15) & (s_top < depth_m) & (np.abs(lx) <= hl) & (np.abs(ly) <= hw)
            depth_m[on_top] = s_top[on_top]
            rgb[on_top] = col

            # Render 4 vertical side faces for realistic side-camera rays & visibility checks
            for face_dir, fdist, tang_dir, t_lim in (
                (np.array([c, s, 0.0]), hl, np.array([-s, c, 0.0]), hw),
                (np.array([-c, -s, 0.0]), hl, np.array([-s, c, 0.0]), hw),
                (np.array([-s, c, 0.0]), hw, np.array([c, s, 0.0]), hl),
                (np.array([s, -c, 0.0]), hw, np.array([c, s, 0.0]), hl),
            ):
                denom = rays_tbl[:, :, 0] * face_dir[0] + rays_tbl[:, :, 1] * face_dir[1]
                p0_rel = np.array([cx, cy, 0.0]) + fdist * face_dir - o_tbl
                with np.errstate(divide="ignore", invalid="ignore"):
                    s_f = float(np.dot(p0_rel, face_dir)) / np.where(np.abs(denom) > 1e-5, denom, np.nan)
                pf = o_tbl + rays_tbl * s_f[:, :, None]
                rel_t = (pf[:, :, 0] - cx) * tang_dir[0] + (pf[:, :, 1] - cy) * tang_dir[1]
                on_f = (
                    np.isfinite(s_f) & (s_f > 0.15) & (s_f < depth_m)
                    & (np.abs(rel_t) <= t_lim) & (pf[:, :, 2] >= z0) & (pf[:, :, 2] <= z1)
                )
                depth_m[on_f] = s_f[on_f]
                rgb[on_f] = side_col

        # Draw dark seam / false-stripe lines on block tops (world XY thickness ~ 1.0 mm)
        for sm in seams_top:
            # sm: p0_xy, p1_xy, z, width_m, dark_val
            p0 = np.array(sm["p0"], dtype=np.float64)
            p1 = np.array(sm["p1"], dtype=np.float64)
            dvec = p1 - p0; L = float(np.linalg.norm(dvec)); udir = dvec / max(1e-6, L)
            nrm = np.array([-udir[1], udir[0]])
            z_sm = sm["z"]
            s_sm = (z_sm - o_tbl[2]) / rays_tbl[:, :, 2]
            hit = o_tbl[:2] + rays_tbl[:, :, :2] * s_sm[:, :, None] - p0
            proj_u = hit @ udir
            proj_v = hit @ nrm
            on_seam = (
                (np.abs(depth_m - s_sm) < 0.002)
                & (proj_u >= 0.0) & (proj_u <= L)
                & (np.abs(proj_v) <= sm.get("half_w_m", 0.0008))
            )
            rgb[on_seam] = sm.get("color", (45, 42, 40))

        # Draw vertical end-face breaks on side view (physical bevel/gap between blocks)
        if is_side:
            for eb in end_breaks:
                rvec, _ = cv2.Rodrigues(T_ct[:3, :3])
                pts = np.array([
                    [eb["x"], eb["y"], eb["z0"]],
                    [eb["x"], eb["y"], eb["z1"]],
                ], dtype=np.float64)
                uv, _ = cv2.projectPoints(pts, rvec, T_ct[:3, 3:4], cam.K_f64, cam.D_f64)
                uv = np.round(uv.reshape(-1, 2)).astype(int)
                cv2.line(rgb, tuple(uv[0]), tuple(uv[1]), (25, 22, 20), thickness=2)

            if occlude_side:
                # Occluder hand/screen blocking view of the blocks in side camera
                cv2.rectangle(rgb, (W // 4, H // 4), (3 * W // 4, 3 * H // 4), (90, 90, 140), -1)
                depth_m[H // 4: 3 * H // 4, W // 4: 3 * W // 4] = 0.25

        # Add realistic D435 spatial depth noise (~0.45 mm wrist, ~0.8 mm side)
        sigma = 0.00045 if not is_side else 0.0008
        noisy_m = depth_m + rng.normal(0.0, sigma, size=depth_m.shape).astype(np.float32)
        depth_u16 = np.clip(np.round(noisy_m * 1000.0), 0, 65535).astype(np.uint16)
        return rgb, depth_u16

    rgb_w, d_w = render_cam(cam_w, Tw, False)
    rgb_s, d_s = render_cam(cam_s, Ts, True)
    return rgb_w, d_w, rgb_s, d_s


def build_printed_scenarios():
    """Construct the complete printed-Jenga physical validation suite."""
    H = 0.015
    return [
        {
            "name": "single_rotated_30deg",
            "expected_count": 1, "expected_yaw": 30.0, "must_resolve": True, "must_abstain": False,
            "blocks": [{"cx": 0.0, "cy": 0.0, "z0": 0.0, "z1": H, "L": 0.075, "W": 0.025, "yaw_rad": np.radians(30.0)}],
        },
        {
            "name": "flush_pair_corroborated",
            "expected_count": 2, "expected_yaw": 0.0, "must_resolve": True, "must_abstain": False,
            "blocks": [
                {"cx": 0.0, "cy": -0.0125, "z0": 0.0, "z1": H, "L": 0.075, "W": 0.025, "yaw_rad": 0.0},
                {"cx": 0.0, "cy": 0.0125, "z0": 0.0, "z1": H, "L": 0.075, "W": 0.025, "yaw_rad": 0.0},
            ],
            "seams": [{"p0": (-0.037, 0.0), "p1": (0.037, 0.0), "z": H, "half_w_m": 0.0009}],
            "end_breaks": [{"x": -0.0375, "y": 0.0, "z0": 0.0, "z1": H}, {"x": 0.0375, "y": 0.0, "z0": 0.0, "z1": H}],
        },
        {
            "name": "intact_1_stripe_single",
            "expected_count": 1, "expected_yaw": 0.0, "must_resolve": True, "must_abstain": False, "is_negative": True,
            "blocks": [{"cx": 0.0, "cy": 0.0, "z0": 0.0, "z1": H, "L": 0.075, "W": 0.025, "yaw_rad": 0.0}],
            # Printed false seam across continuous 75x25 mm block
            "seams": [{"p0": (0.0, -0.012), "p1": (0.0, 0.012), "z": H, "half_w_m": 0.0009}],
        },
        {
            "name": "intact_2_false_stripes_75x25",
            "expected_count": 1, "expected_yaw": 0.0, "must_resolve": True, "must_abstain": False, "is_negative": True,
            # Continuous solid 75x75? No, solid block with TWO plausible false lines at 25 mm pitch!
            "blocks": [{"cx": 0.0, "cy": 0.0, "z0": 0.0, "z1": H, "L": 0.075, "W": 0.075, "yaw_rad": 0.0}],
            "seams": [
                {"p0": (-0.037, -0.0125), "p1": (0.037, -0.0125), "z": H, "half_w_m": 0.0009},
                {"p0": (-0.037, 0.0125), "p1": (0.037, 0.0125), "z": H, "half_w_m": 0.0009},
            ],
            "expected_count_note": "oversized_solid_with_2_stripes_must_not_split",
        },
        {
            "name": "square_75x75_layout_A_0deg_ends_visible",
            "expected_count": 3, "expected_yaw": 0.0, "must_resolve": True, "must_abstain": False,
            "blocks": [
                {"cx": 0.0, "cy": -0.025, "z0": 0.0, "z1": H, "L": 0.075, "W": 0.025, "yaw_rad": 0.0},
                {"cx": 0.0, "cy": 0.0, "z0": 0.0, "z1": H, "L": 0.075, "W": 0.025, "yaw_rad": 0.0},
                {"cx": 0.0, "cy": 0.025, "z0": 0.0, "z1": H, "L": 0.075, "W": 0.025, "yaw_rad": 0.0},
            ],
            "seams": [
                {"p0": (-0.037, -0.0125), "p1": (0.037, -0.0125), "z": H, "half_w_m": 0.0009},
                {"p0": (-0.037, 0.0125), "p1": (0.037, 0.0125), "z": H, "half_w_m": 0.0009},
            ],
            "end_breaks": [
                {"x": -0.0375, "y": -0.0125, "z0": 0.0, "z1": H}, {"x": -0.0375, "y": 0.0125, "z0": 0.0, "z1": H},
            ],
        },
        {
            "name": "square_75x75_layout_B_90deg_ends_visible",
            "expected_count": 3, "expected_yaw": 90.0, "must_resolve": True, "must_abstain": False,
            "blocks": [
                {"cx": -0.025, "cy": 0.0, "z0": 0.0, "z1": H, "L": 0.075, "W": 0.025, "yaw_rad": np.pi / 2},
                {"cx": 0.0, "cy": 0.0, "z0": 0.0, "z1": H, "L": 0.075, "W": 0.025, "yaw_rad": np.pi / 2},
                {"cx": 0.025, "cy": 0.0, "z0": 0.0, "z1": H, "L": 0.075, "W": 0.025, "yaw_rad": np.pi / 2},
            ],
            "seams": [
                {"p0": (-0.0125, -0.037), "p1": (-0.0125, 0.037), "z": H, "half_w_m": 0.0009},
                {"p0": (0.0125, -0.037), "p1": (0.0125, 0.037), "z": H, "half_w_m": 0.0009},
            ],
            "end_breaks": [
                {"x": -0.0125, "y": -0.0375, "z0": 0.0, "z1": H}, {"x": 0.0125, "y": -0.0375, "z0": 0.0, "z1": H},
            ],
        },
        {
            "name": "square_75x75_flush_ends_obscured",
            "expected_count": 3, "expected_yaw": 0.0, "must_resolve": False, "must_abstain": True, "occlude_side": True,
            "blocks": [
                {"cx": 0.0, "cy": -0.025, "z0": 0.0, "z1": H, "L": 0.075, "W": 0.025, "yaw_rad": 0.0},
                {"cx": 0.0, "cy": 0.0, "z0": 0.0, "z1": H, "L": 0.075, "W": 0.025, "yaw_rad": 0.0},
                {"cx": 0.0, "cy": 0.025, "z0": 0.0, "z1": H, "L": 0.075, "W": 0.025, "yaw_rad": 0.0},
            ],
            "seams": [
                {"p0": (-0.037, -0.0125), "p1": (0.037, -0.0125), "z": H, "half_w_m": 0.0009},
                {"p0": (-0.037, 0.0125), "p1": (0.037, 0.0125), "z": H, "half_w_m": 0.0009},
            ],
        },
        {
            "name": "flush_pair_zero_side_contrast",
            "expected_count": 2, "expected_yaw": 0.0, "must_resolve": False, "must_abstain": False,
            "blocks": [
                {"cx": 0.0, "cy": -0.0125, "z0": 0.0, "z1": H, "L": 0.075, "W": 0.025, "yaw_rad": 0.0},
                {"cx": 0.0, "cy": 0.0125, "z0": 0.0, "z1": H, "L": 0.075, "W": 0.025, "yaw_rad": 0.0},
            ],
            "seams": [{"p0": (-0.037, 0.0), "p1": (0.037, 0.0), "z": H, "half_w_m": 0.0009}],
            "end_breaks": [],
        },
        {
            "name": "real_step_3mm_no_side_needed",
            "expected_count": 2, "expected_yaw": 0.0, "must_resolve": True, "must_abstain": False, "occlude_side": True,
            "blocks": [
                {"cx": 0.0, "cy": -0.0125, "z0": 0.0, "z1": H, "L": 0.075, "W": 0.025, "yaw_rad": 0.0},
                {"cx": 0.0, "cy": 0.0125, "z0": 0.0, "z1": H + 0.0035, "L": 0.075, "W": 0.025, "yaw_rad": 0.0},
            ],
        },
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(IMPL_ROOT / "results/synthetic_printed_jenga"))
    args = ap.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    ov_dir = out_dir / "overlays"; ov_dir.mkdir(exist_ok=True)

    cfg = preset_config("printed_jenga")
    cfg.calib_epoch = "synthetic_exact"
    cfg.require_fk_for_moving_wrist = False
    cache = CameraModelCache()
    cam_w, Tw, cam_s, Ts = make_synthetic_cameras(cache)
    dwell_ok = DwellStatus(True, True, True, True, 0.0, 0.1, 0.2, 80.0, "ok")

    results: dict[str, list[dict]] = {p: [] for p in ALL_POLICIES}
    for sc_idx, sc in enumerate(build_printed_scenarios()):
        rgb_w, d_w, rgb_s, d_s = render_synthetic_scene(
            cam_w, Tw, cam_s, Ts, sc["blocks"], sc.get("seams"), sc.get("end_breaks"), sc.get("occlude_side", False),
        )
        vi_w = ViewInput("wrist", cam_w, rgb_w, d_w, Tw, (sc_idx + 1) * 1_000_000_000, (sc_idx + 1) * 1_000_000_000)
        vi_s = ViewInput("tripod", cam_s, rgb_s, d_s, Ts, (sc_idx + 1) * 1_000_000_000, (sc_idx + 1) * 1_000_000_000)
        for pol in ALL_POLICIES:
            res = run_perception_epoch(wrist=vi_w, side=vi_s, config=cfg, policy=pol, dwell=dwell_ok, epoch_id=sc_idx + 1)
            # Evaluate safety/coverage against ground-truth blocks
            n_out = len(res.poses)
            n_amb = len(res.ambiguous_clusters)
            wrong_split_neg = bool(sc.get("is_negative") and (n_out > 1 or (sc["name"].startswith("intact_2") and n_out > 0)))
            wrong_yaw = False
            if n_out == len(sc["blocks"]) and not sc.get("is_negative"):
                max_ye = max(sym_yaw_err_deg(p.yaw_deg_mod, sc["expected_yaw"], 180.0) for p in res.poses)
                wrong_yaw = max_ye > 10.0
            ok_resolved = (n_out == len(sc["blocks"]) and not wrong_yaw) if not sc.get("is_negative") else (n_out <= 1 and not wrong_split_neg)
            results[pol].append({
                "scenario": sc["name"],
                "n_out": n_out,
                "n_amb": n_amb,
                "ok_resolved": ok_resolved,
                "false_confident": wrong_split_neg or wrong_yaw,
                "ms": res.stage_timings_ms["total_ms"],
                "abstained": n_amb > 0,
            })
            if pol in ("merged_corroborated", "merged_wrist_rgb") and sc_idx in (1, 3, 4, 5):
                ov = render_overlay(rgb_w, rgb_s, res, [], cfg.block)
                cv2.imwrite(str(ov_dir / f"{sc['name']}__{pol}.png"), ov)

    summary_lines = [
        "# Synthetic Printed-Jenga (75x25x15 mm) Physical Benchmark",
        "",
        "| Policy | Correct Outcomes (/9) | False-Confident Negatives/Headings | Abstentions | Median / P95 ms |",
        "|---|---:|---:|---:|---:|",
    ]
    for pol, rows in results.items():
        n_ok = sum(r["ok_resolved"] for r in rows)
        n_fc = sum(r["false_confident"] for r in rows)
        n_ab = sum(r["abstained"] for r in rows)
        ms_vals = [r["ms"] for r in rows]
        med = round(float(np.median(ms_vals)), 2)
        p95 = round(float(np.percentile(ms_vals, 95)), 2)
        summary_lines.append(f"| `{pol}` | {n_ok}/{len(rows)} | {n_fc}/{len(rows)} | {n_ab} | {med}/{p95} |")

    (out_dir / "printed_jenga_report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    md = "\n".join(summary_lines) + "\n"
    (out_dir / "printed_jenga_table.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()

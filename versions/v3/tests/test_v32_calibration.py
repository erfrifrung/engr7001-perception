"""V3.2 calibration checks (fault #10, seen live on 26 Nov: every camera had moved 99-274 mm / 3-38 deg while the
static TF publisher still served the 19 Nov calibration, and WP3 kept fusing the views)."""
from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wp3_live.calib_check import (  # noqa: E402
    CalibrationMonitor, TablePlaneCheck, compare_board_fits, transform_delta,
)
from wp3_live.geometry_cache import CameraModel, StaticTFCache  # noqa: E402

W, Hpx = 640, 480
K = np.array([[600.0, 0, 320.0], [0, 600.0, 240.0], [0, 0, 1.0]])
CAM = CameraModel("tripod", W, Hpx, K, np.zeros(5), "tripod_color_optical_frame", 0.001)
BLOCK_H = 0.015


def look_at(eye, target=(0.0, 0.0, 0.0)) -> np.ndarray:
    """T_table_cam (camera -> table) for an optical frame at `eye` looking at `target` (x right, y down, z fwd)."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, z], axis=1)
    T[:3, 3] = eye
    return T


def render_depth(T_true: np.ndarray, blocks=((0.0, 0.0),)) -> np.ndarray:
    """uint16 mm depth of the table plane z=0 (plus 75x25x15 mm blocks) seen through the TRUE camera pose."""
    rays = CAM.rays_f32.reshape(-1, 3).astype(np.float64)
    R, t = T_true[:3, :3], T_true[:3, 3]
    dirs = rays @ R.T                                       # table-frame ray directions (unit depth along z_cam)
    depth = np.full(len(rays), np.inf)
    s = -t[2] / dirs[:, 2]                                  # table plane
    ok = (s > 0) & (np.abs(t[0] + s * dirs[:, 0]) < 0.40) & (np.abs(t[1] + s * dirs[:, 1]) < 0.40)
    depth[ok] = s[ok]
    for bx, by in blocks:                                   # block tops (enough for a plane test)
        s2 = (BLOCK_H - t[2]) / dirs[:, 2]
        px, py = t[0] + s2 * dirs[:, 0], t[1] + s2 * dirs[:, 1]
        on = (s2 > 0) & (np.abs(px - bx) < 0.0375) & (np.abs(py - by) < 0.0125) & (s2 < depth)
        depth[on] = s2[on]
    depth[~np.isfinite(depth)] = 0.0
    return np.round(depth * 1000.0).astype(np.uint16).reshape(Hpx, W)


T_TRUE = look_at((0.05, -0.40, 0.55))
DEPTH = render_depth(T_TRUE)


def _perturb(T, rot_axis=(1, 0, 0), deg=0.0, dxyz=(0, 0, 0), about_table_origin=False):
    P = np.eye(4)
    P[:3, :3] = Rotation.from_rotvec(np.radians(deg) * np.asarray(rot_axis, float)).as_matrix()
    P[:3, 3] = dxyz
    return P @ T if about_table_origin else T @ P


def _check(T_assumed):
    return TablePlaneCheck().check(DEPTH, CAM.rays_f32, 0.001, T_assumed)


# ---------------------------------------------------------------------------------------------- plane check
def test_correct_calibration_passes():
    r = _check(T_TRUE)
    assert r.status == "ok" and r.angle_deg < 0.3 and r.offset_mm < 2.0


def test_tilted_calibration_is_rejected():
    r = _check(_perturb(T_TRUE, (1, 0, 0), 8.0))
    assert r.status == "mismatch" and r.angle_deg > 7.0


def test_small_camera_rotation_is_rejected_by_its_workspace_offset():
    """4 deg about the camera moves the table ~47 mm at the workspace 0.68 m away: a real error, so rejected
    even though the angle alone is under the 6 deg limit."""
    r = _check(_perturb(T_TRUE, (1, 0, 0), 4.0))
    assert r.status == "mismatch" and r.angle_deg < 6.0 and r.offset_mm > 15.0


def test_12nov_bag_tripod_depth_tilt_is_not_rejected():
    """Real data: on the 12 Nov bag the tripod's depth plane is 4.6-4.9 deg off its board calibration while the table
    is at +1 mm inside the workspace (oblique depth bias). V3.2's first threshold (2.5 deg, offset measured at the
    camera: 32-36 mm) rejected it and removed the dual view on the real rig (12/47 -> 2/54 dual-view epochs)."""
    T_assumed = _perturb(T_TRUE, (1, 0, 0), 4.7, about_table_origin=True)   # tilted about a workspace axis
    r = _check(T_assumed)
    assert r.status == "suspect" and r.offset_mm < 3.0


def test_height_error_is_rejected():
    r = _check(_perturb(T_TRUE, dxyz=(0, 0, 0.030), about_table_origin=True))
    assert r.status == "mismatch" and r.offset_mm > 25.0


def test_26nov_scale_move_is_rejected():
    """Tripod on 26 Nov: 274 mm / 38 deg away from the published calibration."""
    T_moved = look_at((0.19, -0.51, 0.57), target=(0.10, 0.12, 0.0))
    r = TablePlaneCheck().check(render_depth(T_moved), CAM.rays_f32, 0.001, T_TRUE)
    assert r.status == "mismatch" and r.angle_deg > 5.0


def test_slide_along_the_table_is_a_documented_blind_spot():
    T_slid = _perturb(T_TRUE, dxyz=(0.030, 0.0, 0.0), about_table_origin=True)
    assert TablePlaneCheck().check(render_depth(T_slid), CAM.rays_f32, 0.001, T_TRUE).status == "ok"


def test_pan_about_the_table_normal_is_a_documented_blind_spot():
    r = _check(_perturb(T_TRUE, (0, 0, 1), 20.0, about_table_origin=True))
    assert r.status == "ok"          # the plane is unchanged: the session-start board check covers this case


def test_no_table_in_view_is_unverified_not_ok():
    r = TablePlaneCheck().check(np.zeros((Hpx, W), np.uint16), CAM.rays_f32, 0.001, T_TRUE)
    assert r.status == "unverified"


# ----------------------------------------------------------------------------------------- monitor hysteresis
def test_monitor_trips_after_two_bad_checks_and_clears_after_three_good():
    m = CalibrationMonitor(period_s=1.0)
    bad = _perturb(T_TRUE, (1, 0, 0), 8.0)
    assert m.observe("tripod", 0.0, DEPTH, CAM.rays_f32, 0.001, bad) == "unverified"
    assert m.observe("tripod", 0.5, DEPTH, CAM.rays_f32, 0.001, bad) == "unverified"   # rate-limited
    assert m.observe("tripod", 1.0, DEPTH, CAM.rays_f32, 0.001, bad) == "mismatch"
    for k, expect in ((2.0, "mismatch"), (3.0, "mismatch"), (4.0, "ok")):
        assert m.observe("tripod", k, DEPTH, CAM.rays_f32, 0.001, T_TRUE) == expect
    assert m.diagnostics("tripod")["last_check"]["status"] == "ok"


def test_unverified_checks_never_clear_a_mismatch():
    m = CalibrationMonitor(period_s=0.0)
    bad = _perturb(T_TRUE, (1, 0, 0), 8.0)
    m.observe("w", 0.0, DEPTH, CAM.rays_f32, 0.001, bad)
    m.observe("w", 1.0, DEPTH, CAM.rays_f32, 0.001, bad)
    for k in range(5):
        assert m.observe("w", 2.0 + k, np.zeros_like(DEPTH), CAM.rays_f32, 0.001, T_TRUE) == "mismatch"


# ------------------------------------------------------------------------------------------- /tf_static refresh
def test_static_tf_cache_expires_so_a_new_tf_static_is_read():
    c = StaticTFCache()
    T0 = np.eye(4)
    c.put("e", "jenga_table", "tripod", T0, now_s=10.0)
    assert c.get("e", "jenga_table", "tripod", now_s=10.5, max_age_s=1.0) is not None
    assert c.get("e", "jenga_table", "tripod", now_s=11.5, max_age_s=1.0) is None      # caller re-reads TF
    assert np.allclose(c.previous("jenga_table", "tripod"), T0)                          # kept for change detection
    assert c.get("e", "jenga_table", "tripod") is not None                               # V3.1 call: no expiry


def test_transform_delta():
    T = _perturb(np.eye(4), (0, 0, 1), 3.0, dxyz=(0.003, 0.004, 0.0))
    dt, dr = transform_delta(np.eye(4), T)
    assert abs(dt - 5.0) < 1e-6 and abs(dr - 3.0) < 1e-6


# -------------------------------------------------------------------------------------- session-start board check
def _fit(links):
    return {"parent": "jenga_table", "passed_stationary_smoke_gate": True,
            "transforms": {k: np.asarray(v).tolist() for k, v in links.items()}}


REF = {"wrist_link": look_at((0.0, -0.05, 0.50)), "tripod_link": T_TRUE, "third_link": look_at((-0.46, 0.20, 0.62))}


def test_board_check_passes_on_the_same_calibration():
    assert compare_board_fits(_fit(REF), _fit(REF))["passed"]


def test_board_check_names_the_camera_that_moved():
    cur = dict(REF, tripod_link=_perturb(T_TRUE, (0, 0, 1), 2.0, about_table_origin=True))   # a pan: plane-blind
    res = compare_board_fits(_fit(REF), _fit(cur))
    assert not res["passed"] and res["failed"] == ["tripod_link"]
    assert res["pairwise"]["third_link->tripod_link"]["dr_deg"] > 1.5


def test_board_check_separates_a_board_move_from_a_rig_move():
    move = _perturb(np.eye(4), (0, 0, 1), 4.0, dxyz=(0.02, 0.0, 0.0))
    cur = {k: move @ v for k, v in REF.items()}            # board moved: every camera "moved" by the same amount
    res = compare_board_fits(_fit(REF), _fit(cur))
    assert not res["passed"]
    assert all(d["dt_mm"] < 0.01 and d["dr_deg"] < 0.01 for d in res["pairwise"].values())


def test_board_check_script_exit_codes(tmp_path):
    a, b = tmp_path / "pub.json", tmp_path / "now.json"
    a.write_text(json.dumps(_fit(REF)))
    b.write_text(json.dumps(_fit(REF)))
    script = ROOT / "scripts" / "check_calibration_drift.py"
    assert subprocess.run([sys.executable, str(script), str(a), str(b)], capture_output=True).returncode == 0
    b.write_text(json.dumps(_fit(dict(REF, wrist_link=_perturb(REF["wrist_link"], dxyz=(0.01, 0, 0))))))
    r = subprocess.run([sys.executable, str(script), str(a), str(b)], capture_output=True, text=True)
    assert r.returncode == 1 and "DRIFT" in r.stdout
    assert subprocess.run([sys.executable, str(script), str(a), str(tmp_path / "x.json")],
                          capture_output=True).returncode == 2

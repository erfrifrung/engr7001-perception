"""Render the portfolio figures and point clouds from the public RGB-D fixture."""

from pathlib import Path
import sys

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "versions/v2-three-view-handoff"))
from scripts.run_printed_jenga_suite import (  # noqa: E402
    build_printed_scenarios,
    make_synthetic_cameras,
    render_synthetic_scene,
)
from wp3_live import ViewInput, preset_config, run_perception_epoch  # noqa: E402
from wp3_live.dwell_gate import DwellStatus  # noqa: E402
from wp3_live.geometry_cache import CameraModelCache, deproject_roi_points  # noqa: E402
from wp3_live.overlay import render_overlay  # noqa: E402

MEDIA = ROOT / "media"
WRIST = "#2654ac"
FIXED = "#af510f"
INK = "#172538"
MUTED = "#46596c"
PAPER = "#f7f9fc"


def write_cloud(role, cloud):
    xyz = cloud["pts_tbl_f32"]
    rgb = cloud["rgb_u8"]
    vertices = np.empty(len(xyz), dtype=[
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    vertices["x"], vertices["y"], vertices["z"] = xyz.T
    vertices["red"], vertices["green"], vertices["blue"] = rgb.T
    header = "\n".join([
        "ply", "format binary_little_endian 1.0",
        f"comment synthetic flush_pair_corroborated camera={role}",
        "comment frame=jenga_table units=metres",
        f"element vertex {len(vertices)}",
        "property float x", "property float y", "property float z",
        "property uchar red", "property uchar green", "property uchar blue",
        "end_header", "",
    ])
    (MEDIA / f"wp3-synthetic-{role}.ply").write_bytes(header.encode("ascii") + vertices.tobytes())
    print(f"{role}: {len(vertices)} points")


def crop(image, bounds):
    x0, y0, x1, y1 = bounds
    return image[y0:y1, x0:x1]


def figure_header(fig, title, subtitle):
    fig.suptitle(title, x=.07, y=.95, ha="left", color=INK, fontsize=24, fontweight="bold")
    fig.text(.07, .855, subtitle, color=MUTED, fontsize=11)


def plot_clouds(wrist_rgb, wrist_cloud, fixed_rgb, fixed_cloud):
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.1), facecolor=PAPER)
    figure_header(fig, "Two cameras, one table frame",
                  "Rendered RGB-D fixture  |  Two flush 75 × 25 × 15 mm blocks")
    fig.subplots_adjust(left=.08, right=.96, bottom=.13, top=.79, wspace=.32, hspace=.54)
    top, section = axes[1]
    for image, rgb, cloud, label, color in (
        (axes[0, 0], wrist_rgb, wrist_cloud, "Wrist", WRIST),
        (axes[0, 1], fixed_rgb, fixed_cloud, "Fixed", FIXED),
    ):
        image.imshow(crop(rgb, cloud["bbox_2d"]), interpolation="nearest")
        image.set_title(f"{label} RGB", color=color, fontweight="bold")
        image.set_axis_off()
        xyz = cloud["pts_tbl_f32"] * 1000.0
        above = xyz[xyz[:, 2] > 2.5]
        slice_points = xyz[np.abs(xyz[:, 0]) < 1.8]
        top.scatter(above[:, 0], above[:, 1], c=color, s=3, linewidths=0, alpha=.45)
        section.scatter(slice_points[:, 1], slice_points[:, 2], c=color, s=14,
                        linewidths=0, alpha=.65, label=label)
    top.set(title="Top view of points above the table", xlabel="Table x (mm)", ylabel="Table y (mm)",
            xlim=(-48, 48), ylim=(-33, 33), aspect="equal")
    section.set(title="Cross-section near x = 0 mm", xlabel="Table y (mm)", ylabel="Height z (mm)",
                xlim=(-48, 48), ylim=(-3, 22))
    section.legend(loc="upper center", ncol=2, frameon=False, fontsize=9)
    for axis in axes[1]:
        axis.set_axisbelow(True)
        axis.grid(color="#e2e9f1")
        axis.spines[["top", "right"]].set_visible(False)
        axis.title.set_fontweight("bold")
    for z in (0, 15):
        section.axhline(z, linestyle=":", linewidth=.8, color=MUTED, zorder=0)
    fig.text(.07, .048, "Both top faces sit at 15 mm. The RGB seam and visible end breaks "
             "provide evidence for the split.", color=MUTED, fontsize=10)
    fig.savefig(MEDIA / "wp3-synthetic-point-clouds.png", dpi=145)
    plt.close(fig)


def plot_occlusion(rgb, bounds, config, two, three):
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.7), facecolor=PAPER)
    figure_header(fig, "A clear view can resolve an occluded seam",
                  "Synthetic flush pair  |  Same wrist image and detector policy in each run")
    fig.subplots_adjust(left=.055, right=.945, bottom=.19, top=.70, wspace=.17)
    for axis, title, result, color in (
        (axes[0], "Wrist + occluded fixed view", two, FIXED),
        (axes[1], "Add a qualified clear view", three, WRIST),
    ):
        overlay = render_overlay(rgb, None, result, [], config.block)
        axis.imshow(crop(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), bounds))
        axis.set_title(title, loc="left", color=color, fontweight="bold", fontsize=13)
        axis.set_axis_off()
        poses, clusters = len(result.poses), len(result.ambiguous_clusters)
        axis.text(0, -.09, f"{poses} poses / {clusters} ambiguous "
                  + ("cluster" if clusters == 1 else "clusters"),
                  transform=axis.transAxes, color=INK, fontsize=11, fontweight="bold", va="top")
        print(f"{title}: {poses} poses, {clusters} ambiguous")
    fig.text(.07, .05, "The clear input uses the same synthetic fixed-camera pose with the occluder removed.",
             color=MUTED, fontsize=9)
    fig.savefig(MEDIA / "wp3-synthetic-occlusion.png", dpi=145)
    plt.close(fig)


def main():
    fixture = next(item for item in build_printed_scenarios()
                   if item["name"] == "flush_pair_corroborated")
    wrist, table_wrist, fixed, table_fixed = make_synthetic_cameras(CameraModelCache())
    fixture_args = (wrist, table_wrist, fixed, table_fixed,
                    fixture["blocks"], fixture["seams"], fixture["end_breaks"])
    wrist_rgb, wrist_depth, fixed_rgb, fixed_depth = render_synthetic_scene(*fixture_args)
    _, _, blocked_rgb, blocked_depth = render_synthetic_scene(*fixture_args, occlude_side=True)
    bounds_m = ((-.060, -.045, -.004), (.060, .045, .020))
    wrist_cloud = deproject_roi_points(wrist_depth, wrist_rgb, wrist, table_wrist, *bounds_m, depth_stride=3)
    fixed_cloud = deproject_roi_points(fixed_depth, fixed_rgb, fixed, table_fixed, *bounds_m, depth_stride=3)

    config = preset_config("printed_jenga")
    config.calib_epoch = "synthetic_portfolio"
    config.require_fk_for_moving_wrist = False
    dwell = DwellStatus(True, True, True, True, 0.0, 0.1, 0.2, 85.0, "ok")
    stamp = 1_000_000_000
    inputs = dict(
        wrist=ViewInput("wrist", wrist, wrist_rgb, wrist_depth, table_wrist, stamp, stamp),
        side=ViewInput("tripod", fixed, blocked_rgb, blocked_depth, table_fixed, stamp, stamp),
        config=config, dwell=dwell, policy="merged_corroborated",
    )
    two = run_perception_epoch(**inputs)
    three = run_perception_epoch(
        **inputs, third=ViewInput("third", fixed, fixed_rgb, fixed_depth, table_fixed, stamp, stamp),
    )
    write_cloud("wrist", wrist_cloud)
    write_cloud("fixed", fixed_cloud)
    plot_clouds(wrist_rgb, wrist_cloud, fixed_rgb, fixed_cloud)
    plot_occlusion(wrist_rgb, wrist_cloud["bbox_2d"], config, two, three)


if __name__ == "__main__":
    main()

"""Configuration dataclasses, verified presets, and deterministic hashing."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BlockSpec:
    """Physical cuboid dimensions and tolerance parameters (metres)."""

    name: str = "printed_jenga"
    length_m: float = 0.0750
    width_m: float = 0.0250
    height_m: float = 0.0150
    edge_loss_m: float = 0.0030
    size_tol: float = 0.22
    min_fill: float = 0.58
    square_symmetry: bool = False
    layers: int = 18

    @property
    def major_m(self) -> float:
        return max(self.length_m, self.width_m)

    @property
    def minor_m(self) -> float:
        return min(self.length_m, self.width_m)

    @property
    def is_square(self) -> bool:
        return self.square_symmetry or abs(self.major_m - self.minor_m) <= 0.004

    @property
    def yaw_symmetry_deg(self) -> float:
        return 90.0 if self.is_square else 180.0


@dataclass(frozen=True)
class CameraProfile:
    """Camera stream & mount role configuration."""

    role: str
    namespace: str
    width: int = 640
    height: int = 480
    fps: int = 15
    depth_scale_m: float = 0.001
    min_depth_m: float = 0.18
    max_depth_m: float = 1.60


@dataclass(frozen=True)
class PolicyThresholds:
    """Tuneable thresholds shared cleanly across interchangeable policies."""

    cell_m: float = 0.0020
    band_lo_factor: float = 0.65
    band_hi_factor: float = 1.45
    min_area_frac: float = 0.35
    split_min_frac: float = 0.40
    # Validated depth cues (measured on flanks away from seam line)
    depth_step_min_m: float = 0.0025
    face_step_min_m: float = 0.0020
    depth_trough_min_m: float = 0.0018
    flank_inner_m: float = 0.0030
    flank_outer_m: float = 0.0120
    flank_min_pts: int = 6
    # Native-RGB seam cues
    rgb_trough_min: float = 5.5
    rgb_grad_min: float = 9.0
    rgb_color_step_min: float = 24.0
    rgb_continuity_min: float = 0.65
    rgb_corroborated_continuity_min: float = 0.70
    shoulder_support_min: float = 0.75
    stripe_max_peaks: int = 2
    blackhat_min: float = 22.0
    # Cross-view / verifier gates
    side_vis_min_r: float = 0.60
    side_rgb_grad_min: float = 10.0
    side_depth_step_min_m: float = 0.0018
    empty_slot_penetration_m: float = 0.020
    empty_slot_vis_min_r: float = 0.80
    occupied_tol_m: float = 0.006
    displaced_protrusion_m: float = 0.005
    verifier_residual_m: float = 0.012
    verifier_sample_frac: float = 0.20
    # Association & tracking
    assoc_gate_xy_m: float = 0.010
    assoc_gate_yaw_deg: float = 30.0
    assoc_uniqueness_margin: float = 2.0
    occlusion_hold_s: float = 1.8
    stale_clear_s: float = 0.5
    retire_s: float = 3.0
    max_age_s: float = 0.35
    max_pair_skew_s: float = 0.12
    max_cross_view_skew_s: float = 0.08
    freeze_timeout_s: float = 1.2
    mds_confirm_frames: int = 1
    # v2 upgrades: conservative contradiction, ray slot clearing, valid depth interp, whole-partition MV
    allow_appearance_contradiction: bool = False
    mv_partition_margin: float = 0.12
    recessed_slot_min_m: float = 0.008
    slot_clear_tol_m: float = 0.006
    max_depth_interp_jump_m: float = 0.04
    # V3: rank minAreaRect edge directions ahead of PCA axes in partition-hypothesis generation (fixes V2's 3 mm
    # end-stagger dead zone on synthetic scenes). False = exact V2 order. On the 12 Nov wooden-prop bag the two
    # orders differ by up to 8.2 deg yaw on one epoch (square props, no ground truth): settle on printed-block GT.
    edge_first_directions: bool = True


@dataclass
class RuntimeConfig:
    """Full pipeline configuration for live and replay execution."""

    profile_name: str = "printed_jenga"
    frame_id: str = "jenga_table"
    roi_xyz_min: tuple[float, float, float] = (-0.16, -0.16, -0.02)
    roi_xyz_max: tuple[float, float, float] = (0.16, 0.16, 0.30)
    row_origin_xyz_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    row_yaw_deg: float = 0.0
    use_lane_prior: bool = False
    block: BlockSpec = field(default_factory=BlockSpec)
    thresholds: PolicyThresholds = field(default_factory=PolicyThresholds)
    wrist: CameraProfile = field(
        default_factory=lambda: CameraProfile(role="wrist", namespace="/perception/wrist", width=1280, height=720, fps=15)
    )
    side: CameraProfile = field(
        default_factory=lambda: CameraProfile(role="tripod", namespace="/perception/tripod", width=640, height=480, fps=15)
    )
    third: CameraProfile = field(
        default_factory=lambda: CameraProfile(role="third", namespace="/perception/third", width=1280, height=720, fps=15)
    )
    use_roi_precrop: bool = True
    roi_pad_px: int = 14
    use_side_verifier: bool = True
    require_fk_for_moving_wrist: bool = True
    require_unique_square_evidence: bool = True
    enable_robot_self_filter: bool = True
    tcp_offset_z_m: float = 0.242
    plane_tolerance_m: float = 0.004
    min_plane_support: float = 0.55
    calib_epoch: str = "unverified"
    extra_roi_crop_frac: tuple[float, float, float, float] | None = None
    enable_temporal_skip: bool = False
    temporal_skip_max_frames: int = 8
    temporal_skip_depth_tol_m: float = 0.0035
    enable_partition_warm_start: bool = True
    enable_parallel_views: bool = False
    # External-mask grouping ablation switch: "greedy" (legacy first-come claiming) or "search"
    # (order-independent single-cuboid group search). Suppression applies to "greedy" only.
    mask_grouping: str = "search"
    mask_suppress_merged: bool = True
    # Top-layer band statistic over cells above layer 1 (False = legacy all-ROI-cells percentile).
    top_band_from_upper_cells: bool = True
    # Carve one fully visible block from an oversized component whose remainder is truncated by an
    # adjacent occluder (taller/no-depth cells). Remainder is published as an ambiguous cluster.
    enable_occlusion_truncated_partition: bool = True
    # V3 tracker (results/V3_RESEARCH.md D4): identities of blocks hidden together stay unverified; DISPLACED
    # tracks re-anchor after N consistent fresh observations. Passed to WP3TemporalTracker by the live node and replay.
    persistent_joint_occlusion: bool = True
    reanchor_after: int = 2
    # V3 region stage (results/V3_RESEARCH.md D1-D3). Off -> exact tuned-V2 behaviour.
    # V3.1: OFF by default. Live M1 runs the V2 path; enable explicitly (JSON key or ROS parameter) to evaluate V3.
    enable_v3_regions: bool = False
    v3_solver: str = "prove"               # "prove" | "enumerate" (STABLE is not a proof) | "adaptive" (sound)
    v3_view_carving: bool = True           # signed-depth see-through carving of UNKNOWN cells (wrist/side/third)
    v3_publish_unexplained: bool = True    # unexplained occupancy becomes an ambiguity cluster instead of vanishing
    v3_robot_unknown: bool = True          # projected robot capsules are UNKNOWN space
    v3_region_budget_ms: float = 250.0     # wall-clock safety net per region
    v3_frame_budget_ms: float = 120.0      # no new region is started after this (regions already done are kept)
    v3_max_nodes: int = 6000               # deterministic per-search node cap (timeout => abstain)
    v3_region_cache: bool = True           # reuse structural verdicts while a region's occupancy is unchanged
    v3_slab_guard: bool = True
    v3_mask_seam_cue: bool = True          # async masks may confirm a seam of geometrically proven blocks only
    v3_fast_deproject: bool = True         # flattened cached ray tables (bit-identical output to V2)             # rectangular block unions need a seam cue (else could be one slab)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def preset_config(name: str) -> RuntimeConfig:
    """Return a clean preset matching lab or historical configurations."""
    key = name.strip().lower()
    if key in ("historical_props", "historical_props_12nov", "props"):
        return RuntimeConfig(
            profile_name="historical_props_12nov",
            frame_id="jenga_table",
            roi_xyz_min=(-0.105, -0.180, -0.025),
            roi_xyz_max=(0.095, 0.065, 0.120),
            row_origin_xyz_m=(-0.008, -0.103, 0.0),
            row_yaw_deg=90.0,
            use_lane_prior=False,
            block=BlockSpec(
                name="historical_props",
                length_m=0.045,
                width_m=0.045,
                height_m=0.038,
                edge_loss_m=0.005,
                size_tol=0.24,
                min_fill=0.55,
                square_symmetry=True,
                layers=3,
            ),
            thresholds=PolicyThresholds(
                cell_m=0.0020,
                band_lo_factor=0.68,
                band_hi_factor=1.40,
                rgb_trough_min=4.5,
                rgb_grad_min=8.0,
                rgb_continuity_min=0.65,
                mds_confirm_frames=1,
            ),
            wrist=CameraProfile(role="wrist", namespace="/perception/wrist", width=640, height=480, fps=6),
            side=CameraProfile(role="tripod", namespace="/perception/tripod", width=640, height=480, fps=6),
            use_roi_precrop=True,
            use_side_verifier=True,
            require_fk_for_moving_wrist=False,
            require_unique_square_evidence=False,
        )
    if key in ("wrist_imu_17nov", "imu_17nov"):
        cfg = preset_config("printed_jenga")
        cfg.profile_name = "wrist_imu_17nov"
        cfg.roi_xyz_min = (-0.16, -0.22, -0.025)
        cfg.roi_xyz_max = (0.16, 0.14, 0.12)
        cfg.extra_roi_crop_frac = (0.24, 0.08, 0.89, 0.77)
        cfg.block = BlockSpec(
            name="wrist_imu_17nov",
            length_m=0.047,
            width_m=0.047,
            height_m=0.038,
            edge_loss_m=0.005,
            size_tol=0.25,
            min_fill=0.55,
            square_symmetry=True,
            layers=2,
        )
        cfg.require_unique_square_evidence = False
        return cfg
    return RuntimeConfig()


def _merge_dataclass(instance: Any, patch: dict[str, Any]) -> Any:
    kwargs = asdict(instance)
    for k, v in patch.items():
        if k in kwargs:
            if isinstance(kwargs[k], tuple) and isinstance(v, list):
                kwargs[k] = tuple(v)
            else:
                kwargs[k] = v
    return type(instance)(**kwargs)


# V3 / V3.1 switches a flat lab JSON may set (previously ignored by the loader, so JSON could not turn V3 off).
_V3_JSON_KEYS = (
    "mask_grouping", "mask_suppress_merged", "top_band_from_upper_cells", "enable_occlusion_truncated_partition",
    "persistent_joint_occlusion", "reanchor_after",
    "enable_v3_regions", "v3_solver", "v3_view_carving", "v3_publish_unexplained", "v3_robot_unknown",
    "v3_region_budget_ms", "v3_frame_budget_ms", "v3_max_nodes", "v3_region_cache", "v3_slab_guard",
    "v3_mask_seam_cue", "v3_fast_deproject",
)


def load_runtime_config(path: str | Path | None, overrides: dict[str, Any] | None = None) -> RuntimeConfig:
    """Load RuntimeConfig from JSON file (supports flat lab JSON or structured RuntimeConfig)."""
    if path is None or str(path).strip() == "":
        cfg = preset_config("printed_jenga")
    else:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        base = raw.get("preset", raw.get("profile_name", "printed_jenga"))
        cfg = preset_config(base)
        for key in (
            "profile_name", "frame_id", "use_lane_prior", "row_yaw_deg", "use_roi_precrop",
            "roi_pad_px", "use_side_verifier", "require_fk_for_moving_wrist",
            "require_unique_square_evidence", "enable_robot_self_filter", "tcp_offset_z_m",
            "plane_tolerance_m", "min_plane_support", "calib_epoch",
            "enable_temporal_skip", "temporal_skip_max_frames", "temporal_skip_depth_tol_m",
            "enable_partition_warm_start", "enable_parallel_views",
            *_V3_JSON_KEYS,
        ):
            if key in raw:
                setattr(cfg, key, raw[key])
        if "roi" in raw and isinstance(raw["roi"], list) and len(raw["roi"]) == 2:
            cfg.roi_xyz_min = tuple(float(x) for x in raw["roi"][0])
            cfg.roi_xyz_max = tuple(float(x) for x in raw["roi"][1])
        if "roi_xyz_min" in raw:
            cfg.roi_xyz_min = tuple(float(x) for x in raw["roi_xyz_min"])
        if "roi_xyz_max" in raw:
            cfg.roi_xyz_max = tuple(float(x) for x in raw["roi_xyz_max"])
        if "row_origin_xyz_m" in raw:
            cfg.row_origin_xyz_m = tuple(float(x) for x in raw["row_origin_xyz_m"])
        if "extra_roi_crop_frac" in raw and raw["extra_roi_crop_frac"] is not None:
            cfg.extra_roi_crop_frac = tuple(float(x) for x in raw["extra_roi_crop_frac"])
        if "block" in raw and isinstance(raw["block"], dict):
            cfg.block = _merge_dataclass(cfg.block, raw["block"])
        if "jenga" in raw and isinstance(raw["jenga"], dict):
            jg = raw["jenga"]
            if "row_origin_xyz_m" in jg:
                cfg.row_origin_xyz_m = tuple(float(x) for x in jg["row_origin_xyz_m"])
            if "row_yaw_deg" in jg:
                cfg.row_yaw_deg = float(jg["row_yaw_deg"])
            if "block_dimensions" in jg and isinstance(jg["block_dimensions"], dict):
                cfg.block = _merge_dataclass(cfg.block, jg["block_dimensions"])
            if "occlusion_seconds" in jg:
                cfg.thresholds = _merge_dataclass(cfg.thresholds, {"occlusion_hold_s": float(jg["occlusion_seconds"])})
        if "thresholds" in raw and isinstance(raw["thresholds"], dict):
            cfg.thresholds = _merge_dataclass(cfg.thresholds, raw["thresholds"])
        if "wrist" in raw and isinstance(raw["wrist"], dict):
            cfg.wrist = _merge_dataclass(cfg.wrist, raw["wrist"])
        if "side" in raw and isinstance(raw["side"], dict):
            cfg.side = _merge_dataclass(cfg.side, raw["side"])
        if "third" in raw and isinstance(raw["third"], dict):
            cfg.third = _merge_dataclass(cfg.third, raw["third"])

    if overrides:
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
            elif hasattr(cfg.thresholds, k):
                cfg.thresholds = _merge_dataclass(cfg.thresholds, {k: v})
    return cfg


def config_hash(cfg: RuntimeConfig) -> str:
    payload = json.dumps(cfg.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def sha256_files(paths: list[Path]) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in sorted(paths):
        if p.is_file():
            out[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out

from dataclasses import replace

from wp3_live.config import preset_config
from wp3_live.geometry_cache import CameraModelCache
from wp3_live.policies import ViewInput, run_perception_epoch
from wp3_live.side_verifier import SideCutVerification, TwoSideCutVerifier
from scripts.run_printed_jenga_suite import make_synthetic_cameras, render_synthetic_scene


class FixedCutVerifier:
    def __init__(self, status):
        self.result = SideCutVerification(status, 1.0, 0.0, 0.0, 12.0, "test")

    def verify_cut(self, *_args):
        return self.result


def test_third_can_add_evidence_and_visible_conflict_is_preserved():
    weak = FixedCutVerifier("INCONCLUSIVE")
    support = FixedCutVerifier("CORROBORATED")
    conflict = FixedCutVerifier("CONTRADICTED")
    verify = TwoSideCutVerifier(weak, support)
    assert verify.verify_cut(None, 0.0, 0.015).status == "CORROBORATED"
    assert verify.third_verdicts == {"CORROBORATED": 1}
    assert TwoSideCutVerifier(support, conflict).verify_cut(None, 0.0, 0.015).status == "CONTRADICTED"
    unavailable = FixedCutVerifier("UNAVAILABLE")
    assert TwoSideCutVerifier(support, unavailable).verify_cut(None, 0.0, 0.015) is support.result


def test_third_is_evaluated_only_when_fresh_registered_and_synchronised():
    camera_wrist, table_wrist, camera_side, table_side = make_synthetic_cameras(CameraModelCache())
    rgb_w, depth_w, rgb_s, depth_s = render_synthetic_scene(
        camera_wrist, table_wrist, camera_side, table_side,
        [
            {"cx": 0.0, "cy": y, "z0": 0.0, "z1": 0.015, "L": 0.075, "W": 0.025, "yaw_rad": 0.0}
            for y in (-0.0125, 0.0125)
        ],
        seams_top=[{"p0": (-0.037, 0.0), "p1": (0.037, 0.0), "z": 0.015, "half_w_m": 0.0009}],
        end_breaks=[],
    )
    ts = 1_000_000_000
    wrist = ViewInput("wrist", camera_wrist, rgb_w, depth_w, table_wrist, ts, ts)
    side = ViewInput("tripod", camera_side, rgb_s, depth_s, table_side, ts, ts)
    third = replace(side, role="third")
    args = dict(wrist=wrist, side=side, config=preset_config("printed_jenga"), policy="whole_partition_mv")
    baseline = run_perception_epoch(**args)
    qualified = run_perception_epoch(**args, third=third)
    assert qualified.evidence_log["third_cut_status"] == "eligible"
    assert sum(qualified.evidence_log["third_cut_verdicts"].values()) > 0
    for unavailable, reason in (
        (replace(third, stamp_ns=ts + 500_000_000), "cross_view_skew"),
        (replace(third, T_table_cam=None), "unavailable_or_unregistered"),
        (replace(third, is_frozen=True), "unavailable_or_unregistered"),
    ):
        fallback = run_perception_epoch(**args, third=unavailable)
        assert fallback.evidence_log["third_cut_status"] == reason
        assert "third_cut_verdicts" not in fallback.evidence_log
        assert [pose.xy_m for pose in fallback.poses] == [pose.xy_m for pose in baseline.poses]

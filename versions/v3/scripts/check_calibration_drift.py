"""Session-start calibration check (V3.2, fault #10): has any camera moved since the calibration being published?

    python3 scripts/register_board.py /tmp/board_now REFERENCE.json --pitch-mm 20   # fresh fit, board on the table
    python3 scripts/check_calibration_drift.py PUBLISHED_FIT.json /tmp/board_now/current_board_tf.json

Exit 0: every camera within --max-mm / --max-deg (keep the published calibration).
Exit 1: drift: publish the fresh fit (publish_board_tf.py) and re-run the table touch-off; WP3 V3.2 picks the new
        /tf_static up within static_tf_refresh_s (older nodes and the twin bridge must be restarted).
Exit 2: unreadable input.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wp3_live.calib_check import compare_board_fits  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("published", type=Path, help="the fit the static TF publisher is serving")
    ap.add_argument("current", type=Path, help="a fresh register_board.py fit")
    ap.add_argument("--max-mm", type=float, default=5.0)
    ap.add_argument("--max-deg", type=float, default=1.0)
    ap.add_argument("--json", action="store_true", help="print the full result as JSON")
    a = ap.parse_args()
    try:
        ref = json.loads(a.published.read_text(encoding="utf-8"))
        cur = json.loads(a.current.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"cannot read fits: {exc}", file=sys.stderr)
        return 2
    if not cur.get("passed_stationary_smoke_gate", False):
        print("WARNING: the fresh fit did not pass register_board.py's smoke gate", file=sys.stderr)
    res = compare_board_fits(ref, cur, a.max_mm, a.max_deg)
    if a.json:
        print(json.dumps(res, indent=2))
    else:
        for link, d in res["links"].items():
            print(f"{link:12s} moved {d['dt_mm']:8.1f} mm {d['dr_deg']:7.2f} deg  {'ok' if d['ok'] else 'DRIFT'}")
        for pair, d in res["pairwise"].items():
            print(f"  {pair:26s} relative change {d['dt_mm']:8.1f} mm {d['dr_deg']:7.2f} deg")
        if res["missing"]:
            print(f"links in only one fit: {', '.join(res['missing'])}")
        print("PASS: calibration still valid" if res["passed"] else
              "FAIL: recalibrate (publish the fresh fit, redo the table touch-off)")
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

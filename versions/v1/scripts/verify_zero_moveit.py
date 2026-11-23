"""Static zero-MoveIt & zero-actuation safety verifier for this WP3 live perception snapshot."""
from __future__ import annotations

import ast
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PATTERNS = [
    re.compile(r"\bmoveit_msgs\b"),
    re.compile(r"\bApplyPlanningScene\b"),
    re.compile(r"\bGetPlanningScene\b"),
    re.compile(r"/apply_planning_scene"),
    re.compile(r"/get_planning_scene"),
    re.compile(r"\bxarm_msgs\b"),
    re.compile(r"\bFollowJointTrajectory\b"),
]

violations: list[str] = []
for p in sorted(ROOT.rglob("*")):
    if not p.is_file() or p.name == "verify_zero_moveit.py":
        continue
    if any(part in ("results", "build", "install", "log", "__pycache__") for part in p.parts):
        continue
    if p.suffix not in (".py", ".xml", ".json", ".yaml", ".sh"):
        continue
    text = p.read_text(encoding="utf-8", errors="replace")
    if p.suffix == ".py":
        tree = ast.parse(text, filename=str(p))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if "moveit" in alias.name.lower() or "xarm" in alias.name.lower():
                        violations.append(f"{p}:{node.lineno} forbidden import {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                if "moveit" in node.module.lower() or "xarm" in node.module.lower():
                    violations.append(f"{p}:{node.lineno} forbidden import {node.module}")
    for pat in FORBIDDEN_PATTERNS:
        for i, line in enumerate(text.splitlines(), start=1):
            if pat.search(line):
                violations.append(f"{p}:{i} matches {pat.pattern}")

if violations:
    for v in violations:
        print("FAIL:", v)
    sys.exit(1)
print("ZERO_MOVEIT_AUDIT: PASS (0 MoveIt/actuation references across wp3_live, ros2_ws, and scripts)")

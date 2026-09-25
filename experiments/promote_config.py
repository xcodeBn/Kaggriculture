"""Preview or embed a trained JSON policy in competition-facing submission.py."""
from __future__ import annotations

import argparse
import ast
import difflib
import json
from datetime import datetime
from pathlib import Path
from pprint import pformat
from typing import Any

from config import ROOT, load_config


SUBMISSION_PATH = ROOT / "submission.py"
TARGET_NAME = "COMPETITION_CONFIG"


def render_submission(source: str, config: dict[str, Any]) -> str:
    """Replace only the COMPETITION_CONFIG assignment, keeping agent logic intact."""
    tree = ast.parse(source)
    target = None
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == TARGET_NAME:
                target = node
                break
        if isinstance(node, ast.Assign) and any(
                isinstance(item, ast.Name) and item.id == TARGET_NAME
                for item in node.targets):
            target = node
            break
    if target is None:
        raise ValueError(f"Could not find top-level {TARGET_NAME} assignment")

    lines = source.splitlines(keepends=True)
    first, last = target.lineno - 1, target.end_lineno
    indent = " " * (len(lines[first]) - len(lines[first].lstrip()))
    assignment = [f"{indent}{TARGET_NAME}: Dict[str, Any] = {{\n"]
    assignment.extend(f"{indent}    {key!r}: {pformat(value)},\n"
                      for key, value in config.items())
    assignment.append(f"{indent}}}\n")
    lines[first:last] = assignment
    return "".join(lines)


def promote(config_path: Path, apply: bool = False,
            submission_path: Path = SUBMISSION_PATH) -> str:
    if not config_path.is_file():
        raise FileNotFoundError(f"Trained config does not exist: {config_path}")
    config = load_config(config_path)
    original = submission_path.read_text(encoding="utf-8")
    updated = render_submission(original, config)
    diff = "".join(difflib.unified_diff(
        original.splitlines(keepends=True), updated.splitlines(keepends=True),
        fromfile=str(submission_path), tofile=f"{submission_path} (promoted config)"))
    if not apply:
        return diff or "No changes: the selected config already matches submission.py.\n"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup_dir = ROOT / "experiments" / "results" / f"promotion_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    (backup_dir / "submission_before.py").write_text(original, encoding="utf-8")
    (backup_dir / "config_promoted.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    submission_path.write_text(updated, encoding="utf-8")
    return f"Promoted {config_path} into {submission_path}\nBackup: {backup_dir}\n{diff}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path,
                        help="trained config JSON, e.g. experiments/results/tpe_RUN/config_best.json")
    parser.add_argument("--apply", action="store_true",
                        help="write the change; without this flag only show a diff")
    parser.add_argument("--submission", type=Path, default=SUBMISSION_PATH,
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    print(promote(args.config, apply=args.apply, submission_path=args.submission), end="")


if __name__ == "__main__":
    main()

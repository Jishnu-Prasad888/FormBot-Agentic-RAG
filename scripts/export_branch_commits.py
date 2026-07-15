#!/usr/bin/env python3
"""Export Python file changes for each local branch into markdown files.

Run from any path inside the repo:
    python scripts/export_branch_commits.py

Creates a `branch_commits/` directory at the repo root with one markdown file
per branch, listing each commit's date, message, and diff for `.py` files.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def run_git(args: Iterable[str]) -> str:
    result = subprocess.run(
        ["git", *args], check=True, text=True, stdout=subprocess.PIPE
    )
    return result.stdout.strip()


def repo_root() -> Path:
    return Path(run_git(["rev-parse", "--show-toplevel"]))


def list_branches() -> list[str]:
    output = run_git(["for-each-ref", "refs/heads", "--format=%(refname:short)"])
    return [line for line in output.splitlines() if line]


def sanitize_branch(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return safe or "branch"


@dataclass
class CommitInfo:
    sha: str
    date: str
    message: str
    diff: str


def collect_commits(branch: str) -> list[CommitInfo]:
    shas = run_git(["rev-list", branch])
    if not shas:
        return []

    commits: list[CommitInfo] = []
    for sha in shas.splitlines():
        meta_raw = run_git(
            ["show", "-s", "--date=iso-strict", "--format=%H%x00%ad%x00%B", sha]
        )
        parts = meta_raw.split("\x00", 2)
        if len(parts) != 3:
            continue

        commit_sha, date, message = parts
        message = message.strip() or "<no commit message>"

        raw_diff = run_git(
            [
                "show",
                "--no-color",
                "--unified=3",
                "--pretty=format:",
                sha,
                "--",
                "*.py",
            ]
        )

        filtered_diff = "\n".join(
            line for line in raw_diff.splitlines() if not line.startswith("-")
        )

        commits.append(CommitInfo(commit_sha, date, message, filtered_diff))

    return commits


def write_markdown(branch: str, commits: list[CommitInfo], out_dir: Path) -> Path:
    safe_name = sanitize_branch(branch)
    path = out_dir / f"{safe_name}.md"

    lines: list[str] = []
    lines.append(f"# Branch {branch}")
    lines.append("")
    lines.append(
        f"Generated on {datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')}"
    )
    lines.append(f"Total commits: {len(commits)}")
    lines.append("")

    if not commits:
        lines.append("_No commits found._")
    else:
        for commit in commits:
            lines.append(f"## {commit.sha} — {commit.date}")
            lines.append("")
            lines.append("Message:")
            lines.append("")
            lines.append(commit.message)
            lines.append("")
            if commit.diff:
                lines.append("```diff")
                lines.append(commit.diff)
                lines.append("```")
            else:
                lines.append("_No Python file changes in this commit._")
            lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    root = repo_root()
    out_dir = root / "branch_commits"
    out_dir.mkdir(exist_ok=True)

    branches = list_branches()
    if not branches:
        print("No local branches found.")
        return

    for branch in branches:
        commits = collect_commits(branch)
        md_path = write_markdown(branch, commits, out_dir)
        print(f"Wrote {md_path.relative_to(root)}")


if __name__ == "__main__":
    main()

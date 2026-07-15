#!/usr/bin/env python3
"""Summarize per-branch commit logs using OpenAI.

Prerequisites:
- Ensure `backend/.env` contains `OPENAI_API_KEY`.
- Install requirements (includes `openai` and `python-dotenv`).

Usage:
    python scripts/summarize_branch_commits.py

Reads markdown files in `branch_commits/`, sends each to OpenAI for a ~2 page
summary, and writes results to `branch_commits_summaries/`.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

from dotenv import load_dotenv
from openai import OpenAI


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_env(env_paths: Iterable[Path]) -> None:
    for env_path in env_paths:
        if env_path.is_file():
            load_dotenv(env_path, override=False)


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY (expected in backend/.env)")
    return OpenAI(api_key=api_key)


def choose_model() -> str:
    # Fixed default per request.
    return "gpt-4o"


def chunk_text(text: str, max_chars: int = 60000) -> List[str]:
    if not text:
        return []

    chunks: List[str] = []
    buffer: List[str] = []
    current = 0

    for block in text.split("\n\n"):
        block_len = len(block) + 2  # account for the separator we will re-add
        if buffer and current + block_len > max_chars:
            chunks.append("\n\n".join(buffer))
            buffer = [block]
            current = block_len
        else:
            buffer.append(block)
            current += block_len

    if buffer:
        chunks.append("\n\n".join(buffer))

    return chunks


def chat_complete(client: OpenAI, model: str, system_prompt: str, user_content: str) -> str:
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )

    return response.choices[0].message.content.strip()


def update_running_summary(client: OpenAI, model: str, prior: str, chunk: str) -> str:
    return chat_complete(
        client,
        model,
        (
            "You maintain an evolving technical summary of a branch commit log."
            " Merge the existing summary with the new chunk, keep it coherent,"
            " deduplicated, and bounded to roughly 800-1200 words overall (never"
            " exceed ~1400). Use concise markdown sections and bullets, focusing"
            " on features, refactors, fixes, data/model changes, risks, and"
            " testing signals. Preserve important dates/names only when useful;"
            " avoid raw diff fragments."
        ),
        (
            "Current summary (may be empty):\n" + (prior or "<none yet>") +
            "\n\nNew chunk to incorporate:\n" + chunk +
            "\n\nReturn only the updated summary."
        ),
    )


def summarize_file(client: OpenAI, model: str, source: Path, dest: Path) -> None:
    content = source.read_text(encoding="utf-8")
    chunks = chunk_text(content)

    if not chunks:
        summary = "_Source file is empty; nothing to summarize._"
    else:
        running_summary = ""
        for idx, chunk in enumerate(chunks, start=1):
            print(f"  Chunk {idx}/{len(chunks)}", flush=True)
            running_summary = update_running_summary(client, model, running_summary, chunk)

        print("  Finalizing summary", flush=True)
        summary = chat_complete(
            client,
            model,
            (
                "You are a release historian. Polish the evolving summary into a"
                " coherent ~2 page (800-1200 words) technical narrative. Highlight"
                " major features, refactors, bug fixes, data/model changes, risks,"
                " and testing notes. Keep concise markdown sections and bullets;"
                " no raw diffs."
            ),
            running_summary,
        )

    lines: list[str] = []
    lines.append(f"# Summary for {source.stem}")
    lines.append("")
    lines.append(f"Source file: {source.name}")
    lines.append(
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')}"
    )
    lines.append(f"Model: {model}")
    lines.append(f"Chunks: {max(len(chunks), 1)}")
    lines.append("")
    lines.append(summary)

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    root = repo_root()
    input_dir = root / "branch_commits"
    output_dir = root / "branch_commits_summaries"

    load_env([root / "backend" / ".env"])
    client = get_client()
    model = choose_model()

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    md_files = sorted(p for p in input_dir.glob("*.md") if "summary" not in p.stem)
    if not md_files:
        raise SystemExit(f"No markdown files found in {input_dir}")

    for path in md_files:
        dest = output_dir / f"{path.stem}_summary.md"
        print(f"Summarizing {path.name} -> {dest.relative_to(root)} ...", flush=True)
        summarize_file(client, model, path, dest)
    print("Done.")


if __name__ == "__main__":
    main()

import datetime as _dt
import json
import sqlite3
from pathlib import Path
from typing import Iterable, Optional


DB_PATH = Path.home() / ".local" / "share" / "opencode" / "opencode.db"
OUTPUT_DIR = Path("session_exports")


def _utc_iso(ms: int) -> str:
    return _dt.datetime.fromtimestamp(ms / 1000, _dt.timezone.utc).isoformat()


def _load_project_id(conn: sqlite3.Connection, worktree: Path) -> Optional[str]:
    cur = conn.execute(
        "select id from project where worktree = ? limit 1", (str(worktree),)
    )
    row = cur.fetchone()
    if row:
        return row[0]

    cur = conn.execute(
        "select project_id from project_directory where directory = ? limit 1",
        (str(worktree),),
    )
    row = cur.fetchone()
    return row[0] if row else None


def _iter_sessions(conn: sqlite3.Connection, project_id: str):
    cur = conn.execute(
        "select id, title, time_created from session where project_id = ? order by time_created",
        (project_id,),
    )
    yield from cur.fetchall()


def _iter_parts(conn: sqlite3.Connection, session_id: str):
    cur = conn.execute(
        """
        select m.time_created, json_extract(m.data,'$.role') as role, p.data
        from message m
        join part p on m.id = p.message_id
        where m.session_id = ?
        order by m.time_created, p.time_created
        """,
        (session_id,),
    )
    yield from cur.fetchall()


def _format_part(time_created: int, role: Optional[str], part_json: str) -> str:
    ts = _utc_iso(time_created)
    try:
        data = json.loads(part_json)
    except Exception:
        return f"[{ts}] {role or 'unknown'} | (unparsed) {part_json}"

    ptype = data.get("type") or "unknown"
    if ptype == "text":
        text = data.get("text", "")
        return f"[{ts}] {role or 'unknown'} | text:\n{text}\n"
    return f"[{ts}] {role or 'unknown'} | {ptype}: {json.dumps(data, ensure_ascii=True)}"


def export_sessions(db_path: Path, worktree: Path, output_dir: Path) -> None:
    if not db_path.exists():
        raise FileNotFoundError(f"opencode db not found at {db_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        project_id = _load_project_id(conn, worktree)
        if not project_id:
            raise RuntimeError(f"No project found for worktree {worktree}")

        for sid, title, created in _iter_sessions(conn, project_id):
            lines: Iterable[str] = []
            parts = list(_iter_parts(conn, sid))
            out_lines = [
                f"Session: {sid}",
                f"Title: {title}",
                f"Created: {_utc_iso(created)}",
                "",
            ]

            if not parts:
                out_lines.append("(no parts found)")
            else:
                for t_created, role, part_json in parts:
                    out_lines.append(_format_part(t_created, role, part_json))

            outfile = output_dir / f"{sid}.txt"
            outfile.write_text("\n".join(out_lines), encoding="utf-8")


if __name__ == "__main__":
    export_sessions(DB_PATH, Path.cwd(), OUTPUT_DIR)

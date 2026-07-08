"""Bulk upload documents from a folder to the Simple RAG API.

Configure the globals below (BASE_URL, FOLDER, METADATA, TIMEOUT) and run:
  python upload_documents.py

Notes:
  - BASE_URL defaults to env SRAG_API_URL or http://localhost:9000.
  - Shows two progress bars: one for uploads, one for indexing (server-side).
  - Failed uploads are logged to failed_uploads.txt if any remain.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import requests
from requests.exceptions import RequestException, Timeout
from tqdm import tqdm


# ── Configurable globals ──────────────────────────────────────────────────────
BASE_URL = os.environ.get("SRAG_API_URL", "http://localhost:9000")
# Set this to the folder containing the documents you want to upload
FOLDER = Path("/path/to/folder")
# Optional metadata sent with every file (set to a dict or None)
METADATA: dict | None = None
TIMEOUT = 180  # seconds


def iter_files(folder: Path) -> Iterable[Path]:
    for path in folder.rglob("*"):
        if path.is_file():
            yield path


def upload_file(path: Path, upload_url: str, metadata: dict | None, timeout: int) -> dict:
    data = {"metadata": json.dumps(metadata)} if metadata else None
    with open(path, "rb") as f:
        files = {"file": (path.name, f)}
        resp = requests.post(upload_url, files=files, data=data, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def main():
    folder: Path = FOLDER.expanduser().resolve()
    if not folder.is_dir():
        raise SystemExit(f"Folder not found: {folder}")

    meta_obj = METADATA

    files = list(iter_files(folder))
    if not files:
        raise SystemExit("No files found to upload.")

    upload_url = BASE_URL.rstrip("/") + "/api/documents/upload"
    print(f"Uploading {len(files)} files to {upload_url}")

    upload_bar = tqdm(total=len(files), desc="Uploading", unit="file")
    index_bar = tqdm(total=len(files), desc="Indexed", unit="file")

    failures_timeout: list[tuple[Path, str]] = []
    failures_other: list[tuple[Path, str]] = []

    for path in files:
        try:
            _ = upload_file(path, upload_url, meta_obj, timeout=TIMEOUT)
            upload_bar.update(1)
            index_bar.update(1)
        except Timeout as exc:
            failures_timeout.append((path, str(exc)))
            upload_bar.set_postfix(error="timeout")
            index_bar.set_postfix(error="timeout")
        except RequestException as exc:
            failures_other.append((path, str(exc)))
            upload_bar.set_postfix(error="request error")
            index_bar.set_postfix(error="request error")
        except Exception as exc:  # noqa: BLE001
            failures_other.append((path, str(exc)))
            upload_bar.set_postfix(error="error")
            index_bar.set_postfix(error="error")

    upload_bar.close()
    index_bar.close()

    combined_failures = failures_timeout + failures_other

    if combined_failures:
        print("Finished first pass with errors:")
        if failures_timeout:
            print("Timeouts:")
            for path, err in failures_timeout:
                print(f"- {path}: {err}")
        if failures_other:
            print("Other errors:")
            for path, err in failures_other:
                print(f"- {path}: {err}")

        retry = input("Retry failed uploads now? (y/N): ").strip().lower() == "y"
        final_failures = combined_failures
        if retry and combined_failures:
            retry_bar = tqdm(total=len(combined_failures), desc="Retrying", unit="file")
            still_failed: list[tuple[Path, str]] = []
            for path, _ in combined_failures:
                try:
                    _ = upload_file(path, upload_url, meta_obj, timeout=TIMEOUT)
                except Exception as exc:  # noqa: BLE001
                    still_failed.append((path, str(exc)))
                    retry_bar.set_postfix(error="error")
                else:
                    retry_bar.set_postfix(error="")
                retry_bar.update(1)
            retry_bar.close()

            if still_failed:
                print("Still failed after retry:")
                for path, err in still_failed:
                    print(f"- {path}: {err}")
                final_failures = still_failed
            else:
                print("All previously failed files uploaded on retry.")
                final_failures = []
        else:
            print("Skipped retry.")
            final_failures = combined_failures

        if final_failures:
            fail_log = Path(__file__).with_name("failed_uploads.txt")
            with open(fail_log, "w") as f:
                for path, err in final_failures:
                    f.write(f"{path}\t{err}\n")
            print(f"Saved failed uploads to {fail_log} for later retry.")
    else:
        print("All files uploaded and indexed successfully.")


if __name__ == "__main__":
    main()

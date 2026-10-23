"""
E6 Test Set Lock Management.

Guarantees that Phase E6 holdout test evaluation is strictly single-use:
- When test data is accessed for evaluation, creates outputs/logs/E6_LOCK containing
  timestamp, user/host, and code version SHA-256 hash.
- Subsequent calls to evaluate test data are rejected with RuntimeError unless the lock
  file is manually deleted by the user.
- Allows reading saved evaluation results and re-plotting without re-evaluating.
"""

from __future__ import annotations

import datetime
import hashlib
import os
from pathlib import Path
from typing import Any

import config

LOGS_DIR = Path(config.LOGS_DIR)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_LOCK_PATH = LOGS_DIR / "E6_LOCK"


def compute_codebase_hash(src_dir: Path | str | None = None) -> str:
    """Computes SHA-256 hash of all python files in src directory."""
    if src_dir is None:
        src_dir = Path(__file__).resolve().parent
    src_path = Path(src_dir)

    hasher = hashlib.sha256()
    for py_file in sorted(src_path.glob("*.py")):
        hasher.update(py_file.name.encode("utf-8"))
        hasher.update(py_file.read_bytes())
    return hasher.hexdigest()


def acquire_e6_lock(
    lock_path: Path | str = DEFAULT_LOCK_PATH,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, str]:
    """
    Acquires the E6 lock before reading the test set.

    Raises:
        RuntimeError: If E6_LOCK already exists, preventing duplicate evaluation.
    """
    path = Path(lock_path)
    if path.is_dir():
        path = path / "E6_LOCK"

    if path.exists():
        content = path.read_text(encoding="utf-8")
        raise RuntimeError(
            f"E6_LOCK file already exists at {path}!\n"
            "The holdout test set has already been evaluated. Duplicate evaluation is strictly prohibited.\n"
            f"Lock details:\n{content}\n"
            "To re-evaluate, the lock file must be manually removed by the user."
        )

    now = datetime.datetime.now().isoformat()
    code_hash = compute_codebase_hash()

    lock_payload_lines = [
        f"LOCKED_AT: {now}",
        f"CODEBASE_HASH: {code_hash}",
        "DESCRIPTION: E6 final holdout test set evaluation locked.",
    ]
    if extra_metadata:
        import json
        lock_payload_lines.append("METADATA:")
        lock_payload_lines.append(json.dumps(extra_metadata, indent=2))

    lock_payload = "\n".join(lock_payload_lines) + "\n"

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(lock_payload, encoding="utf-8")

    return {
        "locked_at": now,
        "code_hash": code_hash,
        "lock_path": str(path),
    }


def is_e6_locked(lock_path: Path | str = DEFAULT_LOCK_PATH) -> bool:
    """Checks whether the E6 evaluation is currently locked."""
    path = Path(lock_path)
    if path.is_dir():
        path = path / "E6_LOCK"
    return path.exists()

"""
Unit tests for E6 lock mechanism.
"""

import pytest
from pathlib import Path

from src.lock_utils import acquire_e6_lock, is_e6_locked, compute_codebase_hash


def test_codebase_hash():
    h = compute_codebase_hash()
    assert isinstance(h, str)
    assert len(h) == 64


def test_e6_lock_lifecycle(tmp_path: Path):
    lock_file = tmp_path / "TEST_E6_LOCK"
    assert not is_e6_locked(lock_file)

    # First acquisition succeeds
    info = acquire_e6_lock(lock_file)
    assert is_e6_locked(lock_file)
    assert lock_file.exists()
    assert "LOCKED_AT" in lock_file.read_text(encoding="utf-8")

    # Second acquisition must fail
    with pytest.raises(RuntimeError, match="E6_LOCK file already exists"):
        acquire_e6_lock(lock_file)

    # Manual deletion unlocks
    lock_file.unlink()
    assert not is_e6_locked(lock_file)

    # Acquisition succeeds again
    info2 = acquire_e6_lock(lock_file)
    assert is_e6_locked(lock_file)

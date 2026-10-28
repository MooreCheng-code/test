"""
tests/test_frozen_hashes.py

Automated regression test verifying that all 11 frozen model components in
outputs/models/final/ match the bitwise SHA-256 hashes recorded in
outputs/logs/e6_preflight.txt Check 7/8.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FINAL_DIR = PROJECT_ROOT / "outputs" / "models" / "final"
PREFLIGHT_PATH = PROJECT_ROOT / "outputs" / "logs" / "e6_preflight.txt"

EXPECTED_FROZEN_FILES = [
    "model_M3.joblib",
    "model_M4.joblib",
    "model_M5.joblib",
    "reference_M3.npy",
    "reference_M4.npy",
    "reference_M5.npy",
    "reference_y.npy",
    "thresholds.json",
    "thresholds.csv",
    "calibrator_platt.joblib",
    "calibrator_isotonic.joblib",
]


def parse_preflight_hashes() -> dict[str, str]:
    """Parse recorded SHA256 hashes from outputs/logs/e6_preflight.txt Check 7/8."""
    assert PREFLIGHT_PATH.exists(), f"Preflight log not found at: {PREFLIGHT_PATH}"
    content = PREFLIGHT_PATH.read_text(encoding="utf-8")

    assert "[Check 7/8]" in content, "Check 7/8 not found in e6_preflight.txt"

    recorded = {}
    in_check7 = False
    for line in content.splitlines():
        if "[Check 7/8]" in line:
            in_check7 = True
            continue
        if in_check7:
            if "-> PASS" in line or (line.strip() and not line.startswith("   ")):
                break
            m = re.match(r"^\s*([^:]+):\s*([0-9a-f]{64})\s*$", line)
            if m:
                recorded[m.group(1).strip()] = m.group(2).strip()

    return recorded


def test_frozen_hashes_match_e6_preflight():
    """Verify all 11 frozen model components match recorded preflight SHA-256 hashes exactly."""
    recorded_hashes = parse_preflight_hashes()

    # Preflight log must record all 11 expected components
    missing_in_log = [fname for fname in EXPECTED_FROZEN_FILES if fname not in recorded_hashes]
    assert not missing_in_log, f"Preflight Check 7/8 missing expected files: {missing_in_log}"
    assert len(recorded_hashes) == len(EXPECTED_FROZEN_FILES), (
        f"Expected exactly {len(EXPECTED_FROZEN_FILES)} recorded files, got {len(recorded_hashes)}"
    )

    # Every file must exist on disk and its SHA256 must match bitwise
    for fname in EXPECTED_FROZEN_FILES:
        target_path = FINAL_DIR / fname
        assert target_path.exists(), f"Frozen model file missing on disk: {target_path}"

        actual_sha = hashlib.sha256(target_path.read_bytes()).hexdigest()
        expected_sha = recorded_hashes[fname]

        assert actual_sha == expected_sha, (
            f"Hash mismatch for {fname}:\n  Actual:   {actual_sha}\n  Expected: {expected_sha}"
        )

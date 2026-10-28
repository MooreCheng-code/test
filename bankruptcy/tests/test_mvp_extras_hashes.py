"""
tests/test_mvp_extras_hashes.py

Automated regression test verifying that all 10 frozen MVP expansion components
in outputs/models/mvp_extras/ match the bitwise SHA-256 hashes and file sizes
recorded in outputs/models/mvp_extras/mvp_extras_hashes.json.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MVP_EXTRAS_DIR = PROJECT_ROOT / "outputs" / "models" / "mvp_extras"
HASH_MANIFEST_PATH = MVP_EXTRAS_DIR / "mvp_extras_hashes.json"

EXPECTED_MVP_EXTRA_FILES = [
    "feature_manifest.json",
    "model_M1.joblib",
    "reference_M1.npy",
    "reference_E_A.npy",
    "calibrator_M1_platt.joblib",
    "calibrator_M3_platt.joblib",
    "calibrator_M4_platt.joblib",
    "thresholds_extras.json",
    "demo_r0_f0_M4.joblib",
    "demo_r0_f0_M1.joblib",
    "oof_demo_batch.csv",
]


def test_mvp_extras_manifest_exists():
    """Verify hash manifest file exists."""
    assert HASH_MANIFEST_PATH.exists(), f"Hash manifest not found at: {HASH_MANIFEST_PATH}"


def test_mvp_extras_hashes_and_sizes_match():
    """Verify all 10 frozen MVP extra components match recorded SHA-256 hashes and sizes."""
    assert HASH_MANIFEST_PATH.exists(), f"Hash manifest not found at: {HASH_MANIFEST_PATH}"

    with open(HASH_MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    recorded_files = manifest.get("files", {})

    # Manifest must record all 10 expected components
    missing_in_manifest = [fname for fname in EXPECTED_MVP_EXTRA_FILES if fname not in recorded_files]
    assert not missing_in_manifest, f"Manifest missing expected files: {missing_in_manifest}"
    assert len(recorded_files) == len(EXPECTED_MVP_EXTRA_FILES), (
        f"Expected exactly {len(EXPECTED_MVP_EXTRA_FILES)} files in manifest, got {len(recorded_files)}"
    )

    # Every file must exist on disk, matching SHA-256 and byte size
    for fname in EXPECTED_MVP_EXTRA_FILES:
        target_path = MVP_EXTRAS_DIR / fname
        assert target_path.exists(), f"MVP extra file missing on disk: {target_path}"

        file_bytes = target_path.read_bytes()
        actual_sha = hashlib.sha256(file_bytes).hexdigest()
        actual_size = len(file_bytes)

        expected_sha = recorded_files[fname]["sha256"]
        expected_size = recorded_files[fname]["size_bytes"]

        assert actual_size == expected_size, (
            f"Size mismatch for {fname}:\n  Actual:   {actual_size}\n  Expected: {expected_size}"
        )
        assert actual_sha == expected_sha, (
            f"Hash mismatch for {fname}:\n  Actual:   {actual_sha}\n  Expected: {expected_sha}"
        )

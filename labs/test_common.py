"""labs/test_common.py - Unit tests for labs/_common.py.

Verifies:
1. Guard assertion halts on full data (6,819 rows) or invalid shapes.
2. Training data loader respects safe 5,455 rows and 176 positives.
3. 5-fold CV splits have ~35-36 positives per fold.
4. load_best_params() successfully loads frozen JSON files read-only.
5. save_manifest() creates valid JSON with Git HEAD and package versions.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from labs._common import (
    assert_train_data_guard,
    get_cv_splitter,
    load_best_params,
    load_train_data,
    save_manifest,
)


class TestLabCommon(unittest.TestCase):
    """Test suite for lab common infrastructure and safety guards."""

    def test_guard_success_on_valid_train_data(self) -> None:
        """Verify that valid 5455 rows and 176 positives pass the guard."""
        X_mock = pd.DataFrame(np.zeros((5455, 10)))
        y_mock = pd.Series([1] * 176 + [0] * (5455 - 176))
        # Should not raise
        assert_train_data_guard(X_mock, y_mock)

    def test_guard_failure_on_full_dataset(self) -> None:
        """Negative test: Passing 6,819 rows must trigger RuntimeError."""
        X_full = pd.DataFrame(np.zeros((6819, 10)))
        y_full = pd.Series([1] * 220 + [0] * (6819 - 220))
        with self.assertRaises(RuntimeError) as ctx:
            assert_train_data_guard(X_full, y_full)
        self.assertIn("GUARD FAILURE", str(ctx.exception))
        self.assertIn("6819 rows", str(ctx.exception))

    def test_guard_failure_on_wrong_row_count(self) -> None:
        """Negative test: Passing wrong row count (e.g. 5000) must trigger RuntimeError."""
        X_wrong = pd.DataFrame(np.zeros((5000, 10)))
        y_wrong = pd.Series([1] * 176 + [0] * (5000 - 176))
        with self.assertRaises(RuntimeError) as ctx:
            assert_train_data_guard(X_wrong, y_wrong)
        self.assertIn("GUARD FAILURE", str(ctx.exception))
        self.assertIn("5455", str(ctx.exception))

    def test_guard_failure_on_wrong_positives(self) -> None:
        """Negative test: Passing wrong positive count (e.g. 150) must trigger RuntimeError."""
        X_mock = pd.DataFrame(np.zeros((5455, 10)))
        y_mock = pd.Series([1] * 150 + [0] * (5455 - 150))
        with self.assertRaises(RuntimeError) as ctx:
            assert_train_data_guard(X_mock, y_mock)
        self.assertIn("GUARD FAILURE", str(ctx.exception))
        self.assertIn("176", str(ctx.exception))

    def test_load_train_data_real(self) -> None:
        """Verify real load_train_data() call returns exact train shape."""
        X, y = load_train_data(check_guard=True)
        self.assertEqual(len(X), 5455)
        self.assertEqual(len(y), 5455)
        self.assertEqual(int(y.sum()), 176)

    def test_cv_splits_positive_counts(self) -> None:
        """Verify 5-fold CV on 5455 train samples yields ~35-36 positives per fold."""
        X, y = load_train_data(check_guard=True)
        skf = get_cv_splitter(n_splits=5, n_repeats=1, seed=42)
        pos_counts = []
        for train_idx, val_idx in skf.split(X, y):
            val_pos = int(y.iloc[val_idx].sum())
            pos_counts.append(val_pos)
            self.assertEqual(len(val_idx), len(X) // 5 + (1 if len(val_idx) > len(X) // 5 else 0))
            self.assertIn(val_pos, [35, 36])
        self.assertEqual(sum(pos_counts), 176)

    def test_load_best_params_all_models(self) -> None:
        """Verify read-only loading of best_params for M1-M5."""
        for m in ["M1", "M2", "M3", "M4", "M5"]:
            data = load_best_params(m)
            self.assertEqual(data["model"], m)
            self.assertIn("best_params", data)
            self.assertIn("imbalance", data)

    def test_save_manifest(self) -> None:
        """Verify run_manifest.json contains required metadata fields."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            params = {"lr": 0.05, "n_est": 100}
            manifest_file = save_manifest(
                output_dir=temp_dir,
                lab_id="test_lab",
                params=params,
                elapsed_seconds=1.23,
                seed=42,
            )
            self.assertTrue(manifest_file.is_file())
            with open(manifest_file, "r", encoding="utf-8") as f:
                content = json.load(f)
            self.assertEqual(content["lab_id"], "test_lab")
            self.assertEqual(content["seed"], 42)
            self.assertIn("git_head", content)
            self.assertIn("packages", content)
            self.assertEqual(content["params"], params)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

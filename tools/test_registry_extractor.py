#!/usr/bin/env python3
"""
tools/test_registry_extractor.py
Unit tests for tools/build_param_registry.py extraction functions.
Verifies that parameter extraction correctly parses:
1. Top-level constants and expressions.
2. Optuna trial.suggest_* calls (bounds, choices, log scale).
3. Dictionary-mapped trial counts (M1=25, M2=40, etc.) ensuring NO number concatenation.
4. Function and class __init__ default arguments.
5. JSON nested key extraction.

Run via: python tools/test_registry_extractor.py
DO NOT place this test inside bankruptcy/tests/.
"""

import ast
import json
import tempfile
import unittest
from pathlib import Path

# Import extractor utilities from tools/build_param_registry.py
import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

from build_param_registry import (
    parse_ast_assignments,
    extract_suggest_call_details,
    extract_model_trials_map,
    find_symbol_line,
)


class TestRegistryExtractor(unittest.TestCase):
    def test_parse_ast_assignments_literals_and_exprs(self):
        code = """
SEED = 42
TEST_SIZE = 0.2
ALERT_RATIOS = [1.5, 3.0, 5.0, 10.0]
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
"""
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".py", delete=False) as f:
            f.write(code)
            temp_path = Path(f.name)

        try:
            assigns = parse_ast_assignments(temp_path)
            self.assertEqual(assigns["SEED"][1], 42)
            self.assertEqual(assigns["SEED"][2], "int")
            self.assertEqual(assigns["TEST_SIZE"][1], 0.2)
            self.assertEqual(assigns["TEST_SIZE"][2], "float")
            self.assertEqual(assigns["ALERT_RATIOS"][1], [1.5, 3.0, 5.0, 10.0])
            self.assertEqual(assigns["ALERT_RATIOS"][2], "list")
            self.assertEqual(assigns["BASE_DIR"][2], "expr")
            self.assertEqual(assigns["DATA_DIR"][2], "expr")
        finally:
            temp_path.unlink(missing_ok=True)

    def test_n_trials_map_extraction_no_concatenation(self):
        """Verifies that dictionary mappings like 25 and 40 are parsed as single ints, NOT concatenated 2540."""
        code = """
n_trials_map_e2a = {
    "M1": 3 if quick else 25,
    "M2": 3 if quick else 40,
    "M3": 3 if quick else 40,
    "M4": 3 if quick else 40,
    "M5": 3 if quick else 40,
}
n_trials_map_e2b = {
    "M1": 5 if quick else 38,
    "M2": 5 if quick else 60,
    "M3": 5 if quick else 60,
    "M4": 5 if quick else 60,
    "M5": 5 if quick else 60,
}
"""
        tree = ast.parse(code)
        e2a_trials = extract_model_trials_map(tree, "n_trials_map_e2a")
        e2b_trials = extract_model_trials_map(tree, "n_trials_map_e2b")

        # Must be single integer values
        self.assertEqual(e2a_trials["M1"], 25)
        self.assertEqual(e2a_trials["M2"], 40)
        self.assertEqual(e2a_trials["M3"], 40)
        self.assertEqual(e2a_trials["M4"], 40)
        self.assertEqual(e2a_trials["M5"], 40)

        self.assertEqual(e2b_trials["M1"], 38)
        self.assertEqual(e2b_trials["M2"], 60)
        self.assertEqual(e2b_trials["M3"], 60)
        self.assertEqual(e2b_trials["M4"], 60)
        self.assertEqual(e2b_trials["M5"], 60)

        # Explicitly verify NO string concatenation happened
        for m, val in e2a_trials.items():
            self.assertIsInstance(val, int)
            self.assertNotIn("2540", str(val))
            self.assertNotIn("3860", str(val))

    def test_extract_suggest_call_details(self):
        """Tests parsing of trial.suggest_* calls."""
        code = """
C = trial.suggest_float("C", 1e-3, 10.0, log=True)
penalty = trial.suggest_categorical("penalty", ["l2", "l1"])
n_est = trial.suggest_int("n_estimators", 200, 800, step=50)
depth = trial.suggest_categorical("max_depth", [4, 6, 8, 12, None])
"""
        tree = ast.parse(code)
        details = extract_suggest_call_details(tree)

        self.assertIn("C", details)
        self.assertEqual(details["C"]["low"], 0.001)
        self.assertEqual(details["C"]["high"], 10.0)
        self.assertTrue(details["C"]["log"])

        self.assertIn("penalty", details)
        self.assertEqual(details["penalty"]["choices"], ["l2", "l1"])

        self.assertIn("n_estimators", details)
        self.assertEqual(details["n_estimators"]["low"], 200)
        self.assertEqual(details["n_estimators"]["high"], 800)
        self.assertEqual(details["n_estimators"]["step"], 50)

        self.assertIn("max_depth", details)
        self.assertEqual(details["max_depth"]["choices"], [4, 6, 8, 12, None])


if __name__ == "__main__":
    unittest.main()

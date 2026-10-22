"""
Unit tests for src/scoring.py inference functions.
"""

import numpy as np
import pandas as pd
import pytest

from src.scoring import to_percentile, ensemble_score, flag, score_dataset


def test_to_percentile_monotonicity_and_bounds():
    ref = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    q = np.array([0.0, 0.1, 0.25, 0.4, 0.5, 0.9])
    pct = to_percentile(q, ref)
    assert len(pct) == len(q)
    assert np.all(pct >= 0.0)
    assert np.all(pct <= 1.0)
    assert np.all(np.diff(pct) >= 0.0)


def test_to_percentile_empty_reference_error():
    with pytest.raises(ValueError, match="reference_scores cannot be empty"):
        to_percentile([0.1, 0.2], np.array([]))


def test_ensemble_score_identical_inputs():
    ref = np.linspace(0, 1, 100)
    scores = np.array([0.2, 0.5, 0.8])
    scores_dict = {"M3": scores, "M4": scores.copy(), "M5": scores.copy()}
    ref_dict = {"M3": ref, "M4": ref, "M5": ref}

    ens = ensemble_score(scores_dict, ref_dict, ["M3", "M4", "M5"])
    single_pct = to_percentile(scores, ref)
    assert np.allclose(ens, single_pct)


def test_ensemble_score_missing_keys():
    ref_dict = {"M3": np.array([0.1, 0.5]), "M4": np.array([0.2, 0.6])}
    scores_dict = {"M3": np.array([0.2]), "M4": np.array([0.3])}

    with pytest.raises(KeyError, match="missing from model_scores_dict"):
        ensemble_score(scores_dict, ref_dict, ["M3", "M4", "M5"])

    scores_dict["M5"] = np.array([0.4])
    with pytest.raises(KeyError, match="missing from reference_dict"):
        ensemble_score(scores_dict, ref_dict, ["M3", "M4", "M5"])


def test_flag_boundary_and_vectorization():
    threshold = 0.75
    scores = np.array([0.74, 0.75, 0.76])
    flags = flag(scores, threshold)
    assert np.array_equal(flags, np.array([False, True, True]))

    assert flag(0.75, threshold) is True or bool(flag(0.75, threshold)) is True
    assert flag(0.74, threshold) is False or bool(flag(0.74, threshold)) is False


def test_score_dataset_type_error():
    with pytest.raises(TypeError, match="must be a pandas DataFrame"):
        score_dataset(np.array([[1, 2], [3, 4]]), components={})

"""
Unit tests for E3: Ensemble, ECDF mapping, and model combinations.
"""

import numpy as np
import pytest
from scipy import stats

from src.ensemble import ecdf_map, compute_ensemble_scores


def test_ecdf_map_monotonicity_and_bounds():
    """ECDF mapping must be monotonically non-decreasing and bounded in [0, 1]."""
    ref = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    query = np.array([-1.0, 0.05, 0.1, 0.25, 0.3, 0.5, 0.8, 10.0])

    pct = ecdf_map(ref, query)

    assert len(pct) == len(query)
    assert np.all(pct >= 0.0)
    assert np.all(pct <= 1.0)
    # Monotonicity
    assert np.all(np.diff(pct) >= 0.0)


def test_ecdf_map_tie_handling_and_symmetry():
    """ECDF mapping handles tied values using symmetric midpoint."""
    # 5 elements with ties: 0.2 appears twice
    ref = np.array([0.1, 0.2, 0.2, 0.5, 0.8])
    # For query = 0.2: left=1, right=3 -> (1+3)/(2*5) = 0.4
    pct = ecdf_map(ref, np.array([0.2]))
    assert np.isclose(pct[0], 0.4)

    # Identical queries must have identical outputs
    q_dups = np.array([0.3, 0.3, 0.3])
    pct_dups = ecdf_map(ref, q_dups)
    assert len(np.unique(pct_dups)) == 1


def test_compute_ensemble_scores_equal_weights():
    """Ensemble with identical model scores equals that single model score."""
    n_samples = 20
    np.random.seed(42)
    s1 = np.random.uniform(0, 1, size=n_samples)

    # Dictionary of 3 models with identical predictions
    scores_dict = {
        "M1": s1,
        "M2": s1.copy(),
        "M3": s1.copy(),
    }
    ens = compute_ensemble_scores(scores_dict, ["M1", "M2", "M3"])
    assert np.allclose(ens, s1)


def test_compute_ensemble_scores_averaging():
    """Ensemble correctly averages model scores."""
    s1 = np.array([0.1, 0.5, 0.9])
    s2 = np.array([0.3, 0.5, 0.7])
    scores_dict = {"M1": s1, "M2": s2}

    ens = compute_ensemble_scores(scores_dict, ["M1", "M2"])
    expected = (s1 + s2) / 2.0
    assert np.allclose(ens, expected)


def test_compute_ensemble_candidate_subsets():
    """Candidate ensembles only include specified model keys."""
    scores_dict = {
        "M1": np.array([0.1, 0.2]),
        "M2": np.array([0.3, 0.4]),
        "M3": np.array([0.5, 0.6]),
        "M4": np.array([0.7, 0.8]),
        "M5": np.array([0.9, 1.0]),
    }

    # E_A: M3, M4, M5
    ens_A = compute_ensemble_scores(scores_dict, ["M3", "M4", "M5"])
    assert np.isclose(ens_A[0], (0.5 + 0.7 + 0.9) / 3.0)
    assert np.isclose(ens_A[1], (0.6 + 0.8 + 1.0) / 3.0)

    # Missing model raises KeyError
    with pytest.raises(KeyError):
        compute_ensemble_scores(scores_dict, ["M1", "M6"])

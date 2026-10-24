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


def test_load_final_components_missing_calibrator_error(tmp_path):
    """Verifies that load_final_components raises FileNotFoundError if calibrators are missing."""
    from src.scoring import load_final_components
    # Create empty dir
    empty_model_dir = tmp_path / "models_empty"
    empty_model_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="Calibrator file"):
        load_final_components(empty_model_dir)


def test_score_dataset_missing_calibrators_raises_error():
    """Verifies that score_dataset raises KeyError if calibrators are missing (never outputs None)."""
    from pathlib import Path
    from src.scoring import load_final_components
    from src.cv import get_train_data
    final_dir = Path(__file__).resolve().parent.parent / "outputs" / "models" / "final"
    components = load_final_components(final_dir)

    # Empty calibrators
    comps_corrupt = dict(components)
    comps_corrupt["calibrators"] = {}

    X_tr, _ = get_train_data()
    sample_df = X_tr.iloc[:5].copy()
    with pytest.raises(KeyError, match="Calibrator"):
        score_dataset(sample_df, comps_corrupt)

    # Missing platt specifically
    comps_corrupt2 = dict(components)
    comps_corrupt2["calibrators"] = {"isotonic": components["calibrators"]["isotonic"]}
    with pytest.raises(KeyError, match="Calibrator 'platt' missing"):
        score_dataset(sample_df, comps_corrupt2)


def test_score_dataset_missing_thresholds_raises_when_flags_requested():
    """Verifies that score_dataset raises KeyError if thresholds are missing but requested."""
    from pathlib import Path
    from src.scoring import load_final_components
    from src.cv import get_train_data
    final_dir = Path(__file__).resolve().parent.parent / "outputs" / "models" / "final"
    components = load_final_components(final_dir)

    comps_no_thresh = dict(components)
    comps_no_thresh.pop("thresholds", None)

    X_tr, _ = get_train_data()
    sample_df = X_tr.iloc[:5].copy()
    with pytest.raises(KeyError, match="Thresholds configuration missing"):
        score_dataset(sample_df, comps_no_thresh, include_threshold_flags=True)

    # When flags are explicitly not requested, missing thresholds should not raise
    df_no_flags = score_dataset(sample_df, comps_no_thresh, include_threshold_flags=False)
    assert not any(c.startswith("flag_") for c in df_no_flags.columns)


"""
Local Feature Attribution and Interpretability Module (src/local_explain.py).

Provides single-case and batch local feature attribution for MVP:
1. M4 (XGBoost): TreeSHAP values (raw margin log-odds) via shap.TreeExplainer.
2. M1 (Regularized Logistic Regression): Standardized linear contributions (w_j * z_j).
3. Modes:
   - mode="final": Uses outputs/models/final/model_M4.joblib and outputs/models/mvp_extras/model_M1.joblib.
   - mode="demo_oof": Uses outputs/models/mvp_extras/demo_r0_f0_M4.joblib and demo_r0_f0_M1.joblib.
   Zero runtime dependency on e5_cache/ or logs/e2_cache/.
4. Missingness safeguards:
   - Rejects samples with missing feature ratio > 20% (raises ValueError).
   - Imputes valid samples with train_medians_95 non-intrusively prior to pipeline transformation.
5. Disclaimers:
   - Clear attribution methodology (log-odds units, M4 vs E_A distinction, multicollinearity notice).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import shap

from src.explain import (
    get_underlying_classifier,
    transform_through_preprocessing,
)
from src.scoring import load_mvp_components

# Project root path
project_root = Path(__file__).resolve().parent.parent

DEFAULT_DISCLAIMER = (
    "本局部歸因基於 {model_desc} 計算，單位為對數勝算比（log-odds）。"
    "當前主要排序與評分模型為 E_A (三模型等權集成)，"
    "單一模型歸因僅供業務審查時掌握風險驅動方向之參考；"
    "由於財務變數間存在高度共線性，相近指標間的權重分配具有替代性，"
    "不代表因果關係或單一變數變動對破產機率的邊際效應。"
)


class LocalExplainer:
    """
    Manages model pipelines and SHAP explainers for local case attribution.
    Caches loaded pipelines and TreeExplainers to ensure sub-second inference.
    """

    def __init__(
        self,
        final_dir: Path | str | None = None,
        extras_dir: Path | str | None = None,
    ):
        self.final_dir = Path(final_dir) if final_dir else project_root / "outputs" / "models" / "final"
        self.extras_dir = Path(extras_dir) if extras_dir else project_root / "outputs" / "models" / "mvp_extras"

        self.pipelines: dict[str, Any] = {}
        self.tree_explainers: dict[str, shap.TreeExplainer] = {}

        # Load manifest for training medians and raw feature order
        manifest_path = self.extras_dir / "feature_manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            self.raw_features_95: list[str] = manifest.get("raw_features_95", [])
            self.train_medians_95: dict[str, float] = manifest.get("train_medians_95", {})
        else:
            self.raw_features_95 = []
            self.train_medians_95 = {}

    def get_pipeline(self, model_name: str, mode: str = "final"):
        """Loads and caches the requested pipeline."""
        key = f"{model_name}_{mode}"
        if key in self.pipelines:
            return self.pipelines[key]

        if mode == "final":
            if model_name == "M4":
                pipe_path = self.final_dir / "model_M4.joblib"
            elif model_name == "M1":
                pipe_path = self.extras_dir / "model_M1.joblib"
            elif model_name == "M3":
                pipe_path = self.final_dir / "model_M3.joblib"
            elif model_name == "M5":
                pipe_path = self.final_dir / "model_M5.joblib"
            else:
                raise ValueError(f"Unsupported model for explanation: '{model_name}'.")
        elif mode == "demo_oof":
            if model_name == "M4":
                pipe_path = self.extras_dir / "demo_r0_f0_M4.joblib"
            elif model_name == "M1":
                pipe_path = self.extras_dir / "demo_r0_f0_M1.joblib"
            else:
                raise ValueError(f"demo_oof mode only supports 'M4' and 'M1', got '{model_name}'.")
        else:
            raise ValueError(f"Invalid mode '{mode}'. Supported: 'final', 'demo_oof'.")

        if not pipe_path.exists():
            raise FileNotFoundError(f"Pipeline file not found: {pipe_path}")

        pipe = joblib.load(pipe_path)
        clf = pipe.named_steps.get("classifier")
        if clf is not None and hasattr(clf, "n_jobs"):
            clf.set_params(n_jobs=1)

        self.pipelines[key] = pipe
        return pipe

    def get_tree_explainer(self, model_name: str, mode: str = "final") -> shap.TreeExplainer:
        """Retrieves or builds cached shap.TreeExplainer."""
        key = f"{model_name}_{mode}"
        if key in self.tree_explainers:
            return self.tree_explainers[key]

        pipe = self.get_pipeline(model_name, mode=mode)
        clf = get_underlying_classifier(pipe)
        explainer = shap.TreeExplainer(clf, model_output="raw")
        self.tree_explainers[key] = explainer
        return explainer

    def explain_case(
        self,
        X_case: pd.DataFrame | pd.Series | dict[str, Any],
        model_name: str = "M4",
        mode: str = "final",
        top_k: int = 5,
    ) -> dict[str, Any]:
        """
        Explains a single company record using either TreeSHAP (M4) or Linear Contributions (M1).

        Parameters:
        - X_case: Single row feature record (DataFrame with 1 row, Series, or dict).
        - model_name: 'M4' (default, TreeSHAP) or 'M1' (Regularized Logistic Regression).
        - mode: 'final' (production frozen) or 'demo_oof' (demo fold r0_f0).
        - top_k: Number of positive and negative drivers to return (default: 5).

        Returns structured dictionary containing base value, attributions, top drivers, and disclaimers.
        """
        if isinstance(X_case, dict):
            df_row = pd.DataFrame([X_case])
        elif isinstance(X_case, pd.Series):
            df_row = pd.DataFrame([X_case.to_dict()])
        elif isinstance(X_case, pd.DataFrame):
            if len(X_case) != 1:
                raise ValueError(f"explain_case expects exactly 1 row, got {len(X_case)}.")
            df_row = X_case.copy()
        else:
            raise TypeError("X_case must be a DataFrame, Series, or dict.")

        # Ensure all 95 raw features are present
        X_95 = pd.DataFrame(index=df_row.index)
        for col in self.raw_features_95:
            if col in df_row.columns:
                X_95[col] = pd.to_numeric(df_row[col], errors="coerce")
            else:
                X_95[col] = np.nan

        # Missingness safeguard (20% maximum)
        missing_count = int(X_95.isna().sum(axis=1).iloc[0])
        missing_ratio = float(missing_count / 95.0)
        if missing_ratio > 0.20:
            raise ValueError(
                f"Sample rejected: missing feature ratio {missing_ratio:.1%} ({missing_count}/95) "
                f"exceeds allowed limit (20%)."
            )

        # Impute missing features with train_medians_95
        if missing_count > 0:
            for col in self.raw_features_95:
                if pd.isna(X_95.loc[X_95.index[0], col]):
                    X_95.loc[X_95.index[0], col] = self.train_medians_95.get(col, 0.0)

        # Retrieve pipeline and transform through preprocessors (steps before classifier)
        pipe = self.get_pipeline(model_name, mode=mode)
        clf = get_underlying_classifier(pipe)
        X_prep, feat_names_72 = transform_through_preprocessing(pipe, X_95)

        # Compute attributions based on model type
        if model_name == "M4":
            explainer = self.get_tree_explainer(model_name, mode=mode)
            X_prep_df = pd.DataFrame(X_prep, columns=feat_names_72)
            shap_values = explainer.shap_values(X_prep_df)

            if isinstance(shap_values, list):
                shap_arr = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                shap_arr = shap_values[:, :, 1]
            else:
                shap_arr = np.asarray(shap_values)

            attributions = shap_arr[0]
            base_val = float(explainer.expected_value) if np.isscalar(explainer.expected_value) else float(explainer.expected_value[0])
            pred_margin = float(clf.predict(X_prep, output_margin=True)[0])
            model_desc = f"M4 (XGBoost TreeSHAP, {mode})"

        elif model_name == "M1":
            coefs = clf.coef_[0]  # shape: (72,)
            intercept = float(clf.intercept_[0])
            attributions = coefs * X_prep[0]
            base_val = intercept
            pred_margin = float(clf.decision_function(X_prep)[0])
            model_desc = f"M1 (正規化回歸係數, {mode})"

        else:
            raise ValueError(f"Local explanation not supported for model '{model_name}'.")

        # Map attributions to feature names and raw/scaled values
        driver_items = []
        for i, fname in enumerate(feat_names_72):
            raw_val = float(X_95[fname].iloc[0]) if fname in X_95.columns else None
            z_val = float(X_prep[0, i])
            attr_val = float(attributions[i])

            driver_items.append({
                "feature": fname,
                "raw_value": raw_val,
                "transformed_value": z_val,
                "attribution": attr_val,
            })

        # Sort drivers: positive attributions increase bankruptcy risk; negative attributions decrease it
        positive_candidates = [d for d in driver_items if d["attribution"] > 0]
        negative_candidates = [d for d in driver_items if d["attribution"] < 0]

        # Top K positive (highest risk drivers)
        positive_sorted = sorted(positive_candidates, key=lambda x: x["attribution"], reverse=True)
        top_positive = positive_sorted[:top_k]

        # Top K negative (strongest protective/reducing drivers)
        negative_sorted = sorted(negative_candidates, key=lambda x: x["attribution"])
        top_negative = negative_sorted[:top_k]

        all_dict = {d["feature"]: d["attribution"] for d in driver_items}

        disclaimer_text = DEFAULT_DISCLAIMER.format(model_desc=model_desc)

        return {
            "model_name": model_name,
            "mode": mode,
            "base_value": base_val,
            "prediction_margin": pred_margin,
            "top_positive_drivers": top_positive,
            "top_negative_drivers": top_negative,
            "all_attributions": all_dict,
            "disclaimer": disclaimer_text,
        }


# Global shared explainer instance
_SHARED_EXPLAINER: Optional[LocalExplainer] = None


def get_local_explainer() -> LocalExplainer:
    """Returns singleton LocalExplainer instance."""
    global _SHARED_EXPLAINER
    if _SHARED_EXPLAINER is None:
        _SHARED_EXPLAINER = LocalExplainer()
    return _SHARED_EXPLAINER


def explain_case(
    X_case: pd.DataFrame | pd.Series | dict[str, Any],
    model_name: str = "M4",
    mode: str = "final",
    top_k: int = 5,
    explainer: Optional[LocalExplainer] = None,
) -> dict[str, Any]:
    """
    Convenience function to explain a single case.
    """
    if explainer is None:
        explainer = get_local_explainer()
    return explainer.explain_case(
        X_case=X_case,
        model_name=model_name,
        mode=mode,
        top_k=top_k,
    )

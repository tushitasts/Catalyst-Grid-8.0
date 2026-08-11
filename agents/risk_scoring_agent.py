"""
Risk Scoring Agent — LightGBM fraud probability + SHAP feature explanations.
Falls back to model feature importances if SHAP is unavailable.
"""
import os
import numpy as np
import pandas as pd
import joblib

import settings


class RiskScoringAgent:
    """Tool: Runs the trained LightGBM model and returns fraud probability + SHAP."""

    def __init__(self, model_path: str = None):
        path = model_path or os.path.join(settings.OUTPUT_DIR, 'lgbm_model.pkl')
        print(f"[RiskScoringAgent] Loading model from {path} ...")
        self.model = joblib.load(path)

        # Try to initialise SHAP
        try:
            import shap
            self.explainer = shap.TreeExplainer(self.model)
            self.shap_available = True
            print("[RiskScoringAgent] SHAP explainer ready.")
        except Exception as e:
            print(f"[RiskScoringAgent] SHAP unavailable ({e}). "
                  "Using global feature importances as fallback.")
            self.shap_available = False

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, case_data: dict, top_k: int = None) -> dict:
        """
        Parameters
        ----------
        case_data : dict from DataAgent.get_case()
        top_k     : number of top features to return

        Returns
        -------
        dict with fraud_probability, prediction, top_shap_features, base_value
        """
        top_k = top_k or settings.SHAP_TOP_K
        X = case_data['feature_vector']
        feature_names = case_data['feature_names']

        # ── Prediction ────────────────────────────────────────────────────────
        fraud_prob = float(self.model.predict_proba(X)[0, 1])
        prediction = int(fraud_prob >= 0.5)

        # ── Explanations ──────────────────────────────────────────────────────
        if self.shap_available:
            top_features, base_value = self._shap_explain(X, feature_names, top_k)
        else:
            top_features = self._importance_fallback(X, feature_names, top_k)
            base_value = None

        return {
            'fraud_probability': fraud_prob,
            'prediction': prediction,
            'top_shap_features': top_features,
            'base_value': base_value,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _shap_explain(self, X: pd.DataFrame, feature_names: list,
                      top_k: int) -> tuple:
        import shap  # noqa: already imported if we reach here

        shap_values = self.explainer.shap_values(X)

        # Binary classification -> list of two arrays [class0, class1]
        if isinstance(shap_values, list):
            sv = shap_values[1][0]          # class-1 (fraud), first sample
            base = float(self.explainer.expected_value[1])
        else:
            sv = shap_values[0]
            base = float(self.explainer.expected_value)

        abs_sv = np.abs(sv)
        top_idx = np.argsort(abs_sv)[::-1][:top_k]

        features = []
        for idx in top_idx:
            features.append({
                'feature': feature_names[idx],
                'value': float(X.iloc[0, idx]),
                'shap_value': float(sv[idx]),
                'direction': (
                    'increases fraud risk' if sv[idx] > 0
                    else 'decreases fraud risk'
                ),
            })
        return features, base

    def _importance_fallback(self, X: pd.DataFrame, feature_names: list,
                             top_k: int) -> list:
        """Use global feature importances when SHAP is unavailable."""
        importances = self.model.feature_importances_
        top_idx = np.argsort(importances)[::-1][:top_k]

        features = []
        for idx in top_idx:
            features.append({
                'feature': feature_names[idx],
                'value': float(X.iloc[0, idx]),
                'shap_value': None,
                'direction': f"importance={importances[idx]}",
            })
        return features

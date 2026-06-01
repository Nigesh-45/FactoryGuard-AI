"""
FactoryGuard AI - Week 3: Explainable AI (XAI) with SHAP
---------------------------------------------------------
Uses SHAP (SHapley Additive exPlanations) to explain
WHY the model flags a specific machine for failure.

Example output:
  "Machine #47 flagged: HIGH RISK
   → Temp_roll_mean_16 = 87.3°C  (+0.42 risk contribution)
   → Pressure_lag_1 = 0.31        (+0.38 risk contribution)
   → Vibration_ema_32 = 12.1      (+0.21 risk contribution)"

This builds trust with on-floor engineering teams.
"""

import shap
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


MODELS_DIR = Path("models")
PLOTS_DIR = Path("reports/shap_plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


class SHAPExplainer:
    """
    Wraps SHAP TreeExplainer for XGBoost model.
    
    Two explanation modes:
    1. Global  → Summary plot: which features matter most overall?
    2. Local   → Force plot: why was THIS specific machine flagged?
    """

    def __init__(self, model_path: str):
        self.pipeline = joblib.load(model_path)
        # Extract the XGBoost model from inside the ImbPipeline
        self.xgb_model = self.pipeline.named_steps["model"]
        self.explainer = shap.TreeExplainer(self.xgb_model)
        print("[SHAP] TreeExplainer initialized.")

    def _get_transformed_features(self, X: pd.DataFrame) -> np.ndarray:
        """
        SMOTE is only applied during training, not inference.
        For SHAP, we just need the raw feature matrix.
        """
        return X.values

    def compute_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values for a feature matrix."""
        X_array = self._get_transformed_features(X)
        shap_values = self.explainer.shap_values(X_array)
        print(f"[SHAP] Computed values for {len(X)} samples.")
        return shap_values

    def global_summary_plot(self, X: pd.DataFrame, save: bool = True):
        """
        Global Feature Importance: Beeswarm summary plot.
        Shows which features push predictions toward failure overall.
        """
        shap_values = self.compute_shap_values(X)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values, X,
            plot_type="dot",
            show=False,
            max_display=20
        )
        plt.title("FactoryGuard AI — Global Feature Impact on Failure Risk", fontsize=14)
        plt.tight_layout()
        if save:
            path = PLOTS_DIR / "global_summary.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"[SHAP] Summary plot saved → {path}")
        plt.show()

    def local_force_plot(self, X: pd.DataFrame, sample_idx: int, save: bool = True):
        """
        Local Explanation: Force plot for a single machine reading.
        Shows exactly which sensor values pushed it into the danger zone.
        """
        shap_values = self.compute_shap_values(X)
        expected_value = self.explainer.expected_value

        print(f"\n[SHAP] Explaining sample #{sample_idx}:")
        print(f"  Predicted failure probability: "
              f"{self.pipeline.predict_proba(X.iloc[[sample_idx]])[:, 1][0]:.2%}")

        force = shap.force_plot(
            expected_value,
            shap_values[sample_idx],
            X.iloc[sample_idx],
            matplotlib=True,
            show=False
        )
        if save:
            path = PLOTS_DIR / f"force_plot_sample_{sample_idx}.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"[SHAP] Force plot saved → {path}")
        plt.show()

    def explain_prediction(self, X_single: pd.DataFrame) -> dict:
        """
        Human-readable explanation for a single machine reading.
        Returns top 3 risk factors with their contribution scores.
        Used by the Flask API (Week 4) to build the JSON response.
        """
        shap_values = self.compute_shap_values(X_single)
        feature_names = X_single.columns.tolist()
        sv = shap_values[0]  # Single sample

        # Rank features by absolute SHAP value
        contributions = sorted(
            zip(feature_names, sv),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        top_factors = [
            {
                "feature": name,
                "value": float(X_single[name].iloc[0]),
                "shap_contribution": round(float(contrib), 4),
                "direction": "↑ Increases Risk" if contrib > 0 else "↓ Reduces Risk"
            }
            for name, contrib in contributions[:3]
        ]

        return {
            "failure_probability": float(
                self.pipeline.predict_proba(X_single)[:, 1][0]
            ),
            "top_risk_factors": top_factors
        }


def run_week3_analysis():
    """Run full SHAP analysis on the saved test set."""
    explainer = SHAPExplainer(str(MODELS_DIR / "factoryguard_xgb.joblib"))
    X_test = pd.read_parquet(MODELS_DIR / "X_test.parquet")

    # Use a sample for faster computation (full set can be slow)
    X_sample = X_test.sample(n=min(500, len(X_test)), random_state=42)

    # Global analysis
    explainer.global_summary_plot(X_sample)

    # Local analysis — explain first high-risk prediction
    explainer.local_force_plot(X_sample, sample_idx=0)

    # Human-readable explanation example
    explanation = explainer.explain_prediction(X_test.iloc[[0]])
    print("\n[SHAP] Human-readable explanation:")
    print(f"  Failure Probability: {explanation['failure_probability']:.2%}")
    for factor in explanation["top_risk_factors"]:
        print(f"  • {factor['feature']}: {factor['value']:.3f} "
              f"→ {factor['direction']} (SHAP: {factor['shap_contribution']})")


if __name__ == "__main__":
    run_week3_analysis()

"""
FactoryGuard AI - Modeling & Hyperparameter Tuning
----------------------------------------------------------
Trains baseline (Logistic Regression), Random Forest,
and high-performance XGBoost models with SMOTE imbalance
handling and RandomizedSearchCV optimization.

Evaluation: F1-Score & Recall (NOT accuracy — failures are rare!)
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, f1_score, recall_score,
    confusion_matrix, roc_auc_score
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb


# ── Constants ──────────────────────────────────────────────────────────────────
TARGET_COL = "failure"
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def load_features(path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load the processed feature dataset and split into X, y."""
    df = pd.read_parquet(path)
    drop_cols = [TARGET_COL, "timestamp", "machine_id"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols]
    y = df[TARGET_COL]
    print(f"[Data] Features: {X.shape} | Class distribution:\n{y.value_counts()}")
    return X, y


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """Print and return key metrics focused on F1 and Recall."""
    y_prob = model.predict_proba(X_test)[:, 1]
    thresh = getattr(model, "critical_threshold", 0.5)
    y_pred = (y_prob >= thresh).astype(int)

    metrics = {
        "model": model_name,
        "f1_score": f1_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "threshold": thresh
    }

    print(f"\n{'='*50}")
    print(f"  {model_name} Results (Threshold: {thresh:.4f})")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Failure"]))
    print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    return metrics


class BaselineTrainer:
    """
    Week 2 Step 1: Logistic Regression baseline.
    Establishes the minimum performance bar to beat.
    Uses SMOTE to handle class imbalance.
    """

    def train(self, X_train, y_train):
        pipeline = ImbPipeline([
            ("smote", SMOTE(random_state=42)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42))
        ])
        pipeline.fit(X_train, y_train)
        print("[Baseline] Logistic Regression trained.")
        return pipeline


class XGBoostTrainer:
    """
    Why XGBoost?
    - Gradient Boosting Decision Trees handle tabular data best
    - scale_pos_weight handles imbalance natively (alternative to SMOTE)
    - Built-in regularization prevents overfitting on noisy sensor data
    """

    PARAM_DIST = {
        "smote": [SMOTE(random_state=42), "passthrough"],
        "model__n_estimators": [100, 200, 300, 500],
        "model__max_depth": [3, 4, 5, 6, 7],
        "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "model__subsample": [0.6, 0.7, 0.8, 1.0],
        "model__colsample_bytree": [0.6, 0.7, 0.8, 1.0],
        "model__min_child_weight": [1, 3, 5],
        "model__gamma": [0, 0.1, 0.2, 0.5],
    }

    def train(self, X_train, y_train, n_iter: int = 30):
        # scale_pos_weight = negative_samples / positive_samples
        scale_pos = (y_train == 0).sum() / (y_train == 1).sum()

        pipeline = ImbPipeline([
            ("smote", SMOTE(random_state=42)),
            ("model", xgb.XGBClassifier(
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1
            ))
        ])

        param_dist = self.PARAM_DIST.copy()
        param_dist["model__scale_pos_weight"] = [1.0, scale_pos, scale_pos / 2]

        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring="f1",          # Optimise for F1, not accuracy!
            cv=5,
            verbose=1,
            random_state=42,
            n_jobs=-1
        )
        search.fit(X_train, y_train)
        print(f"[XGBoost] Best F1 (CV): {search.best_score_:.4f}")
        print(f"[XGBoost] Best params: {search.best_params_}")
        return search.best_estimator_


def find_optimal_threshold(model, X_val, y_val):
    """Finds the decision threshold that maximizes the validation F1 score."""
    y_prob = model.predict_proba(X_val)[:, 1]
    best_thresh = 0.5
    best_f1 = 0.0
    
    # Search from 0.05 to 0.95
    for thresh in np.linspace(0.05, 0.95, 91):
        y_pred = (y_prob >= thresh).astype(int)
        score = f1_score(y_val, y_pred)
        if score > best_f1:
            best_f1 = score
            best_thresh = thresh
            
    print(f"[Threshold Optimizer] Best validation F1: {best_f1:.4f} at threshold: {best_thresh:.4f}")
    return float(best_thresh)


def run_week2_pipeline(features_path: str) -> dict:
    """
    Full Week 2 pipeline: Load -> Split -> Train Baseline -> Train XGBoost -> Evaluate -> Save.
    """
    X, y = load_features(features_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Sub-split training data to find optimal decision threshold
    X_train_fit, X_val, y_train_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # 1. Baseline
    baseline_trainer = BaselineTrainer()
    baseline_model = baseline_trainer.train(X_train_fit, y_train_fit)
    
    base_thresh = find_optimal_threshold(baseline_model, X_val, y_val)
    baseline_model.critical_threshold = base_thresh
    baseline_model.warning_threshold = base_thresh * 0.6
    
    baseline_metrics = evaluate_model(baseline_model, X_test, y_test, "Logistic Regression (Baseline)")

    # 2. XGBoost
    xgb_trainer = XGBoostTrainer()
    xgb_model = xgb_trainer.train(X_train_fit, y_train_fit)
    
    xgb_thresh = find_optimal_threshold(xgb_model, X_val, y_val)
    xgb_model.critical_threshold = xgb_thresh
    xgb_model.warning_threshold = xgb_thresh * 0.6
    
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost (Tuned)")

    # 3. Save best model
    joblib.dump(xgb_model, MODELS_DIR / "factoryguard_xgb.joblib")
    print(f"\n[Pipeline] Best model saved -> {MODELS_DIR / 'factoryguard_xgb.joblib'}")

    # 4. Save test data for Week 3 SHAP analysis
    X_test.to_parquet(MODELS_DIR / "X_test.parquet", index=False)
    y_test.to_frame().to_parquet(MODELS_DIR / "y_test.parquet", index=False)

    return {"baseline": baseline_metrics, "xgboost": xgb_metrics}


if __name__ == "__main__":
    results = run_week2_pipeline("data/processed/modeling_dataset.parquet")


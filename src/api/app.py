"""
FactoryGuard AI - Flask Deployment API & Dashboard Service
-----------------------------------------------------------
Model-as-a-Service endpoint and interactive frontend dashboard.

Features:
- Dynamic live feature engineering using historical sensor data
- Gorgeous responsive glassmorphic operator dashboard
- Telemetry simulation engine for 50 robotic arms
- SHAP explanations visualizer API
"""

import os
import sys
import time
import random
import joblib
import collections
import pandas as pd
from pathlib import Path
from flask import Flask, request, jsonify, render_template

# Resolve project path to import SHAPExplainer
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.explainability.shap_explainer import SHAPExplainer

# ── App Setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")

MODEL_PATH = "models/factoryguard_xgb.joblib"
explainer = SHAPExplainer(MODEL_PATH)
print("[API] FactoryGuard AI model loaded and ready.")

# ── In-Memory Telemetry & Prediction Cache ─────────────────────────────────────
# Cache the last 35 readings per machine to enable real-time rolling calculations
MACHINE_HISTORY = collections.defaultdict(lambda: collections.deque(maxlen=35))
# Cache the latest predictions per machine
MACHINE_PREDICTIONS = {}

def init_history_cache():
    """Load historical sensor data to pre-populate cache and run initial batch predictions."""
    try:
        csv_path = Path("data/raw/sensor_readings.csv")
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            # Sort by timestamp to ensure chronological order
            df = df.sort_values("timestamp")
            
            # 1. Populate history cache
            for m_id, group in df.groupby("machine_id"):
                readings = group[["vibration", "temperature", "pressure"]].to_dict("records")
                MACHINE_HISTORY[m_id].extend(readings)
                
            # 2. Vectorized feature engineering for all machines at once (instant)
            from src.data.data_engineering import TemporalFeatureEngineer
            engineer = TemporalFeatureEngineer()
            df_sorted = df.sort_values(["machine_id", "timestamp"])
            df_features = engineer.engineer(df_sorted)
            
            # Get latest engineered feature row per machine
            df_latest = df_features.groupby("machine_id").last().reset_index()
            
            # 3. Batch prediction using XGBoost
            drop_cols = ["failure", "timestamp", "machine_id"]
            feat_cols = [c for c in df_latest.columns if c not in drop_cols]
            X_batch = df_latest[feat_cols]
            
            probs = explainer.pipeline.predict_proba(X_batch)[:, 1]
            for idx, row in df_latest.iterrows():
                m_id = row["machine_id"]
                MACHINE_PREDICTIONS[m_id] = float(probs[idx])
                
            print(f"[API] Cache and batch predictions populated for {len(MACHINE_HISTORY)} machines.")
        else:
            print("[API] No historical readings found. Starting with empty cache.")
    except Exception as e:
        print(f"[API] Error pre-populating history cache and predictions: {e}")

# Pre-populate cache on startup
init_history_cache()


# ── Feature Engineering helper ───────────────────────────────────────────────
def compute_features_from_history(machine_id: str, vibration: float, temperature: float, pressure: float, append_history: bool = True) -> pd.DataFrame:
    """
    Computes all required engineered features dynamically
    using the historical cache to prevent prediction mismatch.
    """
    if append_history:
        # 1. Add current reading to history
        MACHINE_HISTORY[machine_id].append({
            "vibration": vibration,
            "temperature": temperature,
            "pressure": pressure
        })
    
    # 2. Convert history to DataFrame
    history = list(MACHINE_HISTORY[machine_id])
    if not append_history:
        # If not appending, copy history and temporarily add the current reading for calculation
        history = history.copy()
        if not history:
            history.append({
                "vibration": vibration,
                "temperature": temperature,
                "pressure": pressure
            })
        else:
            history.append({
                "vibration": vibration,
                "temperature": temperature,
                "pressure": pressure
            })
        
    while len(history) < 33:
        history.insert(0, history[0])
        
    df_hist = pd.DataFrame(history)
    
    # 3. Calculate features using the exact same logic as TemporalFeatureEngineer
    SENSOR_COLS = ["vibration", "temperature", "pressure"]
    WINDOWS = [4, 16, 32]
    
    features = {}
    
    # Raw values
    features["vibration"] = vibration
    features["temperature"] = temperature
    features["pressure"] = pressure
    
    for col in SENSOR_COLS:
        for w in WINDOWS:
            # Rolling window excluding the current reading (equivalent to shift(1))
            series = df_hist[col].iloc[:-1]
            
            # Rolling Mean
            features[f"{col}_roll_mean_{w}"] = float(series.tail(w).mean())
            # Rolling Std
            features[f"{col}_roll_std_{w}"] = float(series.tail(w).std()) if len(series) >= 2 else 0.0
            if pd.isna(features[f"{col}_roll_std_{w}"]):
                features[f"{col}_roll_std_{w}"] = 0.0
                
            # Exponential Moving Average
            features[f"{col}_ema_{w}"] = float(series.ewm(span=w, adjust=False).mean().iloc[-1])
            
        # Lags
        features[f"{col}_lag_1"] = float(df_hist[col].iloc[-2])
        features[f"{col}_lag_2"] = float(df_hist[col].iloc[-3])
        
    # Add physics-based interaction features
    features["thermal_stress_index"] = temperature * vibration
    features["pressure_vibration_ratio"] = pressure / (vibration + 1e-5)
    
    # Standard column ordering expected by scikit-learn / XGBoost model
    cols_order = [
        'vibration', 'temperature', 'pressure',
        'vibration_roll_mean_4', 'vibration_roll_std_4', 'vibration_ema_4',
        'vibration_roll_mean_16', 'vibration_roll_std_16', 'vibration_ema_16',
        'vibration_roll_mean_32', 'vibration_roll_std_32', 'vibration_ema_32',
        'vibration_lag_1', 'vibration_lag_2',
        'temperature_roll_mean_4', 'temperature_roll_std_4', 'temperature_ema_4',
        'temperature_roll_mean_16', 'temperature_roll_std_16', 'temperature_ema_16',
        'temperature_roll_mean_32', 'temperature_roll_std_32', 'temperature_ema_32',
        'temperature_lag_1', 'temperature_lag_2',
        'pressure_roll_mean_4', 'pressure_roll_std_4', 'pressure_ema_4',
        'pressure_roll_mean_16', 'pressure_roll_std_16', 'pressure_ema_16',
        'pressure_roll_mean_32', 'pressure_roll_std_32', 'pressure_ema_32',
        'pressure_lag_1', 'pressure_lag_2',
        'thermal_stress_index', 'pressure_vibration_ratio'
    ]
    
    return pd.DataFrame([features])[cols_order]


def validate_input(data: dict) -> tuple[bool, str]:
    """Ensure required sensor fields are present in the request."""
    required_raw = ["vibration", "temperature", "pressure"]
    required_all = [
        'vibration', 'temperature', 'pressure',
        'vibration_roll_mean_4', 'vibration_roll_std_4', 'vibration_ema_4',
        'vibration_roll_mean_16', 'vibration_roll_std_16', 'vibration_ema_16',
        'vibration_roll_mean_32', 'vibration_roll_std_32', 'vibration_ema_32',
        'vibration_lag_1', 'vibration_lag_2',
        'temperature_roll_mean_4', 'temperature_roll_std_4', 'temperature_ema_4',
        'temperature_roll_mean_16', 'temperature_roll_std_16', 'temperature_ema_16',
        'temperature_roll_mean_32', 'temperature_roll_std_32', 'temperature_ema_32',
        'temperature_lag_1', 'temperature_lag_2',
        'pressure_roll_mean_4', 'pressure_roll_std_4', 'pressure_ema_4',
        'pressure_roll_mean_16', 'pressure_roll_std_16', 'pressure_ema_16',
        'pressure_roll_mean_32', 'pressure_roll_std_32', 'pressure_ema_32',
        'pressure_lag_1', 'pressure_lag_2',
        'thermal_stress_index', 'pressure_vibration_ratio'
    ]
    
    # Check if all features are passed directly
    if all(f in data for f in required_all):
        return True, "all"
        
    # Check if at least raw values are provided
    if all(f in data for f in required_raw):
        return True, "raw"
        
    missing = [f for f in required_raw if f not in data]
    return False, f"Missing fields. Require either raw sensor fields {required_raw} or all feature fields. Missing raw: {missing}"


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    """Render the FactoryGuard AI Operator Dashboard."""
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "model": "FactoryGuard AI v1.0"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """Main prediction endpoint (accepts raw sensors or pre-engineered features)."""
    start_time = time.time()

    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON body received"}), 400

    machine_id = data.pop("machine_id", "UNKNOWN")

    is_valid, mode = validate_input(data)
    if not is_valid:
        return jsonify({"error": mode}), 422

    # Build dynamic features if raw sensors are posted
    if mode == "raw":
        vibration = float(data["vibration"])
        temperature = float(data["temperature"])
        pressure = float(data["pressure"])
        X_input = compute_features_from_history(machine_id, vibration, temperature, pressure)
    else:
        # Pre-calculated features provided, map to exact schema order
        cols_order = [
            'vibration', 'temperature', 'pressure',
            'vibration_roll_mean_4', 'vibration_roll_std_4', 'vibration_ema_4',
            'vibration_roll_mean_16', 'vibration_roll_std_16', 'vibration_ema_16',
            'vibration_roll_mean_32', 'vibration_roll_std_32', 'vibration_ema_32',
            'vibration_lag_1', 'vibration_lag_2',
            'temperature_roll_mean_4', 'temperature_roll_std_4', 'temperature_ema_4',
            'temperature_roll_mean_16', 'temperature_roll_std_16', 'temperature_ema_16',
            'temperature_roll_mean_32', 'temperature_roll_std_32', 'temperature_ema_32',
            'temperature_lag_1', 'temperature_lag_2',
            'pressure_roll_mean_4', 'pressure_roll_std_4', 'pressure_ema_4',
            'pressure_roll_mean_16', 'pressure_roll_std_16', 'pressure_ema_16',
            'pressure_roll_mean_32', 'pressure_roll_std_32', 'pressure_ema_32',
            'pressure_lag_1', 'pressure_lag_2',
            'thermal_stress_index', 'pressure_vibration_ratio'
        ]
        X_input = pd.DataFrame([data])[cols_order]

    # Get predictions & SHAP risk factors
    explanation = explainer.explain_prediction(X_input)
    failure_prob = explanation["failure_probability"]
    MACHINE_PREDICTIONS[machine_id] = float(failure_prob)

    # Retrieve optimal thresholds from saved model pipeline
    crit_thresh = getattr(explainer.pipeline, "critical_threshold", 0.7)
    warn_thresh = getattr(explainer.pipeline, "warning_threshold", 0.4)

    if failure_prob >= crit_thresh:
        alert = "🔴 CRITICAL — Schedule maintenance immediately"
    elif failure_prob >= warn_thresh:
        alert = "🟡 WARNING — Monitor closely"
    else:
        alert = "🟢 NORMAL — No action required"

    latency_ms = round((time.time() - start_time) * 1000, 2)
    response = {
        "machine_id": machine_id,
        "failure_probability": round(failure_prob, 4),
        "failure_probability_pct": f"{failure_prob:.1%}",
        "alert_level": alert,
        "top_risk_factors": explanation["top_risk_factors"],
        "latency_ms": latency_ms
    }

    if latency_ms > 50:
        app.logger.warning(f"Latency exceeded 50ms: {latency_ms}ms for machine {machine_id}")

    return jsonify(response), 200


@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    """Batch prediction endpoint for multiple machines."""
    data = request.get_json(force=True)
    machines = data.get("machines", [])
    if not machines:
        return jsonify({"error": "No machines provided"}), 400

    results = []
    for machine_data in machines:
        machine_id = machine_data.pop("machine_id", "UNKNOWN")
        is_valid, mode = validate_input(machine_data)
        if not is_valid:
            continue
            
        if mode == "raw":
            X_input = compute_features_from_history(
                machine_id, 
                float(machine_data["vibration"]), 
                float(machine_data["temperature"]), 
                float(machine_data["pressure"])
            )
        else:
            cols_order = [
                'vibration', 'temperature', 'pressure',
                'vibration_roll_mean_4', 'vibration_roll_std_4', 'vibration_ema_4',
                'vibration_roll_mean_16', 'vibration_roll_std_16', 'vibration_ema_16',
                'vibration_roll_mean_32', 'vibration_roll_std_32', 'vibration_ema_32',
                'vibration_lag_1', 'vibration_lag_2',
                'temperature_roll_mean_4', 'temperature_roll_std_4', 'temperature_ema_4',
                'temperature_roll_mean_16', 'temperature_roll_std_16', 'temperature_ema_16',
                'temperature_roll_mean_32', 'temperature_roll_std_32', 'temperature_ema_32',
                'temperature_lag_1', 'temperature_lag_2',
                'pressure_roll_mean_4', 'pressure_roll_std_4', 'pressure_ema_4',
                'pressure_roll_mean_16', 'pressure_roll_std_16', 'pressure_ema_16',
                'pressure_roll_mean_32', 'pressure_roll_std_32', 'pressure_ema_32',
                'pressure_lag_1', 'pressure_lag_2',
                'thermal_stress_index', 'pressure_vibration_ratio'
            ]
            X_input = pd.DataFrame([machine_data])[cols_order]
            
        explanation = explainer.explain_prediction(X_input)
        results.append({
            "machine_id": machine_id,
            "failure_probability": round(explanation["failure_probability"], 4)
        })

    results.sort(key=lambda x: x["failure_probability"], reverse=True)
    return jsonify({"predictions": results, "count": len(results)}), 200


# ── Interactive Dashboard API Endpoints ───────────────────────────────────────
@app.route("/api/machines", methods=["GET"])
def get_machines():
    """List all cached machines, their current telemetry, and computed risk levels."""
    machines_summary = []
    
    crit_thresh = getattr(explainer.pipeline, "critical_threshold", 0.7)
    warn_thresh = getattr(explainer.pipeline, "warning_threshold", 0.4)
    
    for m_id, readings in MACHINE_HISTORY.items():
        if not readings:
            continue
        last = readings[-1]
        
        prob = MACHINE_PREDICTIONS.get(m_id, 0.0)
        status = "CRITICAL" if prob >= crit_thresh else "WARNING" if prob >= warn_thresh else "NORMAL"
        
        machines_summary.append({
            "machine_id": m_id,
            "vibration": round(last["vibration"], 2),
            "temperature": round(last["temperature"], 2),
            "pressure": round(last["pressure"], 4),
            "failure_probability": round(prob, 4),
            "status": status
        })
        
    machines_summary.sort(key=lambda x: x["machine_id"])
    return jsonify({"machines": machines_summary, "count": len(machines_summary)}), 200


@app.route("/api/machines/<machine_id>/history", methods=["GET"])
def get_machine_history(machine_id):
    """Get the recent historical logs for a machine."""
    readings = MACHINE_HISTORY.get(machine_id, [])
    history_list = []
    for i, r in enumerate(readings):
        history_list.append({
            "index": i,
            "vibration": round(r["vibration"], 2),
            "temperature": round(r["temperature"], 2),
            "pressure": round(r["pressure"], 4)
        })
    return jsonify({"machine_id": machine_id, "history": history_list}), 200


@app.route("/api/machines/<machine_id>/simulate", methods=["GET"])
def simulate_machine_reading(machine_id):
    """Generate a simulated sensor reading for demo purposes."""
    scenario = request.args.get("scenario", "normal") # 'normal' or 'fail'
    
    readings = MACHINE_HISTORY.get(machine_id, [])
    if readings:
        last = readings[-1]
        vib_base = last["vibration"]
        temp_base = last["temperature"]
        pres_base = last["pressure"]
    else:
        vib_base = 10.0
        temp_base = 65.0
        pres_base = 0.4
        
    if scenario == "fail":
        # Inject critical failing sensor profiles
        vibration = vib_base + random.uniform(4.0, 8.0)
        temperature = temp_base + random.uniform(15.0, 25.0)
        pressure = max(0.02, pres_base - random.uniform(0.12, 0.22))
    else:
        # Standard slight operation drift/noise
        if vib_base > 15: vib_base = 10.0
        if temp_base > 80: temp_base = 68.0
        if pres_base < 0.25: pres_base = 0.4
        
        vibration = max(1.0, vib_base + random.uniform(-1.0, 1.0))
        temperature = max(20.0, temp_base + random.uniform(-2.0, 2.0))
        pressure = max(0.05, pres_base + random.uniform(-0.03, 0.03))
        
    simulated_reading = {
        "machine_id": machine_id,
        "vibration": round(vibration, 2),
        "temperature": round(temperature, 2),
        "pressure": round(pressure, 4)
    }
    
    return jsonify(simulated_reading), 200


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

# FactoryGuard AI 🏭
**IoT Predictive Maintenance Engine** | Infotact Solutions — Q4 2025

> Predicts equipment failure 24 hours in advance using streaming sensor data (vibration, temperature, pressure) from 500 robotic arms. Powered by XGBoost + SHAP Explainability.

---

## Project Structure

```
factoryguard/
├── data/
│   ├── raw/                    # Raw sensor CSV logs (place your data here)
│   └── processed/              # Engineered features (auto-generated)
├── models/                     # Saved model artifacts
├── reports/shap_plots/         # SHAP visualizations
├── src/
│   ├── data/
│   │   └── data_engineering.py # Week 1: Ingest, Clean, Feature Engineering
│   ├── models/
│   │   └── train.py            # Week 2: Baseline + XGBoost Training
│   ├── explainability/
│   │   └── shap_explainer.py   # Week 3: SHAP Global + Local Explanations
│   └── api/
│       └── app.py              # Week 4: Flask REST API
├── generate_data.py            # Synthetic sensor data generator
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/Nigesh-45/Factory-Gaurd.git
cd Factory-Gaurd

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate synthetic sensor data (or add your own CSVs to data/raw/)
python generate_data.py
```

---

## Running the Pipeline

### Week 1 — Data Engineering
```bash
python src/data/data_engineering.py
# Output: data/processed/modeling_dataset.parquet
```

### Week 2 — Model Training
```bash
python src/models/train.py
# Output: models/factoryguard_xgb.joblib
```

### Week 3 — SHAP Explainability
```bash
python src/explainability/shap_explainer.py
# Output: reports/shap_plots/global_summary.png
```

### Week 4 — Run the API
```bash
python src/api/app.py
# API running at http://localhost:5000
```

---

## API Usage

### Health Check
```bash
curl http://localhost:5000/health
```

### Single Prediction
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "machine_id": "ARM-047",
    "vibration": 14.2,
    "temperature": 88.5,
    "pressure": 0.29,
    "vibration_roll_mean_4": 13.8,
    "temperature_roll_mean_4": 86.1,
    "pressure_roll_mean_4": 0.31
  }'
```

### Example Response
```json
{
  "machine_id": "ARM-047",
  "failure_probability": 0.8341,
  "failure_probability_pct": "83.4%",
  "alert_level": "🔴 CRITICAL — Schedule maintenance immediately",
  "top_risk_factors": [
    {"feature": "temperature_roll_mean_16", "value": 87.3, "shap_contribution": 0.42, "direction": "↑ Increases Risk"},
    {"feature": "pressure_lag_1", "value": 0.31, "shap_contribution": 0.38, "direction": "↑ Increases Risk"},
    {"feature": "vibration_ema_32", "value": 12.1, "shap_contribution": 0.21, "direction": "↑ Increases Risk"}
  ],
  "latency_ms": 23.4
}
```

---

## Key Concepts

| Concept | Why It Matters |
|---|---|
| **Rolling Window Features** | Captures temporal drift (e.g., gradual temp increase over 4hrs) |
| **SMOTE** | Failures are <1% of data — SMOTE creates synthetic failure samples |
| **F1 + Recall over Accuracy** | A model predicting "no failure" always gets 99% accuracy but is useless |
| **SHAP** | Explains WHY a machine is flagged — builds trust with engineers |
| **joblib serialization** | Saves the trained model for reuse without retraining |

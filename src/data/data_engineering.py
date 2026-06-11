"""
FactoryGuard AI - Week 1: Data Engineering
------------------------------------------
Handles all raw sensor data ingestion, cleaning,
and temporal feature engineering.
"""

import pandas as pd
import numpy as np
from pathlib import Path


class SensorDataLoader:
    """Loads and validates raw IoT sensor CSV logs."""

    def __init__(self, raw_data_path: str):
        self.raw_data_path = Path(raw_data_path)

    def load(self) -> pd.DataFrame:
        """Load all CSV files from the raw data directory."""
        files = list(self.raw_data_path.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV files found in {self.raw_data_path}")
        
        dfs = []
        for f in files:
            df = pd.read_csv(f, parse_dates=["timestamp"])
            dfs.append(df)
        
        combined = pd.concat(dfs, ignore_index=True)
        combined.sort_values("timestamp", inplace=True)
        combined.reset_index(drop=True, inplace=True)
        print(f"[DataLoader] Loaded {len(combined)} rows from {len(files)} file(s).")
        return combined


class SensorDataCleaner:
    """
    Cleans raw sensor data:
    - Handles missing values via interpolation
    - Removes duplicate timestamps per machine
    - Detects and caps outliers using IQR
    """

    SENSOR_COLS = ["vibration", "temperature", "pressure"]

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 1. Drop full duplicate rows
        df.drop_duplicates(inplace=True)

        # 2. Interpolate missing sensor values (linear, time-aware)
        df[self.SENSOR_COLS] = df[self.SENSOR_COLS].interpolate(method="linear", limit_direction="both")

        # 3. Cap outliers using IQR per sensor column
        for col in self.SENSOR_COLS:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower, upper)

        print(f"[Cleaner] Cleaned dataset shape: {df.shape}")
        return df


class TemporalFeatureEngineer:
    """
    WEEK 1 CORE TASK: Rolling Window Feature Engineering.
    
    Creates lag features and rolling statistics to capture
    temporal drift in sensor readings — critical for predicting
    failures 24 hours in advance.

    ⚠️  DATA LEAKAGE GUARD: All rolling/lag operations use
    .shift(1) to ensure we never use the current timestep's
    label to create a feature.
    """

    SENSOR_COLS = ["vibration", "temperature", "pressure"]
    WINDOWS = [4, 16, 32]  # Approx 1hr, 4hr, 8hr at 15-min intervals

    def engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Sort by machine_id and timestamp to ensure correct chronological order per machine
        df.sort_values(["machine_id", "timestamp"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        grouped = df.groupby("machine_id")

        for col in self.SENSOR_COLS:
            # Pre-compute lag 1 and lag 2 grouped to avoid bleeding between machines
            df[f"{col}_lag_1"] = grouped[col].shift(1)
            df[f"{col}_lag_2"] = grouped[col].shift(2)

            # Group on the lag_1 column for rolling computations
            gp_lag = df.groupby("machine_id")[f"{col}_lag_1"]

            for w in self.WINDOWS:
                # Rolling Mean
                df[f"{col}_roll_mean_{w}"] = (
                    gp_lag.rolling(window=w, min_periods=1).mean().reset_index(level=0, drop=True)
                )
                # Rolling Std (volatility indicator)
                df[f"{col}_roll_std_{w}"] = (
                    gp_lag.rolling(window=w, min_periods=1).std().reset_index(level=0, drop=True)
                )
                # Exponential Moving Average
                df[f"{col}_ema_{w}"] = (
                    gp_lag.ewm(span=w, adjust=False).mean().reset_index(level=0, drop=True)
                )

        # Add physics-based interaction features
        df["thermal_stress_index"] = df["temperature"] * df["vibration"]
        df["pressure_vibration_ratio"] = df["pressure"] / (df["vibration"] + 1e-5)

        # Drop rows with NaN introduced by lagging
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

        print(f"[FeatureEngineer] Feature matrix shape: {df.shape}")
        return df


def run_week1_pipeline(raw_path: str, output_path: str) -> pd.DataFrame:
    """
    Full Week 1 pipeline: Load -> Clean -> Feature Engineer -> Save.
    Run this script directly to produce the modeling dataset.
    """
    loader = SensorDataLoader(raw_path)
    cleaner = SensorDataCleaner()
    engineer = TemporalFeatureEngineer()

    df_raw = loader.load()
    df_clean = cleaner.clean(df_raw)
    df_features = engineer.engineer(df_clean)

    output_file = Path(output_path) / "modeling_dataset.parquet"
    df_features.to_parquet(output_file, index=False)
    print(f"[Pipeline] Saved modeling dataset -> {output_file}")
    return df_features


if __name__ == "__main__":
    run_week1_pipeline(
        raw_path="data/raw",
        output_path="data/processed"
    )


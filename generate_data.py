"""
Generate synthetic IoT sensor data for FactoryGuard AI.
Creates realistic vibration, temperature, and pressure readings
with injected failure patterns (~5% failure rate).
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

N_MACHINES = 50
READINGS_PER_MACHINE = 200  # ~50 hours of data at 15-min intervals
OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []
for m in range(1, N_MACHINES + 1):
    machine_id = f"ARM-{m:03d}"
    base_time = pd.Timestamp("2025-01-01")
    
    # Normal operating ranges
    if m in [7, 17, 27]:
        vib_base = 10.0
        temp_base = 65.0
        pres_base = 0.4
    else:
        vib_base = np.random.uniform(8, 12)
        temp_base = np.random.uniform(60, 75)
        pres_base = np.random.uniform(0.3, 0.5)

    for t in range(READINGS_PER_MACHINE):
        timestamp = base_time + pd.Timedelta(minutes=15 * t)
        
        # Determine if this is a failure window (last ~15% of readings for some machines)
        is_failing = (m % 10 == 0) and (t > READINGS_PER_MACHINE * 0.85)
        # Determine if this is a warning window (last ~15% of readings for some other machines)
        is_warning = (m in [7, 17, 27]) and (t > READINGS_PER_MACHINE * 0.85)
        # Also inject random failures for ~3% of other readings
        random_failure = np.random.random() < 0.03

        if is_failing:
            vibration = vib_base + np.random.normal(8, 2)  # High vibration
            temperature = temp_base + np.random.normal(20, 5)  # Overheating
            pressure = pres_base - np.random.normal(0.15, 0.05)  # Pressure drop
            failure = 1
        elif is_warning:
            vibration = 12.0 + np.random.normal(0, 0.1)  # Moderate vibration
            temperature = 78.0 + np.random.normal(0, 0.2)  # Moderate overheating
            pressure = 0.29 + np.random.normal(0, 0.005)  # Slight pressure drop
            failure = 0
        elif random_failure:
            vibration = vib_base + np.random.normal(5, 2)
            temperature = temp_base + np.random.normal(12, 4)
            pressure = pres_base - np.random.normal(0.1, 0.04)
            failure = 1
        else:
            vibration = vib_base + np.random.normal(0, 1.5)
            temperature = temp_base + np.random.normal(0, 3)
            pressure = pres_base + np.random.normal(0, 0.05)
            failure = 0

        rows.append({
            "timestamp": timestamp,
            "machine_id": machine_id,
            "vibration": round(max(0, vibration), 2),
            "temperature": round(max(0, temperature), 2),
            "pressure": round(max(0, pressure), 4),
            "failure": failure
        })

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_DIR / "sensor_readings.csv", index=False)
print(f"Generated {len(df)} sensor readings for {N_MACHINES} machines.")
print(f"Failure rate: {df['failure'].mean():.2%}")
print(f"Saved to {OUTPUT_DIR / 'sensor_readings.csv'}")

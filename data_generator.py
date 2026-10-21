import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


def generate_sample_data() -> pd.DataFrame:
    np.random.seed(42)
    n_normal = 450
    n_anomaly = 50
    total = n_normal + n_anomaly

    start_time = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [start_time + timedelta(hours=i) for i in range(total)]

    equipment_ids = [f'EQ{str(i).zfill(3)}' for i in range(1, 6)]

    temperature_normal = np.random.uniform(60, 90, n_normal)
    vibration_normal = np.random.uniform(0.5, 5.0, n_normal)
    rotational_speed_normal = np.random.uniform(1400, 1600, n_normal)
    pressure_normal = np.random.uniform(2.0, 4.0, n_normal)

    temperature_anomaly = np.random.uniform(100, 120, n_anomaly)
    vibration_anomaly = np.random.uniform(10, 15, n_anomaly)
    rotational_speed_anomaly = np.random.uniform(1800, 2000, n_anomaly)
    pressure_anomaly = np.random.uniform(5.5, 7.0, n_anomaly)

    temperature = np.concatenate([temperature_normal, temperature_anomaly])
    vibration = np.concatenate([vibration_normal, vibration_anomaly])
    rotational_speed = np.concatenate([rotational_speed_normal, rotational_speed_anomaly])
    pressure = np.concatenate([pressure_normal, pressure_anomaly])
    labels = np.concatenate([np.zeros(n_normal, dtype=int), np.ones(n_anomaly, dtype=int)])

    indices = np.random.permutation(total)
    timestamps_arr = np.array(timestamps)[indices]
    equipment_arr = np.array([equipment_ids[i % 5] for i in range(total)])[indices]

    df = pd.DataFrame({
        'timestamp': timestamps_arr,
        'equipment_id': equipment_arr,
        'temperature': np.round(temperature[indices], 2),
        'vibration': np.round(vibration[indices], 3),
        'rotational_speed': np.round(rotational_speed[indices], 1),
        'pressure': np.round(pressure[indices], 3),
        'label': labels[indices]
    })

    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


if __name__ == '__main__':
    df = generate_sample_data()
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_sensor_data.csv')
    df.to_csv(output_path, index=False)
    print(f'Sample data saved to {output_path}')
    print(f'Shape: {df.shape}')
    print(df.head())

import pandas as pd
import numpy as np
from typing import Optional


REQUIRED_COLUMNS = ['timestamp', 'equipment_id', 'temperature', 'vibration', 'rotational_speed', 'pressure']


def load_data(filepath: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f'Data file not found: {filepath}')
    except Exception as e:
        raise ValueError(f'Failed to read CSV file: {e}')
    return preprocess_data(df)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')

    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    except Exception:
        raise ValueError('Could not parse timestamp column as datetime.')

    numeric_cols = ['temperature', 'vibration', 'rotational_speed', 'pressure']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    before = len(df)
    df = df.dropna(subset=numeric_cols)
    after = len(df)
    if before != after:
        print(f'Dropped {before - after} rows with missing numeric values.')

    df['equipment_id'] = df['equipment_id'].astype(str).str.strip()

    df = df.sort_values('timestamp').reset_index(drop=True)

    return df

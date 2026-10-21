import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from typing import Optional, Tuple
import joblib


FEATURES = ['temperature', 'vibration', 'rotational_speed', 'pressure']
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'isolation_forest_model.pkl')


def get_risk_level(row: pd.Series) -> str:
    if row['temperature'] > 100 or row['vibration'] > 10 or row['pressure'] > 6:
        return 'High'
    elif row['temperature'] > 85 or row['vibration'] > 7 or row['pressure'] > 5:
        return 'Medium'
    else:
        return 'Low'


def train_model(df: pd.DataFrame) -> IsolationForest:
    X = df[FEATURES].values
    model = IsolationForest(contamination=0.1, random_state=42, n_estimators=100)
    model.fit(X)
    joblib.dump(model, MODEL_PATH)
    return model


def load_model() -> Optional[IsolationForest]:
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


def predict_anomalies(model: IsolationForest, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    X = df[FEATURES].values
    raw_predictions = model.predict(X)
    df['anomaly'] = np.where(raw_predictions == -1, 1, 0)
    df['anomaly_score'] = model.decision_function(X)
    df['risk_level'] = df.apply(get_risk_level, axis=1)
    return df


def get_or_train_model(df: pd.DataFrame) -> Tuple[IsolationForest, pd.DataFrame]:
    model = load_model()
    if model is None:
        model = train_model(df)
    result_df = predict_anomalies(model, df)
    return model, result_df

# AI-Based Predictive Maintenance Dashboard

An educational Python dashboard that analyzes machine sensor data, detects anomalies using machine learning, and estimates equipment failure risk.

## Features

- **Overview**: KPI cards showing total records, normal readings, anomalies, and high-risk counts
- **Sensor Data**: Browse, filter, and upload sensor data
- **Anomaly Detection**: Interactive time-series charts with anomaly highlights
- **Alerts & Recommendations**: Threshold-based alerts with maintenance guidance
- **Model Info**: Explanation of Isolation Forest, its limitations, and score distributions
- **Download Report**: Export analyzed data as CSV

## Project Structure

```
AI-Based Predictive Maintenance Dashboard/
├── app.py                      # Main Streamlit application
├── data_generator.py           # Generates sample sensor dataset
├── preprocessing.py            # Data loading and cleaning
├── anomaly_model.py            # Isolation Forest model
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── sample_sensor_data.csv      # Generated sample data
└── isolation_forest_model.pkl  # Saved model (auto-created on first run)
```

## Technologies Used

| Library | Purpose |
|---|---|
| Python 3.9+ | Core language |
| Streamlit | Web dashboard |
| Pandas / NumPy | Data processing |
| Scikit-learn | Isolation Forest anomaly detection |
| Plotly | Interactive charts |
| Joblib | Model persistence |

## Setup Instructions (Windows)

### 1. Prerequisites

- Python 3.9 or higher installed
- pip (comes with Python)

### 2. Install Dependencies

Open a terminal (PowerShell or Command Prompt) in the project folder:

```powershell
cd "C:\Users\keert\OneDrive\Desktop\AI-Based Predictive Maintenance Dashboard"
pip install -r requirements.txt
```

### 3. Generate Sample Data

```powershell
python data_generator.py
```

### 4. Run the Dashboard

```powershell
streamlit run app.py
```

Open your browser at: http://localhost:8501

## How to Use

1. The dashboard loads with sample sensor data automatically.
2. Use the **sidebar** to navigate between pages.
3. On the **Sensor Data** page, upload your own CSV file if needed.
4. View anomaly detection results on the **Anomaly Detection** page.
5. Check rule-based alerts on the **Alerts & Recommendations** page.
6. Download the full analyzed report from the **Download Report** page.

## Uploading Your Own Data

Your CSV must include these columns:

| Column | Type | Example |
|---|---|---|
| timestamp | datetime | 2024-01-01 00:00:00 |
| equipment_id | string | EQ001 |
| temperature | float (°C) | 75.4 |
| vibration | float (mm/s) | 2.3 |
| rotational_speed | float (RPM) | 1520.0 |
| pressure | float (bar) | 3.1 |

## Limitations & Disclaimer

> **This is an educational prototype. It is NOT validated for real industrial or defense deployment.**

- Anomalies detected are statistical outliers, not confirmed equipment failures.
- Risk levels are rule-based estimates based on configurable thresholds.
- No labeled ground truth is available, so accuracy/recall cannot be computed honestly.
- The contamination parameter (0.1) is an assumption.

## Live Demo

Try the deployed dashboard:  
https://ai-predictive-maintenance-7nxvcak8lu8435kamsvvlb.streamlit.app/

## GitHub Repository

Source code:  
https://github.com/keerthanakanumuri/AI-Predictive-Maintenance

- Overview page
<img width="1918" height="988" alt="image" src="https://github.com/user-attachments/assets/87bf34aa-80c2-4e2f-9da1-3439b27b8d44" />

- Anomaly Detection charts
<img width="1919" height="950" alt="image" src="https://github.com/user-attachments/assets/07414eae-785c-4f4f-8765-c219a7790f47" />

- Alerts page
<img width="1919" height="1038" alt="image" src="https://github.com/user-attachments/assets/846ba016-86f8-4239-8e3a-e086975498e5" />





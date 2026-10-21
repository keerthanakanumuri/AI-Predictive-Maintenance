import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

from data_generator import generate_sample_data
from preprocessing import preprocess_data, load_data, REQUIRED_COLUMNS
from anomaly_model import get_or_train_model, train_model, predict_anomalies

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DATA_PATH = os.path.join(BASE_DIR, 'sample_sensor_data.csv')

BG_COLOR = '#1a1a2e'
ACCENT_COLOR = '#e94560'
SECONDARY_COLOR = '#0f3460'
TEXT_COLOR = '#eaeaea'
CARD_BG = '#16213e'

st.set_page_config(
    page_title='Predictive Maintenance Dashboard',
    page_icon='⚙️',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.markdown(f"""
    <style>
    .stApp {{
        background-color: {BG_COLOR};
        color: {TEXT_COLOR};
    }}
    section[data-testid="stSidebar"] {{
        background-color: {SECONDARY_COLOR};
    }}
    .metric-card {{
        background-color: {CARD_BG};
        border-left: 4px solid {ACCENT_COLOR};
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }}
    h1, h2, h3, h4 {{
        color: {TEXT_COLOR};
    }}
    .disclaimer {{
        background-color: {CARD_BG};
        border: 1px solid {ACCENT_COLOR};
        padding: 0.75rem 1rem;
        border-radius: 6px;
        font-size: 0.85rem;
        color: #aaa;
        margin-top: 2rem;
    }}
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_default_data():
    if os.path.exists(SAMPLE_DATA_PATH):
        return load_data(SAMPLE_DATA_PATH)
    df = generate_sample_data()
    df.to_csv(SAMPLE_DATA_PATH, index=False)
    return preprocess_data(df)


@st.cache_data
def run_model(df_hash, df):
    _, result_df = get_or_train_model(df)
    return result_df


def get_data_hash(df):
    return str(len(df)) + str(df.columns.tolist())


def render_disclaimer():
    st.markdown(
        '<div class="disclaimer">⚠️ This is an educational prototype. '
        'Not validated for real industrial or defense deployment.</div>',
        unsafe_allow_html=True
    )


def page_overview(df):
    st.title('⚙️ Predictive Maintenance Dashboard')
    st.markdown(
        'This dashboard analyzes machine sensor data, detects anomalies using '
        'Isolation Forest (a machine learning algorithm), and estimates equipment '
        'failure risk using configurable threshold rules. Use the sidebar to navigate.'
    )

    total = len(df)
    normal = int((df['anomaly'] == 0).sum())
    anomalies = int((df['anomaly'] == 1).sum())
    high_risk = int((df['risk_level'] == 'High').sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Total Records', f'{total:,}')
    c2.metric('Normal Readings', f'{normal:,}')
    c3.metric('Anomalies Detected', f'{anomalies:,}', delta=f'{anomalies/total*100:.1f}%', delta_color='inverse')
    c4.metric('High Risk Records', f'{high_risk:,}', delta=f'{high_risk/total*100:.1f}%', delta_color='inverse')

    st.markdown('---')
    col1, col2 = st.columns(2)

    with col1:
        risk_counts = df['risk_level'].value_counts().reset_index()
        risk_counts.columns = ['Risk Level', 'Count']
        color_map = {'High': ACCENT_COLOR, 'Medium': '#f5a623', 'Low': '#27ae60'}
        fig = px.pie(
            risk_counts, names='Risk Level', values='Count',
            title='Risk Level Distribution',
            color='Risk Level', color_discrete_map=color_map,
            hole=0.4
        )
        fig.update_layout(paper_bgcolor=CARD_BG, font_color=TEXT_COLOR, title_font_color=TEXT_COLOR)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        eq_anomaly = df.groupby('equipment_id')['anomaly'].sum().reset_index()
        eq_anomaly.columns = ['Equipment', 'Anomalies']
        fig2 = px.bar(
            eq_anomaly, x='Equipment', y='Anomalies',
            title='Anomalies by Equipment',
            color='Anomalies', color_continuous_scale=[[0, SECONDARY_COLOR], [1, ACCENT_COLOR]]
        )
        fig2.update_layout(paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG, font_color=TEXT_COLOR, title_font_color=TEXT_COLOR)
        st.plotly_chart(fig2, use_container_width=True)

    render_disclaimer()


def page_sensor_data(df, uploaded_file_container):
    st.title('📊 Sensor Data')

    st.subheader('Upload Your Own CSV')
    uploaded = st.file_uploader(
        'Upload a CSV with columns: timestamp, equipment_id, temperature, vibration, rotational_speed, pressure',
        type=['csv']
    )
    if uploaded is not None:
        try:
            user_df = pd.read_csv(uploaded)
            user_df = preprocess_data(user_df)
            _, user_df = get_or_train_model(user_df)
            st.success(f'Uploaded data loaded: {len(user_df)} records.')
            uploaded_file_container['df'] = user_df
        except Exception as e:
            st.error(f'Error loading uploaded file: {e}')
            uploaded_file_container['df'] = None

    display_df = uploaded_file_container.get('df') or df

    st.subheader('Filter Data')
    col1, col2 = st.columns(2)
    with col1:
        eq_options = ['All'] + sorted(display_df['equipment_id'].unique().tolist())
        eq_filter = st.selectbox('Equipment ID', eq_options)
    with col2:
        risk_options = ['All', 'High', 'Medium', 'Low']
        risk_filter = st.selectbox('Risk Level', risk_options)

    filtered = display_df.copy()
    if eq_filter != 'All':
        filtered = filtered[filtered['equipment_id'] == eq_filter]
    if risk_filter != 'All':
        filtered = filtered[filtered['risk_level'] == risk_filter]

    st.markdown(f'Showing **{len(filtered):,}** records')

    display_cols = ['timestamp', 'equipment_id', 'temperature', 'vibration', 'rotational_speed', 'pressure', 'anomaly', 'risk_level']
    available_cols = [c for c in display_cols if c in filtered.columns]
    st.dataframe(filtered[available_cols].reset_index(drop=True), use_container_width=True)

    render_disclaimer()


def page_anomaly_detection(df):
    st.title('🔍 Anomaly Detection')
    st.markdown(
        'Charts below show sensor readings over time. '
        '**Red points** are ML-detected anomalies (Isolation Forest). '
        'This does not guarantee they are actual failures.'
    )

    eq_options = ['All'] + sorted(df['equipment_id'].unique().tolist())
    eq_filter = st.selectbox('Filter by Equipment', eq_options)
    plot_df = df if eq_filter == 'All' else df[df['equipment_id'] == eq_filter]

    label_map = {0: 'Normal', 1: 'Anomaly'}
    plot_df = plot_df.copy()
    plot_df['status'] = plot_df['anomaly'].map(label_map)

    def make_time_chart(y_col, title, y_label):
        fig = px.scatter(
            plot_df, x='timestamp', y=y_col,
            color='status',
            color_discrete_map={'Normal': '#27ae60', 'Anomaly': ACCENT_COLOR},
            title=title,
            labels={y_col: y_label, 'timestamp': 'Time', 'status': 'Status'},
            opacity=0.8
        )
        fig.update_layout(paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG, font_color=TEXT_COLOR, title_font_color=TEXT_COLOR)
        return fig

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(make_time_chart('temperature', 'Temperature Over Time', 'Temperature (°C)'), use_container_width=True)
    with col2:
        st.plotly_chart(make_time_chart('vibration', 'Vibration Over Time', 'Vibration (mm/s)'), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(make_time_chart('pressure', 'Pressure Over Time', 'Pressure (bar)'), use_container_width=True)
    with col4:
        st.plotly_chart(make_time_chart('rotational_speed', 'Rotational Speed Over Time', 'Speed (RPM)'), use_container_width=True)

    st.subheader('Anomaly Count by Equipment')
    eq_anomaly = df.groupby('equipment_id')['anomaly'].sum().reset_index()
    eq_anomaly.columns = ['Equipment', 'Anomaly Count']
    fig_bar = px.bar(eq_anomaly, x='Equipment', y='Anomaly Count', color='Anomaly Count',
                     color_continuous_scale=[[0, SECONDARY_COLOR], [1, ACCENT_COLOR]])
    fig_bar.update_layout(paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG, font_color=TEXT_COLOR)
    st.plotly_chart(fig_bar, use_container_width=True)

    render_disclaimer()


def page_alerts(df):
    st.title('🚨 Alerts & Recommendations')
    st.info('These alerts are based on **configurable threshold rules**, not machine learning predictions. They complement the ML anomaly detection.')

    high_temp = df[df['temperature'] > 100]
    high_vib = df[df['vibration'] > 10]
    high_press = df[df['pressure'] > 6]
    high_speed = df[df['rotational_speed'] > 1800]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('High Temp Alerts', len(high_temp))
    col2.metric('High Vibration Alerts', len(high_vib))
    col3.metric('High Pressure Alerts', len(high_press))
    col4.metric('High Speed Alerts', len(high_speed))

    alerts = []
    if len(high_temp) > 0:
        alerts.append(('🌡️ High Temperature', f'{len(high_temp)} readings above 100°C detected.',
                        'Check cooling systems. Inspect heat exchangers. Reduce load if possible.'))
    if len(high_vib) > 0:
        alerts.append(('📳 High Vibration', f'{len(high_vib)} readings above 10 mm/s detected.',
                        'Inspect bearings and mounts. Check for mechanical imbalance. Schedule lubrication.'))
    if len(high_press) > 0:
        alerts.append(('💨 High Pressure', f'{len(high_press)} readings above 6 bar detected.',
                        'Check pressure relief valves. Inspect seals and pipes. Reduce system pressure.'))
    if len(high_speed) > 0:
        alerts.append(('⚙️ High Rotational Speed', f'{len(high_speed)} readings above 1800 RPM detected.',
                        'Check motor controllers. Verify load coupling. Inspect speed governors.'))

    if alerts:
        for icon_title, desc, rec in alerts:
            with st.expander(f'{icon_title} — {desc}'):
                st.markdown(f'**Recommendation:** {rec}')
    else:
        st.success('No threshold-based alerts found in the current dataset.')

    render_disclaimer()


def page_model_info(df):
    st.title('🤖 Model Information')

    st.subheader('About Isolation Forest')
    st.markdown("""
    **Isolation Forest** is an unsupervised anomaly detection algorithm that isolates observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of that feature.

    - Anomalies are points that require fewer splits to isolate, resulting in shorter path lengths.
    - This model was trained with **contamination=0.1**, meaning roughly 10% of training data is expected to be anomalous.
    - The model operates on four sensor features: temperature, vibration, rotational speed, and pressure.
    """)

    st.subheader('Limitations & Honest Assessment')
    st.warning("""
    **No labeled ground truth is available for this dataset.** Therefore:
    - Accuracy, precision, recall, and F1-score CANNOT be computed honestly.
    - Reported anomalies are statistical outliers, not confirmed equipment failures.
    - The contamination parameter (0.1) is an assumption, not a measured value.
    - Risk levels are rule-based estimates, not physics-based failure models.
    """)

    st.subheader('Anomaly Score Distribution')
    st.markdown('Lower (more negative) anomaly scores indicate more anomalous readings.')
    if 'anomaly_score' in df.columns:
        fig = px.histogram(
            df, x='anomaly_score',
            nbins=50, title='Distribution of Anomaly Scores',
            labels={'anomaly_score': 'Anomaly Score', 'count': 'Count'}
        )
        fig.update_layout(paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG, font_color=TEXT_COLOR, title_font_color=TEXT_COLOR)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('Anomaly scores not available for this dataset.')

    st.subheader('Feature Summary Statistics')
    features = ['temperature', 'vibration', 'rotational_speed', 'pressure']
    available = [f for f in features if f in df.columns]
    st.dataframe(df[available].describe().round(3), use_container_width=True)

    render_disclaimer()


def page_download(df):
    st.title('📥 Download Report')

    total = len(df)
    anomalies = int((df['anomaly'] == 1).sum())
    high_risk = int((df['risk_level'] == 'High').sum())
    medium_risk = int((df['risk_level'] == 'Medium').sum())
    low_risk = int((df['risk_level'] == 'Low').sum())

    st.subheader('Dataset Summary')
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Total Records', f'{total:,}')
    col2.metric('Total Anomalies', f'{anomalies:,}')
    col3.metric('High Risk', f'{high_risk:,}')
    col4.metric('Medium Risk', f'{medium_risk:,}')

    st.subheader('Top Equipment by Anomaly Count')
    eq_summary = df.groupby('equipment_id').agg(
        anomaly_count=('anomaly', 'sum'),
        high_risk_count=('risk_level', lambda x: (x == 'High').sum()),
        total_records=('anomaly', 'count')
    ).reset_index().sort_values('anomaly_count', ascending=False)
    st.dataframe(eq_summary, use_container_width=True)

    st.subheader('Download Full Analyzed Dataset')
    csv_buffer = df.to_csv(index=False)
    st.download_button(
        label='⬇️ Download CSV',
        data=csv_buffer,
        file_name='predictive_maintenance_report.csv',
        mime='text/csv'
    )

    render_disclaimer()


def main():
    st.sidebar.title('Navigation')
    pages = [
        'Overview',
        'Sensor Data',
        'Anomaly Detection',
        'Alerts & Recommendations',
        'Model Info',
        'Download Report'
    ]
    page = st.sidebar.radio('Go to', pages)

    st.sidebar.markdown('---')
    st.sidebar.markdown('**Data Source**')

    with st.spinner('Loading data and running model...'):
        base_df = load_default_data()
        df = run_model(get_data_hash(base_df), base_df)

    uploaded_container = {}

    if page == 'Overview':
        page_overview(df)
    elif page == 'Sensor Data':
        page_sensor_data(df, uploaded_container)
    elif page == 'Anomaly Detection':
        page_anomaly_detection(df)
    elif page == 'Alerts & Recommendations':
        page_alerts(df)
    elif page == 'Model Info':
        page_model_info(df)
    elif page == 'Download Report':
        page_download(df)


if __name__ == '__main__':
    main()

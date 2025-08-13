import streamlit as st
import pandas as pd
import numpy as np
from math import ceil
import os # Import the os module to check for file existence

# --- Backend functions remain the same ---
@st.cache_data
def get_historical_rates_and_summary(df):
    # ... (no changes to this function)
    df.columns = df.columns.str.strip()
    date_cols = ['Informed Consent Date (Local)', 'Screen Fail Date (Local)', 'Randomization Date (Local)']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    df.dropna(subset=['Informed Consent Date (Local)'], inplace=True)
    df['Referral Source'] = df['Referral Source'].str.strip()
    df_1nhealth = df[df['Referral Source'] == '1nHealth'].copy()
    df_site = df[df['Referral Source'] == 'Site'].copy()
    df_1nhealth['ICF_Month'] = df_1nhealth['Informed Consent Date (Local)'].dt.to_period('M')
    df_1nhealth['Rand_Month'] = df_1nhealth['Randomization Date (Local)'].dt.to_period('M')
    df_site['ICF_Month'] = df_site['Informed Consent Date (Local)'].dt.to_period('M')
    df_site['Rand_Month'] = df_site['Randomization Date (Local)'].dt.to_period('M')
    icf_1nhealth_counts = df_1nhealth.groupby('ICF_Month').size()
    icf_site_counts = df_site.groupby('ICF_Month').size()
    rate_1nhealth = df_1nhealth['Rand_Month'].notna().sum() / len(df_1nhealth) if not df_1nhealth.empty else 0
    lag_1nhealth = (df_1nhealth['Randomization Date (Local)'] - df_1nhealth['Informed Consent Date (Local)']).dt.days.mean()
    avg_monthly_icf_1nhealth = icf_1nhealth_counts.mean() if not icf_1nhealth_counts.empty else 0
    rate_site = df_site['Rand_Month'].notna().sum() / len(df_site) if not df_site.empty else 0
    lag_site = (df_site['Randomization Date (Local)'] - df_site['Informed Consent Date (Local)']).dt.days.mean()
    avg_monthly_icf_site = icf_site_counts.mean() if not icf_site_counts.empty else 0
    historical_summary = pd.concat([
        df_1nhealth.groupby('ICF_Month').size().rename('1nHealth ICF Total'),
        df_1nhealth.groupby('Rand_Month').size().rename('1nHealth Rand Total'),
        df_site.groupby('ICF_Month').size().rename('Site ICF Total'),
        df_site.groupby('Rand_Month').size().rename('Site Rand Total')
    ], axis=1).fillna(0).astype(int)
    historical_summary.sort_index(inplace=True)
    rates = {
        '1nHealth': {'rate': rate_1nhealth, 'lag': lag_1nhealth, 'avg_icf': avg_monthly_icf_1nhealth},
        'Site': {'rate': rate_site, 'lag': lag_site, 'avg_icf': avg_monthly_icf_site}
    }
    return historical_summary, rates

def generate_forecast(rates, forecast_horizon_months, edited_icf_forecast):
    # ... (no changes to this function)
    forecast_df = pd.DataFrame(index=edited_icf_forecast.index, columns=[
        '1nHealth ICF Total', '1nHealth Rand Total', 'Site ICF Total', 'Site Rand Total'
    ]).fillna(0.0)
    forecast_df['1nHealth ICF Total'] = edited_icf_forecast['1nHealth ICF Total']
    forecast_df['Site ICF Total'] = edited_icf_forecast['Site ICF Total']
    projected_rands_1nhealth = forecast_df['1nHealth ICF Total'] * rates['1nHealth']['rate']
    projected_rands_site = forecast_df['Site ICF Total'] * rates['Site']['rate']
    days_in_month = 30.44
    for month, rands in projected_rands_1nhealth.items():
        if rands > 0:
            lag_days = rates['1nHealth']['lag'] if pd.notna(rates['1nHealth']['lag']) else 0
            full_months_lag = int(lag_days // days_in_month)
            frac_next_month = (lag_days % days_in_month) / days_in_month
            land_month_1 = month + full_months_lag
            land_month_2 = land_month_1 + 1
            if land_month_1 in forecast_df.index:
                forecast_df.loc[land_month_1, '1nHealth Rand Total'] += rands * (1 - frac_next_month)
            if land_month_2 in forecast_df.index:
                forecast_df.loc[land_month_2, '1nHealth Rand Total'] += rands * frac_next_month
    for month, rands in projected_rands_site.items():
        if rands > 0:
            lag_days = rates['Site']['lag'] if pd.notna(rates['Site']['lag']) else 0
            full_months_lag = int(lag_days // days_in_month)
            frac_next_month = (lag_days % days_in_month) / days_in_month
            land_month_1 = month + full_months_lag
            land_month_2 = land_month_1 + 1
            if land_month_1 in forecast_df.index:
                forecast_df.loc[land_month_1, 'Site Rand Total'] += rands * (1 - frac_next_month)
            if land_month_2 in forecast_df.index:
                forecast_df.loc[land_month_2, 'Site Rand Total'] += rands * frac_next_month
    forecast_df = forecast_df.apply(np.ceil).astype(int)
    return forecast_df

# --- Streamlit Application UI ---

st.set_page_config(layout="wide")
st.title("Blended Enrollment Funnel: 1nHealth vs. Site-Sourced")

# --- Sidebar for Controls ---
with st.sidebar:
    st.header("⚙️ Controls")
    uploaded_file = st.file_uploader("Upload IRT Report (CSV)", type="csv")
    
# --- Main Application Logic ---
raw_df = None
data_loaded = False

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    data_loaded = True
else:
    # **IMPROVED LOGIC**: Check for the sample file, but don't error if it's missing.
    sample_file_path = 'input_file_2.csv'
    if os.path.exists(sample_file_path):
        st.sidebar.write("Using default sample IRT data.")
        raw_df = pd.read_csv(sample_file_path)
        data_loaded = True
    else:
        # If no upload and no sample, just show an info message.
        st.info("👋 Welcome! Please upload an IRT report using the sidebar to begin.")
        st.stop() # Stop the script here until there is data.

# --- App continues only if data is loaded ---
st.sidebar.markdown("---")
forecast_horizon = st.sidebar.slider("Forecast Horizon (Months)", min_value=3, max_value=24, value=6)

hist_summary, rates = get_historical_rates_and_summary(raw_df.copy())

st.sidebar.subheader("Performance Rates (Historical)")
st.sidebar.metric("1nHealth ICF-to-Rand Rate", f"{rates['1nHealth']['rate']:.1%}")
st.sidebar.metric("1nHealth Avg. Lag (Days)", f"{rates['1nHealth']['lag']:.1f}")
st.sidebar.metric("Site ICF-to-Rand Rate", f"{rates['Site']['rate']:.1%}")
st.sidebar.metric("Site Avg. Lag (Days)", f"{rates['Site']['lag']:.1f}")

st.header("Enrollment Forecast")
st.markdown("##### Forecast Assumptions")
st.caption("Default monthly ICFs are based on historical averages. Edit the numbers below to model different scenarios.")

last_hist_month = hist_summary.index.max()
future_months = pd.period_range(start=last_hist_month + 1, periods=forecast_horizon, freq='M')
    
editable_icf_df = pd.DataFrame(index=future_months)
editable_icf_df['1nHealth ICF Total'] = round(rates['1nHealth']['avg_icf'])
editable_icf_df['Site ICF Total'] = round(rates['Site']['avg_icf'])
editable_icf_df.index = editable_icf_df.index.strftime('%Y-%m')

edited_forecast_assumptions = st.data_editor(editable_icf_df)
edited_forecast_assumptions.index = pd.to_period(edited_forecast_assumptions.index, 'M')

forecast_summary = generate_forecast(rates, forecast_horizon, edited_forecast_assumptions)

master_table = pd.concat([hist_summary, forecast_summary])
master_table.index.name = 'Month'
master_table['Overall ICF Total'] = master_table['1nHealth ICF Total'] + master_table['Site ICF Total']
master_table['Overall Rand Total'] = master_table['1nHealth Rand Total'] + master_table['Site Rand Total']
master_table['Overall Running ICF Total'] = master_table['Overall ICF Total'].cumsum()
master_table['Overall Running Rand Total'] = master_table['Overall Rand Total'].cumsum()

st.markdown("##### Master Summary Table (Historical & Forecast)")
st.dataframe(master_table.style.format("{:,.0f}"))

st.markdown("---")
st.header("Visualizations")
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Monthly Performance by Source")
    chart_data_icf = master_table[['1nHealth ICF Total', 'Site ICF Total']]
    chart_data_icf.index = chart_data_icf.index.to_timestamp()
    st.bar_chart(chart_data_icf, y_label="ICFs per Month")

with col2:
    st.markdown("#### Cumulative Enrollment Over Time")
    chart_data_cumulative = master_table[['Overall Running ICF Total', 'Overall Running Rand Total']]
    chart_data_cumulative.index = chart_data_cumulative.index.to_timestamp()
    st.line_chart(chart_data_cumulative)
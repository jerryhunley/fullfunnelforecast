import streamlit as st
import pandas as pd
import numpy as np
from math import ceil
import os

# --- Data Processing and Forecasting Logic (Backend Functions) ---

@st.cache_data
def process_data_and_get_rates(df_1nhealth_raw, df_irt_raw):
    """
    Processes both raw dataframes, creates a blended historical summary,
    and calculates the distinct performance rates for each funnel.
    """
    # --- Process 1nHealth Funnel Data ---
    # In a real scenario, this would use the full multi-stage logic.
    # For this app, we extract the necessary ICF/Rand dates.
    df_1nhealth = df_1nhealth_raw.copy()
    # Assuming the 1nHealth file has columns 'ICF Date' and 'Randomization Date'
    # These column names might need to be adjusted to your actual file.
    df_1nhealth['ICF_Date'] = pd.to_datetime(df_1nhealth['ICF Date'], errors='coerce')
    df_1nhealth['Rand_Date'] = pd.to_datetime(df_1nhealth.get('Randomization Date (Local)'), errors='coerce') # Use .get for optional column
    df_1nhealth.dropna(subset=['ICF_Date'], inplace=True)
    df_1nhealth['ICF_Month'] = df_1nhealth['ICF_Date'].dt.to_period('M')
    df_1nhealth['Rand_Month'] = df_1nhealth['Rand_Date'].dt.to_period('M')

    # --- Process Site-Sourced Data from IRT Report ---
    df_site = df_irt_raw[df_irt_raw['Referral Source'].str.strip() == 'Site'].copy()
    df_site['ICF_Date'] = pd.to_datetime(df_site['Informed Consent Date (Local)'], errors='coerce')
    df_site['Rand_Date'] = pd.to_datetime(df_site['Randomization Date (Local)'], errors='coerce')
    df_site.dropna(subset=['ICF_Date'], inplace=True)
    df_site['ICF_Month'] = df_site['ICF_Date'].dt.to_period('M')
    df_site['Rand_Month'] = df_site['Rand_Date'].dt.to_period('M')

    # --- Calculate Historical Rates for Each Funnel ---
    rate_1nhealth = df_1nhealth['Rand_Month'].notna().sum() / len(df_1nhealth) if not df_1nhealth.empty else 0
    lag_1nhealth = (df_1nhealth['Rand_Date'] - df_1nhealth['ICF_Date']).dt.days.mean()
    avg_monthly_icf_1nhealth = df_1nhealth.groupby('ICF_Month').size().mean() if not df_1nhealth.empty else 0

    rate_site = df_site['Rand_Month'].notna().sum() / len(df_site) if not df_site.empty else 0
    lag_site = (df_site['Rand_Date'] - df_site['ICF_Date']).dt.days.mean()
    avg_monthly_icf_site = df_site.groupby('ICF_Month').size().mean() if not df_site.empty else 0

    rates = {
        '1nHealth': {'rate': rate_1nhealth, 'lag': lag_1nhealth, 'avg_icf': avg_monthly_icf_1nhealth},
        'Site': {'rate': rate_site, 'lag': lag_site, 'avg_icf': avg_monthly_icf_site}
    }
    
    # --- Create Blended Historical Summary ---
    hist_summary = pd.concat([
        df_1nhealth.groupby('ICF_Month').size().rename('1nHealth ICF Total'),
        df_1nhealth.groupby('Rand_Month').size().rename('1nHealth Rand Total'),
        df_site.groupby('ICF_Month').size().rename('Site ICF Total'),
        df_site.groupby('Rand_Month').size().rename('Site Rand Total')
    ], axis=1).fillna(0).astype(int)
    hist_summary.sort_index(inplace=True)

    return hist_summary, rates

def generate_forecast(rates, forecast_horizon_months, edited_icf_forecast):
    """Generates a forecast based on rates and user-edited ICF assumptions for BOTH funnels."""
    forecast_df = pd.DataFrame(index=edited_icf_forecast.index, columns=[
        '1nHealth ICF Total', '1nHealth Rand Total', 'Site ICF Total', 'Site Rand Total'
    ]).fillna(0.0)
    
    forecast_df['1nHealth ICF Total'] = edited_icf_forecast['1nHealth ICF Total']
    forecast_df['Site ICF Total'] = edited_icf_forecast['Site ICF Total']
                    
    projected_rands_1nhealth = forecast_df['1nHealth ICF Total'] * rates['1nHealth']['rate']
    projected_rands_site = forecast_df['Site ICF Total'] * rates['Site']['rate']
    
    days_in_month = 30.44
    
    # Smear Rands for each funnel using its unique lag
    for month, rands in projected_rands_1nhealth.items():
        if rands > 0:
            lag_days = rates['1nHealth']['lag'] if pd.notna(rates['1nHealth']['lag']) else 0
            full_months_lag = int(lag_days // days_in_month)
            frac_next_month = (lag_days % days_in_month) / days_in_month
            land_month_1 = month + full_months_lag
            land_month_2 = land_month_1 + 1
            if land_month_1 in forecast_df.index: forecast_df.loc[land_month_1, '1nHealth Rand Total'] += rands * (1 - frac_next_month)
            if land_month_2 in forecast_df.index: forecast_df.loc[land_month_2, '1nHealth Rand Total'] += rands * frac_next_month

    for month, rands in projected_rands_site.items():
        if rands > 0:
            lag_days = rates['Site']['lag'] if pd.notna(rates['Site']['lag']) else 0
            full_months_lag = int(lag_days // days_in_month)
            frac_next_month = (lag_days % days_in_month) / days_in_month
            land_month_1 = month + full_months_lag
            land_month_2 = land_month_1 + 1
            if land_month_1 in forecast_df.index: forecast_df.loc[land_month_1, 'Site Rand Total'] += rands * (1 - frac_next_month)
            if land_month_2 in forecast_df.index: forecast_df.loc[land_month_2, 'Site Rand Total'] += rands * frac_next_month
                
    return forecast_df.apply(np.ceil).astype(int)

# --- Streamlit Application UI ---
st.set_page_config(layout="wide")
st.title("Blended Enrollment Funnel Forecast")

# --- Sidebar for Controls ---
with st.sidebar:
    st.header("⚙️ Data Upload")
    # The two required file uploaders
    uploaded_1nhealth_file = st.file_uploader("1. Upload 1nHealth Funnel Data (CSV)", type="csv")
    uploaded_irt_file = st.file_uploader("2. Upload Full IRT Report (CSV)", type="csv")
    
# --- Main Application Logic ---
data_loaded = False
if uploaded_1nhealth_file is not None and uploaded_irt_file is not None:
    df_1nhealth_raw = pd.read_csv(uploaded_1nhealth_file)
    df_irt_raw = pd.read_csv(uploaded_irt_file)
    data_loaded = True
else:
    # Handle default file loading for demonstration
    if os.path.exists('input_file_2.csv'):
        st.sidebar.write("---")
        st.sidebar.warning("⚠️ Using default sample data. Please upload your own files.")
        # Using the IRT report as a stand-in for both for demo purposes
        df_1nhealth_raw = pd.read_csv('input_file_2.csv')
        df_irt_raw = pd.read_csv('input_file_2.csv')
        data_loaded = True
    else:
        st.info("👋 Welcome! Please upload both the 1nHealth Funnel Data and the IRT Report to begin.")
        st.stop()

# --- App continues only if data is loaded ---
with st.sidebar:
    st.markdown("---")
    forecast_horizon = st.slider("Forecast Horizon (Months)", min_value=3, max_value=24, value=6)

hist_summary, rates = process_data_and_get_rates(df_1nhealth_raw, df_irt_raw)

with st.sidebar:
    st.subheader("Historical Performance Rates")
    st.metric("1nHealth ICF-to-Rand Rate", f"{rates['1nHealth']['rate']:.1%}")
    st.metric("1nHealth Avg. Lag (Days)", f"{rates['1nHealth']['lag']:.1f}")
    st.metric("Site ICF-to-Rand Rate", f"{rates['Site']['rate']:.1%}")
    st.metric("Site Avg. Lag (Days)", f"{rates['Site']['lag']:.1f}")

# --- Editable Forecast Input Table ---
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

# --- Generate Final Display ---
forecast_summary = generate_forecast(rates, forecast_horizon, edited_forecast_assumptions)
master_table = pd.concat([hist_summary, forecast_summary])
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
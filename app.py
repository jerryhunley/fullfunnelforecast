import streamlit as st
import pandas as pd
import numpy as np
from math import ceil
import os
import io
import re
from datetime import datetime

# --- HELPER FUNCTIONS (FROM ORIGINAL APP) ---

@st.cache_data
def parse_funnel_definition(uploaded_file):
    if uploaded_file is None: return None, None
    bytes_data = uploaded_file.getvalue()
    stringio = io.StringIO(bytes_data.decode("utf-8", errors='replace'))
    try:
        df_funnel_def = pd.read_csv(stringio, sep=None, engine='python', header=None)
    except pd.errors.EmptyDataError:
        return None, None
    parsed_funnel = {}; parsed_stages = []
    for col_idx in df_funnel_def.columns:
        stage_name = str(df_funnel_def[col_idx].iloc[0]).strip()
        if pd.isna(stage_name) or stage_name == "": continue
        parsed_stages.append(stage_name)
        statuses = df_funnel_def[col_idx].iloc[1:].dropna().astype(str).str.strip().tolist()
        if stage_name not in statuses: statuses.append(stage_name)
        parsed_funnel[stage_name] = statuses
    return parsed_funnel, parsed_stages

def parse_history(history_str):
    if pd.isna(history_str): return []
    pattern = re.compile(r"([\w\s().'/:-]+?):\s*(\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}(?:\s*[apAP][mM])?)")
    events = []
    for line in str(history_str).strip().split('\n'):
        match = pattern.match(line.strip())
        if match:
            name, dt_str = match.groups()
            dt_obj = pd.to_datetime(dt_str.strip(), errors='coerce')
            if name and pd.notna(dt_obj): events.append((name.strip(), dt_obj))
    events.sort(key=lambda x: x[1])
    return events

# --- MASTER DATA PROCESSING ---

@st.cache_data
def load_and_process_data(referral_file, funnel_def_file, irt_file):
    # --- Pipeline A: 1nHealth Full Funnel Processing ---
    funnel_def, ordered_stages = parse_funnel_definition(funnel_def_file)
    if not funnel_def: raise ValueError("Could not parse Funnel Definition file.")
    
    df_1nhealth = pd.read_csv(referral_file)
    if 'Lead Stage History' not in df_1nhealth.columns:
        raise ValueError("The '1nHealth Referral Data' file must contain a 'Lead Stage History' column.")

    status_to_stage_map = {status: stage for stage, statuses in funnel_def.items() for status in statuses}
    for stage in ordered_stages: df_1nhealth[f'TS_{stage}'] = pd.NaT

    parsed_history = df_1nhealth['Lead Stage History'].apply(parse_history)
    for idx, events in parsed_history.items():
        if not isinstance(events, list): continue
        for event_name, event_dt in events:
            stage = status_to_stage_map.get(event_name)
            if stage in ordered_stages and pd.isna(df_1nhealth.loc[idx, f'TS_{stage}']):
                df_1nhealth.loc[idx, f'TS_{stage}'] = event_dt
    
    # --- Reconciliation Step ---
    df_irt = pd.read_csv(irt_file)
    df_irt.columns = df_irt.columns.str.strip()
    
    # Get all Subject IDs that are definitively from the 1nHealth funnel
    # This assumes 'Subject ID' is the column name in your referral data
    if 'Subject ID' not in df_1nhealth.columns: raise ValueError("1nHealth Referral Data must contain a 'Subject ID' column.")
    id_col_name = 'Subject ID'
    one_n_health_ids = set(df_1nhealth[id_col_name].unique())

    # --- Pipeline B: Isolate Pure Site-Sourced Data ---
    df_site = df_irt[~df_irt[id_col_name].isin(one_n_health_ids)].copy()
    df_site['ICF_Date'] = pd.to_datetime(df_site['Informed Consent Date (Local)'], errors='coerce')
    df_site['Rand_Date'] = pd.to_datetime(df_site['Randomization Date (Local)'], errors='coerce')
    df_site.dropna(subset=['ICF_Date'], inplace=True)
    df_site['ICF_Month'] = df_site['ICF_Date'].dt.to_period('M')
    df_site['Rand_Month'] = df_site['Rand_Date'].dt.to_period('M')

    # --- Calculate Rates and Lags for Both Funnels ---
    rates_1nhealth, lags_1nhealth = {}, {}
    for i in range(len(ordered_stages) - 1):
        from_stage, to_stage = ordered_stages[i], ordered_stages[i+1]
        ts_from, ts_to = f'TS_{from_stage}', f'TS_{to_stage}'
        denominator = df_1nhealth[ts_from].notna().sum()
        numerator = df_1nhealth[df_1nhealth[ts_from].notna()][ts_to].notna().sum()
        rates_1nhealth[f'{from_stage} -> {to_stage}'] = numerator / denominator if denominator > 0 else 0
        lags_1nhealth[f'{from_stage} -> {to_stage}'] = (df_1nhealth[ts_to] - df_1nhealth[ts_from]).dt.days.mean()

    icf_stage_name = next((s for s in ordered_stages if 'icf' in s.lower()), None)
    enroll_stage_name = next((s for s in ordered_stages if 'enroll' in s.lower() or 'rand' in s.lower()), None)
    if not icf_stage_name: raise ValueError("Funnel Definition must contain a stage with 'ICF' in its name.")
    ts_icf_col, ts_enroll_col = f'TS_{icf_stage_name}', f'TS_{enroll_stage_name}' if enroll_stage_name else None

    # --- Create Blended Historical Summary ---
    hist_summary = pd.concat([
        df_1nhealth.groupby(df_1nhealth[ts_icf_col].dt.to_period('M')).size().rename('1nHealth ICF Total'),
        df_1nhealth.groupby(df_1nhealth[ts_enroll_col].dt.to_period('M')).size().rename('1nHealth Rand Total') if ts_enroll_col else pd.Series(name='1nHealth Rand Total', dtype='int64'),
        df_site.groupby('ICF_Month').size().rename('Site ICF Total'),
        df_site.groupby('Rand_Month').size().rename('Site Rand Total')
    ], axis=1).fillna(0).astype(int)
    hist_summary.sort_index(inplace=True)

    rates = {
        '1nHealth': {'rates': rates_1nhealth, 'lags': lags_1nhealth, 'stages': ordered_stages},
        'Site': {
            'rate': df_site['Rand_Month'].notna().sum() / len(df_site) if not df_site.empty else 0,
            'lag': (df_site['Rand_Date'] - df_site['ICF_Date']).dt.days.mean(),
            'avg_icf': df_site.groupby('ICF_Month').size().mean() if not df_site.empty else 0
        }
    }
    return hist_summary, rates, df_1nhealth

# --- FORECASTING ENGINES ---

def calculate_pipeline_yield_forecast(df_1nhealth_processed, rates, forecast_horizon):
    """Calculates yield from leads ALREADY in the 1nHealth funnel."""
    stages = rates['1nHealth']['stages']
    icf_stage_name = next(s for s in stages if 'icf' in s.lower())
    enroll_stage_name = next((s for s in stages if 'enroll' in s.lower() or 'rand' in s.lower()), None)
    
    in_flight = df_1nhealth_processed[df_1nhealth_processed[f'TS_{icf_stage_name}'].isna()].copy()
    
    # Determine current stage for in-flight leads
    def get_current_stage(row):
        for stage in reversed(stages):
            if pd.notna(row[f'TS_{stage}']):
                return stage, row[f'TS_{stage}']
        return None, None
    in_flight[['current_stage', 'current_stage_ts']] = in_flight.apply(get_current_stage, axis=1, result_type='expand')
    in_flight.dropna(subset=['current_stage'], inplace=True)

    future_months = pd.period_range(start=datetime.now(), periods=forecast_horizon, freq='M')
    forecast_df = pd.DataFrame(0.0, index=future_months, columns=['1nHealth ICF Total', '1nHealth Rand Total'])

    for _, row in in_flight.iterrows():
        prob_to_icf, lag_to_icf = 1.0, 0.0
        start_index = stages.index(row['current_stage'])
        # Project to ICF
        for i in range(start_index, stages.index(icf_stage_name)):
            rate_key = f"{stages[i]} -> {stages[i+1]}"
            prob_to_icf *= rates['1nHealth']['rates'].get(rate_key, 0)
            lag_to_icf += rates['1nHealth']['lags'].get(rate_key, 0)
        
        icf_landing_date = row['current_stage_ts'] + pd.to_timedelta(lag_to_icf, 'D')
        icf_landing_month = pd.Period(icf_landing_date, 'M')
        if icf_landing_month in forecast_df.index:
            forecast_df.loc[icf_landing_month, '1nHealth ICF Total'] += prob_to_icf
        
        # Project from ICF to Enrolled
        if enroll_stage_name:
            prob_to_rand = prob_to_icf * rates['1nHealth']['rates'].get(f"{icf_stage_name} -> {enroll_stage_name}", 0)
            lag_to_rand = lag_to_icf + rates['1nHealth']['lags'].get(f"{icf_stage_name} -> {enroll_stage_name}", 0)
            rand_landing_date = row['current_stage_ts'] + pd.to_timedelta(lag_to_rand, 'D')
            rand_landing_month = pd.Period(rand_landing_date, 'M')
            if rand_landing_month in forecast_df.index:
                forecast_df.loc[rand_landing_month, '1nHealth Rand Total'] += prob_to_rand

    return forecast_df.apply(np.ceil).astype(int)


def generate_future_lead_forecast(rates, forecast_horizon, new_leads_per_month):
    """Forecasts yield from NEW leads entering the 1nHealth funnel."""
    # This is a simplified version of the full cohort projection model
    stages = rates['1nHealth']['stages']
    stage_rates = rates['1nHealth']['rates']
    
    monthly_counts = {stage: 0 for stage in stages}
    if stages: monthly_counts[stages[0]] = new_leads_per_month
    
    for i in range(len(stages) - 1):
        from_stage, to_stage = stages[i], stages[i+1]
        rate = stage_rates.get(f'{from_stage} -> {to_stage}', 0)
        monthly_counts[to_stage] = monthly_counts[from_stage] * rate

    icf_stage_name = next(s for s in stages if 'icf' in s.lower())
    enroll_stage_name = next((s for s in stages if 'enroll' in s.lower() or 'rand' in s.lower()), None)
    
    icf_per_cohort = monthly_counts.get(icf_stage_name, 0)
    rand_per_cohort = monthly_counts.get(enroll_stage_name, 0) if enroll_stage_name else 0

    future_months = pd.period_range(start=datetime.now(), periods=forecast_horizon, freq='M')
    forecast_df = pd.DataFrame(0, index=future_months, columns=['1nHealth ICF Total', '1nHealth Rand Total'])
    forecast_df['1nHealth ICF Total'] = icf_per_cohort
    forecast_df['1nHealth Rand Total'] = rand_per_cohort
    
    return forecast_df.apply(np.ceil).astype(int)

def generate_site_forecast(rates, edited_icf_forecast):
    """Simple forecast for Site-Sourced enrollments."""
    forecast_df = pd.DataFrame(index=edited_icf_forecast.index)
    forecast_df['Site ICF Total'] = edited_icf_forecast['Site ICF Total']
    forecast_df['Site Rand Total'] = 0.0
    
    projected_rands = forecast_df['Site ICF Total'] * rates['Site']['rate']
    lag_days = rates['Site']['lag'] if pd.notna(rates['Site']['lag']) else 0
    days_in_month = 30.44
    full_months_lag = int(lag_days // days_in_month)
    frac_next_month = (lag_days % days_in_month) / days_in_month

    for month, rands in projected_rands.items():
        if rands > 0:
            land_month_1, land_month_2 = month + full_months_lag, month + full_months_lag + 1
            if land_month_1 in forecast_df.index: forecast_df.loc[land_month_1, 'Site Rand Total'] += rands * (1 - frac_next_month)
            if land_month_2 in forecast_df.index: forecast_df.loc[land_month_2, 'Site Rand Total'] += rands * frac_next_month
    
    return forecast_df.apply(np.ceil).astype(int)

# --- STREAMLIT UI ---
st.set_page_config(layout="wide")
st.title("Definitive Blended Enrollment Forecast")

with st.sidebar:
    st.header("⚙️ Data Uploads")
    uploaded_referral_file = st.file_uploader("1. 1nHealth Referral Data (with History)", type="csv")
    uploaded_funnel_def_file = st.file_uploader("2. Funnel Definition (Stage Map)", type=["csv", "tsv"])
    uploaded_irt_file = st.file_uploader("3. Full IRT Report", type="csv")

if uploaded_referral_file and uploaded_funnel_def_file and uploaded_irt_file:
    try:
        hist_summary, rates, df_1nhealth_processed = load_and_process_data(uploaded_referral_file, uploaded_funnel_def_file, uploaded_irt_file)
        
        with st.sidebar:
            st.markdown("---")
            st.subheader("Historical Performance Rates")
            st.metric("Site ICF-to-Rand Rate", f"{rates['Site']['rate']:.1%}")
            st.metric("Site Avg. Lag (Days)", f"{rates['Site']['lag']:.1f}")

        st.header("Enrollment Forecast")
        st.markdown("##### Forecast Assumptions")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**1nHealth Funnel**")
            top_stage = rates['1nHealth']['stages'][0]
            new_leads_per_month = st.number_input(f"New '{top_stage}' per Month", min_value=0, value=100, step=10)
            forecast_horizon = st.slider("Forecast Horizon (Months)", 3, 24, 6, key='horizon')
        
        with col2:
            st.markdown("**Site-Sourced Funnel**")
            st.caption("Edit the default monthly ICFs below.")
            last_hist_month = hist_summary.index.max() if not hist_summary.empty else pd.Period(datetime.now(), 'M')
            future_months = pd.period_range(start=last_hist_month + 1, periods=forecast_horizon, freq='M')
            editable_icf_df = pd.DataFrame(index=future_months)
            editable_icf_df['Site ICF Total'] = round(rates['Site']['avg_icf'])
            editable_icf_df.index = editable_icf_df.index.strftime('%Y-%m')
            edited_site_assumptions = st.data_editor(editable_icf_df)
            edited_site_assumptions.index = pd.PeriodIndex(edited_site_assumptions.index, freq='M')

        # --- GENERATE ALL FORECASTS ---
        forecast_pipeline_yield = calculate_pipeline_yield_forecast(df_1nhealth_processed, rates, forecast_horizon)
        forecast_new_leads = generate_future_lead_forecast(rates, forecast_horizon, new_leads_per_month)
        forecast_site = generate_site_forecast(rates, edited_site_assumptions)
        
        # --- BLEND FORECASTS ---
        forecast_1nhealth_total = forecast_pipeline_yield.add(forecast_new_leads, fill_value=0)
        forecast_summary = forecast_1nhealth_total.join(forecast_site, how='outer').fillna(0).astype(int)

        master_table = pd.concat([hist_summary, forecast_summary])
        master_table['Overall ICF Total'] = master_table['1nHealth ICF Total'] + master_table['Site ICF Total']
        master_table['Overall Rand Total'] = master_table['1nHealth Rand Total'] + master_table['Site Rand Total']
        master_table['Overall Running ICF Total'] = master_table['Overall ICF Total'].cumsum()
        master_table['Overall Running Rand Total'] = master_table['Overall Rand Total'].cumsum()

        st.markdown("##### Master Summary Table (Historical & Forecast)")
        st.dataframe(master_table.style.format("{:,.0f}"))
        
        st.markdown("---")
        st.header("Visualizations")
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            st.markdown("#### Monthly Performance by Source")
            chart_data_icf = master_table[['1nHealth ICF Total', 'Site ICF Total']]
            chart_data_icf.index = chart_data_icf.index.to_timestamp()
            st.bar_chart(chart_data_icf, y_label="ICFs per Month")
        with v_col2:
            st.markdown("#### Cumulative Enrollment Over Time")
            chart_data_cumulative = master_table[['Overall Running ICF Total', 'Overall Running Rand Total']]
            chart_data_cumulative.index = chart_data_cumulative.index.to_timestamp()
            st.line_chart(chart_data_cumulative)

    except ValueError as e:
        st.error(f"Data Processing Error: {e}")
        st.warning("Please ensure all uploaded files are correct and contain the required columns (e.g., 'Lead Stage History', 'Subject ID', 'Referral Source').")
    except Exception as e:
        st.error("A critical error occurred.")
        st.exception(e)

else:
    st.info("👋 Welcome! Please upload all three required files to begin: 1nHealth Referrals, Funnel Definition, and the IRT Report.")
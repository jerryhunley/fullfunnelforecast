import streamlit as st
import pandas as pd
import numpy as np
from math import ceil
import io
import re
from datetime import datetime

# --- HELPER FUNCTIONS FOR 1nHEALTH FORECAST MODEL ---

@st.cache_data
def parse_funnel_definition(uploaded_file):
    if not uploaded_file: return None, None, None
    bytes_data = uploaded_file.getvalue()
    stringio = io.StringIO(bytes_data.decode("utf-8", errors='replace'))
    try: df_funnel_def = pd.read_csv(stringio, sep=None, engine='python', header=None)
    except pd.errors.EmptyDataError: return None, None, None
    parsed_def, ordered_stages, ts_col_map = {}, [], {}
    for col_idx in df_funnel_def.columns:
        stage_name = str(df_funnel_def[col_idx].iloc[0]).strip()
        if pd.isna(stage_name) or not stage_name: continue
        ordered_stages.append(stage_name)
        statuses = df_funnel_def[col_idx].iloc[1:].dropna().astype(str).str.strip().tolist()
        if stage_name not in statuses: statuses.append(stage_name)
        parsed_def[stage_name] = statuses
        ts_col_map[stage_name] = f"TS_{stage_name.replace(' ', '_')}"
    return parsed_def, ordered_stages, ts_col_map

def parse_history_string(history_str):
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

def get_stage_timestamps(row, funnel_def, ordered_stages, ts_col_map):
    timestamps = {ts_col: pd.NaT for ts_col in ts_col_map.values()}
    status_to_stage_map = {s: stage for stage, statuses in funnel_def.items() for s in statuses}
    events = row.get('Parsed_Lead_Stage_History', [])
    for event_name, event_dt in events:
        stage = status_to_stage_map.get(event_name)
        if stage and stage in ordered_stages:
            ts_col = ts_col_map.get(stage)
            if ts_col and pd.isna(timestamps[ts_col]):
                timestamps[ts_col] = event_dt
    return pd.Series(timestamps)

@st.cache_data
def build_1nhealth_forecast_model(_referral_file, _funnel_def_file):
    """
    Reads the 1nHealth files to understand the FUNNEL PROCESS (rates, lags)
    and returns the processed dataframe of leads for pipeline yield calculation.
    THIS FUNCTION DOES NOT CARE ABOUT HISTORICAL COUNTS.
    """
    funnel_def, ordered_stages, ts_col_map = parse_funnel_definition(_funnel_def_file)
    if not funnel_def: raise ValueError("Could not parse Funnel Definition file.")
    
    df = pd.read_csv(_referral_file)
    if 'Lead Stage History' not in df.columns: raise ValueError("1nHealth Referral Data must have 'Lead Stage History'.")
    
    df['Parsed_Lead_Stage_History'] = df['Lead Stage History'].apply(parse_history_string)
    timestamp_df = df.apply(lambda row: get_stage_timestamps(row, funnel_def, ordered_stages, ts_col_map), axis=1)
    df = pd.concat([df, timestamp_df], axis=1)
    
    rates, lags = {}, {}
    for i in range(len(ordered_stages) - 1):
        from_stage, to_stage = ordered_stages[i], ordered_stages[i+1]
        ts_from, ts_to = ts_col_map[from_stage], ts_col_map[to_stage]
        denominator = df[ts_from].notna().sum()
        numerator = df[df[ts_from].notna()][ts_to].notna().sum()
        rates[f"{from_stage} -> {to_stage}"] = numerator / denominator if denominator > 0 else 0
        lags[f"{from_stage} -> {to_stage}"] = (df[ts_to] - df[ts_from]).dt.days.mean()
    
    return df, rates, lags, ordered_stages, ts_col_map

@st.cache_data
def get_historical_summary_and_site_rates(_irt_file):
    """
    Gets ALL historical counts and site-only rates from the IRT file.
    THIS IS THE SINGLE SOURCE OF TRUTH FOR HISTORICALS.
    """
    df_irt = pd.read_csv(_irt_file)
    df_irt.columns = df_irt.columns.str.strip()
    if 'Referral Source' not in df_irt.columns: raise ValueError("IRT Report MUST contain a 'Referral Source' column.")

    df_irt['ICF_Date'] = pd.to_datetime(df_irt['Informed Consent Date (Local)'], errors='coerce')
    df_irt['Rand_Date'] = pd.to_datetime(df_irt['Randomization Date (Local)'], errors='coerce')
    df_irt.dropna(subset=['ICF_Date'], inplace=True)
    df_irt['Referral Source'] = df_irt['Referral Source'].str.strip()
    df_irt['ICF_Month'] = df_irt['ICF_Date'].dt.to_period('M')
    df_irt['Rand_Month'] = df_irt['Rand_Date'].dt.to_period('M')

    historical_summary = pd.concat([
        df_irt[df_irt['Referral Source'] == '1nHealth'].groupby('ICF_Month').size().rename('1nHealth ICF Total'),
        df_irt[df_irt['Referral Source'] == '1nHealth'].groupby('Rand_Month').size().rename('1nHealth Rand Total'),
        df_irt[df_irt['Referral Source'] == 'Site'].groupby('ICF_Month').size().rename('Site ICF Total'),
        df_irt[df_irt['Referral Source'] == 'Site'].groupby('Rand_Month').size().rename('Site Rand Total')
    ], axis=1).fillna(0).astype(int)
    historical_summary.sort_index(inplace=True)

    df_site_only = df_irt[df_irt['Referral Source'] == 'Site']
    site_rates = {
        'rate': len(df_site_only[df_site_only['Rand_Date'].notna()]) / len(df_site_only) if not df_site_only.empty else 0,
        'lag': (df_site_only['Rand_Date'] - df_site_only['ICF_Date']).dt.days.mean(),
        'avg_icf': df_site_only.groupby('ICF_Month').size().mean() if not df_site_only.empty else 0
    }
    return historical_summary, site_rates

# --- FORECASTING ENGINES ---
# ... (These are now correct because they are fed by the correct data sources) ...
def calculate_pipeline_yield(df_1nhealth, rates, lags, ordered_stages, ts_col_map, horizon):
    """The 'Funnel Analysis' component: projects yield from in-flight leads."""
    icf_stage = next((s for s in ordered_stages if 'icf' in s.lower()), None)
    enroll_stage = next((s for s in ordered_stages if 'enroll' in s.lower() or 'rand' in s.lower()), None)
    if not icf_stage: return pd.DataFrame()

    in_flight = df_1nhealth[df_1nhealth[ts_col_map[icf_stage]].isna()].copy()
    
    def get_current_stage(row):
        for stage in reversed(ordered_stages):
            if pd.notna(row[ts_col_map[stage]]): return stage, row[ts_col_map[stage]]
        return None, None
    in_flight[['curr_stage', 'ts']] = in_flight.apply(get_current_stage, axis=1, result_type='expand')
    in_flight.dropna(subset=['curr_stage'], inplace=True)

    future_months = pd.period_range(start=datetime.now(), periods=horizon, freq='M')
    forecast = pd.DataFrame(0.0, index=future_months, columns=['1nHealth ICF Total', '1nHealth Rand Total'])

    for _, row in in_flight.iterrows():
        prob, lag = 1.0, 0.0
        start_idx = ordered_stages.index(row['curr_stage'])
        for i in range(start_idx, len(ordered_stages) - 1):
            from_s, to_s = ordered_stages[i], ordered_stages[i+1]
            prob *= rates.get(f"{from_s} -> {to_s}", 0)
            lag += lags.get(f"{from_s} -> {to_s}", 0)
            
            if to_s == icf_stage and pd.notna(row['ts']) and pd.notna(lag):
                date = row['ts'] + pd.to_timedelta(lag, 'D')
                if pd.Period(date, 'M') in forecast.index: forecast.loc[pd.Period(date, 'M'), '1nHealth ICF Total'] += prob
            if to_s == enroll_stage and pd.notna(row['ts']) and pd.notna(lag):
                date = row['ts'] + pd.to_timedelta(lag, 'D')
                if pd.Period(date, 'M') in forecast.index: forecast.loc[pd.Period(date, 'M'), '1nHealth Rand Total'] += prob
    return forecast

def forecast_new_leads(rates, lags, ordered_stages, horizon, leads_per_month):
    """The 'Projections' component: projects yield from new leads."""
    counts = {s: 0 for s in ordered_stages}
    if ordered_stages: counts[ordered_stages[0]] = leads_per_month
    for i in range(len(ordered_stages) - 1):
        rate = rates.get(f"{ordered_stages[i]} -> {ordered_stages[i+1]}", 0)
        counts[ordered_stages[i+1]] = counts[ordered_stages[i]] * rate
    
    icf_name = next(s for s in ordered_stages if 'icf' in s.lower())
    enroll_name = next((s for s in ordered_stages if 'enroll' in s.lower() or 'rand' in s.lower()), None)
    
    future_months = pd.period_range(start=datetime.now(), periods=horizon, freq='M')
    forecast = pd.DataFrame(0, index=future_months, columns=['1nHealth ICF Total', '1nHealth Rand Total'])
    forecast['1nHealth ICF Total'] = counts.get(icf_name, 0)
    forecast['1nHealth Rand Total'] = counts.get(enroll_name, 0) if enroll_name else 0
    return forecast

def forecast_site_sourced(site_rates, edited_assumptions):
    """The simple forecast for site-sourced leads."""
    forecast = pd.DataFrame(index=edited_assumptions.index)
    forecast['Site ICF Total'] = edited_assumptions['Site ICF Total']
    forecast['Site Rand Total'] = 0.0
    projected_rands = forecast['Site ICF Total'] * site_rates['rate']
    lag = site_rates['lag'] if pd.notna(site_rates['lag']) else 0
    lag_m, lag_f = int(lag // 30.44), (lag % 30.44) / 30.44
    for month, rands in projected_rands.items():
        if rands > 0:
            m1, m2 = month + lag_m, month + lag_m + 1
            if m1 in forecast.index: forecast.loc[m1, 'Site Rand Total'] += rands * (1 - lag_f)
            if m2 in forecast.index: forecast.loc[m2, 'Site Rand Total'] += rands * lag_f
    return forecast

# --- STREAMLIT UI ---
st.set_page_config(layout="wide")
st.title("Definitive Blended Enrollment Forecast")

with st.sidebar:
    st.header("⚙️ Data Uploads")
    uploaded_referral_file = st.file_uploader("1. 1nHealth Referral Data (with History)")
    uploaded_funnel_def_file = st.file_uploader("2. Funnel Definition (Stage Map)")
    uploaded_irt_file = st.file_uploader("3. Full IRT Report (with 'Referral Source')")

if uploaded_referral_file and uploaded_funnel_def_file and uploaded_irt_file:
    try:
        df_1nhealth, rates_1nhealth, lags_1nhealth, ordered_stages, ts_col_map = build_1nhealth_forecast_model(
            uploaded_referral_file, uploaded_funnel_def_file
        )
        hist_summary, site_rates = get_historical_summary_and_site_rates(uploaded_irt_file)
        
        with st.sidebar:
            st.markdown("---")
            st.subheader("Historical Site Performance")
            st.metric("Site ICF-to-Rand Rate", f"{site_rates['rate']:.1%}")
            st.metric("Site Avg. Lag (Days)", f"{site_rates['lag']:.1f}")

        st.header("Enrollment Forecast")
        st.markdown("##### Forecast Assumptions")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**1nHealth Funnel**")
            top_stage = ordered_stages[0]
            new_leads = st.number_input(f"New '{top_stage}' per Month", 0, None, 100, 10)
            horizon = st.slider("Forecast Horizon (Months)", 3, 24, 6, key='horizon')
        
        with col2:
            st.markdown("**Site-Sourced Funnel**")
            st.caption("Edit default monthly ICFs below.")
            last_hist = hist_summary.index.max() if not hist_summary.empty else pd.Period(datetime.now(), 'M')
            future_m = pd.period_range(start=last_hist + 1, periods=horizon, freq='M')
            editable_df = pd.DataFrame({'Site ICF Total': round(site_rates['avg_icf'])}, index=future_m)
            edited_site = st.data_editor(editable_df.rename(index=lambda p: p.strftime('%Y-%m')))
            edited_site.index = pd.PeriodIndex(edited_site.index, freq='M')

        # --- GENERATE & BLEND FORECASTS ---
        yield_forecast = calculate_pipeline_yield(df_1nhealth, rates_1nhealth, lags_1nhealth, ordered_stages, ts_col_map, horizon)
        new_lead_forecast = forecast_new_leads(rates_1nhealth, lags_1nhealth, ordered_stages, horizon, new_leads)
        site_forecast = forecast_site_sourced(site_rates, edited_site)
        
        total_1nhealth_forecast = yield_forecast.add(new_lead_forecast, fill_value=0)
        final_forecast = total_1nhealth_forecast.join(site_forecast, how='outer').fillna(0).apply(np.ceil).astype(int)

        master_table = pd.concat([hist_summary, final_forecast])
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
            chart_data = master_table[['1nHealth ICF Total', 'Site ICF Total']]
            st.bar_chart(chart_data.rename(columns=lambda c: c.replace(' Total', '')), y_label="ICFs per Month")
        with v_col2:
            st.markdown("#### Cumulative Enrollment Over Time")
            st.line_chart(master_table[['Overall Running ICF Total', 'Overall Running Rand Total']])

    except ValueError as e:
        st.error(f"Data Processing Error: {e}")
    except Exception as e:
        st.error(f"A critical error occurred: {e}")
        st.exception(e)

else:
    st.info("👋 Welcome! Please upload all three required files to begin.")
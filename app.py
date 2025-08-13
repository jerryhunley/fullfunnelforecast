# app.py
import streamlit as st
import pandas as pd
import numpy as np
import math
import re
from datetime import datetime, timedelta
import io

# --- 1. Page Configuration ---
st.set_page_config(page_title="Combined Recruitment Forecast", layout="wide")
st.title("🚀 Combined Recruitment Forecast Tool")
st.info("""
This tool provides a complete picture of future recruitment by combining two sources:
1.  **Existing Pipeline:** Expected ICFs and Enrollments from leads already in your funnel.
2.  **New Leads:** Projected ICFs based on your future ad spend and CPQR assumptions.
""")

# --- 2. Core Data Processing and Helper Functions (Robust Versions) ---

# --- Constants for Stage Names ---
STAGE_PASSED_ONLINE_FORM = "Passed Online Form"
STAGE_PRE_SCREENING_ACTIVITIES = "Pre-Screening Activities"
STAGE_SENT_TO_SITE = "Sent To Site"
STAGE_APPOINTMENT_SCHEDULED = "Appointment Scheduled"
STAGE_SIGNED_ICF = "Signed ICF"
STAGE_SCREEN_FAILED = "Screen Failed"
STAGE_ENROLLED = "Enrolled"
STAGE_LOST = "Lost"

@st.cache_data
def parse_funnel_definition(uploaded_file):
    if uploaded_file is None: return None, None, None
    try:
        bytes_data = uploaded_file.getvalue()
        stringio = io.StringIO(bytes_data.decode("utf-8", errors='replace'))
        df_funnel_def = pd.read_csv(stringio, sep=None, engine='python', header=None)

        parsed_funnel_definition = {}; parsed_ordered_stages = []; ts_col_map = {}
        for col_idx in df_funnel_def.columns:
            column_data = df_funnel_def[col_idx]
            stage_name = column_data.iloc[0]
            if pd.isna(stage_name) or str(stage_name).strip() == "": continue
            
            stage_name = str(stage_name).strip().replace('"', '')
            parsed_ordered_stages.append(stage_name)
            
            statuses = column_data.iloc[1:].dropna().astype(str).apply(lambda x: x.strip().replace('"', '')).tolist()
            statuses = [s for s in statuses if s]
            
            if stage_name not in statuses:
                statuses.append(stage_name)
                
            parsed_funnel_definition[stage_name] = statuses
            clean_ts_name = f"TS_{stage_name.replace(' ', '_').replace('(', '').replace(')', '')}"
            ts_col_map[stage_name] = clean_ts_name
            
        if not parsed_ordered_stages:
            st.error("Could not parse stages from Funnel Definition. Check file format.")
            return None, None, None
            
        return parsed_funnel_definition, parsed_ordered_stages, ts_col_map
    except Exception as e:
        st.error(f"Error parsing Funnel Definition file: {e}")
        return None, None, None

def parse_datetime_with_timezone(dt_str):
    if pd.isna(dt_str): return pd.NaT
    dt_str_cleaned = str(dt_str).strip()
    tz_pattern = r'\s+(?:EST|EDT|CST|CDT|MST|MDT|PST|PDT)$'
    dt_str_no_tz = re.sub(tz_pattern, '', dt_str_cleaned)
    return pd.to_datetime(dt_str_no_tz, errors='coerce')

def parse_history_string(history_str):
    if pd.isna(history_str) or str(history_str).strip() == "": return []
    pattern = re.compile(r"([\w\s().'/:-]+?):\s*(\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}(?:\s*[apAP][mM])?(?:\s+[A-Za-z]{3,}(?:T)?)?)")
    raw_lines = str(history_str).strip().split('\n')
    parsed_events = []
    for line in raw_lines:
        line = line.strip()
        if not line: continue
        match = pattern.match(line)
        if match:
            name, dt_str = match.groups()
            name = name.strip()
            dt_obj = parse_datetime_with_timezone(dt_str.strip())
            if name and pd.notna(dt_obj):
                try:
                    py_dt = dt_obj.to_pydatetime()
                    parsed_events.append((name, py_dt))
                except AttributeError:
                    pass
    try:
        parsed_events.sort(key=lambda x: x[1] if pd.notna(x[1]) else datetime.min)
    except TypeError:
        pass
    return parsed_events

def get_stage_timestamps(row, parsed_stage_history_col, parsed_status_history_col, funnel_def, ordered_stgs, ts_col_mapping):
    timestamps = {ts_col_mapping[stage]: pd.NaT for stage in ordered_stgs}
    status_to_stage_map = {}
    if not funnel_def: return pd.Series(timestamps)
    for stage, statuses in funnel_def.items():
        for status in statuses: status_to_stage_map[status] = stage
    
    all_events = []
    stage_hist = row.get(parsed_stage_history_col, [])
    status_hist = row.get(parsed_status_history_col, [])
    if stage_hist: all_events.extend([(name, dt) for name, dt in stage_hist if isinstance(name, str)])
    if status_hist: all_events.extend([(name, dt) for name, dt in status_hist if isinstance(name, str)])
    
    try:
        all_events.sort(key=lambda x: x[1] if pd.notna(x[1]) else datetime.min)
    except TypeError as e:
        pass

    for event_name, event_dt in all_events:
        if pd.isna(event_dt): continue
        event_stage = None
        if event_name in ordered_stgs: event_stage = event_name
        elif event_name in status_to_stage_map: event_stage = status_to_stage_map[event_name]
        
        if event_stage and event_stage in ordered_stgs:
            ts_col_name = ts_col_mapping.get(event_stage)
            if ts_col_name and pd.isna(timestamps[ts_col_name]):
                timestamps[ts_col_name] = event_dt
                
    return pd.Series(timestamps, dtype='datetime64[ns]')

@st.cache_data
def preprocess_referral_data(_df_raw, funnel_def, ordered_stages, ts_col_map):
    if _df_raw is None or funnel_def is None: return None
    df = _df_raw.copy()
    
    submitted_on_col = "Submitted On" if "Submitted On" in df.columns else "Referral Date"
    if submitted_on_col not in df.columns:
        st.error("Data must contain either a 'Submitted On' or 'Referral Date' column.")
        return None
    
    if "Referral Date" in df.columns: df.rename(columns={"Referral Date": "Submitted On"}, inplace=True)
    df["Submitted On_DT"] = df[submitted_on_col].apply(lambda x: parse_datetime_with_timezone(str(x)))
    
    initial_rows = len(df)
    df.dropna(subset=["Submitted On_DT"], inplace=True)
    if (initial_rows - len(df)) > 0: st.warning(f"Dropped {initial_rows - len(df)} rows due to unparseable dates.")
    if df.empty:
        st.error("No valid data remaining after date parsing.")
        return None

    df["Submission_Month"] = df["Submitted On_DT"].dt.to_period('M')
    
    parsed_cols = {}
    for col_name in ['Lead Stage History', 'Lead Status History']:
        if col_name in df.columns:
            parsed_col_name = f"Parsed_{col_name.replace(' ', '_')}"
            df[parsed_col_name] = df[col_name].astype(str).apply(parse_history_string)
            parsed_cols[col_name] = parsed_col_name
        else:
            st.warning(f"History column '{col_name}' not found. Timestamps may be incomplete.")

    parsed_stage_hist_col = parsed_cols.get('Lead Stage History')
    parsed_status_hist_col = parsed_cols.get('Lead Status History')

    if not parsed_stage_hist_col and not parsed_status_hist_col:
        st.error("Neither 'Lead Stage History' nor 'Lead Status History' found. Cannot determine stage progression.")
        return None

    timestamp_cols_df = df.apply(lambda row: get_stage_timestamps(row, parsed_stage_hist_col, parsed_status_hist_col, funnel_def, ordered_stages, ts_col_map), axis=1)
    df = pd.concat([df.drop(columns=[col for col in df.columns if col.startswith('TS_')], errors='ignore'), timestamp_cols_df], axis=1)
    
    for stage, ts_col in ts_col_map.items():
        if ts_col in df.columns:
            df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
            
    return df

def calculate_avg_lag_generic(df, col_from, col_to):
    if not all(c in df.columns and pd.api.types.is_datetime64_any_dtype(df[c]) for c in [col_from, col_to]): return np.nan
    valid_df = df.dropna(subset=[col_from, col_to])
    if valid_df.empty: return np.nan
    diff = valid_df[col_to] - valid_df[col_from]
    diff_positive = diff[diff >= pd.Timedelta(days=0)]
    return diff_positive.mean().total_seconds() / (60*60*24) if not diff_positive.empty else np.nan

@st.cache_data
def calculate_overall_inter_stage_lags(_processed_df, ordered_stages, ts_col_map):
    if _processed_df is None or not ordered_stages: return {}
    lags = {}
    for i in range(len(ordered_stages) - 1):
        stage_from, stage_to = ordered_stages[i], ordered_stages[i+1]
        ts_from, ts_to = ts_col_map.get(stage_from), ts_col_map.get(stage_to)
        if ts_from and ts_to:
            lags[f"{stage_from} -> {stage_to}"] = calculate_avg_lag_generic(_processed_df, ts_from, ts_to)
    return lags

# --- 3. Core Forecasting Functions (Robust Versions) ---

@st.cache_data
def determine_effective_projection_rates(_processed_df, ordered_stages, ts_col_map,
                                          rate_method, rolling_window, manual_rates,
                                          inter_stage_lags_for_maturity):
    if rate_method == 'Manual Input Below':
        return manual_rates, "Manual Input"
    if _processed_df is None or _processed_df.empty:
        return manual_rates, "Manual (No History for Rolling)"

    calculated_rolling_rates = {}
    try:
        hist_counts = _processed_df.groupby("Submission_Month").size().to_frame(name="Total_Submissions")
        for stage_name in ordered_stages:
            ts_col = ts_col_map.get(stage_name)
            if ts_col and ts_col in _processed_df.columns:
                reached_col = f"Reached_{stage_name.replace(' ', '_').replace('(', '').replace(')', '')}"
                stage_counts = _processed_df.dropna(subset=[ts_col]).groupby('Submission_Month').size()
                hist_counts = hist_counts.join(stage_counts.rename(reached_col), how='left')
        hist_counts = hist_counts.fillna(0)
        
        for rate_key, manual_rate_val in manual_rates.items():
            try:
                stage_from, stage_to = rate_key.split(" -> ")
                col_from = f"Reached_{stage_from.replace(' ', '_').replace('(', '').replace(')', '')}"
                col_to = f"Reached_{stage_to.replace(' ', '_').replace('(', '').replace(')', '')}"

                if col_from in hist_counts.columns and col_to in hist_counts.columns:
                    monthly_rates = (hist_counts[col_to] / hist_counts[col_from].replace(0, np.nan)).dropna()
                    if not monthly_rates.empty:
                        window = min(rolling_window, len(monthly_rates)) if rolling_window != 999 else len(monthly_rates)
                        rolling_avg = monthly_rates.rolling(window=window, min_periods=1).mean()
                        if not rolling_avg.empty:
                            calculated_rolling_rates[rate_key] = rolling_avg.iloc[-1]
                        else:
                            calculated_rolling_rates[rate_key] = manual_rate_val
                    else:
                        calculated_rolling_rates[rate_key] = manual_rate_val
                else:
                    calculated_rolling_rates[rate_key] = manual_rate_val
            except Exception:
                calculated_rolling_rates[rate_key] = manual_rate_val
        
        desc = f"Rolling {rolling_window}-Month Avg" if rolling_window != 999 else "Overall Historical Average"
        return calculated_rolling_rates, desc
    except Exception as e:
        st.warning(f"Could not calculate rolling rates due to error: {e}. Falling back to manual rates.")
        return manual_rates, "Manual (Error in Rolling Calc)"


@st.cache_data
def calculate_projections(_processed_df, ordered_stages, ts_col_map, projection_inputs):
    # This is the robust version from the original app, preventing the bug.
    default_return = pd.DataFrame(), np.nan, "N/A", "N/A", pd.DataFrame(), "N/A"
    if _processed_df is None: return default_return
    
    horizon = projection_inputs['horizon']
    future_spend = projection_inputs['spend_dict']
    assumed_cpqr = projection_inputs['cpqr_dict']
    conv_rates = projection_inputs['final_conv_rates']
    inter_stage_lags = projection_inputs.get('inter_stage_lags', {})

    total_lag = 0
    try:
        icf_index = ordered_stages.index(STAGE_SIGNED_ICF)
        for i in range(icf_index):
             key = f"{ordered_stages[i]} -> {ordered_stages[i+1]}"
             total_lag += inter_stage_lags.get(key, 30) # Default 30 days if lag is missing
    except (ValueError, IndexError):
        total_lag = 90 # Fallback lag if stages are not found

    last_hist_month = _processed_df["Submission_Month"].max() if "Submission_Month" in _processed_df and not _processed_df["Submission_Month"].empty else pd.Period(datetime.now(), freq='M') - 1
    proj_start_month = last_hist_month + 1
    future_months = pd.period_range(start=proj_start_month, periods=horizon, freq='M')
    
    cohorts = pd.DataFrame(index=future_months)
    cohorts['Ad_Spend'] = [future_spend.get(m, 0.0) for m in future_months]
    cohorts['CPQR'] = [assumed_cpqr.get(m, 120.0) for m in future_months]
    cohorts['New_QLs'] = (cohorts['Ad_Spend'] / cohorts['CPQR'].replace(0, np.nan)).fillna(0).round().astype(int)

    current_counts = cohorts['New_QLs'].copy()
    if 'icf_index' in locals() and icf_index > 0:
        for i in range(icf_index):
            rate = conv_rates.get(f"{ordered_stages[i]} -> {ordered_stages[i+1]}", 0.0)
            current_counts *= rate
    cohorts['Generated_ICFs'] = current_counts

    max_landing_month = future_months[-1] + int(np.ceil(total_lag / 30.4375)) + 1
    results_index = pd.period_range(start=proj_start_month, end=max_landing_month, freq='M')
    results = pd.DataFrame(0.0, index=results_index, columns=['Projected_ICF_Landed'])
    
    days_in_month = 30.4375
    for cohort_month, row in cohorts.iterrows():
        icfs_generated = row['Generated_ICFs']
        if icfs_generated <= 0: continue
        
        full_lag_months = int(total_lag // days_in_month)
        rem_lag_days = total_lag % days_in_month
        frac_next_month = rem_lag_days / days_in_month
        frac_this_month = 1.0 - frac_next_month

        landing_month_1 = cohort_month + full_lag_months
        landing_month_2 = landing_month_1 + 1

        if landing_month_1 in results.index:
            results.loc[landing_month_1, 'Projected_ICF_Landed'] += icfs_generated * frac_this_month
        if landing_month_2 in results.index:
            results.loc[landing_month_2, 'Projected_ICF_Landed'] += icfs_generated * frac_next_month
            
    final_results = results.reindex(future_months).fillna(0).round().astype(int)
    return final_results, total_lag, "N/A", "N/A", pd.DataFrame(), "Lag calculated from sum of steps."

@st.cache_data
def calculate_pipeline_projection(_processed_df, ordered_stages, ts_col_map, inter_stage_lags, conversion_rates):
    default_return = {'results_df': pd.DataFrame(), 'total_icf_yield': 0, 'total_enroll_yield': 0}
    if _processed_df is None or _processed_df.empty: return default_return

    terminal_ts_cols = [ts_col_map.get(s) for s in [STAGE_ENROLLED, STAGE_SCREEN_FAILED, STAGE_LOST] if s in ts_col_map]
    in_flight_df = _processed_df.copy()
    for ts_col in terminal_ts_cols:
        if ts_col in in_flight_df.columns and pd.api.types.is_datetime64_any_dtype(in_flight_df[ts_col]):
            in_flight_df = in_flight_df[in_flight_df[ts_col].isna()]
    if in_flight_df.empty: return default_return

    def get_current_stage(row, ordered_stages, ts_col_map):
        last_stage, last_ts = None, pd.NaT
        for stage in ordered_stages:
            if stage in [STAGE_ENROLLED, STAGE_SCREEN_FAILED, STAGE_LOST]: continue
            ts_col = ts_col_map.get(stage)
            if ts_col in row and pd.notna(row[ts_col]):
                if pd.isna(last_ts) or row[ts_col] > last_ts:
                    last_ts, last_stage = row[ts_col], stage
        return last_stage, last_ts
    in_flight_df[['current_stage', 'current_stage_ts']] = in_flight_df.apply(lambda row: get_current_stage(row, ordered_stages, ts_col_map), axis=1, result_type='expand')
    in_flight_df.dropna(subset=['current_stage'], inplace=True)

    projections = []
    for _, row in in_flight_df.iterrows():
        prob_to_icf, lag_to_icf = 1.0, 0.0
        start_index = ordered_stages.index(row['current_stage'])
        
        icf_index = ordered_stages.index(STAGE_SIGNED_ICF) if STAGE_SIGNED_ICF in ordered_stages else -1
        if icf_index == -1: continue

        for i in range(start_index, icf_index):
            from_stage, to_stage = ordered_stages[i], ordered_stages[i+1]
            prob_to_icf *= conversion_rates.get(f"{from_stage} -> {to_stage}", 0.0)
            lag_to_icf += inter_stage_lags.get(f"{from_stage} -> {to_stage}", 0.0)
        
        if prob_to_icf > 0:
            icf_date = row['current_stage_ts'] + pd.to_timedelta(lag_to_icf, unit='D')
            prob_icf_to_enroll = conversion_rates.get(f"{STAGE_SIGNED_ICF} -> {STAGE_ENROLLED}", 0.0)
            lag_icf_to_enroll = inter_stage_lags.get(f"{STAGE_SIGNED_ICF} -> {STAGE_ENROLLED}", 0.0)
            enroll_prob = prob_to_icf * prob_icf_to_enroll if pd.notna(prob_icf_to_enroll) else 0
            enroll_date = icf_date + pd.to_timedelta(lag_icf_to_enroll, unit='D') if pd.notna(lag_icf_to_enroll) else pd.NaT
            projections.append({'icf_prob': prob_to_icf, 'icf_date': icf_date, 'enroll_prob': enroll_prob, 'enroll_date': enroll_date})

    if not projections: return default_return

    proj_df = pd.DataFrame(projections)
    all_dates = pd.concat([proj_df['icf_date'], proj_df['enroll_date']]).dropna()
    max_date = all_dates.max() if not all_dates.empty else datetime.now()
    
    start_period = pd.Period(datetime.now(), 'M')
    end_period = pd.Period(max_date, 'M')
    results_index = pd.period_range(start=start_period, end=max(end_period, start_period + 11), freq='M')
    results_df = pd.DataFrame(0.0, index=results_index, columns=['Projected_ICF_Landed', 'Projected_Enrollments_Landed'])

    icf_landed = proj_df.groupby(proj_df['icf_date'].dt.to_period('M'))['icf_prob'].sum()
    enroll_landed = proj_df.groupby(proj_df['enroll_date'].dt.to_period('M'))['enroll_prob'].sum()
    
    results_df['Projected_ICF_Landed'] = results_df.index.map(icf_landed)
    results_df['Projected_Enrollments_Landed'] = results_df.index.map(enroll_landed)
    results_df = results_df.fillna(0).round().astype(int)

    return {
        'results_df': results_df,
        'total_icf_yield': proj_df['icf_prob'].sum(),
        'total_enroll_yield': proj_df['enroll_prob'].sum()
    }

@st.cache_data
def calculate_combined_forecast(_processed_df, ordered_stages, ts_col_map, inter_stage_lags, projection_inputs, funnel_analysis_inputs):
    # Part 1: Projections from Future Spend
    df_from_new_leads, _, _, _, _, _ = calculate_projections(_processed_df, ordered_stages, ts_col_map, projection_inputs)
    if not df_from_new_leads.empty:
        df_from_new_leads = df_from_new_leads.rename(columns={"Projected_ICF_Landed": "ICFs from New Leads"})[['ICFs from New Leads']]

    # Part 2: Projections from Existing Pipeline
    pipeline_results = calculate_pipeline_projection(_processed_df, ordered_stages, ts_col_map, inter_stage_lags, funnel_analysis_inputs['final_conv_rates'])
    df_from_pipeline = pipeline_results.get('results_df', pd.DataFrame())
    if not df_from_pipeline.empty:
        df_from_pipeline = df_from_pipeline.rename(columns={"Projected_ICF_Landed": "ICFs from Pipeline", "Projected_Enrollments_Landed": "Enrollments from Pipeline"})[['ICFs from Pipeline', 'Enrollments from Pipeline']]

    # Part 3: Combine DataFrames
    if isinstance(df_from_new_leads.index, pd.PeriodIndex): df_from_new_leads.index = df_from_new_leads.index.to_timestamp()
    if isinstance(df_from_pipeline.index, pd.PeriodIndex): df_from_pipeline.index = df_from_pipeline.index.to_timestamp()

    combined_df = df_from_new_leads.add(df_from_pipeline, fill_value=0).fillna(0)
    
    # Part 4: Calculate Totals and Cumulative Sums
    for col in ['ICFs from New Leads', 'ICFs from Pipeline', 'Enrollments from Pipeline']:
        if col not in combined_df: combined_df[col] = 0
    
    combined_df['Total Projected ICFs'] = (combined_df['ICFs from New Leads'] + combined_df['ICFs from Pipeline']).round().astype(int)
    combined_df['Enrollments from Pipeline'] = combined_df['Enrollments from Pipeline'].round().astype(int)
    combined_df['ICFs from New Leads'] = combined_df['ICFs from New Leads'].round().astype(int)
    combined_df['ICFs from Pipeline'] = combined_df['ICFs from Pipeline'].round().astype(int)

    combined_df['Cumulative ICFs'] = combined_df['Total Projected ICFs'].cumsum()
    combined_df['Cumulative Enrollments'] = combined_df['Enrollments from Pipeline'].cumsum()

    final_cols_order = ['ICFs from New Leads', 'ICFs from Pipeline', 'Total Projected ICFs', 'Enrollments from Pipeline', 'Cumulative ICFs', 'Cumulative Enrollments']
    combined_df = combined_df.reindex(columns=final_cols_order).fillna(0)

    return {
        'combined_df': combined_df,
        'total_pipeline_icf_yield': pipeline_results.get('total_icf_yield', 0),
        'total_new_lead_icfs': df_from_new_leads['ICFs from New Leads'].sum()
    }


# --- 4. Streamlit UI and Application Logic ---

if 'data_processed' not in st.session_state: st.session_state.data_processed = False

with st.sidebar:
    st.header("⚙️ Setup")
    uploaded_referral_file = st.file_uploader("1. Upload Referral Data (CSV)", type=["csv"])
    uploaded_funnel_def_file = st.file_uploader("2. Upload Funnel Definition (CSV/TSV)", type=["csv", "tsv"])

if uploaded_referral_file and uploaded_funnel_def_file and not st.session_state.data_processed:
    funnel_def, ordered_stages, ts_col_map = parse_funnel_definition(uploaded_funnel_def_file)
    if funnel_def and ordered_stages and ts_col_map:
        try:
            raw_df = pd.read_csv(uploaded_referral_file, low_memory=False)
            st.session_state.processed_df = preprocess_referral_data(raw_df, funnel_def, ordered_stages, ts_col_map)
            if st.session_state.processed_df is not None:
                st.session_state.ordered_stages = ordered_stages
                st.session_state.ts_col_map = ts_col_map
                st.session_state.inter_stage_lags = calculate_overall_inter_stage_lags(st.session_state.processed_df, ordered_stages, ts_col_map)
                st.session_state.data_processed = True
                st.success("Data loaded and processed successfully!")
                st.rerun()
        except Exception as e:
            st.error(f"Error loading referral data: {e}")

if not st.session_state.get('data_processed', False):
    st.warning("Please upload both data files in the sidebar to begin.")
else:
    # --- UI Controls ---
    st.subheader("1. Future Spend & New Lead Assumptions")
    proj_horizon = st.number_input("Projection Horizon (Months)", 1, 48, 12, 1, key='proj_horizon')
    
    last_hist_month = st.session_state.processed_df["Submission_Month"].max()
    _proj_start_month = last_hist_month + 1 if pd.notna(last_hist_month) else pd.Period(datetime.now(), 'M')
    future_months_ui = pd.period_range(start=_proj_start_month, periods=proj_horizon, freq='M')

    col_spend, col_cpqr = st.columns(2)
    with col_spend:
        st.write("Future Monthly Ad Spend:")
        if 'spend_df_cache' not in st.session_state or len(st.session_state.spend_df_cache) != proj_horizon:
            st.session_state.spend_df_cache = pd.DataFrame([{'Month': m.strftime('%Y-%m'), 'Planned_Spend': 20000.0} for m in future_months_ui])
        edited_spend_df = st.data_editor(st.session_state.spend_df_cache, key='spend_editor', use_container_width=True, num_rows="fixed")
        proj_spend_dict = {pd.Period(row['Month'], 'M'): float(row['Planned_Spend']) for _, row in edited_spend_df.iterrows()}
    with col_cpqr:
        st.write("Assumed CPQR ($) per Month:")
        if 'cpqr_df_cache' not in st.session_state or len(st.session_state.cpqr_df_cache) != proj_horizon:
            st.session_state.cpqr_df_cache = pd.DataFrame([{'Month': m.strftime('%Y-%m'), 'Assumed_CPQR': 120.0} for m in future_months_ui])
        edited_cpqr_df = st.data_editor(st.session_state.cpqr_df_cache, key='cpqr_editor', use_container_width=True, num_rows="fixed")
        proj_cpqr_dict = {pd.Period(row['Month'], 'M'): float(row['Assumed_CPQR']) for _, row in edited_cpqr_df.iterrows()}

    st.subheader("2. Funnel Conversion Rate Assumptions")
    rate_method = st.radio("Base Conversion Rates On:", ('Manual Input Below', 'Rolling Historical Average'), key='rate_method', horizontal=True)
    rolling_window = 0
    if rate_method == 'Rolling Historical Average':
        rolling_window = st.selectbox("Select Rolling Window:", [1, 3, 6, 999], index=1, format_func=lambda x: "Overall Average" if x==999 else f"{x}-Month", key='rolling_window')

    manual_rates = {}
    cols_rate = st.columns(3)
    with cols_rate[0]:
        manual_rates[f"{STAGE_PASSED_ONLINE_FORM} -> {STAGE_PRE_SCREENING_ACTIVITIES}"] = st.slider("POF -> PreScreen %", 0.0, 100.0, 95.0, key='cr_qps', step=0.1, format="%.1f%%") / 100.0
        manual_rates[f"{STAGE_PRE_SCREENING_ACTIVITIES} -> {STAGE_SENT_TO_SITE}"] = st.slider("PreScreen -> StS %", 0.0, 100.0, 20.0, key='cr_pssts', step=0.1, format="%.1f%%") / 100.0
    with cols_rate[1]:
        manual_rates[f"{STAGE_SENT_TO_SITE} -> {STAGE_APPOINTMENT_SCHEDULED}"] = st.slider("StS -> Appt %", 0.0, 100.0, 45.0, key='cr_sa', step=0.1, format="%.1f%%") / 100.0
        manual_rates[f"{STAGE_APPOINTMENT_SCHEDULED} -> {STAGE_SIGNED_ICF}"] = st.slider("Appt -> ICF %", 0.0, 100.0, 55.0, key='cr_ai', step=0.1, format="%.1f%%") / 100.0
    with cols_rate[2]:
        manual_rates[f"{STAGE_SIGNED_ICF} -> {STAGE_ENROLLED}"] = st.slider("ICF -> Enrolled %", 0.0, 100.0, 85.0, key='cr_ie', step=0.1, format="%.1f%%") / 100.0
    st.markdown("---")

    if st.button("🚀 Run Combined Forecast", key="run_forecast"):
        effective_rates, rates_method_desc = determine_effective_projection_rates(
            st.session_state.processed_df, st.session_state.ordered_stages, st.session_state.ts_col_map,
            rate_method, rolling_window, manual_rates, st.session_state.inter_stage_lags)
        
        projection_inputs = {
            'horizon': proj_horizon, 'spend_dict': proj_spend_dict, 'cpqr_dict': proj_cpqr_dict,
            'final_conv_rates': effective_rates, 'inter_stage_lags': st.session_state.inter_stage_lags,
            # These are not used in this app's display but are needed for the function signature
            'goal_icf': 9999, 'site_performance_data': pd.DataFrame(), 'icf_variation_percentage': 0
        }
        funnel_analysis_inputs = {'final_conv_rates': effective_rates}

        st.session_state.combined_forecast_data = calculate_combined_forecast(
            st.session_state.processed_df, st.session_state.ordered_stages, st.session_state.ts_col_map,
            st.session_state.inter_stage_lags, projection_inputs, funnel_analysis_inputs)
        st.session_state.rates_desc = rates_method_desc

    if 'combined_forecast_data' in st.session_state:
        results = st.session_state.combined_forecast_data
        combined_df = results['combined_df']
        
        st.header("3. Forecast Results")
        st.caption(f"Using: **{st.session_state.rates_desc}** Conversion Rates")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Projected ICFs (Combined)", f"{combined_df['Total Projected ICFs'].sum():,.0f}")
        col2.metric("from New Leads", f"{results['total_new_lead_icfs']:,.0f}")
        col3.metric("from Existing Pipeline (Total Yield)", f"{results['total_pipeline_icf_yield']:,.1f}")

        st.subheader("Combined Monthly Forecast Table")
        display_df = combined_df.copy()
        if isinstance(display_df.index, pd.DatetimeIndex): display_df.index = display_df.index.strftime('%Y-%m')
        display_df.index.name = "Month"
        st.dataframe(display_df[['ICFs from New Leads', 'ICFs from Pipeline', 'Total Projected ICFs', 'Enrollments from Pipeline']].style.format("{:,.0f}"))

        st.subheader("Cumulative Projections Over Time")
        chart_df = combined_df[['Cumulative ICFs', 'Cumulative Enrollments']].copy()
        st.line_chart(chart_df)
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

# --- 2. Core Data Processing and Helper Functions (Imported from original app) ---

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
    status_to_stage_map = {status: stage for stage, statuses in funnel_def.items() for status in statuses}
    
    all_events = []
    stage_hist = row.get(parsed_stage_history_col, [])
    status_hist = row.get(parsed_status_history_col, [])
    if stage_hist: all_events.extend([(name, dt) for name, dt in stage_hist if isinstance(name, str)])
    if status_hist: all_events.extend([(name, dt) for name, dt in status_hist if isinstance(name, str)])
    
    try:
        all_events.sort(key=lambda x: x[1] if pd.notna(x[1]) else datetime.min)
    except TypeError:
        pass

    for event_name, event_dt in all_events:
        if pd.isna(event_dt): continue
        event_stage = status_to_stage_map.get(event_name) or (event_name if event_name in ordered_stgs else None)
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
    df.rename(columns={"Referral Date": "Submitted On"}, inplace=True)
    df["Submitted On_DT"] = df[submitted_on_col].apply(lambda x: parse_datetime_with_timezone(str(x)))
    
    initial_rows = len(df)
    df.dropna(subset=["Submitted On_DT"], inplace=True)
    if (initial_rows - len(df)) > 0: st.warning(f"Dropped {initial_rows - len(df)} rows due to unparseable dates.")
    if df.empty:
        st.error("No valid data remaining after date parsing.")
        return None

    df["Submission_Month"] = df["Submitted On_DT"].dt.to_period('M')
    
    history_cols = {'Lead Stage History': 'Parsed_Stage_History', 'Lead Status History': 'Parsed_Status_History'}
    for col, parsed_col in history_cols.items():
        if col in df.columns:
            df[parsed_col] = df[col].astype(str).apply(parse_history_string)
        else:
            st.warning(f"History column '{col}' not found. Timestamps may be incomplete.")

    parsed_stage_hist = history_cols['Lead Stage History'] if 'Lead Stage History' in df.columns else None
    parsed_status_hist = history_cols['Lead Status History'] if 'Lead Status History' in df.columns else None

    if not parsed_stage_hist and not parsed_status_hist:
        st.error("Neither 'Lead Stage History' nor 'Lead Status History' found. Cannot determine stage progression.")
        return None

    timestamp_cols_df = df.apply(lambda row: get_stage_timestamps(row, parsed_stage_hist, parsed_status_hist, funnel_def, ordered_stages, ts_col_map), axis=1)
    df = pd.concat([df.drop(columns=[col for col in df.columns if col.startswith('TS_')], errors='ignore'), timestamp_cols_df], axis=1)
    
    for stage, ts_col in ts_col_map.items():
        if ts_col in df.columns:
            df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
            
    return df

def calculate_avg_lag_generic(df, col_from, col_to):
    if not all(c in df.columns for c in [col_from, col_to]): return np.nan
    valid_df = df.dropna(subset=[col_from, col_to])
    if valid_df.empty: return np.nan
    diff = pd.to_datetime(valid_df[col_to]) - pd.to_datetime(valid_df[col_from])
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

# --- 3. Core Forecasting Functions ---

@st.cache_data
def calculate_projections(_processed_df, ordered_stages, ts_col_map, projection_inputs):
    # This is a simplified version of the original function, focused on generating ICFs
    if _processed_df is None: return pd.DataFrame(), np.nan, "N/A", "N/A", pd.DataFrame(), "N/A"
    
    horizon = projection_inputs['horizon']
    future_spend = projection_inputs['spend_dict']
    assumed_cpqr = projection_inputs['cpqr_dict']
    conv_rates = projection_inputs['final_conv_rates']
    inter_stage_lags = projection_inputs.get('inter_stage_lags', {})

    # Calculate overall lag from Passed Online Form to Signed ICF for projection smearing
    pof_to_icf_path = [
        f"{STAGE_PASSED_ONLINE_FORM} -> {STAGE_PRE_SCREENING_ACTIVITIES}",
        f"{STAGE_PRE_SCREENING_ACTIVITIES} -> {STAGE_SENT_TO_SITE}",
        f"{STAGE_SENT_TO_SITE} -> {STAGE_APPOINTMENT_SCHEDULED}",
        f"{STAGE_APPOINTMENT_SCHEDULED} -> {STAGE_SIGNED_ICF}"
    ]
    total_lag = sum(inter_stage_lags.get(key, 30) for key in pof_to_icf_path) if inter_stage_lags else 120

    last_hist_month = _processed_df["Submission_Month"].max() if "Submission_Month" in _processed_df else pd.Period(datetime.now(), freq='M') - 1
    future_months = pd.period_range(start=last_hist_month + 1, periods=horizon, freq='M')
    
    # Create cohorts based on future spend
    cohorts = pd.DataFrame(index=future_months)
    cohorts['Ad_Spend'] = [future_spend.get(m, 0.0) for m in future_months]
    cohorts['CPQR'] = [assumed_cpqr.get(m, 120.0) for m in future_months]
    cohorts['New_QLs'] = (cohorts['Ad_Spend'] / cohorts['CPQR'].replace(0, np.nan)).fillna(0).round().astype(int)

    # Project cohorts through the funnel
    current_counts = cohorts['New_QLs']
    for stage_from, stage_to in zip(ordered_stages[:-1], ordered_stages[1:]):
        rate = conv_rates.get(f"{stage_from} -> {stage_to}", 0.0)
        current_counts *= rate
        if stage_to == STAGE_SIGNED_ICF:
            break
    cohorts['Generated_ICFs'] = current_counts.round().astype(int)

    # Distribute (smear) the generated ICFs into future landing months based on lag
    results = pd.DataFrame(0, index=future_months, columns=['Projected_ICF_Landed'])
    days_in_month = 30.4375
    for cohort_month, row in cohorts.iterrows():
        icfs_generated = row['Generated_ICFs']
        if icfs_generated == 0: continue
        
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

    results['Projected_ICF_Landed'] = results['Projected_ICF_Landed'].round().astype(int)
    # The other return values are not needed for this simplified app but are kept for signature consistency
    return results, total_lag, "N/A", "N/A", pd.DataFrame(), "Lag calculated from sum of steps."

@st.cache_data
def calculate_pipeline_projection(_processed_df, ordered_stages, ts_col_map, inter_stage_lags, conversion_rates):
    default_return = {'results_df': pd.DataFrame(), 'total_icf_yield': 0, 'total_enroll_yield': 0}
    if _processed_df is None or _processed_df.empty: return default_return

    # Filter for leads that are not yet in a terminal state
    terminal_ts_cols = [ts_col_map.get(s) for s in [STAGE_ENROLLED, STAGE_SCREEN_FAILED, STAGE_LOST] if s in ts_col_map]
    in_flight_df = _processed_df.copy()
    for ts_col in terminal_ts_cols:
        if ts_col in in_flight_df.columns:
            in_flight_df = in_flight_df[in_flight_df[ts_col].isna()]
    if in_flight_df.empty: return default_return

    # Determine the latest stage for each in-flight lead
    def get_current_stage(row, ordered_stages, ts_col_map):
        last_stage, last_ts = None, pd.NaT
        for stage in ordered_stages:
            if stage in [STAGE_ENROLLED, STAGE_SCREEN_FAILED, STAGE_LOST]: continue
            ts_col = ts_col_map.get(stage)
            if ts_col and ts_col in row and pd.notna(row[ts_col]):
                if pd.isna(last_ts) or row[ts_col] > last_ts:
                    last_ts, last_stage = row[ts_col], stage
        return last_stage, last_ts
    in_flight_df[['current_stage', 'current_stage_ts']] = in_flight_df.apply(lambda row: get_current_stage(row, ordered_stages, ts_col_map), axis=1, result_type='expand')
    in_flight_df.dropna(subset=['current_stage'], inplace=True)

    # Project forward from current stage
    projections = []
    for _, row in in_flight_df.iterrows():
        prob_to_icf, lag_to_icf = 1.0, 0.0
        start_index = ordered_stages.index(row['current_stage'])
        path_found = False
        for i in range(start_index, len(ordered_stages) - 1):
            from_stage, to_stage = ordered_stages[i], ordered_stages[i+1]
            rate_key = f"{from_stage} -> {to_stage}"
            lag_key = rate_key
            prob_to_icf *= conversion_rates.get(rate_key, 0.0)
            lag_to_icf += inter_stage_lags.get(lag_key, 0.0)
            if to_stage == STAGE_SIGNED_ICF:
                path_found = True
                break
        
        if path_found and prob_to_icf > 0:
            icf_date = row['current_stage_ts'] + pd.to_timedelta(lag_to_icf, unit='D')
            
            # Now project to enrollment
            prob_icf_to_enroll = conversion_rates.get(f"{STAGE_SIGNED_ICF} -> {STAGE_ENROLLED}", 0.0)
            lag_icf_to_enroll = inter_stage_lags.get(f"{STAGE_SIGNED_ICF} -> {STAGE_ENROLLED}", 0.0)
            
            enroll_prob = prob_to_icf * prob_icf_to_enroll
            enroll_date = icf_date + pd.to_timedelta(lag_icf_to_enroll, unit='D')
            
            projections.append({'icf_prob': prob_to_icf, 'icf_date': icf_date, 'enroll_prob': enroll_prob, 'enroll_date': enroll_date})

    if not projections: return default_return

    proj_df = pd.DataFrame(projections)
    proj_df['icf_month'] = proj_df['icf_date'].dt.to_period('M')
    proj_df['enroll_month'] = proj_df['enroll_date'].dt.to_period('M')

    icf_landed = proj_df.groupby('icf_month')['icf_prob'].sum().round().astype(int)
    enroll_landed = proj_df.groupby('enroll_month')['enroll_prob'].sum().round().astype(int)

    results_df = pd.DataFrame(index=pd.period_range(start=datetime.now(), periods=12, freq='M'))
    results_df['Projected_ICF_Landed'] = icf_landed
    results_df['Projected_Enrollments_Landed'] = enroll_landed
    results_df = results_df.fillna(0).astype(int)

    return {
        'results_df': results_df[results_df.index >= pd.Period(datetime.now(), 'M')],
        'total_icf_yield': proj_df['icf_prob'].sum(),
        'total_enroll_yield': proj_df['enroll_prob'].sum()
    }

@st.cache_data
def calculate_combined_forecast(_processed_df, ordered_stages, ts_col_map, inter_stage_lags, projection_inputs, funnel_analysis_inputs):
    # Part 1: Projections from Future Spend (New Leads)
    df_from_new_leads, _, _, _, _, _ = calculate_projections(_processed_df, ordered_stages, ts_col_map, projection_inputs)
    if not df_from_new_leads.empty:
        df_from_new_leads = df_from_new_leads.rename(columns={"Projected_ICF_Landed": "ICFs from New Leads"})[['ICFs from New Leads']]

    # Part 2: Projections from Existing Pipeline
    pipeline_results = calculate_pipeline_projection(
        _processed_df=_processed_df,
        ordered_stages=ordered_stages,
        ts_col_map=ts_col_map,
        inter_stage_lags=inter_stage_lags,
        conversion_rates=funnel_analysis_inputs['final_conv_rates']
    )
    df_from_pipeline = pipeline_results.get('results_df', pd.DataFrame())
    if not df_from_pipeline.empty:
        df_from_pipeline = df_from_pipeline.rename(columns={"Projected_ICF_Landed": "ICFs from Pipeline", "Projected_Enrollments_Landed": "Enrollments from Pipeline"})[['ICFs from Pipeline', 'Enrollments from Pipeline']]

    # Part 3: Combine DataFrames
    if isinstance(df_from_new_leads.index, pd.PeriodIndex): df_from_new_leads.index = df_from_new_leads.index.to_timestamp()
    if isinstance(df_from_pipeline.index, pd.PeriodIndex): df_from_pipeline.index = df_from_pipeline.index.to_timestamp()

    combined_df = df_from_new_leads.add(df_from_pipeline, fill_value=0).fillna(0).astype(int)

    # Part 4: Calculate Totals and Cumulative Sums
    for col in ['ICFs from New Leads', 'ICFs from Pipeline', 'Enrollments from Pipeline']:
        if col not in combined_df: combined_df[col] = 0
    
    combined_df['Total Projected ICFs'] = combined_df['ICFs from New Leads'] + combined_df['ICFs from Pipeline']
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

# --- Session State Initialization ---
if 'data_processed' not in st.session_state: st.session_state.data_processed = False
if 'processed_df' not in st.session_state: st.session_state.processed_df = None
if 'ordered_stages' not in st.session_state: st.session_state.ordered_stages = None
if 'ts_col_map' not in st.session_state: st.session_state.ts_col_map = None
if 'inter_stage_lags' not in st.session_state: st.session_state.inter_stage_lags = None

# --- Sidebar for File Uploads ---
with st.sidebar:
    st.header("⚙️ Setup")
    uploaded_referral_file = st.file_uploader("1. Upload Referral Data (CSV)", type=["csv"], key="referral_uploader")
    uploaded_funnel_def_file = st.file_uploader("2. Upload Funnel Definition (CSV or TSV)", type=["csv", "tsv"], key="funnel_uploader")

# --- Data Loading and Preprocessing ---
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
                st.rerun() # Rerun to update the UI now that data is loaded
        except Exception as e:
            st.error(f"Error loading referral data: {e}")
            
# --- Main Application Panel ---
if not st.session_state.data_processed:
    st.warning("Please upload both data files in the sidebar to begin.")
else:
    # --- Assumption Controls ---
    st.subheader("1. Future Spend & New Lead Assumptions")
    proj_horizon = st.number_input("Projection Horizon (Months)", min_value=1, max_value=48, value=12, step=1, key='combo_proj_horizon')
    
    _proj_start_month = st.session_state.processed_df["Submission_Month"].max() + 1 if "Submission_Month" in st.session_state.processed_df else pd.Period(datetime.now(), freq='M') + 1
    future_months_ui = pd.period_range(start=_proj_start_month, periods=proj_horizon, freq='M')
    
    col_spend, col_cpqr = st.columns(2)
    with col_spend:
        st.write("Future Monthly Ad Spend:")
        if 'combo_spend_df_cache' not in st.session_state or len(st.session_state.combo_spend_df_cache) != proj_horizon:
             st.session_state.combo_spend_df_cache = pd.DataFrame({'Month': future_months_ui.strftime('%Y-%m'), 'Planned_Spend': [20000.0] * proj_horizon})
        edited_spend_df = st.data_editor(st.session_state.combo_spend_df_cache, key='combo_spend_editor', use_container_width=True, num_rows="fixed")
        proj_spend_dict = {pd.Period(row['Month'], freq='M'): float(row['Planned_Spend']) for _, row in edited_spend_df.iterrows()}

    with col_cpqr:
        st.write("Assumed CPQR ($) per Month:")
        if 'combo_cpqr_df_cache' not in st.session_state or len(st.session_state.combo_cpqr_df_cache) != proj_horizon:
            st.session_state.combo_cpqr_df_cache = pd.DataFrame({'Month': future_months_ui.strftime('%Y-%m'), 'Assumed_CPQR': [120.0] * proj_horizon})
        edited_cpqr_df = st.data_editor(st.session_state.combo_cpqr_df_cache, key='combo_cpqr_editor', use_container_width=True, num_rows="fixed")
        proj_cpqr_dict = {pd.Period(row['Month'], freq='M'): float(row['Assumed_CPQR']) for _, row in edited_cpqr_df.iterrows()}

    st.subheader("2. Funnel Conversion Rate Assumptions")
    st.caption("These rates apply to both new leads and the existing pipeline.")
    
    manual_rates = {}
    cols_rate_combo = st.columns(3)
    with cols_rate_combo[0]:
        manual_rates[f"{STAGE_PASSED_ONLINE_FORM} -> {STAGE_PRE_SCREENING_ACTIVITIES}"] = st.slider("POF -> PreScreen %", 0.0, 100.0, 95.0, key='cr_qps', step=0.1, format="%.1f%%") / 100.0
        manual_rates[f"{STAGE_PRE_SCREENING_ACTIVITIES} -> {STAGE_SENT_TO_SITE}"] = st.slider("PreScreen -> StS %", 0.0, 100.0, 20.0, key='cr_pssts', step=0.1, format="%.1f%%") / 100.0
    with cols_rate_combo[1]:
        manual_rates[f"{STAGE_SENT_TO_SITE} -> {STAGE_APPOINTMENT_SCHEDULED}"] = st.slider("StS -> Appt %", 0.0, 100.0, 45.0, key='cr_sa', step=0.1, format="%.1f%%") / 100.0
        manual_rates[f"{STAGE_APPOINTMENT_SCHEDULED} -> {STAGE_SIGNED_ICF}"] = st.slider("Appt -> ICF %", 0.0, 100.0, 55.0, key='cr_ai', step=0.1, format="%.1f%%") / 100.0
    with cols_rate_combo[2]:
        manual_rates[f"{STAGE_SIGNED_ICF} -> {STAGE_ENROLLED}"] = st.slider("ICF -> Enrolled %", 0.0, 100.0, 85.0, key='cr_ie', step=0.1, format="%.1f%%") / 100.0
    
    st.markdown("---")

    # --- Calculation and Display ---
    if st.button("🚀 Run Combined Forecast", key="run_combined_forecast"):
        effective_rates = manual_rates # Simplified for this app, but could add rolling average logic back in if needed
        
        projection_run_inputs = {
            'horizon': proj_horizon,
            'spend_dict': proj_spend_dict,
            'cpqr_dict': proj_cpqr_dict,
            'final_conv_rates': effective_rates,
            'inter_stage_lags': st.session_state.inter_stage_lags,
            'goal_icf': 9999, # Not relevant
            'site_performance_data': pd.DataFrame(), # Not relevant
            'icf_variation_percentage': 0 # Not relevant
        }
        
        funnel_analysis_run_inputs = {'final_conv_rates': effective_rates}

        st.session_state.combined_forecast_data = calculate_combined_forecast(
            _processed_df=st.session_state.processed_df,
            ordered_stages=st.session_state.ordered_stages,
            ts_col_map=st.session_state.ts_col_map,
            inter_stage_lags=st.session_state.inter_stage_lags,
            projection_inputs=projection_run_inputs,
            funnel_analysis_inputs=funnel_analysis_run_inputs
        )
        st.session_state.rates_desc = "Manual Input"

    if 'combined_forecast_data' in st.session_state:
        results = st.session_state.combined_forecast_data
        combined_df = results['combined_df']
        
        st.header("3. Forecast Results")
        st.caption(f"Using: {st.session_state.rates_desc} Conversion Rates")

        total_icfs = combined_df['Total Projected ICFs'].sum()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Projected ICFs (Combined)", f"{total_icfs:,.0f}")
        col2.metric("from New Leads", f"{results['total_new_lead_icfs']:,.0f}")
        col3.metric("from Existing Pipeline", f"{results['total_pipeline_icf_yield']:,.1f}")

        st.subheader("Combined Monthly Forecast Table")
        display_df = combined_df[['ICFs from New Leads', 'ICFs from Pipeline', 'Total Projected ICFs', 'Enrollments from Pipeline']]
        display_df.index.name = "Month"
        st.dataframe(display_df.style.format("{:,.0f}"))

        st.subheader("Cumulative Projections Over Time")
        chart_df = combined_df[['Cumulative ICFs', 'Cumulative Enrollments']].copy()
        st.line_chart(chart_df)
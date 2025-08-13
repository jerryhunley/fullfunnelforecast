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
This tool provides a complete picture of recruitment by showing historical actuals leading into future projections.
- **Historical Data:** Shows actual ICFs and Enrollments that landed in past months.
- **Future Projections:** Combines the yield from the existing pipeline with projections from new ad spend.
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

# --- All data parsing and preprocessing functions are correct and remain unchanged ---
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
            if stage_name not in statuses: statuses.append(stage_name)
            parsed_funnel_definition[stage_name] = statuses
            clean_ts_name = f"TS_{stage_name.replace(' ', '_').replace('(', '').replace(')', '')}"
            ts_col_map[stage_name] = clean_ts_name
        if not parsed_ordered_stages: return None, None, None
        return parsed_funnel_definition, parsed_ordered_stages, ts_col_map
    except Exception as e:
        st.error(f"Error parsing Funnel Definition file: {e}"); return None, None, None

def parse_datetime_with_timezone(dt_str):
    if pd.isna(dt_str): return pd.NaT
    dt_str_cleaned = str(dt_str).strip(); tz_pattern = r'\s+(?:EST|EDT|CST|CDT|MST|MDT|PST|PDT)$'
    dt_str_no_tz = re.sub(tz_pattern, '', dt_str_cleaned)
    return pd.to_datetime(dt_str_no_tz, errors='coerce')

def parse_history_string(history_str):
    if pd.isna(history_str) or str(history_str).strip() == "": return []
    pattern = re.compile(r"([\w\s().'/:-]+?):\s*(\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}(?:\s*[apAP][mM])?(?:\s+[A-Za-z]{3,}(?:T)?)?)")
    raw_lines = str(history_str).strip().split('\n'); parsed_events = []
    for line in raw_lines:
        line = line.strip()
        if not line: continue
        match = pattern.match(line)
        if match:
            name, dt_str = match.groups(); name = name.strip()
            dt_obj = parse_datetime_with_timezone(dt_str.strip())
            if name and pd.notna(dt_obj):
                try: parsed_events.append((name, dt_obj.to_pydatetime()))
                except AttributeError: pass
    try: parsed_events.sort(key=lambda x: x[1] if pd.notna(x[1]) else datetime.min)
    except TypeError: pass
    return parsed_events

def get_stage_timestamps(row, parsed_stage_history_col, parsed_status_history_col, funnel_def, ordered_stgs, ts_col_mapping):
    timestamps = {ts_col_mapping[stage]: pd.NaT for stage in ordered_stgs}
    status_to_stage_map = {status: stage for stage, statuses in funnel_def.items() for status in statuses} if funnel_def else {}
    all_events = []
    stage_hist = row.get(parsed_stage_history_col, [])
    status_hist = row.get(parsed_status_history_col, [])
    if stage_hist: all_events.extend([(name, dt) for name, dt in stage_hist if isinstance(name, str)])
    if status_hist: all_events.extend([(name, dt) for name, dt in status_hist if isinstance(name, str)])
    try: all_events.sort(key=lambda x: x[1] if pd.notna(x[1]) else datetime.min)
    except TypeError: pass
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
        st.error("Data must contain either a 'Submitted On' or 'Referral Date' column."); return None
    if "Referral Date" in df.columns: df.rename(columns={"Referral Date": "Submitted On"}, inplace=True)
    df["Submitted On_DT"] = df[submitted_on_col].apply(lambda x: parse_datetime_with_timezone(str(x)))
    initial_rows = len(df)
    df.dropna(subset=["Submitted On_DT"], inplace=True)
    if initial_rows - len(df) > 0: st.warning(f"Dropped {initial_rows - len(df)} rows due to unparseable dates.")
    if df.empty: st.error("No valid data remaining after date parsing."); return None
    df["Submission_Month"] = df["Submitted On_DT"].dt.to_period('M')
    parsed_cols = {}
    for col_name in ['Lead Stage History', 'Lead Status History']:
        if col_name in df.columns:
            parsed_col_name = f"Parsed_{col_name.replace(' ', '_')}"
            df[parsed_col_name] = df[col_name].astype(str).apply(parse_history_string)
            parsed_cols[col_name] = parsed_col_name
    if not parsed_cols: st.error("Neither 'Lead Stage History' nor 'Lead Status History' found."); return None
    timestamp_cols_df = df.apply(lambda row: get_stage_timestamps(row, parsed_cols.get('Lead Stage History'), parsed_cols.get('Lead Status History'), funnel_def, ordered_stages, ts_col_map), axis=1)
    df = pd.concat([df.drop(columns=[c for c in df.columns if c.startswith('TS_')], errors='ignore'), timestamp_cols_df], axis=1)
    for ts_col in ts_col_map.values():
        if ts_col in df.columns: df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
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
        sf, st = ordered_stages[i], ordered_stages[i+1]
        tsf, tst = ts_col_map.get(sf), ts_col_map.get(st)
        if tsf and tst: lags[f"{sf} -> {st}"] = calculate_avg_lag_generic(_processed_df, tsf, tst)
    return lags

# --- 3. Core Forecasting and Calculation Functions ---

@st.cache_data
def calculate_historical_performance(processed_df, ts_col_map, num_months_history, end_month_period):
    """Calculates actual landed ICFs and Enrollments for past months."""
    if processed_df is None or processed_df.empty or num_months_history <= 0:
        return pd.DataFrame()
    historical_start_month = end_month_period - num_months_history
    historical_months_range = pd.period_range(start=historical_start_month, end=end_month_period - 1, freq='M')
    hist_df = pd.DataFrame(index=historical_months_range)
    icf_ts_col = ts_col_map.get(STAGE_SIGNED_ICF)
    if icf_ts_col and icf_ts_col in processed_df.columns:
        icf_data = processed_df.dropna(subset=[icf_ts_col])
        icf_counts = icf_data.groupby(icf_data[icf_ts_col].dt.to_period('M')).size()
        hist_df['Total Projected ICFs'] = hist_df.index.map(icf_counts)
    enroll_ts_col = ts_col_map.get(STAGE_ENROLLED)
    if enroll_ts_col and enroll_ts_col in processed_df.columns:
        enroll_data = processed_df.dropna(subset=[enroll_ts_col])
        enroll_counts = enroll_data.groupby(enroll_data[enroll_ts_col].dt.to_period('M')).size()
        hist_df['Enrollments from Pipeline'] = hist_df.index.map(enroll_counts)
    return hist_df.fillna(0).astype(int)

@st.cache_data
def determine_effective_projection_rates(_processed_df, ordered_stages, ts_col_map, rate_method, rolling_window, manual_rates, inter_stage_lags_for_maturity):
    """The full, robust function for determining conversion rates."""
    if rate_method == 'Manual Input Below' or _processed_df is None or _processed_df.empty:
        return manual_rates, "Manual Input"
    
    MATURITY_PERIODS_DAYS = {}
    for rate_key in manual_rates.keys():
        avg_lag = inter_stage_lags_for_maturity.get(rate_key)
        MATURITY_PERIODS_DAYS[rate_key] = max(round(1.5 * avg_lag), 20) if pd.notna(avg_lag) and avg_lag > 0 else 45
    
    try:
        hist_counts = _processed_df.groupby("Submission_Month").size().to_frame(name="Total_Submissions")
        for stage in ordered_stages:
            ts_col = ts_col_map.get(stage)
            if ts_col and ts_col in _processed_df.columns:
                reached_col = f"Reached_{stage.replace(' ', '_').replace('(', '').replace(')', '')}"
                counts = _processed_df.dropna(subset=[ts_col]).groupby('Submission_Month').size()
                hist_counts = hist_counts.join(counts.rename(reached_col), how='left')
        hist_counts = hist_counts.fillna(0)

        calculated_rates = {}
        for rate_key, manual_val in manual_rates.items():
            try: from_stage, to_stage = rate_key.split(" -> ")
            except ValueError: continue
            col_from = f"Reached_{from_stage.replace(' ', '_').replace('(', '').replace(')', '')}"
            col_to = f"Reached_{to_stage.replace(' ', '_').replace('(', '').replace(')', '')}"
            if col_from not in hist_counts.columns or col_to not in hist_counts.columns:
                calculated_rates[rate_key] = manual_val; continue
            
            maturity_days = MATURITY_PERIODS_DAYS.get(rate_key, 45)
            mature_hist_counts = hist_counts[hist_counts.index.to_timestamp() + pd.Timedelta(days=maturity_days) < pd.Timestamp(datetime.now())]
            if mature_hist_counts.empty:
                calculated_rates[rate_key] = manual_val; continue

            total_denom = mature_hist_counts[col_from].sum()
            overall_hist_rate = (mature_hist_counts[col_to].sum() / total_denom) if total_denom >= 20 else np.nan
            fallback_rate = overall_hist_rate if pd.notna(overall_hist_rate) else manual_val
            
            monthly_rates = (mature_hist_counts[col_to] / mature_hist_counts[col_from].replace(0, np.nan))
            monthly_rates = monthly_rates.where(mature_hist_counts[col_from] >= 5, fallback_rate).dropna()

            if not monthly_rates.empty:
                win_size = min(rolling_window, len(monthly_rates)) if rolling_window != 999 else len(monthly_rates)
                if win_size > 0:
                    rolling_avg = monthly_rates.rolling(window=win_size, min_periods=1).mean().iloc[-1]
                    calculated_rates[rate_key] = rolling_avg if pd.notna(rolling_avg) else fallback_rate
                else: calculated_rates[rate_key] = fallback_rate
            else: calculated_rates[rate_key] = fallback_rate
                
        desc = f"Rolling {rolling_window}-Month Avg (Matured)" if rolling_window != 999 else "Overall Historical Average (Matured)"
        return calculated_rates, desc
    except Exception as e:
        return manual_rates, f"Manual (Error in Rolling Calc: {e})"

@st.cache_data
def calculate_new_lead_projections(processed_df, ordered_stages, ts_col_map, proj_inputs):
    """The full, robust function for projecting outcomes from future ad spend."""
    if processed_df is None: return pd.DataFrame(columns=['ICFs from New Leads'])
    horizon, future_spend, assumed_cpqr, conv_rates, inter_stage_lags = (
        proj_inputs['horizon'], proj_inputs['spend_dict'], proj_inputs['cpqr_dict'],
        proj_inputs['final_conv_rates'], proj_inputs.get('inter_stage_lags', {}))
    
    projection_segments, total_lag_to_icf = [], 0
    try:
        icf_idx = ordered_stages.index(STAGE_SIGNED_ICF)
        for i in range(icf_idx):
            seg = (ordered_stages[i], ordered_stages[i+1])
            projection_segments.append(seg)
            total_lag_to_icf += inter_stage_lags.get(f"{seg[0]} -> {seg[1]}", 30)
    except (ValueError, IndexError): return pd.DataFrame(columns=['ICFs from New Leads'])

    last_hist_month = processed_df["Submission_Month"].max() if "Submission_Month" in processed_df else pd.Period(datetime.now(),'M')-1
    start_month = last_hist_month + 1
    future_months = pd.period_range(start=start_month, periods=horizon, freq='M')
    
    cohorts = pd.DataFrame(index=future_months)
    cohorts['Ad_Spend'] = [future_spend.get(m, 0) for m in future_months]
    cohorts['CPQR'] = [assumed_cpqr.get(m, 120) for m in future_months]
    cohorts['New_QLs'] = (cohorts['Ad_Spend'] / cohorts['CPQR'].replace(0, np.nan)).fillna(0).round().astype(int)

    counts = cohorts['New_QLs'].copy()
    for stage_from, stage_to in projection_segments:
        counts *= conv_rates.get(f"{stage_from} -> {stage_to}", 0.0)
    cohorts['Generated_ICFs'] = counts

    max_land = future_months[-1] + int(np.ceil(total_lag_to_icf / 30.5)) + 3
    results_idx = pd.period_range(start=start_month, end=max_land, freq='M')
    results = pd.DataFrame(0.0, index=results_idx, columns=['ICFs from New Leads'])

    for month, row in cohorts.iterrows():
        if row['Generated_ICFs'] <= 0: continue
        full_m, rem_d = divmod(total_lag_to_icf, 30.4375)
        frac_next = rem_d / 30.4375
        l1, l2 = month + int(full_m), month + int(full_m) + 1
        if l1 in results.index: results.loc[l1, 'ICFs from New Leads'] += row['Generated_ICFs'] * (1 - frac_next)
        if l2 in results.index: results.loc[l2, 'ICFs from New Leads'] += row['Generated_ICFs'] * frac_next
            
    return results

@st.cache_data
def calculate_pipeline_projection(_processed_df, ordered_stages, ts_col_map, inter_stage_lags, conv_rates):
    """The working function to calculate yield from the existing pipeline."""
    default_return = {'results_df': pd.DataFrame(), 'total_icf_yield': 0, 'total_enroll_yield': 0}
    if _processed_df is None or _processed_df.empty: return default_return

    term_stages = [s for s in [STAGE_ENROLLED, STAGE_SCREEN_FAILED, STAGE_LOST] if s in ts_col_map]
    term_ts_cols = [ts_col_map.get(s) for s in term_stages]
    in_flight = _processed_df.copy()
    for ts_col in term_ts_cols:
        if ts_col in in_flight.columns: in_flight = in_flight[in_flight[ts_col].isna()]
    if in_flight.empty: return default_return

    def get_current_stage(row, stages, ts_map):
        last_stage, last_ts = None, pd.NaT
        for stage in stages:
            if stage in term_stages: continue
            ts_col = ts_map.get(stage)
            if ts_col and ts_col in row and pd.notna(row[ts_col]):
                if pd.isna(last_ts) or row[ts_col] > last_ts:
                    last_ts, last_stage = row[ts_col], stage
        return last_stage
    
    in_flight['current_stage'] = in_flight.apply(lambda row: get_current_stage(row, ordered_stages, ts_col_map), axis=1)
    in_flight['current_stage_ts'] = in_flight.apply(lambda row: row.get(ts_col_map.get(row['current_stage'])) if row['current_stage'] else pd.NaT, axis=1)
    in_flight.dropna(subset=['current_stage', 'current_stage_ts'], inplace=True)

    projections = []
    icf_idx = ordered_stages.index(STAGE_SIGNED_ICF) if STAGE_SIGNED_ICF in ordered_stages else -1
    if icf_idx == -1: return default_return

    for _, row in in_flight.iterrows():
        prob, lag = 1.0, 0.0
        start_idx = ordered_stages.index(row['current_stage'])
        for i in range(start_idx, icf_idx):
            f, t = ordered_stages[i], ordered_stages[i+1]
            prob *= conv_rates.get(f"{f} -> {t}", 0)
            lag += inter_stage_lags.get(f"{f} -> {t}", 0)
        
        if prob > 0:
            icf_date = row['current_stage_ts'] + pd.to_timedelta(lag, unit='D')
            enroll_prob = prob * conv_rates.get(f"{STAGE_SIGNED_ICF} -> {STAGE_ENROLLED}", 0)
            enroll_lag = lag + inter_stage_lags.get(f"{STAGE_SIGNED_ICF} -> {STAGE_ENROLLED}", 0)
            enroll_date = row['current_stage_ts'] + pd.to_timedelta(enroll_lag, unit='D')
            projections.append({'icf_prob': prob, 'icf_date': icf_date, 'enroll_prob': enroll_prob, 'enroll_date': enroll_date})
    
    if not projections: return default_return
    proj_df = pd.DataFrame(projections)
    
    max_date = proj_df[['icf_date', 'enroll_date']].max().max() if not proj_df.empty else datetime.now()
    start_p, end_p = pd.Period(datetime.now(), 'M'), max(pd.Period(max_date, 'M'), pd.Period(datetime.now(), 'M') + 24)
    res_idx = pd.period_range(start=start_p, end=end_p, freq='M')
    res_df = pd.DataFrame(0.0, index=res_idx, columns=['ICFs from Pipeline', 'Enrollments from Pipeline'])

    res_df['ICFs from Pipeline'] = res_df.index.map(proj_df.groupby(proj_df['icf_date'].dt.to_period('M'))['icf_prob'].sum())
    res_df['Enrollments from Pipeline'] = res_df.index.map(proj_df.groupby(proj_df['enroll_date'].dt.to_period('M'))['enroll_prob'].sum())
    
    return {'results_df': res_df.fillna(0), 'total_icf_yield': proj_df['icf_prob'].sum(), 'total_enroll_yield': proj_df['enroll_prob'].sum()}

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
                st.session_state.ordered_stages, st.session_state.ts_col_map = ordered_stages, ts_col_map
                st.session_state.inter_stage_lags = calculate_overall_inter_stage_lags(st.session_state.processed_df, ordered_stages, ts_col_map)
                st.session_state.data_processed = True
                st.success("Data loaded and processed successfully!")
                st.rerun()
        except Exception as e: st.error(f"Error loading referral data: {e}")

if not st.session_state.get('data_processed', False):
    st.warning("Please upload both data files in the sidebar to begin.")
else:
    st.subheader("1. Timeline Assumptions")
    col_hist, col_fut = st.columns(2)
    with col_hist:
        history_months = st.number_input("Months of History to Display", 0, 24, 6, 1, key='hist_months')
    with col_fut:
        proj_horizon = st.number_input("Months of Future Projection", 1, 48, 12, 1, key='proj_horizon')

    st.subheader("2. Future Spend & New Lead Assumptions")
    last_hist_month = st.session_state.processed_df["Submission_Month"].max()
    _proj_start_month = last_hist_month + 1 if pd.notna(last_hist_month) else pd.Period(datetime.now(), 'M')
    future_months_ui = pd.period_range(start=_proj_start_month, periods=proj_horizon, freq='M')

    col_spend, col_cpqr = st.columns(2)
    with col_spend:
        st.write("Future Monthly Ad Spend:")
        if 'spend_df_cache' not in st.session_state or len(st.session_state.spend_df_cache) != proj_horizon:
            st.session_state.spend_df_cache = pd.DataFrame([{'Month':m.strftime('%Y-%m'),'Planned_Spend':20000.0} for m in future_months_ui])
        edited_spend_df = st.data_editor(st.session_state.spend_df_cache, key='spend_editor', use_container_width=True, num_rows="fixed")
        proj_spend_dict = {pd.Period(r['Month'],'M'):float(r['Planned_Spend']) for _, r in edited_spend_df.iterrows()}
    with col_cpqr:
        st.write("Assumed CPQR ($) per Month:")
        if 'cpqr_df_cache' not in st.session_state or len(st.session_state.cpqr_df_cache) != proj_horizon:
            st.session_state.cpqr_df_cache = pd.DataFrame([{'Month':m.strftime('%Y-%m'),'Assumed_CPQR':120.0} for m in future_months_ui])
        edited_cpqr_df = st.data_editor(st.session_state.cpqr_df_cache, key='cpqr_editor', use_container_width=True, num_rows="fixed")
        proj_cpqr_dict = {pd.Period(r['Month'],'M'):float(r['Assumed_CPQR']) for _, r in edited_cpqr_df.iterrows()}

    st.subheader("3. Funnel Conversion Rate Assumptions")
    rate_method = st.radio("Base Conversion Rates On:", ('Manual Input Below', 'Rolling Historical Average'), key='rate_method', horizontal=True)
    rolling_window = 0
    if rate_method == 'Rolling Historical Average':
        rolling_window = st.selectbox("Select Rolling Window:",[1,3,6,999],index=1,format_func=lambda x:"Overall" if x==999 else f"{x}-Mo",key='rolling_win')

    manual_rates = {}
    cols_rate = st.columns(3)
    with cols_rate[0]:
        manual_rates[f"{STAGE_PASSED_ONLINE_FORM} -> {STAGE_PRE_SCREENING_ACTIVITIES}"] = st.slider("POF->PreScreen %",0.0,100.0,95.0,key='cr_1',format="%.1f%%")/100
        manual_rates[f"{STAGE_PRE_SCREENING_ACTIVITIES} -> {STAGE_SENT_TO_SITE}"] = st.slider("PreScreen->StS %",0.0,100.0,20.0,key='cr_2',format="%.1f%%")/100
    with cols_rate[1]:
        manual_rates[f"{STAGE_SENT_TO_SITE} -> {STAGE_APPOINTMENT_SCHEDULED}"]=st.slider("StS->Appt %",0.0,100.0,45.0,key='cr_3',format="%.1f%%")/100
        manual_rates[f"{STAGE_APPOINTMENT_SCHEDULED} -> {STAGE_SIGNED_ICF}"]=st.slider("Appt->ICF %",0.0,100.0,55.0,key='cr_4',format="%.1f%%")/100
    with cols_rate[2]:
        manual_rates[f"{STAGE_SIGNED_ICF} -> {STAGE_ENROLLED}"]=st.slider("ICF->Enrolled %",0.0,100.0,85.0,key='cr_5',format="%.1f%%")/100
    st.markdown("---")

    if st.button("🚀 Run Combined Forecast", key="run_forecast"):
        effective_rates, rates_method_desc = determine_effective_projection_rates(st.session_state.processed_df, st.session_state.ordered_stages, st.session_state.ts_col_map,rate_method, rolling_window, manual_rates, st.session_state.inter_stage_lags)
        st.session_state.rates_desc = rates_method_desc
        
        historical_df = calculate_historical_performance(st.session_state.processed_df, st.session_state.ts_col_map, history_months, _proj_start_month)
        
        proj_inputs = {'horizon': proj_horizon, 'spend_dict': proj_spend_dict, 'cpqr_dict': proj_cpqr_dict, 'final_conv_rates': effective_rates, 'inter_stage_lags': st.session_state.inter_stage_lags}
        
        # --- Run the two independent projections ---
        new_leads_df = calculate_new_lead_projections(st.session_state.processed_df, st.session_state.ordered_stages, st.session_state.ts_col_map, proj_inputs)
        pipeline_results = calculate_pipeline_projection(st.session_state.processed_df, st.session_state.ordered_stages, st.session_state.ts_col_map, st.session_state.inter_stage_lags, effective_rates)
        pipeline_df = pipeline_results['results_df']
        
        # --- Combine the two future projections ---
        future_df = pd.concat([new_leads_df, pipeline_df], axis=1).fillna(0)
        future_df['Total Projected ICFs'] = future_df['ICFs from New Leads'] + future_df['ICFs from Pipeline']
        
        # --- Combine historical and the unified future projection ---
        full_timeline_df = pd.concat([historical_df, future_df]).fillna(0)
        
        all_cols = ['ICFs from New Leads', 'ICFs from Pipeline', 'Total Projected ICFs', 'Enrollments from Pipeline']
        for col in all_cols:
            if col not in full_timeline_df.columns: full_timeline_df[col] = 0
        
        float_cols = full_timeline_df.select_dtypes(include=['float']).columns
        full_timeline_df[float_cols] = full_timeline_df[float_cols].round(0)
        full_timeline_df = full_timeline_df[all_cols].astype(int)
        
        full_timeline_df['Cumulative ICFs'] = full_timeline_df['Total Projected ICFs'].cumsum()
        full_timeline_df['Cumulative Enrollments'] = full_timeline_df['Enrollments from Pipeline'].cumsum()
        
        st.session_state.full_forecast_data = full_timeline_df
        st.session_state.summary_metrics = {
            'total_new_lead_icfs': new_leads_df['ICFs from New Leads'].sum(),
            'total_pipeline_icf_yield': pipeline_results['total_icf_yield']
        }

    if 'full_forecast_data' in st.session_state:
        results_df = st.session_state.full_forecast_data
        summary = st.session_state.summary_metrics
        
        st.header("4. Forecast Results")
        st.caption(f"Using: **{st.session_state.rates_desc}** Conversion Rates")

        col1, col2, col3 = st.columns(3)
        future_icf_sum = results_df.loc[_proj_start_month:]['Total Projected ICFs'].sum()
        col1.metric("Total Projected ICFs (Future Only)", f"{future_icf_sum:,.0f}")
        col2.metric("from New Leads (Future)", f"{summary['total_new_lead_icfs']:,.0f}")
        col3.metric("from Existing Pipeline (Total Yield)", f"{summary['total_pipeline_icf_yield']:,.1f}")

        st.subheader("Historical & Projected Monthly View")
        display_df = results_df.copy()
        if isinstance(display_df.index, pd.PeriodIndex): display_df.index = display_df.index.strftime('%Y-%m')
        display_df.index.name = "Month"
        st.dataframe(display_df[['ICFs from New Leads','ICFs from Pipeline','Total Projected ICFs','Enrollments from Pipeline']].style.format("{:,.0f}"))

        st.subheader("Cumulative Projections Over Time (Historical & Future)")
        chart_df = results_df[['Cumulative ICFs', 'Cumulative Enrollments']].copy()
        if isinstance(chart_df.index, pd.PeriodIndex): chart_df.index = chart_df.index.to_timestamp()
        st.line_chart(chart_df)
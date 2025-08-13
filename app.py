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
            
        if not parsed_ordered_stages:
            st.error("Could not parse stages from Funnel Definition. Check file format.")
            return None, None, None
            
        return parsed_funnel_definition, parsed_ordered_stages, ts_col_map
    except Exception as e:
        st.error(f"Error parsing Funnel Definition file: {e}")
        return None, None, None

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
    
    all_events = []; stage_hist = row.get(parsed_stage_history_col, []); status_hist = row.get(parsed_status_history_col, [])
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
        st.error("Data must contain either a 'Submitted On' or 'Referral Date' column.")
        return None
    
    if "Referral Date" in df.columns: df.rename(columns={"Referral Date": "Submitted On"}, inplace=True)
    df["Submitted On_DT"] = df[submitted_on_col].apply(lambda x: parse_datetime_with_timezone(str(x)))
    
    initial_rows = len(df)
    df.dropna(subset=["Submitted On_DT"], inplace=True); rows_dropped = initial_rows - len(df)
    if rows_dropped > 0: st.warning(f"Dropped {rows_dropped} rows due to unparseable dates.")
    if df.empty:
        st.error("No valid data remaining after date parsing."); return None

    df["Submission_Month"] = df["Submitted On_DT"].dt.to_period('M')
    
    parsed_cols = {}
    for col_name in ['Lead Stage History', 'Lead Status History']:
        if col_name in df.columns:
            parsed_col_name = f"Parsed_{col_name.replace(' ', '_')}"
            df[parsed_col_name] = df[col_name].astype(str).apply(parse_history_string); parsed_cols[col_name] = parsed_col_name
        else: st.warning(f"History column '{col_name}' not found. Timestamps may be incomplete.")

    parsed_stage_hist_col, parsed_status_hist_col = parsed_cols.get('Lead Stage History'), parsed_cols.get('Lead Status History')
    if not parsed_stage_hist_col and not parsed_status_hist_col:
        st.error("Neither 'Lead Stage History' nor 'Lead Status History' found. Cannot determine stage progression.")
        return None

    timestamp_cols_df = df.apply(lambda row: get_stage_timestamps(row, parsed_stage_hist_col, parsed_status_hist_col, funnel_def, ordered_stages, ts_col_map), axis=1)
    df = pd.concat([df.drop(columns=[col for col in df.columns if col.startswith('TS_')], errors='ignore'), timestamp_cols_df], axis=1)
    
    for stage, ts_col in ts_col_map.items():
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
        stage_from, stage_to = ordered_stages[i], ordered_stages[i+1]
        ts_from, ts_to = ts_col_map.get(stage_from), ts_col_map.get(stage_to)
        if ts_from and ts_to: lags[f"{stage_from} -> {stage_to}"] = calculate_avg_lag_generic(_processed_df, ts_from, ts_to)
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
def determine_effective_projection_rates(
    _processed_df, ordered_stages, ts_col_map, rate_method,
    rolling_window, manual_rates, inter_stage_lags_for_maturity
):
    """
    This is the full, robust function for determining conversion rates,
    accounting for data maturity to provide more reliable projections.
    """
    MIN_DENOMINATOR_FOR_RATE_CALC = 5
    DEFAULT_MATURITY_DAYS = 45
    MATURITY_LAG_MULTIPLIER = 1.5
    MIN_EFFECTIVE_MATURITY_DAYS = 20
    MIN_TOTAL_DENOMINATOR_FOR_OVERALL_RATE = 20

    if rate_method == 'Manual Input Below' or _processed_df is None or _processed_df.empty:
        return manual_rates, "Manual Input"

    # --- Determine Maturity Periods for each rate based on its lag ---
    MATURITY_PERIODS_DAYS = {}
    if inter_stage_lags_for_maturity:
        for rate_key in manual_rates.keys():
            avg_lag = inter_stage_lags_for_maturity.get(rate_key)
            if pd.notna(avg_lag) and avg_lag > 0:
                calculated_maturity = round(MATURITY_LAG_MULTIPLIER * avg_lag)
                MATURITY_PERIODS_DAYS[rate_key] = max(calculated_maturity, MIN_EFFECTIVE_MATURITY_DAYS)
            else:
                MATURITY_PERIODS_DAYS[rate_key] = DEFAULT_MATURITY_DAYS
    else: # Fallback if lags aren't available
        for rate_key in manual_rates.keys():
            MATURITY_PERIODS_DAYS[rate_key] = DEFAULT_MATURITY_DAYS
    
    try:
        # --- Build historical counts by submission month cohort ---
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
            try:
                from_stage, to_stage = rate_key.split(" -> ")
            except ValueError: continue

            col_from = f"Reached_{from_stage.replace(' ', '_').replace('(', '').replace(')', '')}"
            col_to = f"Reached_{to_stage.replace(' ', '_').replace('(', '').replace(')', '')}"

            if col_from not in hist_counts.columns or col_to not in hist_counts.columns:
                calculated_rates[rate_key] = manual_val
                continue

            # --- Apply Maturity Logic ---
            maturity_days = MATURITY_PERIODS_DAYS.get(rate_key, DEFAULT_MATURITY_DAYS)
            mature_months_mask = hist_counts.index.to_timestamp() + pd.Timedelta(days=maturity_days) < pd.Timestamp(datetime.now())
            mature_hist_counts = hist_counts[mature_months_mask]
            
            if mature_hist_counts.empty:
                calculated_rates[rate_key] = manual_val
                continue
                
            # Use overall historical rate of MATURE data as a fallback
            total_numerator = mature_hist_counts[col_to].sum()
            total_denominator = mature_hist_counts[col_from].sum()
            overall_hist_rate = (total_numerator / total_denominator) if total_denominator >= MIN_TOTAL_DENOMINATOR_FOR_OVERALL_RATE else np.nan

            # If a month has a small denominator, substitute with overall historical or manual rate
            monthly_numerators = mature_hist_counts[col_to]
            monthly_denominators = mature_hist_counts[col_from]
            
            # Default to the most stable rate (overall historical or manual)
            fallback_rate = overall_hist_rate if pd.notna(overall_hist_rate) else manual_val
            
            monthly_rates = (monthly_numerators / monthly_denominators.replace(0, np.nan))
            # Substitute where denominator is too small
            monthly_rates = monthly_rates.where(monthly_denominators >= MIN_DENOMINATOR_FOR_RATE_CALC, fallback_rate)
            monthly_rates.dropna(inplace=True)

            if not monthly_rates.empty:
                win_size = min(rolling_window, len(monthly_rates)) if rolling_window != 999 else len(monthly_rates)
                if win_size > 0:
                    rolling_avg = monthly_rates.rolling(window=win_size, min_periods=1).mean().iloc[-1]
                    calculated_rates[rate_key] = rolling_avg if pd.notna(rolling_avg) else manual_val
                else:
                    calculated_rates[rate_key] = manual_val
            else:
                calculated_rates[rate_key] = manual_val
                
        desc = f"Rolling {rolling_window}-Month Avg (Matured)" if rolling_window != 999 else "Overall Historical Average (Matured)"
        return calculated_rates, desc
    except Exception:
        return manual_rates, "Manual (Error in Rolling Calc)"

@st.cache_data
def calculate_projections_from_new_leads(_processed_df, ordered_stages, ts_col_map, proj_inputs):
    """Calculates ICFs generated from NEW leads via future spend."""
    if _processed_df is None: return pd.DataFrame()
    
    horizon, future_spend, assumed_cpqr, conv_rates, inter_stage_lags = (
        proj_inputs['horizon'], proj_inputs['spend_dict'], proj_inputs['cpqr_dict'],
        proj_inputs['final_conv_rates'], proj_inputs.get('inter_stage_lags', {}))

    total_lag_to_icf = 0
    try:
        icf_idx = ordered_stages.index(STAGE_SIGNED_ICF)
        for i in range(icf_idx):
            lag_key = f"{ordered_stages[i]} -> {ordered_stages[i+1]}"
            total_lag_to_icf += inter_stage_lags.get(lag_key, 30)
    except (ValueError, IndexError): total_lag_to_icf = 90

    last_hist = _processed_df["Submission_Month"].max() if "Submission_Month" in _processed_df else pd.Period(datetime.now(),'M')-1
    start_month = last_hist + 1
    future_months = pd.period_range(start=start_month, periods=horizon, freq='M')
    
    cohorts = pd.DataFrame(index=future_months)
    cohorts['Ad_Spend'] = [future_spend.get(m, 0) for m in future_months]
    cohorts['CPQR'] = [assumed_cpqr.get(m, 120) for m in future_months]
    cohorts['New_QLs'] = (cohorts['Ad_Spend']/cohorts['CPQR'].replace(0,np.nan)).fillna(0).round().astype(int)

    counts = cohorts['New_QLs'].copy()
    if 'icf_idx' in locals() and icf_idx > 0:
        for i in range(icf_idx):
            rate_key = f"{ordered_stages[i]} -> {ordered_stages[i+1]}"
            counts *= conv_rates.get(rate_key, 0)
    cohorts['Generated_ICFs'] = counts

    max_land_month = future_months[-1] + int(np.ceil(total_lag_to_icf / 30.5)) + 2
    results_idx = pd.period_range(start=start_month, end=max_land_month, freq='M')
    results = pd.DataFrame(0.0, index=results_idx, columns=['ICFs from New Leads'])
    
    for month, row in cohorts.iterrows():
        if row['Generated_ICFs'] <= 0: continue
        full_lag_months, rem_days = divmod(total_lag_to_icf, 30.4375)
        frac_for_next_month = rem_days / 30.4375
        
        land_month_1 = month + int(full_lag_months)
        land_month_2 = month + int(full_lag_months) + 1
        if land_month_1 in results.index:
            results.loc[land_month_1, 'ICFs from New Leads'] += row['Generated_ICFs'] * (1 - frac_for_next_month)
        if land_month_2 in results.index:
            results.loc[land_month_2, 'ICFs from New Leads'] += row['Generated_ICFs'] * frac_for_next_month
            
    return results.reindex(future_months).fillna(0)

@st.cache_data
def calculate_pipeline_projection(_processed_df, ordered_stages, ts_col_map, inter_stage_lags, conv_rates):
    """Calculates future ICFs and Enrollments from EXISTING in-flight leads."""
    if _processed_df is None: return {'results_df': pd.DataFrame(), 'total_icf_yield': 0, 'total_enroll_yield': 0}

    term_ts = [ts_col_map.get(s) for s in [STAGE_ENROLLED, STAGE_SCREEN_FAILED, STAGE_LOST] if s in ts_col_map]
    in_flight = _processed_df.copy()
    for ts in term_ts:
        if ts in in_flight.columns: in_flight = in_flight[in_flight[ts].isna()]
    if in_flight.empty: return {'results_df': pd.DataFrame(), 'total_icf_yield': 0, 'total_enroll_yield': 0}

    def get_curr_stage(row, stages, ts_map):
        last_s, last_t = None, pd.NaT
        for s in stages:
            if s in [STAGE_ENROLLED, STAGE_SCREEN_FAILED, STAGE_LOST]: continue
            ts_col = ts_map.get(s)
            t = row.get(ts_col) if ts_col else None
            if ts_col and pd.notna(t) and (pd.isna(last_t) or t > last_t):
                last_s, last_t = s, t
        return last_s, last_t
    in_flight[['curr_s', 'curr_t']] = in_flight.apply(lambda r: get_curr_stage(r,ordered_stages,ts_col_map), axis=1, result_type='expand')
    in_flight.dropna(subset=['curr_s'], inplace=True)

    projs = []
    icf_idx = ordered_stages.index(STAGE_SIGNED_ICF) if STAGE_SIGNED_ICF in ordered_stages else -1
    if icf_idx == -1: return {'results_df': pd.DataFrame(), 'total_icf_yield': 0, 'total_enroll_yield': 0}

    for _, row in in_flight.iterrows():
        prob_to_icf, lag_to_icf = 1.0, 0.0
        start_idx = ordered_stages.index(row['curr_s'])
        
        path_found = False
        for i in range(start_idx, len(ordered_stages)-1):
            f, t = ordered_stages[i], ordered_stages[i+1]
            prob_to_icf *= conv_rates.get(f"{f} -> {t}", 0)
            lag_to_icf += inter_stage_lags.get(f"{f} -> {t}", 0)
            if t == STAGE_SIGNED_ICF:
                path_found = True
                break
        
        if path_found and prob_to_icf > 0 and pd.notna(row['curr_t']):
            icf_date = row['curr_t'] + pd.to_timedelta(lag_to_icf, unit='D')
            prob_to_enroll = prob_to_icf * conv_rates.get(f"{STAGE_SIGNED_ICF} -> {STAGE_ENROLLED}",0)
            lag_to_enroll = inter_stage_lags.get(f"{STAGE_SIGNED_ICF} -> {STAGE_ENROLLED}",0)
            enroll_date = icf_date + pd.to_timedelta(lag_to_enroll, unit='D') if pd.notna(icf_date) else pd.NaT
            projs.append({'icf_prob':prob_to_icf, 'icf_date':icf_date, 'enroll_prob':prob_to_enroll, 'enroll_date':enroll_date})
    
    if not projs: return {'results_df': pd.DataFrame(), 'total_icf_yield': 0, 'total_enroll_yield': 0}

    proj_df = pd.DataFrame(projs)
    max_d = proj_df[['icf_date', 'enroll_date']].max().max() if not proj_df.empty else datetime.now()
    start_p, end_p = pd.Period(datetime.now(),'M'), max(pd.Period(max_d,'M'), pd.Period(datetime.now(),'M')+24)
    res_idx = pd.period_range(start=start_p, end=end_p, freq='M')
    res_df = pd.DataFrame(0.0, index=res_idx, columns=['ICFs from Pipeline', 'Enrollments from Pipeline'])

    res_df['ICFs from Pipeline'] = res_df.index.map(proj_df.groupby(proj_df['icf_date'].dt.to_period('M'))['icf_prob'].sum())
    res_df['Enrollments from Pipeline'] = res_df.index.map(proj_df.groupby(proj_df['enroll_date'].dt.to_period('M'))['enroll_prob'].sum())
    
    return {'results_df':res_df.fillna(0),'total_icf_yield':proj_df['icf_prob'].sum(),'total_enroll_yield':proj_df['enroll_prob'].sum()}

def calculate_combined_forecast(processed_df, ordered_stages, ts_col_map, inter_stage_lags, proj_inputs, funnel_conv_rates):
    """Orchestrates the calculation of future projections from both new and existing leads."""
    new_leads_df = calculate_projections_from_new_leads(
        processed_df, ordered_stages, ts_col_map, proj_inputs
    )
    pipeline_results = calculate_pipeline_projection(
        processed_df, ordered_stages, ts_col_map, inter_stage_lags, funnel_conv_rates
    )
    pipeline_df = pipeline_results['results_df']
    combined_df = pd.concat([new_leads_df, pipeline_df], axis=1).fillna(0)
    
    if 'ICFs from New Leads' in combined_df.columns and 'ICFs from Pipeline' in combined_df.columns:
        combined_df['Total Projected ICFs'] = combined_df['ICFs from New Leads'] + combined_df['ICFs from Pipeline']
    else:
        combined_df['Total Projected ICFs'] = combined_df.get('ICFs from New Leads', 0) + combined_df.get('ICFs from Pipeline', 0)

    return {
        'combined_df': combined_df,
        'total_new_lead_icfs': new_leads_df['ICFs from New Leads'].sum(),
        'total_pipeline_icf_yield': pipeline_results['total_icf_yield'],
        'total_pipeline_enroll_yield': pipeline_results['total_enroll_yield']
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
                st.session_state.ordered_stages, st.session_state.ts_col_map = ordered_stages, ts_col_map
                st.session_state.inter_stage_lags = calculate_overall_inter_stage_lags(st.session_state.processed_df, ordered_stages, ts_col_map)
                st.session_state.data_processed = True
                st.success("Data loaded and processed successfully!")
                st.rerun()
        except Exception as e:
            st.error(f"Error loading referral data: {e}")

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
        effective_rates, rates_method_desc = determine_effective_projection_rates(
            st.session_state.processed_df, st.session_state.ordered_stages, st.session_state.ts_col_map,
            rate_method, rolling_window, manual_rates, st.session_state.inter_stage_lags)
        st.session_state.rates_desc = rates_method_desc
        
        historical_df = calculate_historical_performance(
            st.session_state.processed_df, st.session_state.ts_col_map,
            history_months, _proj_start_month)
        
        proj_inputs = {
            'horizon': proj_horizon, 'spend_dict': proj_spend_dict, 'cpqr_dict': proj_cpqr_dict,
            'final_conv_rates': effective_rates, 'inter_stage_lags': st.session_state.inter_stage_lags
        }
        future_results = calculate_combined_forecast(
            st.session_state.processed_df, st.session_state.ordered_stages, st.session_state.ts_col_map,
            st.session_state.inter_stage_lags, proj_inputs, effective_rates)
        future_df = future_results['combined_df'].reindex(future_months_ui).fillna(0)
        st.session_state.future_results_cache = future_results

        full_timeline_df = pd.concat([historical_df, future_df]).fillna(0)
        
        all_cols = ['ICFs from New Leads', 'ICFs from Pipeline', 'Total Projected ICFs', 'Enrollments from Pipeline']
        for col in all_cols:
            if col not in full_timeline_df.columns:
                full_timeline_df[col] = 0
        
        float_cols = full_timeline_df.select_dtypes(include=['float']).columns
        full_timeline_df[float_cols] = full_timeline_df[float_cols].round(0)
        full_timeline_df = full_timeline_df[all_cols].astype(int)
        
        full_timeline_df['Cumulative ICFs'] = full_timeline_df['Total Projected ICFs'].cumsum()
        full_timeline_df['Cumulative Enrollments'] = full_timeline_df['Enrollments from Pipeline'].cumsum()
        
        st.session_state.full_forecast_data = full_timeline_df

    if 'full_forecast_data' in st.session_state:
        results_df = st.session_state.full_forecast_data
        
        st.header("4. Forecast Results")
        st.caption(f"Using: **{st.session_state.rates_desc}** Conversion Rates")

        future_cache = st.session_state.future_results_cache
        col1, col2, col3 = st.columns(3)
        future_icf_sum = results_df.loc[_proj_start_month:]['Total Projected ICFs'].sum()
        col1.metric("Total Projected ICFs (Future Only)", f"{future_icf_sum:,.0f}")
        col2.metric("from New Leads (Future)", f"{future_cache['total_new_lead_icfs']:,.0f}")
        col3.metric("from Existing Pipeline (Total Yield)", f"{future_cache['total_pipeline_icf_yield']:,.1f}")

        st.subheader("Historical & Projected Monthly View")
        display_df = results_df.copy()
        display_df.index = display_df.index.strftime('%Y-%m')
        display_df.index.name = "Month"
        st.dataframe(display_df[['ICFs from New Leads','ICFs from Pipeline','Total Projected ICFs','Enrollments from Pipeline']].style.format("{:,.0f}"))

        st.subheader("Cumulative Projections Over Time (Historical & Future)")
        chart_df = results_df[['Cumulative ICFs', 'Cumulative Enrollments']].copy()
        if isinstance(chart_df.index, pd.PeriodIndex): chart_df.index = chart_df.index.to_timestamp()
        st.line_chart(chart_df)
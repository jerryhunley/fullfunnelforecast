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
        if not parsed_ordered_stages: st.error("Could not parse stages from Funnel Definition."); return None, None, None
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
    parsed_events = []
    for line in str(history_str).strip().split('\n'):
        match = pattern.match(line.strip())
        if match:
            name, dt_str = match.groups()
            dt_obj = parse_datetime_with_timezone(dt_str.strip())
            if name.strip() and pd.notna(dt_obj):
                try: parsed_events.append((name.strip(), dt_obj.to_pydatetime()))
                except AttributeError: pass
    try: parsed_events.sort(key=lambda x: x[1] if pd.notna(x[1]) else datetime.min)
    except TypeError: pass
    return parsed_events

def get_stage_timestamps(row, parsed_stage_history_col, parsed_status_history_col, funnel_def, ordered_stgs, ts_col_mapping):
    timestamps = {ts_col_mapping[stage]: pd.NaT for stage in ordered_stgs}
    status_to_stage_map = {status: stage for stage, statuses in funnel_def.items() for status in statuses}
    all_events = []
    if row.get(parsed_stage_history_col): all_events.extend(row.get(parsed_stage_history_col, []))
    if row.get(parsed_status_history_col): all_events.extend(row.get(parsed_status_history_col, []))
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
    df = _df_raw.copy()
    if "Referral Date" in df.columns: df.rename(columns={"Referral Date": "Submitted On"}, inplace=True)
    if "Submitted On" not in df.columns:
        st.error("Data must contain 'Submitted On' or 'Referral Date'."); return None
    df["Submitted On_DT"] = df["Submitted On"].apply(lambda x: parse_datetime_with_timezone(str(x)))
    df.dropna(subset=["Submitted On_DT"], inplace=True)
    if df.empty: st.error("No valid data after date parsing."); return None
    df["Submission_Month"] = df["Submitted On_DT"].dt.to_period('M')
    parsed_cols = {}
    for col in ['Lead Stage History', 'Lead Status History']:
        if col in df.columns:
            parsed_col_name = f"Parsed_{col.replace(' ', '_')}"
            df[parsed_col_name] = df[col].astype(str).apply(parse_history_string)
            parsed_cols[col] = parsed_col_name
    if not parsed_cols: st.error("No Lead History columns found."); return None
    timestamps_df = df.apply(lambda r: get_stage_timestamps(r, parsed_cols.get('Lead Stage History'), parsed_cols.get('Lead Status History'), funnel_def, ordered_stages, ts_col_map), axis=1)
    df = pd.concat([df.drop(columns=[c for c in df.columns if c.startswith('TS_')]), timestamps_df], axis=1)
    for ts_col in ts_col_map.values():
        if ts_col in df.columns: df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
    return df

def calculate_avg_lag_generic(df, col_from, col_to):
    if not all(c in df.columns and pd.api.types.is_datetime64_any_dtype(df[c]) for c in [col_from, col_to]): return np.nan
    diff = df.dropna(subset=[col_from, col_to])[[col_to, col_from]]
    time_diff = diff[col_to] - diff[col_from]
    return time_diff[time_diff >= pd.Timedelta(0)].mean().total_seconds() / (60*60*24)

@st.cache_data
def calculate_overall_inter_stage_lags(_processed_df, ordered_stages, ts_col_map):
    lags = {}
    for i in range(len(ordered_stages) - 1):
        sf, st = ordered_stages[i], ordered_stages[i+1]
        tsf, tst = ts_col_map.get(sf), ts_col_map.get(st)
        if tsf and tst: lags[f"{sf} -> {st}"] = calculate_avg_lag_generic(_processed_df, tsf, tst)
    return lags

# --- 3. Core Forecasting and Calculation Functions ---

@st.cache_data
def calculate_historical_performance(processed_df, ts_col_map, num_months_history, end_month_period):
    if processed_df is None or num_months_history <= 0: return pd.DataFrame()
    start_month = end_month_period - num_months_history
    hist_range = pd.period_range(start=start_month, end=end_month_period - 1, freq='M')
    hist_df = pd.DataFrame(index=hist_range)
    icf_ts, enroll_ts = ts_col_map.get(STAGE_SIGNED_ICF), ts_col_map.get(STAGE_ENROLLED)
    if icf_ts and icf_ts in processed_df.columns:
        hist_df['Total Projected ICFs'] = processed_df.dropna(subset=[icf_ts]).groupby(processed_df[icf_ts].dt.to_period('M')).size()
    if enroll_ts and enroll_ts in processed_df.columns:
        hist_df['Enrollments from Pipeline'] = processed_df.dropna(subset=[enroll_ts]).groupby(processed_df[enroll_ts].dt.to_period('M')).size()
    return hist_df.fillna(0).astype(int)

@st.cache_data
def determine_effective_projection_rates(_processed_df, ordered_stages, ts_col_map, rate_method, rolling_window, manual_rates, inter_stage_lags):
    if rate_method == 'Manual Input Below' or _processed_df is None or _processed_df.empty:
        return manual_rates, "Manual Input"
    
    maturity_days = {key: max(round(1.5 * inter_stage_lags.get(key, 30)), 20) for key in manual_rates.keys()}
    
    try:
        hist_counts = _processed_df.groupby("Submission_Month").size().to_frame(name="Total")
        for stage in ordered_stages:
            ts = ts_col_map.get(stage)
            if ts and ts in _processed_df.columns:
                col = f"Reached_{stage.replace(' ', '_').replace('(', '').replace(')', '')}"
                hist_counts[col] = _processed_df.dropna(subset=[ts]).groupby('Submission_Month').size()
        hist_counts = hist_counts.fillna(0)

        calculated_rates = {}
        for key, manual_val in manual_rates.items():
            try: from_s, to_s = key.split(" -> ")
            except ValueError: continue
            col_f = f"Reached_{from_s.replace(' ', '_').replace('(', '').replace(')', '')}"
            col_t = f"Reached_{to_s.replace(' ', '_').replace('(', '').replace(')', '')}"
            
            if col_f not in hist_counts.columns or col_t not in hist_counts.columns:
                calculated_rates[key] = manual_val; continue

            mature_counts = hist_counts[hist_counts.index.to_timestamp() + pd.Timedelta(days=maturity_days.get(key, 45)) < pd.Timestamp(datetime.now())]
            
            # --- CRITICAL FIX: The final fallback is always the user's manual input ---
            final_fallback_rate = manual_val
            
            if mature_counts.empty:
                calculated_rates[key] = final_fallback_rate; continue

            overall_rate = (mature_counts[col_t].sum() / mature_counts[col_f].sum()) if mature_counts[col_f].sum() >= 20 else np.nan
            
            monthly_rates = (mature_counts[col_t] / mature_counts[col_f].replace(0, np.nan))
            monthly_rates_with_fallback = monthly_rates.where(mature_counts[col_f] >= 5, overall_rate if pd.notna(overall_rate) else np.nan).dropna()

            if not monthly_rates_with_fallback.empty:
                win = min(rolling_window, len(monthly_rates_with_fallback)) if rolling_window != 999 else len(monthly_rates_with_fallback)
                if win > 0:
                    rolling_avg = monthly_rates_with_fallback.rolling(win, min_periods=1).mean().iloc[-1]
                    # Use the calculated rate only if it's a valid, positive number
                    if pd.notna(rolling_avg) and rolling_avg > 0:
                        calculated_rates[key] = rolling_avg
                    else:
                        calculated_rates[key] = final_fallback_rate
                else: calculated_rates[key] = final_fallback_rate
            else: calculated_rates[key] = final_fallback_rate
                
        desc = f"Rolling {rolling_window}-Month Avg (Matured)" if rolling_window != 999 else "Overall Historical Average (Matured)"
        return calculated_rates, desc
    except Exception as e:
        return manual_rates, f"Manual (Error in Rolling Calc: {e})"

@st.cache_data
def calculate_new_lead_projections(processed_df, ordered_stages, ts_col_map, proj_inputs):
    """The working function for projecting outcomes from future ad spend."""
    if processed_df is None: return pd.DataFrame(columns=['ICFs from New Leads'])
    horizon, future_spend, cpqr, rates, lags = proj_inputs.values()
    
    segments, total_lag = [], 0
    try:
        icf_idx = ordered_stages.index(STAGE_SIGNED_ICF)
        for i in range(icf_idx):
            seg = (ordered_stages[i], ordered_stages[i+1])
            segments.append(seg)
            total_lag += lags.get(f"{seg[0]} -> {seg[1]}", 30)
    except (ValueError, IndexError): return pd.DataFrame(columns=['ICFs from New Leads'])

    start_month = processed_df["Submission_Month"].max() + 1 if "Submission_Month" in processed_df else pd.Period(datetime.now(),'M')
    future_months = pd.period_range(start=start_month, periods=horizon, freq='M')
    
    cohorts = pd.DataFrame({'Ad_Spend': [future_spend.get(m, 0) for m in future_months], 'CPQR': [cpqr.get(m, 120) for m in future_months]}, index=future_months)
    cohorts['New_QLs'] = (cohorts['Ad_Spend'] / cohorts['CPQR'].replace(0, np.nan)).fillna(0).round()
    
    counts = cohorts['New_QLs'].copy()
    for stage_from, stage_to in segments:
        counts *= rates.get(f"{stage_from} -> {stage_to}", 0.0)
    cohorts['Generated_ICFs'] = counts

    max_land = future_months[-1] + int(np.ceil(total_lag / 30.5)) + 3
    results = pd.DataFrame(0.0, index=pd.period_range(start=start_month, end=max_land, freq='M'), columns=['ICFs from New Leads'])

    for month, row in cohorts.iterrows():
        if row['Generated_ICFs'] > 0:
            full_m, rem_d = divmod(total_lag, 30.4375)
            frac = rem_d / 30.4375
            l1, l2 = month + int(full_m), month + int(full_m) + 1
            if l1 in results.index: results.loc[l1] += row['Generated_ICFs'] * (1 - frac)
            if l2 in results.index: results.loc[l2] += row['Generated_ICFs'] * frac
            
    return results

@st.cache_data
def calculate_pipeline_projection(_processed_df, ordered_stages, ts_col_map, inter_stage_lags, conv_rates):
    """The working function to calculate yield from the existing pipeline."""
    default = {'results_df': pd.DataFrame(), 'total_icf_yield': 0, 'total_enroll_yield': 0}
    if _processed_df is None or _processed_df.empty: return default
    term_ts = [ts_col_map.get(s) for s in [STAGE_ENROLLED, STAGE_SCREEN_FAILED, STAGE_LOST] if s in ts_col_map]
    in_flight = _processed_df.copy()
    for ts in term_ts:
        if ts in in_flight.columns: in_flight = in_flight[in_flight[ts].isna()]
    if in_flight.empty: return default
    
    def get_curr(row, stgs, ts_map):
        last_s, last_t = None, pd.NaT
        for s in stgs:
            if s in [STAGE_ENROLLED, STAGE_SCREEN_FAILED, STAGE_LOST]: continue
            ts, t = ts_map.get(s), row.get(ts_map.get(s))
            if ts and pd.notna(t) and (pd.isna(last_t) or t > last_t): last_s, last_t = s, t
        return last_s, last_t
    
    in_flight[['curr_s', 'curr_t']] = in_flight.apply(lambda r: get_curr(r, ordered_stages, ts_col_map), axis=1, result_type='expand')
    in_flight.dropna(subset=['curr_s', 'curr_t'], inplace=True)

    projs = []
    icf_idx = ordered_stages.index(STAGE_SIGNED_ICF) if STAGE_SIGNED_ICF in ordered_stages else -1
    if icf_idx == -1: return default

    for _, row in in_flight.iterrows():
        prob, lag = 1.0, 0.0
        start = ordered_stages.index(row['curr_s'])
        for i in range(start, icf_idx):
            f, t = ordered_stages[i], ordered_stages[i+1]
            prob *= conv_rates.get(f"{f} -> {t}", 0)
            lag += inter_stage_lags.get(f"{f} -> {t}", 0)
        if prob > 0:
            icf_d = row['curr_t'] + pd.to_timedelta(lag, unit='D')
            enr_p = prob * conv_rates.get(f"{STAGE_SIGNED_ICF} -> {STAGE_ENROLLED}", 0)
            enr_l = lag + inter_stage_lags.get(f"{STAGE_SIGNED_ICF} -> {STAGE_ENROLLED}", 0)
            enr_d = row['curr_t'] + pd.to_timedelta(enr_l, unit='D')
            projs.append({'ip': prob, 'id': icf_d, 'ep': enr_p, 'ed': enr_d})
    
    if not projs: return default
    proj_df = pd.DataFrame(projs)
    max_d = proj_df[['id', 'ed']].max().max() if not proj_df.empty else datetime.now()
    end_p = max(pd.Period(max_d, 'M'), pd.Period(datetime.now(), 'M') + 24)
    res_idx = pd.period_range(start=datetime.now(), end=end_p, freq='M')
    res = pd.DataFrame(0.0, index=res_idx, columns=['ICFs from Pipeline', 'Enrollments from Pipeline'])
    res['ICFs from Pipeline'] = res.index.map(proj_df.groupby(proj_df['id'].dt.to_period('M'))['ip'].sum())
    res['Enrollments from Pipeline'] = res.index.map(proj_df.groupby(proj_df['ed'].dt.to_period('M'))['ep'].sum())
    
    return {'results_df': res.fillna(0), 'total_icf_yield': proj_df['ip'].sum(), 'total_enroll_yield': proj_df['ep'].sum()}

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
                st.success("Data loaded and processed successfully!"); st.rerun()
        except Exception as e: st.error(f"Error loading referral data: {e}")

if st.session_state.get('data_processed', False):
    st.subheader("1. Timeline Assumptions"); col1, col2 = st.columns(2)
    history_months = col1.number_input("Months of History to Display", 0, 24, 6, 1)
    proj_horizon = col2.number_input("Months of Future Projection", 1, 48, 12, 1)
    
    last_hist_month = st.session_state.processed_df["Submission_Month"].max()
    _proj_start_month = last_hist_month + 1 if pd.notna(last_hist_month) else pd.Period(datetime.now(), 'M')
    future_months_ui = pd.period_range(start=_proj_start_month, periods=proj_horizon, freq='M')
    
    st.subheader("2. Future Spend & New Lead Assumptions"); col1, col2 = st.columns(2)
    with col1:
        st.write("Future Monthly Ad Spend:")
        if 'spend_df_cache' not in st.session_state or len(st.session_state.spend_df_cache) != proj_horizon:
            st.session_state.spend_df_cache = pd.DataFrame([{'Month':m.strftime('%Y-%m'),'Planned_Spend':20000.0} for m in future_months_ui])
        proj_spend_dict = {pd.Period(r['Month'],'M'):float(r['Planned_Spend']) for _, r in st.data_editor(st.session_state.spend_df_cache, key='spend_editor', use_container_width=True, num_rows="fixed").iterrows()}
    with col2:
        st.write("Assumed CPQR ($) per Month:")
        if 'cpqr_df_cache' not in st.session_state or len(st.session_state.cpqr_df_cache) != proj_horizon:
            st.session_state.cpqr_df_cache = pd.DataFrame([{'Month':m.strftime('%Y-%m'),'Assumed_CPQR':120.0} for m in future_months_ui])
        proj_cpqr_dict = {pd.Period(r['Month'],'M'):float(r['Assumed_CPQR']) for _, r in st.data_editor(st.session_state.cpqr_df_cache, key='cpqr_editor', use_container_width=True, num_rows="fixed").iterrows()}

    st.subheader("3. Funnel Conversion Rate Assumptions")
    rate_method = st.radio("Base Rates On:", ('Manual Input Below', 'Rolling Historical Average'), key='rate_method', horizontal=True)
    rolling_window = st.selectbox("Rolling Window:",[1,3,6,999],index=1,format_func=lambda x:"Overall" if x==999 else f"{x}-Mo",key='rolling_win') if rate_method == 'Rolling Historical Average' else 0

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
        effective_rates, rates_desc = determine_effective_projection_rates(st.session_state.processed_df, st.session_state.ordered_stages, st.session_state.ts_col_map,rate_method, rolling_window, manual_rates, st.session_state.inter_stage_lags)
        st.session_state.rates_used = {'rates': effective_rates, 'desc': rates_desc}
        
        hist_df = calculate_historical_performance(st.session_state.processed_df, st.session_state.ts_col_map, history_months, _proj_start_month)
        
        proj_inputs = {'horizon': proj_horizon, 'spend_dict': proj_spend_dict, 'cpqr_dict': proj_cpqr_dict, 'final_conv_rates': effective_rates, 'inter_stage_lags': st.session_state.inter_stage_lags}
        
        new_leads_df = calculate_new_lead_projections(st.session_state.processed_df, st.session_state.ordered_stages, st.session_state.ts_col_map, proj_inputs)
        pipeline_res = calculate_pipeline_projection(st.session_state.processed_df, st.session_state.ordered_stages, st.session_state.ts_col_map, st.session_state.inter_stage_lags, effective_rates)
        pipeline_df = pipeline_res['results_df']
        
        future_df = pd.concat([new_leads_df, pipeline_df], axis=1).fillna(0)
        future_df['Total Projected ICFs'] = future_df['ICFs from New Leads'] + future_df['ICFs from Pipeline']
        
        full_timeline_df = pd.concat([hist_df, future_df]).fillna(0)
        
        all_cols = ['ICFs from New Leads', 'ICFs from Pipeline', 'Total Projected ICFs', 'Enrollments from Pipeline']
        for col in all_cols:
            if col not in full_timeline_df.columns: full_timeline_df[col] = 0
        
        full_timeline_df = full_timeline_df[all_cols].round(0).astype(int)
        full_timeline_df['Cumulative ICFs'] = full_timeline_df['Total Projected ICFs'].cumsum()
        full_timeline_df['Cumulative Enrollments'] = full_timeline_df['Enrollments from Pipeline'].cumsum()
        
        st.session_state.full_forecast_data = full_timeline_df
        st.session_state.summary_metrics = {'new_icfs': new_leads_df['ICFs from New Leads'].sum(), 'pipe_icfs': pipeline_res['total_icf_yield']}

    if 'full_forecast_data' in st.session_state:
        results = st.session_state.full_forecast_data
        summary = st.session_state.summary_metrics
        rates_info = st.session_state.rates_used
        
        st.header("4. Forecast Results")
        with st.expander(f"Rates Used in This Forecast ({rates_info['desc']})", expanded=False):
            st.json({k: f"{v*100:.1f}%" for k, v in rates_info['rates'].items()})

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Projected ICFs (Future)", f"{results.loc[_proj_start_month:]['Total Projected ICFs'].sum():,.0f}")
        c2.metric("from New Leads", f"{summary['new_icfs']:,.0f}")
        c3.metric("from Existing Pipeline", f"{summary['pipe_icfs']:,.1f}")

        st.subheader("Historical & Projected Monthly View")
        display = results.copy()
        if isinstance(display.index, pd.PeriodIndex): display.index = display.index.strftime('%Y-%m')
        display.index.name = "Month"
        st.dataframe(display[all_cols].style.format("{:,.0f}"))

        st.subheader("Cumulative Projections Over Time")
        chart = results[['Cumulative ICFs', 'Cumulative Enrollments']].copy()
        if isinstance(chart.index, pd.PeriodIndex): chart.index = chart.index.to_timestamp()
        st.line_chart(chart)
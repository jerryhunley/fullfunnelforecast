import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
import io

# ==============================
# Page setup
# ==============================
st.set_page_config(page_title="Recruitment Combiner", layout="wide")
st.title("🧮 Recruitment Combiner — Pipeline + Projections")
st.info("This tool combines the results of two independent forecasts: one for the **existing pipeline** (like Funnel Analysis) and one for **new leads from future spend** (like Projections).")

# ==============================
# Helpers
# ==============================
def _strip_us_tz(s: str) -> str:
    tz_pattern = r"\s+(?:EST|EDT|CST|CDT|MST|MDT|PST|PDT)$"
    return re.sub(tz_pattern, "", s.strip()) if isinstance(s, str) else ""

def parse_dt(s):
    if pd.isna(s): return pd.NaT
    cleaned = _strip_us_tz(str(s))
    return pd.to_datetime(cleaned, errors="coerce")

def parse_history_blob(blob: str):
    """Parse history string matching the first app's logic"""
    if pd.isna(blob) or str(blob).strip() == "": 
        return []
    
    events = []
    # Pattern that matches the first app
    pattern = re.compile(r"([\w\s().'/:-]+?):\s*(\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}(?:\s*[apAP][mM])?(?:\s+[A-Za-z]{3,}(?:T)?)?)")
    
    raw_lines = str(blob).strip().split('\n')
    for line in raw_lines:
        line = line.strip()
        if not line: continue
        match = pattern.match(line)
        if match:
            name, dt_str = match.groups()
            name = name.strip()
            dt_obj = parse_dt(dt_str.strip())
            if name and pd.notna(dt_obj):
                try:
                    events.append((name, dt_obj.to_pydatetime() if hasattr(dt_obj, 'to_pydatetime') else dt_obj))
                except AttributeError:
                    events.append((name, dt_obj))
    
    # Sort by datetime
    try:
        events.sort(key=lambda x: x[1] if pd.notna(x[1]) else datetime.min)
    except TypeError:
        pass
    
    return events

@st.cache_data
def parse_funnel_definition(uploaded_file):
    if uploaded_file is None: return None, None, None
    content = uploaded_file.getvalue().decode("utf-8", errors="replace")
    df = pd.read_csv(io.StringIO(content), sep=None, engine="python", header=None)
    funnel_def, ordered, ts_map = {}, [], {}
    for col in df.columns:
        col_series = df[col].dropna().astype(str).str.strip().str.replace('"', '', regex=False)
        if col_series.empty: continue
        stage_name = col_series.iloc[0]
        if not stage_name: continue
        ordered.append(stage_name)
        statuses = [s for s in col_series.iloc[1:] if s]
        if stage_name not in statuses: statuses.append(stage_name)
        funnel_def[stage_name] = statuses
        ts_map[stage_name] = "TS_" + re.sub(r"[^A-Za-z0-9]+", "_", stage_name).strip("_")
    if not ordered: return None, None, None
    return funnel_def, ordered, ts_map

@st.cache_data
def preprocess_referrals(csv_file, funnel_def, ordered_stages, ts_map):
    """Process referrals matching the first app's logic more closely"""
    content = csv_file.getvalue().decode("utf-8", errors="replace")
    df = pd.read_csv(io.StringIO(content))
    
    # Find submission date column
    subm_col = None
    for col_name in ["Submitted On", "Referral Date", "SubmittedOn"]:
        if col_name in df.columns:
            subm_col = col_name
            break
    
    if subm_col is None: 
        raise ValueError("Could not find a submission date column.")
    
    # Parse submission date
    df["Submitted On_DT"] = pd.to_datetime(df[subm_col].apply(lambda x: _strip_us_tz(str(x)) if pd.notna(x) else x), errors="coerce")
    df = df.dropna(subset=["Submitted On_DT"]).copy()
    df["Submission_Month"] = df["Submitted On_DT"].dt.to_period("M")

    # Find history columns
    stage_hist_col = None
    status_hist_col = None
    for col in df.columns:
        if "stage history" in col.lower():
            stage_hist_col = col
        elif "status history" in col.lower():
            status_hist_col = col
    
    if not stage_hist_col and not status_hist_col: 
        raise ValueError("Need a stage or status history column.")

    # Parse history columns
    if stage_hist_col:
        df[f"Parsed_{stage_hist_col.replace(' ', '_')}"] = df[stage_hist_col].apply(parse_history_blob)
    if status_hist_col:
        df[f"Parsed_{status_hist_col.replace(' ', '_')}"] = df[status_hist_col].apply(parse_history_blob)

    # Create status to stage mapping
    status_to_stage = {}
    for stage, statuses in funnel_def.items():
        for status in statuses:
            status_to_stage[status] = stage

    # Initialize timestamp columns
    for ts_col in ts_map.values():
        df[ts_col] = pd.NaT

    # Process each row to get stage timestamps
    for idx, row in df.iterrows():
        all_events = []
        
        # Collect all events from both history columns
        if stage_hist_col:
            parsed_col = f"Parsed_{stage_hist_col.replace(' ', '_')}"
            if parsed_col in row and row[parsed_col]:
                all_events.extend([(name, dt) for name, dt in row[parsed_col] if isinstance(name, str)])
        
        if status_hist_col:
            parsed_col = f"Parsed_{status_hist_col.replace(' ', '_')}"
            if parsed_col in row and row[parsed_col]:
                all_events.extend([(name, dt) for name, dt in row[parsed_col] if isinstance(name, str)])
        
        # Sort all events by time
        try:
            all_events.sort(key=lambda x: x[1] if pd.notna(x[1]) else datetime.min)
        except TypeError:
            pass
        
        # Process events and assign timestamps
        for event_name, event_dt in all_events:
            if pd.isna(event_dt): 
                continue
            
            # Determine which stage this event belongs to
            event_stage = None
            if event_name in ordered_stages:
                event_stage = event_name
            elif event_name in status_to_stage:
                event_stage = status_to_stage[event_name]
            
            # Assign timestamp if we found a valid stage and haven't assigned it yet
            if event_stage and event_stage in ordered_stages:
                ts_col_name = ts_map.get(event_stage)
                if ts_col_name and pd.isna(df.at[idx, ts_col_name]):
                    df.at[idx, ts_col_name] = event_dt
    
    return df

@st.cache_data
def calc_inter_stage_lags(df, ordered_stages, ts_map):
    lags = {}
    for a, b in zip(ordered_stages, ordered_stages[1:]):
        ca, cb = ts_map.get(a), ts_map.get(b)
        if ca in df.columns and cb in df.columns:
            # Both columns must have valid timestamps
            valid_mask = df[ca].notna() & df[cb].notna()
            if valid_mask.any():
                valid_df = df[valid_mask]
                # Calculate lag in days
                diff = (pd.to_datetime(valid_df[cb]) - pd.to_datetime(valid_df[ca])).dt.total_seconds() / (24*3600)
                # Only keep positive lags
                positive_lags = diff[diff >= 0]
                if len(positive_lags) > 0:
                    lags[f"{a} -> {b}"] = float(np.nanmean(positive_lags))
    return lags

@st.cache_data
def overall_rates(df, ordered_stages, ts_map):
    rates = {}
    for a, b in zip(ordered_stages, ordered_stages[1:]):
        ca, cb = ts_map.get(a), ts_map.get(b)
        if ca in df.columns and cb in df.columns:
            denom = df[ca].notna().sum()
            num = (df[ca].notna() & df[cb].notna()).sum()
            rates[f"{a} -> {b}"] = float(num / denom) if denom > 0 else 0.0
        else:
            rates[f"{a} -> {b}"] = 0.0
    return rates

@st.cache_data
def rolling_rates(df, ordered_stages, ts_map, months: int):
    """Calculate rolling rates based on recent data"""
    rates = {}
    
    # Find the most recent timestamp across all stage columns
    ts_cols = [ts_map[s] for s in ordered_stages if ts_map[s] in df.columns]
    all_timestamps = []
    for col in ts_cols:
        valid_ts = df[col].dropna()
        if not valid_ts.empty:
            all_timestamps.extend(valid_ts.tolist())
    
    if all_timestamps:
        anchor = pd.to_datetime(max(all_timestamps))
    else:
        anchor = pd.Timestamp(datetime.now())
    
    cutoff = anchor - pd.DateOffset(months=months)
    
    for a, b in zip(ordered_stages, ordered_stages[1:]):
        ca, cb = ts_map.get(a), ts_map.get(b)
        if ca in df.columns and cb in df.columns:
            # Count leads that entered stage 'a' after cutoff
            recent_mask = df[ca].notna() & (pd.to_datetime(df[ca]) >= cutoff)
            denom = recent_mask.sum()
            # Count those that also reached stage 'b'
            num = (recent_mask & df[cb].notna()).sum()
            rates[f"{a} -> {b}"] = float(num / denom) if denom > 0 else 0.0
        else:
            rates[f"{a} -> {b}"] = 0.0
    return rates

def product_rate_to(target_stage, ordered_stages, rates):
    try: 
        idx = ordered_stages.index(target_stage)
    except ValueError: 
        return 0.0
    result = 1.0
    for i in range(idx):
        if i < len(ordered_stages) - 1:
            result *= rates.get(f"{ordered_stages[i]} -> {ordered_stages[i+1]}", 0.0)
    return result

def total_lag_to(target_stage, ordered_stages, lags):
    try: 
        idx = ordered_stages.index(target_stage)
    except ValueError: 
        return 0.0
    total = 0.0
    for i in range(idx):
        if i < len(ordered_stages) - 1:
            total += lags.get(f"{ordered_stages[i]} -> {ordered_stages[i+1]}", 30.0)
    return total

@st.cache_data
def pipeline_projection(df, ordered_stages, ts_map, rates, lags, icf_stage, final_stage, terminal_extras):
    """Calculate pipeline projection matching the first app's logic"""
    default = {
        'results_df': pd.DataFrame(), 
        'total_icf_yield': 0, 
        'total_enroll_yield': 0, 
        'in_flight_df': pd.DataFrame()
    }
    
    # Define terminal stages
    terminal_stages = set([final_stage] + list(terminal_extras))
    
    # Filter for in-flight leads (not terminated)
    term_cols = [ts_map.get(s) for s in terminal_stages if ts_map.get(s) in df.columns]
    in_flight = df.copy()
    for term_col in term_cols:
        if term_col in in_flight.columns:
            in_flight = in_flight[in_flight[term_col].isna()]
    
    if in_flight.empty:
        return default
    
    # Determine current stage for each lead
    def get_current_stage(row, ordered_stages, ts_map, terminal_stages):
        """Get the furthest non-terminal stage reached"""
        last_stage = None
        last_ts = pd.NaT
        for stage in ordered_stages:
            if stage in terminal_stages:
                continue
            ts_col = ts_map.get(stage)
            if ts_col and ts_col in row.index and pd.notna(row[ts_col]):
                if pd.isna(last_ts) or row[ts_col] > last_ts:
                    last_ts = row[ts_col]
                    last_stage = stage
        return pd.Series([last_stage, last_ts], index=['current_stage', 'current_stage_ts'])
    
    in_flight[['current_stage', 'current_stage_ts']] = in_flight.apply(
        lambda row: get_current_stage(row, ordered_stages, ts_map, terminal_stages), 
        axis=1
    )
    
    # Remove leads with no current stage
    in_flight = in_flight.dropna(subset=['current_stage']).copy()
    
    if in_flight.empty:
        return default
    
    # Get indices for ICF and final stages
    try:
        icf_idx = ordered_stages.index(icf_stage)
        final_idx = ordered_stages.index(final_stage)
    except ValueError:
        return default
    
    # Calculate projections
    all_icf_projections = []
    all_final_projections = []
    
    # Get ICF column
    icf_col = ts_map.get(icf_stage)
    
    # Process leads already at ICF (they have 100% ICF probability)
    if icf_col and icf_col in in_flight.columns:
        already_at_icf = in_flight[in_flight[icf_col].notna()]
        for _, row in already_at_icf.iterrows():
            # 100% probability for ICF (already there)
            all_icf_projections.append({
                'prob': 1.0,
                'lag_to_icf': 0.0,
                'start_date': row[icf_col]
            })
            
            # Calculate probability to final stage
            p_icf_to_final = 1.0
            lag_icf_to_final = 0.0
            for i in range(icf_idx, final_idx):
                if i < len(ordered_stages) - 1:
                    p_icf_to_final *= rates.get(f"{ordered_stages[i]} -> {ordered_stages[i+1]}", 0.0)
                    lag_icf_to_final += lags.get(f"{ordered_stages[i]} -> {ordered_stages[i+1]}", 30.0)
            
            if p_icf_to_final > 0:
                all_final_projections.append({
                    'prob': p_icf_to_final,
                    'lag': lag_icf_to_final,
                    'start_date': row[icf_col]
                })
    
    # Process leads before ICF
    leads_before_icf = in_flight[in_flight[icf_col].isna()] if icf_col and icf_col in in_flight.columns else in_flight
    
    for _, row in leads_before_icf.iterrows():
        if pd.isna(row['current_stage']):
            continue
        
        try:
            start_idx = ordered_stages.index(row['current_stage'])
        except ValueError:
            continue
        
        # Calculate path to ICF
        p_to_icf = 1.0
        lag_to_icf = 0.0
        path_found = False
        
        for i in range(start_idx, len(ordered_stages) - 1):
            from_stage = ordered_stages[i]
            to_stage = ordered_stages[i + 1]
            
            rate_key = f"{from_stage} -> {to_stage}"
            lag_key = f"{from_stage} -> {to_stage}"
            
            p_to_icf *= rates.get(rate_key, 0.0)
            lag_to_icf += lags.get(lag_key, 30.0)
            
            if to_stage == icf_stage:
                path_found = True
                break
        
        if path_found and p_to_icf > 0:
            all_icf_projections.append({
                'prob': p_to_icf,
                'lag_to_icf': lag_to_icf,
                'start_date': row['current_stage_ts']
            })
            
            # Calculate from ICF to final
            p_icf_to_final = 1.0
            lag_icf_to_final = 0.0
            
            for i in range(icf_idx, final_idx):
                if i < len(ordered_stages) - 1:
                    p_icf_to_final *= rates.get(f"{ordered_stages[i]} -> {ordered_stages[i+1]}", 0.0)
                    lag_icf_to_final += lags.get(f"{ordered_stages[i]} -> {ordered_stages[i+1]}", 30.0)
            
            if p_icf_to_final > 0:
                all_final_projections.append({
                    'prob': p_to_icf * p_icf_to_final,
                    'lag': lag_to_icf + lag_icf_to_final,
                    'start_date': row['current_stage_ts']
                })
    
    # Calculate total yields
    total_icf_yield = sum(p['prob'] for p in all_icf_projections)
    total_final_yield = sum(p['prob'] for p in all_final_projections)
    
    # Create monthly projections
    start_month = pd.Period(datetime.now(), 'M')
    idx = pd.period_range(start=start_month, periods=36, freq='M')
    results = pd.DataFrame(0.0, index=idx, columns=["ICFs from Pipeline", f"{final_stage} from Pipeline"])
    
    # Aggregate ICF projections by month
    for proj in all_icf_projections:
        if pd.notna(proj['start_date']) and pd.notna(proj['lag_to_icf']):
            landing_date = proj['start_date'] + timedelta(days=proj['lag_to_icf'])
            landing_month = pd.Period(landing_date, 'M')
            if landing_month in results.index:
                results.loc[landing_month, "ICFs from Pipeline"] += proj['prob']
    
    # Aggregate final stage projections by month
    for proj in all_final_projections:
        if pd.notna(proj['start_date']) and pd.notna(proj['lag']):
            landing_date = proj['start_date'] + timedelta(days=proj['lag'])
            landing_month = pd.Period(landing_date, 'M')
            if landing_month in results.index:
                results.loc[landing_month, f"{final_stage} from Pipeline"] += proj['prob']
    
    # Round for display
    results["ICFs from Pipeline"] = results["ICFs from Pipeline"].round(1)
    results[f"{final_stage} from Pipeline"] = results[f"{final_stage} from Pipeline"].round(1)
    
    return {
        'results_df': results,
        'total_icf_yield': total_icf_yield,
        'total_enroll_yield': total_final_yield,
        'in_flight_df': in_flight
    }

# --- Pipeline Narrative Function ---
@st.cache_data
def generate_pipeline_narrative(in_flight_df, ordered_stages, rates, lags, final_stage):
    if in_flight_df.empty: 
        return []
    
    narrative = []
    stage_counts = in_flight_df['current_stage'].value_counts()
    
    for stage_name in ordered_stages:
        if stage_name == final_stage or stage_name not in stage_counts:
            continue
        
        count = stage_counts[stage_name]
        downstream = []
        p_cum = 1.0
        lag_cum = 0.0
        
        try:
            start_idx = ordered_stages.index(stage_name)
        except ValueError:
            continue
        
        for i in range(start_idx, len(ordered_stages) - 1):
            a, b = ordered_stages[i], ordered_stages[i+1]
            rate = rates.get(f"{a} -> {b}", 0.0)
            lag = lags.get(f"{a} -> {b}", 0.0)
            p_cum *= rate
            lag_cum += lag
            
            downstream.append({
                'stage': b,
                'count': count * p_cum,
                'lag': lag_cum
            })
            
            if b == final_stage:
                break
        
        narrative.append({
            'current_stage': stage_name,
            'count': count,
            'downstream': downstream
        })
    
    return narrative

@st.cache_data
def projections_new_leads(processed_df, ordered_stages, ts_map, horizon_months, spend_by_month, cpqr_by_month, rates, lags, icf_stage, final_stage):
    # Determine start month
    if not processed_df.empty and "Submission_Month" in processed_df.columns:
        start_month = processed_df["Submission_Month"].max() + 1
    else:
        start_month = pd.Period(datetime.now(), "M")
    
    future_months = pd.period_range(start=start_month, periods=horizon_months, freq="M")
    cohorts = pd.DataFrame(index=future_months)
    cohorts["Ad_Spend"] = [float(spend_by_month.get(m, 0.0)) for m in future_months]
    cohorts["CPQR"] = [float(cpqr_by_month.get(m, 120.0)) for m in future_months]
    cohorts["New_QLs"] = (cohorts["Ad_Spend"] / cohorts["CPQR"].replace(0, np.nan)).fillna(0.0)

    # Calculate overall conversion rates and lags
    p_to_icf = product_rate_to(icf_stage, ordered_stages, rates)
    lag_to_icf = total_lag_to(icf_stage, ordered_stages, lags)
    p_to_final = product_rate_to(final_stage, ordered_stages, rates)
    lag_to_final = total_lag_to(final_stage, ordered_stages, lags)

    cohorts["ICFs_generated"] = cohorts["New_QLs"] * p_to_icf
    cohorts["Final_generated"] = cohorts["New_QLs"] * p_to_final

    # Create output dataframe with extended timeline
    max_lag_days = max(lag_to_final, lag_to_icf) + 90
    idx = pd.period_range(start=start_month, periods=horizon_months + int(max_lag_days/30) + 1, freq="M")
    out = pd.DataFrame(0.0, index=idx, columns=["ICFs from New Leads", f"{final_stage} from New Leads"])
    
    # Distribute ICFs and final stage over time
    for m, row in cohorts.iterrows():
        if row["ICFs_generated"] > 0:
            landing_month = (m.to_timestamp() + timedelta(days=lag_to_icf)).to_period("M")
            if landing_month in out.index:
                out.loc[landing_month, "ICFs from New Leads"] += row["ICFs_generated"]
        
        if row["Final_generated"] > 0:
            landing_month = (m.to_timestamp() + timedelta(days=lag_to_final)).to_period("M")
            if landing_month in out.index:
                out.loc[landing_month, f"{final_stage} from New Leads"] += row["Final_generated"]
    
    # Round for display
    out["ICFs from New Leads"] = out["ICFs from New Leads"].round(1)
    out[f"{final_stage} from New Leads"] = out[f"{final_stage} from New Leads"].round(1)
    
    return out

def combine_monthly_tables(pipeline_df, new_df, final_stage_name):
    idx = pipeline_df.index.union(new_df.index)
    both = pd.DataFrame(index=idx)
    
    # Combine all columns
    both["ICFs from Pipeline"] = pipeline_df.get("ICFs from Pipeline", 0.0).reindex(idx).fillna(0.0)
    both[f"{final_stage_name} from Pipeline"] = pipeline_df.get(f"{final_stage_name} from Pipeline", 0.0).reindex(idx).fillna(0.0)
    both["ICFs from New Leads"] = new_df.get("ICFs from New Leads", 0.0).reindex(idx).fillna(0.0)
    both[f"{final_stage_name} from New Leads"] = new_df.get(f"{final_stage_name} from New Leads", 0.0).reindex(idx).fillna(0.0)
    
    # Calculate totals
    both["Total Projected ICFs"] = both["ICFs from Pipeline"] + both["ICFs from New Leads"]
    both[f"Total Projected {final_stage_name}"] = both[f"{final_stage_name} from Pipeline"] + both[f"{final_stage_name} from New Leads"]
    
    # Calculate cumulative
    both["Cumulative ICFs"] = both["Total Projected ICFs"].cumsum()
    both[f"Cumulative {final_stage_name}"] = both[f"Total Projected {final_stage_name}"].cumsum()
    
    return both

def best_guess_stage(ordered_stages, keywords, fallback_idx):
    for s in reversed(ordered_stages):
        if any(k in s.lower() for k in keywords):
            return s
    return ordered_stages[fallback_idx] if len(ordered_stages) > abs(fallback_idx) else ordered_stages[-1]

# ==============================
# Main App
# ==============================
with st.sidebar:
    st.header("1) Upload")
    ref_csv = st.file_uploader("Referral Data (CSV)", type=["csv"])
    funnel_csv = st.file_uploader("Funnel Definition", type=["csv", "tsv"])
    st.header("2) Rates")
    rate_mode = st.radio("Rates from?", ["Manual", "Overall", "Rolling"], index=2, horizontal=True)
    roll_months = st.slider("Rolling window (months)", 1, 12, 1) if rate_mode == "Rolling" else None
    st.header("3) Projections")
    horizon = st.number_input("Projection horizon (months)", 1, 48, 12, 1)

if ref_csv and funnel_csv:
    try:
        funnel_def, ordered, ts_map = parse_funnel_definition(funnel_csv)
        if not funnel_def:
            st.error("Funnel definition invalid.")
            st.stop()
        
        df = preprocess_referrals(ref_csv, funnel_def, ordered, ts_map)
        lags = calc_inter_stage_lags(df, ordered, ts_map)

        st.sidebar.header("Stage Mapping")
        icf_stage = st.sidebar.selectbox("ICF stage", ordered, index=ordered.index(best_guess_stage(ordered, ["icf", "consent", "signed"], -2)))
        final_stage = st.sidebar.selectbox("Final stage", ordered, index=ordered.index(best_guess_stage(ordered, ["enroll", "random", "complete"], -1)))
        
        if ordered.index(final_stage) <= ordered.index(icf_stage):
            st.sidebar.error("Final stage must be after ICF stage.")
            st.stop()

        # Prepare future months for spend/CPQR input
        if not df.empty and "Submission_Month" in df.columns:
            start_m = df["Submission_Month"].max() + 1
        else:
            start_m = pd.Period(datetime.now(), "M")
        
        future_months = pd.period_range(start=start_m, periods=horizon, freq="M")
        
        st.subheader("Future Spend & CPQR")
        colA, colB = st.columns(2)
        spend_df = colA.data_editor(
            pd.DataFrame({"Month": future_months.astype(str), "Ad Spend": [0.0]*len(future_months)}),
            use_container_width=True,
            num_rows="fixed"
        )
        cpqr_df = colB.data_editor(
            pd.DataFrame({"Month": future_months.astype(str), "CPQR": [120.0]*len(future_months)}),
            use_container_width=True,
            num_rows="fixed"
        )
        
        spend_dict = {pd.Period(r["Month"], "M"): r["Ad Spend"] for _, r in spend_df.iterrows()}
        cpqr_dict = {pd.Period(r["Month"], "M"): r["CPQR"] for _, r in cpqr_df.iterrows()}

        # Calculate effective rates based on selection
        if rate_mode == "Overall":
            eff_rates = overall_rates(df, ordered, ts_map)
        elif rate_mode == "Rolling":
            eff_rates = rolling_rates(df, ordered, ts_map, int(roll_months or 3))
        else:  # Manual
            defaults = overall_rates(df, ordered, ts_map)
            st.subheader("Manual Conversion Rates (%)")
            eff_rates = {}
            cols = st.columns(3)
            for i, (a, b) in enumerate(zip(ordered, ordered[1:])):
                default = defaults.get(f"{a} -> {b}", 0.0) * 100
                eff_rates[f"{a} -> {b}"] = cols[i%3].slider(f"{a} → {b}", 0.0, 100.0, round(default, 1), 0.1) / 100

        if st.button("🚀 Run Combined Calculation"):
            # Identify terminal stages
            terminal_extras = []
            for stage in ordered:
                stage_lower = stage.lower()
                if any(term in stage_lower for term in ["screen fail", "lost", "withdrawn", "discontinued", "screen failed"]):
                    terminal_extras.append(stage)
           
           # Run pipeline projection
           pipe_res = pipeline_projection(df, ordered, ts_map, eff_rates, lags, icf_stage, final_stage, terminal_extras)
           
           # Run new leads projection
           new_df = projections_new_leads(df, ordered, ts_map, horizon, spend_dict, cpqr_dict, eff_rates, lags, icf_stage, final_stage)
           
           # Combine results
           combined = combine_monthly_tables(pipe_res['results_df'], new_df, final_stage)

           st.success("✅ Calculation complete!")
           
           # Display summary metrics
           col1, col2, col3 = st.columns(3)
           with col1:
               st.metric("Total ICF Yield (Pipeline)", f"{pipe_res['total_icf_yield']:.1f}")
           with col2:
               st.metric(f"Total {final_stage} Yield (Pipeline)", f"{pipe_res['total_enroll_yield']:.1f}")
           with col3:
               st.metric("In-Flight Leads", f"{len(pipe_res['in_flight_df']):,}")
           
           # Show effective rates being used
           with st.expander("📊 Conversion Rates Being Used", expanded=False):
               rate_df = pd.DataFrame(list(eff_rates.items()), columns=["Transition", "Rate"])
               rate_df["Rate %"] = (rate_df["Rate"] * 100).round(1)
               st.dataframe(rate_df[["Transition", "Rate %"]], use_container_width=True)
           
           # Show lags being used
           with st.expander("⏱️ Inter-Stage Lags (Days)", expanded=False):
               if lags:
                   lag_df = pd.DataFrame(list(lags.items()), columns=["Transition", "Days"])
                   lag_df["Days"] = lag_df["Days"].round(1)
                   st.dataframe(lag_df, use_container_width=True)
               else:
                   st.info("No lag data available")
           
           st.subheader("📅 Combined Monthly Landings")
           
           # Prepare display dataframe
           display = combined.rename(columns={
               f"{final_stage} from Pipeline": f"Final Stage from Pipeline",
               f"{final_stage} from New Leads": f"Final Stage from New Leads",
               f"Total Projected {final_stage}": f"Total Projected Final Stage",
               f"Cumulative {final_stage}": f"Cumulative Final Stage"
           })
           
           display_cols = [
               "ICFs from Pipeline",
               "ICFs from New Leads",
               "Total Projected ICFs",
               "Final Stage from Pipeline",
               "Final Stage from New Leads",
               "Total Projected Final Stage"
           ]
           
           # Show only rows with data
           display_filtered = display[display_cols]
           display_filtered = display_filtered.loc[(display_filtered != 0).any(axis=1)]
           
           st.dataframe(
               display_filtered.round(1).style.format("{:.1f}"),
               use_container_width=True
           )
           
           # Pipeline Breakdown Display
           st.subheader("🔬 Pipeline Breakdown")
           st.caption("This shows how the **'ICFs from Pipeline'** and **'Final Stage from Pipeline'** totals are derived from leads currently in your funnel.")
           
           # Show current stage distribution
           if not pipe_res['in_flight_df'].empty:
               stage_dist = pipe_res['in_flight_df']['current_stage'].value_counts()
               
               col1, col2 = st.columns([1, 2])
               with col1:
                   st.write("**Current Stage Distribution:**")
                   for stage, count in stage_dist.items():
                       st.write(f"- {stage}: {count}")
               
               with col2:
                   # Generate narrative
                   narrative_data = generate_pipeline_narrative(
                       pipe_res['in_flight_df'], 
                       ordered, 
                       eff_rates, 
                       lags, 
                       final_stage
                   )
                   
                   if narrative_data:
                       for step in narrative_data:
                           with st.expander(f"From '{step['current_stage']}' ({step['count']} leads)", expanded=False):
                               st.metric(f"Leads Currently At This Stage", f"{step['count']:,}")
                               st.write("**From this group, we project:**")
                               for proj in step['downstream']:
                                   time_text = f" in **~{proj['lag']:.0f} days**" if proj['lag'] > 0 else ""
                                   st.info(f"**~{proj['count']:.1f}** will advance to **'{proj['stage']}'**{time_text}", icon="➡️")
                   else:
                       st.info("No leads are currently in-flight in the pipeline.")
           else:
               st.info("No leads are currently in-flight in the pipeline.")
           
           # Charts
           st.subheader("📊 Visualizations")
           
           # Prepare chart data
           chart_data = combined[["Total Projected ICFs", f"Total Projected {final_stage}"]].copy()
           chart_data = chart_data.rename(columns={f"Total Projected {final_stage}": "Total Projected Final Stage"})
           
           # Filter to non-zero rows for cleaner chart
           chart_data = chart_data.loc[(chart_data != 0).any(axis=1)]
           
           if not chart_data.empty:
               chart_data.index = chart_data.index.to_timestamp()
               st.line_chart(chart_data)
           else:
               st.info("No projections to visualize")
           
           # Cumulative chart
           st.subheader("📈 Cumulative Projections")
           cum_data = combined[["Cumulative ICFs", f"Cumulative {final_stage}"]].copy()
           cum_data = cum_data.rename(columns={f"Cumulative {final_stage}": "Cumulative Final Stage"})
           
           # Filter to non-zero rows
           cum_data = cum_data.loc[(cum_data != 0).any(axis=1)]
           
           if not cum_data.empty:
               cum_data.index = cum_data.index.to_timestamp()
               st.line_chart(cum_data)
           
           # Downloads
           with st.expander("📥 Downloads"):
               # Combined results
               csv_combined = combined.to_csv().encode("utf-8")
               st.download_button(
                   "⬇️ Combined Results",
                   csv_combined,
                   "combined_projections.csv",
                   "text/csv"
               )
               
               # Pipeline only
               csv_pipeline = pipe_res['results_df'].to_csv().encode("utf-8")
               st.download_button(
                   "⬇️ Pipeline Only",
                   csv_pipeline,
                   "pipeline_projections.csv",
                   "text/csv"
               )
               
               # New leads only
               csv_new = new_df.to_csv().encode("utf-8")
               st.download_button(
                   "⬇️ New Leads Only",
                   csv_new,
                   "new_leads_projections.csv",
                   "text/csv"
               )
               
               # In-flight leads details
               if not pipe_res['in_flight_df'].empty:
                   # Select relevant columns for export
                   export_cols = ['current_stage', 'current_stage_ts']
                   # Add timestamp columns
                   for stage in ordered:
                       ts_col = ts_map.get(stage)
                       if ts_col in pipe_res['in_flight_df'].columns:
                           export_cols.append(ts_col)
                   
                   in_flight_export = pipe_res['in_flight_df'][export_cols].copy()
                   csv_in_flight = in_flight_export.to_csv().encode("utf-8")
                   st.download_button(
                       "⬇️ In-Flight Leads Details",
                       csv_in_flight,
                       "in_flight_leads.csv",
                       "text/csv"
                   )

   except Exception as e:
       st.error(f"An error occurred during processing: {e}")
       st.exception(e)
       
       # Debug information
       with st.expander("🐛 Debug Information"):
           st.write("**Error details:**")
           st.code(str(e))
           
           if 'df' in locals():
               st.write("**Data shape:**", df.shape)
               st.write("**Columns:**", list(df.columns))
               
               # Check timestamp columns
               if 'ts_map' in locals():
                   st.write("**Timestamp columns found:**")
                   for stage, ts_col in ts_map.items():
                       if ts_col in df.columns:
                           non_null = df[ts_col].notna().sum()
                           st.write(f"- {stage} ({ts_col}): {non_null} non-null values")
           
           if 'eff_rates' in locals():
               st.write("**Conversion rates:**")
               st.json(eff_rates)
           
           if 'lags' in locals():
               st.write("**Inter-stage lags:**")
               st.json(lags)
else:
   st.info("👋 Upload your Referral Data and Funnel Definition files on the left to get started.")
   
   with st.expander("ℹ️ File Format Requirements"):
       st.markdown("""
       **Referral Data CSV:**
       - Must have a date column: 'Submitted On', 'Referral Date', or 'SubmittedOn'
       - Must have a history column: 'Lead Stage History' or 'Lead Status History'
       - History format: `Stage Name: MM/DD/YYYY HH:MM AM/PM TZ`
       
       **Funnel Definition CSV/TSV:**
       - First row: Stage names
       - Subsequent rows: Statuses that map to each stage
       - Each column represents one stage in your funnel
       """)
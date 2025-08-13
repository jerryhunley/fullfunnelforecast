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
st.info("This tool combines two forecasts: one for the **existing pipeline** and one for **new leads from future spend**.")

# ==============================
# Helpers
# ==============================
def _strip_us_tz(s: str) -> str:
    tz_pattern = r"\s+(?:EST|EDT|CST|CDT|MST|MDT|PST|PDT)$"
    return re.sub(tz_pattern, "", s.strip()) if isinstance(s, str) else ""

def parse_dt(s):
    if pd.isna(s): return pd.NaT
    return pd.to_datetime(_strip_us_tz(str(s)), errors="coerce")

def parse_history_blob(blob: str):
    if pd.isna(blob): return []
    events = []
    pat = re.compile(r"^(.+?):\s*(.+)$")
    for raw in str(blob).split("\n"):
        m = pat.match(raw.strip())
        if not m: continue
        name, dts = m.groups()
        dt = parse_dt(dts)
        if name and pd.notna(dt):
            events.append((name.strip(), dt))
    events.sort(key=lambda x: x[1])
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
    content = csv_file.getvalue().decode("utf-8", errors="replace")
    df = pd.read_csv(io.StringIO(content))
    subm_col = next((c for c in ["Submitted On", "Referral Date"] if c in df.columns), None)
    if subm_col is None: raise ValueError("Could not find a submission date column.")
    df["Submitted On_DT"] = pd.to_datetime(df[subm_col], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["Submitted On_DT"]).copy()
    df["Submission_Month"] = df["Submitted On_DT"].dt.to_period("M")

    stage_hist_col = next((c for c in df.columns if "stage history" in c.lower()), None)
    status_hist_col = next((c for c in df.columns if "status history" in c.lower()), None)
    if not stage_hist_col and not status_hist_col: raise ValueError("Need a stage or status history column.")

    parsed_stage = df[stage_hist_col].apply(parse_history_blob) if stage_hist_col else pd.Series([[]]*len(df), index=df.index)
    parsed_status = df[status_hist_col].apply(parse_history_blob) if status_hist_col else pd.Series([[]]*len(df), index=df.index)

    status_to_stage = {s: stg for stg, statuses in funnel_def.items() for s in statuses}
    ts_frame = pd.DataFrame(index=df.index, columns=list(ts_map.values()), dtype="datetime64[ns]")
    for i in df.index:
        events = sorted(parsed_stage.get(i, []) + parsed_status.get(i, []), key=lambda x: x[1])
        for name, when in events:
            stg = status_to_stage.get(name, name if name in ordered_stages else None)
            if stg in ordered_stages:
                col = ts_map[stg]
                if pd.isna(ts_frame.at[i, col]): ts_frame.at[i, col] = when
    return pd.concat([df, ts_frame], axis=1)

@st.cache_data
def calc_inter_stage_lags(df, ordered_stages, ts_map):
    lags = {}
    for a, b in zip(ordered_stages, ordered_stages[1:]):
        ca, cb = ts_map.get(a), ts_map.get(b)
        if ca in df.columns and cb in df.columns:
            d = (df[cb] - df[ca]).dt.total_seconds() / (24*3600)
            valid_lags = d[d >= 0]
            if not valid_lags.empty: lags[f"{a} -> {b}"] = float(np.nanmean(valid_lags))
    return lags

@st.cache_data
def calculate_maturity_adjusted_rates(df, ordered_stages, ts_map, lags, manual_rates, is_rolling, window_months):
    """The robust, transit-time-adjusted rate calculation function."""
    maturity_days = {key: max(round(1.5 * lags.get(key, 30)), 20) for key in manual_rates.keys()}
    
    try:
        hist = df.groupby("Submission_Month").size().to_frame("Total")
        for stage in ordered_stages:
            ts = ts_map.get(stage)
            if ts and ts in df.columns:
                col = f"Reached_{stage.replace(' ', '_')}"
                hist[col] = df.dropna(subset=[ts]).groupby(df['Submission_Month']).size()
        hist = hist.fillna(0)

        rates = {}
        for key, manual_val in manual_rates.items():
            try: from_s, to_s = key.split(" -> ")
            except ValueError: continue
            col_f, col_t = f"Reached_{from_s.replace(' ', '_')}", f"Reached_{to_s.replace(' ', '_')}"
            
            if col_f not in hist.columns or col_t not in hist.columns:
                rates[key] = manual_val; continue

            mature_hist = hist[hist.index.to_timestamp() + pd.Timedelta(days=maturity_days.get(key, 45)) < pd.Timestamp(datetime.now())]
            if mature_hist.empty:
                rates[key] = manual_val; continue

            overall_rate = (mature_hist[col_t].sum() / mature_hist[col_f].sum()) if mature_hist[col_f].sum() > 0 else 0
            
            if not is_rolling:
                rates[key] = overall_rate if pd.notna(overall_rate) and overall_rate > 0 else manual_val
            else:
                monthly_rates = (mature_hist[col_t] / mature_hist[col_f].replace(0, np.nan)).where(mature_hist[col_f] >= 5, np.nan).dropna()
                if not monthly_rates.empty:
                    win = min(window_months, len(monthly_rates))
                    rolling_avg = monthly_rates.rolling(win, min_periods=1).mean().iloc[-1]
                    rates[key] = rolling_avg if pd.notna(rolling_avg) and rolling_avg > 0 else manual_val
                else:
                    rates[key] = overall_rate if pd.notna(overall_rate) and overall_rate > 0 else manual_val
        
        desc = f"Maturity-Adjusted ({window_months}-Mo Rolling)" if is_rolling else "Maturity-Adjusted (Overall)"
        return rates, desc, maturity_days
    except Exception:
        return manual_rates, "Manual (error in calculation)", {}

def product_rate_to(target_stage, ordered_stages, rates):
    try: idx = ordered_stages.index(target_stage)
    except ValueError: return 0.0
    return np.prod([float(rates.get(f"{ordered_stages[i]} -> {ordered_stages[i+1]}", 0.0)) for i in range(idx)])

def total_lag_to(target_stage, ordered_stages, lags):
    try: idx = ordered_stages.index(target_stage)
    except ValueError: return 0.0
    return sum(float(lags.get(f"{ordered_stages[i]} -> {ordered_stages[i+1]}", 30.0)) for i in range(idx))

@st.cache_data
def pipeline_projection(df, ordered_stages, ts_map, rates, lags, icf_stage, final_stage, terminal_extras):
    default = {'results_df': pd.DataFrame(), 'total_icf_yield': 0, 'total_enroll_yield': 0, 'in_flight_df': pd.DataFrame()}
    terminal_stages = set([final_stage] + list(terminal_extras))
    term_cols = [ts_map.get(s) for s in terminal_stages if ts_map.get(s) in df.columns]
    is_terminal = pd.Series(False, index=df.index);
    for c in term_cols: is_terminal |= df[c].notna()
    in_flight = df.loc[~is_terminal].copy()

    non_terminal = [s for s in ordered_stages if s not in terminal_stages]
    ts_cols = [ts_map[s] for s in non_terminal if ts_map.get(s) in df.columns]
    if not ts_cols: return default
    in_flight['current_ts'] = in_flight[ts_cols].max(axis=1)
    in_flight = in_flight.dropna(subset=['current_ts']).copy()
    
    stage_map = {ts_map[s]: s for s in non_terminal}
    in_flight['current_stage'] = in_flight[ts_cols].idxmax(axis=1).map(stage_map)
    
    records = []
    try: icf_idx, final_idx = ordered_stages.index(icf_stage), ordered_stages.index(final_stage)
    except ValueError: return default

    for _, row in in_flight.iterrows():
        start_idx = ordered_stages.index(row['current_stage'])
        p_to_icf = product_rate_to(icf_stage, ordered_stages[start_idx:], rates)
        lag_to_icf = total_lag_to(icf_stage, ordered_stages[start_idx:], lags)
        p_to_final = product_rate_to(final_stage, ordered_stages[start_idx:], rates)
        lag_to_final = total_lag_to(final_stage, ordered_stages[start_idx:], lags)
        if p_to_icf > 0:
            icf_month = pd.Period(row['current_ts'] + timedelta(days=lag_to_icf), "M")
            fin_month = pd.Period(row['current_ts'] + timedelta(days=lag_to_final), "M")
            records.append((icf_month, p_to_icf, fin_month, p_to_final))
    
    if not records: return default
    rec_df = pd.DataFrame(records, columns=["icf_m", "p_icf", "fin_m", "p_final"])
    idx = pd.period_range(start=datetime.now(), periods=36, freq="M")
    out = pd.DataFrame(0.0, index=idx, columns=["ICFs from Pipeline", f"{final_stage} from Pipeline"])
    out["ICFs from Pipeline"] = out.index.map(rec_df.groupby("icf_m")["p_icf"].sum()).fillna(0)
    out[f"{final_stage} from Pipeline"] = out.index.map(rec_df.groupby("fin_m")["p_final"].sum()).fillna(0)
    return {'results_df': out, 'total_icf_yield': rec_df['p_icf'].sum(), 'total_enroll_yield': rec_df['p_final'].sum(), 'in_flight_df': in_flight}

@st.cache_data
def projections_new_leads(df, ordered_stages, ts_map, horizon, spend, cpqr, rates, lags, icf_stage, final_stage):
    start = (df["Submission_Month"].max() + 1) if not df.empty else pd.Period(datetime.now(), "M")
    future_months = pd.period_range(start=start, periods=horizon, freq="M")
    cohorts = pd.DataFrame({"Ad_Spend": [spend.get(m,0)], "CPQR": [cpqr.get(m,120)]}, index=future_months)
    cohorts["New_QLs"] = (cohorts["Ad_Spend"] / cohorts["CPQR"].replace(0, np.nan)).fillna(0.0)
    cohorts["ICFs_generated"] = cohorts["New_QLs"] * product_rate_to(icf_stage, ordered, rates)
    cohorts["Final_generated"] = cohorts["New_QLs"] * product_rate_to(final_stage, ordered, rates)
    lag_icf, lag_final = total_lag_to(icf_stage, ordered, lags), total_lag_to(final_stage, ordered, lags)
    
    idx = pd.period_range(start=start, periods=horizon + int(lag_final/30) + 2, freq="M")
    out = pd.DataFrame(0.0, index=idx, columns=["ICFs from New Leads", f"{final_stage} from New Leads"])
    for m, row in cohorts.iterrows():
        if row["ICFs_generated"] > 0: out.loc[(m.to_timestamp() + timedelta(days=lag_icf)).to_period("M")] += row["ICFs_generated"]
        if row["Final_generated"] > 0: out.loc[(m.to_timestamp() + timedelta(days=lag_final)).to_period("M")] += row["Final_generated"]
    return out

def combine_and_display(pipe_res, new_df, final_stage_name):
    idx = pipe_res['results_df'].index.union(new_df.index)
    both = pd.DataFrame(index=idx)
    for col_suffix in ["from Pipeline", "from New Leads"]:
        both[f"ICFs {col_suffix}"] = (pipe_res['results_df'] if 'Pipeline' in col_suffix else new_df).get(f"ICFs {col_suffix}", 0).reindex(idx).fillna(0)
        both[f"{final_stage_name} {col_suffix}"] = (pipe_res['results_df'] if 'Pipeline' in col_suffix else new_df).get(f"{final_stage_name} {col_suffix}", 0).reindex(idx).fillna(0)
    both["Total Projected ICFs"] = both["ICFs from Pipeline"] + both["ICFs from New Leads"]
    both[f"Total Projected {final_stage_name}"] = both[f"{final_stage_name} from Pipeline"] + both[f"{final_stage_name} from New Leads"]
    return both

# ==============================
# Main App
# ==============================
with st.sidebar:
    st.header("1) Upload"); ref_csv, funnel_csv = st.file_uploader("Referral Data"), st.file_uploader("Funnel Definition")
    st.header("2) Rates"); rate_mode = st.radio("Rates from?", ["Manual", "Maturity-Adjusted (Rolling)", "Maturity-Adjusted (Overall)"], index=1, horizontal=True)
    roll_months = st.slider("Rolling window (months)", 1, 12, 3) if "Rolling" in rate_mode else None
    st.header("3) Projections"); horizon = st.number_input("Projection horizon (months)", 1, 48, 12, 1)

if ref_csv and funnel_csv:
    try:
        funnel_def, ordered, ts_map = parse_funnel_definition(funnel_csv)
        if not funnel_def: st.error("Funnel definition invalid."); st.stop()
        
        df = preprocess_referrals(ref_csv, funnel_def, ordered, ts_map)
        lags = calc_inter_stage_lags(df, ordered, ts_map)

        st.sidebar.header("Stage Mapping")
        icf_stage = st.sidebar.selectbox("ICF stage", ordered, index=ordered.index(best_guess_stage(ordered, ["icf", "consent"], -2)))
        final_stage = st.sidebar.selectbox("Final stage", ordered, index=ordered.index(best_guess_stage(ordered, ["enroll", "random"], -1)))
        
        if ordered.index(final_stage) <= ordered.index(icf_stage): st.sidebar.error("Final stage must be after ICF stage."); st.stop()

        start_m = (df["Submission_Month"].max() + 1) if not df.empty else pd.Period(datetime.now(), "M")
        future_months = pd.period_range(start=start_m, periods=horizon, freq="M")
        
        st.subheader("Future Spend & CPQR"); colA, colB = st.columns(2)
        spend_df = colA.data_editor(pd.DataFrame({"Month": future_months.astype(str), "Ad Spend": [20000.0]*len(future_months)}), use_container_width=True, num_rows="fixed")
        cpqr_df = colB.data_editor(pd.DataFrame({"Month": future_months.astype(str), "CPQR": [120.0]*len(future_months)}), use_container_width=True, num_rows="fixed")
        spend_dict = {pd.Period(r["Month"], "M"): r["Ad Spend"] for _, r in spend_df.iterrows()}
        cpqr_dict = {pd.Period(r["Month"], "M"): r["CPQR"] for _, r in cpqr_df.iterrows()}

        if rate_mode == "Manual":
            st.subheader("Manual Conversion Rates (%)"); eff_rates = {}
            cols = st.columns(3)
            for i, (a,b) in enumerate(zip(ordered, ordered[1:])):
                eff_rates[f"{a} -> {b}"] = cols[i%3].slider(f"{a} → {b}", 0.0,100.0, 50.0, 0.1)/100
            rates_desc, maturity_applied = "Manual Input", None
        else:
            manual_rate_placeholders = {f"{a} -> {b}": 0.5 for a, b in zip(ordered, ordered[1:])}
            eff_rates, rates_desc, maturity_applied = calculate_maturity_adjusted_rates(df, ordered, ts_map, lags, manual_rate_placeholders, is_rolling="Rolling" in rate_mode, window_months=int(roll_months or 3))

        if st.button("🚀 Run Combined Calculation"):
            terminal_extras = [s for s in ordered if s.lower() in ["screen failed", "lost"]]
            pipe_res = pipeline_projection(df, ordered, ts_map, eff_rates, lags, icf_stage, final_stage, terminal_extras)
            new_df = projections_new_leads(df, ordered, ts_map, horizon, spend_dict, cpqr_dict, eff_rates, lags, icf_stage, final_stage)
            combined = combine_and_display(pipe_res, new_df, final_stage)
            
            st.success("Done.")
            st.header("Results")
            with st.expander(f"Rates Used in This Forecast ({rates_desc})"):
                st.json({k: f"{v*100:.1f}%" for k, v in eff_rates.items()})
                if maturity_applied: st.write("Maturity Period Applied (Days):"); st.json({k: f"{v:.0f}" for k, v in maturity_applied.items()})
            
            c1,c2,c3 = st.columns(3)
            c1.metric("Total Future ICFs", f"{combined['Total Projected ICFs'].sum():.1f}")
            c2.metric("... from New Leads", f"{combined['ICFs from New Leads'].sum():.1f}")
            c3.metric("... from Pipeline", f"{combined['ICFs from Pipeline'].sum():.1f} (Total Yield: {pipe_res['total_icf_yield']:.1f})")

            st.subheader("📅 Combined Monthly Landings")
            display_cols = ["ICFs from New Leads", "ICFs from Pipeline", "Total Projected ICFs", f"{final_stage} from New Leads", f"{final_stage} from Pipeline", f"Total Projected {final_stage}"]
            st.dataframe(combined[display_cols].round(1).style.format("{:.1f}"), use_container_width=True)

            st.subheader("🔬 Pipeline Breakdown")
            narrative = generate_pipeline_narrative(pipe_res['in_flight_df'], ordered, eff_rates, lags, final_stage)
            if not narrative: st.info("No leads are currently in-flight.")
            else:
                for step in narrative:
                    with st.expander(f"From '{step['current_stage']}' ({step['count']} leads)"):
                        st.metric(f"Leads Currently At Stage", f"{step['count']:,}")
                        for proj in step['downstream']:
                            st.info(f"**~{proj['count']:.1f}** → **'{proj['stage']}'** (in ~{proj['lag']:.0f} days)", icon="➡️")
            
            with st.expander("Downloads"):
                st.download_button("⬇️ Combined", combined.to_csv().encode("utf-8"), "combined.csv")

    except Exception as e: st.error(f"An error occurred: {e}")
else: st.info("Upload your two files on the left to get started.")

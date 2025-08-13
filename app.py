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
    subm_col = next((c for c in ["Submitted On", "Referral Date", "SubmittedOn"] if c in df.columns), None)
    if subm_col is None: raise ValueError("Could not find a submission date column.")
    df["Submitted On_DT"] = pd.to_datetime(df[subm_col], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["Submitted On_DT"]).copy()
    df["Submission_Month"] = df["Submitted On_DT"].dt.to_period("M")

    stage_hist_col = next((c for c in df.columns if "stage history" in c.lower()), None)
    status_hist_col = next((c for c in df.columns if "status history" in c.lower()), None)
    if not stage_hist_col and not status_hist_col: raise ValueError("Need a stage or status history column.")

    parsed_stage = df[stage_hist_col].apply(parse_history_blob) if stage_hist_col else [[]]*len(df)
    parsed_status = df[status_hist_col].apply(parse_history_blob) if status_hist_col else [[]]*len(df)

    status_to_stage = {s: stg for stg, statuses in funnel_def.items() for s in statuses}
    ts_frame = pd.DataFrame(index=df.index, columns=list(ts_map.values()), dtype="datetime64[ns]")
    for i in df.index:
        events = sorted(list(parsed_stage.get(i, [])) + list(parsed_status.get(i, [])), key=lambda x: x[1])
        for name, when in events:
            stg = status_to_stage.get(name, name if name in ordered_stages else None)
            if stg in ordered_stages:
                col = ts_map[stg]
                if pd.isna(ts_frame.at[i, col]): ts_frame.at[i, col] = when
    out = pd.concat([df.reset_index(drop=True), ts_frame.reset_index(drop=True)], axis=1)
    return out

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
def overall_rates(df, ordered_stages, ts_map):
    rates = {}
    for a, b in zip(ordered_stages, ordered_stages[1:]):
        ca, cb = ts_map.get(a), ts_map.get(b)
        denom = df[ca].notna().sum() if ca in df.columns else 0
        num = (df[ca].notna() & df[cb].notna()).sum() if ca in df.columns and cb in df.columns else 0
        rates[f"{a} -> {b}"] = float(num / denom) if denom else 0.0
    return rates

@st.cache_data
def rolling_rates(df, ordered_stages, ts_map, months: int):
    rates = {}
    ts_cols = [c for c in df.columns if c.startswith("TS_")]
    anchor = pd.to_datetime(df[ts_cols].max(axis=1).max()) if not df.empty else pd.Timestamp(datetime.now())
    if pd.isna(anchor): anchor = pd.Timestamp(datetime.now())
    cutoff = anchor - pd.DateOffset(months=months)
    for a, b in zip(ordered_stages, ordered_stages[1:]):
        ca, cb = ts_map.get(a), ts_map.get(b)
        denom = (df[ca].notna() & (df[ca] >= cutoff)).sum() if ca in df.columns else 0
        num = (df[ca].notna() & (df[ca] >= cutoff) & df[cb].notna()).sum() if ca in df.columns and cb in df.columns else 0
        rates[f"{a} -> {b}"] = float(num / denom) if denom else 0.0
    return rates

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
        p_to_icf, lag_to_icf = 1.0, 0.0
        for i in range(start_idx, icf_idx): p_to_icf *= rates.get(f"{ordered_stages[i]}->{ordered_stages[i+1]}",0); lag_to_icf += lags.get(f"{ordered_stages[i]}->{ordered_stages[i+1]}",30)
        p_icf_to_final, lag_icf_to_final = 1.0, 0.0
        for i in range(icf_idx, final_idx): p_icf_to_final *= rates.get(f"{ordered_stages[i]}->{ordered_stages[i+1]}",0); lag_icf_to_final += lags.get(f"{ordered_stages[i]}->{ordered_stages[i+1]}",30)
        if p_to_icf > 0:
            icf_month = pd.Period(row['current_ts'] + timedelta(days=lag_to_icf), "M")
            fin_month = pd.Period(row['current_ts'] + timedelta(days=lag_to_icf+lag_icf_to_final), "M")
            records.append((icf_month, p_to_icf, fin_month, p_to_icf * p_icf_to_final))
    
    if not records: return default
    rec_df = pd.DataFrame(records, columns=["icf_m", "p_icf", "fin_m", "p_final"])
    idx = pd.period_range(start=datetime.now(), periods=36, freq="M")
    out = pd.DataFrame(0.0, index=idx, columns=["ICFs from Pipeline", f"{final_stage} from Pipeline"])
    out["ICFs from Pipeline"] = out.index.map(rec_df.groupby("icf_m")["p_icf"].sum()).fillna(0)
    out[f"{final_stage} from Pipeline"] = out.index.map(rec_df.groupby("fin_m")["p_final"].sum()).fillna(0)
    return {'results_df': out, 'total_icf_yield': rec_df['p_icf'].sum(), 'total_enroll_yield': rec_df['p_final'].sum(), 'in_flight_df': in_flight}

# --- NEW NARRATIVE FUNCTION ---
@st.cache_data
def generate_pipeline_narrative(in_flight_df, ordered_stages, rates, lags, final_stage):
    if in_flight_df.empty: return []
    narrative = []
    stage_counts = in_flight_df['current_stage'].value_counts()
    for stage_name in ordered_stages:
        if stage_name == final_stage or stage_name not in stage_counts: continue
        
        count = stage_counts[stage_name]
        downstream = []
        p_cum = 1.0
        lag_cum = 0.0
        start_idx = ordered_stages.index(stage_name)

        for i in range(start_idx, len(ordered_stages) - 1):
            a, b = ordered_stages[i], ordered_stages[i+1]
            rate = rates.get(f"{a} -> {b}", 0.0)
            lag = lags.get(f"{a} -> {b}", 0.0)
            p_cum *= rate
            lag_cum += lag
            downstream.append({'stage': b, 'count': count * p_cum, 'lag': lag_cum})
        
        narrative.append({'current_stage': stage_name, 'count': count, 'downstream': downstream})
    return narrative

@st.cache_data
def projections_new_leads(processed_df, ordered_stages, ts_map, horizon_months, spend_by_month, cpqr_by_month, rates, lags, icf_stage, final_stage):
    start_month = (processed_df["Submission_Month"].max() + 1) if not processed_df.empty else pd.Period(datetime.now(), "M")
    future_months = pd.period_range(start=start_month, periods=horizon_months, freq="M")
    cohorts = pd.DataFrame(index=future_months)
    cohorts["Ad_Spend"] = [float(spend_by_month.get(m, 0.0)) for m in future_months]
    cohorts["CPQR"] = [float(cpqr_by_month.get(m, 120.0)) for m in future_months]
    cohorts["New_QLs"] = (cohorts["Ad_Spend"] / cohorts["CPQR"].replace(0, np.nan)).fillna(0.0)

    p_to_icf = product_rate_to(icf_stage, ordered_stages, rates)
    lag_to_icf = total_lag_to(icf_stage, ordered_stages, lags)
    p_to_final = product_rate_to(final_stage, ordered_stages, rates)
    lag_to_final = total_lag_to(final_stage, ordered_stages, lags)

    cohorts["ICFs_generated"] = cohorts["New_QLs"] * p_to_icf
    cohorts["Final_generated"] = cohorts["New_QLs"] * p_to_final

    max_lag_days = lag_to_final + 30
    idx = pd.period_range(start=start_month, periods=horizon_months + int(max_lag_days/30) + 1, freq="M")
    out = pd.DataFrame(0.0, index=idx, columns=["ICFs from New Leads", f"{final_stage} from New Leads"])
    for m, row in cohorts.iterrows():
        if row["ICFs_generated"] > 0: out.loc[(m.to_timestamp() + timedelta(days=lag_to_icf)).to_period("M"), "ICFs from New Leads"] += row["ICFs_generated"]
        if row["Final_generated"] > 0: out.loc[(m.to_timestamp() + timedelta(days=lag_to_final)).to_period("M"), f"{final_stage} from New Leads"] += row["Final_generated"]
    return out

def combine_monthly_tables(pipeline_df, new_df, final_stage_name):
    idx = pipeline_df.index.union(new_df.index)
    both = pd.DataFrame(index=idx)
    both["ICFs from Pipeline"] = pipeline_df.get("ICFs from Pipeline", 0.0).reindex(idx).fillna(0.0)
    both[f"{final_stage_name} from Pipeline"] = pipeline_df.get(f"{final_stage_name} from Pipeline", 0.0).reindex(idx).fillna(0.0)
    both["ICFs from New Leads"] = new_df.get("ICFs from New Leads", 0.0).reindex(idx).fillna(0.0)
    both[f"{final_stage_name} from New Leads"] = new_df.get(f"{final_stage_name} from New Leads", 0.0).reindex(idx).fillna(0.0)
    both["Total Projected ICFs"] = both["ICFs from Pipeline"] + both["ICFs from New Leads"]
    both[f"Total Projected {final_stage_name}"] = both[f"{final_stage_name} from Pipeline"] + both[f"{final_stage_name} from New Leads"]
    both["Cumulative ICFs"] = both["Total Projected ICFs"].cumsum()
    both[f"Cumulative {final_stage_name}"] = both[f"Total Projected {final_stage_name}"].cumsum()
    return both

def best_guess_stage(ordered_stages, keywords, fallback_idx):
    for s in reversed(ordered_stages):
        if any(k in s.lower() for k in keywords): return s
    return ordered_stages[fallback_idx] if len(ordered_stages) > abs(fallback_idx) else ordered_stages[-1]

# ==============================
# Main App
# ==============================
with st.sidebar:
    st.header("1) Upload")
    ref_csv, funnel_csv = st.file_uploader("Referral Data (CSV)", type=["csv"]), st.file_uploader("Funnel Definition", type=["csv", "tsv"])
    st.header("2) Rates"); rate_mode = st.radio("Rates from?", ["Manual", "Overall", "Rolling"], index=2, horizontal=True)
    roll_months = st.slider("Rolling window (months)", 1, 12, 3) if rate_mode == "Rolling" else None
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
        
        st.subheader("Future Spend & CPQR")
        colA, colB = st.columns(2)
        spend_df = colA.data_editor(pd.DataFrame({"Month": future_months.astype(str), "Ad Spend": [0.0]*len(future_months)}), use_container_width=True, num_rows="fixed")
        cpqr_df = colB.data_editor(pd.DataFrame({"Month": future_months.astype(str), "CPQR": [120.0]*len(future_months)}), use_container_width=True, num_rows="fixed")
        spend_dict = {pd.Period(r["Month"], "M"): r["Ad Spend"] for _, r in spend_df.iterrows()}
        cpqr_dict = {pd.Period(r["Month"], "M"): r["CPQR"] for _, r in cpqr_df.iterrows()}

        if rate_mode == "Overall": eff_rates = overall_rates(df, ordered, ts_map)
        elif rate_mode == "Rolling": eff_rates = rolling_rates(df, ordered, ts_map, int(roll_months or 3))
        else: # Manual
            defaults = overall_rates(df, ordered, ts_map)
            st.subheader("Manual Conversion Rates (%)")
            eff_rates = {}
            cols = st.columns(3)
            for i, (a,b) in enumerate(zip(ordered, ordered[1:])):
                default = defaults.get(f"{a} -> {b}", 0.0) * 100
                eff_rates[f"{a} -> {b}"] = cols[i%3].slider(f"{a} → {b}", 0.0,100.0, round(default,1), 0.1)/100

        if st.button("🚀 Run Combined Calculation"):
            terminal_extras = [s for s in ordered if s.lower() in ["screen failed", "lost"]]
            pipe_res = pipeline_projection(df, ordered, ts_map, eff_rates, lags, icf_stage, final_stage, terminal_extras)
            new_df = projections_new_leads(df, ordered, ts_map, horizon, spend_dict, cpqr_dict, eff_rates, lags, icf_stage, final_stage)
            combined = combine_monthly_tables(pipe_res['results_df'], new_df, final_stage)

            st.success("Done.")
            st.subheader("📅 Combined Monthly Landings")
            display = combined.rename(columns={f"{final_stage} from Pipeline": f"Final Stage from Pipeline", f"{final_stage} from New Leads": f"Final Stage from New Leads", f"Total Projected {final_stage}": f"Total Projected Final Stage", f"Cumulative {final_stage}": f"Cumulative Final Stage"})
            display_cols = ["ICFs from New Leads", "ICFs from Pipeline", "Total Projected ICFs", "Final Stage from New Leads", "Final Stage from Pipeline", "Total Projected Final Stage"]
            st.dataframe(display[display_cols].round(1).style.format("{:.1f}"), use_container_width=True)

            # --- NEW: Pipeline Breakdown Display ---
            st.subheader("🔬 Pipeline Breakdown")
            st.caption("This shows how the **'ICFs from Pipeline'** and **'Final Stage from Pipeline'** totals are derived from leads currently in your funnel.")
            narrative_data = generate_pipeline_narrative(pipe_res['in_flight_df'], ordered, eff_rates, lags, final_stage)
            if not narrative_data:
                st.info("No leads are currently in-flight in the pipeline.")
            else:
                for step in narrative_data:
                    with st.expander(f"From '{step['current_stage']}' ({step['count']} leads)"):
                        st.metric(f"Leads Currently At This Stage", f"{step['count']:,}")
                        st.write("From this group, we project:")
                        for proj in step['downstream']:
                            time_text = f" in **~{proj['lag']:.0f} days**" if proj['lag'] > 0 else ""
                            st.info(f"**~{proj['count']:.1f}** will advance to **'{proj['stage']}'**{time_text}", icon="➡️")
            # --- End of New Section ---
            
            with st.expander("Downloads"):
                st.download_button("⬇️ Combined", combined.to_csv().encode("utf-8"), "combined.csv", "text/csv")
                st.download_button("⬇️ Pipeline", pipe_res['results_df'].to_csv().encode("utf-8"), "pipeline.csv", "text/csv")
                st.download_button("⬇️ New Leads", new_df.to_csv().encode("utf-8"), "new_leads.csv", "text/csv")

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
else:
    st.info("Upload your two files on the left to get started.")

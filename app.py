import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta

# ==============================
# Page setup
# ==============================
st.set_page_config(page_title="Recruitment Combiner", layout="wide")
st.title("🧮 Recruitment Combiner — Pipeline + Projections")
st.caption("Recreate the original Funnel Analysis (pipeline-only) and Projections (new-lead what‑if) **in the background**, then sum their monthly outputs (ICFs + final-stage).")

# ==============================
# Helpers
# ==============================
def _strip_us_tz(s: str) -> str:
    tz_pattern = r"\s+(?:EST|EDT|CST|CDT|MST|MDT|PST|PDT)$"
    return re.sub(tz_pattern, "", s.strip())

def parse_dt(s):
    if pd.isna(s):
        return pd.NaT
    return pd.to_datetime(_strip_us_tz(str(s)), errors="coerce")

def parse_history_blob(blob: str):
    """Parse lines like 'Status Name: 1/2/2025 14:05 EDT' into list[(name, datetime)]"""
    if pd.isna(blob):
        return []
    events = []
    pat = re.compile(r"^(.+?):\s*(.+)$")
    for raw in str(blob).split("\n"):
        m = pat.match(raw.strip())
        if not m:
            continue
        name, dts = m.groups()
        dt = parse_dt(dts)
        if name and pd.notna(dt):
            events.append((name.strip(), dt))
    events.sort(key=lambda x: x[1])
    return events

@st.cache_data
def parse_funnel_definition(uploaded_file):
    """Auto-detect CSV/TSV. First row = ordered stage names. Rows 2+ in each column = status aliases for that stage."""
    if uploaded_file is None:
        return None, None, None
    import io
    content = uploaded_file.getvalue().decode("utf-8", errors="replace")
    df = pd.read_csv(io.StringIO(content), sep=None, engine="python", header=None)
    funnel_def, ordered, ts_map = {}, [], {}
    for col in df.columns:
        col_series = df[col].dropna().astype(str).str.strip().str.replace('"', '', regex=False)
        if col_series.empty:
            continue
        stage_name = col_series.iloc[0]
        if not stage_name:
            continue
        ordered.append(stage_name)
        statuses = [s for s in col_series.iloc[1:] if s]
        if stage_name not in statuses:
            statuses.append(stage_name)
        funnel_def[stage_name] = statuses
        ts_map[stage_name] = "TS_" + re.sub(r"[^A-Za-z0-9]+", "_", stage_name).strip("_")
    if not ordered:
        return None, None, None
    return funnel_def, ordered, ts_map

@st.cache_data
def preprocess_referrals(csv_file, funnel_def, ordered_stages, ts_map):
    """Return df with TS_* columns and Submission_Month (Period[M])."""
    import io
    content = csv_file.getvalue().decode("utf-8", errors="replace")
    df = pd.read_csv(io.StringIO(content))
    # pick submission column
    subm_col = None
    for cand in ["Submitted On", "Referral Date", "SubmittedOn", "submitted_on"]:
        if cand in df.columns:
            subm_col = cand
            break
    if subm_col is None:
        raise ValueError("Could not find 'Submitted On' or 'Referral Date' column in referral CSV.")
    df["Submitted On_DT"] = pd.to_datetime(df[subm_col], errors="coerce").dt.tz_localize(None)
    df = df[~df["Submitted On_DT"].isna()].copy()
    df["Submission_Month"] = df["Submitted On_DT"].dt.to_period("M")

    # parse histories
    stage_hist_col = None
    status_hist_col = None
    for c in df.columns:
        cl = c.lower()
        if "stage history" in cl and stage_hist_col is None:
            stage_hist_col = c
        if "status history" in cl and status_hist_col is None:
            status_hist_col = c
    if stage_hist_col is None and status_hist_col is None:
        raise ValueError("Need at least one of 'Lead Stage History' or 'Lead Status History' columns.")

    parsed_stage = df[stage_hist_col].apply(parse_history_blob) if stage_hist_col in df.columns else [[]]*len(df)
    parsed_status = df[status_hist_col].apply(parse_history_blob) if status_hist_col in df.columns else [[]]*len(df)

    # status -> stage mapping
    status_to_stage = {}
    for stg, statuses in funnel_def.items():
        for s in statuses:
            status_to_stage[s] = stg

    # build TS_* first-occurrence timestamps for each stage
    ts_frame = pd.DataFrame(index=df.index, columns=[ts_map[s] for s in ordered_stages], dtype="datetime64[ns]")
    for i in df.index:
        events = list(parsed_stage[i]) + list(parsed_status[i])
        events.sort(key=lambda x: x[1])
        for name, when in events:
            stg = status_to_stage.get(name, name if name in ordered_stages else None)
            if stg in ordered_stages:
                col = ts_map[stg]
                if pd.isna(ts_frame.at[i, col]):
                    ts_frame.at[i, col] = when
    out = pd.concat([df.reset_index(drop=True), ts_frame.reset_index(drop=True)], axis=1)
    # ensure optional cols exist
    for c in ["Site", "UTM Source", "UTM Medium"]:
        if c not in out.columns:
            out[c] = np.nan
    return out

@st.cache_data
def calc_inter_stage_lags(df, ordered_stages, ts_map):
    """Average days between consecutive stages; fallback to 30 if insufficient data."""
    lags = {}
    for a, b in zip(ordered_stages, ordered_stages[1:]):
        ca, cb = ts_map[a], ts_map[b]
        if ca in df.columns and cb in df.columns:
            mask = df[ca].notna() & df[cb].notna()
            if mask.any():
                d = (df.loc[mask, cb] - df.loc[mask, ca]).dt.total_seconds() / (24*3600)
                lags[f"{a} -> {b}"] = float(np.nanmean(d.clip(lower=0)))
                continue
        lags[f"{a} -> {b}"] = 30.0
    return lags

@st.cache_data
def overall_rates(df, ordered_stages, ts_map):
    """Compute simple overall stage-to-stage conversion rates from history (no recency filter)."""
    rates = {}
    for a, b in zip(ordered_stages, ordered_stages[1:]):
        ca, cb = ts_map[a], ts_map[b]
        have_a = df[ca].notna()
        have_b = df[cb].notna()
        denom = have_a.sum()
        num = (have_a & have_b).sum()
        rates[f"{a} -> {b}"] = float(num / denom) if denom else 0.0
    return rates

@st.cache_data
def rolling_rates(df, ordered_stages, ts_map, months: int):
    """Compute stage-to-stage conversion using only rows whose A-stage timestamp is within the last N months."""
    rates = {}
    if df.empty:
        return {f"{a} -> {b}": 0.0 for a, b in zip(ordered_stages, ordered_stages[1:])}
    # Use max observed timestamp as 'now' anchor to align with dataset
    anchor = pd.to_datetime(df[[c for c in df.columns if c.startswith("TS_")]].max(axis=1).max())
    if pd.isna(anchor):
        anchor = pd.Timestamp(datetime.now())
    cutoff = anchor - pd.DateOffset(months=months)
    for a, b in zip(ordered_stages, ordered_stages[1:]):
        ca, cb = ts_map[a], ts_map[b]
        have_a_recent = df[ca].notna() & (df[ca] >= cutoff)
        denom = have_a_recent.sum()
        num = (have_a_recent & df[cb].notna()).sum()
        rates[f"{a} -> {b}"] = float(num / denom) if denom else 0.0
    return rates

def product_rate_to(target_stage, ordered_stages, rates):
    """Product of rates from stage 0 up to target_stage (inclusive end)."""
    try:
        idx = ordered_stages.index(target_stage)
    except ValueError:
        return 0.0
    p = 1.0
    for i in range(idx):
        a, b = ordered_stages[i], ordered_stages[i+1]
        p *= float(rates.get(f"{a} -> {b}", 0.0))
    return p

def total_lag_to(target_stage, ordered_stages, lags):
    """Sum of lags from stage 0 up to target_stage (inclusive end)."""
    try:
        idx = ordered_stages.index(target_stage)
    except ValueError:
        return 0.0
    s = 0.0
    for i in range(idx):
        a, b = ordered_stages[i], ordered_stages[i+1]
        s += float(lags.get(f"{a} -> {b}", 30.0))
    return s

@st.cache_data
def pipeline_projection(df, ordered_stages, ts_map, rates, lags, icf_stage, final_stage, terminal_extras):
    """Recreate Funnel Analysis 'Projected Monthly Landings (Future)' for pipeline only (no new leads)."""
    terminal_stages = set([final_stage] + list(terminal_extras))
    term_cols = [ts_map.get(s) for s in terminal_stages if ts_map.get(s) in df.columns]
    is_terminal = pd.Series(False, index=df.index)
    for c in term_cols:
        is_terminal |= df[c].notna()
    in_flight = df.loc[~is_terminal].copy()
    # find current stage per row (latest non-null TS_* among ordered non-terminal stages)
    non_terminal_stages = [s for s in ordered_stages if s not in terminal_stages]
    ts_cols = [ts_map[s] for s in non_terminal_stages if ts_map.get(s) in df.columns]
    if not ts_cols:
        return pd.DataFrame(columns=["ICFs from Pipeline", "Final from Pipeline"])
    in_flight["curr_t"] = in_flight[ts_cols].max(axis=1)

    records = []
    for _, row in in_flight.iterrows():
        last_stage = None
        last_time = row["curr_t"]
        for s in reversed(non_terminal_stages):
            c = ts_map[s]
            if c in in_flight.columns and pd.notna(row[c]):
                last_stage = s
                break
        if last_stage is None or pd.isna(last_time):
            continue
        try:
            start_idx = ordered_stages.index(last_stage)
            icf_idx = ordered_stages.index(icf_stage)
            final_idx = ordered_stages.index(final_stage)
        except ValueError:
            continue

        # accumulate to ICF
        p_to_icf = 1.0
        lag_to_icf = 0.0
        for i in range(start_idx, icf_idx):
            a, b = ordered_stages[i], ordered_stages[i+1]
            p_to_icf *= float(rates.get(f"{a} -> {b}", 0.0))
            lag_to_icf += float(lags.get(f"{a} -> {b}", 30.0))

        # from ICF to final
        p_icf_to_final = 1.0
        lag_icf_to_final = 0.0
        for i in range(icf_idx, final_idx):
            a, b = ordered_stages[i], ordered_stages[i+1]
            p_icf_to_final *= float(rates.get(f"{a} -> {b}", 0.0))
            lag_icf_to_final += float(lags.get(f"{a} -> {b}", 30.0))

        p_to_final = p_to_icf * p_icf_to_final
        if p_to_icf > 0:
            icf_month = pd.Period(last_time + timedelta(days=lag_to_icf), "M")
            fin_month = pd.Period(last_time + timedelta(days=lag_to_icf + lag_icf_to_final), "M")
            records.append((icf_month, p_to_icf, fin_month, p_to_final))

    if not records:
        idx = pd.period_range(start=pd.Period(datetime.now(), "M"), periods=6, freq="M")
        return pd.DataFrame(0.0, index=idx, columns=["ICFs from Pipeline", "Final from Pipeline"])

    rec_df = pd.DataFrame(records, columns=["icf_m", "p_icf", "fin_m", "p_final"])
    idx = pd.period_range(start=min(rec_df["icf_m"].min(), rec_df["fin_m"].min()),
                          end=max(rec_df["icf_m"].max(), rec_df["fin_m"].max()) + 6, freq="M")
    out = pd.DataFrame(0.0, index=idx, columns=["ICFs from Pipeline", "Final from Pipeline"])
    out.loc[rec_df.groupby("icf_m").size().index, "ICFs from Pipeline"] = rec_df.groupby("icf_m")["p_icf"].sum()
    out.loc[rec_df.groupby("fin_m").size().index, "Final from Pipeline"] = rec_df.groupby("fin_m")["p_final"].sum()
    return out

@st.cache_data
def projections_new_leads(processed_df, ordered_stages, ts_map, horizon_months, spend_by_month, cpqr_by_month,
                          rates, lags, icf_stage, final_stage):
    """Recreate Projections table + add final-stage from new leads."""
    start_month = (processed_df["Submission_Month"].max() + 1) if "Submission_Month" in processed_df else pd.Period(datetime.now(), "M")
    future_months = pd.period_range(start=start_month, periods=horizon_months, freq="M")
    cohorts = pd.DataFrame(index=future_months)
    cohorts["Ad_Spend"] = [float(spend_by_month.get(m, 0.0)) for m in future_months]
    cohorts["CPQR"] = [float(cpqr_by_month.get(m, 120.0)) for m in future_months]
    cohorts["New_QLs"] = (cohorts["Ad_Spend"] / cohorts["CPQR"].replace(0, np.nan)).fillna(0.0)

    p_to_icf = product_rate_to(icf_stage, ordered_stages, rates)
    lag_to_icf = total_lag_to(icf_stage, ordered_stages, lags)

    # Multiply rates from ICF to final
    try:
        icf_idx = ordered_stages.index(icf_stage)
        final_idx = ordered_stages.index(final_stage)
    except ValueError:
        icf_idx = None
        final_idx = None
    p_icf_to_final = 1.0
    lag_icf_to_final = 0.0
    if icf_idx is not None and final_idx is not None and final_idx > icf_idx:
        for i in range(icf_idx, final_idx):
            a, b = ordered_stages[i], ordered_stages[i+1]
            p_icf_to_final *= float(rates.get(f"{a} -> {b}", 0.0))
            lag_icf_to_final += float(lags.get(f"{a} -> {b}", 30.0))

    cohorts["ICFs_generated"] = cohorts["New_QLs"] * p_to_icf
    cohorts["Final_generated"] = cohorts["ICFs_generated"] * p_icf_to_final
    cohorts["Cohort_CPICF"] = (cohorts["Ad_Spend"] / cohorts["ICFs_generated"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    # smear into landing months
    land_icf_idx, land_icf_vals = [], []
    land_fin_idx, land_fin_vals = [], []
    for m, row in cohorts.iterrows():
        icf_land_m = (m.to_timestamp() + timedelta(days=lag_to_icf)).to_period("M")
        fin_land_m = (m.to_timestamp() + timedelta(days=lag_to_icf + lag_icf_to_final)).to_period("M")
        land_icf_idx.append(icf_land_m); land_icf_vals.append(row["ICFs_generated"])
        land_fin_idx.append(fin_land_m); land_fin_vals.append(row["Final_generated"])

    idx = pd.period_range(start=min(land_icf_idx + land_fin_idx) if land_icf_idx else start_month, periods=horizon_months + 12, freq="M")
    out = pd.DataFrame(0.0, index=idx, columns=["ICFs from New Leads", "Final from New Leads"])
    if land_icf_idx:
        icf_series = pd.Series(land_icf_vals, index=land_icf_idx).groupby(level=0).sum()
        out.loc[icf_series.index, "ICFs from New Leads"] = icf_series.values
    if land_fin_idx:
        fin_series = pd.Series(land_fin_vals, index=land_fin_idx).groupby(level=0).sum()
        out.loc[fin_series.index, "Final from New Leads"] = fin_series.values
    return out, cohorts

def combine_monthly_tables(pipeline_df, new_df):
    idx = pipeline_df.index.union(new_df.index)
    both = pd.DataFrame(index=idx)
    for col in ["ICFs from Pipeline", "Final from Pipeline"]:
        both[col] = pipeline_df.get(col, 0.0).reindex(idx).fillna(0.0)
    for col in ["ICFs from New Leads", "Final from New Leads"]:
        both[col] = new_df.get(col, 0.0).reindex(idx).fillna(0.0)
    both["Total Projected ICFs"] = both["ICFs from Pipeline"] + both["ICFs from New Leads"]
    both["Total Projected Final"] = both["Final from Pipeline"] + both["Final from New Leads"]
    both["Cumulative ICFs"] = both["Total Projected ICFs"].cumsum()
    both["Cumulative Final"] = both["Total Projected Final"].cumsum()
    return both

def best_guess_icf_stage(ordered_stages):
    for s in ordered_stages:
        if "icf" in s.lower() or "consent" in s.lower():
            return s
    return ordered_stages[-2] if len(ordered_stages) >= 2 else ordered_stages[-1]

def best_guess_final_stage(ordered_stages):
    for s in reversed(ordered_stages):
        sl = s.lower()
        if "enroll" in sl or "random" in sl or "randomization" in sl or "randomised" in sl or "randomized" in sl:
            return s
    return ordered_stages[-1]

# ==============================
# UI — Inputs
# ==============================
with st.sidebar:
    st.header("1) Upload")
    ref_csv = st.file_uploader("Referral Data (CSV)", type=["csv"], key="refcsv")
    funnel_csv = st.file_uploader("Funnel Definition (CSV/TSV)", type=["csv", "tsv"], key="fundef")

    st.header("2) Stage Mapping")
    st.caption("Pick which stage is **ICF signed** and which is your **final outcome** (Randomization/Enrollment).")

    st.header("3) Rates")
    rate_mode = st.radio("How to compute conversion rates?", ["Manual", "Overall (all history)", "Rolling window"], index=2)
    roll_months = st.slider("Rolling window (months)", 1, 12, 3) if rate_mode == "Rolling window" else None
    st.caption("Manual lets you set each adjacent stage→stage %; Rolling computes using only recent rows.")

    st.header("4) Projections")
    horizon = st.number_input("Projection horizon (months)", min_value=1, max_value=48, value=12, step=1)
    st.caption("Enter monthly Spend and CPQR; defaults repeat.")

    st.header("5) Labels (display only)")
    final_label = st.radio("Final stage label for outputs", ["Randomizations", "Enrollments"], index=0)

if ref_csv and funnel_csv:
    try:
        funnel_def, ordered_stages, ts_map = parse_funnel_definition(funnel_csv)
        if not funnel_def:
            st.error("Could not parse the funnel definition file.")
            st.stop()

        # Stage pickers
        col1, col2 = st.columns(2)
        with col1:
            icf_stage = st.selectbox("ICF stage", options=ordered_stages, index=ordered_stages.index(best_guess_icf_stage(ordered_stages)))
        with col2:
            final_stage = st.selectbox("Final outcome stage", options=ordered_stages, index=ordered_stages.index(best_guess_final_stage(ordered_stages)))

        if ordered_stages.index(final_stage) <= ordered_stages.index(icf_stage):
            st.error("Final outcome stage must come **after** the ICF stage in your funnel order.")
            st.stop()

        df = preprocess_referrals(ref_csv, funnel_def, ordered_stages, ts_map)
        lags = calc_inter_stage_lags(df, ordered_stages, ts_map)

        # Prepare spend & CPQR editors
        start_m = (df["Submission_Month"].max() + 1) if "Submission_Month" in df else pd.Period(datetime.now(), "M")
        future_months_ui = pd.period_range(start=start_m, periods=horizon, freq="M")

        st.subheader("Spend & CPQR (for Projections)")
        colA, colB = st.columns(2)
        with colA:
            spend_df = pd.DataFrame({"Month": future_months_ui.astype(str), "Ad_Spend": [0.0]*len(future_months_ui)})
            spend_df = st.data_editor(spend_df, use_container_width=True, num_rows="fixed", key="spend_editor")
            spend_dict = {pd.Period(r["Month"], "M"): float(r["Ad_Spend"]) for _, r in spend_df.iterrows()}
        with colB:
            cpqr_df = pd.DataFrame({"Month": future_months_ui.astype(str), "CPQR": [120.0]*len(future_months_ui)})
            cpqr_df = st.data_editor(cpqr_df, use_container_width=True, num_rows="fixed", key="cpqr_editor")
            cpqr_dict = {pd.Period(r["Month"], "M"): float(r["CPQR"]) for _, r in cpqr_df.iterrows()}

        # Compute base rates according to mode, then allow manual overrides if Manual
        if rate_mode == "Overall (all history)":
            eff_rates = overall_rates(df, ordered_stages, ts_map)
        elif rate_mode == "Rolling window":
            eff_rates = rolling_rates(df, ordered_stages, ts_map, months=int(roll_months or 3))
        else:
            # Manual: build sliders for each adjacent pair
            default_rates = overall_rates(df, ordered_stages, ts_map)
            st.subheader("Manual Conversion Rates (% from → to)")
            eff_rates = {}
            cols = st.columns(3)
            for idx, (a, b) in enumerate(zip(ordered_stages, ordered_stages[1:])):
                with cols[idx % 3]:
                    default_pct = float(default_rates.get(f"{a} -> {b}", 0.0)) * 100.0
                    val = st.slider(f"{a} → {b}", 0.0, 100.0, float(round(default_pct, 1)), 0.1) / 100.0
                    eff_rates[f"{a} -> {b}"] = val

        # Button to run
        if st.button("Run Combined Calculation"):
            # Terminal extras: treat Screen Failed and Lost as terminal even if not present in mapping
            terminal_extras = [s for s in ordered_stages if s.lower() in ["screen failed", "lost"]]

            pipe_df = pipeline_projection(df, ordered_stages, ts_map, eff_rates, lags, icf_stage, final_stage, terminal_extras)
            new_df, cohorts = projections_new_leads(df, ordered_stages, ts_map, horizon, spend_dict, cpqr_dict, eff_rates, lags, icf_stage, final_stage)
            combined = combine_monthly_tables(pipe_df, new_df)

            # Display label mapping for final stage
            renamed = combined.rename(columns={
                "Final from Pipeline": f"{final_label} from Pipeline",
                "Final from New Leads": f"{final_label} from New Leads",
                "Total Projected Final": f"Total Projected {final_label}",
                "Cumulative Final": f"Cumulative {final_label}",
            })

            st.success("Done. Final combined monthly table below.")
            st.subheader("📅 Combined Monthly Landings")
            display = renamed.copy()
            display.index = display.index.astype(str)
            st.dataframe(display[[
                "ICFs from New Leads", "ICFs from Pipeline", "Total Projected ICFs",
                f"{final_label} from New Leads", f"{final_label} from Pipeline", f"Total Projected {final_label}",
                "Cumulative ICFs", f"Cumulative {final_label}"
            ]].round(2), use_container_width=True)

            with st.expander("Underlying tables (generated behind the scenes)"):
                st.markdown("**Pipeline-only (Funnel Analysis → Projected Monthly Landings (Future))**")
                tmp = pipe_df.copy(); tmp.index = tmp.index.astype(str); st.dataframe(tmp.round(2), use_container_width=True)
                st.markdown("**New Leads (Projections → Projected Monthly ICFs & Cohort CPICF)**")
                tmp2 = new_df.copy(); tmp2.index = tmp2.index.astype(str); st.dataframe(tmp2.round(2), use_container_width=True)
                st.markdown("**Cohort Table (for reference)**")
                ctmp = cohorts.copy(); ctmp.index = ctmp.index.astype(str); st.dataframe(ctmp.round(2), use_container_width=True)

            # Downloads
            st.download_button("⬇️ Download Combined CSV", renamed.to_csv().encode("utf-8"), file_name="combined_monthly_landings.csv", mime="text/csv")
            st.download_button("⬇️ Download Pipeline CSV", pipe_df.to_csv().encode("utf-8"), file_name="pipeline_projection.csv", mime="text/csv")
            st.download_button("⬇️ Download New-Leads CSV", new_df.to_csv().encode("utf-8"), file_name="new_leads_projection.csv", mime="text/csv")
            st.download_button("⬇️ Download Cohorts CSV", cohorts.to_csv().encode("utf-8"), file_name="cohort_cpicf.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Upload your two files on the left to get started.")

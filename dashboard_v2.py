"""
dashboard_v2.py — Oil Price Impact Dashboard (v2 — presentation build)
City of Chicago — DPS Assets Analysis

Four-tab Streamlit dashboard:
  Tab 1 — Oil Price vs Budget & Spend (merged, toggle: Budget Appropriations / Actual Spend)
  Tab 2 — Fuel Price Sensitivity (tornado, crude sensitivity, historical validation)
  Tab 3 — Budget Scenarios (MC bands, budget-at-risk, waterfall)
  Tab 4 — Component Exposure (oil shock propagation by component)

Run with:  streamlit run dashboard_v2.py
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

_DASHBOARD_DIR = Path(__file__).parent

from simulation import FleetSim, FacilitiesSim, compute_runchart_data, compute_oil_cost_correlation, generate_wti_forecast
from charts import (
    make_tornado_chart,
    make_crude_sensitivity_chart,
    make_historic_fuel_chart,
    make_fleet_component_chart,
    make_fac_component_chart,
    make_combined_component_chart,
    make_mc_bands_chart,
    make_budget_at_risk_chart,
    make_scenario_waterfall,
    make_mc_scenario_range,
    make_oil_delta_bars,
    make_budget_runchart,
    make_savings_gap_chart,
    make_cost_runchart,
    make_cost_breakdown_bar,
    make_component_shock_bar,
    make_oil_cost_chart,
    make_actual_vs_budget_chart,
    make_actual_summary_chart,
)
from config import SCENARIOS, FOCUS_YEARS, BASE_DIESEL_GAL, SCENARIO_COLORS

# ── Actual spend data — Chicago Data Portal (Payments Dataset) ────────────────
# Source: data.cityofchicago.org/Administration-Finance/Payments/s4vu-giwb
# Fleet fuel: Colonial Oil Industries contracts 129971 (diesel) + 129972 (gasoline)
# Note: FY2021 excludes ~$3.1M World Fuel Services residual (contract wind-down)
# ── Actual energy spend dataset (Chicago Payments cleaned) ───────────────────

ENERGY_SPEND_FILE = _DASHBOARD_DIR / "chicago_energy_clean_strict.xlsx"

@st.cache_data
def load_actual_energy():

    df = pd.read_excel(ENERGY_SPEND_FILE, sheet_name="summary_by_year")

    fuel = df[df["type"] == "fuel"].copy()
    elec = df[df["type"] == "electricity"].copy()

    fuel["value_M"] = fuel["total_spend"] / 1e6
    elec["value_M"] = elec["total_spend"] / 1e6

    fuel = fuel.sort_values("year")
    elec = elec.sort_values("year")

    return fuel, elec

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Oil Price Impact — DPS Chicago",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    '<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">',
    unsafe_allow_html=True,
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* ── Root background ── */
    html, body,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMain"] > div,
    .main, .main > div { background-color: #f1f5f9 !important; }

    /* ── Block container ── */
    .block-container {
        padding-top: 3.5rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        background: transparent !important;
        max-width: 1400px !important;
    }

    /* ── Global text ── */
    p, span, li, td, th { color: #334155 !important; font-family: 'Inter', sans-serif !important; }
    h1 { color: #0f4c8a !important; font-size: 1.5rem !important; font-weight: 700 !important; letter-spacing: -0.01em !important; }
    h2 { color: #1d6fb8 !important; font-size: 1.2rem !important; }
    h3 { color: #1d4ed8 !important; font-size: 1.05rem !important; }
    strong, b { color: #0f172a !important; }

    /* ══════════════════════════════════════════
       METRIC CARDS — border-left accent style
    ══════════════════════════════════════════ */
    div[data-testid="metric-container"],
    [data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-left: 3px solid #3b82f6 !important;
        border-radius: 8px !important;
        padding: 16px 20px !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important;
        transition: box-shadow 0.2s ease !important;
    }
    div[data-testid="metric-container"]:hover {
        box-shadow: 0 4px 14px rgba(59,130,246,0.10) !important;
    }
    div[data-testid="metric-container"] [data-testid="stMetricLabel"] p {
        color: #64748b !important;
        font-size: 0.78rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        font-weight: 500 !important;
    }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }
    div[data-testid="metric-container"] [data-testid="stMetricDelta"]   { color: #ea580c !important; }
    div[data-testid="metric-container"] [data-testid="stMetricDelta"] svg { color: #ea580c !important; }

    /* ══════════════════════════════════════════
       PILLS NAV — top-right tab selector
    ══════════════════════════════════════════ */

    /* Container alignment */
    [data-testid="stPills"] {
        display: flex !important;
        justify-content: flex-end !important;
        flex-wrap: wrap !important;
        gap: 4px !important;
        padding-top: 0.25rem !important;
    }
    [data-testid="stPills"] > div {
        display: flex !important;
        flex-wrap: wrap !important;
        justify-content: flex-end !important;
        gap: 4px !important;
    }

    /* Individual pill buttons */
    [data-testid="stPills"] button,
    button[kind="pillsButton"],
    button[data-testid="stPillsButton"] {
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 20px !important;
        color: #ffffff !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        padding: 5px 14px !important;
        white-space: nowrap !important;
        transition: all 0.15s ease !important;
    }
    [data-testid="stPills"] button:hover,
    button[kind="pillsButton"]:hover {
        background: #eff6ff !important;
        border-color: #3b82f6 !important;
        color: #1d4ed8 !important;
    }
    [data-testid="stPills"] button[aria-selected="true"],
    [data-testid="stPills"] button[data-active="true"],
    [data-testid="stPills"] button[kind="pillsActive"],
    button[kind="pillsActive"],
    button[aria-selected="true"][kind="pillsButton"] {
        background: #dbeafe !important;
        border-color: #3b82f6 !important;
        color: #1d4ed8 !important;
        font-weight: 600 !important;
        box-shadow: 0 0 0 1px rgba(59,130,246,0.20) !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div { background-color: #ffffff !important; border-right: 1px solid #e2e8f0 !important; }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div { color: #334155 !important; font-family: 'Inter', sans-serif !important; }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: #0f4c8a !important; }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: #94a3b8 !important; }

    /* ── Expander — hide toggle button, always show content ── */
    [data-testid="stExpander"] {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
    }
    [data-testid="stExpander"] summary { display: none !important; }

    /* ── Selectbox ── */
    [data-testid="stSelectbox"] > div > div {
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        color: #1e293b !important;
        border-radius: 6px !important;
    }
    [data-testid="stSelectbox"] span { color: #1e293b !important; }

    /* ── Radio ── */
    [data-testid="stRadio"] label { color: #334155 !important; }
    [data-testid="stRadio"] [data-testid="stMarkdown"] p { color: #64748b !important; }

    /* ── Slider ── */
    [data-testid="stSlider"] [data-testid="stTickBarMin"],
    [data-testid="stSlider"] [data-testid="stTickBarMax"] { color: #64748b !important; }

    /* ── Dataframes ── */
    [data-testid="stDataFrame"] iframe { filter: none; border-radius: 8px; border: 1px solid #e2e8f0; }

    /* ── Dividers ── */
    hr { border-color: #e2e8f0 !important; margin: 1.2rem 0 !important; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 5px; background: #f1f5f9; }
    ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }

    /* ── Tab section label ── */
    .tab-header { color: #1d4ed8 !important; font-size: 1.0rem !important; font-weight: 600 !important; margin-bottom: 0.3rem !important; }

    /* ── Spinner ── */
    [data-testid="stSpinner"] p { color: #64748b !important; }

    /* ── Hide fullscreen / expand_more button on charts and containers ── */
    [data-testid="StyledFullScreenButton"],
    button[title="View fullscreen"],
    [data-testid="stElementToolbar"],
    [data-testid="stElementToolbarButton"] { display: none !important; }

    /* ── Hide sidebar collapse/expand buttons ── */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarHeader"],
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"] { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ── Load & cache simulations ──────────────────────────────────────────────────
@st.cache_resource(show_spinner="Running simulations...")
def load_sims():
    fleet = FleetSim()
    fac   = FacilitiesSim()
    return fleet, fac

fleet_sim, fac_sim = load_sims()

@st.cache_data(show_spinner="Building runchart data...")
def get_runchart():
    return compute_runchart_data()

@st.cache_data(show_spinner=False)
def get_tables():
    fleet_det  = fleet_sim.det_scenario_table()
    fleet_mc   = fleet_sim.mc_risk_bands()
    fleet_cs   = fleet_sim.crude_sensitivity()
    fleet_hist = fleet_sim.hist_fuel_by_category()
    fleet_comp = fleet_sim.component_breakdown()
    fac_det    = fac_sim.det_scenario_table()
    fac_mc     = fac_sim.mc_risk_bands()
    fac_cs     = fac_sim.crude_sensitivity()
    return fleet_det, fleet_mc, fleet_cs, fleet_hist, fleet_comp, fac_det, fac_mc, fac_cs

fleet_det, fleet_mc, fleet_cs, fleet_hist, fleet_comp, fac_det, fac_mc, fac_cs = get_tables()

# ── Top-level energy budget helpers (shared by Exec Summary + Oil Price vs) ──
_ENERGY_BUDGET_FILE = _DASHBOARD_DIR / "chicago_energy_budget_2016_2025.xlsx"
_ACCT_CLEAN = {
    "MOTOR VEHICLE DIESEL FUEL": "Motor Vehicle Diesel Fuel",
    "MOTOR VEHL DIESEL FUEL":    "Motor Vehicle Diesel Fuel",
    "Motor Vehicle Diesel Fuel": "Motor Vehicle Diesel Fuel",
    "MOTOR VEHICLE GASOLINE":    "Motor Vehicle Gasoline",
    "GASOLINE":                  "Motor Vehicle Gasoline",
    "Gasoline":                  "Motor Vehicle Gasoline",
    "ALTERNATIVE FUEL":          "Alternative Fuel",
    "Alternative Fuel":          "Alternative Fuel",
    "FUEL OIL":                  "Fuel Oil",
    "Fuel Oil":                  "Fuel Oil",
    "OTHER FUEL":                "Other Fuel",
    "Other Fuel":                "Other Fuel",
    "ELECTRICITY":               "Electricity",
    "Electricity":               "Electricity",
    "NATURAL GAS":               "Natural Gas",
    "Natural Gas":               "Natural Gas",
}

@st.cache_data
def load_budget_df():
    df = pd.read_excel(_ENERGY_BUDGET_FILE, sheet_name="detail_by_account")
    df["Category"]    = df["account"].map(_ACCT_CLEAN)
    df["Fiscal_Year"] = df["year"].astype(int)
    df["Amount_M"]    = pd.to_numeric(df["total_spend"], errors="coerce").fillna(0) / 1e6
    return (df.groupby(["Category", "Fiscal_Year"], as_index=False)
              .agg(Amount_M=("Amount_M", "sum")))

@st.cache_data(show_spinner=False, ttl=3600)
def load_oil_correlation():
    return compute_oil_cost_correlation()


@st.cache_data
def load_energy_spend():
    """
    Reads chicago_energy_clean_strict.xlsx sheet summary_by_year.
    Returns (fuel_df, elec_df) each with columns: year, value_M.
    """
    df = pd.read_excel(ENERGY_SPEND_FILE, sheet_name="summary_by_year")
    df.columns = ["year", "type", "category", "total_spend"]
    df["value_M"] = df["total_spend"] / 1e6
    fuel = df[df["type"] == "fuel"].copy().sort_values("year")
    elec = df[df["type"] == "electricity"].copy().sort_values("year")
    fuel = fuel.groupby("year", as_index=False)["value_M"].sum()
    elec = elec.groupby("year", as_index=False)["value_M"].sum()
    return fuel, elec


@st.cache_data
def load_vendor_detail():
    """Reads vendor_level sheet for vendor-level spend breakdown."""
    df = pd.read_excel(ENERGY_SPEND_FILE, sheet_name="vendor_level")
    df.columns = ["vendor", "type", "category", "year", "total_spend"]
    df["value_M"] = df["total_spend"] / 1e6
    return df


_CDOT_ROADWAY_FILE = _DASHBOARD_DIR / "CDOT_Roadway_Actuals_and_Budget_v3.xlsx"

@st.cache_data
def load_cdot_roadway():
    """
    Load CDOT roadway data from CDOT_Roadway_Actuals_and_Budget_v3.xlsx.
    Returns (df_avb, df_act_wt, df_bud_wt, df_vendor):
      df_avb     — actuals vs budget by year (YEAR, ACTUALS, BUDGET, VARIANCE, VARIANCE_PCT)
      df_act_wt  — actual spend by work type (YEAR, TYPE, TOTAL_SPEND)
      df_bud_wt  — budget by work type (YEAR, TYPE, TOTAL_BUDGET)
      df_vendor  — vendor-level actual payments (VENDOR, TYPE, CATEGORY, YEAR, TOTAL_SPEND, value_M)
    """
    def _clean(df, col_names, key="YEAR"):
        df = df.iloc[1:].reset_index(drop=True)
        df.columns = col_names
        df = df.dropna(subset=[key])
        df[key] = df[key].astype(int)
        return df

    # actuals vs budget
    raw = pd.read_excel(_CDOT_ROADWAY_FILE, sheet_name="actuals_vs_budget", header=1)
    df_avb = _clean(raw, ["YEAR","CATEGORY","ACTUALS","BUDGET","VARIANCE","VARIANCE_PCT"])
    for c in ["ACTUALS","BUDGET","VARIANCE","VARIANCE_PCT"]:
        df_avb[c] = pd.to_numeric(df_avb[c], errors="coerce").fillna(0)
    df_avb = df_avb.sort_values("YEAR").reset_index(drop=True)

    # actual spend by work type
    raw2 = pd.read_excel(_CDOT_ROADWAY_FILE, sheet_name="summary_by_year", header=1)
    df_act_wt = _clean(raw2, ["YEAR","TYPE","CATEGORY","TOTAL_SPEND"])
    df_act_wt["TOTAL_SPEND"] = pd.to_numeric(df_act_wt["TOTAL_SPEND"], errors="coerce").fillna(0)

    # budget by work type
    raw3 = pd.read_excel(_CDOT_ROADWAY_FILE, sheet_name="budget_summary_by_year", header=1)
    df_bud_wt = _clean(raw3, ["YEAR","TYPE","CATEGORY","TOTAL_BUDGET"])
    df_bud_wt["TOTAL_BUDGET"] = pd.to_numeric(df_bud_wt["TOTAL_BUDGET"], errors="coerce").fillna(0)

    # vendor level
    raw4 = pd.read_excel(_CDOT_ROADWAY_FILE, sheet_name="vendor_level", header=1)
    df_vendor = _clean(raw4, ["VENDOR","TYPE","CATEGORY","YEAR","TOTAL_SPEND"], key="YEAR")
    df_vendor["TOTAL_SPEND"] = pd.to_numeric(df_vendor["TOTAL_SPEND"], errors="coerce").fillna(0)
    df_vendor["value_M"]     = df_vendor["TOTAL_SPEND"] / 1e6

    return df_avb, df_act_wt, df_bud_wt, df_vendor


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Oil Price Impact")
    st.markdown("**City of Chicago — DPS**")
    st.divider()

    focus_year = st.selectbox("Focus Year", FOCUS_YEARS, index=0)

    st.divider()
    st.markdown("**Diesel price range**")
    for scen, shock in SCENARIOS.items():
        price = BASE_DIESEL_GAL * (1 + shock)
        badge = "🟢" if shock == 0 else ("🟡" if shock <= 0.25 else ("🟠" if shock <= 0.50 else "🔴"))
        st.markdown(f"{badge} {scen}: **${price:.2f}/gal**")

    st.divider()
    st.markdown("**WTI Scenario**")
    sb_peak_wti = st.slider(
        "Peak WTI ($/bbl)", min_value=60, max_value=150,
        value=90, step=5, key="sb_peak_wti",
        help="Peak WTI price used in the forecast path",
    )
    sb_peak_yr = st.slider(
        "Peak year", min_value=2026, max_value=2030,
        value=2027, step=1, key="sb_peak_yr",
    )
    sb_hl_label = st.select_slider(
        "Reversion speed",
        options=["Rapid (1 yr)", "Medium (2 yr)", "Slow (3 yr)"],
        value="Medium (2 yr)", key="sb_hl",
        help="Speed of mean-reversion toward long-run equilibrium after peak",
    )
    sb_lr = st.slider(
        "Long-run equilibrium ($/bbl)", min_value=45, max_value=90,
        value=65, step=5, key="sb_lr",
    )
    _sb_shock = max(0.0, (sb_peak_wti - 72.0) / 72.0)
    _sb_diesel = 4.10 * (1 + _sb_shock)
    st.caption(
        f"Implied shock: **+{_sb_shock*100:.0f}%**  \n"
        f"Peak diesel: **${_sb_diesel:.2f}/gal**  \n"
        f"Long-run: **${sb_lr}/bbl → ${4.10*(1+(sb_lr-72)/72):.2f}/gal**"
    )

    st.divider()
    st.caption("Fleet: Weibull MC (125 runs)\nFacilities: Sigmoid adoption MC (125 runs)\nEnergy fraction: 28% of Facilities baseline\nPassthrough: 0.21× (elec + nat-gas blend)")


# ── Global WTI scenario (derived from sidebar controls) ──────────────────────
_SB_HL_MAP      = {"Rapid (1 yr)": 1.0, "Medium (2 yr)": 2.0, "Slow (3 yr)": 3.0}
sb_halflife     = _SB_HL_MAP[sb_hl_label]
global_shock_pct = max(0.0, (sb_peak_wti - 72.0) / 72.0)   # fraction vs base WTI

# ── KPI summary cards ─────────────────────────────────────────────────────────
def _get_combined_kpis(yr):
    sev_scen = "Severe shock (+75%)"
    mild_scen = "Mild shock (+25%)"

    fd_sev  = fleet_det[(fleet_det["Fiscal_Year"]==yr) & (fleet_det["Scenario"]==sev_scen)].iloc[0]
    fd_mild = fleet_det[(fleet_det["Fiscal_Year"]==yr) & (fleet_det["Scenario"]==mild_scen)].iloc[0]
    ad_sev  = fac_det[(fac_det["Fiscal_Year"]==yr)   & (fac_det["Scenario"]==sev_scen)].iloc[0]

    fleet_base      = fd_sev["Base_Total_M"]
    fac_base        = ad_sev["Optimized_Base_M"]
    combined_base   = fleet_base + fac_base
    combined_sev    = fleet_base + fd_sev["Delta_Total_M"] + fac_base + ad_sev["Delta_Energy_M"]
    combined_mild   = fleet_base + fd_mild["Delta_Total_M"] + fac_base

    fleet_p95 = float(fleet_mc[(fleet_mc["Fiscal_Year"]==yr) & (fleet_mc["Scenario"]==sev_scen)]["P95_M"].values[0])
    fac_p95   = float(fac_mc[(fac_mc["Fiscal_Year"]==yr)    & (fac_mc["Scenario"]==sev_scen)]["P95_M"].values[0])
    p95_total = fleet_p95 + fac_p95

    return {
        "combined_base":  combined_base,
        "mild_delta":     combined_mild - combined_base,
        "severe_delta":   combined_sev  - combined_base,
        "p95_total":      p95_total,
        "p95_upside":     p95_total - combined_sev,
    }

kpis = _get_combined_kpis(focus_year)

# ── Header: title left, pill nav top-right ────────────────────────────────────
TAB_OPTIONS = [
    "📋 Executive Summary",
    "🛢️ Oil Price vs Fleet",
    "🏢 Oil Price vs Facilities",
    "🛣️ Oil Price vs Roadways",
    "🔩 Component Exposure",
]

_hdr_l, _hdr_r = st.columns([1, 2])
with _hdr_l:
    st.markdown(f"## ⛽ Oil Price Impact — FY{focus_year}")
with _hdr_r:
    active_tab = st.pills(
        "", TAB_OPTIONS,
        default=TAB_OPTIONS[0],
        key="nav_pills",
        label_visibility="collapsed",
    )



# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 0 — Executive Summary                                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
if active_tab == "📋 Executive Summary":
    import numpy as _np

    # ── Compute "your scenario" impacts via interpolation ─────────────────────
    _cs_fleet_ex = fleet_cs[fleet_cs["Fiscal_Year"] == focus_year].sort_values("Crude_Δ_$/bbl")
    _cs_fac_ex   = fac_cs[fac_cs["Fiscal_Year"]   == focus_year].sort_values("Crude_Δ_$/bbl")
    _delta_crude_ex = float(sb_peak_wti - 72.0)

    _sev_scen  = "Severe shock (+75%)"
    _base_scen = "Base ($4.10 diesel)"
    _mild_scen = "Mild shock (+25%)"

    _fleet_sev  = fleet_det[(fleet_det["Fiscal_Year"]==focus_year) & (fleet_det["Scenario"]==_sev_scen)].iloc[0]
    _fleet_base = fleet_det[(fleet_det["Fiscal_Year"]==focus_year) & (fleet_det["Scenario"]==_base_scen)].iloc[0]
    _fleet_mild = fleet_det[(fleet_det["Fiscal_Year"]==focus_year) & (fleet_det["Scenario"]==_mild_scen)].iloc[0]
    _fac_sev    = fac_det[(fac_det["Fiscal_Year"]==focus_year)    & (fac_det["Scenario"]==_sev_scen)].iloc[0]
    _fac_base_r = fac_det[(fac_det["Fiscal_Year"]==focus_year)    & (fac_det["Scenario"]==_base_scen)].iloc[0]

    fleet_base_M   = float(_fleet_base["Base_Total_M"])
    fac_base_M     = float(_fac_base_r["Optimized_Base_M"])
    portfolio_base = fleet_base_M + fac_base_M

    if len(_cs_fleet_ex) >= 2 and _delta_crude_ex > 0:
        _f_exec = float(_np.interp(_delta_crude_ex, _cs_fleet_ex["Crude_Δ_$/bbl"].values, _cs_fleet_ex["Delta_Total_M"].values))
        _a_exec = float(_np.interp(_delta_crude_ex, _cs_fac_ex["Crude_Δ_$/bbl"].values,   _cs_fac_ex["Delta_Energy_M"].values))
    else:
        _f_exec, _a_exec = 0.0, 0.0
    _total_exec_impact = _f_exec + _a_exec
    _exec_pct = (_total_exec_impact / portfolio_base * 100) if portfolio_base > 0 else 0.0

    fleet_p95_ex = float(fleet_mc[(fleet_mc["Fiscal_Year"]==focus_year) & (fleet_mc["Scenario"]==_sev_scen)]["P95_M"].values[0])
    fac_p95_ex   = float(fac_mc[(fac_mc["Fiscal_Year"]==focus_year)    & (fac_mc["Scenario"]==_sev_scen)]["P95_M"].values[0])
    p95_ex       = fleet_p95_ex + fac_p95_ex

    fac_energy_exp    = float(_fac_sev["Delta_Energy_M"])
    fac_energy_exp_pct = (fac_energy_exp / fac_base_M * 100) if fac_base_M > 0 else 0.0
    fleet_sev_delta   = float(_fleet_sev["Delta_Total_M"])
    fleet_sev_total   = fleet_base_M + fleet_sev_delta
    fac_sev_total     = fac_base_M + fac_energy_exp
    fleet_fuel_pct    = (float(_fleet_sev["Delta_Energy_M"]) / fleet_base_M * 100) if fleet_base_M > 0 else 0.0

    # ── Situation banner ───────────────────────────────────────────────────────
    _diesel_ex    = 4.10 * (1 + global_shock_pct)
    _shock_pct_ex = global_shock_pct * 100
    _banner_col   = "#f87171" if global_shock_pct >= 0.50 else ("#fb923c" if global_shock_pct >= 0.25 else "#38bdf8")
    st.markdown(f"""
    <div style="background:linear-gradient(90deg,#0d1b2a 0%,#132336 100%);
                border-left:4px solid {_banner_col};border-radius:8px;
                padding:.75rem 1.5rem;margin-bottom:1rem;
                display:flex;align-items:center;gap:2.5rem;flex-wrap:wrap;">
      <div>
        <div style="color:#94a3b8;font-size:.68rem;font-weight:600;text-transform:uppercase;
                    letter-spacing:.09em;margin-bottom:.2rem;">WTI SCENARIO — FY{focus_year}</div>
        <div style="color:{_banner_col};font-size:1.4rem;font-weight:700;line-height:1.2;">
          ${sb_peak_wti}/bbl &nbsp;▲&nbsp; +{_shock_pct_ex:.0f}% shock
        </div>
        <div style="color:#94a3b8;font-size:.76rem;margin-top:.1rem;">
          Peak {sb_peak_yr} &nbsp;·&nbsp; {sb_hl_label} reversion &nbsp;·&nbsp; Long-run ${sb_lr}/bbl
        </div>
      </div>
      <div style="border-left:1px solid #1e3a5f;padding-left:1.75rem;">
        <div style="color:#64748b;font-size:.68rem;font-weight:600;text-transform:uppercase;letter-spacing:.09em;">PEAK DIESEL</div>
        <div style="color:#e2e8f0;font-size:1.2rem;font-weight:700;">${_diesel_ex:.2f}/gal</div>
      </div>
      <div style="border-left:1px solid #1e3a5f;padding-left:1.75rem;">
        <div style="color:#64748b;font-size:.68rem;font-weight:600;text-transform:uppercase;letter-spacing:.09em;">PORTFOLIO BASE</div>
        <div style="color:#e2e8f0;font-size:1.2rem;font-weight:700;">${portfolio_base:.0f}M</div>
        <div style="color:#475569;font-size:.72rem;">Fleet + Facilities FY{focus_year}</div>
      </div>
      <div style="border-left:1px solid #1e3a5f;padding-left:1.75rem;">
        <div style="color:#64748b;font-size:.68rem;font-weight:600;text-transform:uppercase;letter-spacing:.09em;">SCENARIO IMPACT</div>
        <div style="color:{_banner_col};font-size:1.2rem;font-weight:700;">+${_total_exec_impact:.1f}M</div>
        <div style="color:#475569;font-size:.72rem;">+{_exec_pct:.1f}% vs baseline</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Hero KPI cards ─────────────────────────────────────────────────────────
    def _hero_card(label, value, sub, accent="#0284c7", icon=""):
        return (
            f'<div style="background:#ffffff;border:1px solid #e2e8f0;border-left:4px solid {accent};'
            f'border-radius:10px;padding:1rem 1.25rem;min-height:108px;box-shadow:0 1px 4px rgba(0,0,0,0.06);">'
            f'<div style="color:#64748b;font-size:.68rem;font-weight:600;text-transform:uppercase;'
            f'letter-spacing:.09em;margin-bottom:.3rem;">{icon}&nbsp;{label}</div>'
            f'<div style="color:{accent};font-size:1.55rem;font-weight:700;line-height:1.2;">{value}</div>'
            f'<div style="color:#64748b;font-size:.74rem;margin-top:.3rem;">{sub}</div>'
            f'</div>'
        )

    _hk1, _hk2, _hk3, _hk4 = st.columns(4)
    _imp_col = "#f87171" if _exec_pct >= 10 else ("#fb923c" if _exec_pct >= 5 else "#a78bfa")
    _p95_up  = p95_ex - portfolio_base
    _p95_pct_ex = (_p95_up / portfolio_base * 100) if portfolio_base > 0 else 0.0

    with _hk1:
        st.markdown(_hero_card(
            "Portfolio Baseline",
            f"${portfolio_base:.0f}M",
            f"Fleet ${fleet_base_M:.0f}M  ·  Facilities ${fac_base_M:.0f}M",
            accent="#0284c7", icon="📦"
        ), unsafe_allow_html=True)
    with _hk2:
        st.markdown(_hero_card(
            "Your Scenario Impact",
            f"+${_total_exec_impact:.1f}M",
            f"+{_exec_pct:.1f}% vs baseline  ·  WTI ${sb_peak_wti}/bbl",
            accent=_imp_col, icon="⚡"
        ), unsafe_allow_html=True)
    with _hk3:
        st.markdown(_hero_card(
            "P95 Tail Risk (Severe)",
            f"${p95_ex:.0f}M",
            f"+${_p95_up:.0f}M (+{_p95_pct_ex:.1f}%)  ·  MC 95th pct",
            accent="#f87171", icon="⚠️"
        ), unsafe_allow_html=True)
    with _hk4:
        st.markdown(_hero_card(
            "Facilities Energy Exposure",
            f"+${fac_energy_exp:.1f}M",
            f"+{fac_energy_exp_pct:.1f}% of Facilities baseline  ·  severe shock",
            accent="#fb923c", icon="🏛️"
        ), unsafe_allow_html=True)

    st.markdown("<div style='margin-top:.75rem'></div>", unsafe_allow_html=True)

    # ── Portfolio split — Fleet | Facilities ───────────────────────────────────
    _pl, _pr = st.columns(2)

    def _portfolio_card(rows_html, footer):
        return (
            f'<div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:10px;'
            f'padding:1rem 1.25rem;box-shadow:0 1px 4px rgba(0,0,0,0.06);">'
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:.85rem;">{rows_html}</div>'
            f'<div style="margin-top:.75rem;padding-top:.75rem;border-top:1px solid #e2e8f0;'
            f'color:#64748b;font-size:.7rem;">{footer}</div>'
            f'</div>'
        )

    def _stat(label, val, sub, color="#0f172a"):
        return (
            f'<div><div style="color:#64748b;font-size:.68rem;text-transform:uppercase;'
            f'letter-spacing:.06em;">{label}</div>'
            f'<div style="color:{color};font-size:1.15rem;font-weight:700;">{val}</div>'
            f'<div style="color:#64748b;font-size:.7rem;">{sub}</div></div>'
        )

    with _pl:
        st.markdown("#### 🚗 Fleet Portfolio")
        _fleet_share = fleet_base_M / portfolio_base * 100 if portfolio_base > 0 else 0
        rows = (
            _stat("Baseline Cost",   f"${fleet_base_M:.0f}M",    f"{_fleet_share:.0f}% of portfolio",      color="#0284c7") +
            _stat("Severe Shock",    f"+${fleet_sev_delta:.1f}M", f"${fleet_sev_total:.0f}M total",         color="#dc2626") +
            _stat("Fuel Exposure",   f"{fleet_fuel_pct:.1f}%",    "of Fleet baseline (fuel cost)") +
            _stat("MC P95 (Severe)", f"${fleet_p95_ex:.0f}M",     "95th pct — Weibull 125 runs")
        )
        st.markdown(_portfolio_card(rows, "Fuel · PM materials · CM labor · CapEx replacement"), unsafe_allow_html=True)

    with _pr:
        st.markdown("#### 🏛️ Facilities Portfolio")
        _fac_share = fac_base_M / portfolio_base * 100 if portfolio_base > 0 else 0
        rows = (
            _stat("Optimized Baseline", f"${fac_base_M:.0f}M",      f"{_fac_share:.0f}% of portfolio",       color="#0284c7") +
            _stat("Severe Shock",       f"+${fac_energy_exp:.1f}M",  f"${fac_sev_total:.0f}M total",          color="#dc2626") +
            _stat("Energy Share O&M",   "28%",                        "Electricity + nat-gas + fuel oil") +
            _stat("MC P95 (Severe)",    f"${fac_p95_ex:.0f}M",       "95th pct — Sigmoid MC 125 runs")
        )
        st.markdown(_portfolio_card(rows, "0.21× oil-to-utility passthrough · ~420 buildings · Work/Asset/Demand pillars"), unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)

    # ── WTI vs Cost chart ──────────────────────────────────────────────────────
    st.markdown("#### 🛢️ WTI Crude vs DPS Energy Cost (2016 – 2033)")
    try:
        _hf_ex  = load_budget_df()
        _oc_ex  = load_oil_correlation()
        _yr_ex  = sorted(_hf_ex["Fiscal_Year"].unique())
        _fleet_cats = ["Motor Vehicle Diesel Fuel", "Motor Vehicle Gasoline", "Alternative Fuel"]
        _d_ser  = _hf_ex[_hf_ex["Category"].isin(_fleet_cats)].groupby("Fiscal_Year")["Amount_M"].sum()
        _e_ser  = _hf_ex[_hf_ex["Category"] == "Electricity"].set_index("Fiscal_Year")["Amount_M"]
        _d_dates = [pd.Timestamp(f"{y}-01-01") for y in _yr_ex]
        _d_vals  = [float(_d_ser.get(y, 0)) for y in _yr_ex]
        _e_vals  = [float(_e_ser.get(y, 0)) for y in _yr_ex]
        _base_vol = _d_vals[-1] * 1e6 / 4.10
        _scen_ex  = generate_wti_forecast(
            current_wti        = _oc_ex["current_wti"],
            peak_wti           = float(sb_peak_wti),
            peak_year          = sb_peak_yr,
            halflife_yrs       = sb_halflife,
            long_run_eq        = float(sb_lr),
            sigma              = 0.22,
            base_diesel_vol_gal= _base_vol,
            base_elec_M        = _e_vals[-1],
        )
        _wti_pairs = [(d, p) for d, p in zip(_oc_ex["wti_dates"], _oc_ex["wti_prices"])
                      if pd.Timestamp(d).year >= 2016]
        _chart_data = {
            **_oc_ex,
            "wti_dates":    [p[0] for p in _wti_pairs],
            "wti_prices":   [p[1] for p in _wti_pairs],
            "diesel_dates": _d_dates,
            "diesel_budget": _d_vals,
            "elec_dates":   _d_dates,
            "elec_budget":  _e_vals,
            "diesel_est_2026_M":    None,
            "diesel_budget_2026_M": None,
            **_scen_ex,
        }
        st.plotly_chart(
            make_oil_cost_chart(
                _chart_data,
                diesel_label      = "Fleet Fuel Budget ($M)",
                elec_label        = "Electricity Budget ($M)",
                diesel_proj_label = "Fleet Fuel Projected",
                elec_proj_label   = "Electricity Budget Projected",
                title             = f"WTI Crude vs DPS Energy Budget  (2016 – 2033)  |  Scenario: ${sb_peak_wti}/bbl peak {sb_peak_yr}",
                x_range           = ["2016-01-01", "2033-12-31"],
            ),
            use_container_width=True,
        )
    except Exception as _e:
        st.info(f"Chart unavailable — {_e}")

    st.markdown("<div style='margin-top:.5rem'></div>", unsafe_allow_html=True)

    # ── Key Takeaways ──────────────────────────────────────────────────────────
    st.markdown("#### 📌 Key Takeaways")

    _takeaways = [
        (f"A **+{_shock_pct_ex:.0f}% WTI shock** (${sb_peak_wti}/bbl peak, {sb_peak_yr}) adds an estimated "
         f"**+${_total_exec_impact:.1f}M** to DPS portfolio costs in FY{focus_year} — "
         f"a **{_exec_pct:.1f}% increase** vs the optimized baseline."),
        (f"**Fleet** is the primary direct-fuel consumer. A severe (+75%) shock adds "
         f"**+${fleet_sev_delta:.1f}M** across fuel burn, PM materials, CM labor, and CapEx replacement."),
        (f"**Facilities** face indirect exposure through energy utilities — 28% of O&M is "
         f"energy-linked with a 0.21× oil-to-utility passthrough. A severe shock adds "
         f"**+${fac_energy_exp:.1f}M** (+{fac_energy_exp_pct:.1f}%) to Facilities costs."),
        (f"**Tail risk**: Monte Carlo P95 puts the combined portfolio at **${p95_ex:.0f}M** "
         f"under a severe shock — **+${_p95_up:.0f}M** (+{_p95_pct_ex:.1f}%) above the deterministic median."),
        ("**Mitigation**: Work Modernization, Demand Management, and Asset Management pillars "
         "reduce Facilities baseline through sigmoid adoption curves. Fleet risk is best managed "
         "through procurement timing, fuel-efficient asset acquisition, and fuel hedging strategies."),
    ]

    for i, txt in enumerate(_takeaways):
        _bg = "#0d1b2a" if i % 2 == 0 else "#091424"
        st.markdown(
            f'<div style="background:{_bg};border-left:3px solid #1e3a5f;border-radius:0 6px 6px 0;'
            f'padding:.6rem 1rem;margin-bottom:.4rem;color:#cbd5e1;font-size:.87rem;line-height:1.6;">'
            f'{txt}</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div style="margin-top:1rem;color:#334155;font-size:.7rem;text-align:right;">'
        f'FY{focus_year} &nbsp;·&nbsp; WTI scenario ${sb_peak_wti}/bbl peak ({sb_peak_yr}) &nbsp;·&nbsp; '
        f'City of Chicago DPS Assets Analysis &nbsp;·&nbsp; Adjust scenario via sidebar &nbsp;·&nbsp; '
        f'Explore detail in remaining tabs'
        f'</div>',
        unsafe_allow_html=True,
    )


if False and active_tab == "⚡ Fuel Price Sensitivity":  # removed tab
    # ── Your WTI scenario callout ──────────────────────────────────────────────
    _cs_fleet = fleet_cs[fleet_cs["Fiscal_Year"] == focus_year].sort_values("Crude_Δ_$/bbl")
    _cs_fac   = fac_cs[fac_cs["Fiscal_Year"]   == focus_year].sort_values("Crude_Δ_$/bbl")
    _delta_crude = float(sb_peak_wti - 72.0)
    if len(_cs_fleet) >= 2 and _delta_crude > 0:
        import numpy as _np
        _f_impact = float(_np.interp(
            _delta_crude,
            _cs_fleet["Crude_Δ_$/bbl"].values,
            _cs_fleet["Delta_Total_M"].values,
        ))
        _a_impact = float(_np.interp(
            _delta_crude,
            _cs_fac["Crude_Δ_$/bbl"].values,
            _cs_fac["Delta_Energy_M"].values,
        ))
        _sc1, _sc2, _sc3, _sc4 = st.columns(4)
        with _sc1:
            st.metric("Your WTI Scenario", f"${sb_peak_wti}/bbl",
                      delta=f"Peak {sb_peak_yr} · {sb_hl_label}", delta_color="off")
        with _sc2:
            st.metric("Implied shock vs base", f"+{global_shock_pct*100:.0f}%",
                      delta=f"${4.10*(1+global_shock_pct):.2f}/gal diesel", delta_color="off")
        with _sc3:
            st.metric(f"Est. Fleet impact FY{focus_year}", f"+${_f_impact:.1f}M",
                      delta=f"interpolated at +${_delta_crude:.0f}/bbl crude", delta_color="inverse")
        with _sc4:
            st.metric(f"Est. Facilities impact FY{focus_year}", f"+${_a_impact:.1f}M",
                      delta="energy delta only", delta_color="inverse")
        st.divider()

    st.markdown(f"### Budget Impact by Scenario — FY{focus_year}")

    col_l, col_r = st.columns([1.4, 1])

    with col_l:
        st.markdown('<p class="tab-header">Tornado: Incremental Cost vs Base</p>', unsafe_allow_html=True)
        st.plotly_chart(
            make_tornado_chart(fleet_det, fac_det, focus_year),
            use_container_width=True,
        )

    with col_r:
        st.markdown('<p class="tab-header">Crude Price → Budget Impact</p>', unsafe_allow_html=True)
        st.plotly_chart(
            make_crude_sensitivity_chart(fleet_cs, fac_cs, focus_year),
            use_container_width=True,
        )

    st.divider()

    # Deterministic scenario tables side by side
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Fleet — Deterministic Scenarios**")
        f_show = fleet_det[fleet_det["Fiscal_Year"] == focus_year][[
            "Scenario","Diesel_$/gal","Base_Total_M","Shock_Total_M","Delta_Total_M","Delta_Pct"
        ]].copy()
        f_show.columns = ["Scenario","Diesel $/gal","Base ($M)","Shocked ($M)","Δ ($M)","Δ (%)"]
        st.dataframe(f_show, hide_index=True, use_container_width=True)

    with col_b:
        st.markdown("**Facilities — Deterministic Scenarios**")
        a_show = fac_det[fac_det["Fiscal_Year"] == focus_year][[
            "Scenario","Base_Total_M","Optimized_Base_M","Optimized_Shock_M","Delta_Energy_M","Delta_Pct"
        ]].copy()
        a_show.columns = ["Scenario","Baseline ($M)","Optimized Base","Shock ($M)","Energy Δ ($M)","Δ (%)"]
        st.dataframe(a_show, hide_index=True, use_container_width=True)

    st.divider()

    # Historical fuel chart
    st.markdown("### Historical Fuel & Utilities Budget Validation")
    st.plotly_chart(
        make_historic_fuel_chart(fleet_hist),
        use_container_width=True,
    )

    # Crude sensitivity tables side by side
    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown(f"**Fleet — Cost per Crude Increase (FY{focus_year})**")
        fc_show = fleet_cs[fleet_cs["Fiscal_Year"] == focus_year][[
            "Crude_Δ_$/bbl","Implied_Diesel","Delta_Energy_M","Delta_2nd_Order_M","Delta_Total_M"
        ]].copy()
        fc_show.columns = ["Crude Δ ($/bbl)","Diesel ($/gal)","Energy Δ ($M)","2nd-Order Δ ($M)","Total Δ ($M)"]
        st.dataframe(fc_show, hide_index=True, use_container_width=True)

    with col_d:
        st.markdown(f"**Facilities — Energy Cost per Crude Increase (FY{focus_year})**")
        ac_show = fac_cs[fac_cs["Fiscal_Year"] == focus_year].copy()
        ac_show.columns = ["Year","Crude Δ ($/bbl)","Energy Δ ($M)"]
        st.dataframe(ac_show[["Crude Δ ($/bbl)","Energy Δ ($M)"]], hide_index=True, use_container_width=True)

    with st.expander("Methodology notes", expanded=True):
        st.markdown("""
**Fleet energy elasticity:** Direct linear scaling (gallons × price/gal).
PM/materials: 0.18× per unit oil fraction. Corrective/CM: 0.12×. Replacement CapEx: 0.10×.
Source: GFOA Fleet Benchmarks + DOT Commodity Index Studies.

**Facilities energy fraction:** 28% of annual O&M baseline allocated to energy
(electricity 80%, natural gas 20%). Oil-to-electricity passthrough: 0.15.
Oil-to-natural gas passthrough: 0.45. Weighted: **0.21×**.
Source: DOE CBECS + GFOA municipal building benchmarks.

**Crude-to-diesel mapping:** $0.24/gal retail diesel per $10/bbl WTI (EIA historical ratio).
Chicago diesel pass-through ratio to budget: **1.41×** (12-month lag, r=0.762, 2021–2026).
        """)


if False and active_tab == "📊 Budget Scenarios":  # removed tab
    sel_year = focus_year

    st.markdown(f"### Monte Carlo Risk Bands — FY{sel_year}")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            make_fleet_component_chart(fleet_comp, sel_year),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            make_fac_component_chart(fac_det, sel_year),
            use_container_width=True,
        )

    st.divider()

    col_left, col_right = st.columns(2)
    with col_left:
        st.plotly_chart(
            make_mc_scenario_range(fleet_mc, fac_mc, sel_year),
            use_container_width=True,
        )
    with col_right:
        st.plotly_chart(
            make_oil_delta_bars(fleet_det, fac_det, sel_year),
            use_container_width=True,
        )

    st.divider()

    # ── Your WTI scenario vs pre-computed brackets ─────────────────────────────
    _det_yr = fleet_det[fleet_det["Fiscal_Year"] == sel_year].copy()
    _fdet_yr = fac_det[fac_det["Fiscal_Year"]   == sel_year].copy()
    if len(_det_yr) >= 2:
        import numpy as _np2
        _f_base_M  = float(_det_yr[_det_yr["Scenario"]=="Base ($4.10 diesel)"]["Base_Total_M"].values[0])
        _fa_base_M = float(_fdet_yr[_fdet_yr["Scenario"]=="Base ($4.10 diesel)"]["Optimized_Base_M"].values[0])
        _f_shocks  = _det_yr.sort_values("Shock_Total_M")
        _fa_shocks = _fdet_yr.sort_values("Optimized_Shock_M")
        # Interpolate fleet and facilities cost at global_shock_pct
        _f_shk_fracs = (_f_shocks["Shock_Total_M"] - _f_base_M) / _f_base_M
        _fa_shk_fracs = (_fa_shocks["Optimized_Shock_M"] - _fa_base_M) / _fa_base_M
        _f_scen_M  = float(_np2.interp(global_shock_pct, _f_shk_fracs.values,  _f_shocks["Shock_Total_M"].values))
        _fa_scen_M = float(_np2.interp(global_shock_pct, _fa_shk_fracs.values, _fa_shocks["Optimized_Shock_M"].values))
        _comb_scen = _f_scen_M + _fa_scen_M
        _comb_base = _f_base_M + _fa_base_M

        st.markdown(
            f"**Your WTI scenario** — Peak **${sb_peak_wti}/bbl** in **{sb_peak_yr}** "
            f"(+{global_shock_pct*100:.0f}% shock · {sb_hl_label} reversion · long-run ${sb_lr}/bbl)"
        )
        _yc1, _yc2, _yc3, _yc4 = st.columns(4)
        with _yc1:
            st.metric("Fleet (interp.)", f"${_f_scen_M:.1f}M",
                      delta=f"+${_f_scen_M-_f_base_M:.1f}M vs base", delta_color="inverse")
        with _yc2:
            st.metric("Facilities (interp.)", f"${_fa_scen_M:.1f}M",
                      delta=f"+${_fa_scen_M-_fa_base_M:.1f}M vs base", delta_color="inverse")
        with _yc3:
            st.metric("Combined (interp.)", f"${_comb_scen:.1f}M",
                      delta=f"+${_comb_scen-_comb_base:.1f}M vs base", delta_color="inverse")
        with _yc4:
            # Find which bracket this falls into
            _brackets = {"Base": 0.0, "Mild +25%": 0.25, "Moderate +50%": 0.50, "Severe +75%": 0.75}
            _closest = min(_brackets, key=lambda k: abs(_brackets[k] - global_shock_pct))
            st.metric("Closest scenario bracket", _closest,
                      delta=f"Your shock: {global_shock_pct*100:.0f}%", delta_color="off")
        st.divider()

    # Combined MC summary table
    st.markdown("**Combined Portfolio — MC Risk Bands Summary**")
    fm2 = fleet_mc[fleet_mc["Fiscal_Year"] == sel_year]
    fa2 = fac_mc[fac_mc["Fiscal_Year"]     == sel_year]
    comb = fm2[["Scenario","P10_M","P50_M","P95_M"]].merge(
        fa2[["Scenario","P10_M","P50_M","P95_M"]], on="Scenario", suffixes=("_Fleet","_Fac")
    ).copy()
    comb["Combined P10"] = comb["P10_M_Fleet"] + comb["P10_M_Fac"]
    comb["Combined P50"] = comb["P50_M_Fleet"] + comb["P50_M_Fac"]
    comb["Combined P95"] = comb["P95_M_Fleet"] + comb["P95_M_Fac"]
    comb["Budget at Risk"] = comb["Combined P95"] - comb["Combined P50"]

    base_p50 = float(comb[comb["Scenario"]=="Base ($4.10 diesel)"]["Combined P50"].values[0])
    comb["Δ vs Base P50"] = (comb["Combined P50"] - base_p50).round(1)

    show = comb[["Scenario","Combined P10","Combined P50","Combined P95","Budget at Risk","Δ vs Base P50"]].copy()
    show.columns = ["Scenario","P10 ($M)","P50 ($M)","P95 ($M)","Budget at Risk ($M)","Δ vs Base ($M)"]
    st.dataframe(show, hide_index=True, use_container_width=True)

    with st.expander("Monte Carlo model assumptions", expanded=True):
        st.markdown(f"""
**Fleet MC (125 runs):** Pre-computed from Weibull failure model.
Asset failure times drawn from Weibull(k=3.0, λ=useful_life×1.15).
Oil shock applied post-simulation on energy + PM/CM/CapEx components.

**Facilities MC (125 runs):** Pillar adoption rates drawn from
Normal(r_default, σ={100*0.20:.0f}% × r_default), clipped to [r_min, r_max].
Multiplicative residual model: residual = ∏[1 − min(r_p × σ(t), 0.95)].
Oil shock = energy fraction ({28}%) × passthrough ({0.21}) × shock factor,
added to optimized Facilities cost per run.

**Risk interpretation:**
- P10 = favorable (all programs outperform, low oil)
- P50 = median / central estimate
- P95 = adverse tail (program underperformance + high oil)
        """)


# Budget Trajectory tab removed in v2 — content consolidated into Cost Trajectory
if False:
    rc = get_runchart()
    wti    = rc.get("current_wti", 107.82)
    shock  = rc.get("current_shock", 0.21)
    bridge = rc.get("bridge", {})

    # ── KPI banner
    st.markdown("### Budget Trajectory — 2016 to 2033")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("WTI (Live)", f"${wti:.2f}/bbl",
                  delta=f"+{shock*100:.0f}% vs budget base",
                  delta_color="inverse",
                  help="WTI crude price as of Mar 9, 2026 (Strait of Hormuz crisis)")
    with k2:
        _fy_idx   = rc["proj_years"].index(focus_year) if focus_year in rc["proj_years"] else 0
        yr_base   = rc["baseline"][_fy_idx]
        yr_hr     = rc["high_risk"][_fy_idx]
        st.metric(f"FY{focus_year} Baseline", f"${yr_base:.0f}M")
    with k3:
        _hr_delta = yr_hr - yr_base
        _hr_pct   = _hr_delta / yr_base * 100
        st.metric(f"FY{focus_year} High Risk",
                  f"${yr_hr:.0f}M  (+{_hr_pct:.1f}%)",
                  delta=f"+${_hr_delta:.1f}M oil premium",
                  delta_color="inverse")
    with k4:
        yr_opt     = rc["optimized"][_fy_idx]
        _opt_delta = yr_base - yr_opt
        _opt_pct   = _opt_delta / yr_base * 100
        st.metric(f"FY{focus_year} Optimized",
                  f"${yr_opt:.0f}M  (-{_opt_pct:.1f}%)",
                  delta=f"-${_opt_delta:.1f}M vs baseline",
                  delta_color="normal")

    st.divider()

    # ── Runchart
    st.plotly_chart(make_budget_runchart(rc), use_container_width=True)

    st.divider()

    # ── Gap chart
    st.markdown("### Annual Oil Risk Premium vs Optimization Savings")
    st.plotly_chart(make_savings_gap_chart(rc), use_container_width=True)

    st.divider()

    # ── Scenario table
    st.markdown("**Projected Budget by Scenario ($M)**")
    proj_df = pd.DataFrame({
        "Fiscal Year": rc["proj_years"],
        "Baseline ($M)":  [round(v, 1) for v in rc["baseline"]],
        "High Risk ($M)": [round(v, 1) for v in rc["high_risk"]],
        "Optimized ($M)": [round(v, 1) for v in rc["optimized"]],
        "Oil Premium ($M)":   [round(h - b, 1) for h, b in zip(rc["high_risk"], rc["baseline"])],
        "Opt Savings ($M)":   [round(b - o, 1) for b, o in zip(rc["baseline"], rc["optimized"])],
        "Net Gap ($M)":        [round(h - o, 1) for h, o in zip(rc["high_risk"], rc["optimized"])],
    })
    st.dataframe(proj_df, hide_index=True, use_container_width=True)

    with st.expander("Scenario definitions", expanded=True):
        st.markdown(f"""
**Baseline:** Budget plan trajectory from Facilities `fact_savings` + Fleet pre-computed simulation.
No additional oil shock assumed beyond current budget base ($4.10/gal diesel).

**High Risk:** Current WTI shock (+{shock*100:.0f}%) sustained through projection period.
Fleet uplift = {shock*100:.0f}% × (energy 33% + PM 0.18×15% + CM 0.12×10% + CapEx 0.10×5%) of fleet budget.
Facilities uplift = {shock*100:.0f}% × 28% energy fraction × 0.21 passthrough.
Confidence band = ±1σ using σ_diesel calibrated from FY2016–2025 budget data (chicago_energy_budget_2016_2025.xlsx).

**Optimized:** Baseline minus full L1+L2 pillar savings (Work Modernization, Demand Management,
Asset Management, Vendor & Payment Management, Early Pay). Savings compound annually via
sigmoid adoption curve (AI-accelerated from FY2024).

**Data sources:** Historical actuals from `Historical_Costs` sheet in `asset_model_outputs.xlsx`.
Projections from `fact_savings` sheet. Fleet from `sim_annual.csv`.
        """)




# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  TAB 4 — Component Exposure (oil shock propagation)                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
if active_tab == "🔩 Component Exposure":

    # ── Oil passthrough factors per component ─────────────────────────────────
    # Fleet — Vehicles & Equipment
    _FLEET_VE_PT = {
        "Vehicle Acquisition/Lease":      0.08,
        "Vehicle Upfitting/Configuration": 0.08,
        "Scheduled Maintenance":          0.18,
        "Unscheduled Repairs":            0.12,
        "Tires & Consumables":            0.22,
        "Fuel/Energy":                    1.00,
        "Depreciation/Residual Value":    0.05,
        "Remarketing/Disposal":           0.03,
    }
    # Fleet — Fleet Operations
    _FLEET_OPS_PT = {
        "Fleet Management Platform (FMS)":  0.02,
        "Telematics/GPS Tracking":          0.02,
        "Driver Management & Training":     0.03,
        "Insurance & Risk Management":      0.04,
        "Registration & Title Management":  0.01,
        "Compliance & Regulatory":          0.02,
        "Toll Management":                  0.01,
        "Accident Management":              0.03,
        "Roadside Assistance":              0.05,
        "Motor Pool / Shared Use Coordination": 0.10,
        "Fuel Card Administration":         0.05,
        "Preventive Maintenance Scheduling": 0.15,
        "Parts Inventory Management":       0.10,
        "Vendor/Supplier Management":       0.08,
        "Fleet Rightsizing Analytics":      0.01,
        "Lifecycle Replacement Planning":   0.05,
        "Environmental/Emissions Compliance": 0.03,
        "EV Charging Infrastructure":       0.21,
        "Route Optimization":               0.02,
        "Utilization Tracking & Reporting": 0.01,
        "Capital Planning & Budgeting":     0.02,
        "Warranty Management":              0.05,
        "Driver Safety & MVR Monitoring":   0.01,
        "Mobile Workforce Coordination":    0.02,
        "Fleet Procurement/Sourcing":       0.06,
        "Parking Management":               0.01,
        "Vehicle Storage & Yard Management": 0.05,
        "Fleet Data & Analytics Platform":  0.02,
    }
    # Facilities — Building Systems & Assets
    _FAC_BSA_PT = {
        "HVAC Systems":                     0.28,
        "Electrical Systems & Distribution": 0.21,
        "Plumbing & Water Systems":          0.12,
        "Roof & Building Envelope":          0.05,
        "Fire Protection & Life Safety":     0.04,
        "Elevators & Vertical Transport":    0.21,
        "Structural Systems & Foundations":  0.04,
        "Interior Finishes & Fixtures":      0.05,
        "Parking Structures & Paving":       0.08,
        "Building Automation Systems (BAS)": 0.08,
    }
    # Facilities — Facilities Operations
    _FAC_OPS_PT = {
        "Janitorial & Custodial Services":        0.08,
        "Grounds & Landscaping Maintenance":      0.15,
        "Preventive Maintenance Programs":        0.12,
        "Corrective/Reactive Maintenance":        0.12,
        "Energy Management & Utilities":          0.60,
        "Waste Management & Recycling":           0.15,
        "Pest Control":                           0.08,
        "Security Systems & Access Control":      0.05,
        "Guard Services & Monitoring":            0.05,
        "Space Planning & Move Management":       0.03,
        "CAFM/IWMS Platform":                     0.02,
        "Work Order Management":                  0.02,
        "Capital Project Management":             0.06,
        "Lease Administration":                   0.01,
        "Real Estate Tax & Insurance":            0.02,
        "Environmental Health & Safety (EHS)":    0.04,
        "Indoor Air Quality Management":          0.05,
        "ADA/Accessibility Compliance":           0.02,
        "Hazardous Materials Management":         0.06,
        "Snow & Ice Removal":                     0.35,
        "Signage & Wayfinding":                   0.04,
        "Mail & Package Services":                0.10,
        "Loading Dock Operations":                0.15,
        "Emergency Preparedness & Response":      0.08,
        "Sustainability & LEED Programs":         0.03,
        "Commissioning & Retro-Commissioning":    0.04,
        "Condition Assessment & FCI":             0.04,
        "Deferred Maintenance Planning":          0.03,
        "Vendor/Contractor Management":           0.06,
        "Facility Data & Analytics Platform":     0.02,
    }

    fleet_cfg = pd.DataFrame([
        {"Component": "Vehicle Acquisition/Lease",       "Domain": "Vehicles & Equipment", "Pct_of_Fleet_Cost": 0.1800},
        {"Component": "Vehicle Upfitting/Configuration", "Domain": "Vehicles & Equipment", "Pct_of_Fleet_Cost": 0.0400},
        {"Component": "Scheduled Maintenance",           "Domain": "Vehicles & Equipment", "Pct_of_Fleet_Cost": 0.0900},
        {"Component": "Unscheduled Repairs",             "Domain": "Vehicles & Equipment", "Pct_of_Fleet_Cost": 0.0700},
        {"Component": "Tires & Consumables",             "Domain": "Vehicles & Equipment", "Pct_of_Fleet_Cost": 0.0350},
        {"Component": "Fuel/Energy",                     "Domain": "Vehicles & Equipment", "Pct_of_Fleet_Cost": 0.1200},
        {"Component": "Depreciation/Residual Value",     "Domain": "Vehicles & Equipment", "Pct_of_Fleet_Cost": 0.0450},
        {"Component": "Remarketing/Disposal",            "Domain": "Vehicles & Equipment", "Pct_of_Fleet_Cost": 0.0200},
        {"Component": "Fleet Management Platform (FMS)", "Domain": "Fleet Operations",     "Pct_of_Fleet_Cost": 0.027446},
        {"Component": "Telematics/GPS Tracking",         "Domain": "Fleet Operations",     "Pct_of_Fleet_Cost": 0.022363},
        {"Component": "Driver Management & Training",    "Domain": "Fleet Operations",     "Pct_of_Fleet_Cost": 0.020330},
        {"Component": "Insurance & Risk Management",     "Domain": "Fleet Operations",     "Pct_of_Fleet_Cost": 0.038628},
        {"Component": "Registration & Title Management", "Domain": "Fleet Operations",     "Pct_of_Fleet_Cost": 0.009149},
        {"Component": "Compliance & Regulatory",         "Domain": "Fleet Operations",     "Pct_of_Fleet_Cost": 0.013215},
        {"Component": "Toll Management",                 "Domain": "Fleet Operations",     "Pct_of_Fleet_Cost": 0.006607},
        {"Component": "Accident Management",             "Domain": "Fleet Operations",     "Pct_of_Fleet_Cost": 0.016264},
        {"Component": "Roadside Assistance",             "Domain": "Fleet Operations",     "Pct_of_Fleet_Cost": 0.011182},
        {"Component": "Motor Pool / Shared Use Coordination", "Domain": "Fleet Operations","Pct_of_Fleet_Cost": 0.013215},
        {"Component": "Fuel Card Administration",        "Domain": "Fleet Operations",     "Pct_of_Fleet_Cost": 0.008640},
        {"Component": "Preventive Maintenance Scheduling","Domain": "Fleet Operations",    "Pct_of_Fleet_Cost": 0.022363},
        {"Component": "Parts Inventory Management",      "Domain": "Fleet Operations",     "Pct_of_Fleet_Cost": 0.016264},
        {"Component": "Vendor/Supplier Management",      "Domain": "Fleet Operations",     "Pct_of_Fleet_Cost": 0.013215},
        {"Component": "Fleet Rightsizing Analytics",     "Domain": "Fleet Operations",     "Pct_of_Fleet_Cost": 0.011182},
        {"Component": "Lifecycle Replacement Planning",  "Domain": "Fleet Operations",     "Pct_of_Fleet_Cost": 0.016264},
        {"Component": "Environmental/Emissions Compliance","Domain": "Fleet Operations",   "Pct_of_Fleet_Cost": 0.008640},
        {"Component": "EV Charging Infrastructure",      "Domain": "Fleet Operations",     "Pct_of_Fleet_Cost": 0.013215},
        {"Component": "Route Optimization",              "Domain": "Fleet Operations",     "Pct_of_Fleet_Cost": 0.016264},
        {"Component": "Utilization Tracking & Reporting","Domain": "Fleet Operations",     "Pct_of_Fleet_Cost": 0.011182},
        {"Component": "Capital Planning & Budgeting",    "Domain": "Fleet Operations",     "Pct_of_Fleet_Cost": 0.013215},
        {"Component": "Warranty Management",             "Domain": "Fleet Operations",     "Pct_of_Fleet_Cost": 0.008640},
        {"Component": "Driver Safety & MVR Monitoring",  "Domain": "Fleet Operations",     "Pct_of_Fleet_Cost": 0.011182},
        {"Component": "Mobile Workforce Coordination",   "Domain": "Fleet Operations",     "Pct_of_Fleet_Cost": 0.011182},
        {"Component": "Fleet Procurement/Sourcing",      "Domain": "Fleet Operations",     "Pct_of_Fleet_Cost": 0.013215},
        {"Component": "Parking Management",              "Domain": "Fleet Operations",     "Pct_of_Fleet_Cost": 0.006607},
        {"Component": "Vehicle Storage & Yard Management","Domain": "Fleet Operations",    "Pct_of_Fleet_Cost": 0.008640},
        {"Component": "Fleet Data & Analytics Platform", "Domain": "Fleet Operations",     "Pct_of_Fleet_Cost": 0.011690},
    ])

    fac_cfg = pd.DataFrame([
        {"Component": "HVAC Systems",                        "Domain": "Building Systems & Assets", "Pct_of_Facility_Cost": 0.1200},
        {"Component": "Electrical Systems & Distribution",   "Domain": "Building Systems & Assets", "Pct_of_Facility_Cost": 0.0800},
        {"Component": "Plumbing & Water Systems",            "Domain": "Building Systems & Assets", "Pct_of_Facility_Cost": 0.0550},
        {"Component": "Roof & Building Envelope",            "Domain": "Building Systems & Assets", "Pct_of_Facility_Cost": 0.0750},
        {"Component": "Fire Protection & Life Safety",       "Domain": "Building Systems & Assets", "Pct_of_Facility_Cost": 0.0500},
        {"Component": "Elevators & Vertical Transport",      "Domain": "Building Systems & Assets", "Pct_of_Facility_Cost": 0.0350},
        {"Component": "Structural Systems & Foundations",    "Domain": "Building Systems & Assets", "Pct_of_Facility_Cost": 0.0450},
        {"Component": "Interior Finishes & Fixtures",        "Domain": "Building Systems & Assets", "Pct_of_Facility_Cost": 0.0350},
        {"Component": "Parking Structures & Paving",         "Domain": "Building Systems & Assets", "Pct_of_Facility_Cost": 0.0250},
        {"Component": "Building Automation Systems (BAS)",   "Domain": "Building Systems & Assets", "Pct_of_Facility_Cost": 0.0300},
        {"Component": "Janitorial & Custodial Services",     "Domain": "Facilities Operations",     "Pct_of_Facility_Cost": 0.0525},
        {"Component": "Grounds & Landscaping Maintenance",   "Domain": "Facilities Operations",     "Pct_of_Facility_Cost": 0.0178},
        {"Component": "Preventive Maintenance Programs",     "Domain": "Facilities Operations",     "Pct_of_Facility_Cost": 0.0377},
        {"Component": "Corrective/Reactive Maintenance",     "Domain": "Facilities Operations",     "Pct_of_Facility_Cost": 0.0476},
        {"Component": "Energy Management & Utilities",       "Domain": "Facilities Operations",     "Pct_of_Facility_Cost": 0.0675},
        {"Component": "Waste Management & Recycling",        "Domain": "Facilities Operations",     "Pct_of_Facility_Cost": 0.0119},
        {"Component": "Pest Control",                        "Domain": "Facilities Operations",     "Pct_of_Facility_Cost": 0.0040},
        {"Component": "Security Systems & Access Control",   "Domain": "Facilities Operations",     "Pct_of_Facility_Cost": 0.0218},
        {"Component": "Guard Services & Monitoring",         "Domain": "Facilities Operations",     "Pct_of_Facility_Cost": 0.0307},
        {"Component": "Space Planning & Move Management",    "Domain": "Facilities Operations",     "Pct_of_Facility_Cost": 0.0079},
        {"Component": "CAFM/IWMS Platform",                  "Domain": "Facilities Operations",     "Pct_of_Facility_Cost": 0.0069},
        {"Component": "Work Order Management",               "Domain": "Facilities Operations",     "Pct_of_Facility_Cost": 0.0099},
        {"Component": "Capital Project Management",          "Domain": "Facilities Operations",     "Pct_of_Facility_Cost": 0.0198},
        {"Component": "Lease Administration",                "Domain": "Facilities Operations",     "Pct_of_Facility_Cost": 0.0069},
        {"Component": "Real Estate Tax & Insurance",         "Domain": "Facilities Operations",     "Pct_of_Facility_Cost": 0.0129},
        {"Component": "Environmental Health & Safety (EHS)", "Domain": "Facilities Operations",     "Pct_of_Facility_Cost": 0.0099},
        {"Component": "Indoor Air Quality Management",       "Domain": "Facilities Operations",     "Pct_of_Facility_Cost": 0.0069},
        {"Component": "ADA/Accessibility Compliance",        "Domain": "Facilities Operations",     "Pct_of_Facility_Cost": 0.0059},
        {"Component": "Hazardous Materials Management",      "Domain": "Facilities Operations",     "Pct_of_Facility_Cost": 0.0050},
        {"Component": "Snow & Ice Removal",                  "Domain": "Facilities Operations",     "Pct_of_Facility_Cost": 0.0119},
        {"Component": "Signage & Wayfinding",                "Domain": "Facilities Operations",     "Pct_of_Facility_Cost": 0.0030},
        {"Component": "Mail & Package Services",             "Domain": "Facilities Operations",     "Pct_of_Facility_Cost": 0.0030},
        {"Component": "Loading Dock Operations",             "Domain": "Facilities Operations",     "Pct_of_Facility_Cost": 0.0030},
        {"Component": "Emergency Preparedness & Response",   "Domain": "Facilities Operations",     "Pct_of_Facility_Cost": 0.0079},
        {"Component": "Sustainability & LEED Programs",      "Domain": "Facilities Operations",     "Pct_of_Facility_Cost": 0.0040},
        {"Component": "Commissioning & Retro-Commissioning", "Domain": "Facilities Operations",     "Pct_of_Facility_Cost": 0.0050},
        {"Component": "Condition Assessment & FCI",          "Domain": "Facilities Operations",     "Pct_of_Facility_Cost": 0.0069},
        {"Component": "Deferred Maintenance Planning",       "Domain": "Facilities Operations",     "Pct_of_Facility_Cost": 0.0079},
        {"Component": "Vendor/Contractor Management",        "Domain": "Facilities Operations",     "Pct_of_Facility_Cost": 0.0099},
        {"Component": "Facility Data & Analytics Platform",  "Domain": "Facilities Operations",     "Pct_of_Facility_Cost": 0.0040},
    ])

    # ── Controls ──────────────────────────────────────────────────────────────
    comp_yr    = focus_year
    shock      = global_shock_pct
    shock_label = f"+{shock*100:.0f}% (WTI ${sb_peak_wti}/bbl)"

    st.markdown("### Component-Level Oil Shock Exposure")
    _implied_diesel = 4.10 * (1 + shock)
    st.caption(
        f"WTI scenario: peak **${sb_peak_wti}/bbl** in **{sb_peak_yr}** · "
        f"implied shock **+{shock*100:.0f}%** · diesel **${_implied_diesel:.2f}/gal** · "
        f"reversion **{sb_hl_label}** · long-run **${sb_lr}/bbl** · "
        f"adjust controls in the sidebar ←"
    )

    # Base budgets for the selected year
    _fleet_base_row = fleet_det[
        (fleet_det["Fiscal_Year"] == comp_yr) &
        (fleet_det["Scenario"] == "Base ($4.10 diesel)")
    ]
    _fac_base_row = fac_det[
        (fac_det["Fiscal_Year"] == comp_yr) &
        (fac_det["Scenario"] == "Base ($4.10 diesel)")
    ]
    fleet_base_M = float(_fleet_base_row["Base_Total_M"].values[0]) if len(_fleet_base_row) else 30.0
    fac_base_M   = float(_fac_base_row["Optimized_Base_M"].values[0]) if len(_fac_base_row) else 80.0

    def _build_component_series(cfg_df, pct_col, passthrough_dict, base_M, shock_pct):
        """Return (components, base_vals, delta_vals) for a domain subset."""
        comps, bases, deltas = [], [], []
        for _, row in cfg_df.iterrows():
            comp = str(row["Component"])
            pct  = float(row[pct_col])
            pt   = passthrough_dict.get(comp, 0.0)
            base_val  = round(base_M * pct, 4)
            delta_val = round(base_val * pt * shock_pct, 4)
            comps.append(comp)
            bases.append(base_val)
            deltas.append(delta_val)
        return comps, bases, deltas

    # Fleet — split by domain
    fleet_ve   = fleet_cfg[fleet_cfg["Domain"] == "Vehicles & Equipment"]
    fleet_ops  = fleet_cfg[fleet_cfg["Domain"] == "Fleet Operations"]
    fac_bsa    = fac_cfg[fac_cfg["Domain"] == "Building Systems & Assets"]
    fac_ops    = fac_cfg[fac_cfg["Domain"] == "Facilities Operations"]

    ve_comps,  ve_base,  ve_delta  = _build_component_series(fleet_ve,  "Pct_of_Fleet_Cost",     _FLEET_VE_PT,  fleet_base_M, shock)
    ops_comps, ops_base, ops_delta = _build_component_series(fleet_ops, "Pct_of_Fleet_Cost",     _FLEET_OPS_PT, fleet_base_M, shock)
    bsa_comps, bsa_base, bsa_delta = _build_component_series(fac_bsa,   "Pct_of_Facility_Cost",  _FAC_BSA_PT,   fac_base_M,   shock)
    fo_comps,  fo_base,  fo_delta  = _build_component_series(fac_ops,   "Pct_of_Facility_Cost",  _FAC_OPS_PT,   fac_base_M,   shock)

    # ── Summary KPIs ──────────────────────────────────────────────────────────
    fleet_total_delta = sum(ve_delta) + sum(ops_delta)
    fac_total_delta   = sum(bsa_delta) + sum(fo_delta)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric(f"Fleet Base FY{comp_yr}", f"${fleet_base_M:.1f}M")
    with k2:
        st.metric("Fleet Oil Shock Δ", f"+${fleet_total_delta:.2f}M",
                  delta=f"+{fleet_total_delta/fleet_base_M*100:.1f}% of fleet budget",
                  delta_color="inverse")
    with k3:
        st.metric(f"Facilities Base FY{comp_yr}", f"${fac_base_M:.1f}M")
    with k4:
        st.metric("Facilities Oil Shock Δ", f"+${fac_total_delta:.2f}M",
                  delta=f"+{fac_total_delta/fac_base_M*100:.1f}% of fac budget",
                  delta_color="inverse")

    st.divider()

    # ── Fleet charts (side by side) ───────────────────────────────────────────
    st.markdown(f"#### Fleet — Component Exposure ({shock_label}, FY{comp_yr})")
    fc_l, fc_r = st.columns(2)
    with fc_l:
        st.plotly_chart(
            make_component_shock_bar(
                ve_comps, ve_base, ve_delta,
                title="Vehicles & Equipment",
                height=420,
            ),
            use_container_width=True,
        )
    with fc_r:
        st.plotly_chart(
            make_component_shock_bar(
                ops_comps, ops_base, ops_delta,
                title="Fleet Operations",
                height=700,
            ),
            use_container_width=True,
        )

    st.divider()

    # ── Facilities charts (side by side) ──────────────────────────────────────
    st.markdown(f"#### Facilities — Component Exposure ({shock_label}, FY{comp_yr})")
    fa_l, fa_r = st.columns(2)
    with fa_l:
        st.plotly_chart(
            make_component_shock_bar(
                bsa_comps, bsa_base, bsa_delta,
                title="Building Systems & Assets",
                height=480,
            ),
            use_container_width=True,
        )
    with fa_r:
        st.plotly_chart(
            make_component_shock_bar(
                fo_comps, fo_base, fo_delta,
                title="Facilities Operations",
                height=900,
            ),
            use_container_width=True,
        )

    with st.expander("Passthrough factor methodology", expanded=True):
        st.markdown(f"""
**Oil passthrough factors** estimate how much of a 1% increase in WTI crude translates
to a cost increase in each component. Applied as: `delta = base_cost × passthrough × shock_pct`.

| Exposure level | Passthrough | Examples |
|---|---|---|
| Direct (100%) | 1.00 | Fuel/Energy |
| Very High (21–28%) | 0.21–0.28 | HVAC, Electrical, Elevators, EV Charging |
| High (15–22%) | 0.15–0.22 | Tires, Lubricants/Fluids, Snow & Ice Removal |
| Medium (8–12%) | 0.08–0.12 | Scheduled Maint., Unscheduled Repairs, Waste Mgmt |
| Low (3–6%) | 0.03–0.06 | Acquisition, Procurement, Capital Projects |
| Marginal (1–2%) | 0.01–0.02 | Software platforms, Admin, Compliance |

Sources: GFOA Fleet Benchmarks, DOT Commodity Index Studies, DOE CBECS.
        """)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  TAB — Oil Price vs Fleet                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
if active_tab == "🛢️ Oil Price vs Fleet":

    # ── View toggle ───────────────────────────────────────────────────────────
    oil_view = st.pills(
        "", ["Budget Appropriations", "Actual Spend"],
        default="Budget Appropriations", key="oil_view_fleet",
        label_visibility="collapsed",
    )

    st.divider()

    # ── Compact metric cards + yellow info popover ─────────────────────────
    st.markdown("""<style>
[data-testid="stMetric"]{padding:.25rem .5rem!important;background:#ffffff!important;border:1px solid #e2e8f0!important;border-left:3px solid #3b82f6!important;border-radius:8px!important;box-shadow:0 1px 4px rgba(0,0,0,0.06)!important}
[data-testid="stMetricLabel"] p{font-size:.72rem!important;color:#64748b!important;margin-bottom:.1rem!important}
[data-testid="stMetricValue"]{font-size:1.05rem!important;color:#0f172a!important}
[data-testid="stMetricDelta"] svg,[data-testid="stMetricDelta"] p{font-size:.65rem!important}
[data-testid="stPopover"]>button{background:#f59e0b!important;color:#1e293b!important;
  border-radius:50%!important;width:26px!important;height:26px!important;min-height:0!important;
  padding:0!important;font-size:.85rem!important;font-weight:bold!important;border:none!important;
  line-height:1!important;display:flex!important;align-items:center!important;justify-content:center!important}
</style>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # VIEW A — Budget Appropriations (chicago_energy_budget_2016_2025.xlsx)
    # ══════════════════════════════════════════════════════════════════════════
    if oil_view == "Budget Appropriations":

        hf   = load_budget_df()
        oc_b = load_oil_correlation()

        _fleet_cats_b = ["Motor Vehicle Diesel Fuel", "Motor Vehicle Gasoline", "Alternative Fuel"]
        years_sorted  = sorted(hf["Fiscal_Year"].unique())
        fleet_series  = hf[hf["Category"].isin(_fleet_cats_b)].groupby("Fiscal_Year")["Amount_M"].sum()
        yr_max  = int(fleet_series.index.max())
        yr_prev = yr_max - 1

        st.markdown("### Fleet Fuel Budget vs WTI Crude — FY2016–2025")

        k1, k2, k3 = st.columns(3)
        with k1:
            d_max  = fleet_series.get(yr_max, 0)
            d_prev = fleet_series.get(yr_prev, 0)
            st.metric(f"Fleet Fuel Budget FY{yr_max}", f"${d_max:.2f}M",
                      delta=f"{'+'if (d_max-d_prev)>=0 else ''}{d_max-d_prev:.2f}M vs FY{yr_prev}",
                      delta_color="inverse" if (d_max - d_prev) > 0 else "normal")
        with k2:
            peak_yr  = int(fleet_series.idxmax())
            peak_val = fleet_series.max()
            st.metric("Peak Fleet Budget", f"${peak_val:.2f}M",
                      delta=f"FY{peak_yr}", delta_color="off")
        with k3:
            st.metric("WTI Today", f"${oc_b['current_wti']:.2f}/bbl",
                      delta=f"Crisis peak ${oc_b['spike_wti']:.0f} (Mar 9)",
                      delta_color="inverse")

        # ── Build chart ────────────────────────────────────────────────────────
        fleet_dates = [pd.Timestamp(f"{y}-01-01") for y in years_sorted]
        fleet_vals  = [float(fleet_series.get(y, 0)) for y in years_sorted]
        base_fleet_vol = fleet_vals[-1] * 1e6 / 4.10

        scen_b = generate_wti_forecast(
            current_wti        = oc_b["current_wti"],
            peak_wti           = float(sb_peak_wti),
            peak_year          = sb_peak_yr,
            halflife_yrs       = sb_halflife,
            long_run_eq        = float(sb_lr),
            sigma              = 0.22,
            base_diesel_vol_gal= base_fleet_vol,
            base_elec_M        = None,
        )

        _wti_pairs_b = [(d, p) for d, p in zip(oc_b["wti_dates"], oc_b["wti_prices"])
                        if pd.Timestamp(d).year >= 2016]
        oc_b_chart = {
            **oc_b,
            "wti_dates":            [p[0] for p in _wti_pairs_b],
            "wti_prices":           [p[1] for p in _wti_pairs_b],
            "diesel_dates":         fleet_dates,
            "diesel_budget":        fleet_vals,
            "elec_dates":           fleet_dates,
            "elec_budget":          [0.0] * len(fleet_dates),
            "diesel_est_2026_M":    None,
            "diesel_budget_2026_M": None,
            **scen_b,
        }

        st.plotly_chart(
            make_oil_cost_chart(
                oc_b_chart,
                diesel_label      = "Fleet Fuel Budget ($M)",
                diesel_proj_label = "Fleet Fuel Projected",
                title             = "Fleet Fuel Budget vs WTI Crude  (2016 – 2033)",
                x_range           = ["2016-01-01", "2033-12-31"],
                show_fleet        = True,
                show_elec         = False,
            ),
            use_container_width=True,
        )

        st.divider()

        # ── Appropriations detail table — fleet categories only ────────────────
        st.markdown("**Fleet Fuel Appropriations by Category and Fiscal Year ($M)**")
        hf_fleet = hf[hf["Category"].isin(_fleet_cats_b)]
        pivot = hf_fleet.pivot_table(
            index="Category", columns="Fiscal_Year", values="Amount_M", aggfunc="sum", fill_value=0
        ).reset_index()
        pivot.columns.name = None
        pivot["Total FY2016–2025"] = pivot[[c for c in pivot.columns if isinstance(c, int)]].sum(axis=1)
        for col in [c for c in pivot.columns if isinstance(c, int)] + ["Total FY2016–2025"]:
            pivot[col] = pivot[col].round(3)
        st.dataframe(pivot, hide_index=True, use_container_width=True)

        with st.expander("Source & Methodology", expanded=True):
            st.markdown(f"""
**Primary source:** `chicago_energy_budget_2016_2025.xlsx` — annual appropriations downloaded from
Chicago Data Portal (Budget Ordinance - Appropriations datasets, FY2016–FY2025).

**Fleet accounts:** Motor Vehicle Diesel Fuel, Motor Vehicle Gasoline, Alternative Fuel.

**WTI overlay:** Annual average computed from monthly data (`CL=F` via `yfinance`).
Only years with full coverage are included.

**Fleet Budget projection:** Ornstein-Uhlenbeck mean-reversion model anchored to FY{yr_max}
fleet appropriation as base volume. Passthrough: `$0.024/gal per $/bbl WTI`.
            """)

    # ══════════════════════════════════════════════════════════════════════════
    # VIEW B — Fleet Actual Spend
    # ══════════════════════════════════════════════════════════════════════════
    else:  # oil_view == "Actual Spend"

        oc_s            = load_oil_correlation()
        fuel_es, _elec  = load_energy_spend()
        vend_df         = load_vendor_detail()

        st.markdown("### Fleet Fuel Actual Spend vs WTI Crude — Payments Dataset")

        # ── KPI banner ────────────────────────────────────────────────────────
        recent_fuel   = fuel_es[fuel_es["year"] >= 2021]
        peak_fuel_row = recent_fuel.loc[recent_fuel["value_M"].idxmax()]
        n_yrs_f       = len(recent_fuel)
        total_fuel_r  = recent_fuel["value_M"].sum()

        s1, s2, s3 = st.columns(3)
        with s1:
            st.metric("WTI Today", f"${oc_s['current_wti']:.2f}/bbl",
                      delta=f"Crisis peak ${oc_s['spike_wti']:.0f} (Mar 9)",
                      delta_color="inverse")
        with s2:
            st.metric("Peak Fleet Fuel Spend", f"${peak_fuel_row['value_M']:.1f}M",
                      delta=f"FY{int(peak_fuel_row['year'])}", delta_color="off")
        with s3:
            st.metric(f"Total Fleet Spend FY2021–{int(recent_fuel['year'].max())}",
                      f"${total_fuel_r:.1f}M",
                      delta=f"Avg ${total_fuel_r/n_yrs_f:.1f}M/yr", delta_color="off")

        # Exclude current partial year
        CURRENT_PARTIAL_YEAR = 2026
        fuel_es_plot = fuel_es[fuel_es["year"] < CURRENT_PARTIAL_YEAR]
        fuel_dates_s = [pd.Timestamp(f"{int(y)}-01-01") for y in fuel_es_plot["year"]]
        fuel_vals_s  = fuel_es_plot["value_M"].tolist()

        last_fuel_yr  = int(fuel_es_plot["year"].max())
        last_fuel_M   = float(fuel_es_plot[fuel_es_plot["year"] == last_fuel_yr]["value_M"].values[0])
        base_fuel_vol = last_fuel_M * 1e6 / 4.10

        scen_s = generate_wti_forecast(
            current_wti        = oc_s["current_wti"],
            peak_wti           = float(sb_peak_wti),
            peak_year          = sb_peak_yr,
            halflife_yrs       = sb_halflife,
            long_run_eq        = float(sb_lr),
            sigma              = 0.22,
            base_diesel_vol_gal= base_fuel_vol,
            base_elec_M        = None,
        )

        _wti_pairs_s = [(d, p) for d, p in zip(oc_s["wti_dates"], oc_s["wti_prices"])
                        if pd.Timestamp(d).year >= 2016]
        oc_s_chart = {
            **oc_s,
            "wti_dates":            [p[0] for p in _wti_pairs_s],
            "wti_prices":           [p[1] for p in _wti_pairs_s],
            "diesel_dates":         fuel_dates_s,
            "diesel_budget":        fuel_vals_s,
            "elec_dates":           fuel_dates_s,
            "elec_budget":          [0.0] * len(fuel_dates_s),
            "diesel_est_2026_M":    None,
            "diesel_budget_2026_M": None,
            **scen_s,
        }

        st.plotly_chart(
            make_oil_cost_chart(
                oc_s_chart,
                diesel_label      = "Fleet Fuel Spend ($M)",
                diesel_proj_label = "Fleet Fuel Projected",
                title             = "Fleet Fuel Actual Spend vs WTI Crude  (2016 – 2033)",
                x_range           = ["2016-01-01", "2033-12-31"],
                show_fleet        = True,
                show_elec         = False,
            ),
            use_container_width=True,
        )

        st.divider()

        # ── Vendor breakdown — fleet fuel only ────────────────────────────────
        st.markdown("### Vendor Breakdown — Fleet Fuel Actual Spend")
        vend_fuel  = vend_df[vend_df["type"] == "fuel"].copy()
        vend_fuel  = vend_fuel[vend_fuel["year"] >= 2010]
        vend_pivot = vend_fuel.groupby(["year", "vendor"])["value_M"].sum().reset_index()
        top_vendors = (
            vend_pivot.groupby("vendor")["value_M"].sum().nlargest(8).index.tolist()
        )
        vend_pivot["vendor_label"] = vend_pivot["vendor"].apply(
            lambda v: v if v in top_vendors else "Others"
        )
        vend_agg = vend_pivot.groupby(["year", "vendor_label"])["value_M"].sum().reset_index()
        VENDOR_PALETTE = [
            "#a78bfa","#38bdf8","#34d399","#fb923c",
            "#f87171","#fbbf24","#e879f9","#94a3b8","#64748b",
        ]
        all_vendors = sorted(
            vend_agg["vendor_label"].unique(),
            key=lambda v: vend_agg[vend_agg["vendor_label"]==v]["value_M"].sum(),
            reverse=True,
        )
        from config import PLOTLY_BASE, _AXIS_BASE, _LEGEND_BASE, _MARGIN_BASE
        fig_vend = go.Figure()
        for i, vname in enumerate(all_vendors):
            sub     = vend_agg[vend_agg["vendor_label"] == vname].set_index("year")
            all_yrs = sorted(vend_agg["year"].unique())
            fig_vend.add_trace(go.Bar(
                x=all_yrs,
                y=[float(sub["value_M"].get(y, 0)) for y in all_yrs],
                name=vname,
                marker_color=VENDOR_PALETTE[i % len(VENDOR_PALETTE)],
                hovertemplate=f"<b>{vname}</b><br>%{{x}}  —  $%{{y:.2f}}M<extra></extra>",
            ))
        fig_vend.update_layout(
            **PLOTLY_BASE, barmode="stack", height=420,
            margin=dict(**_MARGIN_BASE), legend=dict(**_LEGEND_BASE),
            title=dict(text="Fleet Fuel Spend by Vendor (2010–2024)",
                       font=dict(size=14, color="#e2e8f0"), x=0),
            xaxis=dict(**_AXIS_BASE, title="Fiscal Year"),
            yaxis=dict(**_AXIS_BASE, title="Spend ($M)"),
        )
        st.plotly_chart(fig_vend, use_container_width=True)

        st.divider()

        # ── Data table ────────────────────────────────────────────────────────
        st.markdown("**Fleet Fuel Spend by Year ($M)**")
        fleet_tbl = fuel_es.rename(columns={"value_M": "Fleet Fuel ($M)", "year": "Fiscal Year"})
        fleet_tbl["Fleet Fuel ($M)"] = fleet_tbl["Fleet Fuel ($M)"].round(3)
        st.dataframe(fleet_tbl.sort_values("Fiscal Year", ascending=False),
                     hide_index=True, use_container_width=True)

        st.divider()

        # ── Excel download ─────────────────────────────────────────────────────
        st.markdown("#### Download Data")
        import io

        def _build_excel_fleet_spend() -> bytes:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as xw:
                fleet_tbl.to_excel(xw, sheet_name="Fleet Fuel by Year", index=False)
                vend_fuel.rename(columns={"value_M": "Spend ($M)"}).to_excel(
                    xw, sheet_name="Vendor Level", index=False)
                wti_hist = pd.DataFrame({
                    "Date": oc_s["wti_dates"], "WTI ($/bbl)": oc_s["wti_prices"],
                })
                wti_hist["Date"] = pd.to_datetime(wti_hist["Date"])
                wti_hist.to_excel(xw, sheet_name="WTI Historical", index=False)
            return buf.getvalue()

        st.download_button(
            label="⬇ Download Excel — Fleet Fuel Spend + WTI",
            data=_build_excel_fleet_spend(),
            file_name="chicago_dps_fleet_fuel_spend_vs_wti.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        with st.expander("Source & Caveats", expanded=True):
            st.markdown("""
**Data source:** `chicago_energy_clean_strict.xlsx` — consolidated actual spend by vendor,
extracted from Chicago Data Portal Payments Dataset.

**Fleet Fuel (type = fuel):**
- Historical coverage 2002–2024; partial or contract-transition years excluded.
- Primary vendor: Colonial Oil Industries, Inc. (contracts 129971 diesel + 129972 gasoline).
- FY2021: excludes ~$3.1M from World Fuel Services (prior contract wind-down).

**Projection:** Ornstein-Uhlenbeck mean-reversion anchored to last actual spend year.
Sigma = 22% (calibrated from FY2021–2024 historical volatility).
            """)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  TAB — Oil Price vs Facilities                                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
if active_tab == "🏢 Oil Price vs Facilities":

    oil_view_fac = st.pills(
        "", ["Budget Appropriations", "Actual Spend"],
        default="Budget Appropriations", key="oil_view_fac",
        label_visibility="collapsed",
    )

    st.divider()

    # ── Compact metric CSS ────────────────────────────────────────────────────
    st.markdown("""<style>
[data-testid="stMetric"]{padding:.25rem .5rem!important;background:#ffffff!important;border:1px solid #e2e8f0!important;border-left:3px solid #3b82f6!important;border-radius:8px!important;box-shadow:0 1px 4px rgba(0,0,0,0.06)!important}
[data-testid="stMetricLabel"] p{font-size:.72rem!important;color:#64748b!important;margin-bottom:.1rem!important}
[data-testid="stMetricValue"]{font-size:1.05rem!important;color:#0f172a!important}
[data-testid="stMetricDelta"] svg,[data-testid="stMetricDelta"] p{font-size:.65rem!important}
[data-testid="stPopover"]>button{background:#f59e0b!important;color:#1e293b!important;
  border-radius:50%!important;width:26px!important;height:26px!important;min-height:0!important;
  padding:0!important;font-size:.85rem!important;font-weight:bold!important;border:none!important;
  line-height:1!important;display:flex!important;align-items:center!important;justify-content:center!important}
</style>""", unsafe_allow_html=True)

    _FAC_CATS = ["Electricity", "Natural Gas", "Fuel Oil"]

    # ══════════════════════════════════════════════════════════════════════════
    # VIEW A — Facilities Budget Appropriations
    # ══════════════════════════════════════════════════════════════════════════
    if oil_view_fac == "Budget Appropriations":

        hf    = load_budget_df()
        oc_fa = load_oil_correlation()

        years_sorted = sorted(hf["Fiscal_Year"].unique())
        fac_series   = hf[hf["Category"].isin(_FAC_CATS)].groupby("Fiscal_Year")["Amount_M"].sum()
        elec_series  = hf[hf["Category"] == "Electricity"].set_index("Fiscal_Year")["Amount_M"]
        ng_series    = hf[hf["Category"] == "Natural Gas"].set_index("Fiscal_Year")["Amount_M"]
        yr_max  = int(fac_series.index.max())
        yr_prev = yr_max - 1

        st.markdown("### Facilities Energy Budget vs WTI Crude — FY2016–2025")

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            f_max  = fac_series.get(yr_max, 0)
            f_prev = fac_series.get(yr_prev, 0)
            st.metric(f"Facilities Budget FY{yr_max}", f"${f_max:.1f}M",
                      delta=f"{'+'if (f_max-f_prev)>=0 else ''}{f_max-f_prev:.1f}M vs FY{yr_prev}",
                      delta_color="inverse" if (f_max - f_prev) > 0 else "normal")
        with k2:
            e_max  = elec_series.get(yr_max, 0)
            e_prev = elec_series.get(yr_prev, 0)
            st.metric(f"Electricity FY{yr_max}", f"${e_max:.1f}M",
                      delta=f"{'+'if (e_max-e_prev)>=0 else ''}{e_max-e_prev:.1f}M vs FY{yr_prev}",
                      delta_color="inverse" if (e_max - e_prev) > 0 else "normal")
        with k3:
            ng_max  = ng_series.get(yr_max, 0)
            ng_prev = ng_series.get(yr_prev, 0)
            st.metric(f"Natural Gas FY{yr_max}", f"${ng_max:.1f}M",
                      delta=f"{'+'if (ng_max-ng_prev)>=0 else ''}{ng_max-ng_prev:.1f}M vs FY{yr_prev}",
                      delta_color="inverse" if (ng_max - ng_prev) > 0 else "normal")
        with k4:
            st.metric("WTI Today", f"${oc_fa['current_wti']:.2f}/bbl",
                      delta=f"Crisis peak ${oc_fa['spike_wti']:.0f} (Mar 9)",
                      delta_color="inverse")

        # ── Build chart ────────────────────────────────────────────────────────
        fac_dates = [pd.Timestamp(f"{y}-01-01") for y in years_sorted]
        fac_vals  = [float(fac_series.get(y, 0)) for y in years_sorted]

        scen_fa = generate_wti_forecast(
            current_wti        = oc_fa["current_wti"],
            peak_wti           = float(sb_peak_wti),
            peak_year          = sb_peak_yr,
            halflife_yrs       = sb_halflife,
            long_run_eq        = float(sb_lr),
            sigma              = 0.22,
            base_diesel_vol_gal= None,
            base_elec_M        = fac_vals[-1],
        )

        _wti_pairs_fa = [(d, p) for d, p in zip(oc_fa["wti_dates"], oc_fa["wti_prices"])
                         if pd.Timestamp(d).year >= 2016]
        oc_fa_chart = {
            **oc_fa,
            "wti_dates":            [p[0] for p in _wti_pairs_fa],
            "wti_prices":           [p[1] for p in _wti_pairs_fa],
            "diesel_dates":         fac_dates,
            "diesel_budget":        [0.0] * len(fac_dates),
            "elec_dates":           fac_dates,
            "elec_budget":          fac_vals,
            "diesel_est_2026_M":    None,
            "diesel_budget_2026_M": None,
            **scen_fa,
        }

        st.plotly_chart(
            make_oil_cost_chart(
                oc_fa_chart,
                elec_label      = "Facilities Budget ($M)",
                elec_proj_label = "Facilities Projected",
                title           = "Facilities Budget vs WTI Crude  (2016 – 2033)",
                x_range         = ["2016-01-01", "2033-12-31"],
                show_fleet      = False,
                show_elec       = True,
            ),
            use_container_width=True,
        )

        st.divider()

        # ── Appropriations detail table — facilities categories ────────────────
        st.markdown("**Facilities Energy Appropriations by Category and Fiscal Year ($M)**")
        hf_fac = hf[hf["Category"].isin(_FAC_CATS)]
        pivot_fac = hf_fac.pivot_table(
            index="Category", columns="Fiscal_Year", values="Amount_M", aggfunc="sum", fill_value=0
        ).reset_index()
        pivot_fac.columns.name = None
        pivot_fac["Total FY2016–2025"] = pivot_fac[[c for c in pivot_fac.columns if isinstance(c, int)]].sum(axis=1)
        for col in [c for c in pivot_fac.columns if isinstance(c, int)] + ["Total FY2016–2025"]:
            pivot_fac[col] = pivot_fac[col].round(3)
        st.dataframe(pivot_fac, hide_index=True, use_container_width=True)

        with st.expander("Source & Methodology", expanded=True):
            st.markdown(f"""
**Primary source:** `chicago_energy_budget_2016_2025.xlsx` — annual appropriations downloaded from
Chicago Data Portal (Budget Ordinance - Appropriations datasets, FY2016–FY2025).

**Facilities accounts:** Electricity (grid power), Natural Gas (heating), Fuel Oil (boilers/heating backup).
Fuel Oil is treated as a facilities energy source because it is burned in building heating systems,
not used in fleet vehicles.

**WTI passthrough to facilities:** 0.21 combined (electricity 80% × 0.15 + natural gas 20% × 0.45).
Fuel Oil has near-direct crude price correlation (petroleum distillate).

**Facilities Budget projection:** Ornstein-Uhlenbeck mean-reversion anchored to FY{yr_max}
facilities total as base value. WTI passthrough factor: 0.15 (electricity grid component).
            """)

    # ══════════════════════════════════════════════════════════════════════════
    # VIEW B — Facilities Actual Spend (electricity only in payments dataset)
    # ══════════════════════════════════════════════════════════════════════════
    else:  # oil_view_fac == "Actual Spend"

        oc_fa           = load_oil_correlation()
        _fuel, elec_es  = load_energy_spend()
        vend_df         = load_vendor_detail()

        st.markdown("### Facilities Electricity Actual Spend vs WTI Crude")

        # ── KPI banner ────────────────────────────────────────────────────────
        recent_elec   = elec_es[elec_es["year"] >= 2021]
        peak_elec_row = recent_elec.loc[recent_elec["value_M"].idxmax()]
        n_yrs_e       = len(recent_elec)
        total_elec_r  = recent_elec["value_M"].sum()

        s1, s2, s3 = st.columns(3)
        with s1:
            st.metric("WTI Today", f"${oc_fa['current_wti']:.2f}/bbl",
                      delta=f"Crisis peak ${oc_fa['spike_wti']:.0f} (Mar 9)",
                      delta_color="inverse")
        with s2:
            st.metric("Peak Electricity Spend", f"${peak_elec_row['value_M']:.1f}M",
                      delta=f"FY{int(peak_elec_row['year'])}", delta_color="off")
        with s3:
            st.metric(f"Total Elec Spend FY2021–{int(recent_elec['year'].max())}",
                      f"${total_elec_r:.1f}M",
                      delta=f"Avg ${total_elec_r/n_yrs_e:.1f}M/yr", delta_color="off")

        # Exclude current partial year
        CURRENT_PARTIAL_YEAR = 2026
        elec_es_plot = elec_es[elec_es["year"] < CURRENT_PARTIAL_YEAR]
        elec_dates_fa = [pd.Timestamp(f"{int(y)}-01-01") for y in elec_es_plot["year"]]
        elec_vals_fa  = elec_es_plot["value_M"].tolist()

        last_elec_yr = int(elec_es_plot["year"].max())
        last_elec_M  = float(elec_es_plot[elec_es_plot["year"] == last_elec_yr]["value_M"].values[0])

        scen_fa_s = generate_wti_forecast(
            current_wti        = oc_fa["current_wti"],
            peak_wti           = float(sb_peak_wti),
            peak_year          = sb_peak_yr,
            halflife_yrs       = sb_halflife,
            long_run_eq        = float(sb_lr),
            sigma              = 0.22,
            base_diesel_vol_gal= None,
            base_elec_M        = last_elec_M,
        )

        _wti_pairs_fa = [(d, p) for d, p in zip(oc_fa["wti_dates"], oc_fa["wti_prices"])
                         if pd.Timestamp(d).year >= 2016]
        oc_fa_s_chart = {
            **oc_fa,
            "wti_dates":            [p[0] for p in _wti_pairs_fa],
            "wti_prices":           [p[1] for p in _wti_pairs_fa],
            "diesel_dates":         elec_dates_fa,
            "diesel_budget":        [0.0] * len(elec_dates_fa),
            "elec_dates":           elec_dates_fa,
            "elec_budget":          elec_vals_fa,
            "diesel_est_2026_M":    None,
            "diesel_budget_2026_M": None,
            **scen_fa_s,
        }

        st.plotly_chart(
            make_oil_cost_chart(
                oc_fa_s_chart,
                elec_label      = "Electricity Spend ($M)",
                elec_proj_label = "Electricity Projected",
                title           = "Electricity Actual Spend vs WTI Crude  (2016 – 2033)",
                x_range         = ["2016-01-01", "2033-12-31"],
                show_fleet      = False,
                show_elec       = True,
            ),
            use_container_width=True,
        )

        st.divider()

        # ── Vendor breakdown — electricity vendors ────────────────────────────
        st.markdown("### Vendor Breakdown — Electricity Actual Spend")
        vend_elec  = vend_df[vend_df["type"] == "electricity"].copy()
        vend_elec  = vend_elec[vend_elec["year"] >= 2010]
        vend_epivot = vend_elec.groupby(["year", "vendor"])["value_M"].sum().reset_index()
        top_evend = (
            vend_epivot.groupby("vendor")["value_M"].sum().nlargest(8).index.tolist()
        )
        vend_epivot["vendor_label"] = vend_epivot["vendor"].apply(
            lambda v: v if v in top_evend else "Others"
        )
        vend_eagg = vend_epivot.groupby(["year", "vendor_label"])["value_M"].sum().reset_index()
        VENDOR_PALETTE_E = [
            "#38bdf8","#34d399","#a78bfa","#fb923c",
            "#f87171","#fbbf24","#e879f9","#94a3b8","#64748b",
        ]
        all_evend = sorted(
            vend_eagg["vendor_label"].unique(),
            key=lambda v: vend_eagg[vend_eagg["vendor_label"]==v]["value_M"].sum(),
            reverse=True,
        )
        from config import PLOTLY_BASE, _AXIS_BASE, _LEGEND_BASE, _MARGIN_BASE
        fig_evend = go.Figure()
        for i, vname in enumerate(all_evend):
            sub     = vend_eagg[vend_eagg["vendor_label"] == vname].set_index("year")
            all_yrs = sorted(vend_eagg["year"].unique())
            fig_evend.add_trace(go.Bar(
                x=all_yrs,
                y=[float(sub["value_M"].get(y, 0)) for y in all_yrs],
                name=vname,
                marker_color=VENDOR_PALETTE_E[i % len(VENDOR_PALETTE_E)],
                hovertemplate=f"<b>{vname}</b><br>%{{x}}  —  $%{{y:.2f}}M<extra></extra>",
            ))
        fig_evend.update_layout(
            **PLOTLY_BASE, barmode="stack", height=420,
            margin=dict(**_MARGIN_BASE), legend=dict(**_LEGEND_BASE),
            title=dict(text="Electricity Spend by Vendor (2010–2024)",
                       font=dict(size=14, color="#e2e8f0"), x=0),
            xaxis=dict(**_AXIS_BASE, title="Fiscal Year"),
            yaxis=dict(**_AXIS_BASE, title="Spend ($M)"),
        )
        st.plotly_chart(fig_evend, use_container_width=True)

        st.divider()

        # ── Data table ────────────────────────────────────────────────────────
        st.markdown("**Electricity Spend by Year ($M)**")
        elec_tbl = elec_es.rename(columns={"value_M": "Electricity ($M)", "year": "Fiscal Year"})
        elec_tbl["Electricity ($M)"] = elec_tbl["Electricity ($M)"].round(3)
        st.dataframe(elec_tbl.sort_values("Fiscal Year", ascending=False),
                     hide_index=True, use_container_width=True)

        st.divider()

        # ── Excel download ─────────────────────────────────────────────────────
        st.markdown("#### Download Data")
        import io

        def _build_excel_fac_spend() -> bytes:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as xw:
                elec_tbl.to_excel(xw, sheet_name="Electricity by Year", index=False)
                vend_elec.rename(columns={"value_M": "Spend ($M)"}).to_excel(
                    xw, sheet_name="Vendor Level", index=False)
                wti_hist = pd.DataFrame({
                    "Date": oc_fa["wti_dates"], "WTI ($/bbl)": oc_fa["wti_prices"],
                })
                wti_hist["Date"] = pd.to_datetime(wti_hist["Date"])
                wti_hist.to_excel(xw, sheet_name="WTI Historical", index=False)
            return buf.getvalue()

        st.download_button(
            label="⬇ Download Excel — Electricity Spend + WTI",
            data=_build_excel_fac_spend(),
            file_name="chicago_dps_electricity_spend_vs_wti.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        with st.expander("Source & Caveats", expanded=True):
            st.markdown("""
**Data source:** `chicago_energy_clean_strict.xlsx` — consolidated actual spend by vendor,
extracted from Chicago Data Portal Payments Dataset.

**Scope:** Only electricity spend is available in the payments dataset at vendor level.
Natural Gas and Fuel Oil actual spend require separate utility billing records.
Use the Budget Appropriations view for the full facilities energy picture
(Electricity + Natural Gas + Fuel Oil combined).

**Electricity vendor:** Constellation NewEnergy, Inc. (contracts 29707 and 198906).
⚠ Includes all Chicago municipal electricity — airports + street lighting (~$22M/yr).
Isolating DFF/2FM buildings only requires ordinance budget detail.

**Projection:** Ornstein-Uhlenbeck mean-reversion anchored to last actual spend year.
WTI passthrough factor: 0.15 (electricity grid oil component).
            """)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  TAB — Oil Price vs Roadways                                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
if active_tab == "🛣️ Oil Price vs Roadways":

    oil_view_rdw = st.pills(
        "", ["Budget Appropriations", "Actual Spend"],
        default="Budget Appropriations", key="oil_view_rdw",
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("""<style>
[data-testid="stMetric"]{padding:.25rem .5rem!important;background:#ffffff!important;border:1px solid #e2e8f0!important;border-left:3px solid #3b82f6!important;border-radius:8px!important;box-shadow:0 1px 4px rgba(0,0,0,0.06)!important}
[data-testid="stMetricLabel"] p{font-size:.72rem!important;color:#64748b!important;margin-bottom:.1rem!important}
[data-testid="stMetricValue"]{font-size:1.05rem!important;color:#0f172a!important}
[data-testid="stMetricDelta"] svg,[data-testid="stMetricDelta"] p{font-size:.65rem!important}
[data-testid="stPopover"]>button{background:#f59e0b!important;color:#1e293b!important;
  border-radius:50%!important;width:26px!important;height:26px!important;min-height:0!important;
  padding:0!important;font-size:.85rem!important;font-weight:bold!important;border:none!important;
  line-height:1!important;display:flex!important;align-items:center!important;justify-content:center!important}
</style>""", unsafe_allow_html=True)

    df_avb, df_act_wt, df_bud_wt, df_rdw_vendor = load_cdot_roadway()
    oc_rdw = load_oil_correlation()

    from config import PLOTLY_BASE, _AXIS_BASE, _LEGEND_BASE, _MARGIN_BASE
    import io

    RDWY_PARTIAL_YEAR = 2026
    RDWY_START        = 2016

    # shared WTI filter
    _wti_pairs_rdw = [
        (d, p) for d, p in zip(oc_rdw["wti_dates"], oc_rdw["wti_prices"])
        if pd.Timestamp(d).year >= RDWY_START
    ]

    # ══════════════════════════════════════════════════════════════════════════
    # VIEW A — Budget Appropriations
    # ══════════════════════════════════════════════════════════════════════════
    if oil_view_rdw == "Budget Appropriations":

        df_bud = df_avb[
            (df_avb["YEAR"] >= RDWY_START) & (df_avb["YEAR"] < RDWY_PARTIAL_YEAR)
        ].copy()

        yr_max_b  = int(df_bud["YEAR"].max())
        yr_prev_b = yr_max_b - 1
        bud_max   = float(df_bud.loc[df_bud["YEAR"] == yr_max_b,  "BUDGET"].values[0])
        bud_prev  = float(df_bud.loc[df_bud["YEAR"] == yr_prev_b, "BUDGET"].values[0])
        peak_bud  = df_bud.loc[df_bud["BUDGET"].idxmax()]

        st.markdown("### CDOT Roadway Budget vs WTI Crude — FY2016–2025")

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric(
                f"Roadway Budget FY{yr_max_b}", f"${bud_max/1e6:.0f}M",
                delta=f"{'+'if (bud_max-bud_prev)>=0 else ''}{(bud_max-bud_prev)/1e6:.0f}M vs FY{yr_prev_b}",
                delta_color="inverse" if (bud_max - bud_prev) > 0 else "normal",
            )
        with k2:
            st.metric(
                "Peak Budget Year", f"${peak_bud['BUDGET']/1e6:.0f}M",
                delta=f"FY{int(peak_bud['YEAR'])}", delta_color="off",
            )
        with k3:
            total_bud_16_25 = float(df_bud["BUDGET"].sum())
            st.metric(
                "Total Budget FY2016–2025", f"${total_bud_16_25/1e6:.0f}M",
                delta=f"Avg ${total_bud_16_25/len(df_bud)/1e6:.0f}M/yr", delta_color="off",
            )
        with k4:
            st.metric(
                "WTI Today", f"${oc_rdw['current_wti']:.2f}/bbl",
                delta=f"Crisis peak ${oc_rdw['spike_wti']:.0f} (Mar 9)",
                delta_color="inverse",
            )

        # ── Run chart ─────────────────────────────────────────────────────────
        bud_dates  = [pd.Timestamp(f"{y}-01-01") for y in df_bud["YEAR"]]
        bud_vals   = (df_bud["BUDGET"] / 1e6).tolist()
        last_bud_M = bud_vals[-1]

        scen_bud = generate_wti_forecast(
            current_wti=oc_rdw["current_wti"], peak_wti=float(sb_peak_wti),
            peak_year=sb_peak_yr, halflife_yrs=sb_halflife,
            long_run_eq=float(sb_lr), sigma=0.22,
            base_diesel_vol_gal=None, base_elec_M=last_bud_M,
        )
        # Roadway projection: OLS trend extrapolation + oil passthrough
        _RDWY_PASS     = 0.22
        _PROJ_YRS_RDW  = [2027, 2028, 2029, 2030, 2031, 2032, 2033]

        _bud_hist_y  = np.array(df_bud["YEAR"].tolist(), dtype=float)
        _bud_hist_v  = np.array((df_bud["BUDGET"] / 1e6).tolist(), dtype=float)
        _bud_slope, _bud_icept = np.polyfit(_bud_hist_y, _bud_hist_v, 1)
        _bud_fitted  = _bud_slope * _bud_hist_y + _bud_icept
        _bud_rmse_frac = float(np.sqrt(np.mean((_bud_hist_v - _bud_fitted) ** 2)) / np.mean(_bud_hist_v))
        _bud_trend   = [_bud_slope * y + _bud_icept for y in _PROJ_YRS_RDW]

        def _rdw_bud_proj(wti_ann):
            return [round(t * (1 + _RDWY_PASS * (w - 72) / 72), 1) for t, w in zip(_bud_trend, wti_ann)]
        def _rdw_bud_hi(wti_ann):
            return [round(t * (1 + _RDWY_PASS * (w - 72) / 72) * (1 + _bud_rmse_frac), 1) for t, w in zip(_bud_trend, wti_ann)]
        def _rdw_bud_lo(wti_ann):
            return [round(max(0.0, t * (1 + _RDWY_PASS * (w - 72) / 72) * (1 - _bud_rmse_frac)), 1) for t, w in zip(_bud_trend, wti_ann)]

        oc_bud_chart = {
            **oc_rdw,
            "wti_dates":            [p[0] for p in _wti_pairs_rdw],
            "wti_prices":           [p[1] for p in _wti_pairs_rdw],
            "diesel_dates":         bud_dates,
            "diesel_budget":        [0.0] * len(bud_dates),
            "elec_dates":           bud_dates,
            "elec_budget":          bud_vals,
            "diesel_est_2026_M":    None,
            "diesel_budget_2026_M": None,
            **scen_bud,
            "elec_proj_base": _rdw_bud_proj(scen_bud["wti_proj_base"]),
            "elec_proj_hi":   _rdw_bud_hi(scen_bud["wti_proj_hi"]),
            "elec_proj_lo":   _rdw_bud_lo(scen_bud["wti_proj_lo"]),
        }
        st.plotly_chart(
            make_oil_cost_chart(
                oc_bud_chart,
                elec_label="Roadway Budget ($M)", elec_proj_label="Roadway Budget Projected",
                title="CDOT Roadway Budget vs WTI Crude  (2016 – 2033)",
                x_range=["2016-01-01", "2033-12-31"],
                show_fleet=False, show_elec=True,
            ),
            use_container_width=True,
        )

        st.divider()

        # ── Budget work type breakdown ─────────────────────────────────────────
        st.markdown("### Annual Roadway Budget by Work Type")

        _BUD_TYPES = [
            "General Construction", "Labor & Operations", "Engineering & Design",
            "Equipment & Maintenance", "Materials & Supplies", "Other Roadway",
        ]
        _BUD_COLORS = {
            "General Construction":  "#7c3aed",
            "Labor & Operations":    "#0284c7",
            "Engineering & Design":  "#059669",
            "Equipment & Maintenance": "#d97706",
            "Materials & Supplies":  "#db2777",
            "Other Roadway":         "#64748b",
        }
        df_bwt = df_bud_wt[
            (df_bud_wt["YEAR"] >= RDWY_START) & (df_bud_wt["YEAR"] < RDWY_PARTIAL_YEAR)
        ].copy()
        all_years_b = sorted(df_bwt["YEAR"].unique())

        fig_bwt = go.Figure()
        for btype in _BUD_TYPES:
            sub = df_bwt[df_bwt["TYPE"] == btype].set_index("YEAR")
            fig_bwt.add_trace(go.Bar(
                x=all_years_b,
                y=[float(sub["TOTAL_BUDGET"].get(y, 0)) / 1e6 for y in all_years_b],
                name=btype,
                marker_color=_BUD_COLORS.get(btype, "#94a3b8"),
                hovertemplate=f"<b>{btype}</b><br>FY%{{x}}  —  $%{{y:.1f}}M<extra></extra>",
            ))
        fig_bwt.update_layout(
            **PLOTLY_BASE, barmode="stack", height=420,
            margin=dict(**_MARGIN_BASE), legend=dict(**_LEGEND_BASE),
            title=dict(text="CDOT Roadway Budget by Work Type (FY2016–2025)",
                       font=dict(size=14, color="#1e293b"), x=0),
            xaxis=dict(**_AXIS_BASE, title="Fiscal Year", dtick=2),
            yaxis=dict(**_AXIS_BASE, title="Budget ($M)"),
        )
        st.plotly_chart(fig_bwt, use_container_width=True)

        st.divider()

        # ── Budget detail table ────────────────────────────────────────────────
        st.markdown("**CDOT Roadway Appropriations by Year ($M)**")
        tbl_bud = df_bud[["YEAR","BUDGET","ACTUALS","VARIANCE","VARIANCE_PCT"]].copy()
        tbl_bud["BUDGET"]       = (tbl_bud["BUDGET"]   / 1e6).round(1)
        tbl_bud["ACTUALS"]      = (tbl_bud["ACTUALS"]  / 1e6).round(1)
        tbl_bud["VARIANCE"]     = (tbl_bud["VARIANCE"] / 1e6).round(1)
        tbl_bud["VARIANCE_PCT"] = (tbl_bud["VARIANCE_PCT"] * 100).round(1)
        tbl_bud = tbl_bud.rename(columns={
            "YEAR": "Fiscal Year", "BUDGET": "Budget ($M)",
            "ACTUALS": "Actuals ($M)", "VARIANCE": "Variance ($M)",
            "VARIANCE_PCT": "Variance (%)",
        }).sort_values("Fiscal Year", ascending=False)
        st.dataframe(tbl_bud, hide_index=True, use_container_width=True)

        with st.expander("Source & Methodology", expanded=True):
            st.markdown(f"""
**Primary source:** `CDOT_Roadway_Actuals_and_Budget_v3.xlsx` — annual approved appropriations
from Chicago Annual Budget Ordinance datasets, FY2011–FY2026.

**Budget work types:** General Construction, Labor & Operations, Engineering & Design,
Equipment & Maintenance, Materials & Supplies, Other Roadway.

**Note on scale:** Budget authority for CDOT roadways reflects multi-year capital program
authorizations. Annual actuals (payments made) are typically 30–70% of budget authority
in any given year, as large capital contracts span multiple fiscal years.

**WTI projection:** Ornstein-Uhlenbeck mean-reversion anchored to FY{yr_max_b} budget total.
Passthrough: 0.15 (infrastructure materials oil component).

**FY2026 excluded** from charts — partial year only.
            """)

    # ══════════════════════════════════════════════════════════════════════════
    # VIEW B — Actual Spend
    # ══════════════════════════════════════════════════════════════════════════
    else:

        df_act = df_avb[
            (df_avb["YEAR"] >= RDWY_START) & (df_avb["YEAR"] < RDWY_PARTIAL_YEAR)
        ].copy()

        yr_max_a  = int(df_act["YEAR"].max())
        yr_prev_a = yr_max_a - 1
        act_max   = float(df_act.loc[df_act["YEAR"] == yr_max_a,  "ACTUALS"].values[0])
        act_prev  = float(df_act.loc[df_act["YEAR"] == yr_prev_a, "ACTUALS"].values[0])
        peak_act  = df_act.loc[df_act["ACTUALS"].idxmax()]
        total_act = float(df_act["ACTUALS"].sum())

        st.markdown("### CDOT Roadway Actual Spend vs WTI Crude — FY2016–2025")

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric(
                f"Roadway Spend FY{yr_max_a}", f"${act_max/1e6:.0f}M",
                delta=f"{'+'if (act_max-act_prev)>=0 else ''}{(act_max-act_prev)/1e6:.0f}M vs FY{yr_prev_a}",
                delta_color="inverse" if (act_max - act_prev) > 0 else "normal",
            )
        with k2:
            st.metric(
                "Peak Annual Spend", f"${peak_act['ACTUALS']/1e6:.0f}M",
                delta=f"FY{int(peak_act['YEAR'])}", delta_color="off",
            )
        with k3:
            st.metric(
                "Total FY2016–2025", f"${total_act/1e6:.0f}M",
                delta=f"Avg ${total_act/len(df_act)/1e6:.0f}M/yr", delta_color="off",
            )
        with k4:
            st.metric(
                "WTI Today", f"${oc_rdw['current_wti']:.2f}/bbl",
                delta=f"Crisis peak ${oc_rdw['spike_wti']:.0f} (Mar 9)",
                delta_color="inverse",
            )

        # ── Run chart ─────────────────────────────────────────────────────────
        act_dates  = [pd.Timestamp(f"{y}-01-01") for y in df_act["YEAR"]]
        act_vals   = (df_act["ACTUALS"] / 1e6).tolist()
        last_act_M = act_vals[-1]

        scen_act = generate_wti_forecast(
            current_wti=oc_rdw["current_wti"], peak_wti=float(sb_peak_wti),
            peak_year=sb_peak_yr, halflife_yrs=sb_halflife,
            long_run_eq=float(sb_lr), sigma=0.22,
            base_diesel_vol_gal=None, base_elec_M=last_act_M,
        )
        _act_hist_y  = np.array(df_act["YEAR"].tolist(), dtype=float)
        _act_hist_v  = np.array((df_act["ACTUALS"] / 1e6).tolist(), dtype=float)
        _act_slope, _act_icept = np.polyfit(_act_hist_y, _act_hist_v, 1)
        _act_fitted  = _act_slope * _act_hist_y + _act_icept
        _act_rmse_frac = float(np.sqrt(np.mean((_act_hist_v - _act_fitted) ** 2)) / np.mean(_act_hist_v))
        _act_trend   = [_act_slope * y + _act_icept for y in _PROJ_YRS_RDW]

        def _rdw_act_proj(wti_ann):
            return [round(t * (1 + _RDWY_PASS * (w - 72) / 72), 1) for t, w in zip(_act_trend, wti_ann)]
        def _rdw_act_hi(wti_ann):
            return [round(t * (1 + _RDWY_PASS * (w - 72) / 72) * (1 + _act_rmse_frac), 1) for t, w in zip(_act_trend, wti_ann)]
        def _rdw_act_lo(wti_ann):
            return [round(max(0.0, t * (1 + _RDWY_PASS * (w - 72) / 72) * (1 - _act_rmse_frac)), 1) for t, w in zip(_act_trend, wti_ann)]

        oc_act_chart = {
            **oc_rdw,
            "wti_dates":            [p[0] for p in _wti_pairs_rdw],
            "wti_prices":           [p[1] for p in _wti_pairs_rdw],
            "diesel_dates":         act_dates,
            "diesel_budget":        [0.0] * len(act_dates),
            "elec_dates":           act_dates,
            "elec_budget":          act_vals,
            "diesel_est_2026_M":    None,
            "diesel_budget_2026_M": None,
            **scen_act,
            "elec_proj_base": _rdw_act_proj(scen_act["wti_proj_base"]),
            "elec_proj_hi":   _rdw_act_hi(scen_act["wti_proj_hi"]),
            "elec_proj_lo":   _rdw_act_lo(scen_act["wti_proj_lo"]),
        }
        st.plotly_chart(
            make_oil_cost_chart(
                oc_act_chart,
                elec_label="Roadway Spend ($M)", elec_proj_label="Roadway Projected",
                title="CDOT Roadway Actual Spend vs WTI Crude  (2016 – 2033)",
                x_range=["2016-01-01", "2033-12-31"],
                show_fleet=False, show_elec=True,
            ),
            use_container_width=True,
        )

        st.divider()

        # ── Stacked bar: actuals by work type ─────────────────────────────────
        st.markdown("### Annual Roadway Spend by Work Type")

        _ACT_TYPES_ORDER = [
            "General Construction", "Other Roadway", "Engineering & Design",
            "Asphalt Paving", "Signals & Electrical", "Concrete", "Utility / Underground",
        ]
        _ACT_COLORS = {
            "General Construction":  "#7c3aed",
            "Other Roadway":         "#0284c7",
            "Engineering & Design":  "#059669",
            "Asphalt Paving":        "#d97706",
            "Signals & Electrical":  "#db2777",
            "Concrete":              "#64748b",
            "Utility / Underground": "#0891b2",
        }
        df_awt = df_act_wt[
            (df_act_wt["YEAR"] >= RDWY_START) & (df_act_wt["YEAR"] < RDWY_PARTIAL_YEAR)
        ].copy()
        all_years_a = sorted(df_awt["YEAR"].unique())

        fig_awt = go.Figure()
        for atype in _ACT_TYPES_ORDER:
            sub = df_awt[df_awt["TYPE"] == atype].set_index("YEAR")
            fig_awt.add_trace(go.Bar(
                x=all_years_a,
                y=[float(sub["TOTAL_SPEND"].get(y, 0)) / 1e6 for y in all_years_a],
                name=atype,
                marker_color=_ACT_COLORS.get(atype, "#94a3b8"),
                hovertemplate=f"<b>{atype}</b><br>FY%{{x}}  —  $%{{y:.1f}}M<extra></extra>",
            ))
        fig_awt.update_layout(
            **PLOTLY_BASE, barmode="stack", height=420,
            margin=dict(**_MARGIN_BASE), legend=dict(**_LEGEND_BASE),
            title=dict(text="CDOT Roadway Spend by Work Type (FY2016–2025)",
                       font=dict(size=14, color="#1e293b"), x=0),
            xaxis=dict(**_AXIS_BASE, title="Fiscal Year", dtick=2),
            yaxis=dict(**_AXIS_BASE, title="Spend ($M)"),
        )
        st.plotly_chart(fig_awt, use_container_width=True)

        st.divider()

        # ── Actuals detail table ───────────────────────────────────────────────
        st.markdown("**CDOT Roadway Actuals vs Budget by Year ($M)**")
        tbl_act = df_act[["YEAR","ACTUALS","BUDGET","VARIANCE","VARIANCE_PCT"]].copy()
        tbl_act["ACTUALS"]      = (tbl_act["ACTUALS"]  / 1e6).round(1)
        tbl_act["BUDGET"]       = (tbl_act["BUDGET"]   / 1e6).round(1)
        tbl_act["VARIANCE"]     = (tbl_act["VARIANCE"] / 1e6).round(1)
        tbl_act["VARIANCE_PCT"] = (tbl_act["VARIANCE_PCT"] * 100).round(1)
        tbl_act = tbl_act.rename(columns={
            "YEAR": "Fiscal Year", "ACTUALS": "Actuals ($M)",
            "BUDGET": "Budget ($M)", "VARIANCE": "Variance ($M)",
            "VARIANCE_PCT": "Variance (%)",
        }).sort_values("Fiscal Year", ascending=False)
        st.dataframe(tbl_act, hide_index=True, use_container_width=True)

        st.divider()

        # ── Excel download ─────────────────────────────────────────────────────
        st.markdown("#### Download Data")

        def _build_excel_rdw_act() -> bytes:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as xw:
                tbl_act.to_excel(xw, sheet_name="Actuals vs Budget", index=False)
                df_rdw_vendor.rename(columns={
                    "VENDOR": "Vendor", "TYPE": "Type", "CATEGORY": "Category",
                    "YEAR": "Fiscal Year", "TOTAL_SPEND": "Total Spend ($)",
                    "value_M": "Total Spend ($M)",
                }).to_excel(xw, sheet_name="Vendor Level", index=False)
                wti_hist = pd.DataFrame({
                    "Date": oc_rdw["wti_dates"], "WTI ($/bbl)": oc_rdw["wti_prices"],
                })
                wti_hist["Date"] = pd.to_datetime(wti_hist["Date"])
                wti_hist.to_excel(xw, sheet_name="WTI Historical", index=False)
            return buf.getvalue()

        st.download_button(
            label="⬇ Download Excel — CDOT Roadway Spend + WTI",
            data=_build_excel_rdw_act(),
            file_name="chicago_cdot_roadway_actuals_vs_wti.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        with st.expander("Source & Caveats", expanded=True):
            st.markdown(f"""
**Data source:** `CDOT_Roadway_Actuals_and_Budget_v3.xlsx` — vendor-level payments
extracted from Chicago Data Portal Payments Dataset (Payments.csv).

**Work type classification (actuals):**
- **General Construction** — road reconstruction, resurfacing, structural work.
- **Other Roadway** — miscellaneous roadway maintenance and services.
- **Engineering & Design** — planning, design, and inspection contracts.
- **Asphalt Paving** — direct petroleum derivative; high oil price correlation.
- **Signals & Electrical** — traffic signal installation and electrical work.
- **Concrete** — portland cement sidewalks, curbs, and structures.
- **Utility / Underground** — utility relocation and underground infrastructure.

**Oil passthrough (blended ≈0.22):**
Asphalt (~20% of spend) ≈ 0.55 passthrough (petroleum distillate).
General Construction (~45%) ≈ 0.15. Remaining types 0.05–0.15.

**WTI projection:** Ornstein-Uhlenbeck mean-reversion anchored to FY{yr_max_a} actuals.
Passthrough: 0.15.  **FY2026 excluded** — partial year only.
            """)


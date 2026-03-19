"""
config.py — Oil Price Impact Dashboard
Combined Fleet + Facilities Simulation
City of Chicago — DPS Assets Analysis

All paths, scenario definitions, elasticities, and chart styles.
Read-only references to Fleet and Facilities v3 data files.
"""

from pathlib import Path

# ── Root paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent

# Fleet simulation outputs (pre-computed CSVs — must live in repo root)
FLEET_SIM_ANNUAL = BASE_DIR / "sim_annual.csv"
FLEET_MC_BY_YEAR = BASE_DIR / "mc_runs_by_year.csv"

# Facilities outputs (must live in repo root)
FACILITIES_EXCEL = BASE_DIR / "asset_model_outputs.xlsx"

# ── Fuel price baselines (FY 2026 base) ──────────────────────────────────────
BASE_DIESEL_GAL  = 4.10   # $/gal diesel (fleet primary fuel)
BASE_GAS_GAL     = 3.80   # $/gal gasoline

# WTI crude to diesel retail passthrough (EIA historical: $0.24/gal per $10/bbl)
CRUDE_TO_DIESEL  = 0.024  # $/gal per $/bbl

# ── Oil shock scenarios ───────────────────────────────────────────────────────
SCENARIOS = {
    "Base ($4.10 diesel)":      0.00,
    "Mild shock (+25%)":        0.25,
    "Moderate shock (+50%)":    0.50,
    "Severe shock (+75%)":      0.75,
}

SCENARIO_COLORS = {
    "Base ($4.10 diesel)":    "#38bdf8",   # sky blue — bright on dark
    "Mild shock (+25%)":      "#a78bfa",   # violet
    "Moderate shock (+50%)":  "#fb923c",   # orange
    "Severe shock (+75%)":    "#f87171",   # red
}

FOCUS_YEARS = [2026, 2027]

# ── Fleet second-order elasticities ──────────────────────────────────────────
# Source: GFOA fleet benchmarks + DOT commodity index studies
FLEET_ELAST_PM      = 0.18   # PM materials / lubricants / tires
FLEET_ELAST_CM      = 0.12   # Corrective maintenance (labor + parts)
FLEET_ELAST_CAPEX   = 0.10   # Replacement CapEx (vehicle manufacturing)

# ── Facilities energy fraction & oil sensitivity ──────────────────────────────
# ~28% of Facilities O&M is energy (electricity + natural gas + fuel oil)
# Source: DOE CBECS + GFOA benchmarks for urban municipal building portfolios
FACILITIES_ENERGY_FRACTION = 0.28

# Oil-to-energy passthrough for facility utilities:
#   Electricity (80% of energy): 0.15 passthrough (generation mix / nat-gas peakers)
#   Natural gas (20% of energy): 0.45 passthrough (tracks oil with ~6-month lag)
# Weighted: 0.80*0.15 + 0.20*0.45 = 0.21
FACILITIES_OIL_PASSTHROUGH = 0.21

# Monte Carlo simulation parameters
MC_RUNS       = 125
MC_SEED       = 42

# Facilities pillar MC uncertainty (sigma as fraction of default rate)
PILLAR_SIGMA_FRAC = 0.20   # ±20% of default rate → 1-sigma uncertainty

# Facilities pillar parameters (mirror of facilities v3 config)
PILLAR_DEFAULTS = {
    "Work_Modernization":   {"r_p": 0.12, "r_min": 0.02, "r_max": 0.25, "k": 1.5, "x0": 3.0},
    "Demand_Management":    {"r_p": 0.07, "r_min": 0.01, "r_max": 0.20, "k": 1.2, "x0": 4.0},
    "Asset_Management":     {"r_p": 0.10, "r_min": 0.02, "r_max": 0.25, "k": 1.0, "x0": 5.0},
    "Vendor_Management":    {"r_p": 0.04, "r_min": 0.01, "r_max": 0.15, "k": 0.8, "x0": 4.0},
    "Payment_Management":   {"r_p": 0.025,"r_min": 0.01, "r_max": 0.12, "k": 1.8, "x0": 4.0},
    "Early_Pay_Management": {"r_p": 0.005,"r_min": 0.00, "r_max": 0.08, "k": 2.0, "x0": 5.0},
}
AI_K   = 0.85
AI_X0  = 3.5
AI_START_YEAR = 2024

# ── Chart palette — light mode, clean slate ───────────────────────────────────
CHART_BG   = "#f8fafc"   # near-white page
CHART_CARD = "#ffffff"   # pure white card

_AXIS_BASE = dict(
    gridcolor="rgba(100,116,139,0.12)",
    zerolinecolor="rgba(100,116,139,0.25)",
    linecolor="rgba(100,116,139,0.20)",
    tickfont=dict(color="#475569", size=11, family="'Inter', sans-serif"),
    title_font=dict(color="#1e293b", size=12, family="'Inter', sans-serif"),
    tickcolor="#94a3b8",
)
_LEGEND_BASE = dict(
    bgcolor="rgba(248,250,252,0.95)",
    bordercolor="rgba(100,116,139,0.25)",
    borderwidth=1,
    font=dict(size=10, color="#1e293b"),
    orientation="h",
    y=-0.22, x=0.5,
    xanchor="center", yanchor="top",
    itemsizing="constant",
    tracegroupgap=4,
)
_MARGIN_BASE = dict(l=60, r=24, t=56, b=90)
PLOTLY_BASE = dict(
    paper_bgcolor=CHART_BG,
    plot_bgcolor=CHART_CARD,
    font=dict(color="#1e293b", family="'Inter', sans-serif", size=11),
    hoverlabel=dict(
        bgcolor="#1e293b",
        bordercolor="rgba(100,116,139,0.40)",
        font=dict(color="#f8fafc", size=12),
    ),
)

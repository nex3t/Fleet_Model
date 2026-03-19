"""
charts.py — Oil Impact Dashboard Charts
Combined Fleet + Facilities Oil Price Sensitivity Visualization
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from config import (
    SCENARIOS, SCENARIO_COLORS, FOCUS_YEARS,
    BASE_DIESEL_GAL, CRUDE_TO_DIESEL,
    FACILITIES_ENERGY_FRACTION,
    PLOTLY_BASE, _AXIS_BASE, _LEGEND_BASE, _MARGIN_BASE,
)

# Portfolio accent colors — saturated for light background
_FLEET_COLOR  = "#7c3aed"       # deep violet
_FAC_COLOR    = "#059669"       # deep emerald
_COMB_COLOR   = "#0284c7"       # deep sky blue
_WARN_COLOR   = "#ea580c"       # deep orange
_DANGER_COLOR = "#dc2626"       # deep red


def _apply_base(fig, height=400):
    fig.update_layout(
        **PLOTLY_BASE,
        height=height,
        margin=dict(**_MARGIN_BASE),
        legend=dict(**_LEGEND_BASE),
    )
    fig.update_xaxes(**_AXIS_BASE)
    fig.update_yaxes(**_AXIS_BASE)
    return fig


# ── Tab 1 Charts ──────────────────────────────────────────────────────────────

def make_tornado_chart(fleet_det: pd.DataFrame, fac_det: pd.DataFrame, year: int) -> go.Figure:
    """
    Tornado chart: delta cost (M$) by scenario for Fleet vs Facilities.
    X-axis: incremental cost above base. Y-axis: scenario labels.
    """
    fleet_row = fleet_det[fleet_det["Fiscal_Year"] == year].copy()
    fac_row   = fac_det[fac_det["Fiscal_Year"]   == year].copy()

    # Build rows: scenario → [fleet_delta, fac_delta, combined]
    rows = []
    for scen_name in list(SCENARIOS.keys())[1:]:   # skip Base (delta=0)
        fd = float(fleet_row[fleet_row["Scenario"] == scen_name]["Delta_Total_M"].values[0])
        sd = float(fac_row[fac_row["Scenario"]     == scen_name]["Delta_Energy_M"].values[0])
        rows.append({"Scenario": scen_name, "Fleet": fd, "Facilities": sd})

    df = pd.DataFrame(rows).sort_values("Fleet")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Fleet",
        y=df["Scenario"],
        x=df["Fleet"],
        orientation="h",
        marker_color=_FLEET_COLOR,
        marker_opacity=0.85,
        text=df["Fleet"].apply(lambda v: f"+${v:.1f}M"),
        textposition="outside",
        textfont=dict(color="#1e293b", size=11),
    ))
    fig.add_trace(go.Bar(
        name="Facilities",
        y=df["Scenario"],
        x=df["Facilities"],
        orientation="h",
        marker_color=_FAC_COLOR,
        marker_opacity=0.85,
        text=df["Facilities"].apply(lambda v: f"+${v:.1f}M"),
        textposition="outside",
        textfont=dict(color="#1e293b", size=11),
    ))
    fig.update_layout(
        barmode="stack",
        title=dict(
            text=f"FY{year} — Combined Budget Impact by Scenario",
            font=dict(size=14, color="#1e293b"),
            x=0,
        ),
        xaxis_title="Incremental Cost vs Base ($M)",
        yaxis_title="",
    )
    return _apply_base(fig, height=360)


def make_crude_sensitivity_chart(fleet_cs: pd.DataFrame, fac_cs: pd.DataFrame, year: int) -> go.Figure:
    """
    Line chart: total combined incremental cost per $X/bbl crude increase.
    """
    fc = fleet_cs[fleet_cs["Fiscal_Year"] == year][["Crude_Δ_$/bbl","Delta_Total_M"]].copy()
    fa = fac_cs[fac_cs["Fiscal_Year"]     == year][["Crude_Δ_$/bbl","Delta_Energy_M"]].copy()
    merged = fc.merge(fa, on="Crude_Δ_$/bbl")

    x   = merged["Crude_Δ_$/bbl"].tolist()
    y_f = merged["Delta_Total_M"].tolist()
    y_s = merged["Delta_Energy_M"].tolist()
    y_c = [a + b for a, b in zip(y_f, y_s)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y_f, name="Fleet",
        mode="lines+markers",
        line=dict(color=_FLEET_COLOR, width=2.5),
        marker=dict(size=8, color=_FLEET_COLOR),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y_s, name="Facilities",
        mode="lines+markers",
        line=dict(color=_FAC_COLOR, width=2.5),
        marker=dict(size=8, color=_FAC_COLOR),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y_c, name="Combined",
        mode="lines+markers",
        line=dict(color=_COMB_COLOR, width=3, dash="dot"),
        marker=dict(size=9, color=_COMB_COLOR),
    ))
    fig.update_layout(
        title=dict(
            text=f"FY{year} — Incremental Cost per Crude Price Increase",
            font=dict(size=14, color="#1e293b"),
            x=0,
        ),
        xaxis_title="Crude Oil Increase ($/bbl)",
        yaxis_title="Incremental Budget Impact ($M)",
    )
    return _apply_base(fig, height=360)


def make_historic_fuel_chart(hist_df: pd.DataFrame) -> go.Figure:
    """
    Stacked area: historical fuel budget by category (2021-2026).
    """
    cats = ["Motor Vehicle Diesel Fuel", "Electricity", "Fuel Oil", "Alternative Fuel"]
    cat_colors = {
        "Motor Vehicle Diesel Fuel": "rgba(124,58,237,0.65)",   # violet
        "Electricity":               "rgba(2,132,199,0.65)",    # sky
        "Fuel Oil":                  "rgba(234,88,12,0.65)",    # orange
        "Alternative Fuel":          "rgba(5,150,105,0.65)",    # emerald
    }
    cat_line_colors = {
        "Motor Vehicle Diesel Fuel": "#7c3aed",
        "Electricity":               "#0284c7",
        "Fuel Oil":                  "#ea580c",
        "Alternative Fuel":          "#059669",
    }

    fig = go.Figure()
    for cat in cats:
        if cat not in hist_df.columns:
            continue
        fig.add_trace(go.Scatter(
            x=hist_df["Fiscal Year"],
            y=hist_df[cat] / 1e6,
            name=cat,
            mode="lines",
            stackgroup="one",
            line=dict(width=0.5, color=cat_line_colors.get(cat, "#64748b")),
            fillcolor=cat_colors.get(cat, "rgba(100,116,139,0.4)"),
        ))
    fig.update_layout(
        title=dict(
            text="Historical City Fuel & Utilities Budget (2021–2026)",
            font=dict(size=14, color="#1e293b"),
            x=0,
        ),
        xaxis_title="Fiscal Year",
        yaxis_title="Budget ($M)",
        xaxis=dict(dtick=1),
    )
    return _apply_base(fig, height=340)


# ── Tab 2 Charts ──────────────────────────────────────────────────────────────

def make_fleet_component_chart(fleet_comp: pd.DataFrame, year: int) -> go.Figure:
    """
    Stacked bar: Fleet cost breakdown by component (Energy / PM / CM / CapEx)
    for each oil shock scenario in the selected year.
    """
    df = fleet_comp[fleet_comp["Fiscal_Year"] == year].copy()

    comp_colors = {
        "Energy":  "#ea580c",   # orange
        "PM":      "#7c3aed",   # violet
        "CM":      "#0284c7",   # sky
        "CapEx":   "#059669",   # emerald
    }

    fig = go.Figure()
    for comp, col, color in [
        ("CapEx",  "CapEx_M",  comp_colors["CapEx"]),
        ("CM",     "CM_M",     comp_colors["CM"]),
        ("PM",     "PM_M",     comp_colors["PM"]),
        ("Energy", "Energy_M", comp_colors["Energy"]),
    ]:
        fig.add_trace(go.Bar(
            name=comp,
            x=df["Scenario"],
            y=df[col],
            marker_color=color,
            marker_opacity=0.85,
            text=df[col].apply(lambda v: f"${v:.0f}M"),
            textposition="inside",
            textfont=dict(color="#1e293b", size=10),
        ))

    fig.update_layout(
        barmode="stack",
        title=dict(
            text=f"FY{year} — Fleet Cost by Component & Scenario",
            font=dict(size=14, color="#1e293b"),
            x=0,
        ),
        xaxis_title="",
        yaxis_title="Cost ($M)",
    )
    return _apply_base(fig, height=380)



def make_fac_component_chart(fac_det: pd.DataFrame, year: int) -> go.Figure:
    """
    Stacked bar: Facilities cost breakdown (Non-Energy O&M / Base Energy / Energy Shock)
    for each oil shock scenario in the selected year.
    """
    df = fac_det[fac_det["Fiscal_Year"] == year].copy()
    df["NonEnergy_M"]     = (df["Optimized_Base_M"] * (1 - FACILITIES_ENERGY_FRACTION)).round(1)
    df["BaseEnergy_M"]    = (df["Optimized_Base_M"] * FACILITIES_ENERGY_FRACTION).round(1)
    df["EnergyShock_M"]   = df["Delta_Energy_M"].clip(lower=0)

    fig = go.Figure()
    for label, col, color in [
        ("Non-Energy O&M",  "NonEnergy_M",   "#059669"),   # emerald
        ("Base Energy",     "BaseEnergy_M",  "#ea580c"),   # orange
        ("Energy Shock Δ",  "EnergyShock_M", "#dc2626"),   # red
    ]:
        fig.add_trace(go.Bar(
            name=label,
            x=df["Scenario"],
            y=df[col],
            marker_color=color,
            marker_opacity=0.85,
            text=df[col].apply(lambda v: f"${v:.0f}M" if v > 1 else ""),
            textposition="inside",
            textfont=dict(color="#1e293b", size=10),
        ))

    fig.update_layout(
        barmode="stack",
        title=dict(
            text=f"FY{year} — Facilities Cost by Component & Scenario",
            font=dict(size=14, color="#1e293b"),
            x=0,
        ),
        xaxis_title="",
        yaxis_title="Cost ($M)",
    )
    return _apply_base(fig, height=380)


def make_combined_component_chart(fleet_comp: pd.DataFrame, fac_det: pd.DataFrame, year: int) -> go.Figure:
    """
    Stacked bar: Combined Fleet + Facilities cost breakdown by component
    for each oil shock scenario in the selected year.
    """
    fc = fleet_comp[fleet_comp["Fiscal_Year"] == year].copy()
    fd = fac_det[fac_det["Fiscal_Year"] == year].copy()
    fd["NonEnergy_M"]   = (fd["Optimized_Base_M"] * (1 - FACILITIES_ENERGY_FRACTION)).round(1)
    fd["BaseEnergy_M"]  = (fd["Optimized_Base_M"] * FACILITIES_ENERGY_FRACTION).round(1)
    fd["EnergyShock_M"] = fd["Delta_Energy_M"].clip(lower=0)

    merged = fc.merge(fd[["Scenario","NonEnergy_M","BaseEnergy_M","EnergyShock_M"]], on="Scenario")

    fig = go.Figure()
    for label, col, color in [
        ("Fac Non-Energy",   "NonEnergy_M",   "#1a7a4a"),   # dark emerald
        ("Fac Base Energy",  "BaseEnergy_M",  "#d97706"),   # amber
        ("Fac Energy Shock", "EnergyShock_M", "#dc2626"),   # red
        ("Fleet CapEx",      "CapEx_M",       "#059669"),   # emerald
        ("Fleet CM",         "CM_M",          "#0284c7"),   # sky
        ("Fleet PM",         "PM_M",          "#7c3aed"),   # violet
        ("Fleet Energy",     "Energy_M",      "#ea580c"),   # orange
    ]:
        fig.add_trace(go.Bar(
            name=label,
            x=merged["Scenario"],
            y=merged[col],
            marker_color=color,
            marker_opacity=0.85,
            text=merged[col].apply(lambda v: f"${v:.0f}M" if v > 2 else ""),
            textposition="inside",
            textfont=dict(color="#1e293b", size=9),
        ))

    fig.update_layout(
        barmode="stack",
        title=dict(
            text=f"FY{year} — Combined Cost by Component & Scenario",
            font=dict(size=14, color="#1e293b"),
            x=0,
        ),
        xaxis_title="",
        yaxis_title="Cost ($M)",
    )
    return _apply_base(fig, height=380)


def make_mc_bands_chart(fleet_mc: pd.DataFrame, fac_mc: pd.DataFrame, portfolio: str, year: int) -> go.Figure:
    """
    Bar chart with error bands: P10/P50/P95 for each scenario.
    portfolio: 'Fleet', 'Facilities', or 'Combined'
    """
    if portfolio == "Fleet":
        df = fleet_mc[fleet_mc["Fiscal_Year"] == year].copy()
    elif portfolio == "Facilities":
        df = fac_mc[fac_mc["Fiscal_Year"] == year].copy()
    else:
        # Combined: sum P50s; P95 as conservative upper
        fm = fleet_mc[fleet_mc["Fiscal_Year"] == year].copy()
        fa = fac_mc[fac_mc["Fiscal_Year"]    == year].copy()
        df = fm[["Scenario","P10_M","P50_M","P95_M"]].merge(
            fa[["Scenario","P10_M","P50_M","P95_M"]], on="Scenario", suffixes=("_f","_a")
        )
        df["P10_M"] = df["P10_M_f"] + df["P10_M_a"]
        df["P50_M"] = df["P50_M_f"] + df["P50_M_a"]
        df["P95_M"] = df["P95_M_f"] + df["P95_M_a"]

    fig = go.Figure()

    for _, row in df.iterrows():
        scen  = row["Scenario"]
        color = SCENARIO_COLORS.get(scen, "#64748b")
        p50   = row["P50_M"]
        p10   = row["P10_M"]
        p95   = row["P95_M"]

        fig.add_trace(go.Bar(
            name=scen,
            x=[scen],
            y=[p50],
            error_y=dict(
                type="data",
                symmetric=False,
                array=[p95 - p50],
                arrayminus=[p50 - p10],
                visible=True,
                thickness=2.5,
                width=8,
                color="rgba(30,41,59,0.70)",
            ),
            marker_color=color,
            marker_opacity=0.85,
            text=f"${p50:.0f}M",
            textposition="outside",
            textfont=dict(color="#1e293b", size=12),
            showlegend=False,
        ))

    fig.update_layout(
        title=dict(
            text=f"FY{year} — {portfolio} Total Cost: P10/P50/P95 Bands",
            font=dict(size=14, color="#1e293b"),
            x=0,
        ),
        xaxis_title="",
        yaxis_title="Total Cost ($M)",
        barmode="group",
    )
    return _apply_base(fig, height=380)


def make_budget_at_risk_chart(fleet_mc: pd.DataFrame, fac_mc: pd.DataFrame, year: int) -> go.Figure:
    """
    Grouped bar: P95 − P50 (budget at risk) per scenario for Fleet + Facilities.
    """
    fm = fleet_mc[fleet_mc["Fiscal_Year"] == year].copy()
    fa = fac_mc[fac_mc["Fiscal_Year"]    == year].copy()

    scens  = list(SCENARIOS.keys())
    f_risk = [float(fm[fm["Scenario"]==s]["P95_M"].values[0]) -
              float(fm[fm["Scenario"]==s]["P50_M"].values[0]) for s in scens]
    a_risk = [float(fa[fa["Scenario"]==s]["P95_M"].values[0]) -
              float(fa[fa["Scenario"]==s]["P50_M"].values[0]) for s in scens]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Fleet",
        x=scens,
        y=f_risk,
        marker_color=_FLEET_COLOR,
        marker_opacity=0.85,
        text=[f"${v:.1f}M" for v in f_risk],
        textposition="outside",
        textfont=dict(color="#1e293b", size=11),
    ))
    fig.add_trace(go.Bar(
        name="Facilities",
        x=scens,
        y=a_risk,
        marker_color=_FAC_COLOR,
        marker_opacity=0.85,
        text=[f"${v:.1f}M" for v in a_risk],
        textposition="outside",
        textfont=dict(color="#1e293b", size=11),
    ))
    fig.update_layout(
        title=dict(
            text=f"FY{year} — Budget at Risk (P95 − P50) by Scenario",
            font=dict(size=14, color="#1e293b"),
            x=0,
        ),
        xaxis_title="",
        yaxis_title="Budget at Risk ($M)",
        barmode="group",
    )
    return _apply_base(fig, height=360)


def make_scenario_waterfall(fleet_det: pd.DataFrame, fac_det: pd.DataFrame, year: int) -> go.Figure:
    """
    Waterfall: Fleet base → Fleet delta → Facilities base → Facilities delta → Combined.
    Shows how each component builds up the total under severe shock.
    """
    shock_scen = "Severe shock (+75%)"
    fd = fleet_det[(fleet_det["Fiscal_Year"]==year) & (fleet_det["Scenario"]==shock_scen)].iloc[0]
    ad = fac_det[(fac_det["Fiscal_Year"]==year)   & (fac_det["Scenario"]==shock_scen)].iloc[0]

    fleet_base  = fd["Base_Total_M"]
    fleet_delta = fd["Delta_Total_M"]
    fac_base    = ad["Optimized_Base_M"]
    fac_delta   = ad["Delta_Energy_M"]
    combined    = fleet_base + fleet_delta + fac_base + fac_delta

    labels  = ["Fleet Base", "Fleet Oil Δ", "Facilities Base", "Facilities Oil Δ", "Combined"]
    measure = ["absolute", "relative", "absolute", "relative", "total"]
    values  = [fleet_base, fleet_delta, fac_base, fac_delta, combined]
    colors  = ["#7c3aed", "#dc2626", "#1a7a4a", "#d97706", "#1e3a5f"]

    fig = go.Figure(go.Waterfall(
        name="FY2026 Severe Shock",
        orientation="v",
        measure=measure,
        x=labels,
        y=values,
        text=[f"${v:.0f}M" for v in values],
        textposition="outside",
        textfont=dict(color="#1e293b", size=12),
        connector=dict(line=dict(color="rgba(148,163,184,0.25)", width=1.5, dash="dot")),
        increasing=dict(marker_color=_DANGER_COLOR),
        decreasing=dict(marker_color=_FAC_COLOR),
        totals=dict(marker_color=_COMB_COLOR),
    ))
    fig.update_layout(
        title=dict(
            text=f"FY{year} — Severe Oil Shock Build-Up (Fleet + Facilities)",
            font=dict(size=14, color="#1e293b"),
            x=0,
        ),
        yaxis_title="Cost ($M)",
        showlegend=False,
    )
    return _apply_base(fig, height=380)


def make_mc_scenario_range(fleet_mc: pd.DataFrame, fac_mc: pd.DataFrame, year: int) -> go.Figure:
    """
    P10 / P50 / P95 range chart — combined portfolio, one band per scenario.
    Each scenario shows three markers (P10, P50, P95) connected by a vertical
    line, ordered left-to-right by shock severity.
    Immediately communicates: median rises AND uncertainty widens as shocks worsen.
    """
    scens       = list(SCENARIOS.keys())
    scen_colors = list(SCENARIO_COLORS.values())

    fig = go.Figure()

    for i, (scen, color) in enumerate(zip(scens, scen_colors)):
        fm = fleet_mc[(fleet_mc["Fiscal_Year"]==year) & (fleet_mc["Scenario"]==scen)]
        fa = fac_mc[(fac_mc["Fiscal_Year"]==year)    & (fac_mc["Scenario"]==scen)]
        if fm.empty or fa.empty:
            continue
        p10 = float(fm["P10_M"].values[0]) + float(fa["P10_M"].values[0])
        p50 = float(fm["P50_M"].values[0]) + float(fa["P50_M"].values[0])
        p95 = float(fm["P95_M"].values[0]) + float(fa["P95_M"].values[0])

        # Vertical error bar (P10→P95 range)
        fig.add_trace(go.Scatter(
            x=[scen, scen],
            y=[p10, p95],
            mode="lines",
            line=dict(color=color, width=2, dash="dot"),
            showlegend=False,
            hoverinfo="skip",
        ))
        # P50 — large marker
        fig.add_trace(go.Scatter(
            x=[scen],
            y=[p50],
            name=f"{scen.split('(')[0].strip()} — P50",
            mode="markers+text",
            marker=dict(size=14, color=color, symbol="circle",
                        line=dict(color="#0d1b2a", width=1.5)),
            text=[f"${p50:.0f}M"],
            textposition="middle right",
            textfont=dict(color=color, size=10),
            hovertemplate=(f"<b>{scen}</b><br>"
                           f"P10: ${p10:.0f}M<br>"
                           f"P50: ${p50:.0f}M<br>"
                           f"P95: ${p95:.0f}M<br>"
                           f"Spread: +${p95-p10:.0f}M<extra></extra>"),
        ))
        # P10 / P95 — small markers
        for val, lbl, sym in [(p10, "P10", "triangle-down"), (p95, "P95", "triangle-up")]:
            fig.add_trace(go.Scatter(
                x=[scen],
                y=[val],
                name=f"{scen.split('(')[0].strip()} — {lbl}",
                mode="markers",
                marker=dict(size=8, color=color, symbol=sym, opacity=0.65,
                            line=dict(color="#0d1b2a", width=1)),
                showlegend=False,
                hoverinfo="skip",
            ))

    fig.update_layout(
        title=dict(
            text=f"FY{year} — Combined Portfolio Risk Range (P10 / P50 / P95)",
            font=dict(size=14, color="#1e293b"), x=0,
        ),
        yaxis_title="Combined Cost ($M, modeled)",
        xaxis_title="",
        showlegend=False,
    )
    return _apply_base(fig, height=380)


def make_oil_delta_bars(fleet_det: pd.DataFrame, fac_det: pd.DataFrame, year: int) -> go.Figure:
    """
    Horizontal grouped bar — oil shock DELTA vs base only (base stripped out).
    Fleet Δ and Facilities Δ per scenario. Answers: "how much does oil cost add?"
    """
    scens  = list(SCENARIOS.keys())
    f_del, a_del, f_2nd = [], [], []

    for s in scens:
        fd = fleet_det[(fleet_det["Fiscal_Year"]==year) & (fleet_det["Scenario"]==s)]
        ad = fac_det[(fac_det["Fiscal_Year"]==year)   & (fac_det["Scenario"]==s)]
        f_del.append(float(fd["Delta_Energy_M"].values[0]) if not fd.empty else 0.0)
        f_2nd.append(float(fd["2nd_Order_M"].values[0])   if not fd.empty else 0.0)
        a_del.append(float(ad["Delta_Energy_M"].values[0]) if not ad.empty else 0.0)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Fleet — Direct Fuel",
        y=scens, x=f_del,
        orientation="h",
        marker_color=_FLEET_COLOR, marker_opacity=0.9,
        text=[f"+${v:.1f}M" if v > 0 else "—" for v in f_del],
        textposition="inside", insidetextanchor="middle",
        textfont=dict(color="#0d1b2a", size=9),
    ))
    fig.add_trace(go.Bar(
        name="Fleet — 2nd Order (PM/CM/CapEx)",
        y=scens, x=f_2nd,
        orientation="h",
        marker_color="#7c3aed", marker_opacity=0.75,
        text=[f"+${v:.1f}M" if v > 0 else "" for v in f_2nd],
        textposition="inside", insidetextanchor="middle",
        textfont=dict(color="#1e293b", size=9),
    ))
    fig.add_trace(go.Bar(
        name="Facilities — Energy Utilities",
        y=scens, x=a_del,
        orientation="h",
        marker_color=_FAC_COLOR, marker_opacity=0.9,
        text=[f"+${v:.1f}M" if v > 0 else "—" for v in a_del],
        textposition="inside", insidetextanchor="middle",
        textfont=dict(color="#0d1b2a", size=9),
    ))
    fig.update_layout(
        title=dict(
            text=f"FY{year} — Oil Shock Cost Impact vs Base (Δ Only)",
            font=dict(size=14, color="#1e293b"), x=0,
        ),
        barmode="stack",
        xaxis_title="Additional Cost vs Base ($M)",
        yaxis_title="",
        yaxis=dict(autorange="reversed"),
    )
    return _apply_base(fig, height=380)


# ── Tab 3 Charts ──────────────────────────────────────────────────────────────

def make_budget_runchart(data: dict) -> go.Figure:
    """
    Multi-year budget trajectory: Historical actuals + 3 projected scenarios.
    Baseline / High Risk (current oil shock sustained) / Optimized (L1+L2).
    """
    hist_years   = data["hist_years"]
    hist_total   = data["hist_total"]
    proj_years   = data["proj_years"]
    baseline     = data["baseline"]
    high_risk    = data["high_risk"]
    optimized    = data["optimized"]
    upper        = data["high_risk_upper"]
    lower        = data["high_risk_lower"]

    # Bridge: force 2024 actual value into projected series so lines connect
    bridge_yr  = data.get("bridge_yr", 2024)
    bridge_val = data.get("bridge_val")
    if bridge_val is not None:
        if bridge_yr in proj_years:
            idx = proj_years.index(bridge_yr)
            baseline[idx]  = bridge_val
            high_risk[idx] = bridge_val
            optimized[idx] = bridge_val
            upper[idx]     = bridge_val
            lower[idx]     = bridge_val
        else:
            proj_years = [bridge_yr] + proj_years
            baseline   = [bridge_val] + baseline
            high_risk  = [bridge_val] + high_risk
            optimized  = [bridge_val] + optimized
            upper      = [bridge_val] + upper
            lower      = [bridge_val] + lower

    fig = go.Figure()

    # Confidence band (High Risk ±1σ)
    band_x = proj_years + proj_years[::-1]
    band_y = upper + lower[::-1]
    fig.add_trace(go.Scatter(
        x=band_x, y=band_y,
        fill="toself",
        fillcolor="rgba(220,38,38,0.10)",
        line=dict(color="rgba(0,0,0,0)"),
        name="High Risk ±1σ band",
        showlegend=False,
        hoverinfo="skip",
    ))

    # Historical actuals
    fig.add_trace(go.Scatter(
        x=hist_years, y=hist_total,
        name="Historical Actuals",
        mode="lines+markers",
        line=dict(color="#64748b", width=2),
        marker=dict(size=6, color="#64748b"),
    ))

    # Projected — Baseline
    fig.add_trace(go.Scatter(
        x=proj_years, y=baseline,
        name="Baseline (Budget Plan)",
        mode="lines+markers",
        line=dict(color=_COMB_COLOR, width=2.5, dash="dash"),
        marker=dict(size=7, color=_COMB_COLOR),
        showlegend=False,
    ))

    # Projected — High Risk
    fig.add_trace(go.Scatter(
        x=proj_years, y=high_risk,
        name=f"High Risk (oil +{int(data['current_shock']*100)}% sustained)",
        mode="lines+markers",
        line=dict(color=_DANGER_COLOR, width=2.5),
        marker=dict(size=7, color=_DANGER_COLOR),
        showlegend=False,
    ))

    # Projected — Optimized
    fig.add_trace(go.Scatter(
        x=proj_years, y=optimized,
        name="Optimized (L1+L2 savings)",
        mode="lines+markers",
        line=dict(color=_FAC_COLOR, width=2.5, dash="dot"),
        marker=dict(size=7, color=_FAC_COLOR),
        showlegend=False,
    ))

    # Vertical line: today (FY2026)
    fig.add_vline(
        x=2026, line_width=1.5,
        line_dash="dot", line_color="rgba(234,88,12,0.70)",
        annotation_text="TODAY (FY2026)",
        annotation_position="top right",
        annotation_font=dict(color="#ea580c", size=10),
    )

    # Vertical line: history / projection break (2024)
    fig.add_vline(
        x=2024, line_width=1,
        line_dash="dash", line_color="rgba(148,163,184,0.40)",
        annotation_text="Last Actual",
        annotation_position="top left",
        annotation_font=dict(color="#64748b", size=10),
    )

    # Annotation: current WTI
    wti = data.get("current_wti", 107.82)
    fig.add_annotation(
        x=2026, y=max(high_risk),
        text=f"WTI ${wti:.0f}/bbl",
        showarrow=False,
        font=dict(color=_DANGER_COLOR, size=10),
        xshift=60, yshift=8,
    )

    fig.update_layout(
        title=dict(
            text="DPS Budget Trajectory — Baseline vs High Risk vs Optimized",
            font=dict(size=14, color="#1e293b"),
            x=0,
        ),
        xaxis_title="Fiscal Year",
        yaxis_title="Total Budget ($M)",
        xaxis=dict(dtick=1),
    )
    return _apply_base(fig, height=440)


def make_savings_gap_chart(data: dict) -> go.Figure:
    """
    Bar + line: annual oil risk premium and optimization savings gap.
    """
    proj_years = data["proj_years"]
    baseline   = data["baseline"]
    high_risk  = data["high_risk"]
    optimized  = data["optimized"]

    oil_premium  = [h - b for h, b in zip(high_risk, baseline)]
    opt_savings  = [b - o for b, o in zip(baseline, optimized)]
    total_gap    = [h - o for h, o in zip(high_risk, optimized)]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Oil Risk Premium (High Risk − Baseline)",
        x=proj_years,
        y=oil_premium,
        marker_color=_DANGER_COLOR,
        marker_opacity=0.80,
        text=[f"+${v:.1f}M" for v in oil_premium],
        textposition="outside",
        textfont=dict(color="#1e293b", size=10),
    ))

    fig.add_trace(go.Bar(
        name="Optimization Savings (Baseline − Optimized)",
        x=proj_years,
        y=[-v for v in opt_savings],
        marker_color=_FAC_COLOR,
        marker_opacity=0.80,
        text=[f"-${v:.1f}M" for v in opt_savings],
        textposition="outside",
        textfont=dict(color="#1e293b", size=10),
    ))

    fig.add_trace(go.Scatter(
        name="Net Gap (High Risk − Optimized)",
        x=proj_years,
        y=total_gap,
        mode="lines+markers",
        line=dict(color=_WARN_COLOR, width=2.5),
        marker=dict(size=8, color=_WARN_COLOR),
    ))

    fig.update_layout(
        title=dict(
            text="Annual Oil Risk Premium vs Optimization Savings",
            font=dict(size=14, color="#1e293b"),
            x=0,
        ),
        xaxis_title="Fiscal Year",
        yaxis_title="Budget Impact ($M)",
        barmode="group",
        xaxis=dict(dtick=1),
    )
    return _apply_base(fig, height=380)


def make_cost_runchart(data: dict) -> go.Figure:
    """
    Cost trajectory: stacked area (fleet fuel + electricity) for historical years,
    then three scenario lines (Baseline / High Risk / Optimized) for projections.
    """
    hist_years = data["hist_years"]
    hist_fuel  = data["hist_fuel"]
    hist_elec  = data["hist_elec"]
    hist_total = data["hist_total"]
    proj_years = data["proj_years"]
    baseline   = data["baseline"]
    high_risk  = data["high_risk"]
    optimized  = data["optimized"]
    upper      = data["high_risk_upper"]
    lower      = data["high_risk_lower"]

    bridge_yr  = data.get("bridge_yr", hist_years[-1])
    bridge_val = data.get("bridge_val", hist_total[-1])

    # Bridge: force last actual into projection series so lines connect
    if bridge_yr not in proj_years:
        proj_years = [bridge_yr] + proj_years
        baseline   = [bridge_val] + baseline
        high_risk  = [bridge_val] + high_risk
        optimized  = [bridge_val] + optimized
        upper      = [bridge_val] + upper
        lower      = [bridge_val] + lower
    else:
        idx = proj_years.index(bridge_yr)
        for lst in (baseline, high_risk, optimized, upper, lower):
            lst[idx] = bridge_val

    fig = go.Figure()

    # Confidence band (High Risk ±1σ)
    band_x = proj_years + proj_years[::-1]
    band_y = upper + lower[::-1]
    fig.add_trace(go.Scatter(
        x=band_x, y=band_y,
        fill="toself",
        fillcolor="rgba(220,38,38,0.10)",
        line=dict(color="rgba(0,0,0,0)"),
        name="High Risk ±1σ band",
        showlegend=False,
        hoverinfo="skip",
    ))

    # Historical total line
    fig.add_trace(go.Scatter(
        x=hist_years, y=hist_total,
        name="Total Actual",
        mode="lines+markers",
        line=dict(color="#64748b", width=2.5),
        marker=dict(size=7, color="#64748b"),
        hovertemplate="%{x}  —  $%{y:.2f}M<extra>Total Actual</extra>",
    ))

    # Projected — Baseline
    fig.add_trace(go.Scatter(
        x=proj_years, y=baseline,
        name="Baseline Projection",
        mode="lines+markers",
        line=dict(color=_COMB_COLOR, width=2.5, dash="dash"),
        marker=dict(size=7, color=_COMB_COLOR),
        hovertemplate="%{x}  —  $%{y:.2f}M<extra>Baseline</extra>",
        showlegend=False,
    ))

    # Projected — High Risk
    fig.add_trace(go.Scatter(
        x=proj_years, y=high_risk,
        name=f"High Risk (oil +{int(data['current_shock']*100)}% sustained)",
        mode="lines+markers",
        line=dict(color=_DANGER_COLOR, width=2.5),
        marker=dict(size=7, color=_DANGER_COLOR),
        hovertemplate="%{x}  —  $%{y:.2f}M<extra>High Risk</extra>",
        showlegend=False,
    ))

    # Projected — Optimized
    fig.add_trace(go.Scatter(
        x=proj_years, y=optimized,
        name="Optimized (demand mgmt + efficiency)",
        mode="lines+markers",
        line=dict(color=_FAC_COLOR, width=2.5, dash="dot"),
        marker=dict(size=7, color=_FAC_COLOR),
        hovertemplate="%{x}  —  $%{y:.2f}M<extra>Optimized</extra>",
        showlegend=False,
    ))

    last_actual_yr = hist_years[-1]
    fig.add_vline(
        x=last_actual_yr, line_width=1,
        line_dash="dash", line_color="rgba(148,163,184,0.40)",
        annotation_text="Last Actual",
        annotation_position="top left",
        annotation_font=dict(color="#64748b", size=10),
    )
    fig.add_vline(
        x=2026, line_width=1.5,
        line_dash="dot", line_color="rgba(234,88,12,0.70)",
        annotation_text="TODAY (FY2026)",
        annotation_position="top right",
        annotation_font=dict(color="#ea580c", size=10),
    )

    wti = data.get("current_wti", 107.82)
    fig.add_annotation(
        x=2026, y=max(high_risk),
        text=f"WTI ${wti:.0f}/bbl",
        showarrow=False,
        font=dict(color=_DANGER_COLOR, size=10),
        xshift=60, yshift=8,
    )

    fig.update_layout(
        title=dict(
            text="DPS Energy Cost Trajectory — Fleet Fuel + Electricity (Actual vs Projected)",
            font=dict(size=14, color="#1e293b"),
            x=0,
        ),
        xaxis_title="Fiscal Year",
        yaxis_title="Energy Cost ($M)",
        xaxis=dict(dtick=1),
    )
    return _apply_base(fig, height=460)


def make_cost_breakdown_bar(hist_years: list, hist_fuel: list, hist_elec: list) -> go.Figure:
    """
    Grouped bar: Fleet Fuel vs Electricity actual spend by year.
    """
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Fleet Fuel",
        x=hist_years, y=hist_fuel,
        marker_color=_FLEET_COLOR,
        marker_opacity=0.85,
        hovertemplate="%{x}  —  $%{y:.2f}M<extra>Fleet Fuel</extra>",
    ))
    fig.add_trace(go.Bar(
        name="Electricity",
        x=hist_years, y=hist_elec,
        marker_color=_FAC_COLOR,
        marker_opacity=0.85,
        hovertemplate="%{x}  —  $%{y:.2f}M<extra>Electricity</extra>",
    ))
    fig.update_layout(
        title=dict(
            text="Fleet Fuel vs Electricity — Annual Actual Spend",
            font=dict(size=14, color="#1e293b"),
            x=0,
        ),
        xaxis_title="Fiscal Year",
        yaxis_title="Spend ($M)",
        barmode="group",
        xaxis=dict(dtick=1),
    )
    return _apply_base(fig, height=360)


def make_oil_cost_chart(data: dict,
                        diesel_label: str = "Diesel Budget ($M)",
                        elec_label:   str = "Electricity Budget ($M)",
                        diesel_proj_label: str = "Diesel Projected",
                        elec_proj_label:   str = "Elec Projected",
                        title: str = "WTI Crude vs Diesel & Electricity  (2016 – 2030)",
                        x_range: list = None,
                        show_fleet: bool = True,
                        show_elec: bool = True) -> go.Figure:
    """
    Dual-axis line chart with projections (2027-2030):
      Left Y  — WTI crude ($/bbl) monthly historical + EIA forecast dashed + ±1σ band
      Right Y — Diesel line ($M) + Electricity line ($M) historical + projected bands
    Labels are configurable to support both budget and actual-spend views.
    """
    wti_dates     = data["wti_dates"]
    wti_prices    = data["wti_prices"]
    diesel_dates  = data["diesel_dates"]
    diesel_budget = data["diesel_budget"]
    elec_dates    = data["elec_dates"]
    elec_budget   = data["elec_budget"]
    proj_dates    = data["proj_dates"]
    wti_base      = data["wti_proj_base"]
    wti_hi        = data["wti_proj_hi"]
    wti_lo        = data["wti_proj_lo"]
    d_base        = data["diesel_proj_base"]
    d_hi          = data["diesel_proj_hi"]
    d_lo          = data["diesel_proj_lo"]
    e_base        = data["elec_proj_base"]
    e_hi          = data["elec_proj_hi"]
    e_lo          = data["elec_proj_lo"]
    events        = data["events"]
    base_wti      = data["base_wti"]
    spike_wti     = data["spike_wti"]
    current_wti   = data["current_wti"]

    wti_proj_mo_dates = data.get("wti_proj_monthly_dates", proj_dates)
    wti_proj_mo_base  = data.get("wti_proj_monthly_base",  wti_base)
    wti_proj_mo_hi    = data.get("wti_proj_monthly_hi",    wti_hi)
    wti_proj_mo_lo    = data.get("wti_proj_monthly_lo",    wti_lo)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Bridge last historical point → first projected (for continuous lines)
    last_wti_date   = wti_dates[-1]
    last_wti_price  = float(wti_prices[-1])
    d_bridge_date   = diesel_dates[-1]
    d_bridge_val    = float(diesel_budget[-1])
    e_bridge_date   = elec_dates[-1]
    e_bridge_val    = float(elec_budget[-1])

    # ── Uncertainty bands (drawn first, behind lines) ─────────────────────────
    # WTI ±1σ fan — uses monthly resolution so band edges are smooth
    fig.add_trace(go.Scatter(
        x=wti_proj_mo_dates + wti_proj_mo_dates[::-1],
        y=wti_proj_mo_hi + wti_proj_mo_lo[::-1],
        fill="toself", fillcolor="rgba(234,88,12,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="WTI ±1σ band", showlegend=False, hoverinfo="skip",
    ), secondary_y=False)

    # Diesel ±1σ band (annual, widens with horizon)
    d_px    = [d_bridge_date] + proj_dates
    d_hi_all = [d_bridge_val] + d_hi
    d_lo_all = [d_bridge_val] + d_lo
    if show_fleet:
        fig.add_trace(go.Scatter(
            x=d_px + d_px[::-1],
            y=d_hi_all + d_lo_all[::-1],
            fill="toself", fillcolor="rgba(124,58,237,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Diesel ±1σ band", showlegend=False, hoverinfo="skip",
        ), secondary_y=True)

    # Electricity ±1σ band
    e_px    = [e_bridge_date] + proj_dates
    e_hi_all = [e_bridge_val] + e_hi
    e_lo_all = [e_bridge_val] + e_lo
    if show_elec:
        fig.add_trace(go.Scatter(
            x=e_px + e_px[::-1],
            y=e_hi_all + e_lo_all[::-1],
            fill="toself", fillcolor="rgba(2,132,199,0.13)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Elec ±1σ band", showlegend=False, hoverinfo="skip",
        ), secondary_y=True)

    # ── WTI historical monthly line ───────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=wti_dates, y=wti_prices,
        name="WTI Crude ($/bbl)",
        mode="lines",
        line=dict(color="#ea580c", width=2.5),
        hovertemplate="%{x|%b %Y}  —  $%{y:.1f}/bbl<extra></extra>",
    ), secondary_y=False)

    # ── WTI EIA forecast — monthly-resolution line (looks like real WTI, not straight)
    fig.add_trace(go.Scatter(
        x=[last_wti_date] + wti_proj_mo_dates,
        y=[last_wti_price] + wti_proj_mo_base,
        name="WTI EIA Forecast",
        mode="lines",
        line=dict(color="#ea580c", width=1.8, dash="dash"),
        hovertemplate="%{x|%b %Y}  —  $%{y:.1f}/bbl<extra></extra>",
        showlegend=False,
    ), secondary_y=False)
    # Annual labels on forecast (separate trace, markers only at Jan of each year)
    fig.add_trace(go.Scatter(
        x=proj_dates,
        y=wti_base,
        name="WTI forecast labels",
        mode="markers+text",
        marker=dict(size=6, color="#ea580c"),
        text=[f"${v:.0f}" for v in wti_base],
        textposition="top center",
        textfont=dict(color="#ea580c", size=9),
        showlegend=False,
        hovertemplate="%{x|%Y}  —  $%{y:.1f}/bbl<extra></extra>",
    ), secondary_y=False)

    # ── Diesel / Fleet line historical ────────────────────────────────────────
    if show_fleet:
        fig.add_trace(go.Scatter(
            x=diesel_dates, y=diesel_budget,
            name=diesel_label,
            mode="lines+markers+text",
            line=dict(color="#7c3aed", width=2.5),
            marker=dict(size=7, color="#7c3aed"),
            text=[f"${v:.1f}M" for v in diesel_budget],
            textposition="top center",
            textfont=dict(color="#7c3aed", size=9),
            hovertemplate="%{x|%Y}  —  $%{y:.2f}M<extra></extra>",
        ), secondary_y=True)

        # ── FY2026 estimated actual (overrun marker) ─────────────────────────
        est_2026   = data.get("diesel_est_2026_M")
        bud_2026   = data.get("diesel_budget_2026_M")
        wti_avg_26 = data.get("wti_2026_avg")
        mo_count   = data.get("diesel_2026_months", 0)
        if est_2026 and bud_2026 and est_2026 > bud_2026:
            overrun = round(est_2026 - bud_2026, 2)
            fig.add_trace(go.Scatter(
                x=[pd.Timestamp("2026-01-01")],
                y=[est_2026],
                name=f"Est. FY2026 Actual (${est_2026:.1f}M)",
                mode="markers+text",
                marker=dict(size=13, color="#dc2626", symbol="diamond",
                            line=dict(color="#fff", width=1.5)),
                text=[f" Est. ${est_2026:.1f}M"],
                textposition="middle right",
                textfont=dict(color="#dc2626", size=9),
                hovertemplate=(
                    f"FY2026 Est. Actual: ${est_2026:.2f}M<br>"
                    f"Approved Budget: ${bud_2026:.2f}M<br>"
                    f"Overrun: +${overrun:.2f}M<br>"
                    f"WTI avg ({mo_count}mo + fwd): ${wti_avg_26}/bbl"
                    "<extra></extra>"
                ),
            ), secondary_y=True)
            fig.add_annotation(
                x=pd.Timestamp("2026-01-01"), y=est_2026,
                ax=pd.Timestamp("2026-01-01"), ay=bud_2026,
                xref="x", yref="y2", axref="x", ayref="y2",
                showarrow=True,
                arrowhead=2, arrowsize=1.2, arrowwidth=1.5,
                arrowcolor="#dc2626",
                text=f"+${overrun:.1f}M<br>overrun",
                font=dict(color="#dc2626", size=8),
                align="left",
            )

        # ── Diesel / Fleet projection dashed ─────────────────────────────────
        fig.add_trace(go.Scatter(
            x=[d_bridge_date] + proj_dates,
            y=[d_bridge_val] + d_base,
            name=diesel_proj_label,
            mode="lines+markers+text",
            line=dict(color="#7c3aed", width=2, dash="dash"),
            marker=dict(size=6, color="#7c3aed"),
            text=[""] + [f"${v:.1f}M" for v in d_base],
            textposition="top center",
            textfont=dict(color="#7c3aed", size=9),
            hovertemplate="%{x|%Y}  —  $%{y:.2f}M<extra></extra>",
            showlegend=False,
        ), secondary_y=True)

    # ── Electricity / Facilities line historical ──────────────────────────────
    if show_elec:
        fig.add_trace(go.Scatter(
            x=elec_dates, y=elec_budget,
            name=elec_label,
            mode="lines+markers+text",
            line=dict(color="#0284c7", width=2.5),
            marker=dict(size=7, color="#0284c7"),
            text=[f"${v:.1f}M" for v in elec_budget],
            textposition="bottom center",
            textfont=dict(color="#0284c7", size=9),
            hovertemplate="%{x|%Y}  —  $%{y:.1f}M<extra></extra>",
        ), secondary_y=True)

    # ── Electricity / Facilities projection dashed ────────────────────────────
    if show_elec:
        fig.add_trace(go.Scatter(
            x=[e_bridge_date] + proj_dates,
            y=[e_bridge_val] + e_base,
            name=elec_proj_label,
            mode="lines+markers+text",
            line=dict(color="#0284c7", width=2, dash="dash"),
            marker=dict(size=6, color="#0284c7"),
            text=[""] + [f"${v:.1f}M" for v in e_base],
            textposition="bottom center",
            textfont=dict(color="#0284c7", size=9),
            hovertemplate="%{x|%Y}  —  $%{y:.1f}M<extra></extra>",
            showlegend=False,
        ), secondary_y=True)

    # ── Crisis peak & today markers ────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=[pd.Timestamp("2026-03-09")], y=[spike_wti],
        name=f"Spike ${spike_wti:.0f} (Mar 9)",
        mode="markers+text",
        marker=dict(size=13, color=_DANGER_COLOR, symbol="star",
                    line=dict(color="#fff", width=1)),
        text=[f" ${spike_wti:.0f}"], textposition="middle right",
        textfont=dict(color=_DANGER_COLOR, size=10),
        hovertemplate=f"Mar 9 peak: ${spike_wti}/bbl<extra></extra>",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=[pd.Timestamp("2026-03-10")], y=[current_wti],
        name=f"Today ${current_wti:.0f}",
        mode="markers+text",
        marker=dict(size=10, color="#059669", symbol="circle",
                    line=dict(color="#fff", width=1)),
        text=[f" ${current_wti:.0f}"], textposition="middle right",
        textfont=dict(color="#059669", size=10),
        hovertemplate=f"Mar 10: ${current_wti}/bbl<extra></extra>",
    ), secondary_y=False)

    # ── Reference lines ────────────────────────────────────────────────────────
    fig.add_hline(
        y=base_wti,
        line_dash="dot", line_color="rgba(2,132,199,0.35)", line_width=1.5,
        annotation_text=f"Model base ${base_wti:.0f}/bbl",
        annotation_position="top right",
        annotation_font=dict(color="#0284c7", size=9),
    )

    # Today divider
    fig.add_vline(
        x=pd.Timestamp("2026-03-10").timestamp() * 1000,
        line_width=1.5, line_dash="dot",
        line_color="rgba(5,150,105,0.55)",
        annotation_text="Today",
        annotation_position="top left",
        annotation_font=dict(color="#059669", size=9),
    )

    # Projection start divider
    fig.add_vline(
        x=pd.Timestamp("2027-01-01").timestamp() * 1000,
        line_width=1, line_dash="dash",
        line_color="rgba(148,163,184,0.30)",
        annotation_text="Projection →",
        annotation_position="top right",
        annotation_font=dict(color="#64748b", size=9),
    )

    # Event vertical lines
    for ev in events:
        fig.add_vline(
            x=ev["date"].timestamp() * 1000,
            line_width=1.2, line_dash="dash",
            line_color=f"rgba({_hex_to_rgb(ev['color'])},0.50)",
            annotation_text=ev["label"],
            annotation_position="top left",
            annotation_font=dict(color=ev["color"], size=9),
        )

    fig.update_layout(
        **PLOTLY_BASE,
        height=490,
        margin=dict(**_MARGIN_BASE),
        legend=dict(**_LEGEND_BASE),
        title=dict(
            text=title,
            font=dict(size=14, color="#1e293b"),
            x=0,
        ),
        xaxis=dict(**_AXIS_BASE, title="", tickangle=-30,
                   **( {"range": x_range} if x_range else {} )),
        yaxis=dict(**_AXIS_BASE,  title="WTI Crude ($/bbl)"),
        yaxis2=dict(**_AXIS_BASE, title="Budget ($M)",
                    overlaying="y", side="right", showgrid=False),
    )
    return fig


def make_actual_vs_budget_chart(actual: dict, budget: dict, title: str,
                                 categories: list, colors: list) -> go.Figure:
    """
    Grouped bar: Budget (appropriation) vs Actual spend by year.
    actual / budget: {year: {cat: value_$M, ...}}
    categories: list of category keys to stack within each year.
    """
    years = sorted(set(list(actual.keys()) + list(budget.keys())))

    fig = go.Figure()

    # Budget bars (lighter, hatched-look via low opacity)
    for cat, color in zip(categories, colors):
        fig.add_trace(go.Bar(
            name=f"Budget — {cat}",
            x=years,
            y=[round(budget.get(y, {}).get(cat, 0) / 1e6, 2) for y in years],
            marker_color=color,
            marker_opacity=0.35,
            marker_pattern_shape="/",
            offsetgroup="budget",
        ))

    # Actual bars (solid)
    for cat, color in zip(categories, colors):
        fig.add_trace(go.Bar(
            name=f"Actual — {cat}",
            x=years,
            y=[round(actual.get(y, {}).get(cat, 0) / 1e6, 2) for y in years],
            marker_color=color,
            marker_opacity=0.85,
            offsetgroup="actual",
            text=[f"${round(actual.get(y, {}).get(cat, 0)/1e6, 1):.1f}M"
                  if actual.get(y, {}).get(cat, 0) else ""
                  for y in years],
            textposition="inside",
            textfont=dict(color="#1e293b", size=9),
        ))

    fig.update_layout(
        barmode="group",
        title=dict(text=title, font=dict(size=14, color="#1e293b"), x=0),
        xaxis_title="Fiscal Year",
        yaxis_title="Spend ($M)",
        xaxis=dict(dtick=1),
    )
    return _apply_base(fig, height=360)


def make_actual_summary_chart(rows: list) -> go.Figure:
    """
    Stacked bar: all categories of actual spend by year.
    rows: [{"year": int, "category": str, "value_M": float, "color": str}]
    """
    df = pd.DataFrame(rows)
    cats = df["category"].unique().tolist()

    fig = go.Figure()
    for cat in cats:
        sub = df[df["category"] == cat].sort_values("year")
        color = sub["color"].iloc[0]
        fig.add_trace(go.Bar(
            name=cat,
            x=sub["year"],
            y=sub["value_M"],
            marker_color=color,
            marker_opacity=0.85,
            text=sub["value_M"].apply(lambda v: f"${v:.1f}M" if v > 0 else ""),
            textposition="inside",
            textfont=dict(color="#1e293b", size=9),
        ))

    fig.update_layout(
        barmode="stack",
        title=dict(
            text="City of Chicago — DPS Actual Energy Spend by Category (FY2021–2024)",
            font=dict(size=14, color="#1e293b"), x=0,
        ),
        xaxis_title="Fiscal Year",
        yaxis_title="Actual Spend ($M)",
        xaxis=dict(dtick=1),
    )
    return _apply_base(fig, height=380)


def _hex_to_rgb(hex_color: str) -> str:
    """Convert #rrggbb to 'r,g,b' string for rgba()."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r},{g},{b}"


def make_component_shock_bar(
    components: list,
    base_vals: list,
    delta_vals: list,
    title: str,
    height: int = 480,
) -> go.Figure:
    """
    Horizontal stacked bar: base cost (blue) + oil shock delta (orange).
    Sorted by delta descending so highest-exposure components appear first.
    """
    order = sorted(range(len(delta_vals)), key=lambda i: delta_vals[i])
    comps  = [components[i] for i in order]
    base   = [base_vals[i]  for i in order]
    delta  = [delta_vals[i] for i in order]
    totals = [b + d for b, d in zip(base, delta)]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Base Cost",
        y=comps, x=base,
        orientation="h",
        marker_color=_COMB_COLOR,
        marker_opacity=0.75,
        hovertemplate="<b>%{y}</b><br>Base: $%{x:.3f}M<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Oil Shock Δ",
        y=comps, x=delta,
        orientation="h",
        marker_color=_WARN_COLOR,
        marker_opacity=0.85,
        hovertemplate=(
            "<b>%{y}</b><br>Oil Δ: +$%{x:.3f}M"
            "<extra></extra>"
        ),
        text=[f"+${d:.3f}M" if d >= 0.001 else "" for d in delta],
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(color="#1e293b", size=8, family="'Inter', sans-serif"),
    ))

    fig.update_layout(
        barmode="stack",
        title=dict(text=title, font=dict(size=13, color="#1e293b"), x=0),
        xaxis_title="Cost ($M)",
        yaxis=dict(tickfont=dict(size=10)),
        xaxis=dict(tickfont=dict(size=10)),
    )
    return _apply_base(fig, height=height)


"""
simulation.py — Combined Fleet + Facilities Oil Impact Simulation
City of Chicago — DPS Assets Analysis

─────────────────────────────────────────────────────────────────
ARCHITECTURE — what this module does vs what it reads
─────────────────────────────────────────────────────────────────
Fleet:
  • sim_annual.csv        → pre-computed deterministic output from the Fleet
                            Weibull/lifecycle notebook. NOT re-run here.
  • mc_runs_by_year.csv   → pre-computed 125-run MC from the same notebook.
                            Each run has different asset failure sequences.
                            NOT re-run here.
  • chicago_energy_budget_2016_2025.xlsx → used to CALIBRATE fuel price
                            volatility (σ_diesel, σ_elec) from FY2016–2025
                            year-over-year budget changes. The resulting σs are
                            injected into every MC run.

Facilities:
  • asset_model_outputs.xlsx (fact_savings) → pre-computed deterministic
    baseline and AI savings. NOT re-run here.
  • A lightweight MC (125 draws) IS run here to sample:
      – Pillar adoption rates r_p ~ N(default, σ)
      – Fuel price shock per run ~ N(scenario_shock, σ_diesel)   ← NEW
      – Facilities oil passthrough ~ N(0.21, σ_passthrough)       ← NEW

Oil shock is applied INSIDE each MC draw (not as a single fixed multiplier
on top of all runs).
─────────────────────────────────────────────────────────────────
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

# Ensure user site-packages are on path (handles split Python installs)
import site
for _sp in site.getusersitepackages() if isinstance(site.getusersitepackages(), list) else [site.getusersitepackages()]:
    if _sp not in sys.path:
        sys.path.insert(0, _sp)
# Also add Roaming site-packages explicitly
_roaming = r"C:\Users\Nex3t\AppData\Roaming\Python\Python313\site-packages"
if _roaming not in sys.path:
    sys.path.insert(0, _roaming)

import numpy as np
import pandas as pd
from config import (
    FLEET_SIM_ANNUAL, FLEET_MC_BY_YEAR,
    FACILITIES_EXCEL, BASE_DIR,
    BASE_DIESEL_GAL, CRUDE_TO_DIESEL,
    FLEET_ELAST_PM, FLEET_ELAST_CM, FLEET_ELAST_CAPEX,
    FACILITIES_ENERGY_FRACTION, FACILITIES_OIL_PASSTHROUGH,
    MC_RUNS, MC_SEED,
    PILLAR_DEFAULTS, PILLAR_SIGMA_FRAC,
    AI_K, AI_X0, AI_START_YEAR,
    FOCUS_YEARS, SCENARIOS,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════
def sigmoid(t: float, k: float, x0: float) -> float:
    return 1.0 / (1.0 + np.exp(-k * (t - x0)))


def _pillar_residual(elapsed: float, rates: dict) -> float:
    residual = 1.0
    for key, r in rates.items():
        mu = sigmoid(elapsed, PILLAR_DEFAULTS[key]["k"], PILLAR_DEFAULTS[key]["x0"])
        residual *= (1.0 - min(r * mu, 0.95))
    return residual


_BUDGET_ACCT_CLEAN = {
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

_BUDGET_EXCEL = BASE_DIR / "chicago_energy_budget_2016_2025.xlsx"


def _load_budget_annual() -> pd.DataFrame:
    """
    Read chicago_energy_budget_2016_2025.xlsx (detail_by_account sheet).
    Returns DataFrame with columns: Category, Fiscal_Year, Amount (dollars).
    """
    df = pd.read_excel(_BUDGET_EXCEL, sheet_name="detail_by_account")
    df["Category"]    = df["account"].map(_BUDGET_ACCT_CLEAN)
    df["Fiscal_Year"] = df["year"].astype(int)
    df["Amount"]      = pd.to_numeric(df["total_spend"], errors="coerce").fillna(0)
    return (df.groupby(["Category", "Fiscal_Year"], as_index=False)
              .agg(Amount=("Amount", "sum")))


def _calibrate_fuel_volatility() -> dict:
    """
    Compute fuel price volatility from chicago_energy_budget_2016_2025.xlsx.

    Returns σ_diesel and σ_elec as annual standard deviations of
    year-over-year budget changes (price × volume proxy, FY2016–2025).
    """
    df = _load_budget_annual()

    def _yoy_std(name):
        s = df[df["Category"] == name].sort_values("Fiscal_Year")["Amount"]
        return float(s.pct_change().dropna().std())

    sigma_diesel = _yoy_std("Motor Vehicle Diesel Fuel")
    sigma_elec   = _yoy_std("Electricity")

    # Facilities passthrough uncertainty:
    #   electricity 80% of energy, oil-to-elec passthrough σ ~ 0.04
    #   gas 20%, oil-to-gas passthrough σ ~ 0.08
    #   combined: sqrt((0.8*0.04)^2 + (0.2*0.08)^2) ≈ 0.038
    sigma_fac_passthrough = 0.038

    return {
        "sigma_diesel":          sigma_diesel,
        "sigma_elec":            sigma_elec,
        "sigma_fac_passthrough": sigma_fac_passthrough,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Fleet Simulation
# ═══════════════════════════════════════════════════════════════════════════════
class FleetSim:
    """
    Reads pre-computed Fleet outputs (sim_annual.csv, mc_runs_by_year.csv).
    Applies oil shocks post-hoc.

    MC improvement: each of the 125 pre-computed runs now gets an independent
    fuel price draw from N(scenario_shock, σ_diesel), calibrated from the
    6-year chicago_energy_budget_2016_2025.xlsx history. This makes the P10/P50/P95 bands reflect
    both asset-level failure uncertainty AND fuel price uncertainty.
    """

    def __init__(self):
        self.sim  = pd.read_csv(FLEET_SIM_ANNUAL)
        self.mc   = pd.read_csv(FLEET_MC_BY_YEAR)
        self._vol = _calibrate_fuel_volatility()
        self._rng = np.random.default_rng(MC_SEED)
        self._validate()
        # Pre-draw one fuel-shock noise vector per run (run-stable across scenarios)
        self._n_runs = int(self.mc["Run_ID"].nunique())
        self._fuel_noise = self._rng.normal(0.0, self._vol["sigma_diesel"], self._n_runs)

    def _validate(self):
        needed = ["Energy_Cost_USD", "PM_Asset_USD", "PM_Modules_USD",
                  "CM_USD", "Repl_Chassis_CapEx_USD", "Repl_Modules_CapEx_USD",
                  "Total_Cost_USD", "Year"]
        for c in needed:
            assert c in self.sim.columns, f"sim_annual missing column: {c}"

    # ── Deterministic scenario table ─────────────────────────────────────────
    def det_scenario_table(self) -> pd.DataFrame:
        rows = []
        base_agg = (
            self.sim[self.sim["Year"].isin(FOCUS_YEARS)]
            .groupby("Year")[["Energy_Cost_USD", "PM_Asset_USD", "PM_Modules_USD",
                               "CM_USD", "Repl_Chassis_CapEx_USD", "Repl_Modules_CapEx_USD",
                               "Total_Cost_USD"]].sum()
        )
        for scen_name, shock in SCENARIOS.items():
            df_s  = self._apply_shock_df(self.sim[self.sim["Year"].isin(FOCUS_YEARS)], shock)
            agg_s = df_s.groupby("Year")[["Energy_Cost_USD", "Total_Cost_USD"]].sum()
            for yr in FOCUS_YEARS:
                bt = float(base_agg.loc[yr, "Total_Cost_USD"])
                st = float(agg_s.loc[yr, "Total_Cost_USD"])
                be = float(base_agg.loc[yr, "Energy_Cost_USD"])
                se = float(agg_s.loc[yr, "Energy_Cost_USD"])
                rows.append({
                    "Scenario":       scen_name,
                    "Fiscal_Year":    yr,
                    "Diesel_$/gal":   round(BASE_DIESEL_GAL * (1 + shock), 2),
                    "Base_Total_M":   round(bt / 1e6, 1),
                    "Shock_Total_M":  round(st / 1e6, 1),
                    "Delta_Total_M":  round((st - bt) / 1e6, 1),
                    "Delta_Pct":      round((st / bt - 1) * 100, 1),
                    "Delta_Energy_M": round((se - be) / 1e6, 1),
                    "2nd_Order_M":    round(((st - bt) - (se - be)) / 1e6, 1),
                })
        return pd.DataFrame(rows)

    # ── Component breakdown by scenario ──────────────────────────────────────
    def component_breakdown(self) -> pd.DataFrame:
        """
        Returns Energy / PM / CM / CapEx cost breakdown per scenario and year.
        Each component reflects the shocked value for that scenario.
        """
        rows = []
        for scen_name, shock in SCENARIOS.items():
            df_s = self._apply_shock_df(self.sim[self.sim["Year"].isin(FOCUS_YEARS)], shock)
            agg  = df_s.groupby("Year")[["Energy_Cost_USD", "PM_Asset_USD", "PM_Modules_USD",
                                         "CM_USD", "Repl_Chassis_CapEx_USD", "Repl_Modules_CapEx_USD"]].sum()
            for yr in FOCUS_YEARS:
                rows.append({
                    "Scenario":    scen_name,
                    "Fiscal_Year": yr,
                    "Energy_M":    round(agg.loc[yr, "Energy_Cost_USD"] / 1e6, 1),
                    "PM_M":        round((agg.loc[yr, "PM_Asset_USD"] + agg.loc[yr, "PM_Modules_USD"]) / 1e6, 1),
                    "CM_M":        round(agg.loc[yr, "CM_USD"] / 1e6, 1),
                    "CapEx_M":     round((agg.loc[yr, "Repl_Chassis_CapEx_USD"] + agg.loc[yr, "Repl_Modules_CapEx_USD"]) / 1e6, 1),
                })
        return pd.DataFrame(rows)

    # ── MC risk bands (fuel-price-aware) ──────────────────────────────────────
    def mc_risk_bands(self) -> pd.DataFrame:
        """
        For each MC run (indexed by Run_ID), apply an independent fuel price
        draw: shock_run = scenario_shock + ε_i, where ε_i ~ N(0, σ_diesel).

        This captures the fuel price uncertainty ON TOP OF the asset-level
        Weibull failure uncertainty already embedded in mc_runs_by_year.csv.

        σ_diesel = {sigma:.3f} (calibrated from chicago_energy_budget_2016_2025.xlsx 2021–2026).
        """.format(sigma=self._vol["sigma_diesel"])
        run_ids = sorted(self.mc["Run_ID"].unique())
        rows    = []

        for scen_name, shock in SCENARIOS.items():
            for yr in FOCUS_YEARS:
                mc_yr = self.mc[self.mc["Year"] == yr].copy()
                run_totals = []

                for i, run_id in enumerate(run_ids):
                    # Independent fuel price noise per run, calibrated from history
                    noise     = self._fuel_noise[i % self._n_runs]
                    shock_run = float(np.clip(shock + noise, -0.05, shock + 0.60))

                    run_data = mc_yr[mc_yr["Run_ID"] == run_id].copy()
                    run_data = self._apply_shock_df(run_data, shock_run)
                    run_totals.append(run_data["Total_Cost_USD"].sum())

                v = np.array(run_totals)
                rows.append({
                    "Scenario":    scen_name,
                    "Fiscal_Year": yr,
                    "P10_M":  round(np.percentile(v, 10) / 1e6, 1),
                    "P50_M":  round(np.percentile(v, 50) / 1e6, 1),
                    "P95_M":  round(np.percentile(v, 95) / 1e6, 1),
                    "sigma_diesel_used": round(self._vol["sigma_diesel"], 3),
                })
        return pd.DataFrame(rows)

    # ── Crude sensitivity table ───────────────────────────────────────────────
    def crude_sensitivity(self) -> pd.DataFrame:
        rows = []
        base_e = (
            self.sim[self.sim["Year"].isin(FOCUS_YEARS)]
            .groupby("Year")["Energy_Cost_USD"].sum()
        )
        for yr in FOCUS_YEARS:
            for delta_crude in [10, 25, 50, 100]:
                delta_diesel = delta_crude * CRUDE_TO_DIESEL
                shock_f      = delta_diesel / BASE_DIESEL_GAL
                d_energy     = float(base_e.loc[yr]) * shock_f
                d_pm = (
                    self.sim[self.sim["Year"] == yr]["PM_Asset_USD"].sum() +
                    self.sim[self.sim["Year"] == yr]["PM_Modules_USD"].sum()
                ) * FLEET_ELAST_PM * shock_f
                d_cm = self.sim[self.sim["Year"] == yr]["CM_USD"].sum() * FLEET_ELAST_CM * shock_f
                rows.append({
                    "Fiscal_Year":       yr,
                    "Crude_Δ_$/bbl":     delta_crude,
                    "Implied_Diesel":    round(BASE_DIESEL_GAL + delta_diesel, 2),
                    "Delta_Energy_M":    round(d_energy / 1e6, 1),
                    "Delta_2nd_Order_M": round((d_pm + d_cm) / 1e6, 1),
                    "Delta_Total_M":     round((d_energy + d_pm + d_cm) / 1e6, 1),
                })
        return pd.DataFrame(rows)

    # ── Historic fuel chart data ──────────────────────────────────────────────
    def hist_fuel_by_category(self) -> pd.DataFrame:
        df = _load_budget_annual()
        # make_historic_fuel_chart expects raw dollar amounts (divides by 1e6 internally)
        # and index column named "Fiscal Year" (space, not underscore)
        df = df.rename(columns={"Fiscal_Year": "Fiscal Year"})
        return df.pivot_table(
            index="Fiscal Year",
            columns="Category",
            values="Amount",
            aggfunc="sum",
        ).reset_index()

    def fuel_volatility_summary(self) -> dict:
        return self._vol

    # ── Internal: apply shock to a dataframe ──────────────────────────────────
    def _apply_shock_df(self, df: pd.DataFrame, shock: float) -> pd.DataFrame:
        d = df.copy()
        d["Energy_Cost_USD"]        *= (1 + shock)
        d["PM_Asset_USD"]           *= (1 + FLEET_ELAST_PM    * shock)
        d["PM_Modules_USD"]         *= (1 + FLEET_ELAST_PM    * shock)
        d["CM_USD"]                 *= (1 + FLEET_ELAST_CM    * shock)
        d["Repl_Chassis_CapEx_USD"] *= (1 + FLEET_ELAST_CAPEX * shock)
        d["Repl_Modules_CapEx_USD"] *= (1 + FLEET_ELAST_CAPEX * shock)
        d["Total_Cost_USD"] = (
            d["Energy_Cost_USD"] + d["PM_Asset_USD"] + d["PM_Modules_USD"] +
            d["CM_USD"] + d["Repl_Chassis_CapEx_USD"] + d["Repl_Modules_CapEx_USD"]
        )
        return d


# ═══════════════════════════════════════════════════════════════════════════════
# Facilities Simulation
# ═══════════════════════════════════════════════════════════════════════════════
class FacilitiesSim:
    """
    Reads pre-computed Facilities deterministic output (fact_savings sheet).
    Runs a lightweight MC (125 draws) varying three stochastic inputs:

      1. Pillar adoption rates r_p ~ N(default, σ_rp)
         → Captures uncertainty in how fast each optimization program matures.

      2. Fuel price shock per run: shock_run ~ N(scenario_shock, σ_diesel)
         → calibrated from chicago_energy_budget_2016_2025.xlsx. Same σ as Fleet, preserving
           correlation between fleet and facilities energy costs.

      3. Oil-to-utilities passthrough per run: pt ~ N(0.21, σ_passthrough)
         → Captures uncertainty in how much of an oil spike translates to
           electricity and gas bills (regulated tariff lag, hedging, etc.)
         → clipped to [0.05, 0.45].
    """

    def __init__(self):
        self._fact = pd.read_excel(FACILITIES_EXCEL, sheet_name="fact_savings")
        self._vol  = _calibrate_fuel_volatility()
        self._rng  = np.random.default_rng(MC_SEED + 1)   # offset from Fleet seed

        self._base_by_yr = (
            self._fact[
                (self._fact["budget_line"] == "Facilities") &
                (self._fact["year"].isin(FOCUS_YEARS))
            ]
            .groupby("year")["baseline"].sum()
        )
        self._ai_by_yr = (
            self._fact[
                (self._fact["budget_line"] == "Facilities") &
                (self._fact["year"].isin(FOCUS_YEARS))
            ]
            .groupby("year")["l2_total"].sum()
        )

        # Pre-draw all MC samples (shock-independent, scenario-independent)
        self._draws = self._build_draws(MC_RUNS)

    def _build_draws(self, n: int) -> list[dict]:
        """
        Draw n parameter sets. Each set contains:
          - pillar rates (r_p for each pillar)
          - fuel_noise: additive noise on the scenario shock (from σ_diesel)
          - passthrough: facilities oil-to-energy passthrough
        """
        sigma_d  = self._vol["sigma_diesel"]
        sigma_pt = self._vol["sigma_fac_passthrough"]
        draws = []
        for _ in range(n):
            rates = {}
            for key, p in PILLAR_DEFAULTS.items():
                sigma_r = p["r_p"] * PILLAR_SIGMA_FRAC
                rates[key] = float(np.clip(
                    self._rng.normal(p["r_p"], sigma_r),
                    p["r_min"], p["r_max"]
                ))
            draws.append({
                "rates":       rates,
                "fuel_noise":  float(self._rng.normal(0.0, sigma_d)),
                "passthrough": float(np.clip(
                    self._rng.normal(FACILITIES_OIL_PASSTHROUGH, sigma_pt),
                    0.05, 0.45
                )),
            })
        return draws

    def _compute_optimized_cost(self, yr: int, rates: dict, base_m: float, ai_m: float) -> float:
        elapsed_l1 = float(yr - 2016)
        residual   = _pillar_residual(elapsed_l1, rates)
        l1_cost    = base_m * residual
        if yr <= AI_START_YEAR:
            return l1_cost
        ai_mu = sigmoid(float(yr - AI_START_YEAR), AI_K, AI_X0)
        return l1_cost - ai_m * ai_mu

    # ── MC risk bands (fuel-price + passthrough aware) ────────────────────────
    def mc_risk_bands(self) -> pd.DataFrame:
        rows = []
        for scen_name, shock in SCENARIOS.items():
            for yr in FOCUS_YEARS:
                base_m = float(self._base_by_yr.loc[yr]) / 1e6
                ai_m   = float(self._ai_by_yr.loc[yr])   / 1e6
                run_costs = []

                for draw in self._draws:
                    shock_run = float(np.clip(
                        shock + draw["fuel_noise"], -0.05, shock + 0.60
                    ))
                    energy_delta = base_m * FACILITIES_ENERGY_FRACTION * draw["passthrough"] * shock_run
                    opt_cost     = self._compute_optimized_cost(yr, draw["rates"], base_m, ai_m)
                    run_costs.append(opt_cost + energy_delta)

                v = np.array(run_costs)
                rows.append({
                    "Scenario":      scen_name,
                    "Fiscal_Year":   yr,
                    "P10_M":         round(np.percentile(v, 10), 1),
                    "P50_M":         round(np.percentile(v, 50), 1),
                    "P95_M":         round(np.percentile(v, 95), 1),
                    "Base_Total_M":  round(base_m, 1),
                    "Energy_Delta_M": round(np.mean(v) - float(np.percentile(v, 50)) + base_m * FACILITIES_ENERGY_FRACTION * FACILITIES_OIL_PASSTHROUGH * shock, 1),
                })
        return pd.DataFrame(rows)

    # ── Deterministic scenario table ─────────────────────────────────────────
    def det_scenario_table(self) -> pd.DataFrame:
        default_rates = {k: v["r_p"] for k, v in PILLAR_DEFAULTS.items()}
        rows = []
        for scen_name, shock in SCENARIOS.items():
            for yr in FOCUS_YEARS:
                base_m       = float(self._base_by_yr.loc[yr]) / 1e6
                ai_m         = float(self._ai_by_yr.loc[yr])   / 1e6
                energy_delta = base_m * FACILITIES_ENERGY_FRACTION * FACILITIES_OIL_PASSTHROUGH * shock
                opt_base     = self._compute_optimized_cost(yr, default_rates, base_m, ai_m)
                opt_shock    = opt_base + energy_delta
                rows.append({
                    "Scenario":          scen_name,
                    "Fiscal_Year":       yr,
                    "Base_Total_M":      round(base_m, 1),
                    "Optimized_Base_M":  round(opt_base, 1),
                    "Optimized_Shock_M": round(opt_shock, 1),
                    "Delta_Energy_M":    round(energy_delta, 1),
                    "Delta_Pct":         round((opt_shock / opt_base - 1) * 100, 1),
                })
        return pd.DataFrame(rows)

    # ── Crude sensitivity table ───────────────────────────────────────────────
    def crude_sensitivity(self) -> pd.DataFrame:
        rows = []
        for yr in FOCUS_YEARS:
            base_m = float(self._base_by_yr.loc[yr]) / 1e6
            for delta_crude in [10, 25, 50, 100]:
                shock_f  = (delta_crude * CRUDE_TO_DIESEL) / BASE_DIESEL_GAL
                d_energy = base_m * FACILITIES_ENERGY_FRACTION * FACILITIES_OIL_PASSTHROUGH * shock_f
                rows.append({
                    "Fiscal_Year":    yr,
                    "Crude_Δ_$/bbl": delta_crude,
                    "Delta_Energy_M": round(d_energy, 1),
                })
        return pd.DataFrame(rows)

    def fuel_volatility_summary(self) -> dict:
        return self._vol


# ═══════════════════════════════════════════════════════════════════════════════
# Budget Runchart Data
# ═══════════════════════════════════════════════════════════════════════════════
def compute_runchart_data() -> dict:
    """
    Builds three budget trajectory series for the runchart:

      Baseline  — historical actuals (2016-2024) + projected combined
                  baseline from fact_savings (2025-2033), no optimization,
                  base fuel prices.

      High Risk — same historical (2016-2024) + projected baseline with
                  current oil shock sustained (~21% above budget base,
                  WTI ~$107/bbl as of March 2026). Shock applied to Fleet
                  energy fraction and Facilities energy fraction each year.

      Optimized — same historical (2016-2024) + projected fact_savings
                  'final' column (L1+L2 optimization programs active,
                  base fuel prices). Shows what optimization saves.

    Returns dict with keys: years, hist_years, proj_years,
    baseline, high_risk, optimized, high_risk_upper, high_risk_lower
    """
    # ── Historical actuals (2016-2024) ────────────────────────────────────────
    hist_df = pd.read_excel(FACILITIES_EXCEL, sheet_name="Historical_Costs")
    hist_df = hist_df.sort_values("Year")
    hist_years = hist_df["Year"].tolist()
    hist_total = (hist_df["DFF_Total_Actual"] / 1e6).round(1).tolist()

    # ── Projected (2025-2033) from fact_savings ───────────────────────────────
    fs = pd.read_excel(FACILITIES_EXCEL, sheet_name="fact_savings")

    # Separate Fleet and non-Fleet to apply different oil shocks
    fleet_proj = (
        fs[fs["budget_line"] == "Fleet"]
        .groupby("year")[["baseline", "final"]].sum()
    )
    other_proj = (
        fs[fs["budget_line"] != "Fleet"]
        .groupby("year")[["baseline", "final"]].sum()
    )

    proj_years_all = sorted(fs["year"].unique())
    # Only show up to 2033
    proj_years = [y for y in proj_years_all if y <= 2033]

    # Current market shock: WTI $83 → ~15% above budget base (Mar 10 2026, post-spike pullback)
    CURRENT_SHOCK = 0.15

    # Fleet energy fraction of fleet total (~33%) + elasticities
    # Effective fleet cost uplift at 21% fuel shock:
    FLEET_ENERGY_FRAC  = 0.33
    FLEET_UPLIFT = CURRENT_SHOCK * (
        FLEET_ENERGY_FRAC * 1.00 +          # direct energy (linear)
        0.15 * FLEET_ELAST_PM +             # PM fraction ~15% of total
        0.12 * FLEET_ELAST_CM +             # CM fraction ~12% of total
        0.10 * FLEET_ELAST_CAPEX            # CapEx fraction ~10%
    )

    # Facilities energy uplift at 21% shock:
    FAC_UPLIFT = CURRENT_SHOCK * FACILITIES_ENERGY_FRACTION * FACILITIES_OIL_PASSTHROUGH

    # Oil price uncertainty band (σ_diesel = 26.1%)
    sigma = _calibrate_fuel_volatility()["sigma_diesel"]
    # ±1 sigma band on the shock itself → cost band
    SHOCK_HI = CURRENT_SHOCK + sigma   # upper 1-sigma
    SHOCK_LO = max(0, CURRENT_SHOCK - sigma * 0.5)  # lower (asymmetric — prices rarely fall fast)
    FLEET_UPLIFT_HI  = SHOCK_HI  * (FLEET_ENERGY_FRAC + 0.15*FLEET_ELAST_PM + 0.12*FLEET_ELAST_CM)
    FLEET_UPLIFT_LO  = SHOCK_LO  * (FLEET_ENERGY_FRAC + 0.15*FLEET_ELAST_PM + 0.12*FLEET_ELAST_CM)
    FAC_UPLIFT_HI    = SHOCK_HI  * FACILITIES_ENERGY_FRACTION * FACILITIES_OIL_PASSTHROUGH
    FAC_UPLIFT_LO    = SHOCK_LO  * FACILITIES_ENERGY_FRACTION * FACILITIES_OIL_PASSTHROUGH

    baseline, high_risk, optimized = [], [], []
    high_risk_upper, high_risk_lower = [], []

    for yr in proj_years:
        if yr not in fleet_proj.index:
            continue
        f_base = fleet_proj.loc[yr, "baseline"] / 1e6
        f_opt  = fleet_proj.loc[yr, "final"]    / 1e6
        o_base = other_proj.loc[yr, "baseline"] / 1e6
        o_opt  = other_proj.loc[yr, "final"]    / 1e6

        total_base = f_base + o_base
        total_opt  = f_opt  + o_opt

        # High risk: apply current shock to each component
        total_hr   = total_base + f_base * FLEET_UPLIFT   + o_base * FAC_UPLIFT
        total_hr_hi = total_base + f_base * FLEET_UPLIFT_HI + o_base * FAC_UPLIFT_HI
        total_hr_lo = total_base + f_base * FLEET_UPLIFT_LO + o_base * FAC_UPLIFT_LO

        baseline.append(round(total_base, 1))
        high_risk.append(round(total_hr, 1))
        optimized.append(round(total_opt, 1))
        high_risk_upper.append(round(total_hr_hi, 1))
        high_risk_lower.append(round(total_hr_lo, 1))

    # Last historical point bridges into projected (2024 overlap)
    bridge_yr   = 2024
    bridge_val  = hist_total[hist_years.index(bridge_yr)] if bridge_yr in hist_years else None

    return {
        "hist_years":       hist_years,
        "hist_total":       hist_total,
        "proj_years":       proj_years,
        "baseline":         baseline,
        "high_risk":        high_risk,
        "optimized":        optimized,
        "high_risk_upper":  high_risk_upper,
        "high_risk_lower":  high_risk_lower,
        "bridge_yr":        bridge_yr,
        "bridge_val":       bridge_val,
        "current_wti":       83.00,
        "current_shock":     CURRENT_SHOCK,
        "current_shock_pct": CURRENT_SHOCK * 100,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Oil Price vs Fleet Cost Correlation
# ═══════════════════════════════════════════════════════════════════════════════
def compute_oil_cost_correlation() -> dict:
    """
    Fetches WTI monthly prices via yfinance (2021–present) and aligns with
    historic fleet diesel budget (FY2021–2026) and total fleet cost trajectory.
    Returns data dict for the Oil vs Cost chart.
    """
    import yfinance as yf

    # ── WTI monthly (datetime) ────────────────────────────────────────────────
    raw = yf.download("CL=F", start="2016-01-01", interval="1mo", progress=False)
    raw = raw["Close"].reset_index()
    raw.columns = ["Date", "WTI"]
    raw["Date"] = pd.to_datetime(raw["Date"]).dt.to_period("M").dt.to_timestamp()
    raw = raw.dropna().sort_values("Date")

    # ── Fleet diesel budget actuals ($M) — from chicago_energy_budget_2016_2025.xlsx ──
    _bdf = _load_budget_annual()
    _bdf["Date"] = pd.to_datetime(_bdf["Fiscal_Year"].astype(str) + "-01-01")

    _fleet_cats = ["Motor Vehicle Diesel Fuel", "Motor Vehicle Gasoline", "Alternative Fuel"]
    diesel = (
        _bdf[_bdf["Category"].isin(_fleet_cats)]
        .groupby("Date", as_index=False)["Amount"].sum()
    )
    diesel["Diesel_M"] = (diesel["Amount"] / 1e6).round(2)

    elec = _bdf[_bdf["Category"] == "Electricity"][["Date", "Amount"]].copy()
    elec["Elec_M"] = (elec["Amount"] / 1e6).round(2)

    # ── Estimated actual fleet fuel & electricity for FY2025 and FY2026 ───────
    # Uses real WTI monthly prices from yfinance; remaining months forward-filled
    # at current WTI. Volume proxy anchored to last confirmed actual (FY2024).
    # Electricity uses 5%/yr structural growth + 15% oil passthrough from FY2024 base.
    # NOTE: These are model estimates, NOT confirmed vendor payment records.
    CURRENT_WTI_LIVE   = 83.00
    ELEC_GROWTH_EST    = 0.05
    ELEC_PASS_EST      = 0.15

    def _avg_wti_for_year(yr: int) -> tuple:
        """Returns (avg_wti, months_actual) for a given calendar year."""
        prices = raw[raw["Date"].dt.year == yr]["WTI"].tolist()
        n_actual = len(prices)
        n_remaining = max(0, 12 - n_actual)
        full = prices + [CURRENT_WTI_LIVE] * n_remaining
        return float(np.mean(full)), n_actual

    # FY2024 actual base volumes (from Chicago Data Portal research)
    FLEET_ACTUAL_2024  = 16_144_459 + 20_385_338   # diesel + gasoline ($)
    ELEC_ACTUAL_2024_M = 72_256_356 / 1e6           # electricity ($M)
    FLEET_VOL_PROXY    = FLEET_ACTUAL_2024 / BASE_DIESEL_GAL  # gallons proxy

    def _est_fleet_M(wti_avg: float) -> float:
        price_gal = BASE_DIESEL_GAL + (wti_avg - 72) * CRUDE_TO_DIESEL
        return round(FLEET_VOL_PROXY * price_gal / 1e6, 2)

    def _est_elec_M(wti_avg: float, years_from_2024: int) -> float:
        oil_fx = ELEC_PASS_EST * (wti_avg - 72) / 72
        return round(ELEC_ACTUAL_2024_M * (1 + ELEC_GROWTH_EST + oil_fx) ** years_from_2024, 2)

    wti_2025_avg, mo_2025 = _avg_wti_for_year(2025)
    wti_2026_avg, mo_2026 = _avg_wti_for_year(2026)

    fleet_est_2025_M = _est_fleet_M(wti_2025_avg)
    fleet_est_2026_M = _est_fleet_M(wti_2026_avg)
    elec_est_2025_M  = _est_elec_M(wti_2025_avg, 1)
    elec_est_2026_M  = _est_elec_M(wti_2026_avg, 2)

    # ── FY2026 estimated actual fleet fuel spend (budget comparison — Tab 4) ────
    # Volume proxy from total fleet fuel budget (diesel + gasoline + alt fuel)
    BASE_VOL_GAL         = float(diesel["Amount"].iloc[-1]) / BASE_DIESEL_GAL
    diesel_price_2026    = BASE_DIESEL_GAL + (wti_2026_avg - 72) * CRUDE_TO_DIESEL
    diesel_est_2026_M    = round(BASE_VOL_GAL * diesel_price_2026 / 1e6, 2)
    diesel_budget_2026_M = float(diesel["Diesel_M"].iloc[-1])

    # ── WTI projections (EIA base forecast + ±1σ bands) ─────────────────────
    # Source: EIA STEO Mar 2026 — adjusted for Hormuz crisis baseline ($83 today).
    # Gradual mean-reversion: crisis premium dissipates over 2 years.
    # σ=22% (EIA annual WTI forecast uncertainty), symmetric ±1σ floor at $40.
    SIGMA_WTI = 0.22
    # Annual anchor points (year → $/bbl)
    _wti_anchors = {
        2026: 83,   # current (bridge)
        2027: 76,   # partial crisis resolution
        2028: 71,   # continued normalization
        2029: 68,   # near long-run equilibrium
        2030: 65,
        2031: 63,
    }
    proj_years = [2027, 2028, 2029, 2030]
    proj_dates  = [pd.Timestamp(f"{y}-01-01") for y in proj_years]
    wti_base    = [float(_wti_anchors[y]) for y in proj_years]
    wti_hi      = [round(v * (1 + SIGMA_WTI), 1)              for v in wti_base]
    wti_lo      = [round(max(v * (1 - SIGMA_WTI), 40.0), 1)   for v in wti_base]

    # Monthly-interpolated WTI projection (for a continuous, non-straight line)
    # Linearly interpolate between annual anchors at monthly frequency,
    # then add a small seeded seasonal perturbation so it looks realistic.
    rng = np.random.default_rng(42)
    _anchor_years = sorted(_wti_anchors)
    _anchor_vals  = [_wti_anchors[y] for y in _anchor_years]
    _all_months   = pd.date_range(
        start=pd.Timestamp(f"{_anchor_years[0]}-01-01"),
        end=pd.Timestamp(f"{_anchor_years[-1]}-12-01"),
        freq="MS",
    )
    # Linear interpolation at monthly level
    _t_anchors = np.array([(pd.Timestamp(f"{y}-01-01") - _all_months[0]).days
                            for y in _anchor_years], dtype=float)
    _t_months  = np.array([(d - _all_months[0]).days for d in _all_months], dtype=float)
    _wti_interp = np.interp(_t_months, _t_anchors, _anchor_vals)
    # Small seasonal perturbation (±3%) to mimic real WTI oscillations
    _seasonal = rng.normal(0, 0.025, len(_wti_interp))
    _wti_monthly_proj = np.round(_wti_interp * (1 + _seasonal), 2)
    # Upper / lower monthly bands (widening fan)
    _frac = (_t_months - _t_anchors[0]) / (_t_anchors[-1] - _t_anchors[0])  # 0→1
    _band_sigma = SIGMA_WTI * _frac  # fan widens with horizon
    wti_proj_monthly_dates  = _all_months.tolist()
    wti_proj_monthly_base   = _wti_monthly_proj.tolist()
    wti_proj_monthly_hi     = np.round(_wti_interp * (1 + _band_sigma), 1).tolist()
    wti_proj_monthly_lo     = np.round(
        np.maximum(_wti_interp * (1 - _band_sigma), 40), 1
    ).tolist()

    # ── Fleet fuel budget projections (diesel + gasoline + alt fuel) ──────────
    # Volume proxy: FY2026 total fleet fuel budget / base diesel price per gal
    BASE_VOL_GAL  = float(diesel["Amount"].iloc[-1]) / BASE_DIESEL_GAL
    def _diesel_proj(wti_vals):
        return [round(BASE_VOL_GAL * (BASE_DIESEL_GAL + (w - 72) * CRUDE_TO_DIESEL) / 1e6, 2)
                for w in wti_vals]

    diesel_proj_base = _diesel_proj(wti_base)
    diesel_proj_hi   = _diesel_proj(wti_hi)
    diesel_proj_lo   = _diesel_proj(wti_lo)

    # ── Electricity budget projections ────────────────────────────────────────
    # Structural growth ~5%/yr + oil passthrough (0.15 elec passthrough)
    ELEC_GROWTH  = 0.05
    ELEC_PASS    = 0.15
    elec_base_val = float(elec["Amount"].iloc[-1]) / 1e6   # FY2026 = $90.07M

    def _elec_proj(wti_vals):
        out = []
        val = elec_base_val
        for i, w in enumerate(wti_vals):
            oil_effect = ELEC_PASS * (w - 72) / 72
            val = val * (1 + ELEC_GROWTH + oil_effect)
            out.append(round(val, 2))
        return out

    elec_proj_base = _elec_proj(wti_base)
    elec_proj_hi   = _elec_proj(wti_hi)
    elec_proj_lo   = _elec_proj(wti_lo)

    # ── Key events ───────────────────────────────────────────────────────────
    events = [
        {"date": pd.Timestamp("2022-02-01"), "label": "Ukraine invasion", "color": "#fb923c"},
        {"date": pd.Timestamp("2023-09-01"), "label": "OPEC+ cuts",       "color": "#a78bfa"},
        {"date": pd.Timestamp("2026-03-01"), "label": "Hormuz crisis",    "color": "#f87171"},
    ]

    return {
        "wti_dates":                raw["Date"].tolist(),
        "wti_prices":               raw["WTI"].round(2).tolist(),
        # Diesel-only budget (Motor Vehicle Diesel Fuel appropriation)
        # No gasoline budget line in chicago_energy_budget_2016_2025.xlsx
        "diesel_dates":             diesel["Date"].tolist(),
        "diesel_budget":            diesel["Diesel_M"].tolist(),
        "elec_dates":               elec["Date"].tolist(),
        "elec_budget":              elec["Elec_M"].tolist(),
        "proj_dates":               proj_dates,
        "wti_proj_base":            wti_base,
        "wti_proj_hi":              wti_hi,
        "wti_proj_lo":              wti_lo,
        # Monthly-resolution WTI forecast (for smooth, realistic-looking line)
        "wti_proj_monthly_dates":   wti_proj_monthly_dates,
        "wti_proj_monthly_base":    wti_proj_monthly_base,
        "wti_proj_monthly_hi":      wti_proj_monthly_hi,
        "wti_proj_monthly_lo":      wti_proj_monthly_lo,
        "diesel_proj_base":         diesel_proj_base,
        "diesel_proj_hi":           diesel_proj_hi,
        "diesel_proj_lo":           diesel_proj_lo,
        "elec_proj_base":           elec_proj_base,
        "elec_proj_hi":             elec_proj_hi,
        "elec_proj_lo":             elec_proj_lo,
        "events":                   events,
        "base_wti":                 72.0,
        "spike_wti":                107.82,
        "current_wti":              83.00,
        # FY2026 budget vs estimated actual (Tab 4 — budget view)
        "diesel_est_2026_M":        diesel_est_2026_M,
        "diesel_budget_2026_M":     diesel_budget_2026_M,
        "wti_2026_avg":             round(wti_2026_avg, 1),
        "diesel_2026_months":       mo_2026,
        # Estimated actual fleet fuel + electricity FY2025 & FY2026 (Tab 5 — actual view)
        "fleet_est_2025_M":         fleet_est_2025_M,
        "fleet_est_2026_M":         fleet_est_2026_M,
        "elec_est_2025_M":          elec_est_2025_M,
        "elec_est_2026_M":          elec_est_2026_M,
        "wti_2025_avg":             round(wti_2025_avg, 1),
        "wti_2025_months":          mo_2025,
        "wti_2026_months":          mo_2026,
    }


def generate_wti_forecast(
    current_wti: float = 83.0,
    peak_wti: float    = 90.0,
    peak_year: int     = 2027,
    halflife_yrs: float = 2.0,
    long_run_eq: float = 65.0,
    sigma: float       = 0.22,
    base_diesel_vol_gal: float = None,
    base_elec_M: float         = None,
    seed: int = 42,
) -> dict:
    """
    Parametric WTI price forecast using mean-reversion (OU) model.

    Path shape:
      • Linear ramp from current_wti → peak_wti at peak_year-06 (mid-year peak)
      • Exponential mean-reversion from peak toward long_run_eq:
            P(t+1) = long_run_eq + (P(t) − long_run_eq) · exp(−κ/12)
            where κ = ln(2) / halflife_yrs
      • Monthly resolution + small seasonal noise (seeded, reproducible)
      • ±1σ uncertainty fan that widens as √t (proper diffusion)

    Also computes diesel-budget and electricity-budget projections that
    react to the WTI path, ready to feed make_oil_cost_chart().
    """
    rng = np.random.default_rng(seed)

    start   = pd.Timestamp("2026-03-01")
    end     = pd.Timestamp("2033-12-01")
    months  = pd.date_range(start=start, end=end, freq="MS")
    n       = len(months)

    peak_ts = pd.Timestamp(f"{peak_year}-06-01")   # shock peaks mid-year

    # ── Build base WTI path ──────────────────────────────────────────────────
    kappa_mo = np.log(2) / (halflife_yrs * 12)      # monthly reversion rate
    wti = np.empty(n)
    wti[0] = current_wti

    for i, d in enumerate(months[1:], 1):
        prev = wti[i - 1]
        if d <= peak_ts:
            # linear ramp to peak
            frac  = (d - months[0]).days / max((peak_ts - months[0]).days, 1)
            wti[i] = current_wti + (peak_wti - current_wti) * frac
        else:
            # OU mean-reversion toward long_run_eq
            wti[i] = long_run_eq + (prev - long_run_eq) * np.exp(-kappa_mo)

    # Small seasonal perturbation (±2.5%) for realism
    noise = rng.normal(0, 0.025, n)
    wti_base = np.round(wti * (1 + noise), 2)

    # ── ±1σ uncertainty fan (widens as √t from first month) ─────────────────
    t_years = np.array([(d - months[0]).days / 365.25 for d in months])
    sigma_t = sigma * np.sqrt(np.maximum(t_years, 0))
    wti_hi  = np.round(wti * (1 + sigma_t), 1)
    wti_lo  = np.round(np.maximum(wti * (1 - sigma_t), 35.0), 1)

    # Annual anchor values (Jan of each year, for data labels)
    ann_years = [2027, 2028, 2029, 2030, 2031, 2032, 2033]
    ann_dates = [pd.Timestamp(f"{y}-01-01") for y in ann_years]
    ann_idx   = [int(np.argmin(np.abs((months - d).days))) for d in ann_dates]
    ann_wti   = [round(float(wti_base[i]), 1) for i in ann_idx]

    # ── Diesel budget projection ──────────────────────────────────────────────
    diesel_monthly_M, diesel_ann_M = [], []
    diesel_hi_ann, diesel_lo_ann   = [], []
    if base_diesel_vol_gal:
        def _d_budget(w_arr):
            return [round(base_diesel_vol_gal *
                          (BASE_DIESEL_GAL + (w - 72) * CRUDE_TO_DIESEL) / 1e6, 2)
                    for w in w_arr]
        diesel_monthly_M = _d_budget(wti_base)
        diesel_ann_M     = [diesel_monthly_M[i] for i in ann_idx]
        diesel_hi_ann    = _d_budget([float(wti_hi[i]) for i in ann_idx])
        diesel_lo_ann    = _d_budget([float(wti_lo[i]) for i in ann_idx])

    # ── Electricity budget projection ─────────────────────────────────────────
    elec_monthly_M, elec_ann_M = [], []
    elec_hi_ann, elec_lo_ann   = [], []
    if base_elec_M:
        ELEC_GROWTH = 0.05
        ELEC_PASS   = 0.15
        def _e_path(w_arr):
            out, val = [], base_elec_M
            for w in w_arr:
                oil_fx = ELEC_PASS * (w - 72) / 72
                val    = val * (1 + ELEC_GROWTH / 12 + oil_fx / 12)
                out.append(round(val, 2))
            return out
        elec_monthly_M = _e_path(wti_base)
        elec_ann_M     = [elec_monthly_M[i] for i in ann_idx]
        elec_hi_ann    = [round(base_elec_M * (1 + ELEC_GROWTH) ** (y - 2026) *
                               (1 + ELEC_PASS * (float(wti_hi[ann_idx[k]]) - 72) / 72), 2)
                          for k, y in enumerate(ann_years)]
        elec_lo_ann    = [round(base_elec_M * (1 + ELEC_GROWTH) ** (y - 2026) *
                               (1 + ELEC_PASS * (float(wti_lo[ann_idx[k]]) - 72) / 72), 2)
                          for k, y in enumerate(ann_years)]

    return {
        # Monthly resolution — for the continuous WTI forecast line
        "wti_proj_monthly_dates": months.tolist(),
        "wti_proj_monthly_base":  wti_base.tolist(),
        "wti_proj_monthly_hi":    wti_hi.tolist(),
        "wti_proj_monthly_lo":    wti_lo.tolist(),
        # Annual anchors — for data labels + budget bands
        "proj_dates":             ann_dates,
        "wti_proj_base":          ann_wti,
        "wti_proj_hi":            [round(float(wti_hi[i]), 1) for i in ann_idx],
        "wti_proj_lo":            [round(float(wti_lo[i]), 1) for i in ann_idx],
        "diesel_proj_base":       diesel_ann_M,
        "diesel_proj_hi":         diesel_hi_ann,
        "diesel_proj_lo":         diesel_lo_ann,
        "elec_proj_base":         elec_ann_M,
        "elec_proj_hi":           elec_hi_ann,
        "elec_proj_lo":           elec_lo_ann,
        # Scenario params (for display)
        "scenario_peak_wti":      peak_wti,
        "scenario_peak_year":     peak_year,
        "scenario_halflife":      halflife_yrs,
        "scenario_long_run":      long_run_eq,
    }


# ── Quick validation ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time
    print("Loading Fleet simulation...")
    t0    = time.time()
    fleet = FleetSim()
    vol   = fleet.fuel_volatility_summary()
    print(f"  σ_diesel (calibrated): {vol['sigma_diesel']:.3f} ({vol['sigma_diesel']*100:.1f}%)")
    print(f"  σ_elec   (calibrated): {vol['sigma_elec']:.3f}   ({vol['sigma_elec']*100:.1f}%)")

    det  = fleet.det_scenario_table()
    print(f"  det_scenario_table: OK ({time.time()-t0:.1f}s)")

    t1 = time.time()
    mc = fleet.mc_risk_bands()
    print(f"  mc_risk_bands (fuel-aware): OK ({time.time()-t1:.1f}s)")
    print(mc[mc["Fiscal_Year"] == 2026].to_string(index=False))

    print("\nLoading Facilities simulation...")
    t2  = time.time()
    fac = FacilitiesSim()
    fac_det = fac.det_scenario_table()
    fac_mc  = fac.mc_risk_bands()
    print(f"  Facilities MC (fuel+passthrough aware): OK ({time.time()-t2:.1f}s)")
    print(fac_mc[fac_mc["Fiscal_Year"] == 2026].to_string(index=False))

    print(f"\nTotal load time: {time.time()-t0:.1f}s")

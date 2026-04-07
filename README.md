# TDCsim — Treasury Deposit Component Simulator

TDCsim is a **stock-flow Treasury funding-chain simulator** built for thesis research. It answers a focused question: **how Treasury issuance, debt service, holder mix, and Treasury cash management change the Treasury-attributed deposit component (TDC), reserves, and the maturity structure of debt**.

The simulator is designed for scenario analysis, not forecasting. It makes the Treasury funding chain explicit so you can compare mechanisms across counterfactuals — different issuance mixes, yield curves, holder preferences, TGA paths, and debt-service dynamics.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q          # run the test suite
python run.py      # run all scenario groups
python run.py tdc_config_optional.yaml  # run the optional-feature example config
```

By default, the project uses the shipped `tdc_config.yaml` and generates a synthetic starting portfolio. Generated CSVs and plots are written to `output/`.

## What the simulator does

TDCsim models a Treasury liability portfolio at the tranche level and steps it forward through time. In each period the engine:

1. **Applies fiscal flows** — spending, taxes, and the primary deficit path.
2. **Updates TGA, reserves, and TDC** — tracks the Treasury General Account, banking-system reserves, and the deposit component.
3. **Pays interest and principal by holder type** — coupon payments, bill maturities, and TIPS/FRN-specific debt service, routed through the correct balance-sheet channels.
4. **Issues new debt** across a configurable maturity and security-type mix — bills, notes, bonds, TIPS, FRNs, and non-marketable liabilities.
5. **Allocates new issuance across holders** using configurable sector preferences, determining who absorbs each tranche and what the deposit/reserve impact is.
6. **Optionally simulates secondary-market preference trading** — a simplified reallocation mechanism where holders trade toward their preferred portfolio mix.
7. **Tracks debt composition, financing cost, CPI/reference CPI paths, and WAM** period by period.

## Core idea: TDC is a component, not "all deposits"

The simulator is built around the idea that Treasury operations can change deposits even when the headline fiscal story looks similar. The deposit effect depends on **who receives Treasury outlays**, **who buys new issuance**, **who receives debt service**, and **whether cash is accumulating in or leaving the TGA**.

That is why a Treasury debt-management problem can be monetary even without changing the headline deficit path.

> **Positive TDC = the Treasury deposit component is increasing. Negative TDC = the Treasury deposit component is decreasing.**

---

## Conceptual definitions

The simulator is motivated by the following accounting identities.

### 1. DU / RU transaction view

$$\Delta D_{TDC} = (G_{dep} - T_{dep}) + (D_{sales} - D_{purch}) + D_{yield}$$

| Term | Meaning |
|---|---|
| $G_{dep} - T_{dep}$ | Net fiscal injection to deposit users (DUs) |
| $D_{sales} - D_{purch}$ | Net Treasury-security sales from DUs to reserve users (RUs) |
| $D_{yield}$ | Interest and principal payments to DUs |

**DUs** are domestic non-banks. **RUs** include banks, the central bank, Treasury/federal accounts, and foreign reserve-linked holders.

### 2. Treasury / central-bank / TGA view

$$\Delta D_{TDC} = (D_{sales} - D_{purch}) + (T_{AV} - T_{Rx} - R_{yield} + R_{Tx}) + \max(0,\; F_{PY} + M_{MT} - F_{OE}) - \Delta TGA$$

| Term | Meaning |
|---|---|
| $D_{sales} - D_{purch}$ | Net TS sales from DUs to RUs |
| $T_{AV} - T_{Rx} - R_{yield} + R_{Tx}$ | Net Treasury flows with RUs |
| $\max(0,\; F_{PY} + M_{MT} - F_{OE})$ | Central-bank factors (remittances, seigniorage) |
| $\Delta TGA$ | Change in the Treasury General Account |

Where $T_{AV}$ is auction receipts, $T_{Rx}$ is Treasury outlays to RUs, $R_{Tx}$ is Treasury receipts from RUs, $F_{PY}$ is CB income, $F_{OE}$ is CB operating expenses, and $M_{MT}$ includes seigniorage.

### 3. Bank-balance-sheet decomposition view

$$\Delta D_{TDC} = (\Delta M - \Delta C - \Delta X) - (\Delta L_{B,DU} + \Delta S_{B,DU} - \Delta CB_B) - \Delta CB_{NB} - \Delta FI_{NonTS}$$

| Term | Meaning |
|---|---|
| $\Delta M - \Delta C - \Delta X$ | DU liquid balances net of non-deposits |
| $\Delta L_{B,DU} + \Delta S_{B,DU} - \Delta CB_B$ | Bank non-TS asset acquisition from DUs (net) |
| $\Delta CB_{NB}$ | CB non-TS operations with non-banks |
| $\Delta FI_{NonTS}$ | Foreign non-TS flows to DUs |

These three views are different ways of isolating the **same Treasury deposit component**.

---

## TDC decomposition

The simulator reports the mechanism directly through a 5-term decomposition:

| Channel | Column | What it captures |
|---|---|---|
| Fiscal flows | `TDC_FiscalFlow` | Net fiscal impact (taxes minus spending) |
| Debt service | `TDC_DebtService` | Deposit effect of interest and principal payments by holder type |
| Auction absorption | `TDC_AuctionAbsorption` | Deposit effect of new debt issuance by holder type |
| Secondary trades | `TDC_SecondaryTrades` | Deposit effect of preference-driven secondary-market trading |
| Other | `TDC_Other` | Reserve transfers, CB expenses, minting flows |

$$TDC_{Change} = TDC_{FiscalFlow} + TDC_{DebtService} + TDC_{AuctionAbsorption} + TDC_{SecondaryTrades} + TDC_{Other}$$

This identity is enforced row-wise and verified by automated tests. It tells you **why** TDC moved, not just that it moved.

---

## Holder types and security types

### Holders

| Holder | Role | Deposit/reserve impact |
|---|---|---|
| `Private` | Domestic non-banks (deposit users) | DU — Treasury flows create/destroy deposits |
| `Banks` | Commercial banks (reserve users) | RU — purchases create deposits; interest receipt is a reserve flow |
| `CB` | Central bank | RU — remittances and balance-sheet operations |
| `Foreign` | Foreign official and private holders | RU — reserve-linked |
| `FedInternal` | Treasury-internal accounts | Intragovernmental — P&I is a TGA wash, no reserve/deposit impact |
| `TrustFunds` | Social Security, Medicare, etc. | Intragovernmental — large non-marketable holdings, P&I is a TGA wash |

### Security types

| Type | Features |
|---|---|
| `Fixed` | Fixed-rate marketable Treasury debt (bills, notes, bonds) |
| `TIPS` | Treasury Inflation-Protected Securities — principal adjusts by CPI index ratio, real coupon rate |
| `FRN` | Floating Rate Notes — coupon resets to benchmark + spread each period |
| `NonMarketable` | Non-marketable Treasury liabilities (trust fund bonds) — interest credited semiannually, not traded |

Bills use **discounted proceeds** (not face value) for TGA/reserve/deposit effects. Secondary trading categorizes by **original maturity** (not remaining maturity).

---

## Scenario levers

The simulator is configuration-driven. Typical scenario levers include:

- **Fiscal path** — spending and tax growth rates
- **TGA target and floor** — Treasury cash management policy
- **Yield curve** — exogenous term structure (static or scenario-specific)
- **Issuance profile** — allocation across bills, notes, bonds, TIPS, FRNs, and non-marketable debt
- **Sector holding preferences** — who absorbs new issuance at auction
- **Secondary-market trading** — optional preference-driven reallocation
- **TIPS parameters** — inflation rate, reference CPI, real coupon
- **FRN parameters** — benchmark rate, spread
- **Dated events** — parameter changes that take effect mid-simulation (e.g., a TGA rebuild starting in Q3, a yield curve shift in year 3)
- **Initial portfolio** — generated from config or loaded from CSV

This makes the project a **counterfactual lab** rather than a one-off chart generator.

## Default scenario set

The shipped `tdc_config.yaml` includes three scenario groups:

1. **Treasury Market Scenarios** — a baseline, a bank-absorption + lower-rates scenario, and a maturity-shift scenario. Compares headline mechanisms.
2. **One-Factor Isolation** — five scenarios that change exactly one lever at a time (issuance mix, holder preferences, yield curve) relative to a common baseline. Supports ceteris paribus attribution.
3. **TDC vs Debt Management** — a combined comparison group focused on debt-management policy.

See `scenario_overview.md` for full scenario descriptions and decomposition logic.

---

## Outputs

A typical run produces:

- **TDC level and change** with the full 5-term decomposition
- **TGA and reserve balances** period by period
- **Government spending, taxes, and primary deficit**
- **Interest payments, principal payments, and debt-service outlays**
- **New issuance by period**
- **Debt held by sector** and **by security type**
- **Aggregate WAM** of the active debt stock
- **Inflation / CPI reference paths** for TIPS accounting
- **Final tranche-level portfolio**
- **Scenario comparison charts** (saved as PNG)

Key result columns:

| Column | Description |
|---|---|
| `TDC_Level` | Cumulative Treasury deposit component |
| `TDC_Change` | Period change in TDC (= sum of decomposition) |
| `Reserves` | Banking-system reserve balance |
| `TGA` | Treasury General Account balance |
| `TotalDebt_Agg` | Total outstanding debt |
| `DebtServiceOutlay_Cumulative` | Cumulative debt-service payments |
| `FinancingCost_Cumulative` | Cumulative financing cost |
| `TIPSInflationAccretion_Cumulative` | Optional cumulative TIPS principal accretion included in financing cost |
| `AuctionDemandShift_AvgAbs` | Mean absolute auction-share shift from rate-sensitive demand overlays |
| `SecondaryDemandShift_AvgAbs` | Mean absolute secondary-target shift from rate-sensitive demand overlays |
| `WAM` | Weighted-average maturity of the active portfolio |

---

## Use cases

TDCsim is especially useful for:

- **Debt-management counterfactuals** — compare issuance profiles (more bills vs. more notes vs. more bonds) and see how TDC, reserves, and financing cost diverge.
- **TGA rebuild and spend-down episodes** — study how TGA refills change TDC and reserves depending on who funds the refill.
- **Holder-mix experiments** — ask whether the same gross issuance has different monetary effects depending on whether banks, non-banks, foreigners, or the CB absorb it.
- **Debt-service channel analysis** — measure how growing debt-service payments feed back into TGA, reserves, and TDC depending on who holds the debt.
- **WAM and maturity-structure analysis** — compare whether shorter or longer maturity structures change TDC, funding needs, or reserve pressure.
- **Explaining why TDC can move differently from debt or deficits** — the main reason the model exists. A larger debt stock or deficit does not mechanically imply a larger TDC if the holder mix and cash-management path change.

---

## Configuration

The main configuration file is `tdc_config.yaml`.

| Section | Purpose |
|---|---|
| `simulation_period` | Start date, end date, frequency, and whether preference trading is enabled |
| `initial_values` | Starting TGA, reserves, and TDC level |
| `initial_portfolio` | Mode (`generated`, `config_derived`, or `file`), generation parameters, target face values by holder |
| `fiscal_params` | Spending and tax assumptions |
| `tga_params` | Treasury cash target and floor |
| `yield_curve` | Exogenous term structure inputs (named curves for scenario overrides) |
| `treasury_issuance_profile` | Issuance mix — bills/notes/bonds/TIPS/FRNs/nonmarketable, maturity distributions |
| `sector_preferences` | Holder absorption preferences for auction allocation |
| `rate_sensitive_demand` | Optional demand elasticities layered on top of auction and secondary preference blocks |
| `tips_params` | Inflation rate, reference CPI, real coupon rate |
| `financing_cost_options` | Optional financing-cost accounting toggles |
| `frn_params` | Benchmark rate and spread |
| `nonmarketable_params` | Trust fund interest crediting rate |
| `scenario_groups` | Scenario batches — each group defines scenarios with parameter overrides and output settings |

Scenarios are defined as overrides on the base config. Each scenario in a group specifies which parameters to change (yield curve, issuance profile, sector preferences, events, etc.), and the engine runs all scenarios in a group and produces comparative plots.

`initial_portfolio` is different: it is loaded once before scenarios run, so changing the starting stock is a run-level choice, not a per-scenario override.

### Optional additions

- `initial_portfolio.mode: config_derived` builds the starting stock from the configured issuance profile, holder preferences, and yield curve instead of the legacy hard-coded maturity menus.
- `rate_sensitive_demand` keeps yields exogenous but lets auction and secondary demand shares flex with yield level, spread-to-anchor, or curve slope. When `enabled: false`, behavior is unchanged.
- `financing_cost_options.include_tips_inflation_accretion: true` adds TIPS principal accretion to `FinancingCost_Period` and exposes `TIPSInflationAccretion_Period` / `TIPSInflationAccretion_Cumulative`.
- The engine now exposes lightweight diagnostics for the new demand system via `AuctionDemandShift_AvgAbs`, `AuctionDemandShift_MaxAbs`, `SecondaryDemandShift_AvgAbs`, and `SecondaryDemandShift_MaxAbs`.

Mergeable examples live in [examples/optional_feature_scenarios.yaml](/Users/shanewray/malus/proj/tdcsim/examples/optional_feature_scenarios.yaml). A ready-to-run alternate config is available at [tdc_config_optional.yaml](/Users/shanewray/malus/proj/tdcsim/tdc_config_optional.yaml), and the chosen auction-demand coefficients are summarized in [examples/calibration_notes.md](/Users/shanewray/malus/proj/tdcsim/examples/calibration_notes.md).

---

## Important assumptions and conventions

1. **TDC is a component, not the whole deposit stock.** The model isolates the Treasury-attributable portion of deposit dynamics.
2. **Private = domestic non-banks.** That is the sector most closely aligned with deposit users.
3. **Intragovernmental holders are split into FedInternal and TrustFunds.** Both are intragovernmental (P&I is a TGA wash, no reserve or deposit impact).
4. **Monetary units are billions USD.** Time units are years (for yields, TTM, WAM).
5. **Transaction values, not mark-to-market.** For TDC accounting, the price that matters is the price at which the security changes hands.
6. **Yield curves are exogenous.** The model does not endogenize rates from demand.
7. **WAM is a scenario statistic, not a sufficient statistic.** It is useful, but does not replace the full holder-by-instrument funding-chain view.

---

## How it works

The runtime flow:

1. `run.py` calls `simulation_core.main()` and can optionally take a config path argument.
2. `simulation_core.py` loads and validates the selected config file, then loads or generates the initial portfolio.
3. `sim_groups.py` dispatches each scenario group (parallel for multiple groups, serial for one).
4. For each scenario, `sim_engine.py` runs the simulation loop period by period — fiscal flows, debt service, issuance, allocation, optional secondary trading — and accumulates results.
5. `sim_plotting.py` writes comparison charts across scenarios in each group.
6. Results are returned as DataFrames with all tracked columns.

## Project layout

```text
run.py                   CLI entry point
tdc_config.yaml          default configuration and scenario groups
scenario_overview.md     scenario descriptions and design notes
requirements.txt         runtime Python dependencies
requirements-dev.txt     runtime + test dependencies
pytest.ini               pytest configuration
LICENSE                  MIT license
src/
  simulation_core.py     config loading, validation, orchestration, portfolio loading/generation
  sim_engine.py          core simulation loop (fiscal, debt service, issuance, allocation)
  sim_pricing.py         yield interpolation, coupon scheduling, accrued interest, bond pricing
  sim_trading.py         secondary-market preference trading
  sim_plotting.py        scenario comparison chart generation
  sim_groups.py          scenario group dispatch and parallel execution
  sim_helpers.py         nested config updates, event application, run validation wrapper
  tdc_validation.py      strict config and event validation
  csv_gen.py             synthetic initial-portfolio generator
  tdc_shared.py          shared constants, schema definitions, dtypes
tests/
  conftest.py            test import path setup
  test_config_and_trading.py     config validation, issuance mechanics, trading logic
  test_engine_and_validation.py  accounting identities, bucket splits, maturity categorization
  test_coupon_schedule.py        coupon date scheduling and accrual
  test_yield_curve.py            yield interpolation and curve handling
output/                  generated portfolios and plots (gitignored)
```

## Testing

The repo includes 75 automated tests covering:

- Coupon scheduling and accrual calculations
- Yield curve interpolation
- Configuration validation and error handling
- Accounting identities (TDC decomposition sums, debt stock consistency)
- Holder bucket splits and maturity categorization
- TIPS, FRN, and non-marketable instrument behavior
- Secondary trading mechanics
- Event validation and application

```bash
# Run tests (install dev dependencies first)
pip install -r requirements-dev.txt
pytest -q
```

---

## Important limitations

This is a mechanism-focused simulator, not a macro forecast model.

- **Yield curves are exogenous.** The model does not endogenize rates from demand. Statements like "more bank demand lowers rates" require an explicit yield curve scenario, not just a preference shift.
- **The legacy generated starting portfolio is stylized.** `initial_portfolio.mode: generated` uses representative maturities and coupons, not a full reconstruction of the actual outstanding Treasury portfolio. `config_derived` reduces this gap but still does not recreate the real Treasury stock.
- **Best used for comparing scenarios**, not estimating real-world outcomes directly.
- **Preference trading is simplified.** It is a single-pass reallocation per period, not a full market microstructure model.
- **By default, TIPS inflation accretion is excluded from financing cost.** Enable `financing_cost_options.include_tips_inflation_accretion` to include it explicitly.
- **Coupon logic assumes sub-semiannual step frequency.** Weekly (the default) is always safe. Monthly or coarser frequencies may miss coupon dates.
- **Coarse frequencies are guarded but still approximate.** `simulation_period.coarse_frequency_action` can be set to `warn`, `allow`, or `error`. The shipped configs use `warn`; weekly remains the recommended mode.

## What this is not

TDCsim is not:

- a GDP or inflation forecasting model,
- a full model of all economy-wide deposits,
- a full market-microstructure model,
- a substitute for empirical estimation of TDC in real-world data,
- investment, legal, or policy advice.

It is a **structured scenario engine** built around Treasury accounting, sector allocation, and funding-chain mechanics.

## License

[MIT](LICENSE)

# CBO Downstream Quickstart

This is the public use path for downstream projects. You should not need to
understand the verifier internals to run scenarios.

## Inputs

Use one locally generated or supplied baseline package and its attestation:

```bash
output/cbo_forecast_release_bound_package.zip
output/cbo_forecast_release_bound_attestation.json
```

That baseline is the fixed CBO input package for the run. Scenario files only
describe what you want to change. Generated packages and attestations are local
artifacts and are not shipped in the repository.

## Generate Example Scenarios

```bash
python scripts/write_cbo_example_scenarios.py \
  --baseline output/cbo_forecast_release_bound_package.zip \
  --attestation output/cbo_forecast_release_bound_attestation.json \
  --output-dir /tmp/tdcsim-cbo-scenarios
```

This writes twenty-three ready-to-run examples:

- `00_baseline_noop.json`: run the baseline without scenario changes.
- `01_rates_inflation_frn_tips.json`: change the nominal curve, inflation, FRN benchmark, TIPS real yield, and real operating cash.
- `02_issuance_maturity_mix.json`: change issuance shares and maturity mix.
- `03_sector_holders.json`: change sector holder preferences from a future date forward.
- `04_fiscal_fed_cash.json`: change primary deficit scale, Fed stock target handling, cash residual, operating cash, and fiscal incidence.
- `05_rate_down_25bp.json`: matched nominal-rate down case with linked FRN benchmark.
- `06_rate_up_25bp.json`: matched nominal-rate up case with linked FRN benchmark.
- `07_issuance_shorter.json`: matched shorter issuance mix, holding TIPS/FRN shares fixed against the longer case.
- `08_issuance_longer.json`: matched longer issuance mix, holding TIPS/FRN shares fixed against the shorter case.
- `09_private_holder_high.json`: high private new-issuance holder share.
- `10_private_holder_low.json`: low private new-issuance holder share.
- `11_primary_deficit_down_1pct.json`: matched primary-deficit down case.
- `12_primary_deficit_up_1pct.json`: matched primary-deficit up case.
- `13_operating_cash_inflation_beta_50.json`: operating-cash-only inflation-beta scenario.
- `14_fed_holdings_scale_1.json`: Fed-holdings-only passthrough scenario.
- `15_empirical_issuance_shorter_uncoupled.json`: moderate empirical shorter-issuance control.
- `16_empirical_issuance_longer_uncoupled.json`: moderate empirical longer-issuance control.
- `17_empirical_shorter_termprem_down_cons.json`: shorter issuance plus conservative long-end rate decline.
- `18_empirical_shorter_termprem_down_central.json`: shorter issuance plus central long-end rate decline.
- `19_empirical_shorter_termprem_down_high.json`: shorter issuance plus high long-end rate decline.
- `20_empirical_longer_termprem_up_cons.json`: longer issuance plus conservative long-end rate rise.
- `21_empirical_longer_termprem_up_central.json`: longer issuance plus central long-end rate rise.
- `22_empirical_longer_termprem_up_high.json`: longer issuance plus high long-end rate rise.

The simulation start date must match the package opening-state date. For the
current release-bound package that is `2026-06-21`. For a short smoke run, add:

```bash
--start-date 2026-06-21 --end-date 2026-06-30
```

## Run A Scenario

After installing the package, use the CLI:

```bash
tdcsim-cbo run \
  --baseline output/cbo_forecast_release_bound_package.zip \
  --attestation output/cbo_forecast_release_bound_attestation.json \
  --scenario /tmp/tdcsim-cbo-scenarios/01_rates_inflation_frn_tips.json \
  --output-dir /tmp/tdcsim-cbo-runs/rates_inflation
```

Or use Python:

```python
from pathlib import Path
from tdcsim_cbo import CboBaselinePackage, CboScenarioSpec, run_cbo_scenario

baseline = CboBaselinePackage.open(
    "output/cbo_forecast_release_bound_package.zip",
    attestation_path="output/cbo_forecast_release_bound_attestation.json",
)
scenario = CboScenarioSpec.from_file("/tmp/tdcsim-cbo-scenarios/01_rates_inflation_frn_tips.json")
run = run_cbo_scenario(baseline, scenario, Path("/tmp/tdcsim-cbo-runs/rates_inflation"))
print(run.results_path)
```

## Main Outputs

Each run writes:

- `tdcsim_cbo_run_manifest.json`: what baseline, scenario, inputs, and outputs were used.
- `compile/compiled/forecast_inputs/`: TDCSIM-compatible forecast input CSV/JSON files.
- `outputs/results_compact.csv.gz`: daily scenario results.
- `outputs/final_portfolio_compact.csv.gz`: final active security portfolio.
- `outputs/tdcsim_period_issuance_flows.csv.gz`: period issuance by instrument, maturity bucket, holder, and private route.
- `outputs/tdcsim_period_principal_flows.csv.gz`: period redemptions/principal by actual holder and instrument, with separate TDC principal-recipient fields on the domestic-ultimate/MMF route.
- `outputs/tdcsim_period_payment_flows.csv.gz`: period interest/payment components with accounting-basis labels.
- `outputs/tdcsim_holder_stocks.csv.gz`: holder stocks by date, sector, instrument, and maturity bucket.
- `outputs/tdcsim_tdc_principal_route_stocks.csv.gz`: security stock by TDC principal-settlement route.
- `outputs/tdcsim_tdc_principal_route_stock_closure.csv.gz`: opening route stock plus issuance less redemption to closing route stock.
- `outputs/tdcsim_debt_target_bridge.csv.gz`: CBO public-debt target to controlled TDCSIM debt bridge.
- `outputs/tdcsim_scenario_metrics.csv.gz`: derived WAM, bill-share, and short-maturity-share metrics.
- `outputs/tdcsim_period_tdc_summary.csv.gz`: period TDC accounting totals, overlap, and ex-overlap TDC.
- `outputs/tdcsim_period_tdc_components.csv.gz`: period TDC components with direct-interest and default TDC-support flags.
- `outputs/summary.json`: small run summary.
- `outputs/catalog.sqlite`: optional artifact catalog when requested by the scenario.

The issuance, principal, and payment tables are event/security-grain tables.
Use `flow_id` for idempotent row ingestion and `security_id` when you need to
join flow rows back to simulated securities. Principal rows expose the actual
redeeming holder in `holder_sector` / `holder_subsector` and the TDC settlement
route in `tdc_principal_recipient_sector` / `tdc_principal_recipient_subsector`.
For RateWall-style net Treasury cashflow work, pair
`tdc_principal_cash_paid_to_du_bil` with DU issuance derived from private
issuance rows, or use the summary table's
`gross_principal_cash_paid_to_du_bil`,
`gross_issuance_proceeds_absorbed_by_du_bil`, and
`net_du_principal_issuance_cashflow_bil`. Payment rows also carry
`accounting_basis` and `is_additive_to_cash_total` so bills and TIPS memo
decompositions are not double-counted as cash.

The TDC summary and component tables are the safer handoff for downstream TDC
accounting. The summary table exposes `tdc_change_bil`,
`overlap_cashflow_bil`, and `tdc_change_ex_overlap_bil`. The component table
marks each component with `enters_direct_interest_support` and
`enters_tdc_deposit_support_default`; no component should enter both legs.
Amounts use the basis
`post_mmf_route_pass_through_pre_ratewall_beta_chi`, so downstream consumers
should not apply the MMF pass-through again.

Useful result columns include:

- `CBORequiredFaceIssuance`
- `NewDebtIssued`
- `AuctionProceeds`
- `CBOControlledDebtTarget`
- `CBOControlledDebtTargetError`
- `PrimaryDeficit`
- `TGA`
- `CBOOperatingCashTarget`
- `CBOCashReconciliationResidual`
- `DebtHeld_Banks`
- `DebtHeld_CB`
- `DebtHeld_Foreign`
- `DebtHeld_Private`
- `DebtHeldByType_Fixed`
- `DebtHeldByType_TIPS`
- `DebtHeldByType_FRN`
- `CBONetInterestDiagnostic`
- `NetInterestDiagnosticStatus`

## Scenario Knobs

The CBO scenario interface supports the main downstream controls:

- Yield curve shocks, including key-rate shocks on generated curve tenors.
- Inflation, FRN benchmark, and TIPS real-yield assumptions.
- Issuance shares and maturity mix, including weighted-average-maturity changes through the maturity distributions. FRNs remain two-year securities in this lane.
- Future-dated holder preference changes for newly issued marketable debt, by security category: bills, notes, bonds, TIPS, and FRNs.
- MMF deposit pass-through, defaulting to `0.97`.
- Primary deficit, debt target, operating cash, Fed stock target, cash residual, and fiscal incidence assumptions.
- Operating cash can be constant nominal, constant real, explicit path-based, or `inflation_beta` where `0.0` means constant nominal and `1.0` means fully inflation-scaled.
- Empirical issuance-duration scenarios can combine an issuance-mix override with a long-end `key_rate_bp` nominal-curve shock. These are externally calibrated assumption scenarios, not endogenous TDCSIM yield-response estimates.

## Boundaries

This lane is a CBO-baseline scenario runner. It does not claim to be the
official CBO model.

Keep these boundaries:

- CBO net interest is diagnostic only. It does not fund issuance.
- Operating cash and cash residual assumptions are not issuance plugs.
- Fed holdings are stock-target holder allocation mechanics, not Fed auction purchases.
- Fed remittances and deferred assets are not modeled in this lane.
- Holder-preference changes affect new issuance from their effective date forward; they do not force secondary-market rebalancing of the existing stock.
- Negative required issuance fails closed unless a scenario explicitly opts into the simplified at-par shortest-public-marketable retirement path.

Those boundaries are why downstream projects can safely change scenario
assumptions without accidentally changing what the CBO lane claims to prove.

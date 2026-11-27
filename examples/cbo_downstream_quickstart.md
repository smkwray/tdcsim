# CBO Downstream Quickstart

How to run scenarios against a CBO baseline package and read their outputs.

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
- `02_issuance_maturity_mix.json`: raise the bill-share target from 22% to 28% between 2027 and the end of 2028.
- `03_sector_holders.json`: change sector holder preferences from a future date forward.
- `04_fiscal_fed_cash.json`: change primary deficit scale, Fed stock target handling, cash residual, operating cash, and fiscal incidence.
- `05_rate_down_25bp.json`: matched nominal-rate down case with linked FRN benchmark.
- `06_rate_up_25bp.json`: matched nominal-rate up case with linked FRN benchmark.
- `07_issuance_shorter.json`: matched shorter case, a 30% bill-share target.
- `08_issuance_longer.json`: matched longer case, a 15% bill-share target.
- `09_private_holder_high.json`: high private new-issuance holder share.
- `10_private_holder_low.json`: low private new-issuance holder share.
- `11_primary_deficit_down_1pct.json`: matched primary-deficit down case.
- `12_primary_deficit_up_1pct.json`: matched primary-deficit up case.
- `13_operating_cash_inflation_beta_50.json`: operating-cash-only inflation-beta scenario.
- `14_fed_holdings_scale_1.json`: Fed-holdings-only passthrough scenario.
- `15_empirical_issuance_shorter_uncoupled.json`: moderate empirical shorter-issuance control (fixed gross issuance shares; see Scenario Knobs).
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
For net Treasury cashflow work, pair
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
- The outstanding bill-share target: `{"mode": "bill_share_target_v1", "bill_share_target": 0.25}`, or a dated path such as `[{"date": "2027-01-01", "bill_share_target": 0.22}, {"date": "2028-12-31", "bill_share_target": 0.28}]` (linear between dates, flat before the first and after the last). Bill terms and the split across notes, bonds, TIPS and FRNs stay those measured from the package's auction data. FRNs remain two-year securities in this lane.
- `replace_shares` instead fixes each security type's share of gross issuance. Bills then run off as they mature, so the outstanding bill share falls well below its opening level within a year; the empirical examples (15–22) use this mode.
- The funding closure: `{"funding_closure": {"mode": "..."}}` with `cbo_operating_cash_target_financing_v1` (default), `cbo_public_debt_target` or `cbo_debt_reference_plus_tga_floor_financing_v1`.
- Future-dated holder preference changes for newly issued marketable debt, by security category: bills, notes, bonds, TIPS, and FRNs.
- MMF deposit pass-through, defaulting to `1.0`. Any lower share requires an explicit ON RRP share for the balance.
- Holder position targets (`holder_position_target`, mode `baseline_relative_stock_v1`): each public holder's Treasury holdings (par, adjusted principal for TIPS) are held at a completed baseline run's daily closing holdings plus a dated increment. One buyer's increment rises linearly to a stated size and then holds; named donors give it up in fixed proportions to their bill holdings at the start, and the remaining routes stay at baseline. The scenario references a target file built from the baseline run (`scripts/write_scenario_family.py --phase positions`) and must declare a simulation interval ending on the last hold date.
- Primary deficit, debt target, operating cash, Fed stock target, cash residual, and fiscal incidence assumptions.
- Operating cash can be constant nominal, constant real, explicit path-based, or `inflation_beta` where `0.0` means constant nominal and `1.0` means fully inflation-scaled.
- Empirical issuance-duration scenarios can combine an issuance-mix override with a long-end `key_rate_bp` nominal-curve shock. The rate shock sizes are calibrated outside TDCSim.

## Model Rules

- By default each day issues what keeps the TGA on its operating-cash path, net of the cash residual; above the path nothing is issued. Debt absorbs cost differences, and `DebtDriftFromReference` reports the gap to CBO's debt path. CBO net interest is reported for comparison.
- Each day's decided issuance first moves bills toward the bill-share target (counting issuance decided but not yet settled), then splits the rest across coupon securities by the auction shares. Each security settles on its Treasury calendar day, listed in the package's issuance calibration; `PendingIssuanceProceeds` is what is decided but not yet settled, so `TGA + PendingIssuanceProceeds` equals the operating-cash target on every day with a need. Within a year the TGA therefore sits below the operating-cash path by the issuance awaiting settlement. `EarlySettlementProceeds` records pending issuance settled early to keep the TGA from going negative. On the last business day of each fiscal year the books close and everything pending settles (`FiscalYearCloseSettlementProceeds`), so the TGA is on the path at each fiscal-year end and fiscal-year-end debt carries no pending issuance. The calendar is Treasury's: bills weekly (the 52-week bill every four weeks); 2-, 3- and 5-year notes monthly; the 7-year monthly until February 2027, then new in March, June, September and December and reopened in the two months after; 10-, 20- and 30-year securities new in February, May, August and November and reopened in the two months after, with 20-year reopenings settling on the Friday of their auction week from June 2026; the floating-rate note new quarterly and reopened monthly; TIPS on Treasury's schedule. Its dates match Treasury's published auction schedules through April 2027; later years extend the same rules. A reopening sells more of the latest issue on its maturity and coupon, at market value plus accrued interest; settlements off the calendar (early or at the close) reopen each security's latest issue. Securities settle on securities business days, payments and fiscal flows fall on Fedwire business days, and nothing moves on other days.
- Under `cbo_public_debt_target` debt follows CBO's path exactly and operating-cash settings move only the TGA; unless the scenario sets an issuance mix, this closure issues fixed gross shares rather than the bill-share rule; under `cbo_debt_reference_plus_tga_floor_financing_v1` debt follows the path and extra borrowing starts only when the TGA would go negative.
- The Fed rolls its maturing holdings over at the auctions settling the day they are paid, as add-ons in proportion to the amounts sold to the public, as far as its holdings path needs; on a falling path the rest runs off. Beyond that it meets its path by buying or selling existing securities at market value. It buys from domestic nonbanks first, then banks, then foreign holders, earliest maturity first; securities with the same maturity go in a fixed order of holder route and contractual terms, so the result does not depend on how holdings are split into rows.
- With holder position targets, after the Fed's operation each day holders above their target sell nominal bills to holders below it at market value, spread across remaining-maturity buckets in proportion to the seller's bills and divided among buyers in the order banks, foreign holders, domestic nonbanks, money funds. Transferred securities keep their terms and original issue proceeds, and redemptions go to the new holder. A run stops if a seller lacks bills, if the targets do not sum to the holdings the public has, if holdings differ from the baseline before the build starts, or if a routed account such as ON RRP would go negative.
- The ordinary fiscal flow paid to the public is CBO's primary deficit plus CBO's Table 4-3 Fed remittances (`CBOTable43RemittanceInFiscalFlow`), because CBO counts remittances as receipts; the opening fiscal year uses the amount remaining after observed DTS receipts at the fiscal splice. The TGA instead receives the remittance the run models (`CBORemittanceCashEffect`, also `CB_Remittance`), so a remittance below Table 4-3 is financed like any other cash need. Remittances have zero direct deposit or reserve effect.
- Fed net income (`CB_NetIncome`) is cash basis: the Fed's Treasury interest as the run pays it (bill discount at maturity, coupons, floating-rate interest, TIPS coupons and inflation compensation at maturity), agency MBS coupon income on the New York Fed's projected runoff path, and the Reserve Banks' other income, less interest on reserve balances, the ON RRP rate on all reverse repos, operating expenses and dividends (2025 levels growing 4 percent a year). IORB and the ON RRP rate are the run's own 3-month rate plus their spreads to the package curve at the opening (3.65 and 3.50 percent on June 18, 2026), so a short-rate scenario moves them. Reserve balances (`CB_ReserveBalances`) are the H.4.1 level of June 17, 2026, moved by the run's Treasury reserve effect, currency growth, MBS runoff and the Fed's non-Treasury income and expenses.
- Net income adds to `CB_EarningsRemittancesDue`, which opens at the H.4.1 line of that name: negative values are the deferred asset (`CB_DeferredAsset`, 236.7 billion at the opening). Nothing is remitted while it is negative; once net income has repaid it, the positive balance is paid to the TGA on the first Fedwire business day from each Wednesday. The rule is system-level: remittances by individual Reserve Banks while the system has a deferred asset, accrual-basis amortization, remunerated deposits other than reserves, valuation gains and losses, and any prepayment response of MBS to rates are not modeled.
- Holder-preference changes apply to new issuance from their effective date; the existing stock keeps its holders.
- Under `cbo_public_debt_target`, a day that requires negative issuance stops the run, unless the scenario selects at-par retirement of the shortest public marketable securities.

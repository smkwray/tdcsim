# CBO Downstream Quickstart

This is the public use path for downstream projects. You should not need to
understand the verifier internals to run scenarios.

## Inputs

Use one baseline package and its attestation:

```bash
output/cbo_forecast_release_bound_package.zip
output/cbo_forecast_release_bound_attestation.json
```

That baseline is the fixed CBO input package. Scenario files only describe what
you want to change.

## Generate Example Scenarios

```bash
python scripts/write_cbo_example_scenarios.py \
  --baseline output/cbo_forecast_release_bound_package.zip \
  --attestation output/cbo_forecast_release_bound_attestation.json \
  --output-dir /tmp/tdcsim-cbo-scenarios
```

This writes five ready-to-run examples:

- `00_baseline_noop.json`: run the baseline without scenario changes.
- `01_rates_inflation_frn_tips.json`: change the nominal curve, inflation, FRN benchmark, TIPS real yield, and real operating cash.
- `02_issuance_maturity_mix.json`: change issuance shares and maturity mix.
- `03_sector_holders.json`: change sector holder preferences from a future date forward.
- `04_fiscal_fed_cash.json`: change primary deficit scale, Fed stock target handling, cash residual, operating cash, and fiscal incidence.

For a short smoke run, add:

```bash
--start-date 2026-10-01 --end-date 2026-10-10
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
- `outputs/summary.json`: small run summary.
- `outputs/catalog.sqlite`: optional artifact catalog when requested by the scenario.

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
- Primary deficit, debt target, operating cash, Fed stock target, cash residual, and fiscal incidence assumptions.

## Boundaries

This lane is a CBO-baseline scenario runner. It does not claim to be the
official CBO model.

Keep these boundaries:

- CBO net interest is diagnostic only. It does not fund issuance.
- Operating cash and cash residual assumptions are not issuance plugs.
- Fed holdings are stock-target holder allocation mechanics, not Fed auction purchases.
- Fed remittances and deferred assets are not modeled in this lane.
- Holder-preference changes affect new issuance from their effective date forward; they do not force secondary-market rebalancing of the existing stock.

Those boundaries are why downstream projects can safely change scenario
assumptions without accidentally changing what the CBO lane claims to prove.

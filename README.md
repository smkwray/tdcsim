# TDCSim

TDCSim simulates the US Treasury funding chain—debt issuance, interest and principal payments, holder allocations, and the Treasury General Account (TGA)—and writes paths for the Treasury Deposit Component (TDC), bank reserves, debt stocks, financing costs, and weighted-average maturity (WAM). It runs either from a YAML scenario configuration or from a Congressional Budget Office (CBO) baseline package.

## Data

- Configuration runner (`run.py`): needs no external downloads when the opening portfolio is generated or derived from the config. Setting `initial_portfolio.mode` to `file` loads a local CSV or Excel security table.
- CBO baseline runner (`tdcsim-cbo`): reads a local CBO baseline package and verification JSON built from CBO Budget and Economic Projection workbooks, Treasury Fiscal Data (Monthly Statement of the Public Debt (MSPD), Daily Treasury Statement (DTS), Monthly Treasury Statement (MTS), Treasury auction results, TIPS CPI detail, and FRN daily indexes), New York Fed System Open Market Account (SOMA) holdings, and opening sector shares from WAMEST (derived from the Financial Accounts of the United States (Z.1)).
- Historical replay (`src/historical_replay.py` and `scripts/build_historical_replay_*.py`): rebuilds quarterly Treasury stocks and flows from local MSPD security detail, Treasury auction results, SOMA holdings, and Z.1 sector holdings.

No API keys are required. Local source data and generated baseline packages are excluded from the repository.

## Install

Requires Python 3.11 or newer. From a source checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt -e .
```

## Usage

### Configuration runner

Run a YAML scenario file (weekly steps by default):

```bash
python run.py tdc_config.yaml
python run.py tdc_config_optional.yaml
```

A configuration defines the fiscal path, yield curve, issuance mix, holder preferences, and dated events across scenario groups. `tdc_config_optional.yaml` exercises TIPS, floating-rate notes (FRNs), rate-sensitive demand, and financing-cost options. A scenario group writes its outputs only when every scenario in the group succeeds. See [scenario_overview.md](scenario_overview.md), [examples/optional_feature_scenarios.yaml](examples/optional_feature_scenarios.yaml), and [examples/calibration_notes.md](examples/calibration_notes.md) for the shipped configurations.

### CBO baseline runner

Validate, run, and verify a JSON scenario against a local baseline package and verification JSON:

```bash
tdcsim-cbo validate \
  --baseline output/cbo_forecast_release_bound_package.zip \
  --attestation output/cbo_forecast_release_bound_attestation.json \
  --scenario SCENARIO.json

tdcsim-cbo run \
  --baseline output/cbo_forecast_release_bound_package.zip \
  --attestation output/cbo_forecast_release_bound_attestation.json \
  --scenario SCENARIO.json \
  --output-dir OUT_DIR

tdcsim-cbo verify \
  --run-dir OUT_DIR \
  --baseline output/cbo_forecast_release_bound_package.zip \
  --attestation output/cbo_forecast_release_bound_attestation.json
```

Helper scripts and subcommands:

- `python scripts/run_cbo_forecast_smoke.py`: builds a local baseline package from CBO workbooks and Treasury files and runs one full-horizon baseline pass.
- `python scripts/verify_cbo_forecast_package.py`: checks the package inputs and recomputes debt drift against CBO's path.
- `python scripts/write_cbo_example_scenarios.py --baseline PACKAGE.zip --attestation ATTESTATION.json --output-dir DIR`: writes 23 runnable example scenarios.
- `python src/ratewall_input_builder.py --opening-book-only --help`: rebuilds the opening security book from local MSPD, SOMA, and WAMEST files.
- `tdcsim-cbo --help`: lists additional subcommands for compiling inputs, exporting forecast state, and comparing policy pairs.

The CBO runner steps daily on Treasury's Fedwire settlement calendar: 4-, 8-, and 17-week bills settle on Tuesdays; 6-, 13-, 26-, and 52-week bills on Thursdays; 3-, 10-, and 30-year nominal securities on the 15th; and 2-, 5-, 7-, and 20-year nominal securities, FRNs, and TIPS at month-end, rolling forward to the next Fedwire business day. Pending issuance closes on September 30 each year. By default, issuance targets the package's operating-cash path and fills bills toward a target share of marketable debt while splitting coupon tenors by trailing 12-month non-Fed auction shares; scenarios can also target CBO's public-debt path or a debt-reference path with a TGA floor. Federal Reserve holdings follow CBO's path through secondary-market trades rather than primary auctions, and a run fails if the TGA goes negative. See [examples/cbo_downstream_quickstart.md](examples/cbo_downstream_quickstart.md) for scenario fields and package layout.

### Reserve routing (TRC)

To route settlement events into a daily Treasury component of reserves (TRC) table, pass `--trc-routing PATH` to `tdcsim-cbo run` or add a `trc_routing` block in a YAML config. The routing JSON sets how foreign holders settle, the share of money market fund (MMF) flows settling through the Federal Reserve's overnight reverse repo facility (ON RRP), and opening balances in other Fed accounts.

## Output

Configuration runs write weekly CSV tables and plots under `output/`. CBO runs write a run manifest, compiled forecast inputs, and compressed daily tables under `outputs/` (dollar amounts in USD billions, maturities in years).

Main files written by a CBO run:

| File | Description |
|---|---|
| `outputs/results_compact.csv.gz` | Daily simulated macro series, in USD billions and years |
| `outputs/final_portfolio_compact.csv.gz` | Closing active security book |
| `outputs/tdcsim_period_tdc_summary.csv.gz` | Period TDC totals and component breakdown, in USD billions |
| `outputs/tdcsim_scenario_metrics.csv.gz` | Weighted-average maturity, bill share, and short-maturity share |
| `outputs/summary.json` | Run summary metrics |

Event-level issuance, redemption, interest, holder-stock, debt-bridge, and optional reserve-routing tables are written in the same `outputs/` directory.

Core columns in the daily results table:

| Column | Description |
|---|---|
| `TDC_Change` | Period TDC flow, in USD billions |
| `TDC_Level` | Cumulative TDC, in USD billions |
| `TGA` | Treasury General Account balance, in USD billions |
| `TotalDebt_Agg` | Total modeled Treasury debt outstanding, in USD billions |
| `FinancingCost_Cumulative` | Cumulative financing cost, in USD billions |
| `WAM` | Weighted-average maturity of modeled debt, in years |

Each period's TDC equals the sum of fiscal flows, debt service, primary auction absorption, secondary trading, and other flows, attributed by the account debited or credited at settlement (domestic nonbank deposits versus banks, the Federal Reserve, Treasury, and foreign accounts). The base reserve series routes reserve effects by holder; enabling TRC routing routes them by settlement account and checks daily closure against Fed nonreserve balances.

## Limits

- Yield curves are exogenous inputs. Changing issuance or holder mix changes rates only when the scenario also supplies a different curve.
- Outputs are accounting-consistent scenario paths rather than causal estimates or forecasts.
- Private holdings split into domestic nonbanks and MMFs, with MMF deposit pass-through defaulting to 0.97 unless overridden.
- TIPS inflation accretion enters cumulative financing cost by default in the CBO runner and is turned off in `tdc_config_optional.yaml`.
- In the configuration runner, secondary trading can reallocate holdings toward configured holder preferences once per step.

## Tests

The default suite runs offline on synthetic fixtures:

```bash
python -m pytest -q
```

Two optional pytest markers cover local-data and memory checks:

```bash
python -m pytest -q -m integration
python -m pytest -q -m memory_stress
```

`integration` requires local source or generated inputs and fails if they are absent.

## License

MIT ([LICENSE](LICENSE)).

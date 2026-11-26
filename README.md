# TDCSim

TDCSim simulates the US Treasury funding chain: debt issuance, interest and principal payments, who buys and holds the debt, and the Treasury General Account (TGA). It writes accounting-consistent paths for the Treasury Deposit Component (TDC), the part of the change in deposits at US banks and credit unions that comes from Treasury payments and trades in Treasury securities ([tdcest](https://github.com/smkwray/tdcest) measures the historical series), along with bank reserves, debt stocks, financing costs and weighted-average maturity (WAM).

It runs from a YAML scenario file (`run.py`, weekly steps) or from a local Congressional Budget Office (CBO) baseline package (`tdcsim-cbo`, daily steps). Yield curves are inputs, so the outputs are accounting scenarios, not forecasts.

## Data

- Configuration runner: needs no downloads when the config generates or derives the opening portfolio. Setting `initial_portfolio.mode` to `file` loads a local CSV or Excel security table.
- CBO baseline runner: reads a local baseline package and its attestation, a JSON file recording the commit, commands and hashes that built it. Packages combine CBO budget and economic projections; Treasury Fiscal Data (the Monthly Statement of the Public Debt (MSPD), Daily and Monthly Treasury Statements, auction results, consumer price index (CPI) detail for Treasury inflation-protected securities (TIPS), and floating-rate note (FRN) indexes); New York Fed System Open Market Account (SOMA) holdings; and opening holder shares from [WAMEST](https://github.com/smkwray/wamest), based on the Financial Accounts of the United States (Z.1).
- Historical replay (`src/historical_replay.py`, `scripts/build_historical_replay_*.py`): rebuilds quarterly Treasury stocks and flows from local MSPD, auction, SOMA and Z.1 files.

No API keys are needed. Source data and baseline packages are not in the repository.

## Install

Python 3.11 or later. From a source checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt -e .
```

## Usage

### Configuration runner

```bash
python run.py tdc_config.yaml
python run.py tdc_config_optional.yaml
```

A configuration sets the fiscal path, yield curve, issuance mix, holder preferences and dated events for groups of scenarios; `tdc_config_optional.yaml` adds TIPS, FRNs and rate-sensitive demand. A group writes outputs only if all its scenarios succeed. [scenario_overview.md](scenario_overview.md) and [examples/](examples/) describe the shipped configurations.

### CBO baseline runner

Validate, run and verify a JSON scenario against a baseline package and attestation, generated locally or supplied:

```bash
PKG=output/cbo_forecast_release_bound_package.zip
ATT=output/cbo_forecast_release_bound_attestation.json
tdcsim-cbo validate --baseline "$PKG" --attestation "$ATT" --scenario SCENARIO.json
tdcsim-cbo run --baseline "$PKG" --attestation "$ATT" --scenario SCENARIO.json --output-dir OUT_DIR
tdcsim-cbo verify --run-dir OUT_DIR --baseline "$PKG" --attestation "$ATT"
```

| Helper | What it does |
|---|---|
| `scripts/run_cbo_forecast_smoke.py` | Builds a local baseline from CBO and Treasury files in `output/cbo_forecast_smoke/` and runs it over the full horizon |
| `scripts/verify_cbo_forecast_package.py PACKAGE` | Checks a baseline package and its reported debt drift from CBO's path; `--write-zip` also zips it |
| `scripts/write_cbo_example_scenarios.py` | Writes 23 example scenarios for a package (`--baseline`, `--attestation`, `--output-dir`) |
| `src/ratewall_input_builder.py --opening-book-only` | Rebuilds the opening security book from local MSPD, SOMA, WAMEST and Treasury auction files named on the command line, pricing it on the input directory's yield curve |

Run the helpers with `python`; `tdcsim-cbo --help` lists the other subcommands. The CBO runner steps through calendar days, but Treasury cash moves only on Fedwire (Federal Reserve settlement system) business days: principal and interest due on a weekend or holiday are paid on the next business day. Each security settles on Treasury's auction calendar, as a new issue or a reopening, on a securities business day (which also excludes Good Friday), and everything pending settles on the last business day of each fiscal year. By default, issuance keeps the TGA on the package's operating-cash path, and a run fails if the TGA goes negative; the [CBO quickstart](examples/cbo_downstream_quickstart.md) covers other funding rules, scenario fields and package layout.

### Reserve routing (TRC)

To write a daily table of the Treasury component of reserves (TRC), the change in bank reserves caused by Treasury and Federal Reserve transactions, pass `--trc-routing PATH` to `tdcsim-cbo run` or add a `trc_routing` block to a YAML config. The routing JSON sets how foreign holders settle, the share of money market fund (MMF) flows that settle through the Fed's overnight reverse repurchase (ON RRP) facility, and opening balances in other Fed accounts. Without routing, reserve effects are assigned by holder; with it, they are assigned by settlement account and checked each day against Fed balances other than reserves.

## Output

Configuration runs write weekly CSV tables and plots under `output/`. A CBO run writes a manifest, compiled inputs and compressed daily tables to its run directory, with tables in `OUT_DIR/outputs/` in USD billions and years.

| File | Contents |
|---|---|
| `outputs/results_compact.csv.gz` | Daily simulated series |
| `outputs/final_portfolio_compact.csv.gz` | Closing security book |
| `outputs/tdcsim_period_tdc_summary.csv.gz` | TDC totals by period and by component |
| `outputs/tdcsim_scenario_metrics.csv.gz` | WAM, bill share and short-maturity share |
| `outputs/summary.json` | Run summary |

Event-level issuance, redemption, interest, holder and reserve-routing tables sit alongside them. Daily results include `TDC_Change`, `TDC_Level` (cumulative), `TGA`, `TotalDebt_Agg`, `FinancingCost_Cumulative` and `WAM`. Each period's TDC sums fiscal flows, debt service, auction purchases, secondary trading and other flows, assigned by the account debited or credited at settlement: domestic nonbank deposits versus banks, the Fed, the Treasury and foreign accounts.

## Limits

- Changing issuance or holder mix moves rates only when the scenario also supplies a different yield curve.
- Private holdings split into domestic nonbanks and MMFs. The share of MMF flows that moves bank deposits defaults to 0.97 unless a scenario overrides it.
- The CBO runner counts TIPS inflation accretion in financing cost by default; `tdc_config_optional.yaml` turns it off.
- In the configuration runner, secondary trading moves holdings toward holder preferences once per step.

## Tests

The default suite runs offline on synthetic fixtures:

```bash
python -m pytest -q
```

Optional markers cover local data (`integration`, which fails if inputs are absent) and memory:

```bash
python -m pytest -q -m integration
python -m pytest -q -m memory_stress
```

## License

MIT. See [LICENSE](LICENSE).

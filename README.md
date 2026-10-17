# TDCSim — Treasury Deposit Component scenarios

TDCSim is a stock-flow simulator for the Treasury funding chain. It compares how
issuance, debt service, holder allocation, and Treasury cash management change the
Treasury-attributed contribution to deposits, reserve balances, and the maturity
structure of debt. Its outputs are accounting-consistent scenarios, not causal
estimates or forecasts.

## Install and validate

Use Python 3.11 or newer from a source checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt -e .
python -m pytest -q
python -m tdcsim_cbo.cli --help
```

The default test suite uses synthetic fixtures. It includes scenario compilation,
execution, output verification, settlement accounting, and failure handling.
`pytest.ini` excludes the integration and memory-stress lanes from this command.

## Choose a runner

| Interface | Inputs | Use |
|---|---|---|
| `python run.py CONFIG.yaml` | YAML configuration; generated or supplied portfolio | Stylized scenario comparisons |
| `tdcsim-cbo` | Verified Congressional Budget Office (CBO) baseline package, attestation, and JSON scenario | Baseline-bound scenarios with independent verification |
| `historical_replay_*` Python modules | Declared historical source inputs | Materialization and accounting reconciliation |

The CBO interface consumes CBO inputs; it is not the official CBO model. Its
`forecast` filenames identify input and output contracts, not a forecasting claim.

### Configuration scenarios

```bash
python run.py tdc_config.yaml
# Alternative configuration with optional demand and financing-cost features:
python run.py tdc_config_optional.yaml
```

These commands run multi-year scenario groups and write local outputs. The shipped
configuration generates a stylized opening portfolio and compares issuance,
holder, and rate assumptions. See [scenario_overview.md](scenario_overview.md)
for the individual comparisons.

A scenario overrides the base fiscal path, yield curve, issuance mix, holder
preferences, or dated events. The opening portfolio is selected once per run.
`initial_portfolio.mode` accepts `generated`, `config_derived`, or `file`;
required malformed inputs and unknown taxonomy labels fail explicitly.

Each group publishes its results only if all its scenarios succeed. Output
settings control CSVs and plots. Optional demand elasticities and Treasury
Inflation-Protected Securities (TIPS) inflation-accretion accounting are described
in [optional_feature_scenarios.yaml](examples/optional_feature_scenarios.yaml)
and [calibration_notes.md](examples/calibration_notes.md).

### CBO baseline scenarios

Supply a baseline package and its attestation, then validate, run, and verify a
scenario. Generated packages and retained runs are not included in this repository.

```bash
tdcsim-cbo validate --help
tdcsim-cbo run --help
tdcsim-cbo verify --help
```

The [downstream quick start](examples/cbo_downstream_quickstart.md) gives the full
commands, example generator, input requirements, output columns, and supported
scenario controls. The package opening date must match the scenario start date.
A passing verification binds the outputs to their inputs and producer and checks
the declared accounting and replay requirements; it does not validate the economic
assumptions as predictions.

## Accounting and interpretation

**TDC** means Treasury Deposit Component. The model records the Treasury-attributed
deposit-settlement flow and its cumulative contribution, rather than the entire
deposit stock. **DU** is the covered domestic nonbank deposit-claim perimeter.
**RS** is its complement in the settlement system. Incidence follows the claim
debited or credited at settlement, not the legal sector of the holder. Existing
producer fields containing `ru_flow` retain their artifact names.

For exact flows measured on the same clock and perimeter, the accounting identity
is

$$\Delta D^P = F^{T,P} + N^P,$$

where $F^{T,P}$ is the Treasury-attributed deposit-settlement flow and $N^P$ contains
all other deposit changes. The coefficient on the exact flow is one. A conditional
projection coefficient estimated from public-data proxies is a separate object.
TDCSim supplies neither that estimate nor evidence identifying post-landing
retention.

The simulator decomposes its period contribution as

$$TDC_{Change}=TDC_{FiscalFlow}+TDC_{DebtService}+TDC_{AuctionAbsorption}+TDC_{SecondaryTrades}+TDC_{Other}.$$

| Component | Modeled mechanism |
|---|---|
| `TDC_FiscalFlow` | Net non-debt-service fiscal payments |
| `TDC_DebtService` | Interest and principal paid through the configured settlement routes |
| `TDC_AuctionAbsorption` | Deposit effect of new Treasury issuance |
| `TDC_SecondaryTrades` | Deposit effect of modeled secondary-market transfers |
| `TDC_Other` | Other explicitly modeled Treasury and central-bank flows |

The decomposition is checked row by row. Bills settle at cash proceeds rather
than face value. TIPS and floating-rate notes (FRNs) carry their own indexation
and payment conventions. The Treasury General Account (**TGA**) is the model's
tracked proxy for Treasury operating cash, so unmodeled cash accounts remain a
boundary of the calculation.

### Treasury component of reserves

The opt-in Treasury component of reserves (**TRC**) uses the same Treasury events
with a separate reserve-settlement route registry. Enable it with `trc_routing`
in the YAML runner or `--trc-routing PATH` in the CBO runner.

The policy must declare foreign settlement routes, the sourced share of money
market fund (MMF) flows settled through overnight reverse repurchase agreements
(**ON RRP**), and opening Federal Reserve nonreserve balances. MMF deposit
pass-through sets deposit incidence; it does not determine reserve incidence.
The daily ledger checks closure and rejects negative route balances.

TRC is emitted separately in `tdcsim_trc_results` and
`tdcsim_trc_metadata.json`. The legacy `Reserves` column is a deprecated
holder-routed Treasury path. It excludes non-Treasury reserve factors and is
neither an observed nor a forecast reserve stock.

## Outputs and downstream use

Configuration runs write CSVs and plots under `output/`. CBO runs write a manifest,
compiled inputs, compact results, final portfolios, settlement-flow tables, and
verification evidence under the requested run directory. The downstream quick
start documents their units and accounting bases.

| Result | Meaning |
|---|---|
| `TDC_Change` | Modeled period Treasury-attributed deposit contribution |
| `TDC_Level` | Opening level plus the cumulative modeled contribution |
| `TGA` | Modeled Treasury General Account balance |
| `TotalDebt_Agg` | Outstanding modeled debt |
| `FinancingCost_Cumulative` | Cumulative financing cost under the selected options |
| `WAM` | Weighted-average maturity of the active portfolio |

[RateWall](https://github.com/smkwray/ratewall) consumes verified scenario exports.
Its deposit-support inputs distinguish cashflow overlap and MMF pass-through;
consumers must preserve those bases rather than apply the adjustments twice.
[TDCest](https://github.com/smkwray/tdcest) owns public-data measurement and
[EA-TDC](https://github.com/smkwray/ea-tdc) owns conditional deposit projections.
Simulation outputs do not replace either evidence chain.

Local `data/`, `output/`, `plots/`, environments, caches, and build products are
ignored. Keep source inputs, their manifests, the executable producer identity,
and output verification together when retaining a run. No downloaded baseline or
historical source data is required for the default synthetic tests.

## Limits

- Yield curves are exogenous. A holder-allocation change does not itself identify
  a rate response; any coupled rate path is an explicit scenario assumption.
- Generated portfolios are stylized. `config_derived` aligns the opening stock
  with configuration assumptions but does not reconstruct the actual Treasury
  portfolio.
- Preference trading is a simplified reallocation mechanism, not a market
  microstructure model. Holder assumptions and fiscal incidence require their own
  justification.
- The configuration runner's weekly stepping and the CBO runner's daily calendar
  have different timing contracts. Coarse configuration frequencies are
  approximate and controlled by `coarse_frequency_action`.
- TIPS inflation accretion enters financing cost only when the corresponding
  option is enabled.
- CBO net interest is diagnostic; it does not finance issuance. Operating cash is
  a scenario proxy, and Federal Reserve holdings use stock-target allocation.
- Accounting closure establishes internal consistency, not causal identification,
  a deposit multiplier, or an empirical retention ratio.

Additional test lanes require their declared inputs or an isolated memory budget:

```bash
python -m pytest -q -m integration
python -m pytest -q -m memory_stress
```

Missing required integration inputs fail rather than skip. These lanes are
separate from a lightweight source-checkout validation.

## License

[MIT](LICENSE)

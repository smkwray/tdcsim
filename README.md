# TDCSim

TDCSim simulates the U.S. Treasury's funding chain: debt issuance, interest and
principal payments, who holds the debt, and the Treasury's cash balance. For each
scenario it reports the Treasury Deposit Component (**TDC**), which is the
Treasury's contribution to bank deposits. It also reports the Treasury's
contribution to bank reserves, the Treasury General Account (**TGA**), and the
maturity structure of the debt. Use it to compare scenarios that change issuance,
holders, interest rates or the fiscal path.

## Install and test

Python 3.11 or newer, from a source checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt -e .
python -m pytest -q
```

The default suite runs on synthetic fixtures and needs no downloaded data. Two
further lanes need local inputs or a dedicated memory budget:

```bash
python -m pytest -q -m integration     # requires local source data; fails if absent
python -m pytest -q -m memory_stress
```

## Runners

| Runner | Entry point | Time step | Opening portfolio |
|---|---|---|---|
| Configuration | `python run.py CONFIG.yaml` | Weekly by default | Generated, derived from the configuration, or loaded from a file |
| CBO baseline | `tdcsim-cbo` | Daily | Security-level book built from Treasury and Federal Reserve data |
| Historical replay | `historical_replay*` modules in `src/` | Quarterly | Treasury's published security history |

### Configuration runner

```bash
python run.py tdc_config.yaml
python run.py tdc_config_optional.yaml   # adds demand elasticities and financing-cost options
```

A configuration sets the fiscal path, yield curve, issuance mix, holder
preferences and dated events, then runs groups of scenarios against each other.
[scenario_overview.md](scenario_overview.md) describes the shipped comparisons,
and [optional_feature_scenarios.yaml](examples/optional_feature_scenarios.yaml)
with [calibration_notes.md](examples/calibration_notes.md) cover the optional
features. `initial_portfolio.mode` selects the opening stock: `generated` (a
stylized portfolio), `config_derived` (consistent with the configured curve and
issuance mix), or `file`. A group writes results only if every scenario in it
succeeds.

### CBO baseline runner

The CBO runner takes the Congressional Budget Office (CBO) baseline as its fiscal
and rate path. It reads a baseline package and attestation, compiles a JSON
scenario against them, runs it and verifies the outputs:

```bash
tdcsim-cbo validate --help
tdcsim-cbo run --help
tdcsim-cbo verify --help
```

The [downstream quick start](examples/cbo_downstream_quickstart.md) covers package
inputs, example scenarios, output files and scenario controls. Each step:

- **Issuance.** Debt held by the public follows CBO's projected path, interpolated
  to the day. New securities settle on the day the path requires them, split by
  the scenario's security-type shares and maturity distribution. The runner
  issues on that daily need, without Treasury's auction calendar.
- **Holders of new issuance.** A holder-preference profile divides each security
  type among banks, foreign holders and private holders. The package ships a
  default profile; scenarios can replace it from a chosen date.
- **Federal Reserve.** Fed holdings follow CBO's projected path. The Fed reaches
  it by buying or selling existing securities from other holders at market
  value; it receives no allocation of new issuance.
- **Payments.** Coupons, principal, floating-rate note (FRN) interest and
  Treasury Inflation-Protected Securities (TIPS) inflation adjustments are paid
  on each security's own dates to its current holder.
- **Cash.** The TGA opens at its Daily Treasury Statement balance. Every cash flow
  is booked on both sides, and a run fails if the TGA goes negative.

**Opening portfolio.** `src/ratewall_input_builder.py` builds the opening book
from Treasury's Monthly Statement of the Public Debt (MSPD). It takes every
marketable security outstanding on the record date (about 460 in May 2026) with
its own coupon and maturity date; the CBO package build adds each TIPS index
ratio and FRN spread from Treasury data. Federal Reserve holdings come security
by security from the New York Fed's System Open Market Account (SOMA) holdings.
The rest of each security is divided among banks, foreign holders, money market
funds (MMFs) and other domestic holders in proportion to each group's holdings of
bills or of longer securities, as estimated by
[WAMEST](https://github.com/smkwray/wamest) from Federal Reserve Financial
Accounts (Z.1) and other public data. Securities that mature between the record
date and the simulation start roll into 3-month bills.

### Historical replay

The replay modules rebuild the historical Treasury stock quarter by quarter from
MSPD security detail and auction results. They allocate each security to Z.1
sectors with SOMA holdings fixed by security, accrue FRN and TIPS interest, and
reconcile sector holdings to Z.1 totals. The forward runners build their own
opening state as described above.

## Accounting

The simulator decomposes each period's TDC as

$$TDC_{Change}=TDC_{FiscalFlow}+TDC_{DebtService}+TDC_{AuctionAbsorption}+TDC_{SecondaryTrades}+TDC_{Other}.$$

| Component | Source |
|---|---|
| `TDC_FiscalFlow` | Spending and receipts other than debt service |
| `TDC_DebtService` | Interest and principal paid to holders |
| `TDC_AuctionAbsorption` | Cash paid for new issuance |
| `TDC_SecondaryTrades` | Modeled trades between holders, including Fed purchases and sales |
| `TDC_Other` | Other modeled Treasury and central-bank flows |

The decomposition is checked row by row. Each flow is attributed to the account
debited or credited at settlement. **DU** is the deposit perimeter: deposit
claims of domestic nonbank holders. **RS** is the rest of the settlement system.
Holders are banks, the Federal Reserve (`CB`), foreign holders and private
holders. Private holdings are split into domestic nonbanks and MMFs, and MMF
flows reach deposits at a pass-through rate that defaults to 0.97. Bills settle
at cash proceeds; TIPS and FRNs follow their own indexation and payment rules.

**Treasury component of reserves (TRC).** The same events can also be routed to
bank reserves. Enable this with `trc_routing` in a configuration or
`--trc-routing PATH` in the CBO runner. The routing file declares how foreign
holders settle, the share of MMF flows that settles through the Fed's overnight
reverse repo facility (ON RRP), and opening balances in other Fed accounts. A
daily ledger checks that the routes close. Results are written to
`tdcsim_trc_results` and `tdcsim_trc_metadata.json`.

## Outputs

Configuration runs write CSVs and plots under `output/`. CBO runs write a
manifest, compiled inputs, daily results, the final portfolio, issuance, principal
and payment flow tables, and verification records to the run directory.

| Column | Meaning |
|---|---|
| `TDC_Change` | Period TDC |
| `TDC_Level` | Opening level plus cumulative TDC |
| `TGA` | Treasury General Account balance |
| `TotalDebt_Agg` | Outstanding modeled debt |
| `FinancingCost_Cumulative` | Cumulative financing cost |
| `WAM` | Weighted-average maturity of the outstanding debt |
| `DebtHeld_*` | Holdings by holder |

`Reserves` is an older series that routes reserve effects by holder; TRC
supersedes it. Columns containing `ru_flow` hold RS flows.

## Model assumptions

- Yield curves are inputs. A change in holders or issuance moves rates only when
  the scenario also changes the curve.
- CBO net interest is reported for comparison. Issuance follows the debt path.
- The configuration runner's secondary trading reallocates holdings toward
  configured holder preferences once per step.
- TIPS inflation accretion enters financing cost when
  `financing_cost_options.include_tips_inflation_accretion` is on.

## Related projects

- [RateWall](https://github.com/smkwray/ratewall) consumes verified CBO-runner
  exports.
- [WAMEST](https://github.com/smkwray/wamest) estimates the maturity mix of
  Treasury holdings by sector and supplies the opening holder split.
- [TDCest](https://github.com/smkwray/tdcest) measures the TDC from public data.
- [EA-TDC](https://github.com/smkwray/ea-tdc) projects deposits conditional on
  the TDC.

## License

[MIT](LICENSE)

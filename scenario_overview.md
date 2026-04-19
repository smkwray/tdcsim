# Simulation Scenarios: Overview and Design

This document describes the scenarios configured in `tdc_config.yaml` and the modeling assumptions driving their results.

**Important:** Yield curves are exogenous scenario inputs. The model does not endogenize rates from demand. Statements like "more bank demand lowers rates" require an explicit yield curve change, not just a holder preference shift.

## Treasury Market Scenarios (3 scenarios)

### 1. Baseline
- **Definition**: Status quo for Treasury issuance and market demand.
- **Yield Curve**: Standard upward-sloping (`yield_curve_default`).
- **Holder Mix**: Historical average allocation (`sector_preferences_default`).
- **Issuance Mix**: Default profile — 20% bills, 60% notes, 20% bonds (`default_issuance_profile`).
- **Purpose**: Benchmark for comparison.

### 2. BanksBuyFromPriv_25bpsLowerRates
- **Mechanism**: Banks absorb a larger share of new issuance (especially bills and notes), combined with a parallel yield curve shift down by ~25bps.
- **Why Outlays Fall**: Lower rates reduce coupon costs on new issuance. The exogenous rate reduction is applied directly via `yield_curve_lower`.
- **Why Deposits Rise**: When banks absorb more Treasury issuance, they avoid the direct DU deposit drain that occurs when domestic nonbanks (`Private`) buy the same issuance. The holder shift from `Private` to `Banks` therefore raises TDC relative to the nonbank-funded case, even without treating primary bank purchases as literal deposit creation.

### 3. MaturityShift_MoreNotes
- **Mechanism**: A combined change — the Treasury shifts issuance toward 2–10y notes (away from long bonds), and banks are assumed to have a stronger preference for notes.
- **Why Outlays Fall**: In the upward-sloping yield curve, 5y notes carry lower coupons than 30y bonds. Shifting issuance to the belly of the curve reduces interest cost.
- **Why Deposits Rise**: The holder preference shift (`sector_preferences_banks_pref_notes`) tilts auction absorption toward banks in the note bucket.
- **Note**: This scenario bundles two mechanisms. The isolation scenarios below decompose them.

## One-Factor Isolation Scenarios (5 scenarios)

These scenarios change exactly one mechanism at a time relative to `Factorial_Baseline`, enabling ceteris paribus attribution. This is a one-factor-at-a-time (OFAT) design, not a full factorial grid.

**Limitation:** OFAT isolation supports statements like "holding X and Y fixed, changing Z does W." It does not identify interaction effects. If issuance mix and holder preferences interact (plausible since preferences operate over maturity buckets), the combined effect may differ from the sum of isolated effects.

### Factorial_Baseline
- Identical overrides to the main Baseline. Serves as the control for all isolation comparisons.

### Factorial_IssuanceOnly
- Changes only `treasury_issuance_profile` (to `policy_issuance_more_notes_fewer_bonds`).
- Isolates the pure effect of shifting the issuance maturity mix.

### Factorial_HolderOnly
- Changes only `sector_preferences` (to `sector_preferences_banks_buy_more`).
- Isolates the effect of banks absorbing a larger share of all maturities.

### Factorial_RateOnly
- Changes only `yield_curve` (to `yield_curve_lower`, ~25bps parallel down-shift).
- Isolates the pure effect of lower borrowing costs.

### Factorial_HolderOnly_NotesPref
- Changes only `sector_preferences` (to `sector_preferences_banks_pref_notes`).
- Isolates the holder shock embedded in `MaturityShift_MoreNotes` (banks prefer notes).

## Decomposition Logic

The isolation scenarios allow the following attribution for the compound scenarios:

- `MaturityShift_MoreNotes` ≈ `Factorial_IssuanceOnly` + `Factorial_HolderOnly_NotesPref` (plus any interaction)
- `BanksBuyFromPriv_25bpsLowerRates` ≈ `Factorial_HolderOnly` + `Factorial_RateOnly` (plus any interaction)

The "plus any interaction" caveat means these are approximate decompositions, not exact additive ones.

## Known Model Limitations

The following are known simplifications in the current simulator. They do not affect the shipped scenario results under the default configuration (weekly frequency, zero TIPS inflation, static yield curves, no FRN issuance), but they constrain the model's applicability outside those settings.

1. **By default, TIPS inflation accretion is excluded from financing cost.** When `cpi_annual_inflation > 0`, TIPS principal grows via index-ratio adjustment, but this accretion is not recorded in `FinancingCost_Period` unless `financing_cost_options.include_tips_inflation_accretion` is enabled. The shipped configuration leaves it off because baseline inflation is zero.

2. **Coupon and accrual logic assumes sub-semiannual step frequency.** The engine pays at most one coupon per tranche per step. For the shipped weekly frequency this is always correct (a weekly step never spans two semiannual coupon dates). Coarser frequencies (monthly or longer) could miss coupons. FRN accrual similarly uses the step's benchmark rate across the full interval rather than resetting on quarterly calendar dates.

3. **The legacy initial portfolio is stylized.** The default `generated` portfolio uses hard-coded maturity menus, coupon ranges, and TIPS reference CPI values that are independent of the scenario's yield curve and issuance profile. Using `initial_portfolio.mode: config_derived` makes the starting stock internally consistent with the configured curve and issuance profile, but it is still a model-generated approximation rather than a reconstruction of the real Treasury stock.

4. **Secondary trading is single-pass per period.** When preference trading is enabled, the engine executes at most one bond trade per seller/buyer/category pair per period. Large imbalances may take multiple periods to resolve. This is a conservative approximation that understates rebalancing speed. (Preference trading is disabled in the shipped configuration.)

5. **Nonmarketable interest crediting is validated for weekly steps only.** The crediting logic checks June 30 and December 31 within the current step's year. A step that crosses a year boundary could miss a credit date. Weekly steps never cross year boundaries within a single step.

6. **Loaded portfolio coercion.** When using `mode: file`, unknown security types are silently converted to `Fixed` and unknown holder types to the default holder. The shipped configuration uses `mode: generated` and is not affected.

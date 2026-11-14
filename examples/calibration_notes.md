# Calibration Notes

## Optional demand calibration

[tdc_config_optional.yaml](../tdc_config_optional.yaml) uses a modest auction-only
demand calibration:

- `Banks.bills.yield_beta = 30.0`
- `Banks.bills.spread_beta = 15.0`
- `Private.bills.yield_beta = -12.0`
- `secondary = {}`

The values were tuned in a short weekly run that matched the optional
configuration's issuance mix, enabled rate-sensitive demand only for bill auction
absorption, and left preference trading off. In that run the mean absolute
auction-demand shift was about `0.0022`, and banks absorbed visibly more bills
than in the baseline.

The elasticity applies only at auction because auction shares map directly onto
the preference columns the engine already uses. Secondary-trading targets are
coarser and keep their configured preferences. The coefficients are scenario
settings chosen to produce moderate comparative statics.

## Source-backed input builder

Choices made in `src/ratewall_input_builder.py`:

- Opening book: Federal Reserve holdings come from SOMA by CUSIP. The non-Fed
  remainder of each security is split by WAMEST sector holdings of bills or of
  longer securities. Depository institutions, including credit unions, route to
  `Banks`; money market funds route to the MMF subbucket of `Private`; all other
  domestic sectors route to the domestic nonbank subbucket.
- New-issuance holder shares blend tdcmix holder priors with recent Z.1 absorption.
  The Z.1 weight is `n_eff / (n_eff + 8.0)`, where `n_eff` counts the usable
  quarters among the most recent eight.
- `Banks` in Z.1 means banks only. Primary-market dealer purchases are
  redistributed by the baseline holder shares.
- Auction-allotment composition enters instrument-specific shares with weight
  `0.20`. Public allotment files identify only dealers, the Fed and foreign
  official buyers, so the overlay is share-weighted.
- Holder-shift scenario intensity is set by `holder_absorption_calibration`, with
  grid `[0.25, 0.50, 0.75, 1.00]` and default `0.50`.
- Synthetic bill proceeds use a price-ratio floor of `0.75` to guard against
  malformed yield or maturity inputs.

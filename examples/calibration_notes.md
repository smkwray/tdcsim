# Optional Demand Calibration Notes

The alternate config in [tdc_config_optional.yaml](/Users/shanewray/malus/proj/tdcsim/tdc_config_optional.yaml) uses a deliberately modest auction-only demand calibration:

- `Banks.bills.yield_beta = 30.0`
- `Banks.bills.spread_beta = 15.0`
- `Private.bills.yield_beta = -12.0`
- `secondary = {}`

These values were chosen to keep the feature in a "noticeable but not mode-changing" range under a short weekly calibration harness:

- configuration shape matched the optional config's issuance mix
- rate-sensitive demand was enabled only for bill auction absorption
- preference trading was left off to isolate the auction channel

In that harness, the selected coefficients produced:

- mean absolute auction-demand shift of about `0.0022`
- visibly higher bank bill absorption than the legacy baseline
- no dependence on endogenous rates or iterative market clearing

Why auction-only:

- auction elasticities map cleanly onto the repo's existing column-sum preference system
- secondary-trading targets in the current model are more approximate and are better left unchanged unless that subsystem is redesigned more explicitly

Interpretation:

- the numbers are scenario coefficients, not empirical estimates
- they are meant to generate moderate comparative statics, not to imply a fitted demand curve

## Source-Backed RateWall Input Builder Notes

These notes document judgment calls in `src/ratewall_input_builder.py`; they are not recalibrations.

- `z1_recent` blend weight `0.50`: equal-weight shrinkage keeps the tdcmix prior live while allowing the latest Z.1 positive-absorption panel to move the central holder mix.
- primary-market instrument blend weight `0.20`: auction-allotment composition informs instrument-specific shares without overriding the broader stock/flow holder priors.
- holder-shift scenario blend weight `0.60`: sensitivity scenarios are intentionally more responsive to tdcmix upper/lower priors while retaining continuity with the central baseline.
- MMF rows collapsed into `Private`: the current tdcsim holder perimeter has no source-backed split for MMF cash-fund routing, so the central path flags the known bias rather than inventing a split.
- dealers mapped to `Banks`: primary-dealer auction allotments are treated as reserve-user settlement bridge activity, not as final holder evidence.
- bill price-ratio floor `0.75`: the floor prevents malformed or extreme yield/maturity inputs from generating implausibly low synthetic bill proceeds in source-backed cohort construction.

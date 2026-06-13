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

- Z.1 shrinkage is now grounded by the Phase 4 T4.1 ingest: `w_z1 = n_eff / (n_eff + 8.0)`. `n_eff` is the winsorized information content of the most recent eight eligible calendar quarters after finite `pos_abs_share_*` filtering, so the trailing all-NaN 2026Q1 panel row no longer silently turns an eight-quarter rule into seven real quarters.
- Z.1 Banks means banks only. `dealer_bridge` is excluded from the final-holder vector and only reduces recent-quarter information content through the core-share rule.
- primary-market instrument blend weight `0.20`: auction-allotment composition informs instrument-specific shares without overriding the broader stock/flow holder priors.
- Auction dealers are routed to an internal `dealer_bridge` and redistributed by the pre-overlay baseline holder shares, not mapped to Banks. Current public primary-market files only expose `dealers`, `fed`, and `foreign_official`, so the overlay is stamped low-coverage and share-weighted.
- holder-shift scenario intensity is YAML-controlled under `holder_absorption_calibration`, with grid `[0.25, 0.50, 0.75, 1.00]` and default `0.50`.
- TIC foreign cross-check is deferred: it is new QA-only plumbing, not a central calibration input for T4.1.
- MMF rows collapsed into `Private`: the current tdcsim holder perimeter has no source-backed split for MMF cash-fund routing, so the central path flags the known bias rather than inventing a split.
- bill price-ratio floor `0.75`: the floor prevents malformed or extreme yield/maturity inputs from generating implausibly low synthetic bill proceeds in source-backed cohort construction.

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

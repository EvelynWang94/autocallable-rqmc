# Main experiment presentation guide

## Primary figure

Use `accuracy_cost_and_interaction.png`. It combines the report's estimator-selection result with the explicit interaction test. A point is circled only when no other tested method/path-count cell is at least as accurate, stable and fast.

## Supporting figures

- `cost_normalized_efficiency.png`: implements the Glynn-Whitt variance-cost comparison and empirical time-to-target view.
- `fair_coupon_regime_sensitivity.png`: shows economic sensitivity and replication uncertainty.
- `trigger_smoothing_regime_map.png`: implements the Wang-aligned local discontinuity diagnostic.
- `sampling_monitoring_approximation_evidence.png`: prevents unlike error sources from being combined.
- `greek_time_to_target.png`: shows why near-KI and near-autocall Greeks must be discussed separately.

## Claim boundaries

- The high-budget M3 value is a sampling reference, not an issuer or dealer quote.
- M3's trivariate approximation warning remains the separate Day 6 audit.
- AC-Smooth is local to the next-observation autocall principal and is not a full-product replacement.

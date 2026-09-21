# Distribution validation - 20 September 2026

- Original regression suite: **25 passed** in 68.21 seconds on the local
  Python environment. GPU tests executed successfully. CuPy emitted a warning
  that `CUDA_PATH` was not detected, but the actual GPU checks passed.
- Synthetic CPU smoke run: M0/M1/M2/M3, 256 paths and two replications each.
  Cash-flow identity, probability-mass and coupon-root checks passed.
- Saved-evidence check: 1,112 Day 9 replication records with the expected
  table sizes; reference mean, SD and SE reproduced from saved replications.

This validation does not rerun the report's full experimental matrix or claim
to remeasure its speed-ups. The archived `gate_summary.csv` files describe
historical experiments. GitHub Actions results, when available, are separate
from these local checks.

The exact local package versions are recorded in `validation-environment.json`.

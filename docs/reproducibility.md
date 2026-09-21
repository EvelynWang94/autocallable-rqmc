# Reproduction guide

## Three levels of reproduction

1. **CPU demonstration and regression tests:** install `.[dev]`, run
   `examples/quickstart.py` and `python -m pytest -q`. Synthetic data; no workbook.
2. **Recheck saved evidence:** run `python scripts/verify_results.py`. This reads
   archived CSVs, checks the 256 + 24 + 352 + 480 row inventory, and recomputes
   reference mean, sample SD and SE. It is not a fresh simulation.
3. **Rerun original experiments:** supply the authorised frozen workbook as
   described in `Data/README.md`, install notebook dependencies and start
   `python -m jupyterlab` from the repository root. Run notebooks in numerical
   order, with Day 3B after Day 3. Day 8 requires a supported NVIDIA GPU and
   a CuPy package matching the locally installed CUDA runtime. Day 9 selects
   its backend using the existing experiment implementation.

All notebooks discover the repository root from their working directory or its
parents. Day 1-5 also support `AP_PROJECT_ROOT`. The original notebook names are
retained for comparison with the report and historical configuration.

## Notebook sequence

| Notebook | Purpose |
| --- | --- |
| Day 1 | Contract layers, deterministic paths, cash-flow schema |
| Day 2 | European options and Black-Scholes validation |
| Day 3 | One-step survival-conditioning pilot |
| Day 3B | Exact single-asset Brownian-bridge benchmark |
| Day 4 | Lee-style exit probability reproduction and failed OOS surrogate gate |
| Day 5 | Direct autocallable pricing and fair coupon |
| Day 6 | M2/M3 bridge-conditioned pricing and approximation audit |
| Day 7 | Greeks, bump diagnostics and local trigger smoothing |
| Day 8 | GPU implementation and numerical equivalence |
| Day 9 | Main matrix, reference, regime analysis and Pareto frontier |

Generated runs write to ignored `outputs/`; committed evidence is in `results/`.
These locations are intentionally distinct so a quick experiment does not
overwrite the published evidence. Day 9 resumes checkpoints within `outputs/`:
use a fresh output directory after changing its budgets, seeds or model inputs.
Never mix checkpoints from different experiment definitions.

## What is preserved and changed

- All six source modules, five original test files and three current configs
  are copied byte-for-byte from the research project.
- Notebook code is preserved apart from repository-root discovery and removal
  of personal absolute paths. Execution outputs/counts are cleared.
- CSVs and figures preserve research evidence; local paths in text manifests
  are replaced with descriptive placeholders. Historical `audit_inventory.csv`
  hashes refer to original files, not transformed repository-distribution files.
- The report retains all 26 pages and its AI-use disclosure; the cover student
  ID is redacted and PDF metadata is reset.
- Raw workbooks, third-party papers, caches, drafts and legacy notebooks are
  excluded. The older project plan and dated status remain historical records.

The dependency ranges in `pyproject.toml` are installation requirements, not a
reconstruction of the original locked environment. `validation-environment.json`
records this packaging run. Consult the report for the limits of original
environment retention and hardware timing portability.

## Numerical boundaries

Replication uncertainty is estimated across independent seeds or Sobol
scrambles. Within a Greek replication, base and bumped values share random
numbers and contractual references remain fixed. Direct finite-grid monitoring,
finite nested bridge estimation and probability projection introduce different
errors; increasing outer sample size alone does not remove all of them.

The final M3 reference is independently randomised but shares the estimator's
approximation structure. Earlier-stage fair coupons correspond to different
calibrations and budgets. The isolated autocall-redemption smoother omits other
cash flows and must not be presented as a complete valuation method.

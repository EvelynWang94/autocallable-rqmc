# Applied Project Current Status

Updated: 2026-08-11

The historical `PROJECT_FREEZE.md` records the 2026-07-30 state. This file is the
current gate status after adding the Day 1 contract layer, the Day 3B exact
Brownian-bridge benchmark, the Day 5 direct autocallable engine, the Day 6
three-asset Brownian-bridge-conditioned M2/M3 estimators, the Day 7
finite-difference Greeks and discrete-trigger smoothing diagnostics, the Day 8
CUDA implementation and CPU/GPU equivalence audit, and the Day 9 main
experiment matrix and cost-normalized estimator comparison.

| Workstream | Status | Current evidence |
| --- | --- | --- |
| Day 1 — market case / RC-L / RC-A layering | **PASS** | `Day1_Contract_Layers_Component_Schema.ipynb`; 10/10 deterministic paths pass; maximum component identity error 0; known fair coupon 0.075 recovered to floating-point precision. |
| Day 2 — European engine and MC/RQMC/Greeks | **PASS (existing evidence retained)** | `Day2_HSBC_M1_RQMC_European_Validation.ipynb`; executed notebook and `outputs/day2_hsbc_m1_rqmc/`. |
| Day 3A — one-step survival pilot | **PASS (existing evidence retained)** | `Day3_Barrier_Conditioning_MC_RQMC_Greeks.ipynb`; executed notebook and `outputs/day3_barrier_conditioning/`. |
| Day 3B — single-asset exact Brownian bridge | **PASS** | `Day3B_Single_Asset_Exact_Brownian_Bridge_Validation.ipynb`; complete M × N grid, exact Bernoulli and conditional-weight bridge estimators, fine-grid MC, analytic benchmark, and Greek bump diagnostics. |
| Day 4 — Lee exit-probability reproduction | **LIMITED / FAIL** | `Day4_Lee_Exit_Probability_Reproduction.ipynb`; all gates except approximation bias pass. OOS q3 RMSE 0.003706 exceeds the frozen 0.002500 target. |
| Day 5 — three-asset direct autocallable and fair coupon | **PASS** | `Day5_Three_Asset_Direct_Autocallable_Fair_Coupon.ipynb`; reusable RC-L/RC-A parser and direct engine, Lee Tables 7/8 checks, component/probability identities, RC-A fair coupon and four required sensitivity dimensions. |
| Day 6 — three-asset Brownian-bridge conditioning | **PASS** | `Day6_Three_Asset_BB_Conditioning_M2_M3.ipynb`; M0–M3 price agreement, exact `g_i`/`h_ij`, fresh nested `q3` fallback, segment/log-weight diagnostics, paired Bernoulli bridge, probability bounds, invariance and approximation-bias audits. |
| Day 7 — Greeks and discrete-trigger smoothing | **PASS WITH CAVEATS** | `Day7_Greeks_and_Discrete_Trigger_Smoothing.ipynb`; three component Deltas/Vegas/Gammas, parallel Vega/Gamma, complete bump grids, CRN, near-KI/near-autocall separation, bridge-seed sensitivity and local AC-Smooth diagnostic. Pure bump-shape Gamma plateau coverage is only 12.5%; every non-plateau estimate is retained. |
| Day 8 - CUDA implementation and numerical equivalence | **PASS** | `Day8_GPU_Implementation_and_Validation.ipynb`; float64 shared-input M0/M1, batched CRN Greeks, hybrid GPU M2/M3, cold/steady timing separation, measured break-even, 9/9 validation gates and 19/19 regression tests. |
| Day 9 - main experiments and Pareto analysis | **PASS** | `Day9_Main_Experiments_and_Pareto_Frontier.ipynb`; 1,112 checkpointed replications, high-budget M3 reference, cost-normalized efficiency, interaction diagnostic, regime and trigger maps, 10/10 gates and 25/25 regression tests. |

## Day 1 evidence

- Frozen layers: HSBC market case, RC-L, and RC-A.
- Continuous knock-in and discrete autocall event clocks are separate.
- Spot-bump policy keeps the strike and absolute barriers fixed.
- Fair-coupon accrual and valuation-scope exclusions are explicit.
- The component schema contains coupon value, early-redemption principal,
  surviving notional, no-KI maturity redemption, terminal KI loss, total value,
  observation-date redemption probability, continuous-KI survival probability,
  and fair coupon.
- Audit outputs: `outputs/day1_contract_layers/`.

## Day 3B evidence

- Continuous analytic down-and-out call price: **4.916409595**.
- Maximum absolute conditional-weight price bias on the required M grid at
  N = 65,536: **0.008423**.
- Direct RQMC monitoring bias: **1.712233** at M = 12 and **0.315855** at M = 504.
- Required monitoring grid: 12, 24, 52, 104, 252, 504.
- Required sample-size grid: 1,024, 4,096, 16,384, 65,536.
- Conditional-weight Delta maximum analytic-FD error: **0.008690**.
- Conditional-weight Vega maximum analytic-FD error: **0.003061** per vol point.
- Weighted Gamma did not meet the strict plateau threshold; the full bump grid,
  replication SD, and instability diagnostic are retained, satisfying the plan's
  “plateau or document instability” condition.
- Audit outputs: `outputs/day3b_exact_brownian_bridge/`.

## Day 5 evidence

- RC-L Table 7 maximum absolute direct-price error versus the reported paper MC:
  **0.761446 per 100**, within the pre-set 1.25 tolerance.
- RC-L Table 7 maximum absolute fair-coupon error: **0.010419** in annual-rate
  units, within the pre-set 0.0200 tolerance.
- Lee Table 8 maximum absolute direct-price error: **0.299435 per 100**.
- RC-A value at the frozen 8.75% coupon: **94.023106 per 100**.
- RC-A fair coupon: **14.915149%**, replication SE **0.158151%**, with a
  normal-approximation 95% interval of **[14.605173%, 15.225125%]**.
- Maximum component identity error: **1.421e-14**; maximum probability-mass
  error: **0**; maximum linear fair-coupon residual: **0**.
- Volatility, correlation, continuous-KI barrier, autocall-schedule and
  monitoring-grid diagnostics are saved under
  `outputs/day5_direct_autocallable/`.
- Day 5 is an M0 direct fine-grid result and does not use the failed Day 4
  logistic approximation. Day 6 uses fresh nested references under each
  current interval signature rather than applying that logistic outside its
  validated domain.
- RC-A remains a stylised market-informed research contract, not HSBC
  `40447DKU1` and not an issuer or dealer market price.

## Day 6 evidence

- Day 4's failed logistic approximation is not reused. The trivariate term uses
  a fresh scrambled-Sobol conditional bridge bank under the current interval
  signature, with the median of three exact-`h_ij`-anchored conditional ratios.
- Maximum M0–M3 price difference versus M0: **0.356807 per 100**, inside the
  pre-set **1.25** tolerance.
- M3 RC-L value: **101.159217 per 100** (replication SE **0.094376**); fair
  coupon: **4.921965%** (SE **0.082768%**).
- M3 RC-A value at the frozen 8.75% coupon: **93.889453 per 100** (replication
  SE **0.034672**); fair coupon: **15.067579%** (SE **0.040050%**).
- Paired Bernoulli bridge versus conditional-weight maximum absolute mean price
  difference: **0.050097 per 100**.
- High-budget fresh nested `q3` audit maximum RMSE: **0.000532**; maximum
  absolute error: **0.002604**. The unused auxiliary direct-survival frequency
  is retained separately so its time-discretisation error is not hidden.
- Final probability-bound violation rate: **0**; numerical underflow count:
  **0**. The largest mean feasible-projection magnitude is **0.000170** and the
  largest interval 99th percentile is **0.008775**. The observed single-path
  maximum **0.058331** is retained as an approximation warning.
- Correlation-to-zero and permutation maximum error: **4.374e-14**. Maximum
  component identity error: **2.842e-14**; probability-mass error and
  fair-coupon root residual are **0**.
- Audit outputs: `outputs/day6_three_asset_bb_conditioning/`. All eight Day 6
  gates pass, so the plan may proceed to Day 7 estimator comparison; this does
  not retroactively validate the Day 4 logistic.

## Day 7 evidence

- Full-product M0 and M3 use an equal outer budget of **512 paths** per
  replication and **4 independent replications**. Every base/plus/minus group
  uses CRN or the identical scrambled Sobol points.
- All **640** requested Greek replication rows are present: three component
  Deltas, Vegas and Gammas, parallel Vega/Gamma, four relative spot bumps and
  three absolute volatility bumps in both near-KI and near-autocall states.
- At the 0.5% active-asset bump in near-KI, the M0/M3 replication-SD ratios are
  **1.31× for Delta**, **1.87× for Vega** and **2.05× for Gamma**, favouring M3.
- At the same bump in near-autocall, the M0/M3 ratios are **0.34× for Delta**,
  **1.79× for Vega** and **0.37× for Gamma**. Brownian-bridge KI conditioning
  therefore does not smooth the discrete autocall trigger and M3 is not
  universally more stable.
- Only **12.5%** of Gamma/parallel-Gamma groups pass the pure bump-shape plateau
  criterion. All are statistically consistent after accounting for the large
  four-replication uncertainty, but this is not presented as a true plateau;
  no small-bump Gamma is hidden.
- Three fresh bridge-bank variants change the tested near-KI active Delta and
  Gamma by only **2.61e-7** and **4.10e-7** in reported units, respectively.
- On the isolated next-observation autocall principal redemption component,
  AC-Smooth agrees with raw RQMC within **0.005583 per 100** and reduces
  replication SD by **97.0× for price**, **61.2× for active Delta** and
  **126.1× for active Gamma**. Rotation orthogonality/correlation errors are
  below **6.67e-16** and inverse-normal probability clipping is zero.
- AC-Smooth remains a B8 discrete-trigger diagnostic, not a Brownian-bridge
  method and not a replacement for full-product pricing. Its promising local
  result requires a future full-product branching implementation before being
  used as a production Greek estimator.
- Audit outputs: `outputs/day7_greeks_trigger_smoothing/`. All nine Day 7
  delivery gates pass with the documented Gamma caveat.

## Day 8 evidence

- The GPU environment is an **NVIDIA GeForce RTX 4060 Laptop GPU (8 GB)** with
  CuPy 14.1.1. All CUDA pricing arithmetic is **float64**.
- All **9/9 Day 8 gates pass** and the full project regression suite reports
  **19/19 tests passed**.
- Under identical normal innovations, the largest M0/M1 CPU/GPU component
  difference is **1.421e-14**. The maximum component identity error is
  **1.421e-14**; probability-mass and fair-coupon root residuals are **0**.
- The measured steady-state break-even on the 52-step-per-year performance grid
  is **N = 8,192** paths. At **N = 131,072**, the measured direct-M0 speed-up is
  **9.28x**, including host-to-device transfer and payoff reduction. Peak CuPy
  memory-pool use in the tested grid is **371,723,776 bytes** (about 354.5 MiB).
- Thirteen base/plus/minus CRN scenarios are evaluated on one GPU scenario axis.
  The maximum CPU/GPU Delta/Vega/Gamma difference is **4.452e-10**.
- M2/M3 are explicitly hybrid, not mislabelled as end-to-end GPU: endpoint GBM,
  exact marginal exits and nested `q3` bridge counts run on CUDA; the validated
  non-integer-order Bessel `h_ij` calculation remains in SciPy on the CPU because
  CuPy 14.1 does not expose `ive`. The largest M2/M3 CPU/GPU component difference
  is **1.110e-15**.
- The primary presentation figure is
  `outputs/day8_gpu_validation/cpu_gpu_core_evidence.png`. Four supporting charts,
  a presentation guide and complete CSV/hash audit are saved under
  `outputs/day8_gpu_validation/`.

## Day 9 evidence

- The complete experiment matrix contains **1,112 formal replications**:
  256 baseline estimator rows, 24 high-budget reference rows, 352 regime rows,
  and 480 next-observation trigger-smoothing rows. Exact checkpoint/resume is
  enabled for every replication table.
- The high-budget M3 reference uses **N = 262,144** and **24 independent Sobol
  scrambles**. The RC-A value is **99.336888 per 100** (replication SE
  **0.001541**) and the corresponding fair coupon is **9.684969%** (SE
  **0.002188 percentage points**).
- At N = 65,536, M3 reaches an observed price RMSE of **0.011833** in a median
  **14.381 seconds** and lies on the price accuracy-cost Pareto frontier. M0 is
  faster for the moderate extrapolated RMSE target of 0.05, so the evidence
  supports M3 as a high-accuracy estimator rather than a universal speed winner.
- The RQMC-conditioning interaction ratio is above one at **3 of 4** tested
  path budgets, ranging from **0.950329** to **2.746141**. Complementarity is
  therefore supported at most tested budgets but is not claimed universally.
- Axis-aligned next-observation smoothing agrees with raw RQMC at all **15/15**
  tested spot/horizon cells and reaches a maximum local variance-reduction
  factor of **29,858.1x**. This remains a local trigger-component diagnostic,
  not a full-product production replacement.
- All **10/10 validation gates pass** and the expanded project regression suite
  reports **25/25 tests passed**. Formal figure filenames and figure content do
  not use Day-number labels.
- The primary presentation figure is
  `outputs/day9_main_experiments/accuracy_cost_and_interaction.png`. Five
  supporting figures, source-to-method notes, presentation guidance, complete
  CSV outputs and a hash inventory are in
  `outputs/day9_main_experiments/`.
- RC-A remains a stylised market-informed research contract and not an HSBC
  issuer quote, dealer mark or tradeable market price.

## Source integrity

The frozen workbook SHA-256 remains:
`CD84790399F9CDA5C5B0ED3E95D973216314CA54DD4BDBA6D9D22C399DBD49E6`.
The pre-layering configuration is preserved as
`config/core_project_config.pre_day1_layering_20260804.json`.

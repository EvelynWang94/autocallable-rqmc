# Literature-to-method map for the main experiment

## Glynn and Whitt (1992)

**Source:** Peter W. Glynn and Ward Whitt, “The Asymptotic Efficiency of Simulation Estimators,” *Operations Research* 40(3), 505–520.

### Achievement

The paper places an estimator and its computational cost in one budget-constrained framework. In the canonical square-error case, asymptotic efficiency is inversely proportional to the product of the estimator’s variance rate and computational cost rate. It also warns that variance reduction alone is not a sufficient ranking when convergence or cost growth is noncanonical.

### Use in this project

- Report `variance × median total runtime`, including random-input generation, transfers and payoff reduction.
- Estimate the empirical RMSE convergence exponent and runtime-growth exponent for M0–M3.
- Report the implied squared-loss efficiency rate and time-to-target price RMSE.
- Use the three-objective price-RMSE, replication-SD and runtime Pareto frontier instead of ranking methods by VRF alone.
- Keep one-off GPU compilation separate from steady replicated cost.

### Boundary

The finite sample experiment does not prove an asymptotic theorem. The fitted rates are empirical diagnostics over the frozen path grid and should not be extrapolated far beyond it without a larger convergence study.

## Wang (2016)

**Source:** Xiaoqun Wang, “Handling Discontinuities in Financial Engineering: Good Path Simulation and Smoothing,” *Operations Research* 64(2), 297–314.

### Achievement

The paper shows that a useful QMC path construction for a discontinuous payoff should align the discontinuity with coordinate axes, not merely reduce effective dimension. It develops a two-step approach: align the discontinuity, then remove or integrate out the active coordinate. The resulting estimator is unbiased and has lower variance under the stated structure. The probability of the active region measures how severe the discontinuity is.

### Use in this project

- Do not assume Brownian bridge or PCA is universally superior for discrete triggers.
- Use the existing Householder rotation to align the next-observation worst-of autocall boundary with one normal coordinate.
- Integrate that coordinate conditionally in `AC-Smooth`, and compare raw versus smoothed RQMC on a near-autocall spot and time-to-observation grid.
- Report call probability, normalized discontinuity severity `4p(1-p)`, VRF and cost-efficiency gain.
- Keep the diagnostic local to the next-observation autocall-principal component; it is not promoted to a full-product estimator.

### Boundary

The full autocallable has multiple observation-date trigger surfaces, coupon memory and continuous knock-in conditioning. Wang’s one-boundary two-step result does not by itself validate a full-product branching implementation. The present experiment therefore uses it only where the aligned one-step event is exact.

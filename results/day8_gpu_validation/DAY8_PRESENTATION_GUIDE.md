# GPU implementation presentation guide

## Implementation focus

1. **Correctness before speed.** CPU and GPU consume identical float64 normal inputs and are checked by payoff component, probability mass and fair-coupon identity.
2. **Break-even, not headline marketing.** Cold/shape-setup and steady-state timings are separate; H2D transfer and reduction stay inside GPU wall time.
3. **Batched Greeks.** Thirteen CRN base/plus/minus scenarios share random input and correlated path work on a GPU scenario axis.
4. **Honest M2/M3 boundary.** Endpoint GBM, marginal exits and nested q3 run on CUDA; exact non-integer Bessel h_ij remains in SciPy CPU because CuPy 14.1 has no ive.
5. **No method substitution.** The established contract, event clocks, Brownian-bridge interface, bump convention and result schema remain unchanged.

## Core chart

Use `cpu_gpu_core_evidence.png` as the primary presentation figure. Its left panel shows the measured break-even (N = 8,192) and 9.28× steady-state speed-up at N = 131,072; its right panel shows that every CPU/GPU numerical-equivalence error is below its frozen tolerance.

## Supporting charts

- `gpu_timing_breakdown.png`: shows that transfer and reduction are included and identifies the dominant direct-pricing stage.
- `batched_greeks_agreement.png`: compares CPU/GPU Delta, Vega and Gamma for all three assets.
- `conditioned_pipeline_timing.png`: shows the M2/M3 hybrid boundary and retained exact CPU Bessel cost.
- `cpu_gpu_performance_scaling.png`: provides the complete cold-start versus steady-state scaling view.

## Headline validated numbers

- Validation gates: 9/9 passed.
- M0/M1 maximum component difference: 1.421e-14.
- M2/M3 maximum component difference: 1.110e-15.
- Batched Greek maximum difference: 4.452e-10.
- Regression tests: 19 passed.

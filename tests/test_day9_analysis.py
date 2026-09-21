import math

import numpy as np
import pandas as pd
import pytest

from day9_analysis import (
    add_pareto_flags,
    convergence_and_time_to_target,
    discontinuity_severity,
    rqmc_conditioning_interaction,
    summarise_baseline,
    summarise_trigger_smoothing,
)


def _reference_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "method": ["M3"] * 4,
            "n_paths": [262144] * 4,
            "total_value": [100.0, 100.1, 99.9, 100.0],
            "fair_coupon": [0.08, 0.081, 0.079, 0.08],
            "coupon_value": [8.0] * 4,
            "early_redemption_principal": [60.0] * 4,
            "surviving_notional": [35.0] * 4,
            "terminal_ki_loss": [-3.0] * 4,
            "continuous_ki_survival_probability": [0.7] * 4,
        }
    )


def test_discontinuity_severity_is_normalized() -> None:
    assert discontinuity_severity(0.0) == 0.0
    assert discontinuity_severity(1.0) == 0.0
    assert discontinuity_severity(0.5) == 1.0
    with pytest.raises(ValueError):
        discontinuity_severity(1.01)


def test_baseline_summary_uses_variance_times_total_cost() -> None:
    rows = []
    for method, spread, runtime in (("M0", 1.0, 1.0), ("M1", 0.5, 2.0)):
        for replication, shock in enumerate((-1.0, 1.0, -1.0, 1.0)):
            rows.append(
                {
                    "method": method,
                    "n_paths": 1024,
                    "replication": replication,
                    "runtime_total_seconds": runtime,
                    "total_value": 100.0 + spread * shock,
                    "fair_coupon": 0.08,
                    "coupon_value": 8.0,
                    "early_redemption_principal": 60.0,
                    "surviving_notional": 35.0,
                    "terminal_ki_loss": -3.0,
                    "continuous_ki_survival_probability": 0.7,
                }
            )
    summary = summarise_baseline(pd.DataFrame(rows), _reference_frame())
    m0 = summary.loc[summary["method"] == "M0"].iloc[0]
    m1 = summary.loc[summary["method"] == "M1"].iloc[0]
    assert m0["variance_times_cost"] == pytest.approx(m0["total_value_sd"] ** 2)
    assert m1["variance_times_cost"] == pytest.approx(
        m1["total_value_sd"] ** 2 * 2.0
    )
    assert m1["relative_cost_efficiency_vs_m0"] > 1.0


def test_interaction_ratio_matches_definition() -> None:
    summary = pd.DataFrame(
        {
            "n_paths": [1024] * 4,
            "method": ["M0", "M1", "M2", "M3"],
            "total_value_sd": [4.0, 2.0, 3.0, 1.0],
        }
    )
    result = rqmc_conditioning_interaction(summary).iloc[0]
    assert result["direct_rqmc_vrf"] == pytest.approx(4.0)
    assert result["conditioned_rqmc_vrf"] == pytest.approx(9.0)
    assert result["interaction_ratio"] == pytest.approx(2.25)
    assert bool(result["supports_complementarity"])


def test_pareto_frontier_removes_dominated_points() -> None:
    points = pd.DataFrame(
        {
            "name": ["fast", "balanced", "dominated"],
            "total_value_rmse": [2.0, 1.0, 3.0],
            "total_value_sd": [2.0, 1.0, 3.0],
            "runtime_median_seconds": [1.0, 2.0, 3.0],
        }
    )
    result = add_pareto_flags(points).set_index("name")
    assert bool(result.loc["fast", "pareto_frontier"])
    assert bool(result.loc["balanced", "pareto_frontier"])
    assert not bool(result.loc["dominated", "pareto_frontier"])


def test_empirical_convergence_recovers_half_rate() -> None:
    n_paths = np.array([1024, 4096, 16384, 65536], dtype=float)
    frame = pd.DataFrame(
        {
            "method": ["M0"] * 4,
            "n_paths": n_paths,
            "total_value_rmse": 10.0 / np.sqrt(n_paths),
            "runtime_median_seconds": n_paths / 1024.0,
        }
    )
    result = convergence_and_time_to_target(frame, 0.05).iloc[0]
    assert result["rmse_convergence_gamma"] == pytest.approx(0.5)
    assert result["cost_growth_beta"] == pytest.approx(1.0)
    assert result["glynn_whitt_squared_loss_rate"] == pytest.approx(1.0)
    assert math.isfinite(result["predicted_time_to_target_seconds"])


def test_trigger_smoothing_reports_vrf_and_agreement() -> None:
    rows = []
    for smooth, values, runtime in (
        (False, [10.0, 12.0, 8.0, 10.0], 1.0),
        (True, [9.8, 10.2, 9.9, 10.1], 1.2),
    ):
        for replication, value in enumerate(values):
            rows.append(
                {
                    "worst_spot_ratio": 1.0,
                    "time_to_observation_days": 2,
                    "smooth": smooth,
                    "replication": replication,
                    "value": value,
                    "runtime_seconds": runtime,
                    "call_probability": 0.5,
                    "probability_clipping_rate": 0.0,
                }
            )
    summary = summarise_trigger_smoothing(pd.DataFrame(rows), 4.0, 0.05).iloc[0]
    assert bool(summary["agreement_pass"])
    assert summary["variance_reduction_factor"] > 10.0
    assert summary["discontinuity_severity"] == pytest.approx(1.0)

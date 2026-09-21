"""Pure analysis and audit helpers for the Day 9 main experiment.

The numerical engines live in the Day 5--8 modules.  This module keeps the
cross-method comparison explicit: sampling error, approximation evidence,
runtime, and discontinuity diagnostics remain separate quantities.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


PRICE_COMPONENTS = (
    "total_value",
    "fair_coupon",
    "coupon_value",
    "early_redemption_principal",
    "surviving_notional",
    "terminal_ki_loss",
    "continuous_ki_survival_probability",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def discontinuity_severity(probability: float | np.ndarray) -> float | np.ndarray:
    """Wang-style normalized severity: zero at 0/1 and one at probability 1/2."""

    probability_array = np.asarray(probability, dtype=float)
    if np.any((probability_array < 0.0) | (probability_array > 1.0)):
        raise ValueError("probability must lie in [0, 1]")
    severity = 4.0 * probability_array * (1.0 - probability_array)
    return float(severity) if severity.ndim == 0 else severity


def _sample_sd(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def _rmse(values: np.ndarray, reference: float) -> float:
    return float(np.sqrt(np.mean(np.square(values - reference))))


def reference_summary(reference_replications: pd.DataFrame) -> pd.DataFrame:
    if reference_replications.empty:
        raise ValueError("reference replications are empty")
    rows: list[dict[str, float | int | str]] = []
    for metric in PRICE_COMPONENTS:
        if metric not in reference_replications:
            continue
        values = reference_replications[metric].astype(float).to_numpy()
        sd = _sample_sd(values)
        rows.append(
            {
                "metric": metric,
                "reference_mean": float(np.mean(values)),
                "reference_sd": sd,
                "reference_se": sd / math.sqrt(len(values)),
                "reference_replications": int(len(values)),
                "reference_n_paths": int(reference_replications["n_paths"].iloc[0]),
                "reference_method": str(reference_replications["method"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def summarise_baseline(
    baseline_replications: pd.DataFrame,
    reference_replications: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise method/N cells against a high-budget reference.

    RMSE is for one replication at the stated path count.  Runtime includes
    random-input generation and all transfers/reductions, but excludes the
    one-off CUDA compilation warm-up.
    """

    if baseline_replications.empty:
        raise ValueError("baseline replications are empty")
    required = {"method", "n_paths", "replication", "runtime_total_seconds"}
    missing = required - set(baseline_replications.columns)
    if missing:
        raise ValueError(f"baseline columns missing: {sorted(missing)}")
    references = reference_summary(reference_replications).set_index("metric")
    rows: list[dict[str, float | int | str]] = []
    for (method, n_paths), group in baseline_replications.groupby(
        ["method", "n_paths"], sort=True
    ):
        runtime = group["runtime_total_seconds"].astype(float).to_numpy()
        row: dict[str, float | int | str] = {
            "method": str(method),
            "n_paths": int(n_paths),
            "replications": int(len(group)),
            "runtime_median_seconds": float(np.median(runtime)),
            "runtime_p10_seconds": float(np.quantile(runtime, 0.10)),
            "runtime_p90_seconds": float(np.quantile(runtime, 0.90)),
        }
        for metric in PRICE_COMPONENTS:
            if metric not in group or metric not in references.index:
                continue
            values = group[metric].astype(float).to_numpy()
            reference = float(references.loc[metric, "reference_mean"])
            mean = float(np.mean(values))
            sd = _sample_sd(values)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
            row[f"{metric}_se"] = sd / math.sqrt(len(values))
            row[f"{metric}_bias"] = mean - reference
            row[f"{metric}_rmse"] = _rmse(values, reference)
        variance = float(row.get("total_value_sd", 0.0)) ** 2
        cost = float(row["runtime_median_seconds"])
        row["variance_times_cost"] = variance * cost
        row["canonical_efficiency"] = (
            1.0 / (variance * cost) if variance > 0.0 and cost > 0.0 else np.inf
        )
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(["n_paths", "method"]).reset_index(drop=True)
    m0_efficiency = (
        summary.loc[summary["method"] == "M0", ["n_paths", "canonical_efficiency"]]
        .rename(columns={"canonical_efficiency": "m0_canonical_efficiency"})
    )
    summary = summary.merge(m0_efficiency, on="n_paths", how="left")
    summary["relative_cost_efficiency_vs_m0"] = (
        summary["canonical_efficiency"] / summary["m0_canonical_efficiency"]
    )
    return summary


def rqmc_conditioning_interaction(baseline_summary: pd.DataFrame) -> pd.DataFrame:
    """Compute (Var(M2)/Var(M3)) / (Var(M0)/Var(M1)) at each N."""

    pivot = baseline_summary.pivot(
        index="n_paths", columns="method", values="total_value_sd"
    )
    required = {"M0", "M1", "M2", "M3"}
    missing = required - set(pivot.columns)
    if missing:
        raise ValueError(f"interaction methods missing: {sorted(missing)}")
    variance = pivot.pow(2)
    result = pd.DataFrame(index=pivot.index)
    result["direct_rqmc_vrf"] = variance["M0"] / variance["M1"]
    result["conditioned_rqmc_vrf"] = variance["M2"] / variance["M3"]
    result["conditioning_vrf_mc"] = variance["M0"] / variance["M2"]
    result["conditioning_vrf_rqmc"] = variance["M1"] / variance["M3"]
    result["interaction_ratio"] = (
        result["conditioned_rqmc_vrf"] / result["direct_rqmc_vrf"]
    )
    result["supports_complementarity"] = result["interaction_ratio"] > 1.0
    return result.reset_index()


def convergence_and_time_to_target(
    baseline_summary: pd.DataFrame,
    target_price_rmse: float,
) -> pd.DataFrame:
    """Fit empirical RMSE and runtime powers and extrapolate a target time."""

    if target_price_rmse <= 0.0:
        raise ValueError("target_price_rmse must be positive")
    rows: list[dict[str, float | str]] = []
    for method, group in baseline_summary.groupby("method", sort=True):
        group = group.sort_values("n_paths")
        n_paths = group["n_paths"].astype(float).to_numpy()
        rmse = group["total_value_rmse"].astype(float).to_numpy()
        runtime = group["runtime_median_seconds"].astype(float).to_numpy()
        valid = (n_paths > 0.0) & (rmse > 0.0) & (runtime > 0.0)
        if valid.sum() < 3:
            continue
        n_paths = n_paths[valid]
        rmse = rmse[valid]
        runtime = runtime[valid]
        rmse_slope, rmse_intercept = np.polyfit(np.log(n_paths), np.log(rmse), 1)
        cost_slope, cost_intercept = np.polyfit(np.log(n_paths), np.log(runtime), 1)
        time_slope, time_intercept = np.polyfit(np.log(runtime), np.log(rmse), 1)
        predicted_time = np.nan
        if time_slope < 0.0:
            predicted_time = float(
                np.exp((math.log(target_price_rmse) - time_intercept) / time_slope)
            )
        gamma = float(-rmse_slope)
        beta = float(cost_slope)
        rows.append(
            {
                "method": str(method),
                "rmse_convergence_gamma": gamma,
                "cost_growth_beta": beta,
                "glynn_whitt_squared_loss_rate": 2.0 * gamma / beta if beta > 0 else np.nan,
                "rmse_vs_time_slope": float(time_slope),
                "target_price_rmse": float(target_price_rmse),
                "predicted_time_to_target_seconds": predicted_time,
                "fit_points": int(valid.sum()),
                "rmse_fit_intercept": float(rmse_intercept),
                "cost_fit_intercept": float(cost_intercept),
            }
        )
    return pd.DataFrame(rows)


def add_pareto_flags(
    points: pd.DataFrame,
    objectives: Sequence[str] = (
        "total_value_rmse",
        "total_value_sd",
        "runtime_median_seconds",
    ),
) -> pd.DataFrame:
    """Mark points that are not dominated under all-minimization objectives."""

    missing = set(objectives) - set(points.columns)
    if missing:
        raise ValueError(f"Pareto objectives missing: {sorted(missing)}")
    result = points.copy().reset_index(drop=True)
    values = result.loc[:, objectives].astype(float).to_numpy()
    finite = np.all(np.isfinite(values), axis=1)
    frontier = np.zeros(len(result), dtype=bool)
    for index, candidate in enumerate(values):
        if not finite[index]:
            continue
        other = values[finite]
        dominated = np.any(
            np.all(other <= candidate, axis=1) & np.any(other < candidate, axis=1)
        )
        frontier[index] = not dominated
    result["pareto_frontier"] = frontier
    return result


def summarise_regimes(regime_replications: pd.DataFrame) -> pd.DataFrame:
    if regime_replications.empty:
        raise ValueError("regime replications are empty")
    grouping = [
        "scenario_id",
        "regime_family",
        "regime_level",
        "contract_id",
        "method",
    ]
    rows: list[dict[str, float | int | str]] = []
    for keys, group in regime_replications.groupby(grouping, sort=False):
        row = dict(zip(grouping, keys))
        row["replications"] = int(len(group))
        for metric in (
            "total_value",
            "fair_coupon",
            "continuous_ki_survival_probability",
        ):
            values = group[metric].astype(float).to_numpy()
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_sd"] = _sample_sd(values)
        row["runtime_median_seconds"] = float(
            np.median(group["runtime_total_seconds"].astype(float))
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summarise_trigger_smoothing(
    replications: pd.DataFrame,
    z_tolerance: float,
    floor_tolerance: float,
) -> pd.DataFrame:
    """Compare raw and aligned conditional-smoothed RQMC by trigger regime."""

    if z_tolerance <= 0.0 or floor_tolerance < 0.0:
        raise ValueError("trigger tolerances must be nonnegative")
    rows: list[dict[str, float | int | bool]] = []
    grouping = ["worst_spot_ratio", "time_to_observation_days"]
    for keys, group in replications.groupby(grouping, sort=True):
        raw = group.loc[~group["smooth"].astype(bool)]
        smooth = group.loc[group["smooth"].astype(bool)]
        if raw.empty or smooth.empty:
            continue
        raw_values = raw["value"].astype(float).to_numpy()
        smooth_values = smooth["value"].astype(float).to_numpy()
        raw_sd = _sample_sd(raw_values)
        smooth_sd = _sample_sd(smooth_values)
        raw_mean = float(np.mean(raw_values))
        smooth_mean = float(np.mean(smooth_values))
        combined_se = math.sqrt(
            raw_sd**2 / len(raw_values) + smooth_sd**2 / len(smooth_values)
        )
        tolerance = max(floor_tolerance, z_tolerance * combined_se)
        probability = float(np.mean(smooth["call_probability"].astype(float)))
        raw_runtime = float(np.median(raw["runtime_seconds"].astype(float)))
        smooth_runtime = float(np.median(smooth["runtime_seconds"].astype(float)))
        raw_variance = raw_sd**2
        smooth_variance = smooth_sd**2
        rows.append(
            {
                "worst_spot_ratio": float(keys[0]),
                "time_to_observation_days": int(keys[1]),
                "replications_raw": int(len(raw)),
                "replications_smooth": int(len(smooth)),
                "raw_value_mean": raw_mean,
                "smooth_value_mean": smooth_mean,
                "absolute_price_difference": abs(raw_mean - smooth_mean),
                "agreement_tolerance": tolerance,
                "agreement_pass": abs(raw_mean - smooth_mean) <= tolerance,
                "raw_replication_sd": raw_sd,
                "smooth_replication_sd": smooth_sd,
                "variance_reduction_factor": (
                    raw_variance / smooth_variance if smooth_variance > 0.0 else np.inf
                ),
                "raw_runtime_median_seconds": raw_runtime,
                "smooth_runtime_median_seconds": smooth_runtime,
                "cost_efficiency_gain": (
                    raw_variance * raw_runtime / (smooth_variance * smooth_runtime)
                    if smooth_variance > 0.0 and smooth_runtime > 0.0
                    else np.inf
                ),
                "call_probability": probability,
                "discontinuity_severity": discontinuity_severity(probability),
                "maximum_probability_clipping_rate": float(
                    smooth["probability_clipping_rate"].astype(float).max()
                ),
            }
        )
    return pd.DataFrame(rows)


def greek_time_to_target(
    greek_summary: pd.DataFrame,
    targets: Mapping[str, float],
    bump: float = 0.005,
) -> pd.DataFrame:
    """Canonical independent-replication time needed to reach a target SE."""

    selected = greek_summary.loc[
        greek_summary["active_asset"].astype(bool)
        & np.isclose(greek_summary["bump"].astype(float), bump)
        & greek_summary["greek"].isin(targets)
    ].copy()
    selected["target_rmse"] = selected["greek"].map(targets).astype(float)
    selected["required_independent_replications"] = np.maximum(
        1.0,
        np.square(selected["estimate_sd"] / selected["target_rmse"]),
    )
    selected["canonical_time_to_target_seconds"] = (
        selected["runtime_seconds_mean"] * selected["required_independent_replications"]
    )
    return selected[
        [
            "scenario",
            "method",
            "greek",
            "asset",
            "bump",
            "estimate_mean",
            "estimate_sd",
            "target_rmse",
            "required_independent_replications",
            "runtime_seconds_mean",
            "canonical_time_to_target_seconds",
        ]
    ].reset_index(drop=True)


def evaluate_gates(
    baseline_replications: pd.DataFrame,
    reference_replications: pd.DataFrame,
    regime_replications: pd.DataFrame,
    trigger_summary: pd.DataFrame,
    interaction: pd.DataFrame,
    config: Mapping[str, object],
    formal_figure_names: Iterable[str],
) -> pd.DataFrame:
    acceptance = config["acceptance"]  # type: ignore[index]
    baseline_cfg = config["baseline"]  # type: ignore[index]
    reference_cfg = config["reference"]  # type: ignore[index]
    regime_cfg = config["regimes"]  # type: ignore[index]
    expected_baseline = (
        len(baseline_cfg["methods"])
        * len(baseline_cfg["path_grid"])
        * int(baseline_cfg["replications"])
    )
    expected_regime_minimum = int(regime_cfg["replications"])
    figures = list(formal_figure_names)
    rows = [
        {
            "gate": "baseline_inventory_complete",
            "observed": len(baseline_replications),
            "threshold": expected_baseline,
            "pass": len(baseline_replications) == expected_baseline,
        },
        {
            "gate": "reference_replications_complete",
            "observed": len(reference_replications),
            "threshold": int(reference_cfg["replications"]),
            "pass": len(reference_replications) >= int(reference_cfg["replications"]),
        },
        {
            "gate": "regime_cells_have_r16",
            "observed": int(
                regime_replications.groupby(["scenario_id", "method"]).size().min()
            ),
            "threshold": expected_regime_minimum,
            "pass": bool(
                regime_replications.groupby(["scenario_id", "method"]).size().min()
                >= expected_regime_minimum
            ),
        },
        {
            "gate": "component_identity",
            "observed": float(
                pd.concat([baseline_replications, reference_replications, regime_replications])[
                    "component_identity_error"
                ].abs().max()
            ),
            "threshold": float(acceptance["component_identity_absolute"]),
            "pass": bool(
                pd.concat([baseline_replications, reference_replications, regime_replications])[
                    "component_identity_error"
                ].abs().max()
                <= float(acceptance["component_identity_absolute"])
            ),
        },
        {
            "gate": "probability_mass",
            "observed": float(
                pd.concat([baseline_replications, reference_replications, regime_replications])[
                    "probability_mass_error"
                ].abs().max()
            ),
            "threshold": float(acceptance["probability_mass_absolute"]),
            "pass": bool(
                pd.concat([baseline_replications, reference_replications, regime_replications])[
                    "probability_mass_error"
                ].abs().max()
                <= float(acceptance["probability_mass_absolute"])
            ),
        },
        {
            "gate": "fair_coupon_root",
            "observed": float(
                pd.concat([baseline_replications, reference_replications, regime_replications])[
                    "fair_coupon_residual"
                ].abs().max()
            ),
            "threshold": float(acceptance["fair_coupon_root_absolute"]),
            "pass": bool(
                pd.concat([baseline_replications, reference_replications, regime_replications])[
                    "fair_coupon_residual"
                ].abs().max()
                <= float(acceptance["fair_coupon_root_absolute"])
            ),
        },
        {
            "gate": "interaction_grid_complete",
            "observed": len(interaction),
            "threshold": int(acceptance["required_interaction_points"]),
            "pass": len(interaction) >= int(acceptance["required_interaction_points"]),
        },
        {
            "gate": "trigger_smoothing_price_agreement",
            "observed": int(trigger_summary["agreement_pass"].sum()),
            "threshold": len(trigger_summary),
            "pass": bool(trigger_summary["agreement_pass"].all()),
        },
        {
            "gate": "formal_figure_names_exclude_day_labels",
            "observed": int(sum("day" in name.lower() for name in figures)),
            "threshold": 0,
            "pass": not any("day" in name.lower() for name in figures),
        },
    ]
    return pd.DataFrame(rows)


def build_audit_inventory(paths: Iterable[str | Path], root: str | Path) -> pd.DataFrame:
    root_path = Path(root).resolve()
    rows = []
    for item in paths:
        path = Path(item).resolve()
        rows.append(
            {
                "relative_path": path.relative_to(root_path).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return pd.DataFrame(rows).sort_values("relative_path").reset_index(drop=True)

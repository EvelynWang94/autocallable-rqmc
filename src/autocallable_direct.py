"""Reusable direct fine-grid engine for the Day 5 autocallable experiments.

The engine keeps discrete coupon/autocall events separate from the pathwise
continuous-KI approximation.  Contract events are simulated on their exact
observation dates; the KI event is approximated on a configurable fine grid.
All monetary outputs are present values per contract notional.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DirectContract:
    contract_id: str
    label: str
    principal: float
    maturity_years: float
    observation_times: tuple[float, ...]
    autocall_flags: tuple[bool, ...]
    autocall_trigger_ratios: tuple[float | None, ...]
    coupon_flags: tuple[bool, ...]
    coupon_trigger_ratios: tuple[float | None, ...]
    coupon_mode: str
    coupon_memory: bool
    coupon_periods_per_year: int
    baseline_annual_coupon: float
    ki_barrier_ratio: float
    claim_boundary: str
    fixed_maturity_no_ki_coupon_rate: float | None = None

    def with_overrides(self, **changes: Any) -> "DirectContract":
        return replace(self, **changes)


def _as_float_tuple(values: Iterable[Any]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def parse_research_contract(config: dict[str, Any], contract_id: str) -> DirectContract:
    """Parse the frozen RC-L/RC-A schemas into one validated interface."""
    raw = config["research_contracts"][contract_id]
    if contract_id == "RC-L":
        times = _as_float_tuple(raw["observation_times_years"])
        triggers = _as_float_tuple(raw["autocall_barrier_ratios"])
        contract = DirectContract(
            contract_id=contract_id,
            label=raw["label"],
            principal=float(raw["principal"]),
            maturity_years=float(raw["maturity_years"]),
            observation_times=times,
            autocall_flags=tuple(True for _ in times),
            autocall_trigger_ratios=triggers,
            coupon_flags=tuple(False for _ in times),
            coupon_trigger_ratios=tuple(None for _ in times),
            coupon_mode="simple_to_redemption",
            coupon_memory=False,
            coupon_periods_per_year=2,
            baseline_annual_coupon=float(raw["coupon"]["baseline_annual_rate"]),
            ki_barrier_ratio=float(raw["knock_in"]["barrier_ratio"]),
            claim_boundary=raw["claim_boundary"],
        )
    elif contract_id == "RC-A":
        schedule = raw["observation_schedule"]
        times = _as_float_tuple(item["time_years_act365"] for item in schedule)
        frequency = raw["coupon"].get("frequency", "quarterly")
        periods_per_year = {"annual": 1, "semiannual": 2, "quarterly": 4, "monthly": 12}.get(
            frequency
        )
        if periods_per_year is None:
            raise ValueError(f"Unsupported coupon frequency: {frequency}")
        contract = DirectContract(
            contract_id=contract_id,
            label=raw["label"],
            principal=float(raw["principal"]),
            maturity_years=float(times[-1]),
            observation_times=times,
            autocall_flags=tuple(bool(item["autocall"]) for item in schedule),
            autocall_trigger_ratios=tuple(
                None if item["autocall_trigger_ratio"] is None else float(item["autocall_trigger_ratio"])
                for item in schedule
            ),
            coupon_flags=tuple(bool(item["coupon"]) for item in schedule),
            coupon_trigger_ratios=tuple(
                None if item["coupon_trigger_ratio"] is None else float(item["coupon_trigger_ratio"])
                for item in schedule
            ),
            coupon_mode="memory_periodic" if raw["coupon"]["memory"] else "periodic",
            coupon_memory=bool(raw["coupon"]["memory"]),
            coupon_periods_per_year=periods_per_year,
            baseline_annual_coupon=float(raw["coupon"]["baseline_annual_rate"]),
            ki_barrier_ratio=float(raw["knock_in"]["barrier_ratio"]),
            claim_boundary=raw["claim_boundary"],
        )
    else:
        raise KeyError(f"Unsupported research contract: {contract_id}")
    validate_contract(contract)
    return contract


def validate_contract(contract: DirectContract) -> None:
    n_obs = len(contract.observation_times)
    aligned = [
        contract.autocall_flags,
        contract.autocall_trigger_ratios,
        contract.coupon_flags,
        contract.coupon_trigger_ratios,
    ]
    if n_obs == 0 or any(len(values) != n_obs for values in aligned):
        raise ValueError("Observation schedule fields must have equal non-zero length")
    times = np.asarray(contract.observation_times, dtype=float)
    if not np.all(np.diff(times) > 0) or not math.isclose(times[-1], contract.maturity_years):
        raise ValueError("Observation times must increase and end at maturity")
    if contract.principal <= 0 or not (0 < contract.ki_barrier_ratio < 1.5):
        raise ValueError("Principal and KI barrier must be positive")
    for enabled, trigger in zip(contract.autocall_flags, contract.autocall_trigger_ratios):
        if enabled and (trigger is None or trigger <= 0):
            raise ValueError("Every enabled autocall requires a positive trigger")
    for enabled, trigger in zip(contract.coupon_flags, contract.coupon_trigger_ratios):
        if contract.coupon_mode in {"periodic", "memory_periodic"} and enabled:
            if trigger is None or trigger <= 0:
                raise ValueError("Every enabled periodic coupon requires a positive trigger")
    if contract.coupon_mode not in {"simple_to_redemption", "periodic", "memory_periodic"}:
        raise ValueError(f"Unsupported coupon mode: {contract.coupon_mode}")


def make_psd_correlation(correlation: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    correlation = np.asarray(correlation, dtype=float)
    if correlation.shape != (3, 3):
        raise ValueError("The direct engine requires a 3x3 correlation matrix")
    correlation = 0.5 * (correlation + correlation.T)
    values, vectors = np.linalg.eigh(correlation)
    values = np.maximum(values, floor)
    repaired = (vectors * values) @ vectors.T
    scale = np.sqrt(np.diag(repaired))
    repaired = repaired / np.outer(scale, scale)
    np.fill_diagonal(repaired, 1.0)
    return repaired


def equicorrelation(rho: float) -> np.ndarray:
    correlation = np.full((3, 3), float(rho))
    np.fill_diagonal(correlation, 1.0)
    return correlation


def stable_seed(base_seed: int, *parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    offset = int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")
    return int((base_seed + offset) % (2**32 - 1))


def monitoring_grid(contract: DirectContract, steps_per_year: int) -> tuple[np.ndarray, dict[int, int]]:
    if steps_per_year <= 0:
        raise ValueError("steps_per_year must be positive")
    regular_steps = int(math.ceil(contract.maturity_years * steps_per_year))
    regular = np.linspace(0.0, contract.maturity_years, regular_steps + 1)
    grid = np.unique(np.concatenate([regular, np.asarray(contract.observation_times)]))
    observation_map: dict[int, int] = {}
    for observation_index, observation_time in enumerate(contract.observation_times):
        grid_index = int(np.argmin(np.abs(grid - observation_time)))
        if not math.isclose(float(grid[grid_index]), observation_time, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("Observation time was not retained in monitoring grid")
        observation_map[grid_index] = observation_index
    return grid, observation_map


def _new_totals(n_observations: int) -> dict[str, Any]:
    return {
        "coupon_annuity": 0.0,
        "fixed_coupon_value": 0.0,
        "early_redemption_principal": 0.0,
        "surviving_notional": 0.0,
        "no_ki_maturity_redemption": 0.0,
        "terminal_ki_loss": 0.0,
        "continuous_ki_survivors": 0,
        "maturity_reached": 0,
        "redemption_counts": np.zeros(n_observations, dtype=np.int64),
    }


def simulate_direct_replication(
    contract: DirectContract,
    risk_free_rate: float,
    dividend_yields: np.ndarray,
    volatilities: np.ndarray,
    correlation: np.ndarray,
    annual_coupon: float,
    n_paths: int,
    steps_per_year: int,
    seed: int,
    batch_size: int = 2048,
    initial_spot_ratios: np.ndarray | None = None,
) -> dict[str, Any]:
    """Simulate one fine-grid direct-MC replication.

    KI is observed at every fine-grid node until redemption.  Discrete coupon
    and autocall tests are evaluated only at exact contract observation times.
    """
    validate_contract(contract)
    if n_paths <= 1 or batch_size <= 0:
        raise ValueError("n_paths must exceed one and batch_size must be positive")
    q = np.asarray(dividend_yields, dtype=float)
    sigma = np.asarray(volatilities, dtype=float)
    if q.shape != (3,) or sigma.shape != (3,) or np.any(sigma <= 0):
        raise ValueError("dividend_yields and volatilities must be positive length-3 vectors")
    repaired_correlation = make_psd_correlation(correlation)
    cholesky = np.linalg.cholesky(repaired_correlation)
    initial_ratios = (
        np.ones(3, dtype=float)
        if initial_spot_ratios is None
        else np.asarray(initial_spot_ratios, dtype=float)
    )
    if initial_ratios.shape != (3,) or np.any(initial_ratios <= 0):
        raise ValueError("initial_spot_ratios must contain three positive values")
    initial_log_state = np.log(initial_ratios)
    grid, observation_map = monitoring_grid(contract, steps_per_year)
    discount_factors = np.exp(-risk_free_rate * np.asarray(contract.observation_times))
    maturity_discount = float(np.exp(-risk_free_rate * contract.maturity_years))
    totals = _new_totals(len(contract.observation_times))
    rng = np.random.default_rng(int(seed))
    started = time.perf_counter()

    for batch_start in range(0, n_paths, batch_size):
        batch = min(batch_size, n_paths - batch_start)
        log_state = np.broadcast_to(initial_log_state, (batch, 3)).copy()
        active = np.ones(batch, dtype=bool)
        knocked_in = np.full(
            batch, np.min(initial_ratios) <= contract.ki_barrier_ratio, dtype=bool
        )
        accrued_periods = np.zeros(batch, dtype=np.int16)
        coupon_annuity = np.zeros(batch, dtype=float)
        fixed_coupon_value = np.zeros(batch, dtype=float)
        early_principal = np.zeros(batch, dtype=float)
        surviving_notional = np.zeros(batch, dtype=float)
        no_ki_redemption = np.zeros(batch, dtype=float)
        terminal_ki_loss = np.zeros(batch, dtype=float)
        call_index = np.full(batch, -1, dtype=np.int16)
        drift = risk_free_rate - q - 0.5 * sigma**2

        for grid_index in range(1, len(grid)):
            dt = float(grid[grid_index] - grid[grid_index - 1])
            normals = rng.standard_normal((batch, 3)) @ cholesky.T
            log_state += drift * dt + sigma * math.sqrt(dt) * normals
            performance = np.exp(log_state)
            knocked_in |= active & (np.min(performance, axis=1) <= contract.ki_barrier_ratio)

            observation_index = observation_map.get(grid_index)
            if observation_index is None:
                continue
            observation_time = contract.observation_times[observation_index]
            worst = np.min(performance, axis=1)
            discount = float(discount_factors[observation_index])

            if contract.coupon_mode in {"periodic", "memory_periodic"}:
                if contract.coupon_flags[observation_index]:
                    accrued_periods[active] += 1
                    coupon_trigger = float(contract.coupon_trigger_ratios[observation_index])
                    coupon_paid = active & (worst >= coupon_trigger)
                    if contract.coupon_memory:
                        paid_periods = accrued_periods[coupon_paid]
                    else:
                        paid_periods = np.ones(int(coupon_paid.sum()), dtype=float)
                    coupon_annuity[coupon_paid] += (
                        contract.principal
                        * paid_periods
                        / contract.coupon_periods_per_year
                        * discount
                    )
                    accrued_periods[coupon_paid] = 0

            if contract.autocall_flags[observation_index]:
                trigger = float(contract.autocall_trigger_ratios[observation_index])
                called = active & (worst >= trigger)
                if np.any(called):
                    call_index[called] = observation_index
                    totals["redemption_counts"][observation_index] += int(called.sum())
                    if contract.coupon_mode == "simple_to_redemption":
                        coupon_annuity[called] += (
                            contract.principal * observation_time * discount
                        )
                    if observation_time < contract.maturity_years - 1e-12:
                        early_principal[called] += contract.principal * discount
                    else:
                        surviving_notional[called] += contract.principal * discount
                        no_ki_at_maturity = called & ~knocked_in
                        no_ki_redemption[no_ki_at_maturity] += contract.principal * discount
                    active[called] = False

        remaining = active
        totals["maturity_reached"] += int(
            np.count_nonzero((call_index < 0) | (call_index == len(contract.observation_times) - 1))
        )
        if np.any(remaining):
            terminal_worst = np.min(np.exp(log_state), axis=1)
            surviving_notional[remaining] += contract.principal * maturity_discount
            maturity_loss = remaining & knocked_in & (terminal_worst < 1.0)
            terminal_ki_loss[maturity_loss] += (
                contract.principal * (terminal_worst[maturity_loss] - 1.0) * maturity_discount
            )
            no_ki = remaining & ~knocked_in
            no_ki_redemption[no_ki] += contract.principal * maturity_discount
            if contract.coupon_mode == "simple_to_redemption":
                if contract.fixed_maturity_no_ki_coupon_rate is None:
                    coupon_annuity[no_ki] += (
                        contract.principal * contract.maturity_years * maturity_discount
                    )
                else:
                    fixed_coupon_value[no_ki] += (
                        contract.principal
                        * contract.fixed_maturity_no_ki_coupon_rate
                        * maturity_discount
                    )

        totals["coupon_annuity"] += float(coupon_annuity.sum())
        totals["fixed_coupon_value"] += float(fixed_coupon_value.sum())
        totals["early_redemption_principal"] += float(early_principal.sum())
        totals["surviving_notional"] += float(surviving_notional.sum())
        totals["no_ki_maturity_redemption"] += float(no_ki_redemption.sum())
        totals["terminal_ki_loss"] += float(terminal_ki_loss.sum())
        totals["continuous_ki_survivors"] += int((~knocked_in).sum())

    scale = 1.0 / n_paths
    coupon_annuity_mean = totals["coupon_annuity"] * scale
    fixed_coupon_mean = totals["fixed_coupon_value"] * scale
    coupon_value = fixed_coupon_mean + annual_coupon * coupon_annuity_mean
    early_principal_mean = totals["early_redemption_principal"] * scale
    surviving_mean = totals["surviving_notional"] * scale
    terminal_loss_mean = totals["terminal_ki_loss"] * scale
    base_value = fixed_coupon_mean + early_principal_mean + surviving_mean + terminal_loss_mean
    total_value = base_value + annual_coupon * coupon_annuity_mean
    additive_sum = coupon_value + early_principal_mean + surviving_mean + terminal_loss_mean
    fair_coupon = (contract.principal - base_value) / coupon_annuity_mean
    fair_coupon_residual = base_value + fair_coupon * coupon_annuity_mean - contract.principal

    result: dict[str, Any] = {
        "contract_id": contract.contract_id,
        "annual_coupon": annual_coupon,
        "coupon_annuity": coupon_annuity_mean,
        "fixed_coupon_value": fixed_coupon_mean,
        "coupon_value": coupon_value,
        "early_redemption_principal": early_principal_mean,
        "surviving_notional": surviving_mean,
        "no_ki_maturity_redemption": totals["no_ki_maturity_redemption"] * scale,
        "terminal_ki_loss": terminal_loss_mean,
        "base_value": base_value,
        "total_value": total_value,
        "additive_sum": additive_sum,
        "component_identity_error": total_value - additive_sum,
        "continuous_ki_survival_probability": totals["continuous_ki_survivors"] * scale,
        "maturity_survival_probability": totals["maturity_reached"] * scale,
        "fair_coupon": fair_coupon,
        "fair_coupon_residual": fair_coupon_residual,
        "n_paths": n_paths,
        "steps_per_year": steps_per_year,
        "grid_steps": len(grid) - 1,
        "seed": int(seed),
        "runtime_seconds": time.perf_counter() - started,
    }
    for asset_index, ratio in enumerate(initial_ratios):
        result[f"initial_spot_ratio_{asset_index + 1}"] = float(ratio)
    for observation_index, count in enumerate(totals["redemption_counts"]):
        result[f"redemption_probability_{observation_index + 1}"] = count * scale
    early_probability = sum(
        result[f"redemption_probability_{index + 1}"]
        for index, time_years in enumerate(contract.observation_times)
        if time_years < contract.maturity_years - 1e-12
    )
    result["probability_mass_error"] = (
        early_probability + result["maturity_survival_probability"] - 1.0
    )
    return result


def run_replications(
    scenario_id: str,
    contract: DirectContract,
    risk_free_rate: float,
    dividend_yields: np.ndarray,
    volatilities: np.ndarray,
    correlation: np.ndarray,
    annual_coupon: float,
    n_paths: int,
    steps_per_year: int,
    replications: int,
    base_seed: int,
    batch_size: int = 2048,
    common_random_numbers: bool = True,
) -> pd.DataFrame:
    rows = []
    for replication in range(replications):
        seed = (
            stable_seed(base_seed, "direct", replication)
            if common_random_numbers
            else stable_seed(base_seed, scenario_id, replication)
        )
        row = simulate_direct_replication(
            contract=contract,
            risk_free_rate=risk_free_rate,
            dividend_yields=dividend_yields,
            volatilities=volatilities,
            correlation=correlation,
            annual_coupon=annual_coupon,
            n_paths=n_paths,
            steps_per_year=steps_per_year,
            seed=seed,
            batch_size=batch_size,
        )
        row.update(
            {
                "scenario_id": scenario_id,
                "replication": replication,
                "risk_free_rate": risk_free_rate,
                "volatility_1": float(volatilities[0]),
                "volatility_2": float(volatilities[1]),
                "volatility_3": float(volatilities[2]),
                "correlation_12": float(correlation[0, 1]),
                "correlation_13": float(correlation[0, 2]),
                "correlation_23": float(correlation[1, 2]),
                "ki_barrier_ratio": contract.ki_barrier_ratio,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summarise_replications(replications: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "total_value",
        "fair_coupon",
        "coupon_value",
        "early_redemption_principal",
        "surviving_notional",
        "terminal_ki_loss",
        "no_ki_maturity_redemption",
        "continuous_ki_survival_probability",
        "maturity_survival_probability",
        "runtime_seconds",
    ]
    rows = []
    for scenario_id, group in replications.groupby("scenario_id", sort=False):
        row: dict[str, Any] = {
            "scenario_id": scenario_id,
            "contract_id": group["contract_id"].iloc[0],
            "replications": len(group),
            "n_paths_per_replication": int(group["n_paths"].iloc[0]),
            "steps_per_year": int(group["steps_per_year"].iloc[0]),
        }
        for metric in metrics:
            values = group[metric].astype(float)
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_sd"] = values.std(ddof=1) if len(values) > 1 else 0.0
            row[f"{metric}_se"] = row[f"{metric}_sd"] / math.sqrt(len(values))
        row["component_identity_max_abs_error"] = group["component_identity_error"].abs().max()
        row["probability_mass_max_abs_error"] = group["probability_mass_error"].abs().max()
        row["fair_coupon_max_abs_residual"] = group["fair_coupon_residual"].abs().max()
        rows.append(row)
    return pd.DataFrame(rows)


def bracketed_bisection(function, lower: float, upper: float, tolerance: float = 1e-12) -> float:
    """Dependency-free bracketed root solver used to cross-check linear C*."""
    f_lower = float(function(lower))
    f_upper = float(function(upper))
    if f_lower == 0:
        return lower
    if f_upper == 0:
        return upper
    if f_lower * f_upper > 0:
        raise ValueError("Root is not bracketed")
    for _ in range(200):
        midpoint = 0.5 * (lower + upper)
        f_midpoint = float(function(midpoint))
        if abs(f_midpoint) <= tolerance or 0.5 * (upper - lower) <= tolerance:
            return midpoint
        if f_lower * f_midpoint <= 0:
            upper = midpoint
            f_upper = f_midpoint
        else:
            lower = midpoint
            f_lower = f_midpoint
    raise RuntimeError("Bracketed root solver did not converge")

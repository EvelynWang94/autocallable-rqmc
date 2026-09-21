"""Three-asset Brownian-bridge conditioning engine used by Day 6.

The outer simulation contains only contract observation-date endpoints.  For
each interval, exact marginal and bivariate Brownian-bridge exit probabilities
are combined with a fresh nested bridge estimate of the trivariate co-exit
term.  The latter deliberately does not reuse the Day 4 logistic approximation,
whose out-of-sample validation gate did not pass.

All values are present values per contract notional.  The module keeps the
continuous knock-in probability separate from discrete coupon and autocall
events and exposes interval-level diagnostics for audit and validation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, MutableMapping

import numpy as np
import pandas as pd
from scipy.special import ive
from scipy.stats import norm, qmc

from autocallable_direct import (
    DirectContract,
    make_psd_correlation,
    monitoring_grid,
    stable_seed,
    validate_contract,
)


PAIR_SPECS = (("12", 0, 1), ("13", 0, 2), ("23", 1, 2))


@dataclass(frozen=True)
class BridgeBank:
    """Reusable conditional Brownian-bridge residual paths for one interval."""

    residuals: np.ndarray
    time_fractions: np.ndarray
    dt: float
    inner_paths: int
    substeps: int
    seed: int


def _broadcast_float(*values: Any) -> tuple[np.ndarray, ...]:
    return tuple(np.broadcast_arrays(*[np.asarray(value, dtype=float) for value in values]))


def marginal_exit(
    endpoint: np.ndarray | float,
    barrier: np.ndarray | float,
    sigma: np.ndarray | float,
    maturity: float,
) -> np.ndarray | float:
    """Lee et al. Lemma 2.1 upper-barrier exit probability."""

    endpoint, barrier, sigma = _broadcast_float(endpoint, barrier, sigma)
    if maturity <= 0 or np.any(sigma <= 0) or np.any(barrier < 0):
        raise ValueError("maturity and sigma must be positive; barrier non-negative")
    exponent = -2.0 * barrier * (barrier - endpoint) / (sigma * sigma * maturity)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = np.where(endpoint >= barrier, 1.0, np.exp(exponent))
    result = np.clip(result, 0.0, 1.0)
    return float(result) if result.ndim == 0 else result


def bivariate_coexit(
    endpoint_i: np.ndarray | float,
    endpoint_j: np.ndarray | float,
    barrier_i: np.ndarray | float,
    barrier_j: np.ndarray | float,
    sigma_i: float,
    sigma_j: float,
    rho: float,
    maturity: float,
    n_terms: int = 12,
) -> np.ndarray | float:
    """Lee et al. Proposition 2.1, vectorised over endpoints and barriers."""

    if not (-1.0 < rho < 1.0):
        raise ValueError("rho must lie strictly between -1 and 1")
    if maturity <= 0 or sigma_i <= 0 or sigma_j <= 0 or n_terms <= 0:
        raise ValueError("maturity, volatilities and n_terms must be positive")
    endpoint_i, endpoint_j, barrier_i, barrier_j = _broadcast_float(
        endpoint_i, endpoint_j, barrier_i, barrier_j
    )
    if np.any(barrier_i < 0) or np.any(barrier_j < 0):
        raise ValueError("upper barriers must be non-negative")
    scalar = endpoint_i.ndim == 0
    xi = np.atleast_1d(endpoint_i)
    xj = np.atleast_1d(endpoint_j)
    bi = np.atleast_1d(barrier_i)
    bj = np.atleast_1d(barrier_j)
    gi = np.atleast_1d(marginal_exit(xi, bi, sigma_i, maturity))
    gj = np.atleast_1d(marginal_exit(xj, bj, sigma_j, maturity))

    # Independence has a simple exact form and is a required Day 6 gate.
    if abs(rho) <= 1e-14:
        independent = gi * gj
        return float(independent[0]) if scalar else independent

    result = np.empty_like(xi, dtype=float)
    both_above = (xi >= bi) & (xj >= bj)
    only_i_above = (xi >= bi) & (xj < bj)
    only_j_above = (xi < bi) & (xj >= bj)
    both_below = ~(both_above | only_i_above | only_j_above)
    result[both_above] = 1.0
    result[only_i_above] = gj[only_i_above]
    result[only_j_above] = gi[only_j_above]

    if np.any(both_below):
        xib = xi[both_below]
        xjb = xj[both_below]
        bib = bi[both_below]
        bjb = bj[both_below]
        root = math.sqrt(1.0 - rho * rho)
        beta = math.acos(-rho)

        z1 = ((xib - bib) / sigma_i - rho * (xjb - bjb) / sigma_j) / root
        z2 = (xjb - bjb) / sigma_j
        z10 = (-bib / sigma_i + rho * bjb / sigma_j) / root
        z20 = -bjb / sigma_j
        d = np.hypot(z1, z2)
        d0 = np.hypot(z10, z20)
        theta = np.clip(np.arctan2(-z2, -z1), 0.0, beta)
        theta0 = np.clip(np.arctan2(-z20, -z10), 0.0, beta)
        argument = d * d0 / maturity
        series = np.zeros_like(argument)
        for term_index in range(1, int(n_terms) + 1):
            order = term_index * math.pi / beta
            angular = np.sin(term_index * math.pi * theta0 / beta) * np.sin(
                term_index * math.pi * theta / beta
            )
            series += angular * ive(order, argument)
        exponent = argument - (z1 * z10 + z2 * z20) / maturity
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            joint_nonexit = 4.0 * math.pi / beta * np.exp(exponent) * series
        result[both_below] = gi[both_below] + gj[both_below] + joint_nonexit - 1.0

    result = np.clip(result, 0.0, np.minimum(gi, gj))
    if np.any(~np.isfinite(result)):
        raise FloatingPointError("non-finite bivariate probability")
    return float(result[0]) if scalar else result


def trivariate_probability_bounds(
    marginal: np.ndarray, pairwise: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Feasible lower/upper bounds for a three-event intersection."""

    g = np.asarray(marginal, dtype=float)
    h = np.asarray(pairwise, dtype=float)
    if g.shape[-1] != 3 or h.shape[-1] != 3:
        raise ValueError("marginal and pairwise arrays must end in length three")
    g1, g2, g3 = np.moveaxis(g, -1, 0)
    h12, h13, h23 = np.moveaxis(h, -1, 0)
    lower = np.maximum.reduce(
        [
            np.zeros_like(g1),
            h12 + h13 - g1,
            h12 + h23 - g2,
            h13 + h23 - g3,
            h12 + h13 + h23 - g1 - g2 - g3,
        ]
    )
    upper = np.minimum.reduce(
        [
            h12,
            h13,
            h23,
            1.0 - g1 - g2 - g3 + h12 + h13 + h23,
        ]
    )
    upper = np.maximum(upper, lower)
    return np.clip(lower, 0.0, 1.0), np.clip(upper, 0.0, 1.0)


def generate_bridge_bank(
    dt: float,
    volatilities: np.ndarray,
    correlation: np.ndarray,
    inner_paths: int,
    substeps: int,
    seed: int,
    method: str = "rqmc",
) -> BridgeBank:
    """Generate endpoint-conditioned residual bridges on the unit interval."""

    sigma = np.asarray(volatilities, dtype=float)
    corr = make_psd_correlation(correlation)
    if dt <= 0 or sigma.shape != (3,) or np.any(sigma <= 0):
        raise ValueError("dt and three volatilities must be positive")
    if inner_paths <= 1 or substeps < 2:
        raise ValueError("inner_paths must exceed one and substeps must be at least two")
    dimension = substeps * 3
    if method == "rqmc":
        sampler = qmc.Sobol(d=dimension, scramble=True, seed=int(seed))
        if inner_paths & (inner_paths - 1) == 0:
            uniforms = sampler.random_base2(int(math.log2(inner_paths)))
        else:
            uniforms = sampler.random(inner_paths)
        normals = norm.ppf(np.clip(uniforms, 1e-12, 1.0 - 1e-12))
    elif method == "mc":
        normals = np.random.default_rng(int(seed)).standard_normal((inner_paths, dimension))
    else:
        raise ValueError("method must be 'mc' or 'rqmc'")
    normals = normals.reshape(inner_paths, substeps, 3)
    correlated = normals @ np.linalg.cholesky(corr).T
    increments = correlated / math.sqrt(substeps)
    brownian = np.cumsum(increments, axis=1)
    fractions = np.arange(1, substeps, dtype=float) / substeps
    bridge = brownian[:, :-1, :] - fractions[None, :, None] * brownian[:, -1:, :]
    residuals = bridge * (sigma * math.sqrt(dt))[None, None, :]
    return BridgeBank(
        residuals=residuals,
        time_fractions=fractions,
        dt=float(dt),
        inner_paths=int(inner_paths),
        substeps=int(substeps),
        seed=int(seed),
    )


def nested_exit_components(
    endpoints: np.ndarray,
    barriers: np.ndarray,
    bank: BridgeBank,
    chunk_size: int = 256,
) -> dict[str, np.ndarray]:
    """Nested estimates of marginal, pair and triple exits and direct survival."""

    x = np.atleast_2d(np.asarray(endpoints, dtype=float))
    b = np.asarray(barriers, dtype=float)
    if b.ndim == 1:
        b = np.broadcast_to(b, x.shape)
    if x.shape != b.shape or x.shape[1] != 3 or np.any(b < 0):
        raise ValueError("endpoints and barriers must be aligned n by 3 arrays")
    n = len(x)
    g_nested = np.empty((n, 3), dtype=float)
    h_nested = np.empty((n, 3), dtype=float)
    q_nested = np.empty(n, dtype=float)
    p_direct = np.empty(n, dtype=float)
    u = bank.time_fractions
    residuals = bank.residuals
    for start in range(0, n, chunk_size):
        stop = min(n, start + chunk_size)
        xc = x[start:stop]
        bc = b[start:stop]
        linear = xc[:, None, None, :] * u[None, None, :, None]
        values = linear + residuals[None, :, :, :]
        exits = np.any(values >= bc[:, None, None, :], axis=2)
        deterministic = (bc <= 0.0) | (xc >= bc)
        exits |= deterministic[:, None, :]
        g_nested[start:stop] = exits.mean(axis=1)
        for column, (_, i, j) in enumerate(PAIR_SPECS):
            h_nested[start:stop, column] = (exits[:, :, i] & exits[:, :, j]).mean(axis=1)
        q_nested[start:stop] = np.all(exits, axis=2).mean(axis=1)
        p_direct[start:stop] = (~np.any(exits, axis=2)).mean(axis=1)
    return {
        "g_nested": g_nested,
        "h_nested": h_nested,
        "q3_nested_direct": q_nested,
        "p3_nested_direct": p_direct,
    }


def segment_probabilities(
    endpoints: np.ndarray,
    barriers: np.ndarray,
    dt: float,
    volatilities: np.ndarray,
    correlation: np.ndarray,
    bank: BridgeBank | None,
    bessel_terms: int = 12,
) -> dict[str, Any]:
    """Compute exact g/h, nested q3 and inclusion-exclusion p3 for one interval."""

    x = np.atleast_2d(np.asarray(endpoints, dtype=float))
    b = np.asarray(barriers, dtype=float)
    if b.ndim == 1:
        b = np.broadcast_to(b, x.shape)
    sigma = np.asarray(volatilities, dtype=float)
    corr = make_psd_correlation(correlation)
    if x.shape != b.shape or x.shape[1] != 3:
        raise ValueError("endpoints and barriers must be aligned n by 3 arrays")

    started_g = time.perf_counter()
    g = np.column_stack(
        [marginal_exit(x[:, i], b[:, i], sigma[i], dt) for i in range(3)]
    )
    runtime_g = time.perf_counter() - started_g

    started_h = time.perf_counter()
    h = np.empty((len(x), 3), dtype=float)
    for column, (_, i, j) in enumerate(PAIR_SPECS):
        h[:, column] = bivariate_coexit(
            x[:, i], x[:, j], b[:, i], b[:, j], sigma[i], sigma[j], corr[i, j], dt,
            n_terms=bessel_terms,
        )
    runtime_h = time.perf_counter() - started_h

    independent = np.max(np.abs(corr[np.triu_indices(3, 1)])) <= 1e-14
    started_q = time.perf_counter()
    if independent:
        nested = {
            "g_nested": g.copy(),
            "h_nested": np.column_stack((g[:, 0] * g[:, 1], g[:, 0] * g[:, 2], g[:, 1] * g[:, 2])),
            "q3_nested_direct": np.prod(g, axis=1),
            "p3_nested_direct": np.prod(1.0 - g, axis=1),
        }
        q_controlled_raw = nested["q3_nested_direct"].copy()
    else:
        if bank is None:
            raise ValueError("a BridgeBank is required outside the independent case")
        nested = nested_exit_components(x, b, bank)
        ratios = np.divide(
            nested["q3_nested_direct"][:, None],
            nested["h_nested"],
            out=np.zeros_like(nested["h_nested"]),
            where=nested["h_nested"] > 0,
        )
        ratios = np.clip(ratios, 0.0, 1.0)
        candidates = h * ratios
        valid = nested["h_nested"] > 0
        # Three equivalent pair-conditionings are available.  Their median is
        # robust to a single noisy low-probability pair ratio while preserving
        # the symmetry between asset labels.  Rows without a valid pair fall
        # back to the direct nested triple-event frequency.
        ordered = np.sort(np.where(valid, candidates, np.inf), axis=1)
        valid_count = valid.sum(axis=1)
        q_controlled_raw = nested["q3_nested_direct"].copy()
        q_controlled_raw[valid_count == 1] = ordered[valid_count == 1, 0]
        q_controlled_raw[valid_count == 2] = 0.5 * (
            ordered[valid_count == 2, 0] + ordered[valid_count == 2, 1]
        )
        q_controlled_raw[valid_count == 3] = ordered[valid_count == 3, 1]
        # If one or more assets has already exited at an interval endpoint, the
        # trivariate term reduces exactly to a lower-dimensional component.
        # Applying this identity avoids treating deterministic endpoint exits
        # as nested-approximation error (important for the close RC-A barrier).
        deterministic = (b <= 0.0) | (x >= b)
        count = deterministic.sum(axis=1)
        one = count == 1
        q_controlled_raw[one & deterministic[:, 0]] = h[one & deterministic[:, 0], 2]
        q_controlled_raw[one & deterministic[:, 1]] = h[one & deterministic[:, 1], 1]
        q_controlled_raw[one & deterministic[:, 2]] = h[one & deterministic[:, 2], 0]
        two = count == 2
        q_controlled_raw[two & ~deterministic[:, 0]] = g[two & ~deterministic[:, 0], 0]
        q_controlled_raw[two & ~deterministic[:, 1]] = g[two & ~deterministic[:, 1], 1]
        q_controlled_raw[two & ~deterministic[:, 2]] = g[two & ~deterministic[:, 2], 2]
        q_controlled_raw[count == 3] = 1.0
    runtime_q = time.perf_counter() - started_q

    lower, upper = trivariate_probability_bounds(g, h)
    q3 = np.clip(q_controlled_raw, lower, upper)
    raw_p3 = 1.0 - g.sum(axis=1) + h.sum(axis=1) - q_controlled_raw
    p3 = 1.0 - g.sum(axis=1) + h.sum(axis=1) - q3
    p3 = np.clip(p3, 0.0, 1.0)
    q3_clipped = np.abs(q3 - q_controlled_raw) > 1e-12
    result: dict[str, Any] = {
        "g": g,
        "h": h,
        "g_nested": nested["g_nested"],
        "h_nested": nested["h_nested"],
        "q3_nested_direct": nested["q3_nested_direct"],
        "q3_controlled_raw": q_controlled_raw,
        "q3_lower": lower,
        "q3_upper": upper,
        "q3": q3,
        "p3_raw": raw_p3,
        "p3": p3,
        "p3_nested_direct": nested["p3_nested_direct"],
        "q3_clipped": q3_clipped,
        "endpoint_breach": np.any((b <= 0.0) | (x >= b), axis=1),
        "runtime_g_seconds": runtime_g,
        "runtime_h_seconds": runtime_h,
        "runtime_q3_seconds": runtime_q,
        "independence_shortcut": independent,
    }
    return result


def generate_endpoint_paths(
    n_paths: int,
    observation_times: tuple[float, ...],
    risk_free_rate: float,
    dividend_yields: np.ndarray,
    volatilities: np.ndarray,
    correlation: np.ndarray,
    method: str,
    seed: int,
    initial_spot_ratios: np.ndarray | None = None,
) -> np.ndarray:
    """Simulate correlated GBM log endpoints at contract observation dates."""

    q_yield = np.asarray(dividend_yields, dtype=float)
    sigma = np.asarray(volatilities, dtype=float)
    corr = make_psd_correlation(correlation)
    initial_ratios = (
        np.ones(3, dtype=float)
        if initial_spot_ratios is None
        else np.asarray(initial_spot_ratios, dtype=float)
    )
    if initial_ratios.shape != (3,) or np.any(initial_ratios <= 0):
        raise ValueError("initial_spot_ratios must contain three positive values")
    n_obs = len(observation_times)
    dimension = n_obs * 3
    if method == "mc":
        normals = np.random.default_rng(int(seed)).standard_normal((n_paths, dimension))
    elif method == "rqmc":
        sampler = qmc.Sobol(d=dimension, scramble=True, seed=int(seed))
        if n_paths & (n_paths - 1) == 0:
            uniforms = sampler.random_base2(int(math.log2(n_paths)))
        else:
            uniforms = sampler.random(n_paths)
        normals = norm.ppf(np.clip(uniforms, 1e-12, 1.0 - 1e-12))
    else:
        raise ValueError("method must be 'mc' or 'rqmc'")
    normals = normals.reshape(n_paths, n_obs, 3) @ np.linalg.cholesky(corr).T
    times = np.asarray(observation_times, dtype=float)
    dts = np.diff(np.concatenate([[0.0], times]))
    drift = risk_free_rate - q_yield - 0.5 * sigma**2
    increments = (
        drift[None, None, :] * dts[None, :, None]
        + sigma[None, None, :] * np.sqrt(dts)[None, :, None] * normals
    )
    return np.log(initial_ratios)[None, None, :] + np.cumsum(increments, axis=1)


def _bank_cache_key(
    dt: float,
    sigma: np.ndarray,
    correlation: np.ndarray,
    inner_paths: int,
    substeps: int,
    seed: int,
) -> tuple[Any, ...]:
    return (
        round(float(dt), 14),
        tuple(np.round(sigma, 14)),
        tuple(np.round(correlation.ravel(), 14)),
        int(inner_paths),
        int(substeps),
        int(seed),
    )


def simulate_conditioned_replication(
    contract: DirectContract,
    risk_free_rate: float,
    dividend_yields: np.ndarray,
    volatilities: np.ndarray,
    correlation: np.ndarray,
    annual_coupon: float,
    n_paths: int,
    outer_method: str,
    seed: int,
    bridge_bank_seed: int,
    inner_paths: int = 256,
    bridge_substeps: int = 16,
    bessel_terms: int = 12,
    bank_cache: MutableMapping[tuple[Any, ...], BridgeBank] | None = None,
    paired_bernoulli: bool = True,
    return_diagnostics: bool = False,
    initial_spot_ratios: np.ndarray | None = None,
) -> dict[str, Any] | tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Price one M2/M3 replication using observation-date endpoints only."""

    validate_contract(contract)
    if n_paths <= 1:
        raise ValueError("n_paths must exceed one")
    sigma = np.asarray(volatilities, dtype=float)
    corr = make_psd_correlation(correlation)
    q_yield = np.asarray(dividend_yields, dtype=float)
    initial_ratios = (
        np.ones(3, dtype=float)
        if initial_spot_ratios is None
        else np.asarray(initial_spot_ratios, dtype=float)
    )
    if initial_ratios.shape != (3,) or np.any(initial_ratios <= 0):
        raise ValueError("initial_spot_ratios must contain three positive values")
    started = time.perf_counter()
    endpoints = generate_endpoint_paths(
        n_paths, contract.observation_times, risk_free_rate, q_yield, sigma, corr,
        method=outer_method, seed=seed, initial_spot_ratios=initial_ratios,
    )
    discounts = np.exp(-risk_free_rate * np.asarray(contract.observation_times))
    maturity_discount = float(discounts[-1])
    log_barrier = math.log(contract.ki_barrier_ratio)
    cache: MutableMapping[tuple[Any, ...], BridgeBank] = {} if bank_cache is None else bank_cache

    active = np.ones(n_paths, dtype=bool)
    cumulative_survival = np.ones(n_paths, dtype=float)
    cumulative_log_survival = np.zeros(n_paths, dtype=float)
    final_survival = np.full(n_paths, np.nan)
    bernoulli_survival = np.ones(n_paths, dtype=bool)
    bernoulli_rng = np.random.default_rng(stable_seed(seed, "paired-bernoulli"))
    accrued_periods = np.zeros(n_paths, dtype=np.int16)
    coupon_annuity_common = np.zeros(n_paths)
    fixed_coupon_common = np.zeros(n_paths)
    early_principal = np.zeros(n_paths)
    surviving_notional = np.zeros(n_paths)
    no_ki_redemption = np.zeros(n_paths)
    terminal_loss = np.zeros(n_paths)
    terminal_loss_bernoulli = np.zeros(n_paths)
    redemption_counts = np.zeros(len(contract.observation_times), dtype=np.int64)
    segment_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    previous = np.broadcast_to(np.log(initial_ratios), (n_paths, 3)).copy()
    previous_time = 0.0

    for observation_index, observation_time in enumerate(contract.observation_times):
        current = endpoints[:, observation_index, :]
        dt = float(observation_time - previous_time)
        active_index = np.flatnonzero(active)
        if len(active_index):
            start_state = previous[active_index]
            end_state = current[active_index]
            # Corollary 5.2: lower-barrier survival becomes upper-barrier survival
            # with endpoint x = start-end and barrier b = start-log(KI).
            transformed_endpoint = start_state - end_state
            transformed_barrier = np.maximum(start_state - log_barrier, 0.0)
            interval_seed = stable_seed(bridge_bank_seed, round(dt, 14))
            key = _bank_cache_key(dt, sigma, corr, inner_paths, bridge_substeps, interval_seed)
            independent = np.max(np.abs(corr[np.triu_indices(3, 1)])) <= 1e-14
            if independent:
                bank = None
            else:
                if key not in cache:
                    cache[key] = generate_bridge_bank(
                        dt, sigma, corr, inner_paths, bridge_substeps, interval_seed, method="rqmc"
                    )
                bank = cache[key]
            probabilities = segment_probabilities(
                transformed_endpoint, transformed_barrier, dt, sigma, corr, bank,
                bessel_terms=bessel_terms,
            )
            p3 = probabilities["p3"]
            cumulative_survival[active_index] *= p3
            with np.errstate(divide="ignore"):
                cumulative_log_survival[active_index] += np.log(p3)
            if paired_bernoulli:
                draws = bernoulli_rng.random(len(active_index))
                bernoulli_survival[active_index] &= draws < p3

            g = probabilities["g"]
            h = probabilities["h"]
            finite_logs = np.log(np.maximum(p3, np.finfo(float).tiny))
            segment_rows.append(
                {
                    "observation_index": observation_index + 1,
                    "observation_time": observation_time,
                    "dt": dt,
                    "active_paths": len(active_index),
                    "g1_mean": g[:, 0].mean(),
                    "g2_mean": g[:, 1].mean(),
                    "g3_mean": g[:, 2].mean(),
                    "h12_mean": h[:, 0].mean(),
                    "h13_mean": h[:, 1].mean(),
                    "h23_mean": h[:, 2].mean(),
                    "q3_nested_direct_mean": probabilities["q3_nested_direct"].mean(),
                    "q3_controlled_raw_mean": probabilities["q3_controlled_raw"].mean(),
                    "q3_mean": probabilities["q3"].mean(),
                    "p3_mean": p3.mean(),
                    "p3_min": p3.min(),
                    "p3_max": p3.max(),
                    "p3_nested_direct_mean": probabilities["p3_nested_direct"].mean(),
                    "raw_probability_bound_violation_rate": probabilities["q3_clipped"].mean(),
                    "q3_projection_abs_mean": np.abs(
                        probabilities["q3"] - probabilities["q3_controlled_raw"]
                    ).mean(),
                    "q3_projection_abs_max": np.abs(
                        probabilities["q3"] - probabilities["q3_controlled_raw"]
                    ).max(),
                    "q3_projection_abs_p99": np.quantile(
                        np.abs(probabilities["q3"] - probabilities["q3_controlled_raw"]),
                        0.99,
                    ),
                    "final_probability_bound_violation_rate": float(
                        np.mean((p3 < 0.0) | (p3 > 1.0))
                    ),
                    "endpoint_breach_rate": probabilities["endpoint_breach"].mean(),
                    "mean_log_segment_weight": finite_logs.mean(),
                    "zero_segment_weight_count": int(np.count_nonzero(p3 <= np.finfo(float).tiny)),
                    "numerical_underflow_count": int(
                        np.count_nonzero(
                            (p3 <= np.finfo(float).tiny) & ~probabilities["endpoint_breach"]
                        )
                    ),
                    "runtime_g_seconds": probabilities["runtime_g_seconds"],
                    "runtime_h_seconds": probabilities["runtime_h_seconds"],
                    "runtime_q3_seconds": probabilities["runtime_q3_seconds"],
                    "independence_shortcut": probabilities["independence_shortcut"],
                }
            )
            weights_now = cumulative_survival[active_index]
            for quantile in (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0):
                weight_rows.append(
                    {
                        "observation_index": observation_index + 1,
                        "observation_time": observation_time,
                        "quantile": quantile,
                        "cumulative_survival_weight": float(np.quantile(weights_now, quantile)),
                        "finite_log_weight": float(
                            np.quantile(
                                np.maximum(cumulative_log_survival[active_index],
                                           math.log(np.finfo(float).tiny)),
                                quantile,
                            )
                        ),
                    }
                )

        performance = np.exp(current)
        worst = np.min(performance, axis=1)
        discount = float(discounts[observation_index])
        if contract.coupon_mode in {"periodic", "memory_periodic"} and contract.coupon_flags[observation_index]:
            accrued_periods[active] += 1
            trigger = float(contract.coupon_trigger_ratios[observation_index])
            paid = active & (worst >= trigger)
            paid_periods = accrued_periods[paid] if contract.coupon_memory else np.ones(paid.sum())
            coupon_annuity_common[paid] += (
                contract.principal * paid_periods / contract.coupon_periods_per_year * discount
            )
            accrued_periods[paid] = 0

        if contract.autocall_flags[observation_index]:
            trigger = float(contract.autocall_trigger_ratios[observation_index])
            called = active & (worst >= trigger)
            if np.any(called):
                redemption_counts[observation_index] += int(called.sum())
                if contract.coupon_mode == "simple_to_redemption":
                    coupon_annuity_common[called] += contract.principal * observation_time * discount
                if observation_time < contract.maturity_years - 1e-12:
                    early_principal[called] += contract.principal * discount
                else:
                    surviving_notional[called] += contract.principal * discount
                    no_ki_redemption[called] += (
                        contract.principal * discount * cumulative_survival[called]
                    )
                final_survival[called] = cumulative_survival[called]
                active[called] = False
        previous = current
        previous_time = observation_time

    remaining = active
    maturity_reached = int(n_paths - redemption_counts[:-1].sum())
    coupon_annuity = coupon_annuity_common.copy()
    coupon_annuity_bernoulli = coupon_annuity_common.copy()
    fixed_coupon_value = fixed_coupon_common.copy()
    fixed_coupon_bernoulli = fixed_coupon_common.copy()
    if np.any(remaining):
        terminal_worst = np.min(np.exp(endpoints[:, -1, :]), axis=1)
        surviving_notional[remaining] += contract.principal * maturity_discount
        loss_candidates = remaining & (terminal_worst < 1.0)
        terminal_loss[loss_candidates] = (
            contract.principal
            * (terminal_worst[loss_candidates] - 1.0)
            * maturity_discount
            * (1.0 - cumulative_survival[loss_candidates])
        )
        terminal_loss_bernoulli[loss_candidates] = (
            contract.principal
            * (terminal_worst[loss_candidates] - 1.0)
            * maturity_discount
            * (~bernoulli_survival[loss_candidates])
        )
        no_ki_redemption[remaining] += (
            contract.principal * maturity_discount * cumulative_survival[remaining]
        )
        if contract.coupon_mode == "simple_to_redemption":
            if contract.fixed_maturity_no_ki_coupon_rate is None:
                coefficient = contract.principal * contract.maturity_years * maturity_discount
                coupon_annuity[remaining] += coefficient * cumulative_survival[remaining]
                coupon_annuity_bernoulli[remaining] += coefficient * bernoulli_survival[remaining]
            else:
                coefficient = (
                    contract.principal * contract.fixed_maturity_no_ki_coupon_rate * maturity_discount
                )
                fixed_coupon_value[remaining] += coefficient * cumulative_survival[remaining]
                fixed_coupon_bernoulli[remaining] += coefficient * bernoulli_survival[remaining]
        final_survival[remaining] = cumulative_survival[remaining]

    if np.any(~np.isfinite(final_survival)):
        raise RuntimeError("final survival weight was not assigned")
    scale = 1.0 / n_paths
    coupon_annuity_mean = coupon_annuity.mean()
    fixed_coupon_mean = fixed_coupon_value.mean()
    early_mean = early_principal.mean()
    surviving_mean = surviving_notional.mean()
    terminal_loss_mean = terminal_loss.mean()
    base_value = fixed_coupon_mean + early_mean + surviving_mean + terminal_loss_mean
    total_value = base_value + annual_coupon * coupon_annuity_mean
    fair_coupon = (contract.principal - base_value) / coupon_annuity_mean
    coupon_value = fixed_coupon_mean + annual_coupon * coupon_annuity_mean

    bern_coupon_annuity_mean = coupon_annuity_bernoulli.mean()
    bern_base = (
        fixed_coupon_bernoulli.mean() + early_mean + surviving_mean + terminal_loss_bernoulli.mean()
    )
    bern_total = bern_base + annual_coupon * bern_coupon_annuity_mean
    result: dict[str, Any] = {
        "contract_id": contract.contract_id,
        "method": "M2" if outer_method == "mc" else "M3",
        "outer_method": outer_method,
        "annual_coupon": annual_coupon,
        "coupon_annuity": coupon_annuity_mean,
        "fixed_coupon_value": fixed_coupon_mean,
        "coupon_value": coupon_value,
        "early_redemption_principal": early_mean,
        "surviving_notional": surviving_mean,
        "no_ki_maturity_redemption": no_ki_redemption.mean(),
        "terminal_ki_loss": terminal_loss_mean,
        "base_value": base_value,
        "total_value": total_value,
        "additive_sum": coupon_value + early_mean + surviving_mean + terminal_loss_mean,
        "component_identity_error": total_value
        - (coupon_value + early_mean + surviving_mean + terminal_loss_mean),
        "continuous_ki_survival_probability": final_survival.mean(),
        "maturity_survival_probability": maturity_reached * scale,
        "fair_coupon": fair_coupon,
        "fair_coupon_residual": base_value + fair_coupon * coupon_annuity_mean - contract.principal,
        "paired_bernoulli_total_value": bern_total,
        "conditional_minus_paired_bernoulli": total_value - bern_total,
        "n_paths": n_paths,
        "endpoint_dimension": len(contract.observation_times) * 3,
        "inner_paths": inner_paths,
        "bridge_substeps": bridge_substeps,
        "bessel_terms": bessel_terms,
        "seed": int(seed),
        "bridge_bank_seed": int(bridge_bank_seed),
        "runtime_seconds": time.perf_counter() - started,
    }
    for asset_index, ratio in enumerate(initial_ratios):
        result[f"initial_spot_ratio_{asset_index + 1}"] = float(ratio)
    for index, count in enumerate(redemption_counts):
        result[f"redemption_probability_{index + 1}"] = count * scale
    early_probability = sum(
        result[f"redemption_probability_{index + 1}"]
        for index, time_years in enumerate(contract.observation_times)
        if time_years < contract.maturity_years - 1e-12
    )
    result["probability_mass_error"] = early_probability + result["maturity_survival_probability"] - 1.0
    if return_diagnostics:
        return result, pd.DataFrame(segment_rows), pd.DataFrame(weight_rows)
    return result


def simulate_direct_rqmc_replication(
    contract: DirectContract,
    risk_free_rate: float,
    dividend_yields: np.ndarray,
    volatilities: np.ndarray,
    correlation: np.ndarray,
    annual_coupon: float,
    n_paths: int,
    steps_per_year: int,
    seed: int,
    initial_spot_ratios: np.ndarray | None = None,
) -> dict[str, Any]:
    """Direct fine-grid scrambled-Sobol estimator (M1) for the Day 6 cross-check."""

    validate_contract(contract)
    sigma = np.asarray(volatilities, dtype=float)
    q_yield = np.asarray(dividend_yields, dtype=float)
    corr = make_psd_correlation(correlation)
    initial_ratios = (
        np.ones(3, dtype=float)
        if initial_spot_ratios is None
        else np.asarray(initial_spot_ratios, dtype=float)
    )
    if initial_ratios.shape != (3,) or np.any(initial_ratios <= 0):
        raise ValueError("initial_spot_ratios must contain three positive values")
    grid, observation_map = monitoring_grid(contract, steps_per_year)
    grid_steps = len(grid) - 1
    dimension = grid_steps * 3
    started = time.perf_counter()
    sampler = qmc.Sobol(d=dimension, scramble=True, seed=int(seed))
    if n_paths & (n_paths - 1) == 0:
        uniforms = sampler.random_base2(int(math.log2(n_paths)))
    else:
        uniforms = sampler.random(n_paths)
    normals = norm.ppf(np.clip(uniforms, 1e-12, 1.0 - 1e-12)).reshape(n_paths, grid_steps, 3)
    normals = normals @ np.linalg.cholesky(corr).T
    discounts = np.exp(-risk_free_rate * np.asarray(contract.observation_times))
    maturity_discount = float(discounts[-1])
    log_state = np.broadcast_to(np.log(initial_ratios), (n_paths, 3)).copy()
    active = np.ones(n_paths, dtype=bool)
    knocked_in = np.full(
        n_paths, np.min(initial_ratios) <= contract.ki_barrier_ratio, dtype=bool
    )
    accrued = np.zeros(n_paths, dtype=np.int16)
    coupon_annuity = np.zeros(n_paths)
    fixed_coupon = np.zeros(n_paths)
    early_principal = np.zeros(n_paths)
    surviving = np.zeros(n_paths)
    no_ki_redemption = np.zeros(n_paths)
    terminal_loss = np.zeros(n_paths)
    redemptions = np.zeros(len(contract.observation_times), dtype=np.int64)
    drift = risk_free_rate - q_yield - 0.5 * sigma**2
    for step in range(grid_steps):
        dt = float(grid[step + 1] - grid[step])
        log_state += drift * dt + sigma * math.sqrt(dt) * normals[:, step, :]
        performance = np.exp(log_state)
        knocked_in |= active & (np.min(performance, axis=1) <= contract.ki_barrier_ratio)
        observation_index = observation_map.get(step + 1)
        if observation_index is None:
            continue
        worst = np.min(performance, axis=1)
        discount = float(discounts[observation_index])
        observation_time = contract.observation_times[observation_index]
        if contract.coupon_mode in {"periodic", "memory_periodic"} and contract.coupon_flags[observation_index]:
            accrued[active] += 1
            paid = active & (worst >= float(contract.coupon_trigger_ratios[observation_index]))
            paid_periods = accrued[paid] if contract.coupon_memory else np.ones(paid.sum())
            coupon_annuity[paid] += (
                contract.principal * paid_periods / contract.coupon_periods_per_year * discount
            )
            accrued[paid] = 0
        if contract.autocall_flags[observation_index]:
            called = active & (worst >= float(contract.autocall_trigger_ratios[observation_index]))
            redemptions[observation_index] += int(called.sum())
            if contract.coupon_mode == "simple_to_redemption":
                coupon_annuity[called] += contract.principal * observation_time * discount
            if observation_time < contract.maturity_years - 1e-12:
                early_principal[called] += contract.principal * discount
            else:
                surviving[called] += contract.principal * discount
                no_ki_redemption[called & ~knocked_in] += contract.principal * discount
            active[called] = False
    remaining = active
    terminal_worst = np.min(np.exp(log_state), axis=1)
    surviving[remaining] += contract.principal * maturity_discount
    loss = remaining & knocked_in & (terminal_worst < 1.0)
    terminal_loss[loss] += contract.principal * (terminal_worst[loss] - 1.0) * maturity_discount
    no_ki = remaining & ~knocked_in
    no_ki_redemption[no_ki] += contract.principal * maturity_discount
    if contract.coupon_mode == "simple_to_redemption":
        if contract.fixed_maturity_no_ki_coupon_rate is None:
            coupon_annuity[no_ki] += contract.principal * contract.maturity_years * maturity_discount
        else:
            fixed_coupon[no_ki] += (
                contract.principal * contract.fixed_maturity_no_ki_coupon_rate * maturity_discount
            )
    coupon_annuity_mean = coupon_annuity.mean()
    fixed_mean = fixed_coupon.mean()
    early_mean = early_principal.mean()
    surviving_mean = surviving.mean()
    loss_mean = terminal_loss.mean()
    base = fixed_mean + early_mean + surviving_mean + loss_mean
    coupon_value = fixed_mean + annual_coupon * coupon_annuity_mean
    total = base + annual_coupon * coupon_annuity_mean
    fair_coupon = (contract.principal - base) / coupon_annuity_mean
    result: dict[str, Any] = {
        "contract_id": contract.contract_id,
        "method": "M1",
        "outer_method": "rqmc-direct",
        "annual_coupon": annual_coupon,
        "coupon_annuity": coupon_annuity_mean,
        "fixed_coupon_value": fixed_mean,
        "coupon_value": coupon_value,
        "early_redemption_principal": early_mean,
        "surviving_notional": surviving_mean,
        "no_ki_maturity_redemption": no_ki_redemption.mean(),
        "terminal_ki_loss": loss_mean,
        "base_value": base,
        "total_value": total,
        "additive_sum": coupon_value + early_mean + surviving_mean + loss_mean,
        "component_identity_error": total - (coupon_value + early_mean + surviving_mean + loss_mean),
        "continuous_ki_survival_probability": (~knocked_in).mean(),
        "maturity_survival_probability": (n_paths - redemptions[:-1].sum()) / n_paths,
        "fair_coupon": fair_coupon,
        "fair_coupon_residual": base + fair_coupon * coupon_annuity_mean - contract.principal,
        "n_paths": n_paths,
        "steps_per_year": steps_per_year,
        "grid_steps": grid_steps,
        "endpoint_dimension": dimension,
        "seed": int(seed),
        "runtime_seconds": time.perf_counter() - started,
    }
    for asset_index, ratio in enumerate(initial_ratios):
        result[f"initial_spot_ratio_{asset_index + 1}"] = float(ratio)
    for index, count in enumerate(redemptions):
        result[f"redemption_probability_{index + 1}"] = count / n_paths
    early_probability = sum(
        result[f"redemption_probability_{index + 1}"]
        for index, time_years in enumerate(contract.observation_times)
        if time_years < contract.maturity_years - 1e-12
    )
    result["probability_mass_error"] = early_probability + result["maturity_survival_probability"] - 1.0
    return result


def summarise_method_replications(replications: pd.DataFrame) -> pd.DataFrame:
    """Summarise price, fair coupon, runtime and paired-bridge comparisons."""

    rows: list[dict[str, Any]] = []
    for (contract_id, method), group in replications.groupby(["contract_id", "method"], sort=False):
        row: dict[str, Any] = {
            "contract_id": contract_id,
            "method": method,
            "replications": len(group),
            "n_paths_per_replication": int(group["n_paths"].iloc[0]),
        }
        for metric in (
            "total_value", "fair_coupon", "coupon_value", "early_redemption_principal",
            "surviving_notional", "terminal_ki_loss", "continuous_ki_survival_probability",
            "runtime_seconds",
        ):
            if metric not in group:
                continue
            values = group[metric].astype(float)
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_sd"] = values.std(ddof=1) if len(values) > 1 else 0.0
            row[f"{metric}_se"] = row[f"{metric}_sd"] / math.sqrt(len(values))
        if "conditional_minus_paired_bernoulli" in group:
            paired = group["conditional_minus_paired_bernoulli"].dropna().astype(float)
            if len(paired):
                row["conditional_minus_bernoulli_mean"] = paired.mean()
                row["conditional_minus_bernoulli_se"] = (
                    paired.std(ddof=1) / math.sqrt(len(paired)) if len(paired) > 1 else 0.0
                )
        row["component_identity_max_abs_error"] = group["component_identity_error"].abs().max()
        row["probability_mass_max_abs_error"] = group["probability_mass_error"].abs().max()
        row["fair_coupon_max_abs_residual"] = group["fair_coupon_residual"].abs().max()
        rows.append(row)
    return pd.DataFrame(rows)

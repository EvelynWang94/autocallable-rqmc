"""Day 7 finite-difference and discrete-autocall smoothing helpers.

The full-product Greeks are obtained by CRN central differences around the M0
and M3 pricing engines.  The AC-Smooth function is intentionally narrower: it
Rao-Blackwellises the *next-observation autocall redemption component* after a
stable Householder rotation.  It is a discrete-trigger diagnostic and is not a
replacement for continuous-barrier Brownian-bridge conditioning.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
from scipy.stats import norm, qmc

from autocallable_direct import DirectContract, make_psd_correlation


def roll_contract_to_autocall(
    contract: DirectContract,
    source_observation_index: int,
    first_time_years: float,
) -> DirectContract:
    """Create a remaining-life diagnostic contract starting before an autocall date."""

    if not (0 <= source_observation_index < len(contract.observation_times)):
        raise IndexError("source_observation_index is outside the contract schedule")
    if not contract.autocall_flags[source_observation_index]:
        raise ValueError("the selected source observation must be an autocall date")
    if first_time_years <= 0:
        raise ValueError("first_time_years must be positive")
    original = np.asarray(contract.observation_times[source_observation_index:], dtype=float)
    shifted = original - original[0] + float(first_time_years)
    selection = slice(source_observation_index, None)
    return contract.with_overrides(
        contract_id=f"{contract.contract_id}-NEAR-AC",
        label=f"{contract.label} — rolled near-autocall diagnostic",
        maturity_years=float(shifted[-1]),
        observation_times=tuple(float(value) for value in shifted),
        autocall_flags=contract.autocall_flags[selection],
        autocall_trigger_ratios=contract.autocall_trigger_ratios[selection],
        coupon_flags=contract.coupon_flags[selection],
        coupon_trigger_ratios=contract.coupon_trigger_ratios[selection],
        claim_boundary=(
            contract.claim_boundary
            + "; remaining-life diagnostic conditional on no prior KI and no accrued coupon memory"
        ),
    )


def central_first_derivative(plus: float, minus: float, bump: float) -> float:
    if bump <= 0:
        raise ValueError("bump must be positive")
    return (float(plus) - float(minus)) / (2.0 * bump)


def central_second_derivative(base: float, plus: float, minus: float, bump: float) -> float:
    if bump <= 0:
        raise ValueError("bump must be positive")
    return (float(plus) - 2.0 * float(base) + float(minus)) / (bump * bump)


def stable_autocall_rotation(correlation: np.ndarray) -> dict[str, np.ndarray | float]:
    """Householder rotation with a positive common-shock final coordinate.

    If ``L`` is the Cholesky factor, the last rotation column is proportional
    to ``L^{-1} 1``.  Therefore every asset has a positive loading on the last
    independent normal and a worst-of upper trigger becomes one scalar lower
    threshold for that coordinate.
    """

    corr = make_psd_correlation(correlation)
    cholesky = np.linalg.cholesky(corr)
    direction = np.linalg.solve(cholesky, np.ones(3))
    direction /= np.linalg.norm(direction)
    e_last = np.array([0.0, 0.0, 1.0])
    difference = e_last - direction
    if np.linalg.norm(difference) <= 1e-14:
        rotation = np.eye(3)
    else:
        householder = difference / np.linalg.norm(difference)
        rotation = np.eye(3) - 2.0 * np.outer(householder, householder)
    mapping = cholesky @ rotation
    if np.any(mapping[:, -1] <= 0):
        raise FloatingPointError("rotation did not produce positive final-coordinate loadings")
    orthogonality_error = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
    correlation_error = float(np.max(np.abs(mapping @ mapping.T - corr)))
    return {
        "rotation": rotation,
        "mapping": mapping,
        "orthogonality_error": orthogonality_error,
        "correlation_error": correlation_error,
        "minimum_final_loading": float(mapping[:, -1].min()),
    }


def autocall_redemption_replication(
    initial_spot_ratios: np.ndarray,
    trigger_ratios: np.ndarray | float,
    time_to_observation: float,
    principal: float,
    risk_free_rate: float,
    dividend_yields: np.ndarray,
    volatilities: np.ndarray,
    correlation: np.ndarray,
    n_paths: int,
    method: str,
    seed: int,
    smooth: bool,
    probability_clip: float = 1e-12,
) -> dict[str, Any]:
    """Price the next-observation autocall principal, raw or AC-smoothed."""

    initial = np.asarray(initial_spot_ratios, dtype=float)
    trigger = np.broadcast_to(np.asarray(trigger_ratios, dtype=float), (3,))
    q_yield = np.asarray(dividend_yields, dtype=float)
    sigma = np.asarray(volatilities, dtype=float)
    if initial.shape != (3,) or q_yield.shape != (3,) or sigma.shape != (3,):
        raise ValueError("initial spots, dividends and volatilities must have length three")
    if np.any(initial <= 0) or np.any(trigger <= 0) or np.any(sigma <= 0):
        raise ValueError("spots, triggers and volatilities must be positive")
    if time_to_observation <= 0 or principal <= 0 or n_paths <= 1:
        raise ValueError("time, principal and n_paths must be positive")
    started = time.perf_counter()
    if method == "mc":
        normals = np.random.default_rng(int(seed)).standard_normal((n_paths, 3))
    elif method == "rqmc":
        sampler = qmc.Sobol(d=3, scramble=True, seed=int(seed))
        if n_paths & (n_paths - 1) == 0:
            uniforms = sampler.random_base2(int(math.log2(n_paths)))
        else:
            uniforms = sampler.random(n_paths)
        normals = norm.ppf(np.clip(uniforms, probability_clip, 1.0 - probability_clip))
    else:
        raise ValueError("method must be 'mc' or 'rqmc'")

    rotation = stable_autocall_rotation(correlation)
    mapping = np.asarray(rotation["mapping"])
    drift = risk_free_rate - q_yield - 0.5 * sigma**2
    deterministic = np.log(initial) + drift * time_to_observation
    scale = sigma * math.sqrt(time_to_observation)
    if smooth:
        partial_shock = normals[:, :2] @ mapping[:, :2].T
        partial_log_state = deterministic[None, :] + scale[None, :] * partial_shock
        final_loading = scale * mapping[:, -1]
        thresholds = (np.log(trigger)[None, :] - partial_log_state) / final_loading[None, :]
        scalar_threshold = np.max(thresholds, axis=1)
        call_weight = norm.sf(scalar_threshold)
        clipped_weight = np.clip(call_weight, probability_clip, 1.0 - probability_clip)
        clipping_rate = float(np.mean(clipped_weight != call_weight))
        call_weight = clipped_weight
    else:
        correlated = normals @ mapping.T
        log_state = deterministic[None, :] + scale[None, :] * correlated
        call_weight = np.all(log_state >= np.log(trigger)[None, :], axis=1).astype(float)
        clipping_rate = 0.0
    discounted_paths = principal * math.exp(-risk_free_rate * time_to_observation) * call_weight
    return {
        "value": float(discounted_paths.mean()),
        "path_sd": float(discounted_paths.std(ddof=1)),
        "call_probability": float(call_weight.mean()),
        "runtime_seconds": time.perf_counter() - started,
        "method": method,
        "smooth": bool(smooth),
        "n_paths": int(n_paths),
        "seed": int(seed),
        "probability_clipping_rate": clipping_rate,
        "orthogonality_error": rotation["orthogonality_error"],
        "correlation_error": rotation["correlation_error"],
        "minimum_final_loading": rotation["minimum_final_loading"],
    }

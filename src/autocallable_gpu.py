"""Day 8 CUDA acceleration and CPU/GPU audit helpers.

The direct M0/M1 implementation keeps the Day 5/6 contract, event clocks,
payoff decomposition and result schema, but evaluates path evolution and
payoffs with NumPy or CuPy from the *same* host-side normal innovations.  This
shared-input interface is the numerical-correctness gate; it is deliberately
separate from random-number-generation benchmarks.

The conditioned M2/M3 implementation is an explicit hybrid.  Endpoint GBM,
marginal bridge exits and nested trivariate bridge counts run on the GPU.  The
non-integer-order Bessel series used by the exact bivariate Lee probability is
kept in SciPy on the CPU because CuPy 14 does not expose ``ive``.  Timings label
that boundary instead of claiming an end-to-end GPU implementation.

All pricing arithmetic is float64.  CuPy is imported lazily so the project can
still import and run CPU tests on machines without CUDA.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import platform
import time
from typing import Any, Iterable, MutableMapping

import numpy as np
import pandas as pd
from scipy.stats import norm, qmc

from autocallable_bb import (
    BridgeBank,
    PAIR_SPECS,
    bivariate_coexit,
    generate_bridge_bank,
    segment_probabilities,
    trivariate_probability_bounds,
)
from autocallable_direct import (
    DirectContract,
    make_psd_correlation,
    monitoring_grid,
    stable_seed,
    validate_contract,
)


FLOAT_DTYPE = np.float64
PROBABILITY_CLIP = 1.0e-12


class GpuUnavailableError(RuntimeError):
    """Raised when a requested CUDA calculation has no working CuPy device."""


@dataclass(frozen=True)
class NormalInput:
    """Host-side independent normal innovations plus generation metadata."""

    values: np.ndarray
    method: str
    seed: int
    generation_seconds: float


def configure_cupy_cache(cache_directory: str | Path) -> Path:
    """Set a project-local CuPy JIT cache before the first CuPy import."""

    path = Path(cache_directory).resolve()
    path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUPY_CACHE_DIR", str(path))
    return path


def _require_cupy():
    try:
        import cupy as cp
    except Exception as exc:  # pragma: no cover - depends on local CUDA install
        raise GpuUnavailableError(f"CuPy could not be imported: {exc}") from exc
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise GpuUnavailableError("CuPy reports zero CUDA devices")
    except GpuUnavailableError:
        raise
    except Exception as exc:  # pragma: no cover - depends on driver state
        raise GpuUnavailableError(f"CUDA runtime is unavailable: {exc}") from exc
    return cp


def gpu_environment() -> dict[str, Any]:
    """Return an audit-friendly CUDA/CuPy environment record."""

    record: dict[str, Any] = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "cupy_available": False,
        "cuda_device_count": 0,
        "precision": "float64",
    }
    try:
        cp = _require_cupy()
        device_count = int(cp.cuda.runtime.getDeviceCount())
        properties = cp.cuda.runtime.getDeviceProperties(0)
        name = properties["name"]
        if isinstance(name, bytes):
            name = name.decode(errors="replace")
        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        record.update(
            {
                "cupy_available": True,
                "cupy_version": cp.__version__,
                "cuda_device_count": device_count,
                "device_name": str(name),
                "compute_capability": f"{properties['major']}.{properties['minor']}",
                "device_memory_total_bytes": int(total_bytes),
                "device_memory_free_bytes_at_probe": int(free_bytes),
                "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
                "cuda_driver_version": int(cp.cuda.runtime.driverGetVersion()),
            }
        )
    except Exception as exc:
        record["unavailable_reason"] = str(exc)
    return record


def warmup_gpu() -> dict[str, Any]:
    """Compile a small float64 path-like kernel and synchronise the device."""

    cp = _require_cupy()
    started = time.perf_counter()
    x = cp.linspace(-1.0, 1.0, 4096, dtype=cp.float64)
    y = cp.exp(x) + cp.minimum(x, 0.0) ** 2
    checksum = float(cp.sum(y).get())
    cp.cuda.Stream.null.synchronize()
    return {
        "compile_warmup_seconds": time.perf_counter() - started,
        "warmup_checksum": checksum,
    }


def generate_normal_input(
    n_paths: int,
    n_steps: int,
    method: str,
    seed: int,
    probability_clip: float = PROBABILITY_CLIP,
) -> NormalInput:
    """Generate independent host normals for shared CPU/GPU M0 or M1 runs."""

    if n_paths <= 1 or n_steps <= 0:
        raise ValueError("n_paths must exceed one and n_steps must be positive")
    started = time.perf_counter()
    dimension = int(n_steps) * 3
    if method == "mc":
        values = np.random.default_rng(int(seed)).standard_normal((n_paths, n_steps, 3))
    elif method == "rqmc":
        sampler = qmc.Sobol(d=dimension, scramble=True, seed=int(seed))
        if n_paths & (n_paths - 1) == 0:
            uniforms = sampler.random_base2(int(math.log2(n_paths)))
        else:
            uniforms = sampler.random(n_paths)
        values = norm.ppf(np.clip(uniforms, probability_clip, 1.0 - probability_clip)).reshape(
            n_paths, n_steps, 3
        )
    else:
        raise ValueError("method must be 'mc' or 'rqmc'")
    values = np.ascontiguousarray(values, dtype=FLOAT_DTYPE)
    return NormalInput(values, method, int(seed), time.perf_counter() - started)


def generate_endpoint_normal_input(
    n_paths: int,
    n_observations: int,
    method: str,
    seed: int,
) -> NormalInput:
    """Generate the lower-dimensional shared input used by M2/M3."""

    return generate_normal_input(n_paths, n_observations, method, seed)


def _validate_market_inputs(
    dividend_yields: np.ndarray,
    volatilities: np.ndarray,
    correlation: np.ndarray,
    initial_spot_ratios: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q_yield = np.asarray(dividend_yields, dtype=FLOAT_DTYPE)
    sigma = np.asarray(volatilities, dtype=FLOAT_DTYPE)
    corr = make_psd_correlation(correlation)
    initial = (
        np.ones(3, dtype=FLOAT_DTYPE)
        if initial_spot_ratios is None
        else np.asarray(initial_spot_ratios, dtype=FLOAT_DTYPE)
    )
    if q_yield.shape != (3,) or sigma.shape != (3,) or initial.shape != (3,):
        raise ValueError("dividends, volatilities and initial spots must have length three")
    if np.any(sigma <= 0.0) or np.any(initial <= 0.0):
        raise ValueError("volatilities and initial spots must be positive")
    return q_yield, sigma, corr, initial


def _scalar_array(backend: str, values: Any) -> np.ndarray:
    if backend == "cpu":
        return np.asarray(values, dtype=FLOAT_DTYPE)
    cp = _require_cupy()
    return cp.asnumpy(values).astype(FLOAT_DTYPE, copy=False)


def direct_price_from_normals(
    contract: DirectContract,
    risk_free_rate: float,
    dividend_yields: np.ndarray,
    volatilities: np.ndarray,
    correlation: np.ndarray,
    annual_coupon: float,
    normals: np.ndarray,
    steps_per_year: int,
    backend: str,
    batch_size: int = 8192,
    initial_spot_ratios: np.ndarray | None = None,
) -> dict[str, Any]:
    """Price direct M0/M1 paths with NumPy or CuPy from identical normals."""

    validate_contract(contract)
    if backend not in {"cpu", "gpu"}:
        raise ValueError("backend must be 'cpu' or 'gpu'")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    q_yield, sigma, corr, initial = _validate_market_inputs(
        dividend_yields, volatilities, correlation, initial_spot_ratios
    )
    grid, observation_map = monitoring_grid(contract, steps_per_year)
    grid_steps = len(grid) - 1
    host_normals = np.asarray(normals, dtype=FLOAT_DTYPE)
    if host_normals.ndim != 3 or host_normals.shape[1:] != (grid_steps, 3):
        raise ValueError(
            f"normals must have shape (n_paths, {grid_steps}, 3); got {host_normals.shape}"
        )
    n_paths = len(host_normals)
    if n_paths <= 1:
        raise ValueError("normals must contain more than one path")

    cp = None
    if backend == "gpu":
        cp = _require_cupy()
        xp = cp
    else:
        xp = np

    started_total = time.perf_counter()
    transfer_seconds = 0.0
    kernel_seconds = 0.0
    reduction_seconds = 0.0
    peak_working_bytes = 0
    cholesky_host = np.linalg.cholesky(corr)
    discounts_host = np.exp(-risk_free_rate * np.asarray(contract.observation_times))
    maturity_discount = float(discounts_host[-1])
    drift_host = risk_free_rate - q_yield - 0.5 * sigma**2

    setup_started = time.perf_counter()
    if backend == "gpu":
        small_transfer_started = time.perf_counter()
        cholesky = cp.asarray(cholesky_host)
        sigma_x = cp.asarray(sigma)
        drift = cp.asarray(drift_host)
        initial_log = cp.asarray(np.log(initial))
        cp.cuda.Stream.null.synchronize()
        transfer_seconds += time.perf_counter() - small_transfer_started
    else:
        cholesky = cholesky_host
        sigma_x = sigma
        drift = drift_host
        initial_log = np.log(initial)
    setup_seconds = time.perf_counter() - setup_started

    numeric_totals = np.zeros(7, dtype=FLOAT_DTYPE)
    integer_totals = np.zeros(2, dtype=np.int64)
    redemption_totals = np.zeros(len(contract.observation_times), dtype=np.int64)

    for batch_start in range(0, n_paths, batch_size):
        batch_stop = min(n_paths, batch_start + batch_size)
        batch = batch_stop - batch_start
        transfer_started = time.perf_counter()
        z = xp.asarray(host_normals[batch_start:batch_stop])
        if backend == "gpu":
            cp.cuda.Stream.null.synchronize()
        transfer_seconds += time.perf_counter() - transfer_started

        kernel_started = time.perf_counter()
        log_state = xp.broadcast_to(initial_log, (batch, 3)).copy()
        active = xp.ones(batch, dtype=bool)
        knocked_in = xp.full(batch, np.min(initial) <= contract.ki_barrier_ratio, dtype=bool)
        accrued = xp.zeros(batch, dtype=xp.int16)
        coupon_annuity = xp.zeros(batch, dtype=xp.float64)
        fixed_coupon = xp.zeros(batch, dtype=xp.float64)
        early_principal = xp.zeros(batch, dtype=xp.float64)
        surviving = xp.zeros(batch, dtype=xp.float64)
        no_ki_redemption = xp.zeros(batch, dtype=xp.float64)
        terminal_loss = xp.zeros(batch, dtype=xp.float64)
        call_index = xp.full(batch, -1, dtype=xp.int16)
        redemptions = xp.zeros(len(contract.observation_times), dtype=xp.int64)

        for step in range(grid_steps):
            dt = float(grid[step + 1] - grid[step])
            correlated = z[:, step, :] @ cholesky.T
            log_state += drift * dt + sigma_x * math.sqrt(dt) * correlated
            performance = xp.exp(log_state)
            knocked_in |= active & (xp.min(performance, axis=1) <= contract.ki_barrier_ratio)
            observation_index = observation_map.get(step + 1)
            if observation_index is None:
                continue
            worst = xp.min(performance, axis=1)
            discount = float(discounts_host[observation_index])
            observation_time = contract.observation_times[observation_index]
            if (
                contract.coupon_mode in {"periodic", "memory_periodic"}
                and contract.coupon_flags[observation_index]
            ):
                accrued += active.astype(xp.int16)
                paid = active & (
                    worst >= float(contract.coupon_trigger_ratios[observation_index])
                )
                paid_periods = accrued if contract.coupon_memory else xp.ones_like(accrued)
                coupon_annuity += (
                    contract.principal
                    * paid_periods
                    / contract.coupon_periods_per_year
                    * discount
                    * paid
                )
                accrued = xp.where(paid, 0, accrued)
            if contract.autocall_flags[observation_index]:
                called = active & (
                    worst >= float(contract.autocall_trigger_ratios[observation_index])
                )
                redemptions[observation_index] += xp.count_nonzero(called)
                call_index = xp.where(called, observation_index, call_index)
                if contract.coupon_mode == "simple_to_redemption":
                    coupon_annuity += (
                        contract.principal * observation_time * discount * called
                    )
                if observation_time < contract.maturity_years - 1.0e-12:
                    early_principal += contract.principal * discount * called
                else:
                    surviving += contract.principal * discount * called
                    no_ki_redemption += contract.principal * discount * (called & ~knocked_in)
                active &= ~called

        remaining = active
        terminal_worst = xp.min(xp.exp(log_state), axis=1)
        surviving += contract.principal * maturity_discount * remaining
        loss = remaining & knocked_in & (terminal_worst < 1.0)
        terminal_loss += (
            contract.principal * (terminal_worst - 1.0) * maturity_discount * loss
        )
        no_ki = remaining & ~knocked_in
        no_ki_redemption += contract.principal * maturity_discount * no_ki
        if contract.coupon_mode == "simple_to_redemption":
            if contract.fixed_maturity_no_ki_coupon_rate is None:
                coupon_annuity += (
                    contract.principal * contract.maturity_years * maturity_discount * no_ki
                )
            else:
                fixed_coupon += (
                    contract.principal
                    * contract.fixed_maturity_no_ki_coupon_rate
                    * maturity_discount
                    * no_ki
                )
        if backend == "gpu":
            cp.cuda.Stream.null.synchronize()
        kernel_seconds += time.perf_counter() - kernel_started

        reduction_started = time.perf_counter()
        numeric_device = xp.stack(
            [
                xp.sum(coupon_annuity),
                xp.sum(fixed_coupon),
                xp.sum(early_principal),
                xp.sum(surviving),
                xp.sum(no_ki_redemption),
                xp.sum(terminal_loss),
                xp.sum((~knocked_in).astype(xp.float64)),
            ]
        )
        integer_device = xp.stack(
            [
                xp.sum(((call_index < 0) | (call_index == len(contract.observation_times) - 1)).astype(xp.int64)),
                xp.asarray(batch, dtype=xp.int64),
            ]
        )
        numeric_totals += _scalar_array(backend, numeric_device)
        integer_totals += _scalar_array(backend, integer_device).astype(np.int64)
        redemption_totals += _scalar_array(backend, redemptions).astype(np.int64)
        if backend == "gpu":
            cp.cuda.Stream.null.synchronize()
            peak_working_bytes = max(
                peak_working_bytes, int(cp.get_default_memory_pool().used_bytes())
            )
        reduction_seconds += time.perf_counter() - reduction_started

    scale = 1.0 / n_paths
    (
        coupon_annuity_mean,
        fixed_coupon_mean,
        early_mean,
        surviving_mean,
        no_ki_mean,
        terminal_loss_mean,
        ki_survivors,
    ) = numeric_totals * scale
    coupon_value = fixed_coupon_mean + annual_coupon * coupon_annuity_mean
    base_value = fixed_coupon_mean + early_mean + surviving_mean + terminal_loss_mean
    total_value = base_value + annual_coupon * coupon_annuity_mean
    additive_sum = coupon_value + early_mean + surviving_mean + terminal_loss_mean
    fair_coupon = (contract.principal - base_value) / coupon_annuity_mean
    maturity_probability = integer_totals[0] * scale
    total_seconds = time.perf_counter() - started_total

    result: dict[str, Any] = {
        "contract_id": contract.contract_id,
        "backend": backend,
        "precision": "float64",
        "annual_coupon": float(annual_coupon),
        "coupon_annuity": float(coupon_annuity_mean),
        "fixed_coupon_value": float(fixed_coupon_mean),
        "coupon_value": float(coupon_value),
        "early_redemption_principal": float(early_mean),
        "surviving_notional": float(surviving_mean),
        "no_ki_maturity_redemption": float(no_ki_mean),
        "terminal_ki_loss": float(terminal_loss_mean),
        "base_value": float(base_value),
        "total_value": float(total_value),
        "additive_sum": float(additive_sum),
        "component_identity_error": float(total_value - additive_sum),
        "continuous_ki_survival_probability": float(ki_survivors),
        "maturity_survival_probability": float(maturity_probability),
        "fair_coupon": float(fair_coupon),
        "fair_coupon_residual": float(
            base_value + fair_coupon * coupon_annuity_mean - contract.principal
        ),
        "n_paths": int(n_paths),
        "steps_per_year": int(steps_per_year),
        "grid_steps": int(grid_steps),
        "batch_size": int(batch_size),
        "setup_seconds": float(setup_seconds),
        "transfer_seconds": float(transfer_seconds),
        "pricing_kernel_seconds": float(kernel_seconds),
        "reduction_seconds": float(reduction_seconds),
        "total_wall_seconds": float(total_seconds),
        "paths_per_second": float(n_paths / total_seconds),
        "peak_memory_pool_bytes": int(peak_working_bytes),
    }
    for asset_index, ratio in enumerate(initial):
        result[f"initial_spot_ratio_{asset_index + 1}"] = float(ratio)
    for observation_index, count in enumerate(redemption_totals):
        result[f"redemption_probability_{observation_index + 1}"] = float(count * scale)
    early_probability = sum(
        result[f"redemption_probability_{index + 1}"]
        for index, observation_time in enumerate(contract.observation_times)
        if observation_time < contract.maturity_years - 1.0e-12
    )
    result["probability_mass_error"] = float(
        early_probability + result["maturity_survival_probability"] - 1.0
    )
    return result


def direct_replication_shared_input(
    contract: DirectContract,
    risk_free_rate: float,
    dividend_yields: np.ndarray,
    volatilities: np.ndarray,
    correlation: np.ndarray,
    annual_coupon: float,
    n_paths: int,
    steps_per_year: int,
    method: str,
    seed: int,
    backend: str,
    batch_size: int = 8192,
    initial_spot_ratios: np.ndarray | None = None,
    normal_input: NormalInput | None = None,
) -> tuple[dict[str, Any], NormalInput]:
    """Generate or reuse input, then run one audited M0/M1 replication."""

    grid, _ = monitoring_grid(contract, steps_per_year)
    supplied = normal_input or generate_normal_input(n_paths, len(grid) - 1, method, seed)
    if supplied.method != method or supplied.values.shape[0] != n_paths:
        raise ValueError("normal_input does not match requested method/path count")
    result = direct_price_from_normals(
        contract,
        risk_free_rate,
        dividend_yields,
        volatilities,
        correlation,
        annual_coupon,
        supplied.values,
        steps_per_year,
        backend,
        batch_size=batch_size,
        initial_spot_ratios=initial_spot_ratios,
    )
    result.update(
        {
            "method": "M0" if method == "mc" else "M1",
            "random_method": method,
            "seed": int(seed),
            "random_input_seconds": float(supplied.generation_seconds),
            "total_with_input_seconds": float(
                result["total_wall_seconds"] + supplied.generation_seconds
            ),
        }
    )
    return result, supplied


def price_scenarios_from_normals(
    contract: DirectContract,
    risk_free_rate: float,
    dividend_yields: np.ndarray,
    correlation: np.ndarray,
    annual_coupon: float,
    normals: np.ndarray,
    steps_per_year: int,
    scenario_labels: Iterable[str],
    initial_spot_scenarios: np.ndarray,
    volatility_scenarios: np.ndarray,
    backend: str,
    batch_size: int = 4096,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Price many CRN bump scenarios together on a CPU or GPU scenario axis."""

    validate_contract(contract)
    if backend not in {"cpu", "gpu"}:
        raise ValueError("backend must be 'cpu' or 'gpu'")
    labels = list(scenario_labels)
    initial_host = np.asarray(initial_spot_scenarios, dtype=FLOAT_DTYPE)
    sigma_host = np.asarray(volatility_scenarios, dtype=FLOAT_DTYPE)
    if initial_host.shape != sigma_host.shape or initial_host.ndim != 2 or initial_host.shape[1] != 3:
        raise ValueError("spot and volatility scenarios must be aligned k by 3 arrays")
    if len(labels) != len(initial_host) or np.any(initial_host <= 0) or np.any(sigma_host <= 0):
        raise ValueError("scenario labels/arrays must align and contain positive values")
    q_yield = np.asarray(dividend_yields, dtype=FLOAT_DTYPE)
    corr = make_psd_correlation(correlation)
    if q_yield.shape != (3,):
        raise ValueError("dividend_yields must have length three")
    grid, observation_map = monitoring_grid(contract, steps_per_year)
    grid_steps = len(grid) - 1
    host_normals = np.asarray(normals, dtype=FLOAT_DTYPE)
    if host_normals.ndim != 3 or host_normals.shape[1:] != (grid_steps, 3):
        raise ValueError("normal input shape does not match the monitoring grid")
    n_paths = len(host_normals)
    scenario_count = len(labels)
    cp = _require_cupy() if backend == "gpu" else None
    xp = cp if backend == "gpu" else np
    discounts = np.exp(-risk_free_rate * np.asarray(contract.observation_times))
    maturity_discount = float(discounts[-1])

    started_total = time.perf_counter()
    transfer_seconds = 0.0
    kernel_seconds = 0.0
    reduction_seconds = 0.0
    small_started = time.perf_counter()
    cholesky = xp.asarray(np.linalg.cholesky(corr))
    initial_log = xp.asarray(np.log(initial_host))
    sigma = xp.asarray(sigma_host)
    drift = risk_free_rate - xp.asarray(q_yield)[None, :] - 0.5 * sigma**2
    if backend == "gpu":
        cp.cuda.Stream.null.synchronize()
    transfer_seconds += time.perf_counter() - small_started

    numeric_totals = np.zeros((scenario_count, 7), dtype=FLOAT_DTYPE)
    maturity_totals = np.zeros(scenario_count, dtype=np.int64)
    redemption_totals = np.zeros((scenario_count, len(contract.observation_times)), dtype=np.int64)

    for batch_start in range(0, n_paths, batch_size):
        batch_stop = min(n_paths, batch_start + batch_size)
        batch = batch_stop - batch_start
        transfer_started = time.perf_counter()
        z = xp.asarray(host_normals[batch_start:batch_stop])
        if backend == "gpu":
            cp.cuda.Stream.null.synchronize()
        transfer_seconds += time.perf_counter() - transfer_started

        kernel_started = time.perf_counter()
        log_state = xp.broadcast_to(initial_log[:, None, :], (scenario_count, batch, 3)).copy()
        active = xp.ones((scenario_count, batch), dtype=bool)
        initial_ki = np.min(initial_host, axis=1) <= contract.ki_barrier_ratio
        knocked_in = xp.broadcast_to(xp.asarray(initial_ki)[:, None], active.shape).copy()
        accrued = xp.zeros((scenario_count, batch), dtype=xp.int16)
        coupon_annuity = xp.zeros_like(active, dtype=xp.float64)
        fixed_coupon = xp.zeros_like(active, dtype=xp.float64)
        early_principal = xp.zeros_like(active, dtype=xp.float64)
        surviving = xp.zeros_like(active, dtype=xp.float64)
        no_ki_redemption = xp.zeros_like(active, dtype=xp.float64)
        terminal_loss = xp.zeros_like(active, dtype=xp.float64)
        call_index = xp.full(active.shape, -1, dtype=xp.int16)
        redemptions = xp.zeros(
            (scenario_count, len(contract.observation_times)), dtype=xp.int64
        )
        for step in range(grid_steps):
            dt = float(grid[step + 1] - grid[step])
            correlated = z[:, step, :] @ cholesky.T
            log_state += (
                drift[:, None, :] * dt
                + sigma[:, None, :] * math.sqrt(dt) * correlated[None, :, :]
            )
            performance = xp.exp(log_state)
            knocked_in |= active & (xp.min(performance, axis=2) <= contract.ki_barrier_ratio)
            observation_index = observation_map.get(step + 1)
            if observation_index is None:
                continue
            worst = xp.min(performance, axis=2)
            discount = float(discounts[observation_index])
            observation_time = contract.observation_times[observation_index]
            if (
                contract.coupon_mode in {"periodic", "memory_periodic"}
                and contract.coupon_flags[observation_index]
            ):
                accrued += active.astype(xp.int16)
                paid = active & (
                    worst >= float(contract.coupon_trigger_ratios[observation_index])
                )
                paid_periods = accrued if contract.coupon_memory else xp.ones_like(accrued)
                coupon_annuity += (
                    contract.principal
                    * paid_periods
                    / contract.coupon_periods_per_year
                    * discount
                    * paid
                )
                accrued = xp.where(paid, 0, accrued)
            if contract.autocall_flags[observation_index]:
                called = active & (
                    worst >= float(contract.autocall_trigger_ratios[observation_index])
                )
                redemptions[:, observation_index] += xp.count_nonzero(called, axis=1)
                call_index = xp.where(called, observation_index, call_index)
                if contract.coupon_mode == "simple_to_redemption":
                    coupon_annuity += contract.principal * observation_time * discount * called
                if observation_time < contract.maturity_years - 1.0e-12:
                    early_principal += contract.principal * discount * called
                else:
                    surviving += contract.principal * discount * called
                    no_ki_redemption += contract.principal * discount * (called & ~knocked_in)
                active &= ~called
        remaining = active
        terminal_worst = xp.min(xp.exp(log_state), axis=2)
        surviving += contract.principal * maturity_discount * remaining
        loss = remaining & knocked_in & (terminal_worst < 1.0)
        terminal_loss += contract.principal * (terminal_worst - 1.0) * maturity_discount * loss
        no_ki = remaining & ~knocked_in
        no_ki_redemption += contract.principal * maturity_discount * no_ki
        if contract.coupon_mode == "simple_to_redemption":
            if contract.fixed_maturity_no_ki_coupon_rate is None:
                coupon_annuity += contract.principal * contract.maturity_years * maturity_discount * no_ki
            else:
                fixed_coupon += (
                    contract.principal
                    * contract.fixed_maturity_no_ki_coupon_rate
                    * maturity_discount
                    * no_ki
                )
        if backend == "gpu":
            cp.cuda.Stream.null.synchronize()
        kernel_seconds += time.perf_counter() - kernel_started

        reduction_started = time.perf_counter()
        numeric_device = xp.stack(
            [
                xp.sum(coupon_annuity, axis=1),
                xp.sum(fixed_coupon, axis=1),
                xp.sum(early_principal, axis=1),
                xp.sum(surviving, axis=1),
                xp.sum(no_ki_redemption, axis=1),
                xp.sum(terminal_loss, axis=1),
                xp.sum((~knocked_in).astype(xp.float64), axis=1),
            ],
            axis=1,
        )
        maturity_device = xp.sum(
            ((call_index < 0) | (call_index == len(contract.observation_times) - 1)).astype(xp.int64),
            axis=1,
        )
        numeric_totals += _scalar_array(backend, numeric_device)
        maturity_totals += _scalar_array(backend, maturity_device).astype(np.int64)
        redemption_totals += _scalar_array(backend, redemptions).astype(np.int64)
        reduction_seconds += time.perf_counter() - reduction_started

    rows: list[dict[str, Any]] = []
    scale = 1.0 / n_paths
    for scenario_index, label in enumerate(labels):
        values = numeric_totals[scenario_index] * scale
        coupon_annuity_mean, fixed_mean, early_mean, surviving_mean, no_ki_mean, loss_mean, ki_survival = values
        base = fixed_mean + early_mean + surviving_mean + loss_mean
        coupon_value = fixed_mean + annual_coupon * coupon_annuity_mean
        total = base + annual_coupon * coupon_annuity_mean
        row: dict[str, Any] = {
            "scenario": label,
            "backend": backend,
            "coupon_annuity": coupon_annuity_mean,
            "fixed_coupon_value": fixed_mean,
            "coupon_value": coupon_value,
            "early_redemption_principal": early_mean,
            "surviving_notional": surviving_mean,
            "no_ki_maturity_redemption": no_ki_mean,
            "terminal_ki_loss": loss_mean,
            "base_value": base,
            "total_value": total,
            "additive_sum": coupon_value + early_mean + surviving_mean + loss_mean,
            "component_identity_error": total - (coupon_value + early_mean + surviving_mean + loss_mean),
            "continuous_ki_survival_probability": ki_survival,
            "maturity_survival_probability": maturity_totals[scenario_index] * scale,
            "fair_coupon": (contract.principal - base) / coupon_annuity_mean,
            "initial_spot_ratio_1": initial_host[scenario_index, 0],
            "initial_spot_ratio_2": initial_host[scenario_index, 1],
            "initial_spot_ratio_3": initial_host[scenario_index, 2],
            "volatility_1": sigma_host[scenario_index, 0],
            "volatility_2": sigma_host[scenario_index, 1],
            "volatility_3": sigma_host[scenario_index, 2],
        }
        for observation_index, count in enumerate(redemption_totals[scenario_index]):
            row[f"redemption_probability_{observation_index + 1}"] = count * scale
        early_probability = sum(
            row[f"redemption_probability_{index + 1}"]
            for index, observation_time in enumerate(contract.observation_times)
            if observation_time < contract.maturity_years - 1.0e-12
        )
        row["probability_mass_error"] = (
            early_probability + row["maturity_survival_probability"] - 1.0
        )
        rows.append(row)
    total_seconds = time.perf_counter() - started_total
    timing = {
        "backend": backend,
        "scenario_count": scenario_count,
        "n_paths": n_paths,
        "transfer_seconds": transfer_seconds,
        "pricing_kernel_seconds": kernel_seconds,
        "reduction_seconds": reduction_seconds,
        "total_wall_seconds": total_seconds,
        "scenario_paths_per_second": scenario_count * n_paths / total_seconds,
    }
    return pd.DataFrame(rows), timing


def central_bump_scenarios(
    initial_spot_ratios: np.ndarray,
    volatilities: np.ndarray,
    spot_relative_bump: float,
    volatility_absolute_bump: float,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Build base plus component spot/vol central-bump scenarios."""

    initial = np.asarray(initial_spot_ratios, dtype=FLOAT_DTYPE)
    sigma = np.asarray(volatilities, dtype=FLOAT_DTYPE)
    if initial.shape != (3,) or sigma.shape != (3,):
        raise ValueError("initial_spot_ratios and volatilities must have length three")
    if spot_relative_bump <= 0 or volatility_absolute_bump <= 0:
        raise ValueError("bumps must be positive")
    labels = ["base"]
    spot_rows = [initial.copy()]
    vol_rows = [sigma.copy()]
    for asset in range(3):
        for direction, sign in (("minus", -1.0), ("plus", 1.0)):
            bumped = initial.copy()
            bumped[asset] *= 1.0 + sign * spot_relative_bump
            labels.append(f"spot_{asset + 1}_{direction}")
            spot_rows.append(bumped)
            vol_rows.append(sigma.copy())
    for asset in range(3):
        for direction, sign in (("minus", -1.0), ("plus", 1.0)):
            bumped = sigma.copy()
            bumped[asset] += sign * volatility_absolute_bump
            labels.append(f"vol_{asset + 1}_{direction}")
            spot_rows.append(initial.copy())
            vol_rows.append(bumped)
    return labels, np.vstack(spot_rows), np.vstack(vol_rows)


def greeks_from_scenario_prices(
    scenario_prices: pd.DataFrame,
    initial_spot_ratios: np.ndarray,
    spot_relative_bump: float,
    volatility_absolute_bump: float,
) -> pd.DataFrame:
    """Convert a central-bump scenario table into component Delta/Gamma/Vega."""

    indexed = scenario_prices.set_index("scenario")
    base = float(indexed.loc["base", "total_value"])
    initial = np.asarray(initial_spot_ratios, dtype=FLOAT_DTYPE)
    rows: list[dict[str, Any]] = []
    for asset in range(3):
        minus = float(indexed.loc[f"spot_{asset + 1}_minus", "total_value"])
        plus = float(indexed.loc[f"spot_{asset + 1}_plus", "total_value"])
        h = initial[asset] * spot_relative_bump
        rows.extend(
            [
                {"greek": "Delta", "asset": asset + 1, "value": (plus - minus) / (2.0 * h)},
                {"greek": "Gamma", "asset": asset + 1, "value": (plus - 2.0 * base + minus) / (h * h)},
            ]
        )
        v_minus = float(indexed.loc[f"vol_{asset + 1}_minus", "total_value"])
        v_plus = float(indexed.loc[f"vol_{asset + 1}_plus", "total_value"])
        rows.append(
            {
                "greek": "Vega",
                "asset": asset + 1,
                "value": (v_plus - v_minus) / (2.0 * volatility_absolute_bump),
            }
        )
    result = pd.DataFrame(rows)
    result["backend"] = str(scenario_prices["backend"].iloc[0])
    return result


def generate_endpoint_paths_from_normals(
    observation_times: tuple[float, ...],
    risk_free_rate: float,
    dividend_yields: np.ndarray,
    volatilities: np.ndarray,
    correlation: np.ndarray,
    normals: np.ndarray,
    backend: str,
    initial_spot_ratios: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Generate observation endpoints on CPU/GPU and return host log paths."""

    if backend not in {"cpu", "gpu"}:
        raise ValueError("backend must be 'cpu' or 'gpu'")
    q_yield, sigma, corr, initial = _validate_market_inputs(
        dividend_yields, volatilities, correlation, initial_spot_ratios
    )
    host_normals = np.asarray(normals, dtype=FLOAT_DTYPE)
    n_obs = len(observation_times)
    if host_normals.ndim != 3 or host_normals.shape[1:] != (n_obs, 3):
        raise ValueError("endpoint normals must have shape (n_paths, n_observations, 3)")
    cp = _require_cupy() if backend == "gpu" else None
    xp = cp if backend == "gpu" else np
    started = time.perf_counter()
    transfer_started = time.perf_counter()
    z = xp.asarray(host_normals)
    cholesky = xp.asarray(np.linalg.cholesky(corr))
    sigma_x = xp.asarray(sigma)
    q_x = xp.asarray(q_yield)
    initial_x = xp.asarray(np.log(initial))
    if backend == "gpu":
        cp.cuda.Stream.null.synchronize()
    transfer_h2d = time.perf_counter() - transfer_started
    kernel_started = time.perf_counter()
    correlated = z @ cholesky.T
    times = np.asarray(observation_times, dtype=FLOAT_DTYPE)
    dts = np.diff(np.concatenate([[0.0], times]))
    dts_x = xp.asarray(dts)
    drift = risk_free_rate - q_x - 0.5 * sigma_x**2
    increments = (
        drift[None, None, :] * dts_x[None, :, None]
        + sigma_x[None, None, :] * xp.sqrt(dts_x)[None, :, None] * correlated
    )
    endpoints_x = initial_x[None, None, :] + xp.cumsum(increments, axis=1)
    if backend == "gpu":
        cp.cuda.Stream.null.synchronize()
    kernel_seconds = time.perf_counter() - kernel_started
    transfer_back_started = time.perf_counter()
    endpoints = _scalar_array(backend, endpoints_x)
    transfer_d2h = time.perf_counter() - transfer_back_started
    return endpoints, {
        "endpoint_h2d_seconds": transfer_h2d,
        "endpoint_kernel_seconds": kernel_seconds,
        "endpoint_d2h_seconds": transfer_d2h,
        "endpoint_total_seconds": time.perf_counter() - started,
    }


def nested_exit_components_gpu(
    endpoints: np.ndarray,
    barriers: np.ndarray,
    bank: BridgeBank,
    chunk_size: int = 256,
) -> dict[str, Any]:
    """Evaluate nested bridge exit indicators on CUDA using a shared bank."""

    cp = _require_cupy()
    x = np.atleast_2d(np.asarray(endpoints, dtype=FLOAT_DTYPE))
    b = np.asarray(barriers, dtype=FLOAT_DTYPE)
    if b.ndim == 1:
        b = np.broadcast_to(b, x.shape)
    if x.shape != b.shape or x.shape[1] != 3:
        raise ValueError("endpoints and barriers must be aligned n by 3 arrays")
    transfer_seconds = 0.0
    kernel_seconds = 0.0
    d2h_seconds = 0.0
    transfer_started = time.perf_counter()
    residuals = cp.asarray(bank.residuals)
    fractions = cp.asarray(bank.time_fractions)
    cp.cuda.Stream.null.synchronize()
    transfer_seconds += time.perf_counter() - transfer_started
    g_nested = np.empty((len(x), 3), dtype=FLOAT_DTYPE)
    h_nested = np.empty((len(x), 3), dtype=FLOAT_DTYPE)
    q_nested = np.empty(len(x), dtype=FLOAT_DTYPE)
    p_direct = np.empty(len(x), dtype=FLOAT_DTYPE)
    for start in range(0, len(x), chunk_size):
        stop = min(len(x), start + chunk_size)
        transfer_started = time.perf_counter()
        xc = cp.asarray(x[start:stop])
        bc = cp.asarray(b[start:stop])
        cp.cuda.Stream.null.synchronize()
        transfer_seconds += time.perf_counter() - transfer_started
        kernel_started = time.perf_counter()
        linear = xc[:, None, None, :] * fractions[None, None, :, None]
        values = linear + residuals[None, :, :, :]
        exits = cp.any(values >= bc[:, None, None, :], axis=2)
        deterministic = (bc <= 0.0) | (xc >= bc)
        exits |= deterministic[:, None, :]
        g_x = cp.mean(exits, axis=1)
        h_x = cp.stack(
            [(exits[:, :, i] & exits[:, :, j]).mean(axis=1) for _, i, j in PAIR_SPECS],
            axis=1,
        )
        q_x = cp.all(exits, axis=2).mean(axis=1)
        p_x = (~cp.any(exits, axis=2)).mean(axis=1)
        cp.cuda.Stream.null.synchronize()
        kernel_seconds += time.perf_counter() - kernel_started
        d2h_started = time.perf_counter()
        g_nested[start:stop] = cp.asnumpy(g_x)
        h_nested[start:stop] = cp.asnumpy(h_x)
        q_nested[start:stop] = cp.asnumpy(q_x)
        p_direct[start:stop] = cp.asnumpy(p_x)
        d2h_seconds += time.perf_counter() - d2h_started
    return {
        "g_nested": g_nested,
        "h_nested": h_nested,
        "q3_nested_direct": q_nested,
        "p3_nested_direct": p_direct,
        "runtime_nested_transfer_seconds": transfer_seconds,
        "runtime_nested_gpu_seconds": kernel_seconds,
        "runtime_nested_d2h_seconds": d2h_seconds,
    }


def segment_probabilities_hybrid_gpu(
    endpoints: np.ndarray,
    barriers: np.ndarray,
    dt: float,
    volatilities: np.ndarray,
    correlation: np.ndarray,
    bank: BridgeBank | None,
    bessel_terms: int = 12,
) -> dict[str, Any]:
    """GPU marginal/nested bridge probabilities with CPU exact Bessel pairs."""

    cp = _require_cupy()
    x = np.atleast_2d(np.asarray(endpoints, dtype=FLOAT_DTYPE))
    b = np.asarray(barriers, dtype=FLOAT_DTYPE)
    if b.ndim == 1:
        b = np.broadcast_to(b, x.shape)
    sigma = np.asarray(volatilities, dtype=FLOAT_DTYPE)
    corr = make_psd_correlation(correlation)
    if x.shape != b.shape or x.shape[1] != 3:
        raise ValueError("endpoints and barriers must be aligned n by 3 arrays")

    marginal_started = time.perf_counter()
    x_gpu = cp.asarray(x)
    b_gpu = cp.asarray(b)
    sigma_gpu = cp.asarray(sigma)
    exponent = -2.0 * b_gpu * (b_gpu - x_gpu) / (sigma_gpu[None, :] ** 2 * dt)
    g_gpu = cp.where(x_gpu >= b_gpu, 1.0, cp.exp(exponent))
    g_gpu = cp.clip(g_gpu, 0.0, 1.0)
    g = cp.asnumpy(g_gpu)
    cp.cuda.Stream.null.synchronize()
    runtime_g = time.perf_counter() - marginal_started

    bivariate_started = time.perf_counter()
    h = np.empty((len(x), 3), dtype=FLOAT_DTYPE)
    for column, (_, i, j) in enumerate(PAIR_SPECS):
        h[:, column] = bivariate_coexit(
            x[:, i], x[:, j], b[:, i], b[:, j], sigma[i], sigma[j], corr[i, j], dt,
            n_terms=bessel_terms,
        )
    runtime_h = time.perf_counter() - bivariate_started

    independent = np.max(np.abs(corr[np.triu_indices(3, 1)])) <= 1.0e-14
    q_started = time.perf_counter()
    if independent:
        nested = {
            "g_nested": g.copy(),
            "h_nested": np.column_stack(
                (g[:, 0] * g[:, 1], g[:, 0] * g[:, 2], g[:, 1] * g[:, 2])
            ),
            "q3_nested_direct": np.prod(g, axis=1),
            "p3_nested_direct": np.prod(1.0 - g, axis=1),
            "runtime_nested_transfer_seconds": 0.0,
            "runtime_nested_gpu_seconds": 0.0,
            "runtime_nested_d2h_seconds": 0.0,
        }
        q_controlled_raw = nested["q3_nested_direct"].copy()
    else:
        if bank is None:
            raise ValueError("a BridgeBank is required outside the independent case")
        nested = nested_exit_components_gpu(x, b, bank)
        ratios = np.divide(
            nested["q3_nested_direct"][:, None],
            nested["h_nested"],
            out=np.zeros_like(nested["h_nested"]),
            where=nested["h_nested"] > 0,
        )
        candidates = h * np.clip(ratios, 0.0, 1.0)
        valid = nested["h_nested"] > 0
        ordered = np.sort(np.where(valid, candidates, np.inf), axis=1)
        valid_count = valid.sum(axis=1)
        q_controlled_raw = nested["q3_nested_direct"].copy()
        q_controlled_raw[valid_count == 1] = ordered[valid_count == 1, 0]
        q_controlled_raw[valid_count == 2] = 0.5 * (
            ordered[valid_count == 2, 0] + ordered[valid_count == 2, 1]
        )
        q_controlled_raw[valid_count == 3] = ordered[valid_count == 3, 1]
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
    runtime_q = time.perf_counter() - q_started
    lower, upper = trivariate_probability_bounds(g, h)
    q3 = np.clip(q_controlled_raw, lower, upper)
    p3_raw = 1.0 - g.sum(axis=1) + h.sum(axis=1) - q_controlled_raw
    p3 = np.clip(1.0 - g.sum(axis=1) + h.sum(axis=1) - q3, 0.0, 1.0)
    return {
        "g": g,
        "h": h,
        "g_nested": nested["g_nested"],
        "h_nested": nested["h_nested"],
        "q3_nested_direct": nested["q3_nested_direct"],
        "q3_controlled_raw": q_controlled_raw,
        "q3_lower": lower,
        "q3_upper": upper,
        "q3": q3,
        "p3_raw": p3_raw,
        "p3": p3,
        "p3_nested_direct": nested["p3_nested_direct"],
        "q3_clipped": np.abs(q3 - q_controlled_raw) > 1.0e-12,
        "endpoint_breach": np.any((b <= 0.0) | (x >= b), axis=1),
        "runtime_g_seconds": runtime_g,
        "runtime_h_seconds": runtime_h,
        "runtime_q3_seconds": runtime_q,
        "runtime_nested_transfer_seconds": nested["runtime_nested_transfer_seconds"],
        "runtime_nested_gpu_seconds": nested["runtime_nested_gpu_seconds"],
        "runtime_nested_d2h_seconds": nested["runtime_nested_d2h_seconds"],
        "independence_shortcut": independent,
        "probability_pipeline": "GPU marginal + GPU nested q3 + CPU SciPy Bessel h_ij",
    }


def conditioned_replication_shared_input(
    contract: DirectContract,
    risk_free_rate: float,
    dividend_yields: np.ndarray,
    volatilities: np.ndarray,
    correlation: np.ndarray,
    annual_coupon: float,
    endpoint_normals: np.ndarray,
    outer_method: str,
    seed: int,
    bridge_bank_seed: int,
    endpoint_backend: str,
    probability_backend: str,
    inner_paths: int = 256,
    bridge_substeps: int = 16,
    bessel_terms: int = 12,
    initial_spot_ratios: np.ndarray | None = None,
    bank_cache: MutableMapping[tuple[Any, ...], BridgeBank] | None = None,
    return_diagnostics: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """M2/M3 price from shared endpoint normals on CPU or the hybrid GPU path."""

    validate_contract(contract)
    if outer_method not in {"mc", "rqmc"}:
        raise ValueError("outer_method must be 'mc' or 'rqmc'")
    if probability_backend not in {"cpu", "gpu-hybrid"}:
        raise ValueError("probability_backend must be 'cpu' or 'gpu-hybrid'")
    q_yield, sigma, corr, initial = _validate_market_inputs(
        dividend_yields, volatilities, correlation, initial_spot_ratios
    )
    n_paths = len(endpoint_normals)
    if n_paths <= 1:
        raise ValueError("endpoint_normals must contain more than one path")
    started = time.perf_counter()
    endpoints, endpoint_timing = generate_endpoint_paths_from_normals(
        contract.observation_times,
        risk_free_rate,
        q_yield,
        sigma,
        corr,
        endpoint_normals,
        backend=endpoint_backend,
        initial_spot_ratios=initial,
    )
    discounts = np.exp(-risk_free_rate * np.asarray(contract.observation_times))
    maturity_discount = float(discounts[-1])
    log_barrier = math.log(contract.ki_barrier_ratio)
    cache: MutableMapping[tuple[Any, ...], BridgeBank] = {} if bank_cache is None else bank_cache

    active = np.ones(n_paths, dtype=bool)
    cumulative_survival = np.ones(n_paths)
    cumulative_log_survival = np.zeros(n_paths)
    final_survival = np.full(n_paths, np.nan)
    bernoulli_survival = np.ones(n_paths, dtype=bool)
    bernoulli_rng = np.random.default_rng(stable_seed(seed, "paired-bernoulli"))
    accrued = np.zeros(n_paths, dtype=np.int16)
    coupon_annuity = np.zeros(n_paths)
    coupon_annuity_bernoulli = np.zeros(n_paths)
    fixed_coupon = np.zeros(n_paths)
    fixed_coupon_bernoulli = np.zeros(n_paths)
    early_principal = np.zeros(n_paths)
    surviving = np.zeros(n_paths)
    no_ki_redemption = np.zeros(n_paths)
    terminal_loss = np.zeros(n_paths)
    terminal_loss_bernoulli = np.zeros(n_paths)
    redemptions = np.zeros(len(contract.observation_times), dtype=np.int64)
    segment_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    previous = np.broadcast_to(np.log(initial), (n_paths, 3)).copy()
    previous_time = 0.0

    for observation_index, observation_time in enumerate(contract.observation_times):
        current = endpoints[:, observation_index, :]
        dt = float(observation_time - previous_time)
        active_index = np.flatnonzero(active)
        if len(active_index):
            start_state = previous[active_index]
            end_state = current[active_index]
            transformed_endpoint = start_state - end_state
            transformed_barrier = np.maximum(start_state - log_barrier, 0.0)
            interval_seed = stable_seed(bridge_bank_seed, round(dt, 14))
            key = (
                round(dt, 14),
                tuple(np.round(sigma, 14)),
                tuple(np.round(corr.ravel(), 14)),
                int(inner_paths),
                int(bridge_substeps),
                int(interval_seed),
            )
            independent = np.max(np.abs(corr[np.triu_indices(3, 1)])) <= 1.0e-14
            if independent:
                bank = None
            else:
                if key not in cache:
                    cache[key] = generate_bridge_bank(
                        dt,
                        sigma,
                        corr,
                        inner_paths,
                        bridge_substeps,
                        interval_seed,
                        method="rqmc",
                    )
                bank = cache[key]
            if probability_backend == "cpu":
                probabilities = segment_probabilities(
                    transformed_endpoint,
                    transformed_barrier,
                    dt,
                    sigma,
                    corr,
                    bank,
                    bessel_terms=bessel_terms,
                )
                probabilities.update(
                    {
                        "runtime_nested_transfer_seconds": 0.0,
                        "runtime_nested_gpu_seconds": 0.0,
                        "runtime_nested_d2h_seconds": 0.0,
                        "probability_pipeline": "NumPy/SciPy CPU reference",
                    }
                )
            else:
                probabilities = segment_probabilities_hybrid_gpu(
                    transformed_endpoint,
                    transformed_barrier,
                    dt,
                    sigma,
                    corr,
                    bank,
                    bessel_terms=bessel_terms,
                )
            p3 = probabilities["p3"]
            cumulative_survival[active_index] *= p3
            with np.errstate(divide="ignore"):
                cumulative_log_survival[active_index] += np.log(p3)
            draws = bernoulli_rng.random(len(active_index))
            bernoulli_survival[active_index] &= draws < p3
            segment_rows.append(
                {
                    "observation_index": observation_index + 1,
                    "observation_time": observation_time,
                    "active_paths": len(active_index),
                    "p3_mean": float(np.mean(p3)),
                    "p3_min": float(np.min(p3)),
                    "p3_max": float(np.max(p3)),
                    "q3_projection_abs_mean": float(
                        np.mean(np.abs(probabilities["q3"] - probabilities["q3_controlled_raw"]))
                    ),
                    "probability_bound_violation_rate": float(
                        np.mean((p3 < 0.0) | (p3 > 1.0))
                    ),
                    "runtime_g_seconds": probabilities["runtime_g_seconds"],
                    "runtime_h_seconds": probabilities["runtime_h_seconds"],
                    "runtime_q3_seconds": probabilities["runtime_q3_seconds"],
                    "runtime_nested_transfer_seconds": probabilities[
                        "runtime_nested_transfer_seconds"
                    ],
                    "runtime_nested_gpu_seconds": probabilities["runtime_nested_gpu_seconds"],
                    "runtime_nested_d2h_seconds": probabilities["runtime_nested_d2h_seconds"],
                    "probability_pipeline": probabilities["probability_pipeline"],
                }
            )
            for quantile in (0.0, 0.01, 0.5, 0.99, 1.0):
                weight_rows.append(
                    {
                        "observation_index": observation_index + 1,
                        "observation_time": observation_time,
                        "quantile": quantile,
                        "cumulative_survival_weight": float(
                            np.quantile(cumulative_survival[active_index], quantile)
                        ),
                        "finite_log_weight": float(
                            np.quantile(
                                np.maximum(
                                    cumulative_log_survival[active_index],
                                    math.log(np.finfo(float).tiny),
                                ),
                                quantile,
                            )
                        ),
                    }
                )

        performance = np.exp(current)
        worst = np.min(performance, axis=1)
        discount = float(discounts[observation_index])
        if (
            contract.coupon_mode in {"periodic", "memory_periodic"}
            and contract.coupon_flags[observation_index]
        ):
            accrued[active] += 1
            paid = active & (
                worst >= float(contract.coupon_trigger_ratios[observation_index])
            )
            paid_periods = accrued[paid] if contract.coupon_memory else np.ones(paid.sum())
            coupon_annuity[paid] += (
                contract.principal
                * paid_periods
                / contract.coupon_periods_per_year
                * discount
            )
            coupon_annuity_bernoulli[paid] = coupon_annuity[paid]
            accrued[paid] = 0
        if contract.autocall_flags[observation_index]:
            called = active & (
                worst >= float(contract.autocall_trigger_ratios[observation_index])
            )
            redemptions[observation_index] += int(called.sum())
            if contract.coupon_mode == "simple_to_redemption":
                amount = contract.principal * observation_time * discount
                coupon_annuity[called] += amount
                coupon_annuity_bernoulli[called] += amount
            if observation_time < contract.maturity_years - 1.0e-12:
                early_principal[called] += contract.principal * discount
            else:
                surviving[called] += contract.principal * discount
                no_ki_redemption[called] += (
                    contract.principal * discount * cumulative_survival[called]
                )
            final_survival[called] = cumulative_survival[called]
            active[called] = False
        previous = current
        previous_time = observation_time

    remaining = active
    maturity_reached = int(n_paths - redemptions[:-1].sum())
    if np.any(remaining):
        terminal_worst = np.min(np.exp(endpoints[:, -1, :]), axis=1)
        surviving[remaining] += contract.principal * maturity_discount
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
                    contract.principal
                    * contract.fixed_maturity_no_ki_coupon_rate
                    * maturity_discount
                )
                fixed_coupon[remaining] += coefficient * cumulative_survival[remaining]
                fixed_coupon_bernoulli[remaining] += coefficient * bernoulli_survival[remaining]
        final_survival[remaining] = cumulative_survival[remaining]

    if np.any(~np.isfinite(final_survival)):
        raise RuntimeError("final survival weight was not assigned")
    coupon_annuity_mean = coupon_annuity.mean()
    fixed_mean = fixed_coupon.mean()
    early_mean = early_principal.mean()
    surviving_mean = surviving.mean()
    loss_mean = terminal_loss.mean()
    base = fixed_mean + early_mean + surviving_mean + loss_mean
    total = base + annual_coupon * coupon_annuity_mean
    coupon_value = fixed_mean + annual_coupon * coupon_annuity_mean
    fair_coupon = (contract.principal - base) / coupon_annuity_mean
    bern_base = (
        fixed_coupon_bernoulli.mean()
        + early_mean
        + surviving_mean
        + terminal_loss_bernoulli.mean()
    )
    bern_total = bern_base + annual_coupon * coupon_annuity_bernoulli.mean()
    segment_frame = pd.DataFrame(segment_rows)
    result: dict[str, Any] = {
        "contract_id": contract.contract_id,
        "method": "M2" if outer_method == "mc" else "M3",
        "outer_method": outer_method,
        "endpoint_backend": endpoint_backend,
        "probability_backend": probability_backend,
        "precision": "float64",
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
        "continuous_ki_survival_probability": final_survival.mean(),
        "maturity_survival_probability": maturity_reached / n_paths,
        "fair_coupon": fair_coupon,
        "fair_coupon_residual": base + fair_coupon * coupon_annuity_mean - contract.principal,
        "paired_bernoulli_total_value": bern_total,
        "conditional_minus_paired_bernoulli": total - bern_total,
        "n_paths": n_paths,
        "endpoint_dimension": len(contract.observation_times) * 3,
        "inner_paths": inner_paths,
        "bridge_substeps": bridge_substeps,
        "bessel_terms": bessel_terms,
        "seed": int(seed),
        "bridge_bank_seed": int(bridge_bank_seed),
        "endpoint_h2d_seconds": endpoint_timing["endpoint_h2d_seconds"],
        "endpoint_kernel_seconds": endpoint_timing["endpoint_kernel_seconds"],
        "endpoint_d2h_seconds": endpoint_timing["endpoint_d2h_seconds"],
        "probability_g_seconds": float(segment_frame["runtime_g_seconds"].sum()),
        "probability_h_bessel_cpu_seconds": float(segment_frame["runtime_h_seconds"].sum()),
        "probability_q3_seconds": float(segment_frame["runtime_q3_seconds"].sum()),
        "nested_gpu_kernel_seconds": float(segment_frame["runtime_nested_gpu_seconds"].sum()),
        "nested_gpu_transfer_seconds": float(
            segment_frame["runtime_nested_transfer_seconds"].sum()
            + segment_frame["runtime_nested_d2h_seconds"].sum()
        ),
        "total_wall_seconds": time.perf_counter() - started,
        "pipeline_claim_boundary": (
            "full CPU reference"
            if probability_backend == "cpu"
            else "hybrid GPU: endpoint/marginal/nested on CUDA; exact Bessel pairs and event loop on CPU"
        ),
    }
    for asset_index, ratio in enumerate(initial):
        result[f"initial_spot_ratio_{asset_index + 1}"] = float(ratio)
    for index, count in enumerate(redemptions):
        result[f"redemption_probability_{index + 1}"] = count / n_paths
    early_probability = sum(
        result[f"redemption_probability_{index + 1}"]
        for index, observation_time in enumerate(contract.observation_times)
        if observation_time < contract.maturity_years - 1.0e-12
    )
    result["probability_mass_error"] = (
        early_probability + result["maturity_survival_probability"] - 1.0
    )
    if return_diagnostics:
        return result, segment_frame, pd.DataFrame(weight_rows)
    return result


def compare_result_components(
    cpu_result: dict[str, Any],
    gpu_result: dict[str, Any],
    fields: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Create a deterministic component-by-component CPU/GPU audit table."""

    selected = list(fields or (
        "coupon_annuity",
        "coupon_value",
        "early_redemption_principal",
        "surviving_notional",
        "no_ki_maturity_redemption",
        "terminal_ki_loss",
        "base_value",
        "total_value",
        "continuous_ki_survival_probability",
        "maturity_survival_probability",
        "fair_coupon",
    ))
    rows = []
    for field in selected:
        cpu_value = float(cpu_result[field])
        gpu_value = float(gpu_result[field])
        absolute = abs(gpu_value - cpu_value)
        rows.append(
            {
                "field": field,
                "cpu_value": cpu_value,
                "gpu_value": gpu_value,
                "absolute_difference": absolute,
                "relative_difference": absolute / max(abs(cpu_value), 1.0e-15),
            }
        )
    return pd.DataFrame(rows)


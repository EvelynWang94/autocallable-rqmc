import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autocallable_bb import generate_bridge_bank, segment_probabilities
from autocallable_direct import parse_research_contract
from autocallable_gpu import (
    central_bump_scenarios,
    compare_result_components,
    conditioned_replication_shared_input,
    configure_cupy_cache,
    direct_price_from_normals,
    generate_endpoint_normal_input,
    generate_normal_input,
    gpu_environment,
    price_scenarios_from_normals,
    segment_probabilities_hybrid_gpu,
)


configure_cupy_cache(ROOT / "tmp" / "cupy_test_cache")


def _gpu_or_skip():
    environment = gpu_environment()
    if not environment["cupy_available"]:
        pytest.skip(environment.get("unavailable_reason", "CUDA is unavailable"))


def _contract():
    config = json.loads((ROOT / "config" / "core_project_config.json").read_text(encoding="utf-8"))
    return parse_research_contract(config, "RC-A")


def _market():
    return {
        "risk_free_rate": 0.03,
        "dividend_yields": np.array([0.01, 0.005, 0.012]),
        "volatilities": np.array([0.18, 0.26, 0.22]),
        "correlation": np.array(
            [[1.0, 0.62, 0.48], [0.62, 1.0, 0.42], [0.48, 0.42, 1.0]]
        ),
        "initial": np.array([1.05, 1.03, 1.02]),
    }


def test_direct_shared_input_cpu_gpu_component_agreement():
    _gpu_or_skip()
    contract = _contract()
    market = _market()
    steps_per_year = 12
    from autocallable_direct import monitoring_grid

    grid, _ = monitoring_grid(contract, steps_per_year)
    normal_input = generate_normal_input(256, len(grid) - 1, "mc", 1401)
    common = dict(
        contract=contract,
        risk_free_rate=market["risk_free_rate"],
        dividend_yields=market["dividend_yields"],
        volatilities=market["volatilities"],
        correlation=market["correlation"],
        annual_coupon=contract.baseline_annual_coupon,
        normals=normal_input.values,
        steps_per_year=steps_per_year,
        batch_size=128,
        initial_spot_ratios=market["initial"],
    )
    cpu = direct_price_from_normals(**common, backend="cpu")
    gpu = direct_price_from_normals(**common, backend="gpu")
    comparison = compare_result_components(cpu, gpu)
    assert comparison["absolute_difference"].max() < 1.0e-10
    assert abs(cpu["component_identity_error"]) < 1.0e-12
    assert abs(gpu["component_identity_error"]) < 1.0e-12
    assert abs(cpu["probability_mass_error"]) < 1.0e-12
    assert abs(gpu["probability_mass_error"]) < 1.0e-12


def test_batched_bump_scenarios_cpu_gpu_agreement():
    _gpu_or_skip()
    contract = _contract()
    market = _market()
    steps_per_year = 12
    from autocallable_direct import monitoring_grid

    grid, _ = monitoring_grid(contract, steps_per_year)
    normal_input = generate_normal_input(128, len(grid) - 1, "rqmc", 1402)
    labels, spots, volatilities = central_bump_scenarios(
        market["initial"], market["volatilities"], 0.005, 0.005
    )
    common = dict(
        contract=contract,
        risk_free_rate=market["risk_free_rate"],
        dividend_yields=market["dividend_yields"],
        correlation=market["correlation"],
        annual_coupon=contract.baseline_annual_coupon,
        normals=normal_input.values,
        steps_per_year=steps_per_year,
        scenario_labels=labels,
        initial_spot_scenarios=spots,
        volatility_scenarios=volatilities,
        batch_size=128,
    )
    cpu, _ = price_scenarios_from_normals(**common, backend="cpu")
    gpu, _ = price_scenarios_from_normals(**common, backend="gpu")
    joined = cpu[["scenario", "total_value"]].merge(
        gpu[["scenario", "total_value"]], on="scenario", suffixes=("_cpu", "_gpu")
    )
    assert np.max(np.abs(joined["total_value_gpu"] - joined["total_value_cpu"])) < 1.0e-10


def test_hybrid_segment_probability_matches_cpu_reference():
    _gpu_or_skip()
    market = _market()
    rng = np.random.default_rng(1403)
    endpoints = rng.normal(0.02, 0.12, size=(48, 3))
    barriers = np.full((48, 3), 0.25)
    bank = generate_bridge_bank(
        0.25,
        market["volatilities"],
        market["correlation"],
        inner_paths=64,
        substeps=8,
        seed=1404,
        method="rqmc",
    )
    cpu = segment_probabilities(
        endpoints,
        barriers,
        0.25,
        market["volatilities"],
        market["correlation"],
        bank,
        bessel_terms=8,
    )
    gpu = segment_probabilities_hybrid_gpu(
        endpoints,
        barriers,
        0.25,
        market["volatilities"],
        market["correlation"],
        bank,
        bessel_terms=8,
    )
    for field in ("g", "h", "g_nested", "h_nested", "q3_nested_direct", "p3"):
        assert np.max(np.abs(gpu[field] - cpu[field])) < 1.0e-12


def test_conditioned_shared_input_cpu_hybrid_gpu_agreement():
    _gpu_or_skip()
    contract = _contract()
    market = _market()
    normal_input = generate_endpoint_normal_input(
        64, len(contract.observation_times), "rqmc", 1405
    )
    common = dict(
        contract=contract,
        risk_free_rate=market["risk_free_rate"],
        dividend_yields=market["dividend_yields"],
        volatilities=market["volatilities"],
        correlation=market["correlation"],
        annual_coupon=contract.baseline_annual_coupon,
        endpoint_normals=normal_input.values,
        outer_method="rqmc",
        seed=1405,
        bridge_bank_seed=2405,
        inner_paths=32,
        bridge_substeps=6,
        bessel_terms=8,
        initial_spot_ratios=market["initial"],
    )
    cpu = conditioned_replication_shared_input(
        **common, endpoint_backend="cpu", probability_backend="cpu"
    )
    gpu = conditioned_replication_shared_input(
        **common, endpoint_backend="gpu", probability_backend="gpu-hybrid"
    )
    comparison = compare_result_components(cpu, gpu)
    assert comparison["absolute_difference"].max() < 1.0e-10
    assert abs(gpu["component_identity_error"]) < 1.0e-12
    assert abs(gpu["probability_mass_error"]) < 1.0e-12


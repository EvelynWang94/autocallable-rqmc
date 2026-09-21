from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autocallable_direct import (  # noqa: E402
    bracketed_bisection,
    equicorrelation,
    parse_research_contract,
    simulate_direct_replication,
)


def load_config():
    return json.loads((ROOT / "config" / "core_project_config.json").read_text(encoding="utf-8"))


class DirectEngineTests(unittest.TestCase):
    def test_parser_preserves_contract_layers(self):
        config = load_config()
        rc_l = parse_research_contract(config, "RC-L")
        rc_a = parse_research_contract(config, "RC-A")
        self.assertEqual(rc_l.coupon_mode, "simple_to_redemption")
        self.assertEqual(rc_a.coupon_mode, "memory_periodic")
        self.assertEqual(rc_l.ki_barrier_ratio, 0.45)
        self.assertEqual(rc_a.ki_barrier_ratio, 0.75)
        self.assertFalse(rc_a.autocall_flags[0])

    def test_small_direct_run_preserves_identities(self):
        contract = parse_research_contract(load_config(), "RC-L")
        result = simulate_direct_replication(
            contract=contract,
            risk_free_rate=0.03,
            dividend_yields=np.zeros(3),
            volatilities=np.full(3, 0.2),
            correlation=equicorrelation(0.4),
            annual_coupon=0.06,
            n_paths=512,
            steps_per_year=24,
            seed=20260805,
            batch_size=256,
        )
        self.assertLessEqual(abs(result["component_identity_error"]), 1e-12)
        self.assertLessEqual(abs(result["probability_mass_error"]), 1e-12)
        self.assertLessEqual(abs(result["fair_coupon_residual"]), 1e-12)

    def test_bracketed_solver_recovers_linear_root(self):
        root = bracketed_bisection(
            lambda value: 80.0 + 250.0 * value - 100.0,
            0.0,
            0.30,
        )
        self.assertLessEqual(abs(root - 0.08), 1e-10)


if __name__ == "__main__":
    unittest.main()

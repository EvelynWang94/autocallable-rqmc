import os
import sys
import unittest

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from autocallable_direct import DirectContract, equicorrelation, simulate_direct_replication  # noqa: E402
from autocallable_greeks import (  # noqa: E402
    autocall_redemption_replication,
    central_first_derivative,
    central_second_derivative,
    roll_contract_to_autocall,
    stable_autocall_rotation,
)


class Day7GreekHelperTests(unittest.TestCase):
    @staticmethod
    def contract():
        return DirectContract(
            contract_id="TEST",
            label="test",
            principal=100.0,
            maturity_years=1.0,
            observation_times=(0.25, 0.5, 1.0),
            autocall_flags=(False, True, True),
            autocall_trigger_ratios=(None, 1.0, 0.9),
            coupon_flags=(True, True, True),
            coupon_trigger_ratios=(0.75, 0.75, 0.75),
            coupon_mode="memory_periodic",
            coupon_memory=True,
            coupon_periods_per_year=4,
            baseline_annual_coupon=0.08,
            ki_barrier_ratio=0.70,
            claim_boundary="test",
        )

    def test_central_differences(self):
        self.assertAlmostEqual(central_first_derivative(1.21, 0.81, 0.1), 2.0)
        self.assertAlmostEqual(central_second_derivative(1.0, 1.21, 0.81, 0.1), 2.0)

    def test_roll_contract_keeps_aligned_remaining_schedule(self):
        rolled = roll_contract_to_autocall(self.contract(), 1, 2.0 / 365.0)
        self.assertEqual(len(rolled.observation_times), 2)
        self.assertTrue(rolled.autocall_flags[0])
        self.assertAlmostEqual(rolled.observation_times[0], 2.0 / 365.0)
        self.assertAlmostEqual(rolled.maturity_years, 0.5 + 2.0 / 365.0)

    def test_householder_rotation_is_valid(self):
        rotation = stable_autocall_rotation(equicorrelation(0.4))
        self.assertLess(rotation["orthogonality_error"], 1e-12)
        self.assertLess(rotation["correlation_error"], 1e-12)
        self.assertGreater(rotation["minimum_final_loading"], 0.0)

    def test_raw_and_smoothed_autocall_prices_agree(self):
        common = dict(
            initial_spot_ratios=np.array([1.002, 1.001, 0.999]),
            trigger_ratios=1.0,
            time_to_observation=2.0 / 365.0,
            principal=100.0,
            risk_free_rate=0.04,
            dividend_yields=np.array([0.01, 0.01, 0.01]),
            volatilities=np.array([0.18, 0.22, 0.25]),
            correlation=equicorrelation(0.4),
            n_paths=16384,
            method="rqmc",
            seed=123,
        )
        raw = autocall_redemption_replication(**common, smooth=False)
        smooth = autocall_redemption_replication(**common, smooth=True)
        self.assertLess(abs(raw["value"] - smooth["value"]), 0.35)

    def test_direct_engine_accepts_initial_spot_ratios(self):
        row = simulate_direct_replication(
            contract=self.contract(),
            risk_free_rate=0.03,
            dividend_yields=np.zeros(3),
            volatilities=np.full(3, 0.20),
            correlation=equicorrelation(0.40),
            annual_coupon=0.08,
            n_paths=128,
            steps_per_year=24,
            seed=9,
            initial_spot_ratios=np.array([0.80, 1.0, 1.0]),
        )
        self.assertEqual(row["initial_spot_ratio_1"], 0.80)
        self.assertTrue(np.isfinite(row["total_value"]))


if __name__ == "__main__":
    unittest.main()

import math
import os
import sys
import unittest

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from autocallable_bb import (  # noqa: E402
    bivariate_coexit,
    generate_bridge_bank,
    marginal_exit,
    segment_probabilities,
    simulate_conditioned_replication,
    trivariate_probability_bounds,
)
from autocallable_direct import DirectContract, equicorrelation  # noqa: E402


class BrownianBridgeProbabilityTests(unittest.TestCase):
    def test_appendix_c_marginals_and_bivariates(self):
        endpoint = np.array([0.01, 0.02, 0.03])
        barrier = np.full(3, 0.10)
        sigma = np.full(3, 0.20)
        maturity = 0.50
        g = marginal_exit(endpoint, barrier, sigma, maturity)
        expected_g = np.exp(-2.0 * barrier * (barrier - endpoint) / (sigma**2 * maturity))
        np.testing.assert_allclose(g, expected_g, atol=1e-14)
        pairs = np.array(
            [
                bivariate_coexit(endpoint[0], endpoint[1], 0.10, 0.10, 0.20, 0.20, 0.40, maturity),
                bivariate_coexit(endpoint[0], endpoint[2], 0.10, 0.10, 0.20, 0.20, 0.40, maturity),
                bivariate_coexit(endpoint[1], endpoint[2], 0.10, 0.10, 0.20, 0.20, 0.40, maturity),
            ]
        )
        np.testing.assert_allclose(pairs, np.array([0.2375, 0.2565, 0.2790]), atol=6e-4)

    def test_vectorised_barriers_match_scalar_calls(self):
        xi = np.array([0.01, -0.02, 0.04])
        xj = np.array([0.02, 0.01, -0.03])
        bi = np.array([0.10, 0.11, 0.12])
        bj = np.array([0.09, 0.10, 0.13])
        vector = bivariate_coexit(xi, xj, bi, bj, 0.2, 0.25, 0.35, 0.5)
        scalar = np.array(
            [bivariate_coexit(xi[k], xj[k], bi[k], bj[k], 0.2, 0.25, 0.35, 0.5) for k in range(3)]
        )
        np.testing.assert_allclose(vector, scalar, atol=1e-13)

    def test_zero_correlation_degenerates_exactly(self):
        endpoints = np.array([[0.01, 0.02, 0.03], [-0.03, 0.04, 0.00]])
        barriers = np.array([[0.10, 0.11, 0.12], [0.08, 0.09, 0.10]])
        sigma = np.array([0.20, 0.25, 0.30])
        result = segment_probabilities(
            endpoints, barriers, 0.5, sigma, np.eye(3), bank=None
        )
        expected = np.prod(1.0 - result["g"], axis=1)
        np.testing.assert_allclose(result["p3"], expected, atol=1e-14)
        np.testing.assert_allclose(result["q3"], np.prod(result["g"], axis=1), atol=1e-14)
        self.assertTrue(result["independence_shortcut"])

    def test_probability_bounds_and_permutation_invariance(self):
        endpoints = np.array([[0.01, 0.02, 0.03], [-0.01, 0.04, 0.02]])
        barriers = np.full_like(endpoints, 0.10)
        sigma = np.full(3, 0.20)
        result = segment_probabilities(endpoints, barriers, 0.5, sigma, np.eye(3), None)
        lower, upper = trivariate_probability_bounds(result["g"], result["h"])
        self.assertTrue(np.all(result["q3"] >= lower - 1e-14))
        self.assertTrue(np.all(result["q3"] <= upper + 1e-14))
        self.assertTrue(np.all((result["p3"] >= 0.0) & (result["p3"] <= 1.0)))
        perm = np.array([2, 0, 1])
        permuted = segment_probabilities(
            endpoints[:, perm], barriers[:, perm], 0.5, sigma[perm], np.eye(3), None
        )
        np.testing.assert_allclose(result["p3"], permuted["p3"], atol=1e-14)

    def test_nested_probability_outputs_are_finite(self):
        sigma = np.full(3, 0.20)
        corr = equicorrelation(0.40)
        bank = generate_bridge_bank(0.5, sigma, corr, 128, 12, 123)
        result = segment_probabilities(
            np.array([[0.01, 0.02, 0.03], [0.03, 0.02, 0.01]]),
            np.full((2, 3), 0.10),
            0.5,
            sigma,
            corr,
            bank,
        )
        for key in ("g", "h", "q3", "p3", "q3_nested_direct", "p3_nested_direct"):
            self.assertTrue(np.all(np.isfinite(result[key])))
        self.assertTrue(np.all((result["p3"] >= 0.0) & (result["p3"] <= 1.0)))


class ConditionedPricingTests(unittest.TestCase):
    @staticmethod
    def _contract() -> DirectContract:
        return DirectContract(
            contract_id="TEST",
            label="two-date test",
            principal=100.0,
            maturity_years=1.0,
            observation_times=(0.5, 1.0),
            autocall_flags=(True, True),
            autocall_trigger_ratios=(0.10, 0.10),
            coupon_flags=(False, False),
            coupon_trigger_ratios=(None, None),
            coupon_mode="simple_to_redemption",
            coupon_memory=False,
            coupon_periods_per_year=2,
            baseline_annual_coupon=0.06,
            ki_barrier_ratio=0.50,
            claim_boundary="unit test",
        )

    def test_component_probability_identities_and_stop_after_call(self):
        result, segment, weights = simulate_conditioned_replication(
            contract=self._contract(),
            risk_free_rate=0.03,
            dividend_yields=np.zeros(3),
            volatilities=np.full(3, 0.20),
            correlation=equicorrelation(0.40),
            annual_coupon=0.06,
            n_paths=128,
            outer_method="rqmc",
            seed=19,
            bridge_bank_seed=29,
            inner_paths=64,
            bridge_substeps=8,
            return_diagnostics=True,
        )
        self.assertLess(abs(result["component_identity_error"]), 1e-12)
        self.assertLess(abs(result["probability_mass_error"]), 1e-12)
        self.assertLess(abs(result["fair_coupon_residual"]), 1e-12)
        self.assertEqual(len(segment), 1)
        self.assertEqual(set(weights["observation_index"]), {1})
        self.assertAlmostEqual(result["redemption_probability_1"], 1.0, places=12)
        self.assertEqual(result["endpoint_dimension"], 6)

    def test_mc_and_rqmc_conditioned_paths_execute(self):
        values = []
        for method in ("mc", "rqmc"):
            row = simulate_conditioned_replication(
                contract=self._contract(),
                risk_free_rate=0.03,
                dividend_yields=np.zeros(3),
                volatilities=np.full(3, 0.20),
                correlation=equicorrelation(0.40),
                annual_coupon=0.06,
                n_paths=64,
                outer_method=method,
                seed=7,
                bridge_bank_seed=8,
                inner_paths=32,
                bridge_substeps=6,
            )
            self.assertTrue(math.isfinite(row["total_value"]))
            values.append(row["total_value"])
        self.assertLess(abs(values[0] - values[1]), 0.5)


if __name__ == "__main__":
    unittest.main()

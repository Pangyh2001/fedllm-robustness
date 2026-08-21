import unittest
from collections import OrderedDict

import torch

from fedrda_experiments.aggregation import (
    clean_preserving_residual,
    fedrda_residual,
    qfedavg_update,
    risk_aware_update,
    residual_diagnostics,
    sfat_update,
    tail_reweighted_residual,
)
from fedrda_experiments.state import flatten


def state(values):
    return OrderedDict(weight=torch.tensor(values, dtype=torch.float32))


class AggregationTests(unittest.TestCase):
    def test_conflict_diagnostics_detect_opposite_vectors(self):
        diagnostics = residual_diagnostics(
            [state([1.0, 0.0]), state([-1.0, 0.0])],
            [0.5, 0.5],
        )
        self.assertEqual(diagnostics.average_pair_cosine, -1.0)
        self.assertEqual(diagnostics.conflict_rate, 1.0)

    def test_tail_reweight_moves_toward_tail(self):
        result = tail_reweighted_residual(
            [state([1.0, 0.0]), state([0.0, 1.0])],
            [0.5, 0.5],
            [0, 1],
            {1},
            3.0,
        )
        self.assertGreater(flatten(result)[1], flatten(result)[0])

    def test_fedrda_satisfies_weak_constraints(self):
        residuals = [
            state([1.0, 0.0]),
            state([-0.8, 0.6]),
            state([0.0, 1.0]),
        ]
        result, metrics = fedrda_residual(
            residuals,
            [1 / 3] * 3,
            [0, 1, 2],
            {1},
            rho=10.0,
            kappa=0.1,
        )
        vector = flatten(result)
        self.assertTrue(metrics["success"])
        self.assertGreater(vector.norm(), 0)
        unit = vector / vector.norm()
        for residual in residuals:
            direction = flatten(residual)
            direction = direction / direction.norm()
            self.assertGreaterEqual(torch.dot(unit, direction), -0.1001)

    def test_sfat_upweights_low_loss_client(self):
        result, weights = sfat_update(
            [state([1.0, 0.0]), state([0.0, 1.0])],
            [0.1, 1.0],
            top_k=1,
            multiplier=2.0,
            use_slack=True,
        )
        self.assertGreater(weights[0], weights[1])
        self.assertGreater(flatten(result)[0], flatten(result)[1])

    def test_clean_preserving_residual_removes_conflict(self):
        result, metrics = clean_preserving_residual(
            state([1.0, 0.0]),
            state([-1.0, 1.0]),
            norm_cap=2.0,
        )
        self.assertTrue(metrics["conflict_removed"])
        self.assertGreaterEqual(torch.dot(flatten(result), torch.tensor([1.0, 0.0])), 0)

    def test_qfedavg_emphasizes_high_loss_update(self):
        result, metrics = qfedavg_update(
            [state([1.0, 0.0]), state([0.0, 1.0])],
            [0.1, 1.0],
            learning_rate=0.1,
            q=1.0,
        )
        self.assertGreater(flatten(result)[1], flatten(result)[0])
        self.assertGreater(metrics["denominator"], 0)

    def test_risk_aware_update_emphasizes_weak_client(self):
        result, metrics = risk_aware_update(
            [state([1.0, 0.0]), state([0.0, 1.0])],
            [0.5, 0.5],
            [0, 1],
            {0: 0.2, 1: 1.0},
            {1},
            temperature=1.0,
            tail_multiplier=2.0,
            weight_cap=3.0,
        )
        self.assertGreater(metrics["weights"][1], metrics["weights"][0])
        self.assertGreater(flatten(result)[1], flatten(result)[0])

if __name__ == "__main__":
    unittest.main()

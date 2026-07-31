import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / 'flow_metrics.py'
SPEC = importlib.util.spec_from_file_location('flow_metrics', MODULE_PATH)
flow_metrics = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(flow_metrics)


class FlowMetricsTests(unittest.TestCase):
    def test_balanced_accuracy_is_mean_population_recall(self):
        result = flow_metrics.compute_prediction_metrics(
            np.array([1, 1, 2, 2, 2]),
            np.array([1, 2, 2, 2, 1]),
            ['balanced_accuracy', 'recall'],
        )

        self.assertAlmostEqual(result['balanced_accuracy'], (0.5 + 2 / 3) / 2)
        self.assertEqual(result['balanced_accuracy'], result['recall_macro'])

    def test_completely_missed_population_contributes_zero_f1(self):
        result = flow_metrics.compute_prediction_metrics(
            np.array([1, 1, 2, 2]),
            np.array([1, 1, 1, 1]),
            ['f1', 'precision'],
        )

        self.assertEqual(result['per_population']['2']['precision'], 0.0)
        self.assertEqual(result['per_population']['2']['f1'], 0.0)
        self.assertAlmostEqual(result['f1_macro'], 1 / 3)

    def test_missing_prediction_is_scored_as_rejection(self):
        result = flow_metrics.compute_prediction_metrics(
            np.array([1, 1, 0]),
            np.array([1, np.nan, np.nan]),
            ['accuracy'],
        )

        self.assertEqual(result['n_cells'], 2)
        self.assertEqual(result['accuracy'], 0.5)
        self.assertEqual(result['n_truth_positive'], 2)
        self.assertEqual(result['n_truth_zero'], 1)
        self.assertEqual(result['n_pred_zero_on_truth_positive'], 1)
        self.assertEqual(result['n_pred_zero_on_truth_zero'], 1)
        self.assertEqual(result['n_pred_missing_mapped_to_zero'], 1)
        self.assertEqual(result['rejection_rate_on_truth_positive'], 0.5)


if __name__ == '__main__':
    unittest.main()

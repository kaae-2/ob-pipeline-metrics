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
    def test_balanced_accuracy_is_mean_one_vs_rest_balanced_accuracy(self):
        result = flow_metrics.compute_prediction_metrics(
            np.array([1, 1, 2, 2, 3, 3]),
            np.array([1, 2, 2, 3, 3, 3]),
            ['balanced_accuracy', 'recall'],
        )

        expected_per_population = {
            '1': (0.5 + 1.0) / 2,
            '2': (0.5 + 0.75) / 2,
            '3': (1.0 + 0.75) / 2,
        }
        for population, expected in expected_per_population.items():
            self.assertAlmostEqual(
                result['per_population'][population]['specificity'],
                2 * expected - result['per_population'][population]['recall'],
            )
            self.assertAlmostEqual(
                result['per_population'][population]['balanced_accuracy'],
                expected,
            )

        self.assertAlmostEqual(
            result['balanced_accuracy'],
            sum(expected_per_population.values()) / len(expected_per_population),
        )
        self.assertNotEqual(result['balanced_accuracy'], result['recall_macro'])

    def test_completely_missed_population_contributes_zero_f1(self):
        result = flow_metrics.compute_prediction_metrics(
            np.array([1, 1, 2, 2]),
            np.array([1, 1, 1, 1]),
            ['f1', 'precision'],
            training_support_by_label={'1': 2, '2': 2},
        )

        self.assertEqual(result['per_population']['2']['precision'], 0.0)
        self.assertEqual(result['per_population']['2']['f1'], 0.0)
        self.assertAlmostEqual(result['f1_macro'], 1 / 3)

    def test_population_absent_from_training_still_contributes_to_metrics(self):
        result = flow_metrics.compute_prediction_metrics(
            np.array([1, 1, 2, 2]),
            np.array([1, 1, 1, 1]),
            ['f1', 'precision', 'recall'],
            training_support_by_label={'1': 2, '2': 0},
        )

        absent = result['per_population']['2']
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'scaling_rate']:
            self.assertEqual(absent[metric], 0.0)
        self.assertEqual(absent['tp'], 0)
        self.assertEqual(absent['fn'], 2)
        self.assertEqual(absent['support'], 2)
        self.assertAlmostEqual(result['f1_macro'], 1 / 3)

    def test_population_absent_from_training_uses_actual_predictions(self):
        result = flow_metrics.compute_prediction_metrics(
            np.array([1, 1, 2, 2]),
            np.array([1, 1, 2, 2]),
            ['f1', 'precision', 'recall'],
            training_support_by_label={'1': 2, '2': 0},
        )

        absent = result['per_population']['2']
        self.assertEqual(absent['precision'], 1.0)
        self.assertEqual(absent['recall'], 1.0)
        self.assertEqual(absent['f1'], 1.0)
        self.assertEqual(result['f1_macro'], 1.0)

    def test_training_support_is_read_from_split_audit(self):
        metadata = {
            'split_audit': {
                'populations': [
                    {'id': 1, 'training_support': 4},
                    {'id': 2.0, 'training_support': 0},
                ],
            },
        }

        self.assertEqual(
            flow_metrics._read_training_support_by_label(metadata),
            {'1': 4.0, '2': 0.0},
        )

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

    def test_all_missing_predictions_are_recognized_as_not_run(self):
        self.assertTrue(
            flow_metrics.predictions_are_all_missing(
                np.array([np.nan, np.nan]),
            )
        )
        self.assertFalse(
            flow_metrics.predictions_are_all_missing(
                np.array([np.nan, 1]),
            )
        )
        self.assertFalse(flow_metrics.predictions_are_all_missing(np.array([])))


if __name__ == '__main__':
    unittest.main()

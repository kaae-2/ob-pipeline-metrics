import sys
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(MODULE_DIR))
try:
    import batch_flow_metrics
finally:
    sys.path.pop(0)


class BatchFlowMetricsTests(unittest.TestCase):
    def test_artifact_without_one_vs_rest_metrics_needs_rerun(self):
        population = {
            'support': 2,
            'f1': 1.0,
            'precision': 1.0,
            'recall': 1.0,
            'accuracy': 1.0,
            'scaling_rate': 0.5,
        }
        payload = {
            'results': {
                'run0': {'per_population': {'1': population, '2': population.copy()}},
            },
        }

        with patch.object(
            batch_flow_metrics.flow_metrics,
            '_read_json_maybe_gzip',
            return_value=payload,
        ):
            self.assertTrue(batch_flow_metrics.artifact_needs_rerun('old.json.gz'))

    def test_artifact_with_finite_one_vs_rest_metrics_is_current(self):
        population = {
            'support': 2,
            'f1': 1.0,
            'precision': 1.0,
            'recall': 1.0,
            'specificity': 1.0,
            'balanced_accuracy': 1.0,
            'accuracy': 1.0,
            'scaling_rate': 0.5,
        }
        payload = {
            'results': {
                'run0': {'per_population': {'1': population, '2': population.copy()}},
            },
        }

        with patch.object(
            batch_flow_metrics.flow_metrics,
            '_read_json_maybe_gzip',
            return_value=payload,
        ):
            self.assertFalse(batch_flow_metrics.artifact_needs_rerun('new.json.gz'))


if __name__ == '__main__':
    unittest.main()

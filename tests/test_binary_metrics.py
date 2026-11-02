"""CPU regression checks for binary benchmark scoring."""

import unittest

import numpy as np

from src.scripts.finetune_dlb_binary_classification import _average_precision_binary


class AveragePrecision(unittest.TestCase):
    def test_all_tied_scores_ignore_sample_order(self):
        scores = np.array([0.5, 0.5])
        for labels in ([1, 0], [0, 1]):
            self.assertAlmostEqual(_average_precision_binary(np.array(labels), scores), 0.5)

    def test_mixed_ties_ignore_sample_order(self):
        labels = np.array([1, 0, 1, 0])
        scores = np.array([0.9, 0.9, 0.2, 0.1])
        for order in ([0, 1, 2, 3], [1, 0, 2, 3], [3, 2, 1, 0]):
            self.assertAlmostEqual(_average_precision_binary(labels[order], scores[order]), 7 / 12)

    def test_distinct_scores(self):
        labels = np.array([1, 0, 1])
        scores = np.array([0.9, 0.5, 0.1])
        self.assertAlmostEqual(_average_precision_binary(labels, scores), 5 / 6)


if __name__ == "__main__":
    unittest.main()

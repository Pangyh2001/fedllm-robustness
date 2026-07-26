import unittest

import numpy as np

from fedrda_experiments.data import (
    balanced_label_skew_partitions,
    shared_dirichlet_partitions,
)


class DataPartitionTests(unittest.TestCase):
    def test_shared_dirichlet_partition_uses_every_example_once(self):
        train_labels = np.repeat(np.arange(4), 100)
        test_labels = np.repeat(np.arange(4), 20)
        train, test = shared_dirichlet_partitions(
            train_labels,
            test_labels,
            num_clients=5,
            alpha=0.5,
            min_samples=4,
            rng=np.random.default_rng(7),
        )
        self.assertEqual(
            sorted(index for part in train for index in part),
            list(range(len(train_labels))),
        )
        self.assertEqual(
            sorted(index for part in test for index in part),
            list(range(len(test_labels))),
        )
        self.assertGreaterEqual(min(map(len, train)), 4)
        self.assertGreaterEqual(min(map(len, test)), 1)

    def test_balanced_label_skew_uses_every_example_and_equal_sizes(self):
        train_labels = np.repeat(np.arange(4), 101)
        test_labels = np.repeat(np.arange(4), 21)
        train, test = balanced_label_skew_partitions(
            train_labels,
            test_labels,
            num_clients=5,
            alpha=0.1,
            rng=np.random.default_rng(9),
        )
        self.assertEqual(
            sorted(index for part in train for index in part),
            list(range(len(train_labels))),
        )
        self.assertEqual(
            sorted(index for part in test for index in part),
            list(range(len(test_labels))),
        )
        self.assertLessEqual(max(map(len, train)) - min(map(len, train)), 1)
        self.assertLessEqual(max(map(len, test)) - min(map(len, test)), 1)


if __name__ == "__main__":
    unittest.main()

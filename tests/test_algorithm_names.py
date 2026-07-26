import unittest

from fedrda_experiments.config import FederatedConfig
from fedrda_experiments.runner import SUPPORTED_ALGORITHMS


class AlgorithmNameTests(unittest.TestCase):
    def test_complete_method_is_named_fedrda(self):
        self.assertEqual(FederatedConfig().algorithm, "fedrda")
        self.assertIn("fedrda", SUPPORTED_ALGORITHMS)

    def test_deprecated_fedeat_names_are_removed(self):
        self.assertNotIn("fedeat", SUPPORTED_ALGORITHMS)
        self.assertNotIn("fedeat_tail", SUPPORTED_ALGORITHMS)


if __name__ == "__main__":
    unittest.main()

import unittest
from posteriorModel import F

class TestPosteriorModel(unittest.TestCase):
    def test_F(self):
        s = {
            "Parameters": [0.1, 0.2, 0.3],
            "Reference Evaluations": [],
            "Standard Deviation": [],
            "error_fit": []
        }
        T = [1, 2, 3]
        F(s, T)

        # Assert that the "Reference Evaluations" list is populated
        self.assertEqual(len(s["Reference Evaluations"]), len(T))

        # Assert that the "Standard Deviation" list is populated
        self.assertEqual(len(s["Standard Deviation"]), len(T))

        # Assert that the "error_fit" list is populated
        self.assertEqual(len(s["error_fit"]), len(T))

if __name__ == '__main__':
    unittest.main()
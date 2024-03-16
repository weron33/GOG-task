import os
import unittest

from typing import List, Tuple

os.environ.setdefault('MODEL_NAME', 'LightGCN')

from main import get_recommendations


class TestMain(unittest.TestCase):
    def setUp(self) -> None:
        self._negative_input = 'Some string'
        self._negative_output = TypeError

        self._positive_input = 9
        self._target_output_length = 10

    def test_negative_input(self):
        self.assertRaises(
            self._negative_output,
            get_recommendations,
            self._negative_input
        )

    def test_positive_output_type(self):
        ans = get_recommendations(self._positive_input)
        for item in ans:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)
            self.assertIsInstance(item[0], int)
            self.assertIsInstance(item[1], float)

    def test_positive_output_length(self):
        ans = get_recommendations(self._positive_input)
        self.assertEquals(len(ans), self._target_output_length)


if __name__ == '__main__':
    unittest.main()

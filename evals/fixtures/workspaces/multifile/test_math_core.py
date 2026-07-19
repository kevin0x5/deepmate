import unittest

from math_core import total


class MathCoreTests(unittest.TestCase):
    def test_total(self) -> None:
        self.assertEqual(total([2, 3, 5]), 10)


if __name__ == "__main__":
    unittest.main()

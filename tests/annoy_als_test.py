from __future__ import print_function

import unittest

from implicit.annoy_als import AnnoyAlternatingLeastSquares

from .recommender_base_test import TestRecommenderBaseMixin


class AnnoyALSTest(unittest.TestCase, TestRecommenderBaseMixin):
    def _get_model(self):
        return AnnoyAlternatingLeastSquares(factors=3, regularization=0)


if __name__ == "__main__":
    unittest.main()

import unittest

from implicit.lmf import LogisticMatrixFactorization

from .recommender_base_test import TestRecommenderBaseMixin


class LMFTest(unittest.TestCase, TestRecommenderBaseMixin):
    def _get_model(self):
        return LogisticMatrixFactorization(factors=3, regularization=0, use_gpu=False)


if __name__ == "__main__":
    unittest.main()

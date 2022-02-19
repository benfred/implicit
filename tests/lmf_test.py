import unittest

from recommender_base_test import RecommenderBaseTestMixin

from implicit.lmf import LogisticMatrixFactorization


class LMFTest(unittest.TestCase, RecommenderBaseTestMixin):
    def _get_model(self):
        return LogisticMatrixFactorization(
            factors=3, regularization=0, use_gpu=False, random_state=43
        )

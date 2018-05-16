import unittest

from implicit.bpr import BayesianPersonalizedRanking
from implicit.cuda import HAS_CUDA

from .recommender_base_test import TestRecommenderBaseMixin


class BPRTest(unittest.TestCase, TestRecommenderBaseMixin):

    def _get_model(self):
        return BayesianPersonalizedRanking(factors=3, regularization=0, use_gpu=False)


if HAS_CUDA:
    class BPRGPUTest(unittest.TestCase, TestRecommenderBaseMixin):

        def _get_model(self):
            return BayesianPersonalizedRanking(factors=3, regularization=0, use_gpu=True)


if __name__ == "__main__":
    unittest.main()

import unittest

from scipy.sparse import csr_matrix

from implicit.bpr import BayesianPersonalizedRanking
from implicit.gpu import HAS_CUDA

from .recommender_base_test import RecommenderBaseTestMixin


class BPRTest(unittest.TestCase, RecommenderBaseTestMixin):
    def _get_model(self):
        return BayesianPersonalizedRanking(
            factors=3, regularization=0, use_gpu=False, random_state=42
        )


if HAS_CUDA:

    class BPRGPUTest(unittest.TestCase, RecommenderBaseTestMixin):
        def _get_model(self):
            return BayesianPersonalizedRanking(
                factors=3,
                regularization=0,
                use_gpu=True,
                learning_rate=0.05,
                random_state=42,
            )


# Test issue #264 causing crashes on empty matrices
def test_fit_empty_matrix():
    raw = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    return BayesianPersonalizedRanking(use_gpu=False).fit(csr_matrix(raw), show_progress=False)


# Test issue #264 causing crashes on almost empty matrices
def test_fit_almost_empty_matrix():
    raw = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    return BayesianPersonalizedRanking(use_gpu=False).fit(csr_matrix(raw), show_progress=False)

import unittest

from recommender_base_test import RecommenderBaseTestMixin
from scipy.sparse import csr_matrix

from implicit.bpr import BayesianPersonalizedRanking
from implicit.gpu import HAS_CUDA


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
                learning_rate=0.1,
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


def test_fit_callback():
    num_called = 0

    def callback(epoch, elapsed, correct, skipped):
        nonlocal num_called
        num_called += 1

    raw = [
        [1, 1, 0, 1, 0, 0],
        [0, 1, 1, 1, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 1],
        [0, 1, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 1],
    ]
    model = BayesianPersonalizedRanking(iterations=5, use_gpu=False)

    model.fit(csr_matrix(raw), show_progress=False, callback=callback)
    assert num_called == model.iterations

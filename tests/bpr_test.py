import unittest

from implicit.bpr import BayesianPersonalizedRanking
from scipy.sparse import csr_matrix

from implicit.cuda import HAS_CUDA
from .recommender_base import RecommenderBaseMixin


class BPR(unittest.TestCase, RecommenderBaseMixin):

    def _get_model(self):
        return BayesianPersonalizedRanking(factors=3, regularization=0, use_gpu=False)

    # Test issue #264 causing crashes on empty matrices
    def test_fit_empty_matrix(self):
        raw = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        return BayesianPersonalizedRanking().fit(csr_matrix(raw))

    # Test issue #264 causing crashes on almost empty matrices
    def test_fit_almost_empty_matrix(self):
        raw = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        return BayesianPersonalizedRanking().fit(csr_matrix(raw))


if HAS_CUDA:
    class BPRGPU(unittest.TestCase, RecommenderBaseMixin):

        def _get_model(self):
            return BayesianPersonalizedRanking(factors=3, regularization=0, use_gpu=True)

if __name__ == "__main__":
    unittest.main()

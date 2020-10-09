import unittest

from scipy.sparse import csr_matrix

from implicit.cml import CollaborativeMetricLearning

from .recommender_base_test import TestRecommenderBaseMixin


class LMFTest(unittest.TestCase, TestRecommenderBaseMixin):
    def _get_model(self):
        return CollaborativeMetricLearning(factors=10, regularization=1.0,
                                           use_gpu=False,
                                           random_state=43)

    def test_fit_empty_matrix(self):
        raw = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        model = self._get_model()
        model.fit(csr_matrix(raw), show_progress=False)


if __name__ == "__main__":
    unittest.main()

import unittest

from scipy.sparse import csr_matrix

from implicit.cml import CollaborativeMetricLearning

from .recommender_base_test import TestRecommenderBaseMixin


class CMLTest(unittest.TestCase, TestRecommenderBaseMixin):
    def _get_model(self, factors=8):
        return CollaborativeMetricLearning(factors=3,
                                           use_gpu=False,
                                           random_state=33)

    def test_fit_empty_matrix(self):
        raw = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        model = self._get_model()
        model.fit(csr_matrix(raw), show_progress=False)


if __name__ == "__main__":
    unittest.main()

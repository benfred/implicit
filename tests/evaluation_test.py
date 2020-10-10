from __future__ import print_function

import unittest

import numpy as np
from scipy.sparse import csr_matrix, random

from implicit.evaluation import recommender_split, train_test_split


class EvaluationTest(unittest.TestCase):
    def _get_sample_matrix(self):
        return csr_matrix((np.random.random((10, 10)) > 0.5).astype(np.float64))

    def _get_implicit_matrix(self):
        m = random(1000, 500, density=0.1, format="csr")
        m[m > 0] = 1
        return m.tocoo()

    def test_split(self):
        seed = np.random.randint(1000)
        mat = self._get_sample_matrix()
        train, test = train_test_split(mat, 0.8, seed)
        train2, test2 = train_test_split(mat, 0.8, seed)
        self.assertTrue(np.all(train.todense() == train2.todense()))

    def test_recommender_split_single_sample(self):
        mat = self._get_implicit_matrix()
        train, test = recommender_split(mat, n_samples=1)
        self.assertTrue(((train + test) - mat).nnz == 0)


if __name__ == "__main__":
    unittest.main()

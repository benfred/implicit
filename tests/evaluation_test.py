from __future__ import print_function

import unittest

import numpy as np
from scipy.sparse import csr_matrix

from implicit.evaluation import train_test_split


class EvaluationTest(unittest.TestCase):
    def _get_sample_matrix(self):
        return csr_matrix((np.random.random((10, 10)) > 0.5).astype(np.float64))

    def test_split(self):
        seed = np.random.randint(1000)
        mat = self._get_sample_matrix()
        train, test = train_test_split(mat, 0.8, seed)
        train2, test2 = train_test_split(mat, 0.8, seed)
        self.assertTrue(np.all(train.todense() == train2.todense()))

if __name__ == "__main__":
    unittest.main()

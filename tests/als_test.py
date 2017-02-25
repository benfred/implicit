from __future__ import print_function

import unittest

import numpy as np
from scipy.sparse import csr_matrix

from implicit.als import AlternatingLeastSquares

from .recommender_base_test import TestRecommenderBaseMixin


class ALSTest(unittest.TestCase, TestRecommenderBaseMixin):
    def _get_model(self):
        return AlternatingLeastSquares(factors=3, regularization=0)

    def test_factorize(self):
        counts = csr_matrix([[1, 1, 0, 1, 0, 0],
                             [0, 1, 1, 1, 0, 0],
                             [1, 0, 1, 0, 0, 0],
                             [1, 1, 0, 0, 0, 0],
                             [0, 0, 1, 1, 0, 1],
                             [0, 1, 0, 0, 0, 1],
                             [0, 0, 0, 0, 1, 1]], dtype=np.float64)

        # try all 8 variants of native/python, cg/cholesky, and
        # 64 vs 32 bit factors
        for dtype in (np.float32, np.float64):
            for use_cg in (False, True):
                for use_native in (True, False):
                    try:
                        model = AlternatingLeastSquares(factors=6,
                                                        regularization=1e-10,
                                                        dtype=dtype,
                                                        use_native=use_native,
                                                        use_cg=use_cg)
                        np.random.seed(23)
                        model.fit(counts * 2)
                        rows, cols = model.item_factors, model.user_factors

                    except Exception as e:
                        self.fail(msg="failed to factorize matrix. Error=%s"
                                      " dtype=%s, cg=%s, native=%s"
                                      % (e, dtype, use_cg, use_native))

                    reconstructed = rows.dot(cols.T)
                    for i in range(counts.shape[0]):
                        for j in range(counts.shape[1]):
                            self.assertAlmostEqual(counts[i, j], reconstructed[i, j],
                                                   delta=0.0001,
                                                   msg="failed to reconstruct row=%s, col=%s,"
                                                       " value=%.5f, dtype=%s, cg=%s, native=%s"
                                                       % (i, j, reconstructed[i, j], dtype, use_cg,
                                                          use_native))


if __name__ == "__main__":
    unittest.main()

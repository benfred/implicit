from __future__ import print_function

import unittest

import numpy as np
from scipy.sparse import csr_matrix

from implicit.als import AlternatingLeastSquares

from .recommender_base_test import TestRecommenderBaseMixin


class ALSTest(unittest.TestCase, TestRecommenderBaseMixin):
    def _get_model(self):
        return AlternatingLeastSquares(factors=3, regularization=0)

    def test_cg_nan(self):
        # test issue with CG code that was causing NaN values in output:
        # https://github.com/benfred/implicit/issues/19#issuecomment-283164905
        raw = [[0.0, 2.0, 1.5, 1.33333333, 1.25, 1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 2.0, 1.5, 1.33333333, 1.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 2.0, 1.5, 1.33333333, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 2.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.5, 1.33333333, 1.25, 1.2],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.5, 1.33333333, 1.25],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.5, 1.33333333],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.5],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        counts = csr_matrix(raw, dtype=np.float64)
        for use_native in (True, False):
            model = AlternatingLeastSquares(factors=3,
                                            regularization=0.01,
                                            dtype=np.float64,
                                            use_native=use_native,
                                            use_cg=True)
            model.fit(counts)
            rows, cols = model.item_factors, model.user_factors

            self.assertFalse(np.isnan(np.sum(cols)))
            self.assertFalse(np.isnan(np.sum(rows)))

    def test_factorize(self):
        counts = csr_matrix([[1, 1, 0, 1, 0, 0],
                             [0, 1, 1, 1, 0, 0],
                             [1, 0, 1, 0, 0, 0],
                             [1, 1, 0, 0, 0, 0],
                             [0, 0, 1, 1, 0, 1],
                             [0, 1, 0, 0, 0, 1],
                             [0, 0, 0, 0, 1, 1]], dtype=np.float64)
        user_items = counts * 2

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
                        model.fit(user_items)
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

    def test_explain(self):
        counts = csr_matrix([[1, 1, 0, 1, 0, 0],
                             [0, 1, 1, 1, 0, 0],
                             [1, 0, 1, 0, 0, 0],
                             [1, 1, 0, 0, 0, 0],
                             [0, 0, 1, 1, 0, 1],
                             [0, 1, 0, 0, 0, 1],
                             [0, 0, 0, 0, 1, 1]], dtype=np.float64)
        user_items = counts * 2

        model = AlternatingLeastSquares(factors=4,
                                        regularization=1e-1,
                                        use_native=False,
                                        use_cg=False,
                                        iterations=10)
        np.random.seed(23)
        model.fit(user_items)
        model.item_factors, model.user_factors

        userid = 0
        recs = model.recommend(userid, user_items, N=10, recalculate_user=True)
        top_rec, score = recs[0]
        score_explained, contributions, W = model.explain(userid, user_items, itemid=top_rec)
        self.assertAlmostEqual(score, score_explained, 4)
        self.assertAlmostEqual(score, sum(s for _, s in contributions), 4)


if __name__ == "__main__":
    unittest.main()

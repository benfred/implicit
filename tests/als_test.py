from __future__ import print_function

import unittest

import numpy as np
from scipy.sparse import csr_matrix, random

from implicit.als import AlternatingLeastSquares
from implicit.cuda import HAS_CUDA

from .recommender_base_test import TestRecommenderBaseMixin


class ALSTest(unittest.TestCase, TestRecommenderBaseMixin):

    def _get_model(self):
        return AlternatingLeastSquares(factors=3, regularization=0, use_gpu=False)

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
                                            use_cg=True,
                                            use_gpu=False)
            model.fit(counts, show_progress=False)
            rows, cols = model.item_factors, model.user_factors

            self.assertFalse(np.isnan(np.sum(cols)))
            self.assertFalse(np.isnan(np.sum(rows)))

    def test_cg_nan2(self):
        # test out Nan appearing in CG code (from https://github.com/benfred/implicit/issues/106)
        Ciu = random(m=100, n=100, density=0.0005, format='coo', dtype=np.float32,
                     random_state=42, data_rvs=None).T.tocsr()

        configs = [{'use_native': True, 'use_gpu': False}, {'use_native': False, 'use_gpu': False}]
        if HAS_CUDA:
            configs.append({'use_gpu': True})

        for options in configs:
            model = AlternatingLeastSquares(factors=32, regularization=10, iterations=10,
                                            dtype=np.float32, **options)
            model.fit(Ciu, show_progress=False)

            self.assertTrue(np.isfinite(model.item_factors).all())
            self.assertTrue(np.isfinite(model.user_factors).all())

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
        options = [(dtype, cg, native, False)
                   for dtype in (np.float32, np.float64)
                   for cg in (False, True)
                   for native in (False, True)]

        # also try out GPU support if available
        if HAS_CUDA:
            options.append((np.float32, False, False, True))

        for dtype, use_cg, use_native, use_gpu in options:
            try:
                model = AlternatingLeastSquares(factors=6,
                                                regularization=0,
                                                dtype=dtype,
                                                use_native=use_native,
                                                use_cg=use_cg,
                                                use_gpu=use_gpu)
                np.random.seed(23)
                model.fit(user_items, show_progress=False)
                rows, cols = model.item_factors, model.user_factors

            except Exception as e:
                self.fail(msg="failed to factorize matrix. Error=%s"
                              " dtype=%s, cg=%s, native=%s gpu=%s"
                              % (e, dtype, use_cg, use_native, use_gpu))

            reconstructed = rows.dot(cols.T)
            for i in range(counts.shape[0]):
                for j in range(counts.shape[1]):
                    self.assertAlmostEqual(counts[i, j], reconstructed[i, j],
                                           delta=0.0001,
                                           msg="failed to reconstruct row=%s, col=%s,"
                                               " value=%.5f, dtype=%s, cg=%s, native=%s gpu=%s"
                                               % (i, j, reconstructed[i, j], dtype, use_cg,
                                                  use_native, use_gpu))

    def test_explain(self):
        counts = csr_matrix([[1, 1, 0, 1, 0, 0],
                             [0, 1, 1, 1, 0, 0],
                             [1, 4, 1, 0, 7, 0],
                             [1, 1, 0, 0, 0, 0],
                             [9, 0, 4, 1, 0, 1],
                             [0, 1, 0, 0, 0, 1],
                             [0, 0, 2, 0, 1, 1]], dtype=np.float64)
        user_items = counts * 2
        item_users = user_items.T

        model = AlternatingLeastSquares(factors=4,
                                        regularization=20,
                                        use_native=False,
                                        use_cg=False,
                                        iterations=100)
        np.random.seed(23)
        model.fit(user_items, show_progress=False)

        userid = 0

        # Assert recommendation is the the same if we recompute user vectors
        recs = model.recommend(userid, item_users, N=10)
        recalculated_recs = model.recommend(userid, item_users, N=10, recalculate_user=True)
        for (item1, score1), (item2, score2) in zip(recs, recalculated_recs):
            self.assertEqual(item1, item2)
            self.assertAlmostEqual(score1, score2, 4)

        # Assert explanation makes sense
        top_rec, score = recalculated_recs[0]
        score_explained, contributions, W = model.explain(userid, item_users, itemid=top_rec)
        scores = [s for _, s in contributions]
        items = [i for i, _ in contributions]
        self.assertAlmostEqual(score, score_explained, 4)
        self.assertAlmostEqual(score, sum(scores), 4)
        self.assertEqual(scores, sorted(scores, reverse=True), "Scores not in order")
        self.assertEqual([0, 2, 3, 4], sorted(items), "Items not seen by user")

        # Assert explanation with precomputed user weights is correct
        top_score_explained, top_contributions, W = model.explain(
            userid, item_users, itemid=top_rec, user_weights=W, N=2)
        top_scores = [s for _, s in top_contributions]
        top_items = [i for i, _ in top_contributions]
        self.assertEqual(2, len(top_contributions))
        self.assertAlmostEqual(score, top_score_explained, 4)
        self.assertEqual(scores[:2], top_scores)
        self.assertEqual(items[:2], top_items)

    def test_recommend_all(self):
        item_users = self.get_checker_board(50)
        user_items = item_users.T.tocsr()

        model = self._get_model()
        model.fit(item_users, show_progress=False)

        recs = model.recommend_all(user_items, N=1, show_progress=False)
        for userid in range(50):
            self.assertEqual(len(recs[userid]), 1)

            # the top item recommended should be the same as the userid:
            # its the one withheld item for the user that is liked by
            # all the other similar users
            self.assertEqual(recs[userid][0], userid)

        # try asking for more items than possible
        self.assertRaises(ValueError, model.recommend_all, user_items, N=10000, show_progress=False)
        self.assertRaises(
            ValueError, model.recommend_all, user_items, filter_items=list(range(10000)),
            show_progress=False)

        # filter recommended items using an additional filter list
        recs = model.recommend_all(user_items, N=1, filter_items=[0], show_progress=False)
        self.assertTrue(0 not in recs)


if __name__ == "__main__":
    unittest.main()

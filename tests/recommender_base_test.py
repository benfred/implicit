""" Common test functions for all recommendation models """

from __future__ import print_function

import numpy as np
from scipy.sparse import csr_matrix


class TestRecommenderBaseMixin(object):
    """ Mixin to test a bunch of common functionality in models
    deriving from RecommenderBase """

    def _get_model(self):
        raise NotImplementedError()

    def test_recommend(self):
        item_users = self.get_checker_board(50)
        user_items = item_users.T.tocsr()

        model = self._get_model()
        model.fit(item_users)

        for userid in range(50):
            recs = model.recommend(userid, user_items, N=1)
            self.assertEqual(len(recs), 1)

            # the top item recommended should be the same as the userid:
            # its the one withheld item for the user that is liked by
            # all the other similar users
            self.assertEqual(recs[0][0], userid)

    def test_similar_items(self):
        model = self._get_model()
        model.fit(self.get_checker_board(50))
        for itemid in range(50):
            recs = model.similar_items(itemid, N=10)
            for r, _ in recs:
                self.assertEqual(r % 2, itemid % 2)

    def get_checker_board(self, X):
        """ Returns a 'checkerboard' matrix: where every even userid has liked
        every even itemid and every odd userid has liked every odd itemid.
        The diagonal is withheld for testing recommend methods """
        ret = np.zeros((X, X))
        for i in range(X):
            for j in range(i % 2, X, 2):
                ret[i, j] = 1.0
        return csr_matrix(ret - np.eye(X))

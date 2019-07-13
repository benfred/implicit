""" Common test functions for all recommendation models """

from __future__ import print_function

import pickle

import numpy as np
from scipy.sparse import csr_matrix

from implicit.evaluation import precision_at_k
from implicit.nearest_neighbours import ItemItemRecommender


class TestRecommenderBaseMixin(object):
    """ Mixin to test a bunch of common functionality in models
    deriving from RecommenderBase """

    def _get_model(self):
        raise NotImplementedError()

    def test_recommend(self):
        item_users = self.get_checker_board(50)
        user_items = item_users.T.tocsr()

        model = self._get_model()
        model.fit(item_users, show_progress=False)

        for userid in range(50):
            recs = model.recommend(userid, user_items, N=1)
            self.assertEqual(len(recs), 1)

            # the top item recommended should be the same as the userid:
            # its the one withheld item for the user that is liked by
            # all the other similar users
            self.assertEqual(recs[0][0], userid)

        # try asking for more items than possible,
        # should return only the available items
        # https://github.com/benfred/implicit/issues/22
        recs = model.recommend(0, user_items, N=10000)
        self.assertTrue(len(recs))

        # filter recommended items using an additional filter list
        # https://github.com/benfred/implicit/issues/26
        recs = model.recommend(0, user_items, N=1, filter_items=[0])
        self.assertTrue(0 not in dict(recs))

    def test_recalculate_user(self):
        item_users = self.get_checker_board(50)
        user_items = item_users.T.tocsr()

        model = self._get_model()
        model.fit(item_users, show_progress=False)

        for userid in range(item_users.shape[1]):
            recs = model.recommend(userid, user_items, N=1)
            self.assertEqual(len(recs), 1)
            user_vector = user_items[userid]

            # we should get the same item if we recalculate_user
            try:
                recs_from_liked = model.recommend(userid=0, user_items=user_vector,
                                                  N=1, recalculate_user=True)
                self.assertEqual(recs[0][0], recs_from_liked[0][0])

                # TODO: if we set regularization for the model to be sufficiently high, the
                # scores from recalculate_user are slightly different. Investigate
                # (could be difference between CG and cholesky optimizers?)
                self.assertAlmostEqual(recs[0][1], recs_from_liked[0][1], places=4)
            except NotImplementedError:
                # some models don't support recalculating user on the fly, and thats ok
                pass

    def test_evaluation(self):
        item_users = self.get_checker_board(50)
        user_items = item_users.T.tocsr()

        model = self._get_model()
        model.fit(item_users, show_progress=False)

        # we've withheld the diagnoal for testing, and have verified that in test_recommend
        # it is returned for each user. So p@1 should be 1.0
        p = precision_at_k(model, user_items.tocsr(), csr_matrix(np.eye(50)), K=1,
                           show_progress=False)
        self.assertEqual(p, 1)

    def test_similar_users(self):

        model = self._get_model()
        # calculating similar users in nearest-neighbours is not implemented yet
        if isinstance(model, ItemItemRecommender):
            return
        model.fit(self.get_checker_board(50), show_progress=False)
        for userid in range(50):
            recs = model.similar_users(userid, N=10)
            for r, _ in recs:
                self.assertEqual(r % 2, userid % 2)

    def test_similar_items(self):
        model = self._get_model()
        model.fit(self.get_checker_board(50), show_progress=False)
        for itemid in range(50):
            recs = model.similar_items(itemid, N=10)
            for r, _ in recs:
                self.assertEqual(r % 2, itemid % 2)

    def test_zero_length_row(self):
        # get a matrix where a row/column is 0
        item_users = self.get_checker_board(50).todense()
        item_users[42] = 0
        item_users[:, 42] = 0

        # also set the last row/column to 0 (test out problem reported here
        # https://github.com/benfred/implicit/issues/86#issuecomment-373385686)
        item_users[49] = 0
        item_users[:, 49] = 0

        model = self._get_model()
        model.fit(csr_matrix(item_users), show_progress=False)

        # item 42 has no users, shouldn't be similar to anything
        for itemid in range(40):
            recs = model.similar_items(itemid, 10)
            self.assertTrue(42 not in [r for r, _ in recs])

    def test_dtype(self):
        # models should be able to accept input of either float32 or float64
        item_users = self.get_checker_board(50)
        model = self._get_model()
        model.fit(item_users.astype(np.float64), show_progress=False)

        model = self._get_model()
        model.fit(item_users.astype(np.float32), show_progress=False)

    def test_rank_items(self):
        item_users = self.get_checker_board(50)
        user_items = item_users.T.tocsr()

        model = self._get_model()
        model.fit(item_users, show_progress=False)

        for userid in range(50):
            selected_items = np.random.randint(50, size=10).tolist()
            ranked_list = model.rank_items(userid, user_items, selected_items)
            ordered_items = [itemid for (itemid, score) in ranked_list]

            # ranked list should have same items
            self.assertEqual(set(ordered_items), set(selected_items))

            wrong_neg_items = [-1, -3, -5]
            wrong_pos_items = [51, 300, 200]

            # rank_items should raise IndexError if selected items contains wrong itemids

            with self.assertRaises(IndexError):
                wrong_item_list = selected_items + wrong_neg_items
                model.rank_items(userid, user_items, wrong_item_list)
            with self.assertRaises(IndexError):
                wrong_item_list = selected_items + wrong_pos_items
                model.rank_items(userid, user_items, wrong_item_list)

    def test_pickle(self):
        item_users = self.get_checker_board(50)
        model = self._get_model()
        model.fit(item_users, show_progress=False)

        pickled = pickle.dumps(model)
        pickle.loads(pickled)

    def get_checker_board(self, X):
        """ Returns a 'checkerboard' matrix: where every even userid has liked
        every even itemid and every odd userid has liked every odd itemid.
        The diagonal is withheld for testing recommend methods """
        ret = np.zeros((X, X))
        for i in range(X):
            for j in range(i % 2, X, 2):
                ret[i, j] = 1.0
        return csr_matrix(ret - np.eye(X))

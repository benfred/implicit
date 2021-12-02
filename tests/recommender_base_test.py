""" Common test functions for all recommendation models """

from __future__ import print_function

import pickle
import random

import numpy as np
from scipy.sparse import csr_matrix

from implicit.evaluation import precision_at_k
from implicit.nearest_neighbours import ItemItemRecommender


def get_checker_board(X):
    """Returns a 'checkerboard' matrix: where every even userid has liked
    every even itemid and every odd userid has liked every odd itemid.
    The diagonal is withheld for testing recommend methods"""
    ret = np.zeros((X, X))
    for i in range(X):
        for j in range(i % 2, X, 2):
            ret[i, j] = 1.0
    return csr_matrix(ret - np.eye(X))


class RecommenderBaseTestMixin:
    """Mixin to test a bunch of common functionality in models
    deriving from RecommenderBase"""

    def _get_model(self):
        raise NotImplementedError()

    def test_recommend(self):
        item_users = get_checker_board(50)
        user_items = item_users.T.tocsr()

        model = self._get_model()
        model.fit(item_users, show_progress=False)

        for userid in range(50):
            ids, _ = model.recommend(userid, user_items, N=1)
            self.assertEqual(len(ids), 1)

            # the top item recommended should be the same as the userid:
            # its the one withheld item for the user that is liked by
            # all the other similar users
            self.assertEqual(ids[0], userid)

        # try asking for more items than possible,
        # should return only the available items
        # https://github.com/benfred/implicit/issues/22
        ids, _ = model.recommend(0, user_items, N=10000)
        self.assertTrue(len(ids))

        # filter recommended items using an additional filter list
        # https://github.com/benfred/implicit/issues/26
        ids, _ = model.recommend(0, user_items, N=1, filter_items=[0])
        self.assertTrue(0 not in set(ids))

    def test_recommend_batch(self):
        user_items = get_checker_board(50)

        model = self._get_model()
        model.fit(user_items, show_progress=False)

        ids, _ = model.recommend(np.arange(50), user_items, N=1)
        for userid in range(50):
            assert len(ids[userid]) == 1

            # the top item recommended should be the same as the userid:
            # its the one withheld item for the user that is liked by
            # all the other similar users
            assert ids[userid][0] == userid

        userids = np.array([2, 3, 4])
        ids, _ = model.recommend(userids, user_items, N=1)

        for i, userid in enumerate(userids):
            assert ids[i][0] == userid

        # filter recommended items using an additional filter list
        ids, _ = model.recommend(userids, user_items, N=1, filter_items=[0])
        for i, _ in enumerate(userids):
            assert 0 not in ids[i]

    def test_recalculate_user(self):
        item_users = get_checker_board(50)
        user_items = item_users.T.tocsr()

        model = self._get_model()
        model.fit(item_users, show_progress=False)

        try:
            batch_ids, batch_scores = model.recommend(
                np.arange(50), user_items, N=1, recalculate_user=True
            )
        except NotImplementedError:
            # some models don't support recalculating user on the fly, and that's ok
            return

        for userid in range(item_users.shape[1]):
            ids, scores = model.recommend(userid, user_items, N=1)
            self.assertEqual(len(ids), 1)
            user_vector = user_items[userid]

            # we should get the same item if we recalculate_user
            ids_from_liked, scores_from_liked = model.recommend(
                userid=0, user_items=user_vector, N=1, recalculate_user=True
            )
            self.assertEqual(ids[0], ids_from_liked[0])

            # TODO: if we set regularization for the model to be sufficiently high, the
            # scores from recalculate_user are slightly different. Investigate
            # (could be difference between CG and cholesky optimizers?)
            self.assertAlmostEqual(scores[0], scores_from_liked[0], places=4)

            # we should also get the same from the batch recommend call already done
            self.assertEqual(batch_ids[userid][0], ids_from_liked[0])
            self.assertAlmostEqual(batch_scores[userid][0], scores_from_liked[0], places=4)

    def test_evaluation(self):
        item_users = get_checker_board(50)
        user_items = item_users.T.tocsr()

        model = self._get_model()
        model.fit(item_users, show_progress=False)

        # we've withheld the diagonal for testing, and have verified that in test_recommend
        # it is returned for each user. So p@1 should be 1.0
        p = precision_at_k(
            model, user_items.tocsr(), csr_matrix(np.eye(50)), K=1, show_progress=False
        )
        self.assertEqual(p, 1)

    def test_similar_users(self):
        model = self._get_model()
        # calculating similar users in nearest-neighbours is not implemented yet
        if isinstance(model, ItemItemRecommender):
            return
        model.fit(get_checker_board(50), show_progress=False)
        for userid in range(50):
            ids, _ = model.similar_users(userid, N=10)
            for r in ids:
                self.assertEqual(r % 2, userid % 2)

    def test_similar_users_batch(self):
        model = self._get_model()
        # calculating similar users in nearest-neighbours is not implemented yet
        if isinstance(model, ItemItemRecommender):
            return
        model.fit(get_checker_board(256), show_progress=False)
        userids = np.arange(50)
        ids, scores = model.similar_users(userids, N=10)

        self.assertEqual(ids.shape, (50, 10))

        for userid in userids:
            # first user returned should be itself, and score should be ~1.0
            self.assertEqual(ids[userid][0], userid)
            self.assertAlmostEqual(scores[userid][0], 1.0, places=4)

            # the rest of the users should be even or odd depending on the userid
            for r in ids[userid]:
                self.assertEqual(r % 2, userid % 2)

    def test_similar_users_filter(self):
        model = self._get_model()
        # calculating similar users in nearest-neighbours is not implemented yet
        if isinstance(model, ItemItemRecommender):
            return

        model.fit(get_checker_board(256), show_progress=False)
        userids = np.arange(50)

        ids, _ = model.similar_users(userids, N=10, filter_users=np.arange(52) * 5)
        for userid in userids:
            for r in ids[userid]:
                self.assertTrue(r % 5 != 0)

        selected = np.arange(10)
        ids, _ = model.similar_users(userids, N=10, users=selected)
        for userid in userids:
            self.assertEqual(set(ids[userid]), set(selected))

    def test_similar_items(self):
        model = self._get_model()
        model.fit(get_checker_board(256), show_progress=False)
        for itemid in range(50):
            ids, _ = model.similar_items(itemid, N=10)
            for r in ids:
                self.assertEqual(r % 2, itemid % 2)

    def test_similar_items_batch(self):
        model = self._get_model()
        user_items = get_checker_board(256)
        model.fit(user_items, show_progress=False)
        itemids = np.arange(50)

        def check_results(ids):
            self.assertEqual(ids.shape, (50, 10))
            for itemid in itemids:
                # first item returned should be itself
                self.assertEqual(ids[itemid][0], itemid)

                # the rest of the items should be even or odd depending on the itemid
                for r in ids[itemid]:
                    self.assertEqual(r % 2, itemid % 2)

        ids, _ = model.similar_items(itemids, N=10)
        check_results(ids)
        try:
            ids, _ = model.similar_items(
                itemids, N=10, recalculate_item=True, react_users=user_items.T.tocsr()
            )
            check_results(ids)
        except NotImplementedError:
            # some models don't support recalculating user on the fly, and that's ok
            pass

    def test_similar_items_filter(self):
        model = self._get_model()

        model.fit(get_checker_board(256), show_progress=False)
        itemids = np.arange(50)

        ids, _ = model.similar_items(itemids, N=10, filter_items=np.arange(52) * 5)
        for itemid in itemids:
            for r in ids[itemid]:
                self.assertTrue(r % 5 != 0)

        selected = np.arange(10)
        ids, _ = model.similar_items(itemids, N=10, items=selected)
        for itemid in itemids:
            self.assertEqual(set(ids[itemid]), set(selected))

    def test_zero_length_row(self):
        # get a matrix where a row/column is 0
        item_users = get_checker_board(50).todense()
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
            ids, _ = model.similar_items(itemid, 10)
            self.assertTrue(42 not in ids)

    def test_dtype(self):
        # models should be able to accept input of either float32 or float64
        item_users = get_checker_board(50)
        model = self._get_model()
        model.fit(item_users.astype(np.float64), show_progress=False)

        model = self._get_model()
        model.fit(item_users.astype(np.float32), show_progress=False)

    def test_rank_items(self):
        item_users = get_checker_board(50)
        user_items = item_users.T.tocsr()

        model = self._get_model()
        model.fit(item_users, show_progress=False)

        for userid in range(50):
            selected_items = random.sample(range(50), 10)

            ids, _ = model.recommend(
                userid, user_items, items=selected_items, filter_already_liked_items=False
            )

            # ranked list should have same items
            self.assertEqual(set(ids), set(selected_items))

            wrong_neg_items = [-1, -3, -5]
            wrong_pos_items = [51, 300, 200]

            # rank_items should raise IndexError if selected items contains wrong itemids
            with self.assertRaises(IndexError):
                wrong_item_list = selected_items + wrong_neg_items
                model.recommend(userid, user_items, items=wrong_item_list)
            with self.assertRaises(IndexError):
                wrong_item_list = selected_items + wrong_pos_items
                model.recommend(userid, user_items, items=wrong_item_list)

    def test_rank_items_batch(self):
        item_users = get_checker_board(50)
        user_items = item_users.T.tocsr()

        model = self._get_model()
        model.fit(item_users, show_progress=False)

        selected_items = np.arange(10) * 3
        ids, _ = model.recommend(np.arange(50), user_items, items=selected_items)

        for userid in range(50):
            current_ids = ids[userid]
            self.assertEqual(set(current_ids), set(selected_items))

    def test_pickle(self):
        item_users = get_checker_board(50)
        model = self._get_model()
        model.fit(item_users, show_progress=False)

        pickled = pickle.dumps(model)
        pickle.loads(pickled)

    def get_checker_board(self, X):
        """Returns a 'checkerboard' matrix: where every even userid has liked
        every even itemid and every odd userid has liked every odd itemid.
        The diagonal is withheld for testing recommend methods"""
        ret = np.zeros((X, X))
        for i in range(X):
            for j in range(i % 2, X, 2):
                ret[i, j] = 1.0
        return csr_matrix(ret - np.eye(X))

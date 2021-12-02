import time

import numpy as np

import implicit.gpu

from ..cpu.matrix_factorization_base import _filter_items_from_sparse_matrix
from ..recommender_base import RecommenderBase


class MatrixFactorizationBase(RecommenderBase):
    """Base class for MF models running on the GPU.

    This adds support for inference to run on the GPU as well as training.

    Attributes
    ----------
    item_factors : implicit.gpu.Matrix
        Array of latent factors for each item in the training set
    user_factors : implicit.gpu.Matrix
        Array of latent factors for each user in the training set
    """

    def __init__(self):
        self.item_factors = None
        self.user_factors = None
        self._item_norms = None
        self._user_norms = None
        self._user_norms_host = None
        self._item_norms_host = None
        self._knn = implicit.gpu.KnnQuery()

    def recommend(
        self,
        userid,
        user_items,
        N=10,
        filter_already_liked_items=True,
        filter_items=None,
        recalculate_user=False,
        items=None,
    ):
        if recalculate_user:
            raise NotImplementedError("recalculate_user isn't support on GPU yet")

        item_factors = self.item_factors
        if items is not None:
            if filter_items:
                raise ValueError("Can't set both items and filter_items in recommend call")

            items = np.array(items)
            items.sort()
            item_factors = item_factors[items]

            # check selected items are in the model
            if items.max() >= self.item_factors.shape[0] or items.min() < 0:
                raise IndexError("Some itemids are not in the model")

        if filter_items is not None:
            filter_items = implicit.gpu.IntVector(np.array(filter_items, dtype="int32"))

        query_filter = None
        if filter_already_liked_items:
            query_filter = user_items[userid]

            # if we've been given a list of explicit itemids to rank, we need to filter down
            if items is not None:
                query_filter = _filter_items_from_sparse_matrix(items, query_filter)

            if query_filter.nnz:
                query_filter = implicit.gpu.COOMatrix(query_filter.tocoo())
            else:
                query_filter = None

        # calculate the top N items, removing the users own liked items from the results
        ids, scores = self._knn.topk(
            item_factors,
            self.user_factors[userid],
            N,
            query_filter=query_filter,
            item_filter=filter_items,
        )

        if np.isscalar(userid):
            ids, scores = ids[0], scores[0]

        if items is not None:
            ids = items[ids]

        return ids, scores

    recommend.__doc__ = RecommenderBase.recommend.__doc__

    @property
    def user_norms(self):
        if self._user_norms is None:
            self._user_norms = implicit.gpu.calculate_norms(self.user_factors)
            self._user_norms_host = self._user_norms.to_numpy().reshape(self._user_norms.shape[1])
        return self._user_norms

    @property
    def item_norms(self):
        if self._item_norms is None:
            self._item_norms = implicit.gpu.calculate_norms(self.item_factors)
            self._item_norms_host = self._item_norms.to_numpy().reshape(self._item_norms.shape[1])
        return self._item_norms

    def similar_users(self, userid, N=10, filter_users=None, users=None):
        norms = self.user_norms
        user_factors = self.user_factors
        if users is not None:
            if filter_users:
                raise ValueError("Can't set both users and filter_users in similar_users call")

            users = np.array(users)
            user_factors = user_factors[users]

            # TODO: we should be able to do this all on the GPU
            norms = implicit.gpu.Matrix(self._user_norms_host[users].reshape(1, len(users)))

            # check selected items are in the model
            if users.max() >= self.user_factors.shape[0] or users.min() < 0:
                raise IndexError("Some userids in the users parameter are not in the model")

        if filter_users is not None:
            filter_users = implicit.gpu.IntVector(np.array(filter_users, dtype="int32"))

        ids, scores = self._knn.topk(
            user_factors, self.user_factors[userid], N, norms, item_filter=filter_users
        )

        if users is not None:
            ids = users[ids]

        user_norms = self._user_norms_host[userid]
        if np.isscalar(userid):
            ids, scores = ids[0], scores[0]
            scores /= user_norms
        else:
            scores /= user_norms[:, None]
        return ids, scores

    similar_users.__doc__ = RecommenderBase.similar_users.__doc__

    def similar_items(
        self, itemid, N=10, react_users=None, recalculate_item=False, filter_items=None, items=None
    ):
        if recalculate_item:
            raise NotImplementedError("recalculate_item isn't support on GPU yet")

        item_factors = self.item_factors
        norms = self.item_norms
        if items is not None:
            if filter_items:
                raise ValueError("Can't set both items and filter_items in similar_items call")

            items = np.array(items)

            # TODO: we should be able to do this all on the GPU
            norms = implicit.gpu.Matrix(self._item_norms_host[items].reshape(1, len(items)))
            item_factors = item_factors[items]

            # check selected items are in the model
            if items.max() >= self.item_factors.shape[0] or items.min() < 0:
                raise IndexError("Some itemids are not in the model")

        if filter_items is not None:
            filter_items = implicit.gpu.IntVector(np.array(filter_items, dtype="int32"))

        ids, scores = self._knn.topk(
            item_factors, self.item_factors[itemid], N, norms, item_filter=filter_items
        )

        if items is not None:
            ids = items[ids]

        item_norms = self._item_norms_host[itemid]
        if np.isscalar(itemid):
            ids, scores = ids[0], scores[0]
            scores /= item_norms
        else:
            scores /= item_norms[:, None]
        return ids, scores

    similar_items.__doc__ = RecommenderBase.similar_items.__doc__

    def __getstate__(self):
        return {
            "item_factors": self.item_factors.to_numpy() if self.item_factors else None,
            "user_factors": self.user_factors.to_numpy() if self.user_factors else None,
        }

    def __setstate__(self, state):
        self.item_factors = implicit.gpu.Matrix(state["item_factors"])
        self.user_factors = implicit.gpu.Matrix(state["user_factors"])


def check_random_state(random_state):
    """Validate the random state, r

    Check a random seed or existing numpy RandomState
    and get back an initialized RandomState.

    Parameters
    ----------
    random_state : int, None or RandomState
        The existing RandomState. If None, or an int, will be used
        to seed a new curand RandomState generator
    """
    if isinstance(random_state, np.random.RandomState):
        # we need to convert from numpy random state our internal random state
        return implicit.gpu.RandomState(random_state.randint(2 ** 31))

    # otherwise try to initialize a new one, and let it fail through
    # on the numpy side if it doesn't work
    return implicit.gpu.RandomState(random_state or int(time.time()))

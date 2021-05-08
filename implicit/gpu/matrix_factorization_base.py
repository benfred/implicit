import itertools
import time

import numpy as np

import implicit.gpu

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
        self._knn = implicit.gpu.KnnQuery()

    def recommend(
        self,
        userid,
        user_items,
        N=10,
        filter_already_liked_items=True,
        filter_items=None,
        recalculate_user=False,
    ):
        if recalculate_user:
            raise NotImplementedError("recalculate_user isn't support on GPU yet")
        liked = set()
        if filter_already_liked_items:
            liked.update(user_items[userid].indices)
        if filter_items:
            liked.update(filter_items)
        count = N + len(liked)

        # calculate the top N items, removing the users own liked items from the results
        # TODO: own like filtering (direct in topk class
        ids, scores = self._knn.topk(self.item_factors, self.user_factors[userid], count)
        return list(
            itertools.islice((rec for rec in zip(ids[0], scores[0]) if rec[0] not in liked), N)
        )

    recommend.__doc__ = RecommenderBase.recommend.__doc__

    def rank_items(self, userid, user_items, selected_items, recalculate_user=False):
        if recalculate_user:
            raise NotImplementedError("recalculate_user isn't support on GPU yet")

        # check selected items are  in the model
        if max(selected_items) >= self.item_factors.shape[0] or min(selected_items) < 0:
            raise IndexError("Some of selected itemids are not in the model")

        item_factors = self.item_factors[selected_items]
        user = self.user_factors[userid]

        # once we have item_factors here, this should work
        ids, scores = self._knn.topk(item_factors, user, len(selected_items))
        ids = np.array(selected_items)[ids]

        print(ids)
        print(scores)
        return list(zip(ids[0], scores[0]))

    rank_items.__doc__ = RecommenderBase.rank_items.__doc__

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

    def similar_users(self, userid, N=10):
        ids, scores = self._knn.topk(
            self.user_factors, self.user_factors[int(userid)], N, self.user_norms
        )
        scores /= self._user_norms_host[userid]
        return list(zip(ids[0], scores[0]))

    similar_users.__doc__ = RecommenderBase.similar_users.__doc__

    def similar_items(self, itemid, N=10, react_users=None, recalculate_item=False):
        if recalculate_item:
            raise NotImplementedError("recalculate_item isn't support on GPU yet")
        ids, scores = self._knn.topk(
            self.item_factors, self.item_factors[int(itemid)], N, self.item_norms
        )
        scores /= self._item_norms_host[itemid]
        return list(zip(ids[0], scores[0]))

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

try:
    import cupy as cp
except ImportError:
    pass

import numpy as np
import itertools

from ..recommender_base import RecommenderBase


class MatrixFactorizationBase(RecommenderBase):
    """ Base class for MF models running on the GPU.

    This adds support for inference to run on the GPU as well as training.
    Factors are stored as cupy arrays.

    Attributes
    ----------
    item_factors : cupy.array
        Array of latent factors for each item in the training set
    user_factors : cupy.array
        Array of latent factors for each user in the training set
    """

    def __init__(self):
        self.item_factors = None
        self.user_factors = None
        self._item_norms = None
        self._user_norms = None

    def recommend(self, userid, user_items,
                  N=10, filter_already_liked_items=True, filter_items=None, recalculate_user=False):
        if recalculate_user:
            raise NotImplementedError("recalculate_user isn't support on GPU yet")

        user = self.user_factors[userid]

        liked = set()
        if filter_already_liked_items:
            liked.update(user_items[userid].indices)
        if filter_items:
            liked.update(filter_items)

        # calculate the top N items, removing the users own liked items from the results
        scores = self.item_factors.dot(user)

        count = N + len(liked)
        if count < len(scores):
            ids = cp.argpartition(scores, -count)[-count:]
            best = sorted(zip(ids.tolist(), scores[ids].tolist()), key=lambda x: -x[1])
        else:
            best = sorted(enumerate(scores.tolist()), key=lambda x: -x[1])

        return list(itertools.islice((rec for rec in best if rec[0] not in liked), N))

    recommend.__doc__ = RecommenderBase.recommend.__doc__

    def rank_items(self, userid, user_items, selected_items, recalculate_user=False):
        if recalculate_user:
            raise NotImplementedError("recalculate_user isn't support on GPU yet")

        user = self.user_factors[userid]

        # check selected items are  in the model
        if max(selected_items) >= self.item_factors.shape[0] or min(selected_items) < 0:
            raise IndexError("Some of selected itemids are not in the model")

        item_factors = self.item_factors[selected_items]
        # calculate relevance scores of given items w.r.t the user
        scores = item_factors.dot(user)

        # return sorted results
        return sorted(zip(selected_items, scores), key=lambda x: -x[1])

    rank_items.__doc__ = RecommenderBase.rank_items.__doc__

    @property
    def user_norms(self):
        if self._user_norms is None:
            self._user_norms = cp.linalg.norm(self.user_factors, axis=-1)
            # don't divide by zero in similar_items, replace with small value
            self._user_norms[self._user_norms == 0] = 1e-10
        return self._user_norms

    @property
    def item_norms(self):
        if self._item_norms is None:
            self._item_norms = cp.linalg.norm(self.item_factors, axis=-1)
            # don't divide by zero in similar_items, replace with small value
            self._item_norms[self._item_norms == 0] = 1e-10
        return self._item_norms

    def similar_users(self, userid, N=10):
        factor = self.user_factors[userid]
        norm = self.user_norms[userid]
        scores = self.user_factors.dot(factor) / (norm * self.user_norms)
        best = cp.argpartition(scores, -N)[-N:]
        return sorted(zip(best.tolist(), scores[best].tolist()), key=lambda x: -x[1])

    similar_users.__doc__ = RecommenderBase.similar_users.__doc__

    def similar_items(self, itemid, N=10, react_users=None, recalculate_item=False):
        if recalculate_item:
            raise NotImplementedError("recalculate_item isn't support on GPU yet")
        factor = self.item_factors[itemid]
        norm = self.item_norms[itemid]
        scores = self.item_factors.dot(factor) / (norm * self.item_norms)
        best = cp.argpartition(scores, -N)[-N:]
        return sorted(zip(best.tolist(), scores[best].tolist()), key=lambda x: -x[1])

    similar_items.__doc__ = RecommenderBase.similar_items.__doc__


def check_random_state(random_state):
    """Validate the random state, r

    Check a random seed or existing numpy RandomState
    and get back an initialized RandomState.

    Parameters
    ----------
    random_state : int, None or RandomState
        The existing RandomState. If None, or an int, will be used
        to seed a new cupy RandomState.
    """
    if isinstance(random_state, cp.random.RandomState):
        return random_state

    if isinstance(random_state, np.random.RandomState):
        # we need to convert from numpy random state to cupy random state.
        return cp.random.RandomState(random_state.randint(2**63))

    # otherwise try to initialize a new one, and let it fail through
    # on the numpy side if it doesn't work
    return cp.random.RandomState(random_state)

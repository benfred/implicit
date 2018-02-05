""" Base class for recommendation algorithms in this package """

import itertools
from abc import ABCMeta, abstractmethod

import numpy as np


class RecommenderBase(object):
    """ Defines the interface that all recommendations models here expose """
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, item_users):
        """
        Trains the model on a sparse matrix of item/user/weight

        Parameters
        ----------
        item_user : csr_matrix
            A matrix of shape (number_of_items, number_of_users). The nonzero
            entries in this matrix are the items that are liked by each user.
            The values are how confidant you are that the item is liked by the user.
        """
        pass

    @abstractmethod
    def recommend(self, userid, user_items, N=10, filter_items=None, recalculate_user=False):
        """
        Recommends items for a user

        Calculates the N best recommendations for a user, and returns a list of itemids, score.

        Parameters
        ----------
        userid : int
            The userid to calculate recommendations for
        user_items : csr_matrix
            A sparse matrix of shape (number_users, number_items). This lets us look
            up the liked items and their weights for the user. This is used to filter out
            items that have already been liked from the output, and to also potentially
            calculate the best items for this user.
        N : int, optional
            The number of results to return
        filter_items : sequence of ints, optional
            List of extra item ids to filter out from the output
        recalculate_user : bool, optional
            When true, don't rely on stored user state and instead recalculate from the
            passed in user_items

        Returns
        -------
        list
            List of (itemid, score) tuples
        """
        pass

    @abstractmethod
    def similar_items(self, itemid, N=10):
        """
        Calculates a list of similar items

        Parameters
        ----------
        itemid : int
            The row id of the item to retrieve similar items for
        N : int, optional
            The number of similar items to return

        Returns
        -------
        list
            List of (itemid, score) tuples
        """
        pass


class MatrixFactorizationBase(RecommenderBase):
    """ MatrixFactorizationBase contains common functionality for recommendation models.

    Attributes
    ----------
    item_factors : ndarray
        Array of latent factors for each item in the training set
    user_factors : ndarray
        Array of latent factors for each user in the training set
     """
    def __init__(self):
        # learned parameters
        self.item_factors = None
        self.user_factors = None

        # cache of item norms (useful for calculating similar items)
        self._item_norms = None

    def recommend(self, userid, user_items, N=10, filter_items=None, recalculate_user=False):
        user = self._user_factor(userid, user_items, recalculate_user)

        # calculate the top N items, removing the users own liked items from the results
        liked = set(user_items[userid].indices)
        scores = self.item_factors.dot(user)
        if filter_items:
            liked.update(filter_items)

        count = N + len(liked)
        if count < len(scores):
            ids = np.argpartition(scores, -count)[-count:]
            best = sorted(zip(ids, scores[ids]), key=lambda x: -x[1])
        else:
            best = sorted(enumerate(scores), key=lambda x: -x[1])
        return list(itertools.islice((rec for rec in best if rec[0] not in liked), N))

    recommend.__doc__ = RecommenderBase.recommend.__doc__

    def _user_factor(self, userid, user_items, recalculate_user=False):
        if recalculate_user:
            return self.recalculate_user(userid, user_items)
        else:
            return self.user_factors[userid]

    def recalculate_user(self, userid, user_items):
        raise NotImplementedError("recalculate_user is not supported with this model")

    def similar_items(self, itemid, N=10):
        scores = self.item_factors.dot(self.item_factors[itemid]) / self.item_norms
        best = np.argpartition(scores, -N)[-N:]
        return sorted(zip(best, scores[best] / self.item_norms[itemid]), key=lambda x: -x[1])

    similar_items.__doc__ = RecommenderBase.similar_items.__doc__

    @property
    def item_norms(self):
        if self._item_norms is None:
            self._item_norms = np.linalg.norm(self.item_factors, axis=-1)
            # don't divide by zero in similar_items, replace with small value
            self._item_norms[self._item_norms == 0] = 1e-10
        return self._item_norms

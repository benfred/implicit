""" Base class for recommendation algorithms in this package """

import itertools
from abc import ABCMeta, abstractmethod


class RecommenderBase(object):
    """ Defines the interface that recommendations models here expose """
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, item_users):
        """ Trains the model on a sparse matrix of item/user/weight """
        pass

    @abstractmethod
    def liked(self, userid, user_items):
        """ Returns items liked by a user """
        pass

    @abstractmethod
    def best_recommendations(self, userid, user_items, N, recalculate_user=False):
        """ Returns N best recommendations for a user """
        pass

    def recommend(
        self,
        userid,
        user_items,
        N=10,
        filter_items=None,
        recalculate_user=False,
        filter_liked=True
    ):
        """ Recommends items for a user"""

        filtered = set(filter_items if filter_items is not None else [])

        if (filter_liked):
            liked = self.liked(userid, user_items)
            filtered.update(liked)

        best = self.best_recommendations(
            userid,
            user_items,
            N + len(filtered),
            recalculate_user=recalculate_user
        )

        return list(itertools.islice((rec for rec in best if rec[0] not in filtered), N))

    @abstractmethod
    def similar_items(self, itemid, N=10):
        """ Returns related items for an item """
        pass

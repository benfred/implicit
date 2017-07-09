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
    def recommend(self, userid, user_items, N=10, filter_items=None, recalculate_user=False, filter_liked = True):
        """ Recommends items for a user"""
        pass

    @abstractmethod
    def similar_items(self, itemid, N=10):
        """ Returns related items for an item """
        pass

    def slice_recommendations(self, N, best, liked, filter_liked = True):
        """ Returns N recommendations from best candidates, either filtering or not those already liked """
        return list(itertools.islice((rec for rec in best if not filter_liked or rec[0] not in liked), N))

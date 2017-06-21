""" Base class for recommendation algorithms in this package """

from abc import ABCMeta, abstractmethod


class RecommenderBase(object):
    """ Defines the interface that recommendations models here expose """
    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, item_users):
        """ Trains the model on a sparse matrix of item/user/weight """
        pass

    @abstractmethod
    def recommend(self, userid, user_items, N=10, filter_items=None, recalculate_user=False):
        """ Recommends items for a user"""
        pass

    @abstractmethod
    def similar_items(self, itemid, N=10):
        """ Returns related items for an item """
        pass

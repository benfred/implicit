""" Base class for recommendation algorithms in this package """

from abc import ABCMeta, abstractmethod


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

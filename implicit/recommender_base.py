""" Base class for recommendation algorithms in this package """
import warnings
from abc import ABCMeta, abstractmethod


class ModelFitError(Exception):
    pass


class RecommenderBase:
    """Defines the interface that all recommendations models here expose"""

    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, user_items):
        """
        Trains the model on a sparse matrix of item/user/weight

        Parameters
        ----------
        user_items : csr_matrix
            A matrix of shape (number_of_users, number_of_items). The nonzero
            entries in this matrix are the items that are liked by each user.
            The values are how confident you are that the item is liked by the user.
        """

    @abstractmethod
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
        """
        Recommends items for a user

        Calculates the N best recommendations for a user, and returns a list of itemids, score.

        Parameters
        ----------
        userid : Union[int, array_like]
            The userid or array of userids to calculate recommendations for
        user_items : csr_matrix
            A sparse matrix of shape (number_users, number_items). This lets us look
            up the liked items and their weights for the user. This is used to filter out
            items that have already been liked from the output, and to also potentially
            calculate the best items for this user.
        N : int, optional
            The number of results to return
        filter_already_liked_items: bool, optional
            When true, don't return items present in the training set that were rated
            by the specified user.
        filter_items : sequence of ints, optional
            List of extra item ids to filter out from the output
        recalculate_user : bool, optional
            When true, don't rely on stored user state and instead recalculate from the
            passed in user_items
        items: array_like, optional
            Array of extra item ids. When set this will only rank the items in this array instead
            of ranking every item the model was fit for. This parameter cannot be used with
            filter_items

        Returns
        -------
        tuple
            Tuple of (itemids, scores) arrays. When calculating for a single user these array will
            be 1-dimensional with N items. When passed an array of userids, these will be
            2-dimensional arrays with a row for each user.
        """

    @abstractmethod
    def similar_users(self, userid, N=10, filter_users=None, users=None):
        """
        Calculates the most similar users for a userid or array of userids

        Parameters
        ----------
        userid : Union[int, array_like]
            The userid or an array of userids to retrieve similar users for.
        N : int, optional
            The number of similar users to return
        filter_users: array_like, optional
            An array of user ids to filter out from the results being returned
        users: array_like, optional
            An array of user ids to include in the output. If not set all users in the training
            set will be included. Cannot be used with the filter_users options

        Returns
        -------
        tuple
            Tuple of (userids, scores) arrays
        """

    @abstractmethod
    def similar_items(
        self, itemid, N=10, react_users=None, recalculate_item=False, filter_items=None, items=None
    ):
        """
        Calculates a list of similar items

        Parameters
        ----------
        itemid : Union[int, array_like]
            The item id or an array of item ids to retrieve similar items for
        N : int, optional
            The number of similar items to return
        react_users : csr_matrix, optional
            A sparse matrix of shape (number_items, number_users). This lets us look
            up the reacted users and their weights for the item.
        recalculate_item : bool, optional
            When true, don't rely on stored item state and instead recalculate from the
            passed in react_users
        filter_items: array_like, optional
            An array of item ids to filter out from the results being returned
        items: array_like, optional
            An array of item ids to include in the output. If not set all items in the training
            set will be included. Cannot be used with the filter_items options

        Returns
        -------
        tuple
            Tuple of (itemids, scores) arrays
        """

    def rank_items(self, userid, user_items, selected_items, recalculate_user=False):
        """
        Rank given items for a user and returns sorted item list.

        Deprecated. Use recommend with the 'items' parameter instead
        """
        warnings.warn(
            "rank_items is deprecated. Use recommend with the 'items' parameter instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.recommend(
            userid,
            user_items,
            recalculate_user=recalculate_user,
            items=selected_items,
            filter_already_liked_items=False,
        )

""" Base class for recommendation algorithms in this package """
import warnings
from abc import ABCMeta, abstractmethod

import numpy as np


class ModelFitError(Exception):
    pass


class RecommenderBase(metaclass=ABCMeta):
    """Defines a common interface for all recommendation models"""

    @abstractmethod
    def fit(self, user_items, show_progress=True, callback=None):
        """
        Trains the model on a sparse matrix of user/item/weight

        Parameters
        ----------
        user_items : csr_matrix
            A sparse CSR matrix of shape (number_of_users, number_of_items). The nonzero
            entries in this matrix are the items that are liked by each user.
            The values are how confident you are that the item is liked by the user.
        show_progress : bool, optional
            Whether to show a progress bar during fitting
        callback: Callable, optional
            Callable function on each epoch with such arguments as epoch, elapsed time and progress
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
        Recommends items for users.

        This method allows you to calculate the top N recommendations for a user or
        batch of users. Passing an array of userids instead of a single userid will
        tend to be more efficient, and allows multi-thread processing on the CPU.

        This method has options for filtering out items from the results. You can both
        filter out items that have already been liked by the user with the
        filter_already_liked_items parameter, as well as pass in filter_items to filter
        out other items for all users in the batch. By default all items in the training
        dataset are scored, but by setting the 'items' parameter you can restrict down to
        a subset.

        Example usage::

            # calculate the top recommendations for a single user
            ids, scores = model.recommend(0, user_items[0])

            # calculate the top recommendations for a batch of users
            userids = np.arange(10)
            ids, scores = model.recommend(userids, user_items[userids])

        Parameters
        ----------
        userid : Union[int, array_like]
            The userid or array of userids to calculate recommendations for
        user_items : csr_matrix
            A sparse matrix of shape (users, number_items). This lets us look
            up the liked items and their weights for the user. This is used to filter out
            items that have already been liked from the output, and to also potentially
            recalculate the user representation. Each row in this sparse matrix corresponds
            to a row in the userid parameter: that is the first row in this matrix contains
            the liked items for the first user in the userid array.
        N : int, optional
            The number of results to return
        filter_already_liked_items: bool, optional
            When true, don't return items present in the training set that were rated
            by the specified user.
        filter_items : array_like, optional
            List of extra item ids to filter out from the output
        recalculate_user : bool, optional
            When true, don't rely on stored user embeddings and instead recalculate from the
            passed in user_items. This option isn't supported by all models.
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
        self, itemid, N=10, recalculate_item=False, item_users=None, filter_items=None, items=None
    ):
        """
        Calculates a list of similar items

        Parameters
        ----------
        itemid : Union[int, array_like]
            The item id or an array of item ids to retrieve similar items for
        N : int, optional
            The number of similar items to return
        recalculate_item : bool, optional
            When true, don't rely on stored item state and instead recalculate from the
            passed in item_users
        item_users : csr_matrix, optional
            A sparse matrix of shape (itemid, number_users). This lets us look
            up the users for each item. This is only needs to be set when setting
            recalculate_item=True. This should have the same number of rows as
            the itemid parameter, with the first row of the sparse matrix corresponding
            to the first item in the itemid array.
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

    @abstractmethod
    def save(self, file):
        """Saves the model to a file, using the numpy `.npz` format

        Parameters
        ----------
        file : str or io.IOBase
            Either the filename or an open file-like object to save the model to

        See Also
        --------
        load
        numpy.savez
        """

    @classmethod
    def load(cls, fileobj_or_path) -> "RecommenderBase":
        """Loads the model from a file

        Parameters
        ----------
        fileobj_or_path : str or io.IOBase
            Either the filename or an open file-like object to load the model from

        Returns
        -------
        RecommenderBase
            The model loaded up from disk

        See Also
        --------
        save
        numpy.load
        """
        if isinstance(fileobj_or_path, str) and not fileobj_or_path.endswith(".npz"):
            fileobj_or_path = fileobj_or_path + ".npz"
        with np.load(fileobj_or_path, allow_pickle=False) as data:
            ret = cls()
            for k, v in data.items():
                if k == "dtype":
                    v = np.dtype(str(v))
                elif v.shape == ():
                    v = v.item()
                setattr(ret, k, v)
            return ret

    def rank_items(self, userid, user_items, selected_items, recalculate_user=False):
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

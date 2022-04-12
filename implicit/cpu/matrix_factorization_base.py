""" Base class for recommendation algorithms in this package """
import warnings

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix

from ..recommender_base import ModelFitError, RecommenderBase
from .topk import topk


class MatrixFactorizationBase(RecommenderBase):
    """MatrixFactorizationBase contains common functionality for recommendation models.

    Attributes
    ----------
    item_factors : ndarray
        Array of latent factors for each item in the training set
    user_factors : ndarray
        Array of latent factors for each user in the training set
    num_threads : int
        The number of threads to use for batch recommendation calls and fitting the
        model. Setting to 0 will use all CPU cores on the machine
    """

    def __init__(self, num_threads=0):
        # learned parameters
        self.item_factors = None
        self.user_factors = None

        # cache of user, item norms (useful for calculating similar items)
        self._user_norms, self._item_norms = None, None
        self.num_threads = num_threads

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
        if filter_already_liked_items or recalculate_user:
            if not isinstance(user_items, csr_matrix):
                raise ValueError("user_items needs to be a CSR sparse matrix")
            user_count = 1 if np.isscalar(userid) else len(userid)
            if user_items.shape[0] != user_count:
                raise ValueError("user_items must contain 1 row for every user in userids")

        user = self._user_factor(userid, user_items, recalculate_user)

        item_factors = self.item_factors

        # if we have an item list to restrict down to, we need to filter the item_factors
        # and filter_query_items
        if items is not None:
            N = min(N, len(items))
            if filter_items:
                raise ValueError("Can't set both items and filter_items in recommend call")

            items = np.array(items)
            items.sort()
            item_factors = item_factors[items]

            # check selected items are in the model
            if items.max() >= self.item_factors.shape[0] or items.min() < 0:
                raise IndexError("Some itemids in the items parameter in are not in the model")

        # get a CSR matrix of items to filter per-user
        filter_query_items = None
        if filter_already_liked_items:
            filter_query_items = user_items

            # if we've been given a list of explicit itemids to rank, we need to filter down
            if items is not None:
                filter_query_items = _filter_items_from_sparse_matrix(items, filter_query_items)

        ids, scores = topk(
            item_factors,
            user,
            N,
            filter_query_items=filter_query_items,
            filter_items=filter_items,
            num_threads=self.num_threads,
        )

        if np.isscalar(userid):
            ids, scores = ids[0], scores[0]

        # if we've been given an explicit items list, remap the ids
        if items is not None:
            ids = items[ids]

        return ids, scores

    def recommend_all(
        self,
        user_items,
        N=10,
        recalculate_user=False,
        filter_already_liked_items=True,
        filter_items=None,
        users_items_offset=0,
    ):
        warnings.warn(
            "recommend_all is deprecated. Use recommend with an array of userids instead",
            DeprecationWarning,
        )

        userids = np.arange(user_items.shape[0]) + users_items_offset
        if users_items_offset:
            adjusted = lil_matrix(
                (user_items.shape[0] + users_items_offset, user_items.shape[1]),
                dtype=user_items.dtype,
            )
            adjusted[users_items_offset:] = user_items
            user_items = adjusted.tocsr()

        ids, _ = self.recommend(
            userids,
            user_items,
            N=N,
            filter_already_liked_items=filter_already_liked_items,
            filter_items=filter_items,
            recalculate_user=recalculate_user,
        )
        return ids

    recommend.__doc__ = RecommenderBase.recommend.__doc__

    def _user_factor(self, userid, user_items, recalculate_user=False):
        if recalculate_user:
            return self.recalculate_user(userid, user_items)
        return self.user_factors[userid]

    def _item_factor(self, itemid, item_users, recalculate_item=False):
        if recalculate_item:
            return self.recalculate_item(itemid, item_users)
        return self.item_factors[itemid]

    def recalculate_user(self, userid, user_items):
        raise NotImplementedError("recalculate_user is not supported with this model")

    def recalculate_item(self, itemid, item_users):
        raise NotImplementedError("recalculate_item is not supported with this model")

    def similar_users(self, userid, N=10, filter_users=None, users=None):
        user_factors = self.user_factors
        norms = self.user_norms
        norm = norms[userid]

        # if we have an user list to restrict down to, we need to filter the user_factors
        if users is not None:
            if filter_users:
                raise ValueError("Can't set both users and filter_users in similar_users call")

            users = np.array(users)
            user_factors = user_factors[users]
            norms = norms[users]

            # check selected items are in the model
            if users.max() >= self.user_factors.shape[0] or users.min() < 0:
                raise IndexError("Some userids in the users parameter are not in the model")

        factor = self.user_factors[userid]
        ids, scores = self._get_similarity_score(
            factor, norm, user_factors, norms, N, filter_items=filter_users
        )
        if users is not None:
            ids = users[ids]

        return ids, scores

    similar_users.__doc__ = RecommenderBase.similar_users.__doc__

    def similar_items(
        self, itemid, N=10, recalculate_item=False, item_users=None, filter_items=None, items=None
    ):
        factor = self._item_factor(itemid, item_users, recalculate_item)
        factors = self.item_factors
        norms = self.item_norms

        if recalculate_item:
            if np.isscalar(itemid):
                norm = np.linalg.norm(factor)
                norm = norm if norm != 0 else 1e-10
            else:
                norm = np.linalg.norm(factor, axis=1)
                norm[norm == 0] = 1e-10
        else:
            norm = norms[itemid]

        # if we have an item list to restrict down to, we need to filter the item_factors
        if items is not None:
            if filter_items:
                raise ValueError("Can't set both items and filter_items in similar_items call")

            items = np.array(items)
            factors = factors[items]
            norms = norms[items]

            # check selected items are in the model
            if items.max() >= self.item_factors.shape[0] or items.min() < 0:
                raise IndexError("Some itemids in the items parameter are not in the model")

        ids, scores = self._get_similarity_score(
            factor, norm, factors, norms, N, filter_items=filter_items
        )
        if items is not None:
            ids = items[ids]
        return ids, scores

    similar_items.__doc__ = RecommenderBase.similar_items.__doc__

    def _get_similarity_score(self, factor, norm, factors, norms, N, filter_items=None):
        ids, scores = topk(
            factors,
            factor,
            N,
            item_norms=norms,
            filter_items=filter_items,
            num_threads=self.num_threads,
        )
        if np.isscalar(norm):
            ids, scores = ids[0], scores[0]
            scores /= norm
        else:
            scores /= norm[:, None]
        return ids, scores

    @property
    def user_norms(self):
        if self._user_norms is None:
            self._user_norms = np.linalg.norm(self.user_factors, axis=-1)
            # don't divide by zero in similar_items, replace with small value
            self._user_norms[self._user_norms == 0] = 1e-10
        return self._user_norms

    @property
    def item_norms(self):
        if self._item_norms is None:
            self._item_norms = np.linalg.norm(self.item_factors, axis=-1)
            # don't divide by zero in similar_items, replace with small value
            self._item_norms[self._item_norms == 0] = 1e-10
        return self._item_norms

    def _check_fit_errors(self):
        is_nan = np.any(np.isnan(self.user_factors), axis=None)
        is_nan |= np.any(np.isnan(self.item_factors), axis=None)
        if is_nan:
            raise ModelFitError("NaN encountered in factors")


def _filter_items_from_sparse_matrix(items, query_items):
    """Remaps all the ids in query_items down to match the position
    in the items filter. Requires items to be sorted"""
    filter_query_items = query_items.tocoo()

    positions = np.searchsorted(items, filter_query_items.col)
    positions = np.clip(positions, 0, len(items) - 1)

    filter_query_items.data[items[positions] != filter_query_items.col] = 0
    filter_query_items.col = positions
    filter_query_items.eliminate_zeros()
    return filter_query_items.tocsr()

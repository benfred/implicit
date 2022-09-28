import logging

import annoy
import numpy as np
from scipy.sparse import csr_matrix

import implicit.gpu
from implicit.recommender_base import RecommenderBase
from implicit.utils import _batch_call, _filter_items_from_results, augment_inner_product_matrix

log = logging.getLogger("implicit")


class AnnoyModel(RecommenderBase):

    """Speeds up inference calls to MatrixFactorization models by using an
    `Annoy <https://github.com/spotify/annoy>`_ index to calculate similar items and
    recommend items.

    Parameters
    ----------
    model : MatrixFactorizationBase
        A matrix factorization model to use for the factors
    n_trees : int, optional
        The number of trees to use when building the Annoy index. More trees gives higher precision
        when querying.
    search_k : int, optional
        Provides a way to search more trees at runtime, giving the ability to have more accurate
        results at the cost of taking more time.
    approximate_similar_items : bool, optional
        whether or not to build an Annoy index for computing similar_items
    approximate_recommend : bool, optional
        whether or not to build an Annoy index for the recommend call

    Attributes
    ----------
    similar_items_index : annoy.AnnoyIndex
        Annoy index for looking up similar items in the cosine space formed by the latent
        item_factors

    recommend_index : annoy.AnnoyIndex
        Annoy index for looking up similar items in the inner product space formed by the latent
        item_factors
    """

    def __init__(
        self,
        model,
        approximate_similar_items=True,
        approximate_recommend=True,
        n_trees=50,
        search_k=-1,
    ):
        self.model = model

        self.similar_items_index = None
        self.recommend_index = None
        self.max_norm = None

        self.approximate_similar_items = approximate_similar_items
        self.approximate_recommend = approximate_recommend

        self.n_trees = n_trees
        self.search_k = search_k

    def fit(self, Cui, show_progress=True, callback=None):
        # train the model
        self.model.fit(Cui, show_progress, callback)

        item_factors = self.model.item_factors
        if implicit.gpu.HAS_CUDA and isinstance(item_factors, implicit.gpu.Matrix):
            item_factors = item_factors.to_numpy()
        item_factors = item_factors.astype("float32")

        # build up an Annoy Index with all the item_factors (for calculating
        # similar items)
        if self.approximate_similar_items:
            log.debug("Building annoy similar items index")

            self.similar_items_index = annoy.AnnoyIndex(item_factors.shape[1], "angular")
            for i, row in enumerate(item_factors):
                self.similar_items_index.add_item(i, row)
            self.similar_items_index.build(self.n_trees)

        # build up a separate index for the inner product (for recommend
        # methods)
        if self.approximate_recommend:
            log.debug("Building annoy recommendation index")
            self.max_norm, extra = augment_inner_product_matrix(item_factors)
            self.recommend_index = annoy.AnnoyIndex(extra.shape[1], "angular")
            for i, row in enumerate(extra):
                self.recommend_index.add_item(i, row)
            self.recommend_index.build(self.n_trees)

    def similar_items(
        self, itemid, N=10, recalculate_item=False, item_users=None, filter_items=None, items=None
    ):
        if items is not None and self.approximate_similar_items:
            raise NotImplementedError("using an items filter isn't supported with ANN lookup")

        count = N
        if filter_items is not None:
            count += len(filter_items)

        if not self.approximate_similar_items:
            return self.model.similar_items(
                itemid,
                N,
                recalculate_item=recalculate_item,
                item_users=item_users,
                filter_items=filter_items,
                items=items,
            )

        # annoy doesn't have a batch mode we can use
        if not np.isscalar(itemid):
            return _batch_call(
                self.similar_items,
                itemid,
                N=N,
                recalculate_item=recalculate_item,
                item_users=item_users,
                filter_items=filter_items,
            )

        # support recalculate_item if possible. TODO: refactor this
        if hasattr(self.model, "_item_factor"):
            factor = self.model._item_factor(
                itemid, item_users, recalculate_item
            )  # pylint: disable=protected-access
        elif recalculate_item:
            raise NotImplementedError(f"recalculate_item isn't supported with {self.model}")
        else:
            factor = self.model.item_factors[itemid]
            if implicit.gpu.HAS_CUDA and isinstance(factor, implicit.gpu.Matrix):
                factor = factor.to_numpy()

        if len(factor.shape) != 1:
            factor = factor.squeeze()

        ids, scores = self.similar_items_index.get_nns_by_vector(
            factor, N, search_k=self.search_k, include_distances=True
        )
        ids, scores = np.array(ids), np.array(scores)

        if filter_items is not None:
            ids, scores = _filter_items_from_results(itemid, ids, scores, filter_items, N)

        return ids, 1 - (scores**2) / 2

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
        if (filter_already_liked_items or recalculate_user) and not isinstance(
            user_items, csr_matrix
        ):
            raise ValueError("user_items needs to be a CSR sparse matrix")

        if items is not None and self.approximate_recommend:
            raise NotImplementedError("using a 'items' list with ANN search isn't supported")

        if not self.approximate_recommend:
            return self.model.recommend(
                userid,
                user_items,
                N=N,
                filter_already_liked_items=filter_already_liked_items,
                filter_items=filter_items,
                recalculate_user=recalculate_user,
                items=items,
            )

        # batch computation isn't supported by annoy, fallback to looping over items
        if not np.isscalar(userid):
            return _batch_call(
                self.recommend,
                userid,
                user_items=user_items,
                N=N,
                filter_already_liked_items=filter_already_liked_items,
                filter_items=filter_items,
                recalculate_user=recalculate_user,
                items=items,
            )

        # support recalculate_user if possible (TODO: come back to this since its a bit of a hack)
        if hasattr(self.model, "+_user_factor"):
            user = self.model._user_factor(
                userid, user_items, recalculate_user
            )  # pylint: disable=protected-access
        elif recalculate_user:
            raise NotImplementedError(f"recalculate_user isn't supported with {self.model}")
        else:
            user = self.model.user_factors[userid]
            if implicit.gpu.HAS_CUDA and isinstance(user, implicit.gpu.Matrix):
                user = user.to_numpy()

        # calculate the top N items, removing the users own liked items from
        # the results
        count = N
        if filter_items:
            count += len(filter_items)
            filter_items = np.array(filter_items)

        if filter_already_liked_items:
            user_likes = user_items[0].indices
            filter_items = (
                np.append(filter_items, user_likes) if filter_items is not None else user_likes
            )
            count += len(user_likes)

        query = np.append(user, 0)
        ids, scores = self.recommend_index.get_nns_by_vector(
            query, count, include_distances=True, search_k=self.search_k
        )
        ids, scores = np.array(ids), np.array(scores)

        if filter_items is not None:
            ids, scores = _filter_items_from_results(userid, ids, scores, filter_items, N)

        # convert the distances from euclidean to cosine distance,
        # and then rescale the cosine distance to go back to inner product
        scaling = self.max_norm * np.linalg.norm(query)
        scores = scaling * (1 - (scores**2) / 2)
        return ids, scores

    def similar_users(self, userid, N=10, filter_users=None, users=None):
        raise NotImplementedError(
            "similar_users isn't implemented with Annoy yet. (note: you can call "
            " self.model.similar_models to get the same functionality on the inner model class)"
        )

    def save(self, file):
        raise NotImplementedError(".save isn't implemented for Annoy yet")

    @classmethod
    def load(cls, file):
        raise NotImplementedError(".load isn't implemented for Annoy yet")

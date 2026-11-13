import logging
import warnings

import mlxvs
import numpy as np
from scipy.sparse import csr_matrix

import implicit.gpu
from implicit.recommender_base import RecommenderBase
from implicit.utils import _batch_call, _filter_items_from_results, augment_inner_product_matrix

log = logging.getLogger("implicit")


class _Index:
    """An inner product index of a set of factors: a CAGRA graph searched on the GPU,
    or exact search on the GPU for small sets and large result counts"""

    def __init__(self, factors, graph_degree, intermediate_graph_degree, itopk_size):
        self.factors = np.ascontiguousarray(factors, dtype=np.float32)
        self.itopk_size = itopk_size
        self.graph = None
        if len(self.factors) > intermediate_graph_degree:
            self.graph = mlxvs.CagraIndex.build(
                self.factors,
                graph_degree,
                "inner_product",
                intermediate_graph_degree=intermediate_graph_degree,
            )

    def search(self, queries, count):
        """returns (scores, ids) of the count rows of largest inner product with each query"""
        count = min(count, len(self.factors))
        if self.graph is not None and count <= 1024:
            ids, scores = self.graph.search(queries, count, itopk_size=max(self.itopk_size, count))
        elif count <= 256:
            ids, scores = mlxvs.brute_force_knn(self.factors, queries, count, "inner_product")
        else:
            # beyond the GPU top-k: exact search with numpy
            all_scores = queries @ self.factors.T
            ids = np.argsort(-all_scores, axis=1)[:, :count]
            scores = np.take_along_axis(all_scores, ids, axis=1)
        return np.asarray(scores), np.asarray(ids)


class MLXVSModel(RecommenderBase):
    """
    Speeds up inference calls to MatrixFactorization models on the GPU of Apple silicon Macs, by
    using `mlxvs <https://github.com/mikepapadim/mlxvs>`_ to build CAGRA graph indices of the
    latent factors (the graph search of NVIDIA's cuVS, on Apple's MLX). Needs macOS 14 or later on
    Apple silicon.

    Parameters
    ----------
    model : MatrixFactorizationBase
        A matrix factorization model to use for the factors
    approximate_similar_items : bool, optional
        whether or not to build an index for computing similar_items
    approximate_recommend : bool, optional
        whether or not to build an index for the recommend call
    graph_degree : int, optional
        The number of neighbours of each item in the search graph.
    intermediate_graph_degree : int, optional
        The number of neighbours in the k-NN graph pruned into the search graph. Item sets of at
        most this many items are searched exactly.
    itopk_size : int, optional
        The length of each query's search list: higher is more accurate and slower.

    Attributes
    ----------
    similar_items_index :
        Index for looking up similar items in the cosine space formed by the latent item_factors

    recommend_index :
        Index for looking up similar items in the inner product space formed by the latent
        item_factors
    """

    def __init__(
        self,
        model,
        approximate_similar_items=True,
        approximate_recommend=True,
        graph_degree=32,
        intermediate_graph_degree=64,
        itopk_size=64,
    ):
        self.model = model
        self.similar_items_index = None
        self.recommend_index = None

        self.approximate_similar_items = approximate_similar_items
        self.approximate_recommend = approximate_recommend

        # hyper-parameters for mlxvs
        self.graph_degree = graph_degree
        self.intermediate_graph_degree = intermediate_graph_degree
        self.itopk_size = itopk_size
        super().__init__()

    def _index(self, factors):
        return _Index(factors, self.graph_degree, self.intermediate_graph_degree, self.itopk_size)

    def fit(self, Cui, show_progress=True, callback=None):
        self.model.fit(Cui, show_progress, callback=callback)

        item_factors = self.model.item_factors
        if implicit.gpu.HAS_CUDA and isinstance(item_factors, implicit.gpu.Matrix):
            item_factors = item_factors.to_numpy()
        item_factors = item_factors.astype("float32")

        if self.approximate_recommend:
            log.debug("Building mlxvs recommendation index")

            # graph search finds the largest inner products poorly when the norms of the items
            # vary: add a dimension so that every item has the same norm (queries get a 0 there),
            # as the annoy and nmslib models do
            _, augmented = augment_inner_product_matrix(item_factors)
            self.recommend_index = self._index(augmented)

        if self.approximate_similar_items:
            log.debug("Building mlxvs similar items index")

            # cosine similarity: an inner product index on normalized vectors
            norms = np.linalg.norm(item_factors, axis=1)
            norms[norms == 0] = 1e-10
            self.similar_items_index = self._index((item_factors.T / norms).T)

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

        # support recalculate_item if possible
        if hasattr(self.model, "_item_factor"):
            factors = self.model._item_factor(itemid, item_users, recalculate_item)  # pylint: disable=protected-access
        elif recalculate_item:
            raise NotImplementedError(f"recalculate_item isn't supported with {self.model}")
        else:
            factors = self.model.item_factors[itemid]
            if implicit.gpu.HAS_CUDA and isinstance(factors, implicit.gpu.Matrix):
                factors = factors.to_numpy()

        factors = np.array(factors, dtype="float32")
        if np.isscalar(itemid):
            factors = factors.reshape(1, -1)
        factors /= np.maximum(np.linalg.norm(factors, axis=1), 1e-10)[:, None]

        scores, ids = self.similar_items_index.search(factors, count)

        if np.isscalar(itemid):
            ids, scores = ids[0], scores[0]

        if filter_items is not None:
            ids, scores = _filter_items_from_results(itemid, ids, scores, filter_items, N)

        return ids, scores

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

        # filter_already_liked_items needs a different number of results per user: query one
        # user at a time
        if filter_already_liked_items and not np.isscalar(userid):
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

        if not self.approximate_recommend:
            warnings.warn("Calling recommend on a MLXVSModel with approximate_recommend=False")
            return self.model.recommend(
                userid,
                user_items,
                N=N,
                filter_already_liked_items=filter_already_liked_items,
                filter_items=filter_items,
                recalculate_user=recalculate_user,
                items=items,
            )

        if hasattr(self.model, "_user_factor"):
            user = self.model._user_factor(userid, user_items, recalculate_user)  # pylint: disable=protected-access
        elif recalculate_user:
            raise NotImplementedError(f"recalculate_user isn't supported with {self.model}")
        else:
            user = self.model.user_factors[userid]
            if implicit.gpu.HAS_CUDA and isinstance(user, implicit.gpu.Matrix):
                user = user.to_numpy()

        # calculate the top N items, removing the users own liked items from the results
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

        query = np.array(user, dtype="float32")
        if np.isscalar(userid):
            query = query.reshape(1, -1)
        # the extra dimension of the augmented item factors
        query = np.append(query, np.zeros((len(query), 1), dtype="float32"), axis=1)

        scores, ids = self.recommend_index.search(query, count)

        if np.isscalar(userid):
            ids, scores = ids[0], scores[0]

        if filter_items is not None:
            ids, scores = _filter_items_from_results(userid, ids, scores, filter_items, N)

        return ids, scores

    def similar_users(self, userid, N=10, filter_users=None, users=None):
        raise NotImplementedError(
            "similar_users isn't implemented with mlxvs yet. (note: you can call "
            " self.model.similar_models to get the same functionality on the inner model class)"
        )

    def save(self, file):
        raise NotImplementedError(".save isn't implemented for mlxvs yet")

    @classmethod
    def load(cls, file):
        raise NotImplementedError(".load isn't implemented for mlxvs yet")

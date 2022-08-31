import logging
import warnings

import faiss
import numpy as np
from scipy.sparse import csr_matrix

import implicit.gpu
from implicit.recommender_base import RecommenderBase
from implicit.utils import _batch_call, _filter_items_from_results

log = logging.getLogger("implicit")


# pylint:  disable=no-value-for-parameter


class FaissModel(RecommenderBase):
    """
    Speeds up inference calls to MatrixFactorization models by using
    `Faiss <https://github.com/facebookresearch/faiss>`_ to create approximate nearest neighbours
    indices of the latent factors.

    Parameters
    ----------
    model : MatrixFactorizationBase
        A matrix factorization model to use for the factors
    nlist : int, optional
        The number of cells to use when building the Faiss index.
    nprobe : int, optional
        The number of cells to visit to perform a search.
    use_gpu : bool, optional
        Whether or not to enable run Faiss on the GPU. Requires faiss to have been
        built with GPU support.
    approximate_similar_items : bool, optional
        whether or not to build an Faiss index for computing similar_items
    approximate_recommend : bool, optional
        whether or not to build an Faiss index for the recommend call

    Attributes
    ----------
    similar_items_index : faiss.IndexIVFFlat
        Faiss index for looking up similar items in the cosine space formed by the latent
        item_factors

    recommend_index : faiss.IndexIVFFlat
        Faiss index for looking up similar items in the inner product space formed by the latent
        item_factors
    """

    def __init__(
        self,
        model,
        approximate_similar_items=True,
        approximate_recommend=True,
        nlist=400,
        nprobe=20,
        use_gpu=implicit.gpu.HAS_CUDA,
    ):
        self.model = model
        self.similar_items_index = None
        self.recommend_index = None
        self.quantizer = None
        self.gpu_resources = None
        self.factors = None

        self.approximate_similar_items = approximate_similar_items
        self.approximate_recommend = approximate_recommend

        # hyper-parameters for FAISS
        self.nlist = nlist
        self.nprobe = nprobe
        self.use_gpu = use_gpu
        super().__init__()

    def fit(self, Cui, show_progress=True, callback=None):
        self.model.fit(Cui, show_progress, callback=callback)

        item_factors = self.model.item_factors
        if implicit.gpu.HAS_CUDA and isinstance(item_factors, implicit.gpu.Matrix):
            item_factors = item_factors.to_numpy()
        item_factors = item_factors.astype("float32")

        self.factors = item_factors.shape[1]

        self.quantizer = faiss.IndexFlat(self.factors)

        if self.use_gpu:
            self.gpu_resources = faiss.StandardGpuResources()

        if self.approximate_recommend:
            log.debug("Building faiss recommendation index")

            # build up a inner product index here
            if self.use_gpu:
                index = faiss.GpuIndexIVFFlat(
                    self.gpu_resources, self.factors, self.nlist, faiss.METRIC_INNER_PRODUCT
                )
            else:
                index = faiss.IndexIVFFlat(
                    self.quantizer, self.factors, self.nlist, faiss.METRIC_INNER_PRODUCT
                )

            index.train(item_factors)
            index.add(item_factors)
            index.nprobe = self.nprobe
            self.recommend_index = index

        if self.approximate_similar_items:
            log.debug("Building faiss similar items index")

            # likewise build up cosine index for similar_items, using an inner product
            # index on normalized vectors`
            norms = np.linalg.norm(item_factors, axis=1)
            norms[norms == 0] = 1e-10

            normalized = (item_factors.T / norms).T.astype("float32")
            if self.use_gpu:
                index = faiss.GpuIndexIVFFlat(
                    self.gpu_resources, self.factors, self.nlist, faiss.METRIC_INNER_PRODUCT
                )
            else:
                index = faiss.IndexIVFFlat(
                    self.quantizer, self.factors, self.nlist, faiss.METRIC_INNER_PRODUCT
                )

            index.train(normalized)
            index.add(normalized)
            index.nprobe = self.nprobe
            self.similar_items_index = index

    def similar_items(
        self, itemid, N=10, recalculate_item=False, item_users=None, filter_items=None, items=None
    ):
        if items is not None and self.approximate_similar_items:
            raise NotImplementedError("using an items filter isn't supported with ANN lookup")

        count = N
        if filter_items is not None:
            count += len(filter_items)

        if not self.approximate_similar_items or (self.use_gpu and count >= 1024):
            return self.model.similar_items(
                itemid,
                N,
                recalculate_item=recalculate_item,
                item_users=item_users,
                filter_items=filter_items,
                items=items,
            )

        # support recalculate_item if possible. TODO: refactor this
        if hasattr(self.model, "_item_factor"):
            factors = self.model._item_factor(
                itemid, item_users, recalculate_item
            )  # pylint: disable=protected-access
        elif recalculate_item:
            raise NotImplementedError(f"recalculate_item isn't supported with {self.model}")
        else:
            factors = self.model.item_factors[itemid]
            if implicit.gpu.HAS_CUDA and isinstance(factors, implicit.gpu.Matrix):
                factors = factors.to_numpy()

        if np.isscalar(itemid):
            factors /= np.linalg.norm(factors)
            factors = factors.reshape(1, -1)
        else:
            factors /= np.linalg.norm(factors, axis=1)[:, None]

        scores, ids = self.similar_items_index.search(factors.astype("float32"), count)

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

        # batch computation is tricky with filter_already_liked_items (requires querying a
        # different number of rows per user). Instead just fallback to a faiss query per user
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
            warnings.warning("Calling recommend on a FaissModel with approximate_recommend=False")
            return self.model.recommend(
                userid,
                user_items,
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

        # the GPU variant of faiss doesn't support returning more than 1024 results.
        # fall back to the exact match when this happens
        if self.use_gpu and count >= 1024:
            return self.model.recommend(
                userid,
                user_items,
                N=N,
                filter_already_liked_items=filter_already_liked_items,
                filter_items=filter_items,
                recalculate_user=recalculate_user,
                items=items,
            )

        if np.isscalar(userid):
            query = user.reshape(1, -1).astype("float32")
        else:
            query = user.astype("float32")

        scores, ids = self.recommend_index.search(query, count)

        if np.isscalar(userid):
            ids, scores = ids[0], scores[0]

        if filter_items is not None:
            ids, scores = _filter_items_from_results(userid, ids, scores, filter_items, N)

        return ids, scores

    def similar_users(self, userid, N=10, filter_users=None, users=None):
        raise NotImplementedError(
            "similar_users isn't implemented with Faiss yet. (note: you can call "
            " self.model.similar_models to get the same functionality on the inner model class)"
        )

    def save(self, file):
        raise NotImplementedError(".save isn't implemented for Faiss yet")

    @classmethod
    def load(cls, file):
        raise NotImplementedError(".load isn't implemented for Faiss yet")

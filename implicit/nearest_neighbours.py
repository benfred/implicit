import numpy as np
from numpy import bincount, log, log1p, sqrt
from scipy.sparse import coo_matrix, csr_matrix

from ._nearest_neighbours import NearestNeighboursScorer, all_pairs_knn
from .recommender_base import RecommenderBase
from .utils import _batch_call


class ItemItemRecommender(RecommenderBase):
    """Base class for Item-Item Nearest Neighbour recommender models
    here.

    Parameters
    ----------
    K : int, optional
        The number of neighbours to include when calculating the item-item
        similarity matrix
    num_threads : int, optional
        The number of threads to use for fitting the model. Specifying 0
        means to default to the number of cores on the machine.
    """

    def __init__(self, K=20, num_threads=0):
        self.similarity = None
        self.K = K
        self.num_threads = num_threads
        self.scorer = None

    def fit(self, weighted, show_progress=True, callback=None):
        """Computes and stores the similarity matrix"""
        if callback:
            raise NotImplementedError("callback isn't support on ItemItemRecommender.fit")

        self.similarity = all_pairs_knn(
            weighted,
            self.K,
            show_progress=show_progress,
            num_threads=self.num_threads,
        ).tocsr()
        self.scorer = NearestNeighboursScorer(self.similarity)

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
        if not isinstance(user_items, csr_matrix):
            raise ValueError("user_items needs to be a CSR sparse matrix")

        if not np.isscalar(userid):
            if user_items.shape[0] != len(userid):
                raise ValueError("user_items must contain 1 row for every user in userids")

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

        if filter_items is not None and items is not None:
            raise ValueError("Can't specify both filter_items and items")

        if filter_items is not None:
            N += len(filter_items)
        elif items is not None:
            items = np.array(items)
            N = self.similarity.shape[0]
            # check if items contains itemids that are not in the model(user_items)
            if items.max() >= N or items.min() < 0:
                raise IndexError("Some of selected itemids are not in the model")

        ids, scores = self.scorer.recommend(
            user_items.indptr,
            user_items.indices,
            user_items.data,
            K=N,
            remove_own_likes=filter_already_liked_items,
        )

        if filter_items is not None:
            mask = np.in1d(ids, filter_items, invert=True)
            ids, scores = ids[mask][:N], scores[mask][:N]

        elif items is not None:
            mask = np.in1d(ids, items)
            ids, scores = ids[mask], scores[mask]

            # returned items should be equal to input selected items
            missing = items[np.in1d(items, ids, invert=True)]
            if missing.size:
                ids = np.append(ids, missing)
                scores = np.append(scores, np.full(missing.size, -np.finfo(scores.dtype).max))

        return ids, scores

    def similar_users(self, userid, N=10, filter_users=None, users=None):
        raise NotImplementedError("similar_users isn't implemented for item-item recommenders")

    def similar_items(
        self, itemid, N=10, recalculate_item=False, item_users=None, filter_items=None, items=None
    ):
        if recalculate_item:
            raise NotImplementedError("Recalculate_item isn't implemented")

        if not np.isscalar(itemid):
            return _batch_call(
                self.similar_items, itemid, N=N, filter_items=filter_items, items=items
            )

        if filter_items is not None and items is not None:
            raise ValueError("Can't specify both filter_items and items")

        if itemid >= self.similarity.shape[0]:
            return np.array([]), np.array([])

        ids = self.similarity[itemid].indices
        scores = self.similarity[itemid].data

        if filter_items is not None:
            mask = np.in1d(ids, filter_items, invert=True)
            ids, scores = ids[mask], scores[mask]

        elif items is not None:
            mask = np.in1d(ids, items)
            ids, scores = ids[mask], scores[mask]

            # returned items should be equal to input selected items
            missing = items[np.in1d(items, ids, invert=True)]
            if missing.size:
                ids = np.append(ids, missing)
                scores = np.append(scores, np.full(missing.size, -np.finfo(scores.dtype).max))

        best = np.argsort(scores)[::-1][:N]
        return ids[best], scores[best]

    def __getstate__(self):
        state = self.__dict__.copy()
        # scorer isn't picklable
        del state["scorer"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.similarity is not None:
            self.scorer = NearestNeighboursScorer(self.similarity)
        else:
            self.scorer = None

    def save(self, fileobj_or_path):
        args = {"K": self.K}
        m = self.similarity
        if m is not None:
            args.update(
                {"shape": m.shape, "data": m.data, "indptr": m.indptr, "indices": m.indices}
            )
        np.savez(fileobj_or_path, **args)

    @classmethod
    def load(cls, fileobj_or_path):
        # numpy.save automatically appends a npz suffic, numpy.load doesn't apparently
        if isinstance(fileobj_or_path, str) and not fileobj_or_path.endswith(".npz"):
            fileobj_or_path = fileobj_or_path + ".npz"

        with np.load(fileobj_or_path, allow_pickle=False) as data:
            ret = cls()
            if data.get("data") is not None:
                similarity = csr_matrix(
                    (data["data"], data["indices"], data["indptr"]), shape=data["shape"]
                )
                ret.similarity = similarity
                ret.scorer = NearestNeighboursScorer(similarity)
            ret.K = data["K"]
            return ret


class CosineRecommender(ItemItemRecommender):
    """An Item-Item Recommender on Cosine distances between items"""

    def fit(self, counts, show_progress=True, callback=None):
        # cosine distance is just the dot-product of a normalized matrix
        ItemItemRecommender.fit(self, normalize(counts.T).T, show_progress, callback)


class TFIDFRecommender(ItemItemRecommender):
    """An Item-Item Recommender on TF-IDF distances between items"""

    def fit(self, counts, show_progress=True, callback=None):
        weighted = normalize(tfidf_weight(counts.T)).T
        ItemItemRecommender.fit(self, weighted, show_progress, callback)


class BM25Recommender(ItemItemRecommender):
    """An Item-Item Recommender on BM25 distance between items"""

    def __init__(self, K=20, K1=1.2, B=0.75, num_threads=0):
        super().__init__(K, num_threads)
        self.K1 = K1
        self.B = B

    def fit(self, counts, show_progress=True, callback=None):
        weighted = bm25_weight(counts.T, self.K1, self.B).T
        ItemItemRecommender.fit(self, weighted, show_progress, callback)


def tfidf_weight(X):
    """Weights a Sparse Matrix by TF-IDF Weighted"""
    X = coo_matrix(X)

    # calculate IDF
    N = float(X.shape[0])
    idf = log(N) - log1p(bincount(X.col))

    # apply TF-IDF adjustment
    X.data = sqrt(X.data) * idf[X.col]
    return X


def normalize(X):
    """equivalent to scipy.preprocessing.normalize on sparse matrices
    , but lets avoid another dependency just for a small utility function"""
    X = coo_matrix(X)
    X.data = X.data / sqrt(bincount(X.row, X.data**2))[X.row]
    return X


def bm25_weight(X, K1=100, B=0.8):
    """Weighs each row of a sparse matrix X  by BM25 weighting"""
    # calculate idf per term (user)
    X = coo_matrix(X)

    N = float(X.shape[0])
    idf = log(N) - log1p(bincount(X.col))

    # calculate length_norm per document (artist)
    row_sums = np.ravel(X.sum(axis=1))
    average_length = row_sums.mean()
    length_norm = (1.0 - B) + B * row_sums / average_length

    # weight matrix rows by bm25
    X.data = X.data * (K1 + 1.0) / (K1 * length_norm[X.row] + X.data) * idf[X.col]
    return X

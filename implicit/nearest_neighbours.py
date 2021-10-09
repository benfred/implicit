import numpy
from numpy import bincount, log, log1p, sqrt
from scipy.sparse import coo_matrix, csr_matrix

from ._nearest_neighbours import NearestNeighboursScorer, all_pairs_knn
from .recommender_base import RecommenderBase


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

    def fit(self, weighted, show_progress=True):
        """Computes and stores the similarity matrix"""
        self.similarity = all_pairs_knn(
            weighted, self.K, show_progress=show_progress, num_threads=self.num_threads
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
    ):
        """returns the best N recommendations for a user given its id"""
        if userid >= user_items.shape[0]:
            raise ValueError("userid is out of bounds of the user_items matrix")

        # recalculate_user is ignored because this is not a model based algorithm
        items = N
        if filter_items:
            items += len(filter_items)

        ids, scores = self.scorer.recommend(
            userid,
            user_items.indptr,
            user_items.indices,
            user_items.data,
            K=items,
            remove_own_likes=filter_already_liked_items,
        )

        if filter_items:
            mask = numpy.in1d(ids, filter_items, invert=True)
            ids, scores = ids[mask][:N], scores[mask][:N]

        return ids, scores

    def rank_items(self, userid, user_items, selected_items, recalculate_user=False):
        """Rank given items for a user and returns sorted item list"""
        # check if selected_items contains itemids that are not in the model(user_items)
        if max(selected_items) >= user_items.shape[1] or min(selected_items) < 0:
            raise IndexError("Some of selected itemids are not in the model")

        selected_items = numpy.array(selected_items)

        # calculate the relevance scores
        liked_vector = user_items.getrow(userid)
        recommendations = liked_vector.dot(self.similarity)

        # remove items that are not in the selected_items
        ids, scores = recommendations.indices, recommendations.data
        mask = numpy.in1d(ids, selected_items)
        ids, scores = ids[mask], scores[mask]

        # returned items should be equal to input selected items
        missing = selected_items[numpy.in1d(selected_items, ids, invert=True)]
        if missing.size:
            ids = numpy.append(ids, missing)
            scores = numpy.append(scores, numpy.full(missing.size, -numpy.finfo(scores.dtype).max))
        return ids, scores

    def similar_users(self, userid, N=10):
        raise NotImplementedError("Not implemented Yet")

    def similar_items(self, itemid, N=10):
        """Returns a list of the most similar other items"""
        if itemid >= self.similarity.shape[0]:
            return numpy.array([]), numpy.array([])

        ids = self.similarity[itemid].indices
        scores = self.similarity[itemid].data
        best = numpy.argsort(scores)[::-1][:N]
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

    def save(self, filename):
        m = self.similarity
        numpy.savez(
            filename, data=m.data, indptr=m.indptr, indices=m.indices, shape=m.shape, K=self.K
        )

    @classmethod
    def load(cls, filename):
        # numpy.save automatically appends a npz suffic, numpy.load doesn't apparently
        if not filename.endswith(".npz"):
            filename = filename + ".npz"

        m = numpy.load(filename)
        similarity = csr_matrix((m["data"], m["indices"], m["indptr"]), shape=m["shape"])

        ret = cls()
        ret.similarity = similarity
        ret.scorer = NearestNeighboursScorer(similarity)
        ret.K = m["K"]
        return ret


class CosineRecommender(ItemItemRecommender):
    """An Item-Item Recommender on Cosine distances between items"""

    def fit(self, counts, show_progress=True):
        # cosine distance is just the dot-product of a normalized matrix
        ItemItemRecommender.fit(self, normalize(counts), show_progress)


class TFIDFRecommender(ItemItemRecommender):
    """An Item-Item Recommender on TF-IDF distances between items"""

    def fit(self, counts, show_progress=True):
        weighted = normalize(tfidf_weight(counts))
        ItemItemRecommender.fit(self, weighted, show_progress)


class BM25Recommender(ItemItemRecommender):
    """An Item-Item Recommender on BM25 distance between items"""

    def __init__(self, K=20, K1=1.2, B=0.75, num_threads=0):
        super(BM25Recommender, self).__init__(K, num_threads)
        self.K1 = K1
        self.B = B

    def fit(self, counts, show_progress=True):
        weighted = bm25_weight(counts, self.K1, self.B)
        ItemItemRecommender.fit(self, weighted, show_progress)


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
    , but lets avoid another depedency just for a small utility function"""
    X = coo_matrix(X)
    X.data = X.data / sqrt(bincount(X.row, X.data ** 2))[X.row]
    return X


def bm25_weight(X, K1=100, B=0.8):
    """Weighs each row of a sparse matrix X  by BM25 weighting"""
    # calculate idf per term (user)
    X = coo_matrix(X)

    N = float(X.shape[0])
    idf = log(N) - log1p(bincount(X.col))

    # calculate length_norm per document (artist)
    row_sums = numpy.ravel(X.sum(axis=1))
    average_length = row_sums.mean()
    length_norm = (1.0 - B) + B * row_sums / average_length

    # weight matrix rows by bm25
    X.data = X.data * (K1 + 1.0) / (K1 * length_norm[X.row] + X.data) * idf[X.col]
    return X

import itertools

import numpy
from numpy import bincount, log, log1p, sqrt
from scipy.sparse import coo_matrix, csr_matrix

from ._nearest_neighbours import all_pairs_knn
from .recommender_base import RecommenderBase
from .utils import nonzeros


class ItemItemRecommender(RecommenderBase):
    """ Base class for Item-Item Nearest Neighbour recommender models
    here """
    def __init__(self, K=20):
        self.similarity = None
        self.K = K

    def fit(self, weighted):
        """ Computes and stores the similarity matrix """
        self.similarity = all_pairs_knn(weighted, self.K).tocsr()

    def recommend(self, userid, user_items, N=10, filter_items=None, recalculate_user=False):
        """ returns the best N recommendations for a user given its id"""
        # recalculate_user is ignored because this is not a model based algorithm
        liked_vector = user_items[userid]

        # calculate the top related items
        recommendations = liked_vector.dot(self.similarity)
        best = sorted(zip(recommendations.indices, recommendations.data), key=lambda x: -x[1])

        # remove users own liked items from the output
        liked = set(liked_vector.indices)
        if filter_items:
            liked.update(filter_items)

        return list(itertools.islice((rec for rec in best if rec[0] not in liked), N))

    def similar_items(self, itemid, N=10):
        """ Returns a list of the most similar other items """
        if itemid >= self.similarity.shape[0]:
            return []

        return sorted(list(nonzeros(self.similarity, itemid)), key=lambda x: -x[1])[:N]

    def save(self, filename):
        m = self.similarity
        numpy.savez(filename, data=m.data, indptr=m.indptr, indices=m.indices, shape=m.shape,
                    K=self.K)

    @classmethod
    def load(cls, filename):
        # numpy.save automatically appends a npz suffic, numpy.load doesn't apparently
        if not filename.endswith(".npz"):
            filename = filename + ".npz"

        m = numpy.load(filename)
        similarity = csr_matrix((m['data'], m['indices'], m['indptr']), shape=m['shape'])

        ret = cls()
        ret.similarity = similarity
        ret.K = m['K']
        return ret


class CosineRecommender(ItemItemRecommender):
    """ An Item-Item Recommender on Cosine distances between items """
    def fit(self, counts):
        # cosine distance is just the dot-product of a normalized matrix
        ItemItemRecommender.fit(self, normalize(counts))


class TFIDFRecommender(ItemItemRecommender):
    """ An Item-Item Recommender on TF-IDF distances between items """
    def fit(self, counts):
        weighted = normalize(tfidf_weight(counts))
        ItemItemRecommender.fit(self, weighted)


class BM25Recommender(ItemItemRecommender):
    """ An Item-Item Recommender on BM25 distance between items """
    def __init__(self, K=20, K1=1.2, B=.75):
        super(BM25Recommender, self).__init__(K)
        self.K1 = K1
        self.B = B

    def fit(self, counts):
        weighted = bm25_weight(counts, self.K1, self.B)
        ItemItemRecommender.fit(self, weighted)


def tfidf_weight(X):
    """ Weights a Sparse Matrix by TF-IDF Weighted """
    X = coo_matrix(X)

    # calculate IDF
    N = float(X.shape[0])
    idf = log(N) - log1p(bincount(X.col))

    # apply TF-IDF adjustment
    X.data = sqrt(X.data) * idf[X.col]
    return X


def normalize(X):
    """ equivalent to scipy.preprocessing.normalize on sparse matrices
    , but lets avoid another depedency just for a small utility function """
    X = coo_matrix(X)
    X.data = X.data / sqrt(bincount(X.row, X.data ** 2))[X.row]
    return X


def bm25_weight(X, K1=100, B=0.8):
    """ Weighs each row of a sparse matrix X  by BM25 weighting """
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

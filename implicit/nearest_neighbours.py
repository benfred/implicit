import numpy
from numpy import bincount, log, sqrt
from scipy.sparse import coo_matrix, csr_matrix

from ._nearest_neighbours import all_pairs_knn
from .utils import nonzeros


class ItemItemRecommender(object):
    """ Base class for Item-Item Nearest Neighbour recommender models
    here """
    def __init__(self):
        self.similarity = None

    def fit(self, weighted, K):
        """ Computes and stores the similarity matrix """
        self.similarity = all_pairs_knn(weighted, K).tocsr()

    def similar_items(self, itemid):
        """ Returns a list of the most similar other items """
        return sorted(list(nonzeros(self.similarity, itemid)), key=lambda x: -x[1])

    def save(self, filename):
        m = self.similarity
        numpy.savez(filename, data=m.data, indptr=m.indptr, indices=m.indices, shape=m.shape)

    @classmethod
    def load(cls, filename):
        # numpy.savez automatically appends a npz suffic, numpy.load doesn't apparently
        if not filename.endswith(".npz"):
            filename = filename + ".npz"

        m = numpy.load(filename)
        similarity = csr_matrix((m['data'], m['indices'], m['indptr']), shape=m['shape'])

        ret = cls()
        ret.similarity = similarity
        return ret


class CosineRecommender(ItemItemRecommender):
    """ An Item-Item Recommender on Cosine distances between items """
    def fit(self, counts, K):
        # cosine distance is just the dot-product of a normalized matrix
        ItemItemRecommender.fit(self, normalize(counts), K)


class TFIDFRecommender(ItemItemRecommender):
    """ An Item-Item Recommender on TF-IDF distances between items """
    def fit(self, counts, K):
        weighted = normalize(tfidf_weight(counts))
        ItemItemRecommender.fit(self, weighted, K)


class BM25Recommender(ItemItemRecommender):
    """ An Item-Item Recommender on BM25 distance between items """
    def __init__(self, K1=1.2, B=.75):
        self.K1 = K1
        self.B = B

    def fit(self, counts, K):
        weighted = bm25_weight(counts, self.K1, self.B)
        ItemItemRecommender.fit(self, weighted, K)


def tfidf_weight(X):
    """ Weights a Sparse Matrix by TF-IDF Weighted """
    X = coo_matrix(X)

    # calculate IDF
    N = float(X.shape[0])
    idf = log(N / (1 + bincount(X.col)))

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
    idf = log(N / (1 + bincount(X.col)))

    # calculate length_norm per document (artist)
    row_sums = numpy.ravel(X.sum(axis=1))
    average_length = row_sums.mean()
    length_norm = (1.0 - B) + B * row_sums / average_length

    # weight matrix rows by bm25
    X.data = X.data * (K1 + 1.0) / (K1 * length_norm[X.row] + X.data) * idf[X.col]
    return X

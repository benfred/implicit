import unittest

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import implicit

from .recommender_base_test import RecommenderBaseTestMixin


class BM25Test(unittest.TestCase, RecommenderBaseTestMixin):
    def _get_model(self):
        return implicit.nearest_neighbours.BM25Recommender(K=50)


class TFIDFTest(unittest.TestCase, RecommenderBaseTestMixin):
    def _get_model(self):
        return implicit.nearest_neighbours.TFIDFRecommender(K=50)


class CosineTest(unittest.TestCase, RecommenderBaseTestMixin):
    def _get_model(self):
        return implicit.nearest_neighbours.CosineRecommender(K=50)


def test_all_pairs_knn():
    counts = csr_matrix(
        [
            [5, 1, 0, 9, 0, 0],
            [0, 2, 1, 1, 0, 0],
            [7, 0, 3, 0, 0, 0],
            [1, 8, 0, 0, 0, 0],
            [0, 0, 4, 4, 0, 0],
            [0, 3, 0, 0, 0, 2],
            [0, 0, 0, 0, 6, 0],
        ],
        dtype=np.float64,
    )
    counts = implicit.nearest_neighbours.tfidf_weight(counts).tocsr()

    # compute all neighbours using matrix dot product
    all_neighbours = counts.T.dot(counts).tocsr()
    K = 3
    knn = implicit.nearest_neighbours.all_pairs_knn(counts, K, show_progress=False).tocsr()

    for rowid in range(counts.shape[1]):
        # make sure values match
        for colid, data in zip(knn[rowid].indices, knn[rowid].data):
            assert pytest.approx(all_neighbours[rowid, colid]) == data

        # make sure top K selected
        row = all_neighbours[rowid]
        assert set(knn[rowid].indices) == set(
            colid for colid, _ in sorted(zip(row.indices, row.data), key=lambda x: -x[1])[:K]
        )

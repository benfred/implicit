import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import implicit

from .recommender_base_test import get_checker_board


@pytest.mark.skipif(not implicit.gpu.HAS_CUDA, reason="needs cuda build")
@pytest.mark.parametrize("k", [4, 16, 64, 128, 1000])
@pytest.mark.parametrize("batch", [1, 10, 100])
@pytest.mark.parametrize("temp_memory", [500_000_000, 5_000_000])
def test_topk_ascending(k, batch, temp_memory):
    num_items = 10000
    factors = 10
    items = np.arange(num_items * factors).reshape((num_items, factors)).astype("float32")
    queries = np.arange(batch * factors).reshape((batch, factors)).astype("float32")
    _check_knn_queries(items, queries, k, max_temp_memory=temp_memory)


@pytest.mark.skipif(not implicit.gpu.HAS_CUDA, reason="needs cuda build")
@pytest.mark.parametrize("k", [4, 64, 128])
@pytest.mark.parametrize("batch", [1, 10, 100])
@pytest.mark.parametrize("temp_memory", [500_000_000, 500_000])
def test_topk_random(k, batch, temp_memory):
    num_items = 1000
    factors = 10
    np.random.seed(0)
    items = np.random.uniform(size=(num_items, factors)).astype("float32")
    queries = np.random.uniform(size=(batch, factors)).astype("float32")
    _check_knn_queries(items, queries, k, max_temp_memory=temp_memory)


def _check_knn_queries(items, queries, k=5, max_temp_memory=500_000_000):
    # compute distances on the gpu
    knn = implicit.gpu._cuda.KnnQuery(max_temp_memory=max_temp_memory)
    ids, distances = knn.topk(
        implicit.gpu._cuda.Matrix(items), implicit.gpu._cuda.Matrix(queries), k
    )

    # compute on the cpu
    batch = queries.dot(items.T)
    exact_ids = np.flip(np.argsort(batch)[:, -k:], axis=1)
    exact_distances = np.zeros(exact_ids.shape)
    for r in range(batch.shape[0]):
        exact_distances[r] = batch[r][exact_ids[r]]

    # make sure that we match
    assert_array_equal(ids, exact_ids)
    assert_allclose(distances, exact_distances, rtol=1e-06)


@pytest.mark.skipif(not implicit.gpu.HAS_CUDA, reason="needs cuda build")
def test_calculate_norms():
    num_items = 100
    factors = 8
    items = np.arange(num_items * factors).reshape((num_items, factors)).astype("float32")
    norms = (
        implicit.gpu._cuda.calculate_norms(implicit.gpu._cuda.Matrix(items))
        .to_numpy()
        .reshape(num_items)
    )
    np_norms = np.linalg.norm(items, axis=1)
    assert_allclose(norms, np_norms)


@pytest.mark.skipif(not implicit.gpu.HAS_CUDA, reason="needs cuda build")
@pytest.mark.parametrize(
    "model_class", [implicit.als.AlternatingLeastSquares, implicit.bpr.BayesianPersonalizedRanking]
)
@pytest.mark.parametrize("from_gpu", [True, False])
def test_cpu_gpu_conversion(model_class, from_gpu):
    model = model_class(use_gpu=from_gpu, factors=32)
    user_plays = get_checker_board(50)
    model.fit(user_plays)
    converted = model.to_cpu() if from_gpu else model.to_gpu()
    assert_allclose(
        model.recommend(0, user_plays[0]),
        converted.recommend(0, user_plays[0]),
        rtol=1e-3,
        atol=1e-3,
    )

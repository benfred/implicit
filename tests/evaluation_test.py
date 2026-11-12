import numpy as np
import pytest
from scipy.sparse import csr_matrix, random

import implicit
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.evaluation import (
    leave_k_out_split,
    precision_at_k,
    ranking_metrics_at_k,
    train_test_split,
)
from implicit.lmf import LogisticMatrixFactorization

pytest.importorskip("h5py")

from implicit.datasets.movielens import get_movielens  # noqa
from recommender_base_test import get_checker_board  # noqa


def _get_sample_matrix():
    return csr_matrix((np.random.random((10, 10)) > 0.5).astype(np.float64))


def _get_matrix():
    mat = random(100, 100, density=0.5, format="csr", dtype=np.float32)
    return mat.tocoo()


def test_train_test_split():
    seed = np.random.randint(1000)
    mat = _get_sample_matrix()
    train, _ = train_test_split(mat, 0.8, seed)
    train2, _ = train_test_split(mat, 0.8, seed)
    assert np.all(train.todense() == train2.todense())


def test_leave_k_out_returns_correct_shape():
    """
    Test that the output matrices are of the same shape as the input matrix.
    """

    mat = _get_matrix()
    train, test = leave_k_out_split(mat, K=1)
    assert train.shape == mat.shape
    assert test.shape == mat.shape


def test_leave_k_out_outputs_produce_input():
    """
    Test that the sum of the output matrices is equal to the input matrix (i.e.
    that summing the output matrices produces the input matrix).
    """

    mat = _get_matrix()
    train, test = leave_k_out_split(mat, K=1)
    assert ((train + test) - mat).nnz == 0


def test_leave_k_split_is_reservable():
    """
    Test that the sum of the train and test set equals the input.
    """

    mat = _get_matrix()
    train, test = leave_k_out_split(mat, K=1)

    # check all matrices are positive, non-zero
    assert mat.sum() > 0
    assert test.sum() > 0
    assert train.sum() > 0

    # check sum of train + test = input
    assert ((train + test) - mat).nnz == 0


def test_leave_k_out_gets_correct_train_only_shape():
    """Test that the correct number of users appear *only* in the train set."""

    mat = _get_matrix()
    train, test = leave_k_out_split(mat, K=1, train_only_size=0.8)
    train_only = ~np.isin(np.unique(train.tocoo().row), test.tocoo().row)

    assert train_only.sum() == int(train.shape[0] * 0.8)


def test_leave_k_out_raises_error_for_k_less_than_zero():
    """
    Test that an error is raised when K < 0.
    """
    with pytest.raises(ValueError):
        leave_k_out_split(None, K=0)


def test_leave_k_out_raises_error_for_invalid_train_only_size_lower_bound():
    """
    Test that an error is raised when train_only_size < 0.
    """
    with pytest.raises(ValueError):
        leave_k_out_split(None, K=1, train_only_size=-1.0)


def test_leave_k_out_raises_error_for_invalid_train_only_size_upper_bound():
    """
    Test that an error is raised when train_only_size >= 1.
    """
    with pytest.raises(ValueError):
        leave_k_out_split(None, K=1, train_only_size=1.0)


def test_evaluate_movielens_100k():
    _, ratings = get_movielens(variant="100k")

    # remove things < min_rating, and convert to implicit dataset
    # by considering ratings as a binary preference only
    min_rating = 3.0
    ratings.data[ratings.data < min_rating] = 0
    ratings.eliminate_zeros()
    ratings.data = np.ones(len(ratings.data))

    user_ratings = ratings.T.tocsr()
    train, test = train_test_split(user_ratings)

    model = implicit.als.AlternatingLeastSquares()
    model.fit(train)

    assert precision_at_k(model, train, test) > 0.2


@pytest.mark.parametrize(
    "model",
    [
        AlternatingLeastSquares(factors=8, use_gpu=False, random_state=42),
        BayesianPersonalizedRanking(factors=8, use_gpu=False, random_state=42),
        LogisticMatrixFactorization(factors=8, use_gpu=False, random_state=42),
    ],
)
def test_ranking_metrics_k_larger_than_number_of_items(model):
    user_items = get_checker_board(50)
    model.fit(user_items, show_progress=False)
    test_user_items = csr_matrix(np.eye(50))

    # asking for more items than there are in the model should only return the available
    # items, and the metrics shouldn't count any item twice
    ids, _ = model.recommend(np.arange(50), user_items, N=100)
    assert ids.shape == (50, 50)

    all_items = ranking_metrics_at_k(model, user_items, test_user_items, K=50, show_progress=False)
    too_many = ranking_metrics_at_k(model, user_items, test_user_items, K=100, show_progress=False)

    assert too_many == pytest.approx(all_items)
    assert all(0 <= value <= 1 for value in too_many.values())

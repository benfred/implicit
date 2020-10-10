# distutils: language = c++
# cython: language_level=3

import cython
import numpy as np
from cython.parallel import parallel, prange
from scipy.sparse import coo_matrix, csr_matrix
from tqdm.auto import tqdm

from libc.math cimport fmin
from libc.stdlib cimport free, malloc
from libc.string cimport memset
from libcpp.unordered_set cimport unordered_set

from .utils import check_random_state


def train_test_split(ratings, train_percentage=0.8, random_state=None):
    """ Randomly splits the ratings matrix into two matrices for training/testing.

    Parameters
    ----------
    ratings : coo_matrix
        A sparse matrix to split
    train_percentage : float
        What percentage of ratings should be used for training
    random_state : int, None or RandomState
        The existing RandomState. If None, or an int, will be used
        to seed a new numpy RandomState.
    Returns
    -------
    (train, test) : csr_matrix, csr_matrix
        A tuple of csr_matrices for training/testing """

    ratings = ratings.tocoo()
    random_state = check_random_state(random_state)
    random_index = random_state.random_sample(len(ratings.data))
    train_index = random_index < train_percentage
    test_index = random_index >= train_percentage

    train = csr_matrix((ratings.data[train_index],
                        (ratings.row[train_index], ratings.col[train_index])),
                       shape=ratings.shape, dtype=ratings.dtype)

    test = csr_matrix((ratings.data[test_index],
                       (ratings.row[test_index], ratings.col[test_index])),
                      shape=ratings.shape, dtype=ratings.dtype)

    test.data[test.data < 0] = 0
    test.eliminate_zeros()

    return train, test


def _choose(rng, int n, float frac):
    size = max(1, int(n * frac))
    return rng.choice(n, size=size, replace=False)


def _single_sample(arr):
    idx = arr.argsort()
    bins = np.bincount(arr)
    rand = np.random.rand(bins.sum()) + np.repeat(np.arange(len(bins)), bins)
    offset = np.r_[0, bins[: -1].cumsum()]
    return idx[rand.argsort()[offset]]


def recommender_split(
    mat, int n_samples = 1, float train_only_size=0.0, random_state=None
):

    random_state = check_random_state(random_state)

    if n_samples < 1:
        raise ValueError("The 'n_samples' must be >= 1.")

    users = mat.row  # user ids, assumes sorted (!)
    items = mat.col  # item ids
    data = mat.data

    unique_users, counts = np.unique(users, return_counts=True)

    # get only users with n + 1 interactions
    candidate_mask = counts > n_samples + 1

    # keep a given subset of users _only_ in the training set.
    if train_only_size > 0.0:
        train_only_mask = ~np.isin(
            unique_users, _choose(random_state, len(unique_users), train_only_size)
        )
        candidate_mask = train_only_mask & candidate_mask

    # get unique users who appear in the test set
    unique_candidate_users = unique_users[candidate_mask]
    full_candidate_mask = np.isin(users, unique_candidate_users)

    # get all users, items and ratings that match specified requirements to be
    # included in test set.
    candidate_users = users[full_candidate_mask]
    candidate_items = items[full_candidate_mask]
    candidate_data = data[full_candidate_mask]

    if n_samples == 1:
        # sample a single item for each candidate user
        # (i.e. get index of each sampled item)
        test_idx = _single_sample(candidate_users)
    else:
        # todo: can this be vectorized too?
        test_idx = np.ravel(
            [random_state.choice(np.where(candidate_users==user)[0], n_samples) for user in unique_candidate_users]
        )

    # get all remaining remaining candidate user-item pairs, and prepare to append to training set.
    train_idx = np.setdiff1d(np.arange(len(candidate_users), dtype=int), test_idx)

    # build test matrix
    test_users = candidate_users[test_idx]
    test_items = candidate_items[test_idx]
    test_data = candidate_data[test_idx]
    test_mat = coo_matrix((test_data, (test_users, test_items)), shape=mat.shape)

    # build training matrix
    train_users = np.r_[users[~full_candidate_mask], candidate_users[train_idx]]
    train_items = np.r_[items[~full_candidate_mask], candidate_items[train_idx]]
    train_data = np.r_[data[~full_candidate_mask], candidate_data[train_idx]]
    train_mat = coo_matrix((train_data, (train_users, train_items)), shape=mat.shape)

    return train_mat, test_mat



@cython.boundscheck(False)
def precision_at_k(model, train_user_items, test_user_items, int K=10,
                   show_progress=True, int num_threads=1):
    """ Calculates P@K for a given trained model

    Parameters
    ----------
    model : RecommenderBase
        The fitted recommendation model to test
    train_user_items : csr_matrix
        Sparse matrix of user by item that contains elements that were used
            in training the model
    test_user_items : csr_matrix
        Sparse matrix of user by item that contains withheld elements to
        test on
    K : int
        Number of items to test on
    show_progress : bool, optional
        Whether to show a progress bar
    num_threads : int, optional
        The number of threads to use for testing. Specifying 0 means to default
        to the number of cores on the machine. Note: aside from the ALS and BPR
        models, setting this to more than 1 will likely hurt performance rather than
        help.

    Returns
    -------
    float
        the calculated p@k
    """
    return ranking_metrics_at_k(
        model, train_user_items, test_user_items, K, show_progress, num_threads)['precision']


@cython.boundscheck(False)
def mean_average_precision_at_k(model, train_user_items, test_user_items, int K=10,
                                show_progress=True, int num_threads=1):
    """ Calculates MAP@K for a given trained model

    Parameters
    ----------
    model : RecommenderBase
        The fitted recommendation model to test
    train_user_items : csr_matrix
        Sparse matrix of user by item that contains elements that were used in training the model
    test_user_items : csr_matrix
        Sparse matrix of user by item that contains withheld elements to test on
    K : int
        Number of items to test on
    show_progress : bool, optional
        Whether to show a progress bar
    num_threads : int, optional
        The number of threads to use for testing. Specifying 0 means to default
        to the number of cores on the machine. Note: aside from the ALS and BPR
        models, setting this to more than 1 will likely hurt performance rather than
        help.

    Returns
    -------
    float
        the calculated MAP@k
    """
    return ranking_metrics_at_k(
        model, train_user_items, test_user_items, K, show_progress, num_threads)['map']


@cython.boundscheck(False)
def ndcg_at_k(model, train_user_items, test_user_items, int K=10,
              show_progress=True, int num_threads=1):
    """ Calculates ndcg@K for a given trained model

    Parameters
    ----------
    model : RecommenderBase
        The fitted recommendation model to test
    train_user_items : csr_matrix
        Sparse matrix of user by item that contains elements that were used in training the model
    test_user_items : csr_matrix
        Sparse matrix of user by item that contains withheld elements to test on
    K : int
        Number of items to test on
    show_progress : bool, optional
        Whether to show a progress bar
    num_threads : int, optional
        The number of threads to use for testing. Specifying 0 means to default
        to the number of cores on the machine. Note: aside from the ALS and BPR
        models, setting this to more than 1 will likely hurt performance rather than
        help.

    Returns
    -------
    float
        the calculated ndcg@k
    """
    return ranking_metrics_at_k(
        model, train_user_items, test_user_items, K, show_progress, num_threads)['ndcg']


@cython.boundscheck(False)
def AUC_at_k(model, train_user_items, test_user_items, int K=10,
             show_progress=True, int num_threads=1):
    """ Calculate limited AUC for a given trained model

    Parameters
    ----------
    model : RecommenderBase
        The fitted recommendation model to test
    train_user_items : csr_matrix
        Sparse matrix of user by item that contains elements that were used in training the model
    test_user_items : csr_matrix
        Sparse matrix of user by item that contains withheld elements to test on
    K : int
        Number of items to test on
    show_progress : bool, optional
        Whether to show a progress bar
    num_threads : int, optional
        The number of threads to use for testing. Specifying 0 means to default
        to the number of cores on the machine. Note: aside from the ALS and BPR
        models, setting this to more than 1 will likely hurt performance rather than
        help.

    Returns
    -------
    float
        the calculated ndcg@k
    """
    return ranking_metrics_at_k(
        model, train_user_items, test_user_items, K, show_progress, num_threads)['auc']


@cython.boundscheck(False)
def ranking_metrics_at_k(model, train_user_items, test_user_items, int K=10,
                         show_progress=True, int num_threads=1):
    """ Calculates ranking metrics for a given trained model

    Parameters
    ----------
    model : RecommenderBase
        The fitted recommendation model to test
    train_user_items : csr_matrix
        Sparse matrix of user by item that contains elements that were used
            in training the model
    test_user_items : csr_matrix
        Sparse matrix of user by item that contains withheld elements to
        test on
    K : int
        Number of items to test on
    show_progress : bool, optional
        Whether to show a progress bar
    num_threads : int, optional
        The number of threads to use for testing. Specifying 0 means to default
        to the number of cores on the machine. Note: aside from the ALS and BPR
        models, setting this to more than 1 will likely hurt performance rather than
        help.

    Returns
    -------
    float
        the calculated p@k
    """
    if not isinstance(train_user_items, csr_matrix):
        train_user_items = train_user_items.tocsr()

    if not isinstance(test_user_items, csr_matrix):
        test_user_items = test_user_items.tocsr()

    cdef int users = test_user_items.shape[0], items = test_user_items.shape[1]
    cdef int u, i
    # precision
    cdef double relevant = 0, pr_div = 0, total = 0
    # map
    cdef double mean_ap = 0, ap = 0
    # ndcg
    cdef double[:] cg = (1.0 / np.log2(np.arange(2, K + 2)))
    cdef double[:] cg_sum = np.cumsum(cg)
    cdef double ndcg = 0, idcg
    # auc
    cdef double mean_auc = 0, auc, hit, miss, num_pos_items, num_neg_items

    cdef int[:] test_indptr = test_user_items.indptr
    cdef int[:] test_indices = test_user_items.indices

    cdef int * ids
    cdef unordered_set[int] * likes

    progress = tqdm(total=users, disable=not show_progress)

    with nogil, parallel(num_threads=num_threads):
        ids = <int * > malloc(sizeof(int) * K)
        likes = new unordered_set[int]()
        try:
            for u in prange(users, schedule='guided'):
                # if we don't have any test items, skip this user
                if test_indptr[u] == test_indptr[u+1]:
                    with gil:
                        progress.update(1)
                    continue
                memset(ids, -1, sizeof(int) * K)

                with gil:
                    recs = model.recommend(u, train_user_items, N=K)
                    for i in range(len(recs)):
                        ids[i] = recs[i][0]
                    progress.update(1)

                # mostly we're going to be blocked on the gil here,
                # so try to do actual scoring without it
                likes.clear()
                for i in range(test_indptr[u], test_indptr[u+1]):
                    likes.insert(test_indices[i])

                pr_div += fmin(K, likes.size())
                ap = 0
                hit = 0
                miss = 0
                auc = 0
                idcg = cg_sum[min(K, likes.size()) - 1]
                num_pos_items = likes.size()
                num_neg_items = items - num_pos_items

                for i in range(K):
                    if likes.find(ids[i]) != likes.end():
                        relevant += 1
                        hit += 1
                        ap += hit / (i + 1)
                        ndcg += cg[i] / idcg
                    else:
                        miss += 1
                        auc += hit
                auc += ((hit + num_pos_items) / 2.0) * (num_neg_items - miss)
                mean_ap += ap / fmin(K, likes.size())
                mean_auc += auc / (num_pos_items * num_neg_items)
                total += 1
        finally:
            free(ids)
            del likes

    progress.close()
    return {
        "precision": relevant / pr_div,
        "map": mean_ap / total,
        "ndcg": ndcg / total,
        "auc": mean_auc / total
    }

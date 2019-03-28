# distutils: language = c++
# cython: language_level=3

import tqdm
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
import cython
from cython.operator import dereference
from cython.parallel import parallel, prange
from libc.stdlib cimport malloc, free
from libc.string cimport memset
from libc.math cimport fmin

from libcpp.unordered_set cimport unordered_set

from math import ceil
cdef extern from "topnc.h":
    cdef void fargsort_c(float A[], int n_row, int m_row, int m_cols, int ktop, int B[]) nogil


def train_test_split(ratings, train_percentage=0.8):
    """ Randomly splits the ratings matrix into two matrices for training/testing.

    Parameters
    ----------
    ratings : coo_matrix
        A sparse matrix to split
    train_percentage : float
        What percentage of ratings should be used for training

    Returns
    -------
    (train, test) : csr_matrix, csr_matrix
        A tuple of csr_matrices for training/testing """
    ratings = ratings.tocoo()

    random_index = np.random.random(len(ratings.data))
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
    if not isinstance(train_user_items, csr_matrix):
        train_user_items = train_user_items.tocsr()

    if not isinstance(test_user_items, csr_matrix):
        test_user_items = test_user_items.tocsr()

    cdef int users = test_user_items.shape[0], u, i
    cdef double relevant = 0, total = 0
    cdef int[:] test_indptr = test_user_items.indptr
    cdef int[:] test_indices = test_user_items.indices

    cdef int * ids
    cdef unordered_set[int] * likes

    progress = tqdm.tqdm(total=users, disable=not show_progress)

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
                memset(ids, 0, sizeof(int) * K)

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

                total += fmin(K, likes.size())

                for i in range(K):
                    if likes.find(ids[i]) != likes.end():
                        relevant += 1
        finally:
            free(ids)
            del likes

    progress.close()
    return relevant / total


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
    # TODO: there is a fair amount of boilerplate here that is cut and paste
    # from precision_at_k. refactor it out.
    if not isinstance(train_user_items, csr_matrix):
        train_user_items = train_user_items.tocsr()

    if not isinstance(test_user_items, csr_matrix):
        test_user_items = test_user_items.tocsr()

    cdef int users = test_user_items.shape[0], u, i, total = 0
    cdef double mean_ap = 0, ap = 0, relevant = 0
    cdef int[:] test_indptr = test_user_items.indptr
    cdef int[:] test_indices = test_user_items.indices

    cdef int * ids
    cdef unordered_set[int] * likes

    progress = tqdm.tqdm(total=users, disable=not show_progress)

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
                memset(ids, 0, sizeof(int) * K)

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

                ap = 0
                relevant = 0
                for i in range(K):
                    if likes.find(ids[i]) != likes.end():
                        relevant = relevant + 1
                        ap = ap + relevant / (i + 1)
                mean_ap += ap / fmin(K, likes.size())
                total += 1
        finally:
            free(ids)
            del likes

    progress.close()
    return mean_ap / total


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def ALS_recommend_all(model, users_items, int k=10, int threads=1, show_progress=True,
                      recalculate_user=False, filter_already_liked_items=False):

    if not isinstance(users_items, csr_matrix):
        users_items = users_items.tocsr()
    factors_items = model.item_factors.T

    cdef:
        int users_c = users_items.shape[0], items_c = users_items.shape[1], batch = threads * 10
        int u_b, u_low, u_high, u_len, u
    A = np.zeros((batch, items_c), dtype=np.float32)
    cdef:
        int users_c_b = ceil(users_c / batch)
        float[:, ::1] A_mv = A
        float * A_mv_p = &A_mv[0, 0]
        int[:, ::1] B_mv = np.zeros((users_c, k), dtype=np.intc)
        int * B_mv_p = &B_mv[0, 0]

    progress = tqdm.tqdm(total=users_c, disable=not show_progress)
    for u_b in range(users_c_b):
        u_low = u_b * batch
        u_high = min([(u_b + 1) * batch, users_c])
        u_len = u_high - u_low
        users_factors = np.vstack([
            model._user_factor(u, users_items, recalculate_user)
            for u
            in range(u_low, u_high, 1)
        ]).astype(np.float32)
        users_factors.dot(factors_items, out=A[:u_len])
        if filter_already_liked_items:
            A[users_items[u_low:u_high].nonzero()] = 0
        for u in prange(u_len, nogil=True, num_threads=threads, schedule='dynamic'):
            fargsort_c(A_mv_p, u, batch * u_b + u, items_c, k, B_mv_p)
        progress.update(u_len)
    progress.close()
    return np.asarray(B_mv)

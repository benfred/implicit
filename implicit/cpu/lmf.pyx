import cython

from cython cimport floating, integral

import logging
import multiprocessing
import time

from cython.parallel import parallel, prange, threadid
from tqdm.auto import tqdm

from libc.math cimport exp, sqrt
from libc.stdlib cimport free, malloc
from libc.string cimport memcpy, memset
from libcpp cimport bool
from libcpp.algorithm cimport binary_search

import random

import numpy as np
import scipy.sparse

from libcpp.vector cimport vector

from ..utils import check_csr, check_random_state
from .matrix_factorization_base import MatrixFactorizationBase

log = logging.getLogger("implicit")


cdef extern from "<random>" namespace "std":
    cdef cppclass mt19937:
        mt19937(unsigned int)

    cdef cppclass uniform_int_distribution[T]:
        uniform_int_distribution(T, T)
        T operator()(mt19937) nogil


cdef class RNGVector(object):
    """ This class creates one c++ rng object per thread, and enables us to randomly sample
    liked/disliked items here in a thread safe manner """
    cdef vector[mt19937] rng
    cdef vector[uniform_int_distribution[long]] dist

    def __init__(self, int num_threads, long rows, long[:] rng_seeds):
        if len(rng_seeds) != num_threads:
            raise ValueError("length of RNG seeds must be equal to num_threads")

        cdef int i
        for i in range(num_threads):
            self.rng.push_back(mt19937(rng_seeds[i]))
            self.dist.push_back(uniform_int_distribution[long](0, rows))

    cdef inline long generate(self, int thread_id) noexcept nogil:
        return self.dist[thread_id](self.rng[thread_id])


class LogisticMatrixFactorization(MatrixFactorizationBase):
    """ Logistic Matrix Factorization

    A collaborative filtering recommender model that learns probabilistic distribution
    whether user like it or not. Algorithm of the model is described in
    `Logistic Matrix Factorization for Implicit Feedback Data
    <https://web.stanford.edu/~rezab/nips2014workshop/submits/logmat.pdf>`

    Parameters
    ----------
    factors : int, optional
        The number of latent factors to compute
    learning_rate : float, optional
        The learning rate to apply for updates during training
    regularization : float, optional
        The regularization factor to use
    dtype : data-type, optional
        Specifies whether to generate 64 bit or 32 bit floating point factors
    iterations : int, optional
        The number of training epochs to use when fitting the data
    neg_prop : int, optional
        The proportion of negative samples. i.e.) "neg_prop = 30" means if user have seen 5 items,
        then 5 * 30 = 150 negative samples are used for training.
    num_threads : int, optional
        The number of threads to use for fitting the model and batch recommendation calls.
        Specifying 0 means to default to the number of cores on the machine.
    random_state : int, RandomState or None, optional
        The random state for seeding the initial item and user factors.
        Default is None.

    Attributes
    ----------
    item_factors : ndarray
        Array of latent factors for each item in the training set
    user_factors : ndarray
        Array of latent factors for each user in the training set
    """

    def __init__(self, factors=30, learning_rate=1.00, regularization=0.6, dtype=np.float32,
                 iterations=30, neg_prop=30, num_threads=0,
                 random_state=None):
        super(LogisticMatrixFactorization, self).__init__(num_threads=num_threads)

        self.factors = factors
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.regularization = regularization
        self.dtype = np.dtype(dtype)
        self.neg_prop = neg_prop
        self.random_state = random_state

    @cython.cdivision(True)
    @cython.boundscheck(False)
    def fit(self, user_items, show_progress=True, callback=None):
        """ Factorizes the user_items matrix

        Parameters
        ----------
        user_items: csr_matrix
            Matrix of confidences for the liked items. This matrix should be a csr_matrix where
            the rows of the matrix are the user, and the columns are the items that are liked by
            the user.
            BPR ignores the weight value of the matrix right now - it treats non zero entries
            as a binary signal that the user liked the item.
        show_progress : bool, optional
            Whether to show a progress bar
        callback: Callable, optional
            Callable function on each epoch with such arguments as epoch, elapsed time and progress
        """
        rs = check_random_state(self.random_state)

        # for now, all we handle is float 32 values
        if user_items.dtype != np.float32:
            user_items = user_items.astype(np.float32)

        users, items = user_items.shape

        user_items = check_csr(user_items)
        item_users = user_items.T.tocsr()

        if not item_users.has_sorted_indices:
            item_users.sort_indices()
        if not user_items.has_sorted_indices:
            user_items.sort_indices()

        # this basically calculates the 'row' attribute of a COO matrix
        # without requiring us to get the whole COO matrix
        user_counts = np.ediff1d(user_items.indptr)
        item_counts = np.bincount(user_items.indices, minlength=items)

        # Reserve last two elements of user factors, and item factors to be bias.
        # user_factors[-1] = 1, item_factors[-2] = 1
        # user_factors[-2] = user bias, item factors[-1] = item bias
        # This significantly simplifies both training, and serving
        if self.item_factors is None:
            self.item_factors = rs.standard_normal(size=(items, self.factors + 2), dtype=np.float32)
            self.item_factors[:, -1] = 1.0

            # set factors to all zeros for items without any ratings
            self.item_factors[item_counts == 0] = np.zeros(self.factors + 2)

        if self.user_factors is None:
            self.user_factors = rs.standard_normal(size=(users, self.factors + 2), dtype=np.float32)
            self.user_factors[:, -2] = 1.0

            # set factors to all zeros for users without any ratings
            self.user_factors[user_counts == 0] = np.zeros(self.factors + 2)

        # For Adagrad update
        user_vec_deriv_sum = np.zeros((users, self.factors + 2)).astype(np.float32)
        item_vec_deriv_sum = np.zeros((items, self.factors + 2)).astype(np.float32)

        cdef int num_threads = self.num_threads
        if not num_threads:
            num_threads = multiprocessing.cpu_count()

        # initialize RNG's, one per thread. Also pass the seeds for each thread's RNG
        cdef long[:] rng_seeds = rs.integers(0, 2**31, size=num_threads, dtype="long")
        cdef RNGVector rng = RNGVector(num_threads, len(user_items.data) - 1, rng_seeds)

        log.debug("Running %i LMF training epochs", self.iterations)
        with tqdm(total=self.iterations, disable=not show_progress) as progress:
            for epoch in range(self.iterations):
                s = time.time()
                # user update
                lmf_update(rng, user_vec_deriv_sum,
                           self.user_factors, self.item_factors,
                           user_items.indices, user_items.indptr, user_items.data,
                           self.learning_rate, self.regularization, self.neg_prop, num_threads)
                self.user_factors[:, -2] = 1.0
                # item update
                lmf_update(rng, item_vec_deriv_sum,
                           self.item_factors, self.user_factors,
                           item_users.indices, item_users.indptr, item_users.data,
                           self.learning_rate, self.regularization, self.neg_prop, num_threads)
                self.item_factors[:, -1] = 1.0
                progress.update(1)
                if callback:
                    callback(epoch, time.time() - s)

        self._check_fit_errors()

    def save(self, fileobj_or_path):
        args = dict(
            user_factors=self.user_factors,
            item_factors=self.item_factors,
            regularization=self.regularization,
            factors=self.factors,
            learning_rate=self.learning_rate,
            neg_prop=self.neg_prop,
            num_threads=self.num_threads,
            iterations=self.iterations,
            dtype=self.dtype.name,
            random_state=self.random_state)

        # filter out 'None' valued args, since we can't go np.load on
        # them without using pickle
        args = {k:v for k,v in args.items() if v is not None}
        np.savez(fileobj_or_path, **args)


@cython.cdivision(True)
cdef inline floating sigmoid(floating x) noexcept nogil:
    if x >= 0:
        return 1 / (1+exp(-x))
    else:
        z = exp(x)
        return z / (1 + z)


@cython.cdivision(True)
@cython.boundscheck(False)
def lmf_update(RNGVector rng, floating[:, :] deriv_sum_sq,
               floating[:, :] user_vectors, floating[:, :] item_vectors,
               integral[:] indices, integral[:] indptr, floating[:] data,
               floating lr, floating reg, integral neg_prop,
               integral num_threads):

    cdef integral n_users = user_vectors.shape[0]
    cdef integral n_items = item_vectors.shape[1]
    cdef integral n_factors = user_vectors.shape[1]

    cdef integral u, i, it, c, _, index, f
    cdef integral thread_id
    cdef floating* deriv
    cdef floating score, z, temp
    cdef floating exp_r
    cdef int user_seen_item

    with nogil, parallel(num_threads=num_threads):
        deriv = <floating*> malloc(sizeof(floating) * n_factors)
        thread_id = threadid()
        try:
            for u in prange(n_users, schedule='static'):
                if indptr[u] == indptr[u + 1]:
                    continue
                user_seen_item = indptr[u + 1] - indptr[u]
                memset(deriv, 0, sizeof(floating) * n_factors)

                # Positive item indices: c_ui* y_i
                for index in range(indptr[u], indptr[u + 1]):
                    i = indices[index]
                    for _ in range(n_factors):
                        deriv[_] += data[index] * item_vectors[i, _]

                # Positive Item Indices (c_ui * exp(y_ui)) / (1 + exp(y_ui)) * y_i
                for index in range(indptr[u], indptr[u + 1]):
                    exp_r = 0
                    i = indices[index]
                    for _ in range(n_factors):
                        exp_r += user_vectors[u, _] * item_vectors[i, _]
                    z = sigmoid(exp_r) * data[index]
                    for _ in range(n_factors):
                        deriv[_] = deriv[_] - z * item_vectors[i, _]

                # Negative(Sampled) Item Indices exp(y_ui) / (1 + exp(y_ui)) * y_i
                for _ in range(min(n_items, user_seen_item * neg_prop)):
                    index = rng.generate(thread_id)
                    i = indices[index]
                    exp_r = 0
                    for _ in range(n_factors):
                        exp_r = exp_r + (user_vectors[u, _] * item_vectors[i, _])
                    z = sigmoid(exp_r)

                    for _ in range(n_factors):
                        deriv[_] = deriv[_] - z * item_vectors[i, _]
                for _ in range(n_factors):
                    deriv[_] -= reg * user_vectors[u, _]
                    deriv_sum_sq[u, _] += deriv[_] * deriv[_]

                    # a small constant is added for numerical stability
                    user_vectors[u, _] += (lr / (sqrt(1e-6 + deriv_sum_sq[u, _]))) * deriv[_]
        finally:
            free(deriv)

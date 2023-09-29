import cython

from cython cimport floating, integral

import logging
import multiprocessing
import time

from cython.parallel import parallel, prange
from tqdm.auto import tqdm

from libc.math cimport exp
from libcpp cimport bool
from libcpp.algorithm cimport binary_search

import random

import numpy as np
import scipy.sparse

from libcpp.vector cimport vector

from ..utils import check_csr, check_random_state
from .matrix_factorization_base import MatrixFactorizationBase

log = logging.getLogger("implicit")

# thin wrapper around omp_get_thread_num (since referencing directly will cause OSX
# build to fail)
cdef extern from "implicit/cpu/bpr.h" namespace "implicit" nogil:
    cdef int get_thread_num()


cdef extern from "<random>" namespace "std":
    cdef cppclass mt19937:
        mt19937(unsigned int)

    cdef cppclass uniform_int_distribution[T]:
        uniform_int_distribution(T, T)
        T operator()(mt19937) noexcept nogil


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


@cython.boundscheck(False)
cdef bool has_non_zero(integral[:] indptr, integral[:] indices,
                       integral rowid, integral colid) noexcept nogil:
    """ Given a CSR matrix, returns whether the [rowid, colid] contains a non zero.
    Assumes the CSR matrix has sorted indices """
    return binary_search(&indices[indptr[rowid]], &indices[indptr[rowid + 1]], colid)


class BayesianPersonalizedRanking(MatrixFactorizationBase):
    """ Bayesian Personalized Ranking

    A recommender model that learns  a matrix factorization embedding based off minimizing the
    pairwise ranking loss described in the paper `BPR: Bayesian Personalized Ranking from Implicit
    Feedback <https://arxiv.org/pdf/1205.2618.pdf>`_.

    Parameters
    ----------
    factors : int, optional
        The number of latent factors to compute
    learning_rate : float, optional
        The learning rate to apply for SGD updates during training
    regularization : float, optional
        The regularization factor to use
    dtype : data-type, optional
        Specifies whether to generate 64 bit or 32 bit floating point factors
    iterations : int, optional
        The number of training epochs to use when fitting the data
    verify_negative_samples: bool, optional
        When sampling negative items, check if the randomly picked negative item has actually
        been liked by the user. This check increases the time needed to train but usually leads
        to better predictions.
    num_threads : int, optional
        The number of threads to use for fitting the model and batch recommend calls.
        Specifying 0 means to default to the number of cores on the machine.
    random_state : int, RandomState, Generator or None, optional
        The random state for seeding the initial item and user factors.
        Default is None.

    Attributes
    ----------
    item_factors : ndarray
        Array of latent factors for each item in the training set
    user_factors : ndarray
        Array of latent factors for each user in the training set
    """
    def __init__(self, factors=100, learning_rate=0.01, regularization=0.01, dtype=np.float32,
                 iterations=100, num_threads=0,
                 verify_negative_samples=True, random_state=None):
        super(BayesianPersonalizedRanking, self).__init__(num_threads=num_threads)

        self.factors = factors
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.regularization = regularization
        self.dtype = np.dtype(dtype)
        self.verify_negative_samples = verify_negative_samples
        self.random_state = random_state

    @cython.cdivision(True)
    @cython.boundscheck(False)
    def fit(self, user_items, show_progress=True, callback=None):
        """ Factorizes the user_items matrix

        Parameters
        ----------
        user_items: csr_matrix
            Matrix of confidences for the liked items. This matrix should be a csr_matrix where
            the rows of the matrix are the user, and the columns are the items liked by that user.
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

        user_items = check_csr(user_items)
        users, items = user_items.shape

        # We need efficient user lookup for case of removing own likes
        if self.verify_negative_samples and not user_items.has_sorted_indices:
            user_items.sort_indices()

        # this basically calculates the 'row' attribute of a COO matrix
        # without requiring us to get the whole COO matrix
        user_counts = np.ediff1d(user_items.indptr)
        userids = np.repeat(np.arange(users), user_counts).astype(user_items.indices.dtype)

        # create factors if not already created.
        # Note: the final dimension is for the item bias term - which is set to a 1 for all users
        # this simplifies interfacing with approximate nearest neighbours libraries etc
        if self.item_factors is None:
            self.item_factors = (rs.random((items, self.factors + 1), dtype=self.dtype) - .5)
            self.item_factors /= self.factors

            # set factors to all zeros for items without any ratings
            item_counts = np.bincount(user_items.indices, minlength=items)
            self.item_factors[item_counts == 0] = np.zeros(self.factors + 1)

        if self.user_factors is None:
            self.user_factors = (rs.random((users, self.factors + 1), dtype=self.dtype) - .5)
            self.user_factors /= self.factors

            # set factors to all zeros for users without any ratings
            self.user_factors[user_counts == 0] = np.zeros(self.factors + 1)

            self.user_factors[:, self.factors] = 1.0

        # invalidate cached norms
        self._user_norms = self._item_norms = None

        # we accept num_threads = 0 as indicating to create as many threads as we have cores,
        # but in that case we need the number of cores, since we need to initialize RNG state per
        # thread. Get the appropriate value back from openmp
        cdef int num_threads = self.num_threads
        if not num_threads:
            num_threads = multiprocessing.cpu_count()

        # initialize RNG's, one per thread. Also pass the seeds for each thread's RNG
        cdef long[:] rng_seeds = rs.integers(0, 2**31, size=num_threads, dtype="long")
        cdef RNGVector rng = RNGVector(num_threads, len(user_items.data) - 1, rng_seeds)

        log.debug("Running %i BPR training epochs", self.iterations)
        with tqdm(total=self.iterations, disable=not show_progress) as progress:
            for epoch in range(self.iterations):
                s = time.time()
                correct, skipped = bpr_update(rng, userids, user_items.indices, user_items.indptr,
                                              self.user_factors, self.item_factors,
                                              self.learning_rate, self.regularization, num_threads,
                                              self.verify_negative_samples)
                progress.update(1)
                total = len(user_items.data)
                if total != 0 and total != skipped:
                    progress.set_postfix(
                        {"train_auc": "%.2f%%" % (100.0 * correct / (total - skipped)),
                         "skipped": "%.2f%%" % (100.0 * skipped / total)})

                if callback:
                    callback(epoch, time.time() - s, correct, skipped)

        self._check_fit_errors()

    def to_gpu(self) -> "implicit.gpu.bpr.BayesianPersonalizedRanking":
        """Converts this model to an equivalent version running on the gpu"""
        import implicit.gpu.bpr

        ret = implicit.gpu.bpr.BayesianPersonalizedRanking(
            factors=self.factors,
            learning_rate=self.learning_rate,
            regularization=self.regularization,
            iterations=self.iterations,
            verify_negative_samples=self.verify_negative_samples,
            random_state=self.random_state,
        )

        if self.user_factors is not None:
            ret.user_factors = implicit.gpu.Matrix(self.user_factors)
        if self.item_factors is not None:
            ret.item_factors = implicit.gpu.Matrix(self.item_factors)
        return ret

    def save(self, fileobj_or_path):
        args = dict(user_factors=self.user_factors,
            item_factors=self.item_factors,
            regularization=self.regularization,
            factors=self.factors,
            learning_rate=self.learning_rate,
            verify_negative_samples=self.verify_negative_samples,
            num_threads=self.num_threads,
            iterations=self.iterations,
            dtype=self.dtype.name,
            random_state=self.random_state
        )

        # filter out 'None' valued args, since we can't go np.load on
        # them without using pickle
        args = {k:v for k,v in args.items() if v is not None}
        np.savez(fileobj_or_path, **args)


@cython.cdivision(True)
@cython.boundscheck(False)
def bpr_update(RNGVector rng,
               integral[:] userids, integral[:] itemids, integral[:] indptr,
               floating[:, :] X, floating[:, :] Y,
               float learning_rate, float reg, int num_threads,
               bool verify_neg):
    cdef integral users = X.shape[0], items = Y.shape[0]
    cdef long samples = len(userids), i, liked_index, disliked_index, correct = 0, skipped = 0
    cdef integral j, liked_id, disliked_id, thread_id
    cdef floating z, score, temp

    cdef floating * user
    cdef floating * liked
    cdef floating * disliked

    cdef integral factors = X.shape[1] - 1

    with nogil, parallel(num_threads=num_threads):

        thread_id = get_thread_num()
        for i in prange(samples, schedule='static'):
            liked_index = rng.generate(thread_id)
            liked_id = itemids[liked_index]

            # if the user has liked the item, skip this for now
            disliked_index = rng.generate(thread_id)
            disliked_id = itemids[disliked_index]

            if verify_neg and has_non_zero(indptr, itemids, userids[liked_index], disliked_id):
                skipped += 1
                continue

            # get pointers to the relevant factors
            user, liked, disliked = &X[userids[liked_index], 0], &Y[liked_id, 0], &Y[disliked_id, 0]

            # compute the score
            score = 0
            for j in range(factors + 1):
                score = score + user[j] * (liked[j] - disliked[j])
            z = 1.0 / (1.0 + exp(score))

            if z < .5:
                correct += 1

            # update the factors via sgd.
            for j in range(factors):
                temp = user[j]
                user[j] += learning_rate * (z * (liked[j] - disliked[j]) - reg * user[j])
                liked[j] += learning_rate * (z * temp - reg * liked[j])
                disliked[j] += learning_rate * (-z * temp - reg * disliked[j])

            # update item bias terms (last column of factorized matrix)
            liked[factors] += learning_rate * (z - reg * liked[factors])
            disliked[factors] += learning_rate * (-z - reg * disliked[factors])

    return correct, skipped

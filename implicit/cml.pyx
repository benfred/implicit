import itertools
import logging
import multiprocessing
import random
import time

import cython
import numpy as np
import scipy.sparse
import tqdm
from cython.parallel import parallel, prange, threadid
from cython cimport floating, integral
from libc.math cimport exp
from libc.math cimport sqrt
from libc.math cimport log10

from libcpp cimport bool
from libcpp.algorithm cimport binary_search
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy, memset
from libcpp.vector cimport vector

from .recommender_base import MatrixFactorizationBase
from .utils import check_random_state

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

    cdef inline long generate(self, int thread_id) nogil:
        return self.dist[thread_id](self.rng[thread_id])


@cython.boundscheck(False)
cdef bool has_non_zero(integral[:] indptr, integral[:] indices,
                       integral rowid, integral colid) nogil:
    """ Given a CSR matrix, returns whether the [rowid, colid] contains a non zero.
    Assumes the CSR matrix has sorted indices """
    return binary_search(&indices[indptr[rowid]], &indices[indptr[rowid + 1]], colid)


class CollaborativeMetricLearning(MatrixFactorizationBase):
    """ Collaborative Metric Learning
    A collaborative filtering recommender model that learns
    to embed user and item factors into euclidean metric space,
    and recommend items to users by their euclidean distance.
    Algorithm of the model is described in
    `Collaborative Metric Learning
    <http://www.cs.cornell.edu/~ylongqi/paper/HsiehYCLBE17.pdf>`

    Parameters
    ----------
    factors : int, optional
        The number of latent factors to compute
    learning_rate : float, optional
        The learning rate to apply for updates during training
    regularization : float, optional
        The regularization factor to use.
        This model exploits cogswell covariance loss
        `<https://arxiv.org/abs/1511.06068>`.
        Currently Not implemented yet.
    dtype : data-type, optional
        Specifies whether to generate 64 bit or 32 bit floating point factors
    iterations : int, optional
        The number of training epochs to use when fitting the data
    neg_sampling : int, optional
        The size of negative sampling estimating ranking weight.
    threshold : float, optional
        The threshold to find violation in the personalized ranking.
        See `<http://www.cs.cornell.edu/~ylongqi/paper/HsiehYCLBE17.pdf>`
        for reference.
    use_gpu : bool, optional
        Fit on the GPU if available
    num_threads : int, optional
        The number of threads to use for fitting the model. This only
        applies for the native extensions. Specifying 0 means to default
        to the number of cores on the machine.
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
    def __init__(self, factors=50, learning_rate=0.10, regularization=0.0, dtype=np.float32,
                 iterations=30, neg_sampling=100, threshold=1.0, use_gpu=False, num_threads=0,
                 random_state=None):
        super(CollaborativeMetricLearning, self).__init__()

        self.factors = factors
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.regularization = regularization
        self.dtype = dtype
        self.threshold = threshold
        self.use_gpu = use_gpu
        self.num_threads = num_threads
        self.neg_sampling = neg_sampling
        self.random_state = random_state

        # TODO: Add GPU training
        if self.use_gpu:
            raise NotImplementedError("GPU version of LMF is not implemeneted yet!")

    @cython.cdivision(True)
    @cython.boundscheck(False)
    def fit(self, item_users, show_progress=True):
        """ Factorizes the item_users matrix

        Parameters
        ----------
        item_users: coo_matrix
            Matrix of confidences for the liked items. This matrix should be a coo_matrix where
            the rows of the matrix are the item, and the columns are the users that liked that item.
            BPR ignores the weight value of the matrix right now - it treats non zero entries
            as a binary signal that the user liked the item.
        show_progress : bool, optional
            Whether to show a progress bar
        """
        rs = check_random_state(self.random_state)

        # for now, all we handle is float 32 values
        if item_users.dtype != np.float32:
            item_users = item_users.astype(np.float32)

        items, users = item_users.shape

        item_users = item_users.tocsr()
        user_items = item_users.T.tocsr()

        if not item_users.has_sorted_indices:
            item_users.sort_indices()
        if not user_items.has_sorted_indices:
            user_items.sort_indices()

        # it requires twice more memory but, is efficient for training CML model.
        user_items_coo = user_items.tocoo()

        # this basically calculates the 'row' attribute of a COO matrix
        # without requiring us to get the whole COO matrix
        user_counts = np.ediff1d(user_items.indptr)
        item_counts = np.bincount(user_items.indices, minlength=items)

        if self.item_factors is None:
            self.item_factors = rs.normal(scale=1.0 / (self.factors ** 0.5),
                                          size=(items, self.factors)).astype(np.float32)
            # set factors to all zeros for items without any ratings
            self.item_factors[item_counts == 0] = np.zeros(self.factors)

        if self.user_factors is None:
            self.user_factors = rs.normal(scale=1.0 / (self.factors ** 0.5),
                                          size=(users, self.factors)).astype(np.float32)

            # set factors to all zeros for users without any ratings
            self.user_factors[user_counts == 0] = np.zeros(self.factors)

        # For Adagrad update
        user_vec_deriv_sum = np.zeros((users, self.factors)).astype(np.float32)
        item_vec_deriv_sum = np.zeros((items, self.factors)).astype(np.float32)

        cdef int num_threads = self.num_threads
        if not num_threads:
            num_threads = multiprocessing.cpu_count()

        # initialize RNG's
        # It utilizes two RNG. First for sampling items uniformly,
        # and second for sampling items proportional to their popularity.
        cdef long[:] rng_seeds = rs.randint(0, 2**31, size=num_threads)
        unique_items = np.unique(user_items_coo.col)
        cdef RNGVector rng_items = RNGVector(num_threads, len(unique_items) - 1, rng_seeds)
        cdef RNGVector rng_coo = RNGVector(num_threads, len(user_items.data) - 1, rng_seeds)

        log.debug("Running %i LMF training epochs", self.iterations)
        with tqdm.tqdm(total=self.iterations, disable=not show_progress) as progress:
            for epoch in range(self.iterations):
                t = cml_update(rng_items, rng_coo, unique_items,
                               user_vec_deriv_sum, item_vec_deriv_sum,
                               self.user_factors, self.item_factors,
                               user_items.indices, user_items.indptr, user_items.data,
                               user_items_coo.row, user_items_coo.col,
                               self.threshold, self.learning_rate, self.regularization,
                               self.neg_sampling, num_threads)
                progress.update(1)

        self.user_factors[user_counts == 0] = 10000.0 * np.ones(self.factors)
        self.item_factors[item_counts == 0] = 10000.0 * np.ones(self.factors)
        self._check_fit_errors()

    def similar_users(self, userid, N=10):
        factor = self.user_factors[userid]
        factors = self.user_factors
        return self._get_similarity_score(factor, factors, N)

    def similar_items(self, itemid, N=10, react_users=None, recalculate_item=False):
        factor = self._item_factor(itemid, react_users, recalculate_item)
        factors = self.item_factors
        return self._get_similarity_score(factor, factors, N)

    def recommend(self, userid, user_items,
                  N=10, filter_already_liked_items=True,
                  filter_items=None, recalculate_user=False):
        user = self._user_factor(userid, user_items, recalculate_user)
        liked = set()
        if filter_already_liked_items:
            liked.update(user_items[userid].indices)
        if filter_items:
            liked.update(filter_items)

        # Unlike other factor based recommenders, CML exploits L2 distance as a similarity measure.
        diff = (user - self.item_factors)
        scores = -(diff * diff).sum(1)
        count = N + len(liked)
        if count < len(scores):
            ids = np.argpartition(scores, -count)[-count:]
            best = sorted(zip(ids, scores[ids]), key=lambda x: -x[1])
        else:
            best = sorted(enumerate(scores), key=lambda x: -x[1])
        return list(itertools.islice((rec for rec in best if rec[0] not in liked), N))

    def _get_similarity_score(self, factor, factors, N):
        diff = (factor - factors)
        scores = -(diff * diff).sum(1)
        best = np.argpartition(scores, -N)[-N:]
        return sorted(zip(best, scores[best]), key=lambda x: -x[1])


@cython.cdivision(True)
@cython.boundscheck(False)
def cml_update(RNGVector rng_items, RNGVector rng_coo, integral[:] unique_items,
               floating[:, :] u_deriv_sum_sq, floating[:, :] i_deriv_sum_sq,
               floating[:, :] user_vectors, floating[:, :] item_vectors,
               integral[:] indices, integral[:] indptr, floating[:] data,
               integral[:] row, integral[:] col,
               floating threshold, floating lr, floating reg, integral neg_sampling,
               integral num_threads):
    cdef float loss = 0.0
    cdef integral samples = len(indices)
    cdef integral n_users = user_vectors.shape[0]
    cdef integral n_items = item_vectors.shape[1]
    cdef integral n_factors = user_vectors.shape[1]

    cdef integral u, i, j, it, c, _, __, index, f
    cdef integral sampled_neg_items
    cdef integral thread_id
    cdef floating* u_deriv
    cdef floating* i_deriv
    cdef floating* j_deriv
    cdef floating* cov
    cdef floating* vec_avg

    cdef floating score, z, temp
    cdef int user_seen_item
    cdef floating weight
    cdef floating *uij
    cdef integral* neg_sample_cnts
    cdef floating* tmps
    with nogil, parallel(num_threads=num_threads):
        neg_sample_cnts = <integral*>malloc(sizeof(integral) * num_threads)
        tmps = <floating *> malloc(sizeof(floating) * num_threads)
        cov = <floating*> malloc(sizeof(floating) * n_factors * n_factors)
        vec_avg = <floating*> malloc(sizeof(floating) * n_factors)
        u_deriv = <floating*> malloc(sizeof(floating) * n_factors)
        i_deriv = <floating*> malloc(sizeof(floating) * n_factors)
        j_deriv = <floating*> malloc(sizeof(floating) * n_factors)
        memset(cov, 0, sizeof(floating) * n_factors * n_factors)
        memset(vec_avg, 0, sizeof(floating) * n_factors)
        uij = <floating*> malloc(sizeof(floating) * 2)

        thread_id = threadid()
        try:
            for __ in prange(samples, schedule='static'):
                memset(u_deriv, 0, sizeof(floating) * n_factors)
                memset(i_deriv, 0, sizeof(floating) * n_factors)
                memset(j_deriv, 0, sizeof(floating) * n_factors)
                index = rng_coo.generate(thread_id)
                u, i = row[index], col[index]

                uij[0] = 0
                uij[1] = 0
                for _ in range(n_factors):
                    uij[0] += (user_vectors[u][_] - item_vectors[i][_]) ** 2

                # Sample negative items until the condition is statisfied.
                neg_sample_cnts[thread_id] = 0
                while neg_sample_cnts[thread_id] < neg_sampling:
                    neg_sample_cnts[thread_id] += 1
                    while True:
                        j = unique_items[rng_items.generate(thread_id)]
                        # j should be negative item for user u
                        if not has_non_zero(indptr, indices, u, j):
                            break

                    uij[1] = 0
                    for _ in range(n_factors):
                        uij[1] += (user_vectors[u][_] - item_vectors[j][_]) ** 2

                    # Assume here that j is negative item, that user u has not interacted with j
                    if threshold + uij[0] - uij[1] > 0:
                        break
                if neg_sample_cnts[thread_id] == neg_sampling:
                    # No update
                    continue

                loss += threshold + uij[0] - uij[1]
                weight = log10(1.0 + (n_items // neg_sample_cnts[thread_id]))
                # Factor update
                for _ in range(n_factors):
                    u_deriv[_] = -weight * (item_vectors[i][_] - item_vectors[j][_])
                    u_deriv_sum_sq[u, _] += u_deriv[_] * u_deriv[_]
                for _ in range(n_factors):
                    i_deriv[_] = weight * lr * (item_vectors[i][_] - user_vectors[u][_])
                    i_deriv_sum_sq[i, _] += i_deriv[_] * i_deriv[_]
                for _ in range(n_factors):
                    j_deriv[_] = -weight * (item_vectors[j][_] - user_vectors[u][_])
                    i_deriv_sum_sq[j, _] += j_deriv[_] * j_deriv[_]

                for _ in range(n_factors):
                    user_vectors[u][_] -= (lr / (sqrt(1e-8 + u_deriv_sum_sq[u, _]))) * u_deriv[_]
                    item_vectors[i][_] -= (lr / (sqrt(1e-8 + i_deriv_sum_sq[i, _]))) * i_deriv[_]
                    item_vectors[j][_] -= (lr / (sqrt(1e-8 + i_deriv_sum_sq[j, _]))) * j_deriv[_]
                # 3.4 Add Regularization.
                # How to get this value approximately, and quite easily...?

                # Forcing Updated params in unit sphere
                tmps[thread_id] = 0.0
                for _ in range(n_factors):
                    tmps[thread_id] += user_vectors[u][_] * user_vectors[u][_]

                tmps[thread_id] = max(1.0, tmps[thread_id])
                for _ in range(n_factors):
                    user_vectors[u][_] /= tmps[thread_id]

                tmps[thread_id] = 0
                for _ in range(n_factors):
                    tmps[thread_id] += item_vectors[i][_] * item_vectors[i][_]
                tmps[thread_id] = max(1.0, tmps[thread_id])
                for _ in range(n_factors):
                    item_vectors[i][_] /= tmps[thread_id]

                tmps[thread_id] = 0
                for _ in range(n_factors):
                    tmps[thread_id] += item_vectors[j][_] * item_vectors[j][_]
                tmps[thread_id] = max(1.0, tmps[thread_id])
                for _ in range(n_factors):
                    item_vectors[j][_] /= tmps[thread_id]

        finally:
            free(neg_sample_cnts)
            free(tmps)
            free(cov)
            free(vec_avg)
            free(u_deriv)
            free(i_deriv)
            free(j_deriv)
            pass
    return loss

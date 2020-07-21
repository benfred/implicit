import itertools
import logging
import multiprocessing
import random
import time

import cython
from cython cimport floating, integral
from cython.parallel import parallel, prange, threadid

import numpy as np
import scipy.sparse
import tqdm


from libc.stdlib cimport malloc, free
from libc.string cimport memcpy, memset
from libc.math cimport exp, sqrt
from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set

from .recommender_base import MatrixFactorizationBase
from .utils import check_random_state

cimport scipy.linalg.cython_blas as cython_blas

log = logging.getLogger("implicit")


cdef inline floating dot(int *n, floating *sx, int *incx, floating *sy, int *incy) nogil:
    if floating is double:
        return cython_blas.ddot(n, sx, incx, sy, incy)
    else:
        return cython_blas.sdot(n, sx, incx, sy, incy)


cdef inline void axpy(int * n, floating * da, floating * dx, int * incx, floating * dy,
                      int * incy) nogil:
    if floating is double:
        cython_blas.daxpy(n, da, dx, incx, dy, incy)
    else:
        cython_blas.saxpy(n, da, dx, incx, dy, incy)

cdef extern from "<random>" namespace "std":
    cdef cppclass mt19937:
        mt19937(unsigned int)

    cdef cppclass uniform_int_distribution[T]:
        uniform_int_distribution(T, T)
        T operator()(mt19937) nogil

cdef inline floating cap(floating x) nogil:
    cdef floating one = 1.0
    return max(min(one, x), -one)

cdef inline floating sigmoid(floating x) nogil:
    cdef floating one = 1.0
    return one / (one + exp(-x))

cdef inline floating tanh(floating x) nogil:
    cdef floating one = 1.0
    cdef floating exp2x = exp(2.0 * x)
    return (exp2x - one) / (exp2x + one)

cdef inline floating sigmoid_deriv(floating sig_x) nogil:
    cdef floating one = 1.0
    return sig_x * (one - sig_x)

cdef inline floating tanh_deriv(floating tanh_x) nogil:
    cdef floating one = 1.0
    return one - tanh_x * tanh_x

cdef inline floating identity(floating x) nogil:
    return x

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


class CollaborativeDenoisingAutoEncoder(MatrixFactorizationBase):
    """ Collaborative Denoising AutoEncoder
    Algorithm of the model is described
    in Collaborative Denoising Auto-Encoders for
    Top-N Recommender Systems <https://dl.acm.org/doi/10.1145/2835776.2835837>

    A few notes:
        - Negative sampling reduces training time signifcantly,
        with affordable performance degrading.
        - Lock on gradient calculation does not help to compensate
        increased training time.
        - It is quite hard to add several different losses,
        especially losses involving softmax operation.
        Thus just implemented simple form:
        tanh for in activation, sigmoid for out actiation, and Binary Cross Entropy Loss.
        TODO: if found effective scheme to add different losses and activations, do.
    Parameters:
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
    dropout: float, optional
        The keep probability of dropout ratio.
        If dropout set to be 0.9, 10% of items will be dropped during training.
        Left item factors are magnified by 1.0 / dropout.
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

    def __init__(self, factors=64, learning_rate=0.001,
                 regularization=0.01, dtype=np.float32,
                 iterations=20, num_threads=0,
                 neg_prop=30,
                 dropout=0.5,
                 use_gpu=False,
                 random_state=None):
        super(CollaborativeDenoisingAutoEncoder, self).__init__()
        self.factors = factors
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.regularization = regularization
        self.dtype = dtype
        self.use_gpu = use_gpu
        self.num_threads = num_threads
        self.neg_prop = neg_prop
        self.dropout = dropout
        self.random_state = random_state
        self.user_factors = None
        self.item_factors = None
        self.W = None
        self.b = None

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

        # this basically calculates the 'row' attribute of a COO matrix
        # without requiring us to get the whole COO matrix
        user_counts = np.ediff1d(user_items.indptr)
        item_counts = np.bincount(user_items.indices, minlength=items)

        if self.item_factors is None:
            self.item_factors = rs.normal(0, 0.001, size=(items, self.factors)).astype(self.dtype)

        if self.user_factors is None:
            self.user_factors = rs.normal(0, 0.001, size=(users, self.factors)).astype(self.dtype)

        if self.W is None:
            self.W = rs.normal(0, 0.001, size=(items, self.factors)).astype(self.dtype)
        if self.b is None:
            self.b = np.zeros(shape=(items,)).astype(self.dtype)

        # For Adagrad update
        u_factors_d_sum = np.zeros(shape=(users, self.factors)).astype(self.dtype)
        i_factors_d_sum = np.zeros(shape=(items, self.factors)).astype(self.dtype)
        W_d_sum = np.zeros(shape=(items, self.factors)).astype(self.dtype)
        b_d_sum = np.zeros(shape=(items,)).astype(self.dtype)

        cdef int num_threads = self.num_threads
        if not num_threads:
            num_threads = multiprocessing.cpu_count()

        # initialize RNG's, one per thread. Also pass the seeds for each thread's RNG
        cdef long[:] rng_seeds = rs.randint(0, 2**31, size=num_threads)
        cdef RNGVector rng = RNGVector(num_threads, len(user_items.data) - 1, rng_seeds)
        cdef RNGVector rng2 = RNGVector(num_threads, 10000 + 1, rng_seeds)
        log.debug("Running %i CDAE training epochs", self.iterations)
        with tqdm.tqdm(total=self.iterations, disable=not show_progress) as progress:
            for epoch in range(self.iterations):
                cdae_update(rng, rng2,
                            u_factors_d_sum,
                            i_factors_d_sum,
                            W_d_sum,
                            b_d_sum,
                            self.user_factors,
                            self.item_factors,
                            self.W,
                            self.b,
                            user_items.indices, user_items.indptr,
                            self.neg_prop, self.learning_rate, self.regularization, self.dropout,
                            num_threads)
                progress.update(1)

    def recommend(self, userid, user_items,
                  N=10, filter_already_liked_items=True, filter_items=None):

        z = np.zeros(self.factors).astype(self.dtype)
        for itemid in user_items[userid].indices:
            z += self.item_factors[itemid]
        z += self.user_factors[userid]
        z = np.tanh(z)

        liked = set()
        if filter_already_liked_items:
            liked.update(user_items[userid].indices)
        if filter_items:
            liked.update(filter_items)

        # calculate the top N items, removing the users own liked items from the results
        scores = ((self.W * z).sum(-1) + self.b)

        count = N + len(liked)
        if count < len(scores):
            ids = np.argpartition(scores, -count)[-count:]
            best = sorted(zip(ids, scores[ids]), key=lambda x: -x[1])
        else:
            best = sorted(enumerate(scores), key=lambda x: -x[1])
        return list(itertools.islice((rec for rec in best if rec[0] not in liked), N))


@cython.cdivision(True)
@cython.boundscheck(False)
def cdae_update(RNGVector rng, RNGVector rng2,
                floating[:, :] u_factors_d_sum,
                floating[:, :] i_factors_d_sum,
                floating[:, :] W_d_sum,
                floating[:] b_d_sum,
                floating[:, :] user_factors,
                floating[:, :] item_factors,
                floating[:, :] W,
                floating[:] b,
                integral[:] indices,
                integral[:] indptr,
                int neg_prop,
                float lr,
                float reg,
                float dropout,
                integral num_threads):

    cdef integral n_users = user_factors.shape[0]
    cdef integral n_items = item_factors.shape[0]
    cdef int factors = user_factors.shape[1]
    cdef int zero = 0, one = 1
    cdef integral u, i, it, c, index, uc=0
    cdef integral thread_id
    cdef floating* deriv
    cdef floating* b_d
    cdef floating* z
    cdef floating* w_d
    cdef floating* dldz

    cdef floating s
    cdef floating grad
    cdef int user_seen_item
    cdef floating dlds
    cdef integral _
    cdef floating beta = 1.0
    cdef unordered_set[integral] * likes
    cdef vector[integral] * exploited
    with nogil, parallel(num_threads=num_threads):
        z = <floating*> malloc(sizeof(floating) * factors)
        w_d = <floating*> malloc(sizeof(floating) * factors)
        dldz = <floating*> malloc(sizeof(floating) * factors)
        deriv = <floating*> malloc(sizeof(floating) * factors)
        b_d = <floating*> malloc(sizeof(floating) * n_items)
        thread_id = threadid()
        likes = new unordered_set[integral]()
        exploited = new vector[integral]()
        try:
            for u in prange(n_users, schedule='guided'):
                if indptr[u] == indptr[u + 1]:
                    continue
                user_seen_item = indptr[u + 1] - indptr[u]

                memset(deriv, 0, sizeof(floating) * factors)
                memset(z, 0, sizeof(floating) * factors)
                memset(dldz, 0, sizeof(floating) * factors)
                likes.clear()
                exploited.clear()

                # Positive item indices: c_ui* y_i
                for _ in range(factors):
                    z[_] += user_factors[u, _]
                uc = 0
                for index in range(indptr[u], indptr[u + 1]):
                    i = indices[index]
                    likes.insert(i)
                    if rng2.generate(thread_id) / 10000.0 <= dropout:
                        exploited.push_back(i)
                        uc += 1
                        for _ in range(factors):
                            z[_] += item_factors[i, _] / dropout

                for _ in range(factors):
                    z[_] = tanh(z[_])

                # Positive item indices

                for index in range(indptr[u], indptr[u + 1]):
                    i = indices[index]
                    s = sigmoid(dot(&factors, z, &one, &item_factors[i, 0], &one))
                    dlds = sigmoid_deriv(s) / s

                    for _ in range(factors):
                        w_d[_] = dlds * z[_]
                        grad = cap(w_d[_] + reg * W[i, _])
                        W_d_sum[i, _] += grad ** 2
                        W[i, _] -= lr * cap(w_d[_] + reg * W[i, _]) / sqrt(beta + W_d_sum[i, _])
                    b_d[i] = dlds
                    grad = cap(b_d[i] + reg * b[i])
                    b_d_sum[i] += grad ** 2
                    b[i] -= lr * grad / sqrt(beta + b_d_sum[i])

                    axpy(&factors, &dlds, &item_factors[i, 0], &one, dldz, &one)

                for _ in range(min(n_items // 10, neg_prop * user_seen_item)):
                    index = rng.generate(thread_id)
                    i = indices[index]
                    if likes.find(i) != likes.end():
                        continue
                    s = sigmoid(dot(&factors, z, &one, &item_factors[i, 0], &one))
                    dlds = -sigmoid_deriv(s) / (1. - s)

                    for _ in range(factors):
                        w_d[_] = dlds * z[_]
                        grad = cap(w_d[_] + reg * W[i, _])
                        W_d_sum[i, _] += grad ** 2
                        W[i, _] -= lr * grad / sqrt(beta + W_d_sum[i, _])

                    b_d[i] = dlds
                    b_d_sum[i] += b_d[i] ** 2
                    b[i] -= lr * cap(b_d[i] + reg * b[i]) / sqrt(beta + b_d_sum[i])
                    axpy(&factors, &dlds, &item_factors[i, 0], &one, dldz, &one)

                memset(deriv, 0, sizeof(floating) * factors)
                for _ in range(factors):
                    deriv[_] = dldz[_] * tanh_deriv(z[_])
                    grad = cap(deriv[_] + reg * user_factors[u, _])
                    u_factors_d_sum[u, _] += grad ** 2
                    user_factors[u, _] -= lr * grad / sqrt(beta + u_factors_d_sum[u, _])

                for index in range(exploited[0].size()):
                    i = exploited[0][index]
                    for _ in range(factors):
                        grad = cap(deriv[_] / dropout + reg * item_factors[i, _])
                        i_factors_d_sum[i, _] += grad ** 2
                        item_factors[i, _] -= lr * grad / sqrt(beta + i_factors_d_sum[i, _])

        finally:
            free(z)
            free(w_d)
            free(dldz)
            free(deriv)
            free(b_d)

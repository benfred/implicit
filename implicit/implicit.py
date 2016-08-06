""" Implicit Alternating Least Squares """
import numpy as np
import time
import os
import logging
from . import _implicit

log = logging.getLogger("implicit")


def alternating_least_squares(Cui, factors, regularization=0.01,
                              iterations=15, use_native=True, num_threads=0,
                              dtype=np.float64):
    """ factorizes the matrix Cui using an implicit alternating least squares
    algorithm

    Args:
        Cui (csr_matrix): Confidence Matrix
        factors (int): Number of factors to extract
        regularization (double): Regularization parameter to use
        iterations (int): Number of alternating least squares iterations to
        run
        num_threads (int): Number of threads to run least squares iterations.
        0 means to use all CPU cores.

    Returns:
        tuple: A tuple of (row, col) factors
    """
    _check_open_blas()

    users, items = Cui.shape

    X = np.random.rand(users, factors).astype(dtype) * 0.01
    Y = np.random.rand(items, factors).astype(dtype) * 0.01

    Cui, Ciu = Cui.tocsr(), Cui.T.tocsr()

    solver = _implicit.least_squares if use_native else least_squares

    for iteration in range(iterations):
        s = time.time()
        solver(Cui, X, Y, regularization, num_threads)
        solver(Ciu, Y, X, regularization, num_threads)
        log.debug("finished iteration %i in %s", iteration, time.time() - s)

    return X, Y


def least_squares(Cui, X, Y, regularization, num_threads):
    """ For each user in Cui, calculate factors Xu for them
    using least squares on Y.

    Note: this is at least 10 times slower than the cython version included
    here.
    """
    users, factors = X.shape
    YtY = Y.T.dot(Y)

    for u in range(users):
        # accumulate YtCuY + regularization*I in A
        A = YtY + regularization * np.eye(factors)

        # accumulate YtCuPu in b
        b = np.zeros(factors)

        for i, confidence in nonzeros(Cui, u):
            factor = Y[i]
            A += (confidence - 1) * np.outer(factor, factor)
            b += confidence * factor

        # Xu = (YtCuY + regularization * I)^-1 (YtCuPu)
        X[u] = np.linalg.solve(A, b)


def nonzeros(m, row):
    """ returns the non zeroes of a row in csr_matrix """
    for index in range(m.indptr[row], m.indptr[row+1]):
        yield m.indices[index], m.data[index]


def _check_open_blas():
    """ checks to see if using OpenBlas. If so, warn if the number of threads isn't set to 1
    (causes perf issues) """
    if np.__config__.get_info('openblas_info') and os.environ.get('OPENBLAS_NUM_THREADS') != '1':
        log.warn("OpenBLAS detected. Its highly recommend to set the environment variable "
                 "'export OPENBLAS_NUM_THREADS=1' to disable its internal multithreading")

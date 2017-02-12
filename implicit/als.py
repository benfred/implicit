""" Implicit Alternating Least Squares """
import logging
import time

import numpy as np

from . import _als
from .utils import check_open_blas, nonzeros

log = logging.getLogger("implicit")


def alternating_least_squares(Cui, factors, regularization=0.01,
                              iterations=15,
                              use_native=True, use_cg=True,
                              num_threads=0, dtype=np.float64,
                              calculate_training_loss=False):
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
    check_open_blas()

    users, items = Cui.shape

    X = np.random.rand(users, factors).astype(dtype) * 0.01
    Y = np.random.rand(items, factors).astype(dtype) * 0.01

    Cui, Ciu = Cui.tocsr(), Cui.T.tocsr()

    solver = _als.least_squares if use_native else least_squares
    if use_cg:
        solver = _als.least_squares_cg if use_native else least_squares_cg

    for iteration in range(iterations):
        s = time.time()
        solver(Cui, X, Y, regularization, num_threads=num_threads)
        solver(Ciu, Y, X, regularization, num_threads=num_threads)
        log.debug("finished iteration %i in %s", iteration, time.time() - s)

        if calculate_training_loss:
            loss = _als.calculate_loss(Cui, X, Y, regularization, num_threads)
            log.debug("loss at iteration %i is %s", iteration, loss)

    return X, Y


def least_squares(Cui, X, Y, regularization, num_threads=0):
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


def least_squares_cg(Cui, X, Y, regularization, num_threads=0, cg_steps=3):
    users, factors = X.shape
    YtY = Y.T.dot(Y) + regularization * np.eye(factors, dtype=Y.dtype)

    for u in range(users):
        # start from previous iteration
        x = X[u]

        # calculate residual error r = (YtCuPu - (YtCuY.dot(Xu)
        r = -YtY.dot(x)
        for i, confidence in nonzeros(Cui, u):
            r += (confidence - (confidence - 1) * Y[i].dot(x)) * Y[i]

        p = r.copy()
        rsold = r.dot(r)

        for it in range(cg_steps):
            # calculate Ap = YtCuYp - without actually calculating YtCuY
            Ap = YtY.dot(p)
            for i, confidence in nonzeros(Cui, u):
                Ap += (confidence - 1) * Y[i].dot(p) * Y[i]

            # standard CG update
            alpha = rsold / p.dot(Ap)
            x += alpha * p
            r -= alpha * Ap
            rsnew = r.dot(r)
            p = r + (rsnew / rsold) * p
            rsold = rsnew

        X[u] = x

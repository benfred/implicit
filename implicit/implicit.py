""" Implicit Alternating Least Squares """
from __future__ import print_function

import numpy as np
import time

try:
    from . import _implicit
    has_cython = True
except:
    has_cython = False


def alternating_least_squares(Cui, factors, regularization=0.01,
                              iterations=15, use_native=has_cython):
    """ factorizes the matrix Cui using an implicit alternating least squares
    algorithm

    Args:
        Cui (csr_matrix): Confidence Matrix
        factors (int): Number of factors to extract
        regularization (double): Regularization parameter to use
        iterations (int): Number of alternating least squares iterations to
        run

    Returns:
        tuple: A tuple of (row, col) factors
    """

    users, items = Cui.shape

    X = np.random.rand(users, factors) * 0.01
    Y = np.random.rand(items, factors) * 0.01

    Cui, Ciu = Cui.tocsr(), Cui.T.tocsr()

    solver = _implicit.least_squares if use_native else least_squares

    for iteration in range(iterations):
        s = time.time()
        solver(Cui, X, Y, regularization)
        solver(Ciu, Y, X, regularization)
        print("finished iteration %i in %s" % (iteration, time.time() - s))

    return X, Y


def least_squares(Cui, X, Y, regularization):
    """ For each user in Cui, calculate factors Xu for them
    using least squares on Y.

    Note: this is at least 10 times slower than the cython version included
    here.
    """
    users, factors = X.shape
    YtY = Y.T.dot(Y)

    for u in range(users):
        # accumulate YtCuY + regulariation*I in A
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

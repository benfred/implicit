""" Implicit Alternating Least Squares """
import itertools
import logging
import time

import numpy as np

from . import _als
from .recommender_base import RecommenderBase
from .utils import check_open_blas, nonzeros

log = logging.getLogger("implicit")


class AlternatingLeastSquares(RecommenderBase):
    """ A Recommendation Model based off the algorithms described in the paper 'Collaborative
        Filtering for Implicit Feedback Datasets' with perfomance optimizations described in
        'Applications of the Conjugate Gradient Method for Implicit Feedback Collaborative
        Filtering.'
    """
    def __init__(self, factors=100, regularization=0.01, dtype=np.float64,
                 use_native=True, use_cg=True,
                 iterations=15, calculate_training_loss=False, num_threads=0):
        # parameters on how to factorize
        self.factors = factors
        self.regularization = regularization

        # options on how to fit the model
        self.dtype = dtype
        self.use_native = use_native
        self.use_cg = use_cg
        self.iterations = iterations
        self.calculate_training_loss = calculate_training_loss
        self.num_threads = num_threads

        # learned parameters
        self.item_factors = None
        self.user_factors = None

        # cache of item norms (useful for calculating similar items)
        self._item_norms = None

        check_open_blas()

    def fit(self, item_users):
        """ Factorizes the matrix Cui. This must be called before trying to recommend items.
        After calling this method, the members 'user_factors' and 'item_factors' will be
        initialized with a latent factor model of the input data

        Args:
            item_users (csr_matrix): Matrix of confidences for the liked items. This matrix
                should be a csr_matrix where the rows of the matrix are the
                item, the columns are the users that liked that item, and the
                value is the confidence that the user liked the item.
        """
        Ciu, Cui = item_users.tocsr(), item_users.T.tocsr()
        items, users = Ciu.shape

        # Initialize the variables randomly if they haven't already been set
        if self.user_factors is None:
            self.user_factors = np.random.rand(users, self.factors).astype(self.dtype) * 0.01
        if self.item_factors is None:
            self.item_factors = np.random.rand(items, self.factors).astype(self.dtype) * 0.01

        # invalidate cached norms
        self._item_norms = None

        solver = self.solver

        # alternate between learning the user_factors from the item_factors and vice-versa
        for iteration in range(self.iterations):
            s = time.time()
            solver(Cui, self.user_factors, self.item_factors, self.regularization,
                   num_threads=self.num_threads)
            solver(Ciu, self.item_factors, self.user_factors, self.regularization,
                   num_threads=self.num_threads)
            log.debug("finished iteration %i in %s", iteration, time.time() - s)

            if self.calculate_training_loss:
                loss = _als.calculate_loss(Cui, self.user_factors, self.item_factors,
                                           self.regularization, num_threads=self.num_threads)
                log.debug("loss at iteration %i is %s", iteration, loss)

    def recommend(self, userid, user_items, N=10):
        """ Returns the top N ranked items for a single user """
        scores = self.item_factors.dot(self.user_factors[userid])

        # calcualte the top N items, removing the users own liked items from the results
        liked = set(user_items[userid].indices)
        ids = np.argpartition(scores, -(N + len(liked)))[-(N + len(liked)):]
        best = sorted(zip(ids, scores[ids]), key=lambda x: -x[1])
        return list(itertools.islice((rec for rec in best if rec[0] not in liked), N))

    def similar_items(self, itemid, N=10):
        """ Return the top N similar items for itemid. """
        scores = self.item_factors.dot(self.item_factors[itemid]) / self.item_norms
        best = np.argpartition(scores, -N)[-N:]
        return sorted(zip(best, scores[best] / self.item_norms[itemid]), key=lambda x: -x[1])

    @property
    def item_norms(self):
        if self._item_norms is None:
            self._item_norms = np.linalg.norm(self.item_factors, axis=-1)
        return self._item_norms

    @property
    def solver(self):
        if self.use_cg:
            return _als.least_squares_cg if self.use_native else least_squares_cg
        return _als.least_squares if self.use_native else least_squares


def alternating_least_squares(Ciu, factors, **kwargs):
    """ factorizes the matrix Cui using an implicit alternating least squares
    algorithm. Note: this method is deprecated, consider moving to the
    AlternatingLeastSquares class instead

    """
    log.warning("This method is deprecated. Please use the AlternatingLeastSquares"
                " class instead")

    model = AlternatingLeastSquares(factors=factors, **kwargs)
    model.fit(Ciu)
    return model.item_factors, model.user_factors


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

""" Implicit Alternating Least Squares """
import heapq
import itertools
import logging
import time

import numpy as np
import scipy

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

        # cache for item factors squared
        self._YtY = None

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

        # invalidate cached norms and squared factors
        self._item_norms = None
        self._YtY = None

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

    def recommend(self, userid, user_items, N=10, filter_items=None, recalculate_user=False):
        """ Returns the top N ranked items for a single user, given its id """
        user = self._user_factor(userid, user_items, recalculate_user)
        # calculate the top N items, removing the users own liked items from the results
        liked = set(user_items[userid].indices)
        scores = self.item_factors.dot(user)
        if filter_items:
            liked.update(filter_items)

        count = N + len(liked)
        if count < len(scores):
            ids = np.argpartition(scores, -count)[-count:]
            best = sorted(zip(ids, scores[ids]), key=lambda x: -x[1])
        else:
            best = sorted(enumerate(scores), key=lambda x: -x[1])
        return list(itertools.islice((rec for rec in best if rec[0] not in liked), N))

    def _user_factor(self, userid, user_items, recalculate_user=False):
        if not recalculate_user:
            return self.user_factors[userid]
        return user_factor(self.item_factors, self.YtY,
                           user_items.tocsr(), userid,
                           self.regularization, self.factors)

    def explain(self, userid, user_items, itemid, user_weights=None, N=10):
        """ Returns the predicted rating for an user x item pair,
            the explanation (the contribution from the top N items the user liked),
            and a user latent factor weight that can be cached if you want to
            get more than one explanation for the same user.
        """
        # user_weights = Cholesky decomposition of Wu^-1
        # from section 5 of the paper CF for Implicit Feedback Datasets
        user_items = user_items.tocsr()
        if user_weights is None:
            A, _ = user_linear_equation(self.item_factors, self.YtY,
                                        user_items, userid,
                                        self.regularization, self.factors)
            user_weights = scipy.linalg.cho_factor(A)
        seed_item = self.item_factors[itemid]

        # weighted_item = y_i^t W_u
        weighted_item = scipy.linalg.cho_solve(user_weights, seed_item)

        total_score = 0.0
        h = []
        for i, (itemid, confidence) in enumerate(nonzeros(user_items, userid)):
            factor = self.item_factors[itemid]
            # s_u^ij = (y_i^t W^u) y_j
            score = weighted_item.dot(factor) * confidence
            total_score += score
            contribution = (score, itemid)
            if i < N:
                heapq.heappush(h, contribution)
            else:
                heapq.heappushpop(h, contribution)

        items = (heapq.heappop(h) for i in range(len(h)))
        top_contributions = list((i, s) for s, i in items)[::-1]
        return total_score, top_contributions, user_weights

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

    @property
    def YtY(self):
        if self._YtY is None:
            Y = self.item_factors
            self._YtY = Y.T.dot(Y)
        return self._YtY


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
    users, n_factors = X.shape
    YtY = Y.T.dot(Y)

    for u in range(users):
        X[u] = user_factor(Y, YtY, Cui, u, regularization, n_factors)


def user_linear_equation(Y, YtY, Cui, u, regularization, n_factors):
    # Xu = (YtCuY + regularization * I)^-1 (YtCuPu)
    # YtCuY + regularization * I = YtY + regularization * I + Yt(Cu-I)

    # accumulate YtCuY + regularization*I in A
    A = YtY + regularization * np.eye(n_factors)

    # accumulate YtCuPu in b
    b = np.zeros(n_factors)

    for i, confidence in nonzeros(Cui, u):
        factor = Y[i]
        A += (confidence - 1) * np.outer(factor, factor)
        b += confidence * factor
    return A, b


def user_factor(Y, YtY, Cui, u, regularization, n_factors):
    # Xu = (YtCuY + regularization * I)^-1 (YtCuPu)
    A, b = user_linear_equation(Y, YtY, Cui, u, regularization, n_factors)
    return np.linalg.solve(A, b)


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
            if rsnew < 1e-10:
                break
            p = r + (rsnew / rsold) * p
            rsold = rsnew

        X[u] = x

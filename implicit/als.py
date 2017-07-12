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
    """ Alternating Least Squares

    A Recommendation Model based off the algorithms described in the paper 'Collaborative
    Filtering for Implicit Feedback Datasets' with performance optimizations described in
    'Applications of the Conjugate Gradient Method for Implicit Feedback Collaborative
    Filtering.'

    Parameters
    ----------
    factors : int, optional
        The number of latent factors to compute
    regularization : float, optional
        The regularization factor to use
    dtype : data-type, optional
        Specifies whether to generate 64 bit or 32 bit floating point factors
    use_native : bool, optional
        Use native extensions to speed up model fitting
    use_cg : bool, optional
        Use a faster Conjugate Gradient solver to calculate factors
    iterations : int, optional
        The number of ALS iterations to use when fitting data
    calculate_training_loss : bool, optional
        Whether to log out the training loss at each iteration
    num_threads : int, optional
        The number of threads to use for fitting the model. This only
        applies for the native extensions. Specifying 0 means to default
        to the number of cores on the machine.

    Attributes
    ----------
    item_factors : ndarray
        Array of latent factors for each item in the training set
    user_factors : ndarray
        Array of latent factors for each user in the training set
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
        """ Factorizes the item_users matrix.

        After calling this method, the members 'user_factors' and 'item_factors' will be
        initialized with a latent factor model of the input data.

        The item_users matrix does double duty here. It defines which items are liked by which
        users (P_iu in the original paper), as well as how much confidence we have that the user
        liked the item (C_iu).

        The negative items are implicitly defined: This code assumes that non-zero items in the
        item_users matrix means that the user liked the item. The negatives are left unset in this
        sparse matrix: the library will assume that means Piu = 0 and Ciu = 1 for all these items.

        Parameters
        ----------
        item_users: csr_matrix
            Matrix of confidences for the liked items. This matrix should be a csr_matrix where
            the rows of the matrix are the item, the columns are the users that liked that item,
            and the value is the confidence that the user liked the item.
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

    recommend.__doc__ = RecommenderBase.recommend.__doc__

    def recommend_batch(self, user_ids=None, item_ids=None, N=10, ignore_pairs=None):
        """ Batch computation of recommendations, optionally for subsets of users and items.

        Parameters
        ---------
        user_ids : arraylike, optional
            If given, restricts the users recommendations are made for. If not given,
            recommendations are made for all users.
        item_ids : arraylike, optional
            If given, restricts the items recommendations are made for. If not given,
            recommendations are made for all items.
        N : int > 0, optional
            Maximum number of items to recommend per user. If there are fewer items available for
            a user than N, all available items are returned.
        ignore_pairs : tuple-like, optional
            Indices (user ID and item ID) of user-item pairs that are not eligible for
            recommendation. If given, the user cannot be recommended the item.
            If you want to ignore liked events, input the result of user_items.nonzero() here.
            See also np.nonzero().

        Returns
        ---------
        recommendations : list
            A list with an element per user (in order of user_ids). Each element is a list of up to
            N (item_id, weight) tuples.
        """

        # Fetch and subset factorizations
        user_latent = self.user_factors  # all users x factors
        item_latent = self.item_factors  # all items x factors

        if user_ids is None:
            user_ids = np.arange(user_latent.shape[0])
        if item_ids is None:
            item_ids = np.arange(item_latent.shape[0])

        n_users = len(user_ids)
        n_items = len(item_ids)
        user_latent = user_latent[user_ids, :]  # user_ids x factors
        item_latent = item_latent[item_ids, :]  # item_ids x factors

        # Calculate the recommendation weights
        weights = user_latent.dot(item_latent.T)  # users x items

        # Optionally remove weights for items with high confidences
        if ignore_pairs is not None:
            drop_row_idx, drop_col_idx = ignore_pairs
            weights[drop_row_idx, drop_col_idx] = np.nan

        item_labels = np.repeat(item_ids[np.newaxis, :], n_users, axis=0)  # users x items

        # Advanced indexing! Adapted from https://stackoverflow.com/a/33141247/3275967
        row_selector = np.arange(n_users)[:, np.newaxis]

        # Select top N items per row iff there are N or more items available.
        if N < n_items:
            # Indices of the top N weights per row
            indices = np.argpartition(-weights, N, axis=1)[:, :N]  # users x N
            # Advanced indexing: For each row (user) in weights, take the top N weights
            weights = weights[row_selector, indices]  # users x N
            # Separate but related, the item IDs of the top N weights per user, same ordering as in
            # weights.
            item_labels = item_labels[row_selector, indices]  # users x N

        # Sort the weights and their associated item IDs
        indices = np.argsort(-weights)
        weights = weights[row_selector, indices]
        item_labels = item_labels[row_selector, indices]

        # Prepare output: A list of recommendations per user. The recommendations are themselves
        # lists of (item_id, weight) tuples.
        recommendations = []
        for user_id in range(n_users):
            user_recommendations = []
            for column in range(min(N, n_items)):
                w = weights[user_id, column]
                if np.isnan(w):
                    # nothing more to do here, only nans remaining in the row
                    break
                recommended_item = (item_labels[user_id, column], w)
                user_recommendations.append(recommended_item)
            recommendations.append(user_recommendations)

        return recommendations

    def _user_factor(self, userid, user_items, recalculate_user=False):
        if not recalculate_user:
            return self.user_factors[userid]
        return user_factor(self.item_factors, self.YtY,
                           user_items.tocsr(), userid,
                           self.regularization, self.factors)

    def explain(self, userid, user_items, itemid, user_weights=None, N=10):
        """ Provides explanations for why the item is liked by the user.

        Parameters
        ---------
        userid : int
            The userid to explain recommendations for
        user_items : csr_matrix
            Sparse matrix containing the liked items for the user
        itemid : int
            The itemid to explain recommendations for
        user_weights : ndarray, optional
            Precomputed Cholesky decomposition of the weighted user liked items.
            Useful for speeding up repeated calls to this function, this value
            is returned
        N : int, optional
            The number of liked items to show the contribution for

        Returns
        -------
        total_score : float
            The total predicted score for this user/item pair
        top_contributions : list
            A list of the top N (itemid, score) contributions for this user/item pair
        user_weights : ndarray
            A factorized representation of the user. Passing this in to
            future 'explain' calls will lead to noticeable speedups
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
        scores = self.item_factors.dot(self.item_factors[itemid]) / self.item_norms
        best = np.argpartition(scores, -N)[-N:]
        return sorted(zip(best, scores[best] / self.item_norms[itemid]), key=lambda x: -x[1])

    similar_items.__doc__ = RecommenderBase.similar_items.__doc__

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

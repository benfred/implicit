import logging
import time

import numpy as np
from tqdm.auto import tqdm

import implicit.gpu
from implicit.gpu.matrix_factorization_base import MatrixFactorizationBase, check_random_state
from implicit.utils import check_csr

log = logging.getLogger("implicit")


class AlternatingLeastSquares(MatrixFactorizationBase):
    """Alternating Least Squares

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
    alpha : float, optional
        The weight to give to positive examples.
    dtype : data-type, optional
        Specifies whether to use 16 bit or 32 bit floating point factors
    iterations : int, optional
        The number of ALS iterations to use when fitting data
    calculate_training_loss : bool, optional
        Whether to log out the training loss at each iteration
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

    def __init__(
        self,
        factors=64,
        regularization=0.01,
        alpha=1.0,
        dtype=np.float32,
        iterations=15,
        calculate_training_loss=False,
        random_state=None,
    ):
        if not implicit.gpu.HAS_CUDA:
            raise ValueError("No CUDA extension has been built, can't train on GPU.")

        super().__init__()

        # parameters on how to factorize
        self.factors = factors
        self.regularization = regularization
        self.alpha = alpha
        self.dtype = dtype

        # options on how to fit the model
        self.iterations = iterations
        self.calculate_training_loss = calculate_training_loss
        self.fit_callback = None
        self.random_state = random_state
        self.cg_steps = 3

        # cached access to properties
        self._solver = None
        self._YtY = None
        self._XtX = None

    def fit(self, user_items, show_progress=True, callback=None):
        """Factorizes the user_items matrix.

        After calling this method, the members 'user_factors' and 'item_factors' will be
        initialized with a latent factor model of the input data.

        The user_items matrix does double duty here. It defines which items are liked by which
        users (P_ui in the original paper), as well as how much confidence we have that the user
        liked the item (C_ui.

        The negative items are implicitly defined: This code assumes that positive items in the
        user_items matrix means that the user liked the item. The negatives are left unset in this
        sparse matrix: the library will assume that means Piu = 0 and Ciu = 1 for all these items.
        Negative items can also be passed with a higher confidence value by passing a negative
        value, indicating that the user disliked the item.

        Parameters
        ----------
        user_items: csr_matrix
            Matrix of confidences for the liked items. This matrix should be a csr_matrix where
            the rows of the matrix are the user, the columns are the items liked by that user,
            and the value is the confidence that the user liked the item.
        show_progress : bool, optional
            Whether to show a progress bar during fitting
        callback: Callable, optional
            Callable function on each epoch with such arguments as epoch, elapsed time and progress
        """
        # initialize the random state
        random_state = check_random_state(self.random_state)

        # TODO: allow passing in cupy arrays on gpu
        Cui = check_csr(user_items)

        if Cui.dtype != np.float32:
            Cui = Cui.astype(np.float32)

        s = time.time()
        Ciu = Cui.T.tocsr()
        log.debug("Calculated transpose in %.3fs", time.time() - s)

        items, users = Ciu.shape

        s = time.time()

        # Initialize the variables randomly if they haven't already been set
        if self.user_factors is None:
            self.user_factors = random_state.uniform(
                users, self.factors, low=-0.5 / self.factors, high=0.5 / self.factors
            )
            self.user_factors = self.user_factors.astype(self.dtype)

        if self.item_factors is None:
            self.item_factors = random_state.uniform(
                items, self.factors, low=-0.5 / self.factors, high=0.5 / self.factors
            )
            self.item_factors = self.item_factors.astype(self.dtype)

        log.debug("Initialized factors in %s", time.time() - s)

        # invalidate cached norms and squared factors
        self._item_norms = self._user_norms = None
        self._item_norms_host = self._user_norms_host = None
        self._YtY = self._XtX = None

        Ciu = implicit.gpu.CSRMatrix(Ciu)
        Cui = implicit.gpu.CSRMatrix(Cui)
        X = self.user_factors
        Y = self.item_factors
        loss = None

        _YtY = implicit.gpu.Matrix.zeros(self.factors, self.factors)
        _XtX = implicit.gpu.Matrix.zeros(self.factors, self.factors)

        log.debug("Running %i ALS iterations", self.iterations)
        with tqdm(total=self.iterations, disable=not show_progress) as progress:
            for iteration in range(self.iterations):
                s = time.time()
                self.solver.calculate_yty(Y, _YtY, self.regularization)
                self.solver.least_squares(Cui, X, _YtY, Y, self.cg_steps)

                self.solver.calculate_yty(X, _XtX, self.regularization)
                self.solver.least_squares(Ciu, Y, _XtX, X, self.cg_steps)
                progress.update(1)

                if self.calculate_training_loss:
                    loss = self.solver.calculate_loss(Cui, X, Y, self.regularization)
                    progress.set_postfix({"loss": loss})

                    if not show_progress:
                        log.info("loss %.4f", loss)

                # Backward compatibility
                if not callback:
                    callback = self.fit_callback
                if callback:
                    callback(iteration, time.time() - s, loss)

        if self.calculate_training_loss:
            log.info("Final training loss %.4f", loss)

    def recalculate_user(self, userid, user_items):
        if self.alpha != 1.0:
            user_items = self.alpha * user_items

        users = 1 if np.isscalar(userid) else len(userid)
        user_factors = implicit.gpu.Matrix.zeros(users, self.factors).astype(self.dtype)
        Cui = implicit.gpu.CSRMatrix(user_items)

        self.solver.least_squares(
            Cui, user_factors, self.YtY, self.item_factors, cg_steps=self.factors
        )
        return user_factors[0] if np.isscalar(userid) else user_factors

    def recalculate_item(self, itemid, item_users):
        if self.alpha != 1.0:
            item_users = self.alpha * item_users

        items = 1 if np.isscalar(itemid) else len(itemid)
        item_factors = implicit.gpu.Matrix.zeros(items, self.factors).astype(self.dtype)
        Ciu = implicit.gpu.CSRMatrix(item_users)
        self.solver.least_squares(
            Ciu, item_factors, self.XtX, self.user_factors, cg_steps=self.factors
        )
        return item_factors[0] if np.isscalar(itemid) else item_factors

    def partial_fit_users(self, userids, user_items):
        """Incrementally updates user factors

        This method updates factors for users specified by userids, given a
        sparse matrix of items that they have interacted with before. This
        allows you to retrain only parts of the model with new data, and
        avoid a full retraining when new users appear - or the liked
        items for an existing user change.

        Parameters
        ----------
        userids : array_like
            An array of userids to calculate new factors for
        user_items : csr_matrix
            Sparse matrix containing the liked items for each user
        """
        if len(userids) != user_items.shape[0]:
            raise ValueError("user_items must contain 1 row for every user in userids")

        # recalculate factors for each user in the input
        user_factors = self.recalculate_user(userids, user_items)

        # ensure that we have enough storage for any new users
        users, factors = self.user_factors.shape
        max_userid = max(userids)
        if max_userid >= users:
            # TODO: grow exponentially ?
            self.user_factors.resize(max_userid + 1, factors)

        self.user_factors.assign_rows(userids, user_factors)

        # clear any cached properties that are invalidated by this update
        self._user_norms = self._user_norms_host = None
        self._XtX = None

    def partial_fit_items(self, itemids, item_users):
        """Incrementally updates item factors

        This method updates factors for items specified by itemids, given a
        sparse matrix of users that have interacted with them. This
        allows you to retrain only parts of the model with new data, and
        avoid a full retraining when new users appear - or the liked
        users for an existing item change.

        Parameters
        ----------
        itemids : array_like
            An array of itemids to calculate new factors for
        item_users : csr_matrix
            Sparse matrix containing the liked users for each item in itemids
        """
        if len(itemids) != item_users.shape[0]:
            raise ValueError("item_users must contain 1 row for every user in itemids")

        # recalculate factors for each item in the input
        item_factors = self.recalculate_item(itemids, item_users)

        # ensure that we have enough storage for any new items
        items, factors = self.item_factors.shape
        max_itemid = max(itemids)
        if max_itemid >= items:
            # TODO: grow exponentially ?
            self.item_factors.resize(max_itemid + 1, factors)

        # update the stored factors with the newly calculated values
        self.item_factors.assign_rows(itemids, item_factors)

        # clear any cached properties that are invalidated by this update
        self._item_norms = self._item_norms_host = None
        self._YtY = None

    @property
    def solver(self):
        if self._solver is None:
            self._solver = implicit.gpu.LeastSquaresSolver()
        return self._solver

    @property
    def YtY(self):
        if self._YtY is None:
            self._YtY = implicit.gpu.Matrix.zeros(self.factors, self.factors)
            self.solver.calculate_yty(self.item_factors, self._YtY, self.regularization)
        return self._YtY

    @property
    def XtX(self):
        if self._XtX is None:
            self._XtX = implicit.gpu.Matrix.zeros(self.factors, self.factors)
            self.solver.calculate_yty(self.user_factors, self._XtX, self.regularization)
        return self._XtX

    def to_cpu(self) -> implicit.cpu.als.AlternatingLeastSquares:
        """Converts this model to an equivalent version running on the CPU"""
        ret = implicit.cpu.als.AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            alpha=self.alpha,
            dtype=self.dtype,
            iterations=self.iterations,
            calculate_training_loss=self.calculate_training_loss,
            random_state=self.random_state,
        )
        ret.user_factors = self.user_factors.to_numpy() if self.user_factors is not None else None
        ret.item_factors = self.item_factors.to_numpy() if self.item_factors is not None else None
        return ret

    def __getstate__(self):
        state = super().__getstate__()
        state["_solver"] = None
        state["_XtX"] = self._XtX.to_numpy() if self._XtX is not None else None
        state["_YtY"] = self._YtY.to_numpy() if self._YtY is not None else None
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        if self._XtX is not None:
            self._XtX = implicit.gpu.Matrix(self._XtX)
        if self._YtY is not None:
            self._YtY = implicit.gpu.Matrix(self._YtY)


def calculate_loss(Cui, X, Y, regularization, solver=None):
    """Calculates the loss for an ALS model"""
    if not isinstance(Cui, implicit.gpu.CSRMatrix):
        Cui = implicit.gpu.CSRMatrix(Cui)
    if not isinstance(X, implicit.gpu.Matrix):
        X = implicit.gpu.Matrix(X)
    if not isinstance(Y, implicit.gpu.Matrix):
        Y = implicit.gpu.Matrix(Y)
    if solver is None:
        solver = implicit.gpu.LeastSquaresSolver()

    return solver.calculate_loss(Cui, X, Y, regularization)

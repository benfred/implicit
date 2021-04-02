import logging
import time

import numpy as np
import scipy
import scipy.sparse
from tqdm.auto import tqdm

import implicit.gpu

from .matrix_factorization_base import MatrixFactorizationBase, check_random_state

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
        iterations=15,
        calculate_training_loss=False,
        random_state=None,
    ):
        if not implicit.gpu.HAS_CUDA:
            raise ValueError("No CUDA extension has been built, can't train on GPU.")

        super(AlternatingLeastSquares, self).__init__()

        # parameters on how to factorize
        self.factors = factors
        self.regularization = regularization

        # options on how to fit the model
        self.iterations = iterations
        self.calculate_training_loss = calculate_training_loss
        self.fit_callback = None
        self.random_state = random_state
        self.cg_steps = 3

    def fit(self, item_users, show_progress=True):
        """Factorizes the item_users matrix.

        After calling this method, the members 'user_factors' and 'item_factors' will be
        initialized with a latent factor model of the input data.

        The item_users matrix does double duty here. It defines which items are liked by which
        users (P_iu in the original paper), as well as how much confidence we have that the user
        liked the item (C_iu).

        The negative items are implicitly defined: This code assumes that positive items in the
        item_users matrix means that the user liked the item. The negatives are left unset in this
        sparse matrix: the library will assume that means Piu = 0 and Ciu = 1 for all these items.
        Negative items can also be passed with a higher confidence value by passing a negative
        value, indicating that the user disliked the item.

        Parameters
        ----------
        item_users: csr_matrix
            Matrix of confidences for the liked items. This matrix should be a csr_matrix where
            the rows of the matrix are the item, the columns are the users that liked that item,
            and the value is the confidence that the user liked the item.
        show_progress : bool, optional
            Whether to show a progress bar during fitting
        """
        # initialize the random state
        random_state = check_random_state(self.random_state)

        # TODO: allow passing in cupy arrays on gpu
        Ciu = item_users
        if not isinstance(Ciu, scipy.sparse.csr_matrix):
            s = time.time()
            log.debug("Converting input to CSR format")
            Ciu = Ciu.tocsr()
            log.debug("Converted input to CSR in %.3fs", time.time() - s)

        if Ciu.dtype != np.float32:
            Ciu = Ciu.astype(np.float32)

        s = time.time()
        Cui = Ciu.T.tocsr()
        log.debug("Calculated transpose in %.3fs", time.time() - s)

        items, users = Ciu.shape

        s = time.time()

        # Initialize the variables randomly if they haven't already been set
        if self.user_factors is None:
            self.user_factors = random_state.uniform(
                users, self.factors, low=-0.5 / self.factors, high=0.5 / self.factors
            )
        if self.item_factors is None:
            self.item_factors = random_state.uniform(
                items, self.factors, low=-0.5 / self.factors, high=0.5 / self.factors
            )

        log.debug("Initialized factors in %s", time.time() - s)

        # invalidate cached norms and squared factors
        self._item_norms = self._user_norms = None

        Ciu = implicit.gpu.CSRMatrix(Ciu)
        Cui = implicit.gpu.CSRMatrix(Cui)
        X = self.user_factors
        Y = self.item_factors

        solver = implicit.gpu.LeastSquaresSolver(self.factors)
        log.debug("Running %i ALS iterations", self.iterations)
        with tqdm(total=self.iterations, disable=not show_progress) as progress:
            for iteration in range(self.iterations):
                s = time.time()
                solver.least_squares(Cui, X, Y, self.regularization, self.cg_steps)
                solver.least_squares(Ciu, Y, X, self.regularization, self.cg_steps)
                progress.update(1)

                if self.fit_callback:
                    self.fit_callback(iteration, time.time() - s)

                if self.calculate_training_loss:
                    loss = solver.calculate_loss(Cui, X, Y, self.regularization)
                    progress.set_postfix({"loss": loss})

        if self.calculate_training_loss:
            log.info("Final training loss %.4f", loss)

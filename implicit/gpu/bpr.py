import logging

try:
    import cupy as cp
except ImportError:
    pass
import numpy as np
from tqdm.auto import tqdm

import implicit.gpu

from .matrix_factorization_base import MatrixFactorizationBase, check_random_state

log = logging.getLogger("implicit")


class BayesianPersonalizedRanking(MatrixFactorizationBase):
    """Bayesian Personalized Ranking

    A recommender model that learns  a matrix factorization embedding based off minimizing the
    pairwise ranking loss described in the paper `BPR: Bayesian Personalized Ranking from Implicit
    Feedback <https://arxiv.org/pdf/1205.2618.pdf>`_.

    Parameters
    ----------
    factors : int, optional
        The number of latent factors to compute
    learning_rate : float, optional
        The learning rate to apply for SGD updates during training
    regularization : float, optional
        The regularization factor to use
    iterations : int, optional
        The number of training epochs to use when fitting the data
    verify_negative_samples: bool, optional
        When sampling negative items, check if the randomly picked negative item has actually
        been liked by the user. This check increases the time needed to train but usually leads
        to better predictions.
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
        factors=100,
        learning_rate=0.01,
        regularization=0.01,
        dtype=np.float32,
        iterations=100,
        verify_negative_samples=True,
        random_state=None,
    ):
        super(BayesianPersonalizedRanking, self).__init__()
        if not implicit.gpu.HAS_CUDA:
            raise ValueError("No CUDA extension has been built, can't train on GPU.")

        if (factors + 1) % 32:
            padding = 32 - (factors + 1) % 32
            log.warning(
                "GPU training requires factor size to be a multiple of 32 - 1."
                " Increasing factors from %i to %i.",
                factors,
                factors + padding,
            )
            factors += padding

        self.factors = factors
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.regularization = regularization
        self.verify_negative_samples = verify_negative_samples
        self.random_state = random_state

    def fit(self, item_users, show_progress=True):
        """Factorizes the item_users matrix

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

        # We need efficient user lookup for case of removing own likes
        # TODO: might make more sense to just changes inputs to be users by items instead
        # but that would be a major breaking API change
        user_items = item_users.T.tocsr()
        if not user_items.has_sorted_indices:
            user_items.sort_indices()

        # this basically calculates the 'row' attribute of a COO matrix
        # without requiring us to get the whole COO matrix
        user_counts = np.ediff1d(user_items.indptr)
        userids = np.repeat(np.arange(users), user_counts).astype(user_items.indices.dtype)

        # create factors if not already created.
        # Note: the final dimension is for the item bias term - which is set to a 1 for all users
        # this simplifies interfacing with approximate nearest neighbours libraries etc
        if self.item_factors is None:
            self.item_factors = rs.rand(items, self.factors + 1, dtype=cp.float32) - 0.5
            self.item_factors /= self.factors

            # set factors to all zeros for items without any ratings
            item_counts = np.bincount(user_items.indices, minlength=items)
            self.item_factors[item_counts == 0] = cp.zeros(self.factors + 1)

        if self.user_factors is None:
            self.user_factors = rs.rand(users, self.factors + 1, dtype=cp.float32) - 0.5
            self.user_factors /= self.factors

            # set factors to all zeros for users without any ratings
            self.user_factors[user_counts == 0] = cp.zeros(self.factors + 1)

            self.user_factors[:, self.factors] = 1.0

        self._item_norms = self._user_norms = None

        userids = implicit.gpu.CuIntVector(userids)
        itemids = implicit.gpu.CuIntVector(user_items.indices)
        indptr = implicit.gpu.CuIntVector(user_items.indptr)

        X = implicit.gpu.CuDenseMatrix(self.user_factors)
        Y = implicit.gpu.CuDenseMatrix(self.item_factors)

        log.debug("Running %i BPR training epochs", self.iterations)
        with tqdm(total=self.iterations, disable=not show_progress) as progress:
            for epoch in range(self.iterations):
                correct, skipped = implicit.gpu.cu_bpr_update(
                    userids,
                    itemids,
                    indptr,
                    X,
                    Y,
                    self.learning_rate,
                    self.regularization,
                    rs.randint(2 ** 31),
                    self.verify_negative_samples,
                )
                progress.update(1)
                total = len(user_items.data)
                if total != 0 and total != skipped:
                    progress.set_postfix(
                        {
                            "correct": "%.2f%%" % (100.0 * correct / (total - skipped)),
                            "skipped": "%.2f%%" % (100.0 * skipped / total),
                        }
                    )

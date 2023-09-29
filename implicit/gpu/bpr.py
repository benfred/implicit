import logging
import time

import numpy as np
from tqdm.auto import tqdm

import implicit.gpu

from ..utils import check_csr, check_random_state
from .matrix_factorization_base import MatrixFactorizationBase

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
    random_state : int, RandomState, Generator or None, optional
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
        super().__init__()
        if not implicit.gpu.HAS_CUDA:
            raise ValueError("No CUDA extension has been built, can't train on GPU.")

        self.factors = factors
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.regularization = regularization
        self.verify_negative_samples = verify_negative_samples
        self.random_state = random_state

    def fit(self, user_items, show_progress=True, callback=None):
        """Factorizes the user_items matrix

        Parameters
        ----------
        user_items: csr_matrix
            Matrix of confidences for the liked items. This matrix should be a csr_matrix where
            the rows of the matrix are the user, and the columns are the items liked by that user.
            BPR ignores the weight value of the matrix right now - it treats non zero entries
            as a binary signal that the user liked the item.
        show_progress : bool, optional
            Whether to show a progress bar
        callback: Callable, optional
            Callable function on each epoch with such arguments as epoch, elapsed time and progress
        """
        rs = check_random_state(self.random_state)
        user_items = check_csr(user_items)

        # for now, all we handle is float 32 values
        if user_items.dtype != np.float32:
            user_items = user_items.astype(np.float32)

        users, items = user_items.shape

        # We need efficient user lookup for case of removing own likes
        if self.verify_negative_samples and not user_items.has_sorted_indices:
            user_items.sort_indices()

        # this basically calculates the 'row' attribute of a COO matrix
        # without requiring us to get the whole COO matrix
        user_counts = np.ediff1d(user_items.indptr)
        userids = np.repeat(np.arange(users), user_counts).astype(user_items.indices.dtype)

        # create factors if not already created.
        # Note: the final dimension is for the item bias term - which is set to a 1 for all users
        # this simplifies interfacing with approximate nearest neighbours libraries etc
        if self.item_factors is None:
            item_factors = rs.random((items, self.factors + 1), "float32") - 0.5
            item_factors /= self.factors

            # set factors to all zeros for items without any ratings
            item_counts = np.bincount(user_items.indices, minlength=items)
            item_factors[item_counts == 0] = np.zeros(self.factors + 1)
            self.item_factors = implicit.gpu.Matrix(item_factors)

        if self.user_factors is None:
            user_factors = rs.random((users, self.factors + 1), "float32") - 0.5
            user_factors /= self.factors

            # set factors to all zeros for users without any ratings
            user_factors[user_counts == 0] = np.zeros(self.factors + 1)
            user_factors[:, self.factors] = 1.0

            self.user_factors = implicit.gpu.Matrix(user_factors)

        self._item_norms = self._user_norms = None

        userids = implicit.gpu.IntVector(userids)
        itemids = implicit.gpu.IntVector(user_items.indices)
        indptr = implicit.gpu.IntVector(user_items.indptr)

        X = self.user_factors
        Y = self.item_factors

        log.debug("Running %i BPR training epochs", self.iterations)
        with tqdm(total=self.iterations, disable=not show_progress) as progress:
            for _epoch in range(self.iterations):
                s = time.time()
                correct, skipped = implicit.gpu.bpr_update(
                    userids,
                    itemids,
                    indptr,
                    X,
                    Y,
                    self.learning_rate,
                    self.regularization,
                    rs.integers(2**31),
                    self.verify_negative_samples,
                )
                progress.update(1)
                total = len(user_items.data)
                if total and total != skipped:
                    progress.set_postfix(
                        {
                            "train_auc": f"{100.0 * correct / (total - skipped):0.2f}%",
                            "skipped": f"{100.0 * skipped / total:0.2f}%",
                        }
                    )
                if callback:
                    callback(_epoch, time.time() - s, correct, skipped)

    def to_cpu(self) -> implicit.cpu.bpr.BayesianPersonalizedRanking:
        """Converts this model to an equivalent version running on the cpu"""
        ret = implicit.cpu.bpr.BayesianPersonalizedRanking(
            factors=self.factors,
            learning_rate=self.learning_rate,
            regularization=self.regularization,
            iterations=self.iterations,
            verify_negative_samples=self.verify_negative_samples,
            random_state=self.random_state,
        )
        ret.user_factors = self.user_factors.to_numpy() if self.user_factors is not None else None
        ret.item_factors = self.item_factors.to_numpy() if self.item_factors is not None else None
        return ret

import numpy as np

import implicit.cpu.bpr
import implicit.gpu.bpr


def BayesianPersonalizedRanking(
    factors=100,
    learning_rate=0.01,
    regularization=0.01,
    dtype=np.float32,
    iterations=100,
    use_gpu=implicit.gpu.HAS_CUDA,
    num_threads=0,
    verify_negative_samples=True,
    random_state=None,
):
    """Bayesian Personalized Ranking

    A recommender model that learns  a matrix factorization embedding based off minimizing the
    pairwise ranking loss described in the paper `BPR: Bayesian Personalized Ranking from Implicit
    Feedback <https://arxiv.org/pdf/1205.2618.pdf>`_.

    This factory function returns either the cpu implementation from implicit.cpu.bpr or
    the gpu implementation from implicit.gpu.bpr depending on the value of the use_gpu flag.

    Parameters
    ----------
    factors : int, optional
        The number of latent factors to compute
    learning_rate : float, optional
        The learning rate to apply for SGD updates during training
    regularization : float, optional
        The regularization factor to use
    dtype : data-type, optional
        Specifies whether to generate 64 bit or 32 bit floating point factors
    use_gpu : bool, optional
        Fit on the GPU if available
    iterations : int, optional
        The number of training epochs to use when fitting the data
    verify_negative_samples: bool, optional
        When sampling negative items, check if the randomly picked negative item has actually
        been liked by the user. This check increases the time needed to train but usually leads
        to better predictions.
    num_threads : int, optional
        The number of threads to use for fitting the model. This only
        applies for the native extensions. Specifying 0 means to default
        to the number of cores on the machine.
    random_state : int, RandomState or None, optional
        The random state for seeding the initial item and user factors.
        Default is None.
    """

    if use_gpu:
        return implicit.gpu.bpr.BayesianPersonalizedRanking(
            factors,
            learning_rate,
            regularization,
            iterations=iterations,
            verify_negative_samples=verify_negative_samples,
            random_state=random_state,
        )
    return implicit.cpu.bpr.BayesianPersonalizedRanking(
        factors,
        learning_rate,
        regularization,
        dtype=dtype,
        num_threads=num_threads,
        iterations=iterations,
        verify_negative_samples=verify_negative_samples,
        random_state=random_state,
    )

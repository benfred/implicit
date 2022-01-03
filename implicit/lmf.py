import numpy as np

import implicit.cpu.lmf


def LogisticMatrixFactorization(
    factors=30,
    learning_rate=1.00,
    regularization=0.6,
    dtype=np.float32,
    iterations=30,
    neg_prop=30,
    use_gpu=False,
    num_threads=0,
    random_state=None,
):
    """Logistic Matrix Factorization

    A collaborative filtering recommender model that learns probabilistic distribution
    whether user like it or not. Algorithm of the model is described in
    `Logistic Matrix Factorization for Implicit Feedback Data
    <https://web.stanford.edu/~rezab/nips2014workshop/submits/logmat.pdf>`

    Parameters
    ----------
    factors : int, optional
        The number of latent factors to compute
    learning_rate : float, optional
        The learning rate to apply for updates during training
    regularization : float, optional
        The regularization factor to use
    dtype : data-type, optional
        Specifies whether to generate 64 bit or 32 bit floating point factors
    iterations : int, optional
        The number of training epochs to use when fitting the data
    neg_prop : int, optional
        The proportion of negative samples. i.e.) "neg_prop = 30" means if user have seen 5 items,
        then 5 * 30 = 150 negative samples are used for training.
    use_gpu : bool, optional
        Fit on the GPU if available
    num_threads : int, optional
        The number of threads to use for fitting the model. This only
        applies for the native extensions. Specifying 0 means to default
        to the number of cores on the machine.
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
    if use_gpu:
        raise NotImplementedError
    return implicit.cpu.lmf.LogisticMatrixFactorization(
        factors,
        learning_rate,
        regularization,
        dtype=dtype,
        iterations=iterations,
        neg_prop=neg_prop,
        num_threads=num_threads,
        random_state=random_state,
    )

import numpy as np

import implicit.cpu.als
import implicit.gpu.als


def AlternatingLeastSquares(
    factors=100,
    regularization=0.01,
    alpha=1.0,
    dtype=np.float32,
    use_native=True,
    use_cg=True,
    use_gpu=implicit.gpu.HAS_CUDA,
    iterations=15,
    calculate_training_loss=False,
    num_threads=0,
    random_state=None,
):
    """Alternating Least Squares

    A Recommendation Model based off the algorithms described in the paper 'Collaborative
    Filtering for Implicit Feedback Datasets' with performance optimizations described in
    'Applications of the Conjugate Gradient Method for Implicit Feedback Collaborative
    Filtering.'

    This factory function switches between the cpu and gpu implementations found in
    implicit.cpu.als.AlternatingLeastSquares and implicit.gpu.als.AlternatingLeastSquares
    depending on the use_gpu flag.

    Parameters
    ----------
    factors : int, optional
        The number of latent factors to compute
    regularization : float, optional
        The regularization factor to use
    alpha : float, optional
        The weight to give to positive examples.
    dtype : data-type, optional
        Specifies whether to generate 64 bit or 32 bit or 16 bit floating point factors
    use_native : bool, optional
        Use native extensions to speed up model fitting
    use_cg : bool, optional
        Use a faster Conjugate Gradient solver to calculate factors
    use_gpu : bool, optional
        Fit on the GPU if available, default is to run on GPU only if available
    iterations : int, optional
        The number of ALS iterations to use when fitting data
    calculate_training_loss : bool, optional
        Whether to log out the training loss at each iteration
    num_threads : int, optional
        The number of threads to use for fitting the model. This only
        applies for the native extensions. Specifying 0 means to default
        to the number of cores on the machine.
    random_state : int, np.random.RandomState or None, optional
        The random state for seeding the initial item and user factors.
        Default is None.
    """
    if use_gpu:
        return implicit.gpu.als.AlternatingLeastSquares(
            factors,
            regularization,
            alpha,
            dtype=dtype,
            iterations=iterations,
            calculate_training_loss=calculate_training_loss,
            random_state=random_state,
        )
    return implicit.cpu.als.AlternatingLeastSquares(
        factors,
        regularization,
        alpha,
        dtype,
        use_native,
        use_cg,
        iterations,
        calculate_training_loss,
        num_threads,
        random_state,
    )

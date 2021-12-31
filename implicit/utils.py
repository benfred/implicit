import logging
import os

import numpy as np


def nonzeros(m, row):
    """returns the non zeroes of a row in csr_matrix"""
    for index in range(m.indptr[row], m.indptr[row + 1]):
        yield m.indices[index], m.data[index]


_checked_blas_config = False


def check_blas_config():
    """checks to see if using OpenBlas/Intel MKL. If so, warn if the number of threads isn't set
    to 1 (causes severe perf issues when training - can be 10x slower)"""
    # don't warn repeatedly
    global _checked_blas_config  # pylint: disable=global-statement
    if _checked_blas_config:
        return
    _checked_blas_config = True

    if np.__config__.get_info("openblas_info") and os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        logging.warning(
            "OpenBLAS detected. Its highly recommend to set the environment variable "
            "'export OPENBLAS_NUM_THREADS=1' to disable its internal multithreading"
        )
    if np.__config__.get_info("blas_mkl_info") and os.environ.get("MKL_NUM_THREADS") != "1":
        logging.warning(
            "Intel MKL BLAS detected. Its highly recommend to set the environment "
            "variable 'export MKL_NUM_THREADS=1' to disable its internal "
            "multithreading"
        )


def check_random_state(random_state):
    """Validate the random state.

    Check a random seed or existing numpy RandomState
    and get back an initialized RandomState.

    Parameters
    ----------
    random_state : int, None or RandomState
        The existing RandomState. If None, or an int, will be used
        to seed a new numpy RandomState.
    """
    # if it's an existing random state, pass through
    if isinstance(random_state, np.random.RandomState):
        return random_state
    # otherwise try to initialize a new one, and let it fail through
    # on the numpy side if it doesn't work
    return np.random.RandomState(random_state)


def _batch_call(func, ids, *args, N=10, **kwargs):
    # we're running in batch mode, just loop over each item and call the scalar version of the
    # function
    output_ids = np.zeros((len(ids), N), dtype=np.int32)
    output_scores = np.zeros((len(ids), N), dtype=np.float32)

    for i, idx in enumerate(ids):
        batch_ids, batch_scores = func(idx, *args, N=N, **kwargs)

        # pad out to N items if we're returned fewer
        missing_items = N - len(batch_ids)
        if missing_items > 0:
            batch_ids = np.append(batch_ids, np.full(missing_items, -1))
            batch_scores = np.append(
                batch_scores, np.full(missing_items, -np.finfo(np.float32).max)
            )

        output_ids[i] = batch_ids[:N]
        output_scores[i] = batch_scores[:N]

    return output_ids, output_scores

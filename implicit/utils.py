import os
import warnings

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
        warnings.warn(
            "OpenBLAS detected. Its highly recommend to set the environment variable "
            "'export OPENBLAS_NUM_THREADS=1' to disable its internal multithreading"
        )
    if np.__config__.get_info("blas_mkl_info") and os.environ.get("MKL_NUM_THREADS") != "1":
        warnings.warn(
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


def augment_inner_product_matrix(factors):
    """This function transforms a factor matrix such that an angular nearest neighbours search
    will return top related items of the inner product.

    This involves transforming each row by adding one extra dimension as suggested in the paper:
    "Speeding Up the Xbox Recommender System Using a Euclidean Transformation for Inner-Product
    Spaces" https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/XboxInnerProduct.pdf

    Basically this involves transforming each feature vector so that they have the same norm, which
    means the cosine of this transformed vector is proportional to the dot product (if the other
    vector in the cosine has a 0 in the extra dimension)."""
    norms = np.linalg.norm(factors, axis=1)
    max_norm = norms.max()

    # add an extra dimension so that the norm of each row is the same
    # (max_norm)
    extra_dimension = np.sqrt(max_norm**2 - norms**2)
    return max_norm, np.append(factors, extra_dimension.reshape(norms.shape[0], 1), axis=1)


def _batch_call(func, ids, *args, N=10, **kwargs):
    # we're running in batch mode, just loop over each item and call the scalar version of the
    # function
    output_ids = np.zeros((len(ids), N), dtype=np.int32)
    output_scores = np.zeros((len(ids), N), dtype=np.float32)

    user_items = kwargs.pop("user_items") if "user_items" in kwargs else None
    item_users = kwargs.pop("item_users") if "item_users" in kwargs else None

    # pylint: disable=unsubscriptable-object
    for i, idx in enumerate(ids):
        current_kwargs = kwargs
        if user_items is not None:
            current_kwargs = dict(user_items=user_items[i], **kwargs)
        elif item_users is not None:
            current_kwargs = dict(item_users=item_users[i], **kwargs)

        batch_ids, batch_scores = func(idx, *args, N=N, **current_kwargs)

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


def _filter_items_from_results(queryid, ids, scores, filter_items, N):
    if np.isscalar(queryid):
        mask = np.in1d(ids, filter_items, invert=True)
        ids, scores = ids[mask][:N], scores[mask][:N]
    else:
        rows = len(queryid)
        filtered_scores = np.zeros((rows, N), dtype=scores.dtype)
        filtered_ids = np.zeros((rows, N), dtype=ids.dtype)
        for row in range(rows):
            mask = np.in1d(ids[row], filter_items, invert=True)
            filtered_ids[row] = ids[row][mask][:N]
            filtered_scores[row] = scores[row][mask][:N]
        ids, scores = filtered_ids, filtered_scores
    return ids, scores

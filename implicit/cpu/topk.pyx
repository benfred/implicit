import cython
import numpy as np

from cython cimport floating, integral

from cython.parallel import parallel, prange


cdef extern from "implicit/cpu/select.h" namespace "implicit" nogil:
    cdef void select[T](const T * batch,
                        int batch_rows, int batch_columns, int k,
                        int * ids, T * distances) noexcept nogil


def topk(items, query, int k, item_norms=None, filter_query_items=None, filter_items=None, int num_threads=0):
    if len(query.shape) == 1:
        query = query.reshape((1, len(query)))

    cdef int query_rows = query.shape[0]
    indices = np.zeros((query_rows, k), dtype="int32")
    distances = np.zeros((query_rows, k), dtype=query.dtype)

    # TODO: figure out appropriate batch size from available memory
    cdef int batch_size = 100 # TODO

    cdef int batches = (query_rows // batch_size)
    if query_rows % batch_size:
        batches += 1

    # if we're only running one batch, don't create a threadpool
    if batches == 1:
        _topk_batch(items, query, k, 0, query_rows, indices, distances, item_norms=item_norms, filter_query_items=filter_query_items, filter_items=filter_items)
        return indices, distances

    cdef int startidx, endidx, batch

    for batch in prange(batches, schedule="guided", num_threads=num_threads, nogil=True):
        startidx = batch * batch_size
        endidx = min(startidx + batch_size, query_rows)
        with gil:
            _topk_batch(items, query, k, startidx, endidx, indices, distances, item_norms=item_norms, filter_query_items=filter_query_items, filter_items=filter_items)

    return indices, distances

def _topk_batch(items, query, int k, int startidx, int endidx, int[:, :] indices, floating[:, :] distances,
                item_norms=None, filter_query_items=None, filter_items=None):
    batch_distances = query[startidx: endidx].dot(items.T)
    if item_norms is not None:
        batch_distances = batch_distances / item_norms

    neginf = -np.finfo(batch_distances.dtype).max
    if filter_query_items is not None:
        for i, idx in enumerate(range(startidx, endidx)):
            batch_distances[i, filter_query_items[idx].indices] = neginf
    if filter_items is not None:
        batch_distances[:, filter_items] = neginf

    cdef floating * c_distances = &distances[0, 0]
    cdef int * c_indices = &indices[0, 0]

    cdef floating[:, :] batch_view = batch_distances
    cdef floating * c_batch = &batch_view[0, 0]
    cdef int rows = batch_view.shape[0]
    cdef int cols = batch_view.shape[1]

    with nogil:
        select(c_batch, rows, cols, k, c_indices + startidx * k, c_distances + startidx * k)


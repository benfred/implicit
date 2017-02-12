import cython
import numpy as np
import scipy.sparse

from cython.operator import dereference
from cython.parallel import parallel, prange

from libcpp.vector cimport vector
from libcpp.utility cimport pair

cdef extern from "nearest_neighbours.h" namespace "implicit" nogil:
    cdef cppclass TopK[Index, Value]:
        TopK(size_t K)
        vector[pair[Value, Index]] results

    cdef cppclass SparseMatrixMultiplier[Index, Value]:
        SparseMatrixMultiplier(Index item_count)
        void add(Index index, Value value)
        void foreach[Function](Function & f)


@cython.boundscheck(False)
def all_pairs_knn(items, unsigned int K=100, int num_threads=0):
    """ Returns the top K nearest neighbours for each row in the matrix.
    """
    items = items.tocsr()
    users = items.T.tocsr()

    cdef int item_count = items.shape[0]
    cdef int i, u, index1, index2, j
    cdef double w1, w2

    cdef int[:] item_indptr = items.indptr, item_indices = items.indices
    cdef double[:] item_data = items.data

    cdef int[:] user_indptr = users.indptr, user_indices = users.indices
    cdef double[:] user_data = users.data

    cdef SparseMatrixMultiplier[int, double] * neighbours
    cdef TopK[int, double] * topk
    cdef pair[double, int] result

    # holds triples of output
    cdef double[:] values = np.zeros(item_count * K)
    cdef long[:] rows = np.zeros(item_count * K, dtype=int)
    cdef long[:] cols = np.zeros(item_count * K, dtype=int)

    with nogil, parallel(num_threads=num_threads):
        # allocate memory per thread
        neighbours = new SparseMatrixMultiplier[int, double](item_count)
        topk = new TopK[int, double](K)

        try:
            for i in prange(item_count, schedule='guided'):
                for index1 in range(item_indptr[i], item_indptr[i+1]):
                    u = item_indices[index1]
                    w1 = item_data[index1]

                    for index2 in range(user_indptr[u], user_indptr[u+1]):
                        neighbours.add(user_indices[index2], user_data[index2] * w1)

                topk.results.clear()
                neighbours.foreach(dereference(topk))

                index2 = K * i
                for result in topk.results:
                    rows[index2] = i
                    cols[index2] = result.second
                    values[index2] = result.first
                    index2 = index2 + 1

        finally:
            del neighbours
            del topk

    return scipy.sparse.coo_matrix((values, (rows, cols)),
                                   shape=(item_count, item_count))

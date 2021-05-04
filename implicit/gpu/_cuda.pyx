""" Various thin cython wrappers on top of CUDA functions """

import cython
import numpy as np
from cython.operator import dereference

from cython cimport view
from libcpp cimport bool
from libcpp.utility cimport move, pair

from .als cimport LeastSquaresSolver as CppLeastSquaresSolver
from .bpr cimport bpr_update as cpp_bpr_update
from .knn cimport KnnQuery as CppKnnQuery
from .matrix cimport COOMatrix as CppCOOMatrix
from .matrix cimport CSRMatrix as CppCSRMatrix
from .matrix cimport Matrix as CppMatrix
from .matrix cimport Vector as CppVector
from .matrix cimport calculate_norms as cpp_calculate_norms
from .random cimport RandomState as CppRandomState


cdef class RandomState(object):
    cdef CppRandomState * c_random

    def __cinit__(self, long seed=42):
        self.c_random = new CppRandomState(seed)

    def uniform(self, rows, cols, float low=0, float high=1.0):
        ret = Matrix(None)
        ret.c_matrix = new CppMatrix(self.c_random.uniform(rows, cols, low, high))
        return ret

    def randn(self, rows, cols, float mean=0, float stddev=1):
        ret = Matrix(None)
        ret.c_matrix = new CppMatrix(self.c_random.randn(rows, cols, mean, stddev))
        return ret

    def __dealloc__(self):
        del self.c_random


cdef class KnnQuery(object):
    cdef CppKnnQuery * c_knn

    def __cinit__(self, size_t max_temp_memory=500_000_000):
        self.c_knn = new CppKnnQuery(max_temp_memory)

    def __dealloc__(self):
        del self.c_knn

    def topk(self, Matrix items, Matrix m, int k, Matrix item_norms=None):
        cdef CppMatrix * queries = m.c_matrix
        cdef int rows = queries.rows
        cdef int[:, :] x
        cdef float[:, :] y

        cdef float * c_item_norms = NULL
        if item_norms:
            c_item_norms = item_norms.c_matrix.data

        indices = np.zeros((rows, k), dtype="int32")
        distances = np.zeros((rows, k), dtype="float32")
        x = indices
        y = distances

        self.c_knn.topk(dereference(items.c_matrix), dereference(queries), k,
                        &x[0, 0], &y[0, 0], c_item_norms)

        return indices, distances


cdef class Matrix(object):
    cdef CppMatrix * c_matrix

    def __cinit__(self, X):
        if X is None:
            self.c_matrix = NULL
            return

        cdef float[:, :] c_X
        cdef long data

        # see if the input support CAI (cupy/pytorch/cudf etc)
        cai = getattr(X, "__cuda_array_interface__", None)
        if cai:
            shape = cai["shape"]
            data = cai["data"][0]
            self.c_matrix = new CppMatrix(shape[0], shape[1], <float*>data, False)
        else:
            # otherwise assume we're a buffer on host
            c_X = X
            self.c_matrix = new CppMatrix(X.shape[0], X.shape[1], &c_X[0, 0], True)

    @property
    def shape(self):
        return self.c_matrix.rows, self.c_matrix.cols

    def __getitem__(self, idx):
        cdef int i
        cdef IntVector ids
        ret = Matrix(None)
        if isinstance(idx, slice):
            if idx.step and idx.step != 1:
                raise ValueError(f"Can't slice matrix with step {idx.step} yet")

            start = idx.start if idx.start is not None else 0
            stop = idx.stop if idx.stop is not None else self.c_matrix.rows
            ret.c_matrix = new CppMatrix(dereference(self.c_matrix), start, stop)

        elif isinstance(idx, int):
            i = idx
            ret.c_matrix = new CppMatrix(dereference(self.c_matrix), i)

        else:
            try:
                idx = np.array(idx).astype("int32")
            except Exception:
                raise ValueError(f"don't know how to handle __getitem__ on {idx}")

            if len(idx.shape) != 1:
                raise ValueError(f"don't know how to handle __getitem__ on {idx}")

            if ((idx < 0) | (idx >= self.c_matrix.rows)).any():
                raise ValueError(f"row id out of range for selecting items from matrix")

            ids = IntVector(idx)
            ret.c_matrix = new CppMatrix(dereference(self.c_matrix), dereference(ids.c_vector))

        return ret

    def to_numpy(self):
        ret = np.zeros((self.c_matrix.rows, self.c_matrix.cols), dtype="float32")
        cdef float[:, :] temp = ret
        self.c_matrix.to_host(&temp[0, 0])
        return ret

    def __repr__(self):
        return f"Matrix({str(self.to_numpy())})"

    def __str__(self):
        return str(self.to_numpy())

    def __dealloc__(self):
        if self.c_matrix is not NULL:
            del self.c_matrix


cdef class IntVector(object):
    cdef CppVector[int] * c_vector

    def __cinit__(self, int[:] data):
        self.c_vector = new CppVector[int](len(data), &data[0])

    def __dealloc__(self):
        del self.c_vector


cdef class CSRMatrix(object):
    cdef CppCSRMatrix * c_matrix

    def __cinit__(self, X):
        cdef int[:] indptr = X.indptr
        cdef int[:] indices = X.indices
        cdef float[:] data = X.data.astype(np.float32)
        self.c_matrix = new CppCSRMatrix(X.shape[0], X.shape[1], len(X.data),
                                         &indptr[0], &indices[0], &data[0])

    def __dealloc__(self):
        del self.c_matrix

cdef class COOMatrix(object):
    cdef CppCOOMatrix* c_matrix

    def __cinit__(self, X):
        cdef int[:] row = X.row
        cdef int[:] col = X.col
        cdef float[:] data = X.data.astype(np.float32)
        self.c_matrix = new CppCOOMatrix(X.shape[0], X.shape[1], len(X.data),
                                         &row[0], &col[0], &data[0])

    def __dealloc__(self):
        del self.c_matrix


cdef class LeastSquaresSolver(object):
    cdef CppLeastSquaresSolver * c_solver

    def __cinit__(self, int factors):
        self.c_solver = new CppLeastSquaresSolver(factors)

    def least_squares(self, CSRMatrix cui, Matrix X, Matrix Y,
                      float regularization, int cg_steps):
        self.c_solver.least_squares(dereference(cui.c_matrix), X.c_matrix, dereference(Y.c_matrix),
                                    regularization, cg_steps)

    def calculate_loss(self, CSRMatrix cui, Matrix X, Matrix Y,
                       float regularization):
        return self.c_solver.calculate_loss(dereference(cui.c_matrix), dereference(X.c_matrix),

                                            dereference(Y.c_matrix), regularization)

    def __dealloc__(self):
        del self.c_solver


def calculate_norms(Matrix items):
    ret = Matrix(None)
    ret.c_matrix = new CppMatrix(cpp_calculate_norms(dereference(items.c_matrix)))
    return ret


def bpr_update(IntVector userids, IntVector itemids, IntVector indptr,
               Matrix X, Matrix Y,
               float learning_rate, float regularization, long seed, bool verify_negative):
    ret = cpp_bpr_update(dereference(userids.c_vector),
                         dereference(itemids.c_vector),
                         dereference(indptr.c_vector),
                         X.c_matrix, Y.c_matrix,
                         learning_rate, regularization, seed, verify_negative)
    return ret.first, ret.second

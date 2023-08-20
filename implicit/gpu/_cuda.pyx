""" Various thin cython wrappers on top of CUDA functions """

import cython
import numpy as np
from cython.operator import dereference

from cython cimport view
from libc.stdint cimport uint16_t
from libcpp cimport bool
from libcpp.utility cimport move, pair

from implicit.utils import check_csr

from .als cimport LeastSquaresSolver as CppLeastSquaresSolver
from .bpr cimport bpr_update as cpp_bpr_update
from .knn cimport KnnQuery as CppKnnQuery
from .matrix cimport COOMatrix as CppCOOMatrix
from .matrix cimport CSRMatrix as CppCSRMatrix
from .matrix cimport Matrix as CppMatrix
from .matrix cimport Vector as CppVector
from .random cimport RandomState as CppRandomState
from .utils cimport get_device_count as cpp_get_device_count


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

    def __cinit__(self, size_t max_temp_memory=0):
        self.c_knn = new CppKnnQuery(max_temp_memory)

    def __dealloc__(self):
        del self.c_knn

    @cython.boundscheck(False)
    def topk(self, Matrix items, Matrix m, int k, Matrix item_norms=None,
             COOMatrix query_filter=None, IntVector item_filter=None):
        cdef CppMatrix * queries = m.c_matrix
        cdef CppCOOMatrix * c_query_filter = NULL
        cdef CppVector[int] * c_item_filter = NULL
        cdef size_t rows = queries.rows
        cdef int[:, :] indices_view
        cdef float[:, :] distances_view

        cdef CppMatrix * c_item_norms = NULL
        if item_norms is not None:
            c_item_norms = item_norms.c_matrix

        if query_filter is not None:
            c_query_filter = query_filter.c_matrix

        if item_filter is not None:
            c_item_filter = item_filter.c_vector

        indices = np.zeros((rows, k), dtype="int32")
        distances = np.zeros((rows, k), dtype="float32")
        indices_view = indices
        distances_view = distances

        with nogil:
            self.c_knn.topk(dereference(items.c_matrix), dereference(queries), k,
                            &indices_view[0, 0], &distances_view[0, 0], c_item_norms, c_query_filter, c_item_filter)

        return indices, distances

cdef class Matrix(object):
    cdef CppMatrix * c_matrix

    def __cinit__(self, X):
        if X is None:
            self.c_matrix = NULL
            return

        cdef float[:, :] temp_float
        cdef uint16_t[:, :] temp_half
        cdef long data

        # see if the input support CAI (cupy/pytorch/cudf etc)
        cai = getattr(X, "__cuda_array_interface__", None)
        if cai:
            shape = cai["shape"]
            data = cai["data"][0]
            itemsize = int(cai["typestr"][2])
            self.c_matrix = new CppMatrix(shape[0], shape[1], <void*>data, False, itemsize)
        else:
            # otherwise assume we're a buffer on host

            if X.dtype.char == "f":
                temp_float = X
                self.c_matrix = new CppMatrix(X.shape[0], X.shape[1], &temp_float[0, 0], True, 4)
            elif X.dtype.char == "e":
                temp_half = X.view(np.uint16)
                self.c_matrix = new CppMatrix(X.shape[0], X.shape[1], &temp_half[0, 0], True, 2)
            else:
                raise ValueError(f"unhandled dtype for GPU Matrix {X.dtype}")

    @classmethod
    def zeros(cls, rows, cols):
        ret = Matrix(None)
        ret.c_matrix = new CppMatrix(rows, cols, NULL, True, 4)
        return ret

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
                raise IndexError(f"don't know how to handle __getitem__ on {idx}")

            if len(idx.shape) == 0:
                idx = idx.reshape([1])

            if len(idx.shape) != 1:
                raise IndexError(f"don't know how to handle __getitem__ on {idx} - shape={idx.shape}")

            if ((idx < 0) | (idx >= self.c_matrix.rows)).any():
                raise IndexError(f"row id out of range for selecting items from matrix")

            ids = IntVector(idx)
            ret.c_matrix = new CppMatrix(dereference(self.c_matrix), dereference(ids.c_vector))

        return ret

    def assign_rows(self, rowids, Matrix other):
        cdef IntVector rows
        rows = IntVector(np.array(rowids).astype("int32"))
        self.c_matrix.assign_rows(dereference(rows.c_vector), dereference(other.c_matrix))

    def astype(self, dtype):
        dtype = np.dtype(dtype)
        _ALLOWED_DTYPES = (np.float16, np.float32)
        if dtype not in _ALLOWED_DTYPES:
            raise ValueError(f"Invalid dtype '{dtype}' for GPU model. "
                             f"Allowed dtypes are: {_ALLOWED_DTYPES}")

        cdef int itemsize = dtype.itemsize
        ret = Matrix(None)
        ret.c_matrix = new CppMatrix(self.c_matrix.astype(itemsize))
        return ret

    def resize(self, size_t rows, size_t cols):
        self.c_matrix.resize(rows, cols)

    def to_numpy(self):
        cdef float[:, :] temp_float
        cdef uint16_t[:, :] temp_half

        if self.c_matrix.itemsize == 4:
            ret = np.zeros((self.c_matrix.rows, self.c_matrix.cols), dtype="float32")
            temp_float = ret
            self.c_matrix.to_host(&temp_float[0, 0])
            return ret
        elif self.c_matrix.itemsize == 2:
            ret = np.zeros((self.c_matrix.rows, self.c_matrix.cols), dtype="float16")
            temp_half = ret.view(np.uint16)
            self.c_matrix.to_host(&temp_half[0, 0])
            return ret
        else:
            raise ValueError(f"Invalid itemsize {self.c_matrix.itemsize}")

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
        X = check_csr(X)

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

    def __cinit__(self):
        self.c_solver = new CppLeastSquaresSolver()

    def least_squares(self, CSRMatrix cui, Matrix X, Matrix YtY, Matrix Y, int cg_steps):
        with nogil:
            self.c_solver.least_squares(dereference(cui.c_matrix), X.c_matrix,
                                        dereference(YtY.c_matrix), dereference(Y.c_matrix),
                                        cg_steps)
    def calculate_loss(self, CSRMatrix cui, Matrix X, Matrix Y,
                       float regularization):
        cdef float loss
        with nogil:
            loss = self.c_solver.calculate_loss(dereference(cui.c_matrix), dereference(X.c_matrix),
                                                dereference(Y.c_matrix), regularization)
        return loss


    def calculate_yty(self, Matrix Y, Matrix YtY, float regularization):
        with nogil:
            self.c_solver.calculate_yty(dereference(Y.c_matrix), YtY.c_matrix, regularization)

    def __dealloc__(self):
        del self.c_solver


def calculate_norms(Matrix items):
    ret = Matrix(None)
    ret.c_matrix = new CppMatrix(items.c_matrix.calculate_norms())
    return ret


def get_device_count():
    return cpp_get_device_count()


def bpr_update(IntVector userids, IntVector itemids, IntVector indptr,
               Matrix X, Matrix Y,
               float learning_rate, float regularization, long seed, bool verify_negative):
    with nogil:
        ret = cpp_bpr_update(dereference(userids.c_vector),
                             dereference(itemids.c_vector),
                             dereference(indptr.c_vector),
                             X.c_matrix, Y.c_matrix,
                             learning_rate, regularization, seed, verify_negative)
    return ret.first, ret.second

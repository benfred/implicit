""" Various thin cython wrappers on top of CUDA functions """
import cython
import numpy as np
from cython.operator import dereference

from libcpp cimport bool
from libcpp.utility cimport pair

from .matrix cimport CSRMatrix as CppCSRMatrix
from .matrix cimport COOMatrix as CppCOOMatrix
from .matrix cimport Matrix as CppMatrix
from .matrix cimport Vector as CppVector

from .bpr cimport bpr_update as cpp_bpr_update
from .als cimport LeastSquaresSolver as CppLeastSquaresSolver

cdef class Matrix(object):
    cdef CppMatrix * c_matrix

    def __cinit__(self, X):
        cdef float[:, :] c_X
        cdef long data
        cai = getattr(X, "__cuda_array_interface__", None)
        if cai:
            shape = cai["shape"]
            data = cai["data"][0]
            self.c_matrix = new CppMatrix(shape[0], shape[1], <float*>data, False)
        else:
            # otherwise assume we're a buffer on host
            c_X = X
            self.c_matrix = new CppMatrix(X.shape[0], X.shape[1], &c_X[0, 0], True)

    def to_host(self, float[:, :] X):
        self.c_matrix.to_host(&X[0, 0])

    def __dealloc__(self):
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


def bpr_update(IntVector userids, IntVector itemids, IntVector indptr,
                  Matrix X, Matrix Y,
                  float learning_rate, float regularization, long seed, bool verify_negative):
    ret = cpp_bpr_update(dereference(userids.c_vector),
                         dereference(itemids.c_vector),
                         dereference(indptr.c_vector),
                         X.c_matrix, Y.c_matrix,
                         learning_rate, regularization, seed, verify_negative)
    return ret.first, ret.second

""" Various thin cython wrappers on top of CUDA functions """
import numpy as np
import cython
from cython.operator import dereference

cdef extern from "als.h" namespace "implicit" nogil:
    cdef cppclass CudaCSRMatrix:
        CudaCSRMatrix(int rows, int cols, int nonzeros,
                      const int * indptr, const int * indices, const float * data) except +

    cdef cppclass CudaCOOMatrix:
        CudaCOOMatrix(int rows, int cols, int nonzeros,
                      const int * row, const int * col, const float * data) except +

    cdef cppclass CudaDenseMatrix:
        CudaDenseMatrix(int rows, int cols, const float * data) except +
        void to_host(float * output) except +

    cdef cppclass CudaLeastSquaresSolver:
        CudaLeastSquaresSolver(int factors) except +
        void least_squares(const CudaCSRMatrix & Cui, CudaDenseMatrix * X,
                           const CudaDenseMatrix & Y, float regularization, int cg_steps) except +

        float calculate_loss(const CudaCSRMatrix & Cui, const CudaDenseMatrix & X,
                             const CudaDenseMatrix & Y, float regularization) except +


cdef extern from "bpr.h" namespace "implicit" nogil:
    cdef int bpr_update(const CudaCOOMatrix & Ciu,
                        CudaDenseMatrix * X,
                        CudaDenseMatrix * Y,
                        float learning_rate, float regularization, long seed) except +


cdef class CuDenseMatrix(object):
    cdef CudaDenseMatrix* c_matrix

    def __cinit__(self, float[:, :] X):
        self.c_matrix = new CudaDenseMatrix(X.shape[0], X.shape[1], &X[0, 0])

    def to_host(self, float[:, :] X):
        self.c_matrix.to_host(&X[0, 0])

    def __dealloc__(self):
        del self.c_matrix


cdef class CuCSRMatrix(object):
    cdef CudaCSRMatrix* c_matrix

    def __cinit__(self, X):
        cdef int[:] indptr = X.indptr
        cdef int[:] indices = X.indices
        cdef float[:] data = X.data.astype(np.float32)
        self.c_matrix = new CudaCSRMatrix(X.shape[0], X.shape[1], len(X.data),
                                          &indptr[0], &indices[0], &data[0])

    def __dealloc__(self):
        del self.c_matrix

cdef class CuCOOMatrix(object):
    cdef CudaCOOMatrix* c_matrix

    def __cinit__(self, X):
        cdef int[:] row = X.row
        cdef int[:] col = X.col
        cdef float[:] data = X.data.astype(np.float32)
        self.c_matrix = new CudaCOOMatrix(X.shape[0], X.shape[1], len(X.data),
                                          &row[0], &col[0], &data[0])

    def __dealloc__(self):
        del self.c_matrix


cdef class CuLeastSquaresSolver(object):
    cdef CudaLeastSquaresSolver * c_solver

    def __cinit__(self, int factors):
        self.c_solver = new CudaLeastSquaresSolver(factors)

    def least_squares(self, CuCSRMatrix cui, CuDenseMatrix X, CuDenseMatrix Y,
                      float regularization, int cg_steps):
        self.c_solver.least_squares(dereference(cui.c_matrix), X.c_matrix, dereference(Y.c_matrix),
                                    regularization, cg_steps)

    def calculate_loss(self, CuCSRMatrix cui, CuDenseMatrix X, CuDenseMatrix Y,
                       float regularization):
        return self.c_solver.calculate_loss(dereference(cui.c_matrix), dereference(X.c_matrix),

                                            dereference(Y.c_matrix), regularization)

    def __dealloc__(self):
        del self.c_solver


def cu_bpr_update(CuCOOMatrix ciu, CuDenseMatrix X, CuDenseMatrix Y,
                  float learning_rate, float regularization, long seed):
    return bpr_update(dereference(ciu.c_matrix), X.c_matrix, Y.c_matrix,
                      learning_rate, regularization, seed)

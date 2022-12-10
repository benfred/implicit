from .matrix cimport CSRMatrix, Matrix


cdef extern from "implicit/gpu/als.h" namespace "implicit::gpu" nogil:
    cdef cppclass LeastSquaresSolver:
        LeastSquaresSolver() except +

        void calculate_yty(const Matrix & Y, Matrix * YtY, float regularization) except +

        void least_squares(const CSRMatrix & Cui, Matrix * X,
                           const Matrix & YtY, const Matrix & Y,
                           int cg_steps) except +

        float calculate_loss(const CSRMatrix & Cui, const Matrix & X,
                             const Matrix & Y, float regularization) except +

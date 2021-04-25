from .matrix cimport CSRMatrix, Matrix


cdef extern from "als.h" namespace "implicit::gpu" nogil:
    cdef cppclass LeastSquaresSolver:
        LeastSquaresSolver(int factors) except +
        void least_squares(const CSRMatrix & Cui, Matrix * X,
                           const Matrix & Y, float regularization, int cg_steps) except +

        float calculate_loss(const CSRMatrix & Cui, const Matrix & X,
                             const Matrix & Y, float regularization) except +

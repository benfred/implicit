import numpy
import cython
from cython.parallel import parallel, prange
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy

# requires scipy v0.16
cimport scipy.linalg.cython_lapack as cython_lapack
cimport scipy.linalg.cython_blas as cython_blas

@cython.boundscheck(False)
def least_squares(Cui, double [:, :] X, double [:, :] Y, double regularization, int num_threads):
    cdef int [:] indptr = Cui.indptr, indices = Cui.indices
    cdef double [:] data = Cui.data

    cdef int users = X.shape[0], factors = X.shape[1], u, i, j, index, err, one = 1
    cdef double confidence, temp

    YtY = numpy.dot(numpy.transpose(Y), Y)

    cdef double[:, :] initialA = YtY + regularization * numpy.eye(factors)
    cdef double[:] initialB = numpy.zeros(factors)

    cdef double * A
    cdef double * b
    cdef int * pivot

    with nogil, parallel(num_threads = num_threads):
        # allocate temp memory for each thread
        A = <double *> malloc(sizeof(double) * factors * factors)
        b = <double *> malloc(sizeof(double) * factors)
        pivot = <int *> malloc(sizeof(int) * factors)
        try:
            for u in prange(users, schedule='guided'):
                # For each user u calculate
                # Xu = (YtCuY + regularization*I)i^-1 * YtYCuPu

                # Build up A = YtCuY + reg * I and b = YtCuPu
                memcpy(A, &initialA[0, 0], sizeof(double) * factors * factors)
                memcpy(b, &initialB[0], sizeof(double) * factors)

                for index in range(indptr[u], indptr[u+1]):
                    i = indices[index]
                    confidence = data[index]

                    # b += Yi Cui Pui
                    # Pui is implicit, its defined to be 1 for non-zero entries
                    cython_blas.daxpy(&factors, &confidence, &Y[i, 0], &one, b, &one)

                    # A += Yi^T Cui Yi
                    # Since we've already added in YtY, we subtract 1 from confidence
                    for j in range(factors):
                        temp = (confidence - 1) * Y[i, j]
                        cython_blas.daxpy(&factors, &temp, &Y[i, 0], &one, A + j * factors, &one)

                cython_lapack.dposv("U", &factors, &one, A, &factors, b, &factors, &err);

                # fall back to using a LU decomposition if this fails
                if err:
                    cython_lapack.dgesv(&factors, &one, A, &factors, pivot, b, &factors, &err)

                if not err:
                    memcpy(&X[u, 0], b, sizeof(double) * factors)

                else:
                    with gil:
                        raise ValueError("Singular matrix (err=%i) on row %i" % (err, u))

        finally:
            free(A)
            free(b)
            free(pivot)

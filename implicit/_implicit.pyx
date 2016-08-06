import numpy
import cython
from cython cimport floating
from cython.parallel import parallel, prange
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy

# requires scipy v0.16
cimport scipy.linalg.cython_lapack as cython_lapack
cimport scipy.linalg.cython_blas as cython_blas

# lapack/blas wrappers for cython fused types
cdef inline void axpy(int * n, floating * da, floating * dx, int * incx, floating * dy, int * incy) nogil:
    if floating is double:
        cython_blas.daxpy(n, da, dx, incx, dy, incy)
    else:
        cython_blas.saxpy(n, da, dx, incx, dy, incy)

cdef inline void posv(char * u, int * n, int * nrhs, floating * a, int * lda, floating * b, int * ldb, int * info) nogil:
    if floating is double:
        cython_lapack.dposv(u, n, nrhs, a, lda, b, ldb, info)
    else:
        cython_lapack.sposv(u, n, nrhs, a, lda, b, ldb, info)

cdef inline void gesv(int * n, int * nrhs, floating * a, int * lda, int * piv, floating * b, int * ldb, int * info) nogil:
    if floating is double:
        cython_lapack.dgesv(n, nrhs, a, lda, piv, b, ldb, info)
    else:
        cython_lapack.sgesv(n, nrhs, a, lda, piv, b, ldb, info)


@cython.boundscheck(False)
def least_squares(Cui, floating [:, :] X, floating [:, :] Y, double regularization, int num_threads):
    dtype = numpy.float64 if floating is double else numpy.float32

    cdef int [:] indptr = Cui.indptr, indices = Cui.indices
    cdef double [:] data = Cui.data

    cdef int users = X.shape[0], factors = X.shape[1], u, i, j, index, err, one = 1
    cdef floating confidence, temp

    YtY = numpy.dot(numpy.transpose(Y), Y)

    cdef floating[:, :] initialA = YtY + regularization * numpy.eye(factors, dtype=dtype)
    cdef floating[:] initialB = numpy.zeros(factors, dtype=dtype)

    cdef floating * A
    cdef floating * b
    cdef int * pivot

    with nogil, parallel(num_threads = num_threads):
        # allocate temp memory for each thread
        A = <floating *> malloc(sizeof(floating) * factors * factors)
        b = <floating *> malloc(sizeof(floating) * factors)
        pivot = <int *> malloc(sizeof(int) * factors)
        try:
            for u in prange(users, schedule='guided'):
                # For each user u calculate
                # Xu = (YtCuY + regularization*I)i^-1 * YtYCuPu

                # Build up A = YtCuY + reg * I and b = YtCuPu
                memcpy(A, &initialA[0, 0], sizeof(floating) * factors * factors)
                memcpy(b, &initialB[0], sizeof(floating) * factors)

                for index in range(indptr[u], indptr[u+1]):
                    i = indices[index]
                    confidence = data[index]

                    # b += Yi Cui Pui
                    # Pui is implicit, its defined to be 1 for non-zero entries
                    axpy(&factors, &confidence, &Y[i, 0], &one, b, &one)

                    # A += Yi^T Cui Yi
                    # Since we've already added in YtY, we subtract 1 from confidence
                    for j in range(factors):
                        temp = (confidence - 1) * Y[i, j]
                        axpy(&factors, &temp, &Y[i, 0], &one, A + j * factors, &one)

                posv("U", &factors, &one, A, &factors, b, &factors, &err);

                # fall back to using a LU decomposition if this fails
                if err:
                    gesv(&factors, &one, A, &factors, pivot, b, &factors, &err)

                if not err:
                    memcpy(&X[u, 0], b, sizeof(floating) * factors)

                else:
                    with gil:
                        raise ValueError("Singular matrix (err=%i) on row %i" % (err, u))

        finally:
            free(A)
            free(b)
            free(pivot)

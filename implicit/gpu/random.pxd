from .matrix cimport Matrix


cdef extern from "implicit/gpu/random.h" namespace "implicit::gpu" nogil:
    cdef cppclass RandomState:
        RandomState(long rows) except +
        Matrix uniform(size_t rows, size_t cols, float low, float high) except +
        Matrix randn(size_t rows, size_t cols, float mean, float stdev) except +

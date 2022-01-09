from .matrix cimport Matrix


cdef extern from "implicit/gpu/random.h" namespace "implicit::gpu" nogil:
    cdef cppclass RandomState:
        RandomState(long rows) except +
        Matrix uniform(int rows, int cols, float low, float high) except +
        Matrix randn(int rows, int cols, float mean, float stdev) except +

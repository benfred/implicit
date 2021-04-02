from libcpp cimport bool

cdef extern from "matrix.h" namespace "implicit::gpu" nogil:
    cdef cppclass CSRMatrix:
        CSRMatrix(int rows, int cols, int nonzeros,
                  const int * indptr, const int * indices, const float * data) except +

    cdef cppclass COOMatrix:
        COOMatrix(int rows, int cols, int nonzeros,
                  const int * row, const int * col, const float * data) except +

    cdef cppclass Vector[T]:
        Vector(int size, T * data)

    cdef cppclass Matrix:
        Matrix(int rows, int cols, float * data, bool host) except +
        void to_host(float * output) except +


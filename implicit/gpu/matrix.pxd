from libcpp cimport bool


cdef extern from "matrix.h" namespace "implicit::gpu" nogil:
    cdef cppclass CSRMatrix:
        CSRMatrix(int rows, int cols, int nonzeros,
                  const int * indptr, const int * indices, const float * data) except +

    cdef cppclass COOMatrix:
        COOMatrix(int rows, int cols, int nonzeros,
                  const int * row, const int * col, const float * data) except +

    cdef cppclass Vector[T]:
        Vector(int size, T * data) except +
        void to_host(T * output) except +
        T * data
        int size

    cdef cppclass Matrix:
        Matrix(int rows, int cols, float * data, bool host) except +
        Matrix(const Matrix & other, int rowid) except +
        Matrix(const Matrix & other, int start, int end) except +
        Matrix(const Matrix & other, const Vector[int] & rowids) except +
        Matrix(Matrix && other) except +
        void to_host(float * output) except +
        int rows, cols
        float * data

    Matrix calculate_norms(const Matrix & items) except +

    cdef cppclass KnnQuery:
        KnnQuery()
        void query(const Matrix & items, const Matrix & queries, int k,
                   int * indices, float * distances) except +
        void argpartition(const Matrix & items, int k, int * indices, float * distances) except +
        void argsort(Matrix * items, int * indices) except +

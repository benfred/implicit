from libcpp cimport bool


cdef extern from "implicit/gpu/matrix.h" namespace "implicit::gpu" nogil:
    cdef cppclass CSRMatrix:
        CSRMatrix(int rows, int cols, int nonzeros,
                  const int * indptr, const int * indices, const float * data) except +

    cdef cppclass COOMatrix:
        COOMatrix(int rows, int cols, int nonzeros,
                  const int * row, const int * col, const float * data) except +

    cdef cppclass Vector[T]:
        Vector(size_t size, T * data) except +
        void to_host(T * output) except +
        T * data
        size_t size

    cdef cppclass Matrix:
        Matrix(size_t rows, size_t cols, void * data, bool host, size_t itemsize) except +
        Matrix(const Matrix & other, size_t rowid) except +
        Matrix(const Matrix & other, size_t start, size_t end) except +
        Matrix(const Matrix & other, const Vector[int] & rowids) except +
        Matrix(Matrix && other) except +
        void to_host(void * output) except +
        void resize(size_t rows, size_t cols) except +
        void assign_rows(const Vector[int] & rowids, const Matrix & other) except +
        Matrix calculate_norms() except +
        Matrix astype(size_t itemsize) except +
        size_t rows, cols
        void * data
        size_t itemsize

    cdef cppclass KnnQuery:
        KnnQuery()
        void query(const Matrix & items, const Matrix & queries, int k,
                   int * indices, float * distances) except +
        void argpartition(const Matrix & items, int k, int * indices, float * distances) except +
        void argsort(Matrix * items, int * indices) except +

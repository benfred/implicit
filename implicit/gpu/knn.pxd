from .matrix cimport COOMatrix, Matrix, Vector


cdef extern from "implicit/gpu/knn.h" namespace "implicit::gpu" nogil:
    cdef cppclass KnnQuery:
        KnnQuery(size_t max_temp_memory) except +

        void topk(const Matrix & items, const Matrix & query, int k,
                  int * indices, float * distances,
                  Matrix * item_norms,
                  COOMatrix * query_filter,
                  Vector[int] * item_filter) except +

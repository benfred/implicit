from .matrix cimport Matrix


cdef extern from "knn.h" namespace "implicit::gpu" nogil:
    cdef cppclass KnnQuery:
        KnnQuery(size_t max_temp_memory) except +

        void topk(const Matrix & items, const Matrix & query, int k,
                  int * indices, float * distances, float * item_norms) except +

#ifndef IMPLICIT_GPU_KNN_H_
#define IMPLICIT_GPU_KNN_H_
#include <memory>

#include "implicit/gpu/matrix.h"

struct cublasContext;

namespace implicit { namespace gpu {
struct StackAllocator;

class KnnQuery {
public:
    KnnQuery(size_t temp_memory=512000000);
    ~KnnQuery();
    cublasContext * blas_handle;

    void topk(const Matrix & items, const Matrix & query, int k,
              int * indices, float * distances,
              float * item_norms = NULL);

    void argpartition(const Matrix & items, int k, int * indices, float * distances);
    void argsort(const Matrix & items, int * indices, float * distances);

protected:
    size_t max_temp_memory;
    std::unique_ptr<StackAllocator> alloc;
};
}}  // namespace implicit/gpu
#endif  // IMPLICIT_GPU_KNN_H_

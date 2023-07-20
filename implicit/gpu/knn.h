#ifndef IMPLICIT_GPU_KNN_H_
#define IMPLICIT_GPU_KNN_H_
#include <memory>

#include <raft/core/resources.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

#include "implicit/gpu/matrix.h"

namespace implicit {
namespace gpu {
class KnnQuery {
public:
  KnnQuery(size_t temp_memory = 0);
  ~KnnQuery();

  void topk(const Matrix &items, const Matrix &query, int k, int *indices,
            float *distances, Matrix *item_norms = NULL,
            const COOMatrix *query_filter = NULL,
            Vector<int> *item_filter = NULL);

  template <typename T>
  void topk_impl(const Matrix &items, const Matrix &query, int k, int *indices,
                 float *distances, Matrix *item_norms = NULL,
                 const COOMatrix *query_filter = NULL,
                 Vector<int> *item_filter = NULL);

protected:
  std::unique_ptr<rmm::mr::device_memory_resource> mr;
  raft::resources handle;
  size_t max_temp_memory;
};
} // namespace gpu
} // namespace implicit
#endif // IMPLICIT_GPU_KNN_H_

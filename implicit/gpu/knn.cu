#include <vector>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <cuda_runtime.h>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/transform.h>

#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/matrix/select_k.cuh>

#include "implicit/gpu/knn.h"
#include "implicit/gpu/utils.h"

namespace implicit {
namespace gpu {
namespace {
// faiss seems to have issues when distances contain -FLT_MAX, and can return a
// '-1' in the indices returned, instead of an actual valid row number. When we
// filter, instead of setting to -FLT_MAX, set to the next smallest valid
// float32 value.
const static float _FLT_MAX = FLT_MAX;
const static uint32_t UINT_FILTER_DISTANCE =
    (*reinterpret_cast<const uint32_t *>(&_FLT_MAX)) - 1;
const static float FLT_FILTER_DISTANCE =
    -*reinterpret_cast<const float *>(&UINT_FILTER_DISTANCE);
} // namespace

bool is_host_memory(void *address) {
  cudaPointerAttributes attr;
  auto err = cudaPointerGetAttributes(&attr, address);
  if (err == cudaErrorInvalidValue) {
    return true;
  }

#if __CUDACC_VER_MAJOR__ >= 10
  return attr.type == cudaMemoryTypeHost ||
         attr.type == cudaMemoryTypeUnregistered;
#else
  return attr.memoryType == cudaMemoryTypeHost ||
         attr.memoryType == cudaMemoryTypeUnregistered;
#endif
}

KnnQuery::KnnQuery(size_t temp_memory) {
  if (!temp_memory) {
    // use half of free GPU memory, limited to 4GB max
    size_t free, total;
    CHECK_CUDA(cudaMemGetInfo(&free, &total));
    temp_memory = std::min(free / 2, size_t{4294967296});
  }

  // pad out to 256 bytes if necessary
  size_t padding = temp_memory % 256;
  if (padding) {
    temp_memory += 256 - padding;
  }

  max_temp_memory = temp_memory;

  static rmm::mr::cuda_memory_resource upstream_mr;
  mr.reset(new rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>(
      &upstream_mr, max_temp_memory, max_temp_memory));
}

void KnnQuery::topk(const Matrix &items, const Matrix &query, int k,
                    int *indices, float *distances, Matrix *item_norms,
                    const COOMatrix *query_filter, Vector<int> *item_filter) {
  if (query.cols != items.cols) {
    throw std::invalid_argument(
        "Must have same number of columns in each matrix for topk");
  }

  if (query.itemsize != items.itemsize) {
    throw std::invalid_argument("Must have same dtype in each matrix for topk");
  }

  if (items.itemsize == 4) {
    topk_impl<float>(items, query, k, indices, distances, item_norms,
                     query_filter, item_filter);
  } else if (items.itemsize == 2) {
    topk_impl<half>(items, query, k, indices, distances, item_norms,
                    query_filter, item_filter);
  } else {
    throw std::invalid_argument("invalid itemsize for topk");
  }
}

namespace {

template <typename T>
void gemm(cublasContext *blas_handle, const Matrix &items, const Matrix &batch,
          Matrix *output);

template <>
void gemm<float>(cublasContext *blas_handle, const Matrix &items,
                 const Matrix &batch, Matrix *output) {
  // matrix multiple the items by the batch, store in distances
  float alpha = 1.0, beta = 0.;

  CHECK_CUBLAS(cublasSgemm(blas_handle, CUBLAS_OP_T, CUBLAS_OP_N, items.rows,
                           batch.rows, items.cols, &alpha, items, items.cols,
                           batch, batch.cols, &beta, *output, output->cols));
}

template <>
void gemm<half>(cublasContext *blas_handle, const Matrix &items,
                const Matrix &batch, Matrix *output) {
  // matrix multiple the items by the batch, store in distances
  float alpha = 1.0, beta = 0.;

  // our factors are float16, but we accumulate into a float32
  CHECK_CUBLAS(cublasSgemmEx(
      blas_handle, CUBLAS_OP_T, CUBLAS_OP_N, items.rows, batch.rows, items.cols,
      &alpha, items.data, CUDA_R_16F, items.cols, batch.data, CUDA_R_16F,
      batch.cols, &beta, output->data, CUDA_R_32F, output->cols));
}
} // namespace

template <typename T>
void KnnQuery::topk_impl(const Matrix &items, const Matrix &query, int k,
                         int *indices, float *distances, Matrix *item_norms,
                         const COOMatrix *query_filter,
                         Vector<int> *item_filter) {
  auto stream = raft::resource::get_cuda_stream(handle);

  size_t available_temp_memory = max_temp_memory;

  // limit to temp memory 8GB or so (causes some issues if we have over 2^31
  // entries in our matrix)
  available_temp_memory = std::min(available_temp_memory, size_t{4294967296});

  float *host_distances = NULL;
  std::unique_ptr<rmm::device_uvector<float>> distances_storage;
  size_t distances_size = query.rows * k * sizeof(float);
  if (is_host_memory(distances)) {
    host_distances = distances;
    distances_storage.reset(
        new rmm::device_uvector<float>(query.rows * k, stream, mr.get()));
    distances = distances_storage->data();
    available_temp_memory -= distances_size;
  }

  int *host_indices = NULL;
  std::unique_ptr<rmm::device_uvector<int>> indices_storage;
  size_t indices_size = query.rows * k * sizeof(int);
  if (is_host_memory(indices)) {
    host_indices = indices;
    indices_storage.reset(
        new rmm::device_uvector<int>(query.rows * k, stream, mr.get()));
    indices = indices_storage->data();
    available_temp_memory -= indices_size;
  }

  // extra temp memory for sorting the output
  available_temp_memory -= (query.rows * k * (sizeof(float) + sizeof(int))) * 3;

  size_t batch_size = (available_temp_memory /
                       (sizeof(float) * static_cast<size_t>(items.rows)));

  batch_size = std::min(batch_size, query.rows);
  batch_size = std::max(batch_size, static_cast<size_t>(1));

  rmm::device_uvector<T> temp_mem(batch_size * items.rows, stream, mr.get());
  Matrix temp_distances(batch_size, items.rows, temp_mem.data(), false);

  for (int start = 0; start < query.rows; start += batch_size) {
    auto end = std::min(query.rows, start + batch_size);

    Matrix batch(query, start, end);
    temp_distances.rows = batch.rows;

    auto blas_handle = raft::resource::get_cublas_handle(handle);
    gemm<T>(blas_handle, items, batch, &temp_distances);

    auto thrust_policy = raft::resource::get_thrust_policy(handle);
    // If we have norms (cosine distance etc) normalize the results here
    if (item_norms != NULL) {
      auto count = thrust::make_counting_iterator<size_t>(0);
      int cols = temp_distances.cols;
      int item_norm_cols = items.rows;
      float *data = temp_distances;
      const float *item_norms_data = *item_norms;
      thrust::for_each(thrust_policy, count,
                       count + (static_cast<size_t>(temp_distances.rows) *
                                static_cast<size_t>(temp_distances.cols)),
                       [=] __device__(size_t i) {
                         int col = i % cols;
                         if (col < item_norm_cols) {
                           data[i] /= item_norms_data[col];
                         }
                       });
    }

    if (item_filter != NULL) {
      auto count = thrust::make_counting_iterator<size_t>(0);
      float *data = temp_distances;
      int *items = item_filter->data;
      int items_size = item_filter->size;
      int cols = temp_distances.cols;

      float filter_distance = FLT_FILTER_DISTANCE;
      thrust::for_each(thrust_policy, count,
                       count + items_size * temp_distances.rows,
                       [=] __device__(size_t i) {
                         int col = items[i % items_size];
                         int row = i / items_size;
                         data[row * cols + col] = filter_distance;
                       });
    }

    if (query_filter != NULL) {
      auto count = thrust::make_counting_iterator<size_t>(0);
      int *row = query_filter->row;
      int *col = query_filter->col;
      float *data = temp_distances;
      int items = temp_distances.cols;
      float filter_distance = FLT_FILTER_DISTANCE;
      thrust::for_each(thrust_policy, count, count + query_filter->nonzeros,
                       [=] __device__(int i) {
                         if ((row[i] >= start) && (row[i] < end)) {
                           data[(row[i] - start) * items + col[i]] =
                               filter_distance;
                         }
                       });
    }

    auto current_k = std::min(k, static_cast<int>(temp_distances.cols));

    auto distance_view = raft::make_device_matrix_view<const float, int64_t>(
        temp_distances, temp_distances.rows, temp_distances.cols);

    raft::matrix::select_k<float, int>(
        handle, distance_view, std::nullopt,
        raft::make_device_matrix_view<float, int64_t>(
            distances + start * k, temp_distances.rows, current_k),
        raft::make_device_matrix_view<int, int64_t>(
            indices + start * k, temp_distances.rows, current_k),
        false, true);
    // TODO: callback per batch (show progress etc)
  }

  raft::resource::sync_stream(handle);

  if (host_indices) {
    CHECK_CUDA(cudaMemcpy(host_indices, indices, indices_size,
                          cudaMemcpyDeviceToHost));
  }

  if (host_distances) {
    CHECK_CUDA(cudaMemcpy(host_distances, distances, distances_size,
                          cudaMemcpyDeviceToHost));
  }
}

KnnQuery::~KnnQuery() {}

} // namespace gpu
} // namespace implicit

#include <vector>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <faiss/gpu/utils/BlockSelectKernel.cuh>
#include <faiss/gpu/utils/DeviceTensor.cuh>
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

#include "implicit/gpu/knn.h"
#include "implicit/gpu/utils.h"

namespace implicit {
namespace gpu {
namespace {
const static int TILE_GROUPS = 32;
const static int MAX_TILE_ROWS = 32;

template <typename T> inline constexpr float NEG_FLT_MAX = -FLT_MAX;
template <> inline constexpr float NEG_FLT_MAX<half> = -65500.0;

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

template <typename T>
void copy_columns(const T *input, int rows, int cols, T *output,
                  int output_cols) {
  auto count = thrust::make_counting_iterator<int>(0);
  thrust::for_each(count, count + (rows * output_cols), [=] __device__(int i) {
    int col = i % output_cols;
    int row = i / output_cols;
    output[col + row * output_cols] = input[col + row * cols];
  });
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

  CHECK_CUBLAS(cublasCreate(&blas_handle));
}

void KnnQuery::topk(const Matrix &items, const Matrix &query, int k,
                    int *indices, void *distances, Matrix *item_norms,
                    const COOMatrix *query_filter, Vector<int> *item_filter) {
  if (query.cols != items.cols) {
    throw std::invalid_argument(
        "Must have same number of columns in each matrix for topk");
  }

  if (query.itemsize != items.itemsize) {
    throw std::invalid_argument("Must have same dtype in each matrix for topk");
  }

  if (items.itemsize == 4) {
    topk_impl<float>(items, query, k, indices,
                     reinterpret_cast<float *>(distances), item_norms,
                     query_filter, item_filter);
  } else if (items.itemsize == 2) {
    topk_impl<half>(items, query, k, indices,
                    reinterpret_cast<half *>(distances), item_norms,
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
  half alpha = 1.0, beta = 0.;

  half *output_data = *output;

  CHECK_CUBLAS(cublasHgemm(blas_handle, CUBLAS_OP_T, CUBLAS_OP_N, items.rows,
                           batch.rows, items.cols, &alpha, items, items.cols,
                           batch, batch.cols, &beta, output_data,
                           output->cols));
}
} // namespace

template <typename T>
void KnnQuery::topk_impl(const Matrix &items, const Matrix &query, int k,
                         int *indices, T *distances, Matrix *item_norms,
                         const COOMatrix *query_filter,
                         Vector<int> *item_filter) {
  rmm::cuda_stream_view stream;

  size_t available_temp_memory = max_temp_memory;

  // limit to temp memory 8GB or so (causes some issues if we have over 2^31
  // entries in our matrix)
  available_temp_memory = std::min(available_temp_memory, size_t{4294967296});

  T *host_distances = NULL;
  std::unique_ptr<rmm::device_uvector<T>> distances_storage;
  size_t distances_size = query.rows * k * sizeof(T);
  if (is_host_memory(distances)) {
    host_distances = distances;
    distances_storage.reset(
        new rmm::device_uvector<T>(query.rows * k, stream, mr.get()));
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

  // Create temporary memory for storing results. We're padding out temp memory
  // so that we can tile the columns (break up a single row to multiple top-k
  // operations) if there aren't many rows in the input
  size_t temp_distances_cols = items.rows;
  size_t padding = 0;

  // just in case we're tiling each row, we'l need some temp memory for that too
  size_t tile_memory =
      TILE_GROUPS * MAX_TILE_ROWS * k * (sizeof(T) + sizeof(int));
  bool allow_tiling = tile_memory * 4 < available_temp_memory;
  if (allow_tiling) {
    padding = temp_distances_cols % TILE_GROUPS;
    if (padding) {
      temp_distances_cols += TILE_GROUPS - padding;
    }
    available_temp_memory -= tile_memory;
  }

  // We need 6 copies of the matrix for argsort code - and then some
  // extra memory per SM as well.
  size_t batch_size = (available_temp_memory /
                       (sizeof(T) * static_cast<size_t>(temp_distances_cols)));
  if (k >= GPU_MAX_SELECTION_K) {
    batch_size *= 0.15;
  }

  batch_size = std::min(batch_size, static_cast<size_t>(query.rows));
  batch_size = std::max(batch_size, static_cast<size_t>(1));

  rmm::device_uvector<T> temp_mem(batch_size * temp_distances_cols, stream,
                                  mr.get());

  Matrix temp_distances(batch_size, temp_distances_cols, temp_mem.data(), false,
                        items.itemsize);

  // Fill temp_distances if we're padding so that results don't appear
  if (padding) {
    T *temp_data = temp_distances;
    thrust::device_ptr<T> data = thrust::device_pointer_cast(temp_data);
    thrust::fill(data, data + temp_distances.rows * temp_distances.cols,
                 NEG_FLT_MAX<T>);
  }

  for (int start = 0; start < query.rows; start += batch_size) {
    auto end = std::min(query.rows, start + static_cast<int>(batch_size));

    Matrix batch(query, start, end);
    temp_distances.rows = batch.rows;

    gemm<T>(blas_handle, items, batch, &temp_distances);

    // If we have norms (cosine distance etc) normalize the results here
    if (item_norms != NULL) {
      auto count = thrust::make_counting_iterator<size_t>(0);
      int cols = temp_distances.cols;
      int item_norm_cols = items.rows;
      T *data = temp_distances;
      T *item_norms_data = *item_norms;
      thrust::for_each(count,
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
      T *data = temp_distances;
      int *items = item_filter->data;
      int items_size = item_filter->size;
      int cols = temp_distances.cols;

      // TODO: fix this for half
      T filter_distance = FLT_FILTER_DISTANCE;
      thrust::for_each(count, count + items_size * temp_distances.rows,
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
      T *data = temp_distances;
      int items = temp_distances.cols;
      // TODO: fix this for half
      T filter_distance = FLT_FILTER_DISTANCE;
      thrust::for_each(
          count, count + query_filter->nonzeros, [=] __device__(int i) {
            if ((row[i] >= start) && (row[i] < end)) {
              data[(row[i] - start) * items + col[i]] = filter_distance;
            }
          });
    }

    argpartition<T>(temp_distances, k, indices + start * k,
                    distances + start * k, allow_tiling);

    // TODO: callback per batch (show progress etc)
  }

  if (host_indices) {
    CHECK_CUDA(cudaMemcpy(host_indices, indices, indices_size,
                          cudaMemcpyDeviceToHost));
  }

  if (host_distances) {
    CHECK_CUDA(cudaMemcpy(host_distances, distances, distances_size,
                          cudaMemcpyDeviceToHost));
  }
}

template <typename T>
void KnnQuery::argpartition(const Matrix &items, int k, int *indices,
                            T *distances, bool allow_tiling) {
  k = std::min(k, items.cols);

  if (k >= GPU_MAX_SELECTION_K) {
    rmm::cuda_stream_view stream;
    rmm::device_uvector<int> temp_indices(items.rows * items.cols, stream,
                                          mr.get());
    rmm::device_uvector<T> temp_distances(items.rows * items.cols, stream,
                                          mr.get());
    argsort<T>(items, temp_indices.data(), temp_distances.data());
    copy_columns(temp_distances.data(), items.rows, items.cols, distances, k);
    copy_columns(temp_indices.data(), items.rows, items.cols, indices, k);
    return;
  }

  int rows = items.rows;
  int cols = items.cols;

  // faiss runBlockSelect isn't the fastest option when there aren't that many
  // rows, since each row in the query only gets a single thread block to
  // process it. For queries with a small number of rows, we're going to break
  // up each row into TILE_GROUPS sub-rows, in one runBlockSelect, and then
  // combine the results from those in a final select op.
  bool tile_rows =
      (rows <= MAX_TILE_ROWS) && (cols % TILE_GROUPS == 0) && (cols >= 65536);
  if (allow_tiling && tile_rows) {
    // Run the first block select on the sub-rows
    int rows_tile = rows * TILE_GROUPS;
    int cols_tile = cols / TILE_GROUPS;

    rmm::cuda_stream_view stream;
    rmm::device_uvector<int> temp_indices(rows_tile * k, stream, mr.get());
    rmm::device_uvector<T> temp_distances(rows_tile * k, stream, mr.get());

    const T *items_data = items;
    faiss::gpu::DeviceTensor<T, 2, true> items_tensor(
        const_cast<T *>(items_data), {rows_tile, cols_tile});
    faiss::gpu::DeviceTensor<T, 2, true> temp_distances_tensor(
        temp_distances.data(), {rows_tile, k});
    faiss::gpu::DeviceTensor<int, 2, true> temp_indices_tensor(
        temp_indices.data(), {rows_tile, k});
    faiss::gpu::runBlockSelect(items_tensor, temp_distances_tensor,
                               temp_indices_tensor, true, k, 0);

    // Calculate the true index for all the topk results (since the current
    // temp_indices will be relative to the split values)
    auto count = thrust::make_counting_iterator<size_t>(0);
    int *temp_indices_ptr = temp_indices.data();
    thrust::for_each(count, count + rows_tile * k, [=] __device__(int i) {
      int offset = cols_tile * ((i / k) % TILE_GROUPS);
      temp_indices_ptr[i] += offset;
    });

    // reshape the temp tensors we calculated in the first pass, and then get
    // the actual output
    faiss::gpu::DeviceTensor<T, 2, true> temp_input_distances_tensor(
        temp_distances.data(), {rows, k * TILE_GROUPS});
    faiss::gpu::DeviceTensor<int, 2, true> temp_input_indices_tensor(
        temp_indices.data(), {rows, k * TILE_GROUPS});
    faiss::gpu::DeviceTensor<T, 2, true> distances_tensor(distances, {rows, k});
    faiss::gpu::DeviceTensor<int, 2, true> indices_tensor(indices, {rows, k});
    faiss::gpu::runBlockSelectPair(temp_input_distances_tensor,
                                   temp_input_indices_tensor, distances_tensor,
                                   indices_tensor, true, k, 0);
  } else {
    const T *items_data = items;

    faiss::gpu::DeviceTensor<T, 2, true> items_tensor(
        const_cast<T *>(items_data), {rows, cols});
    faiss::gpu::DeviceTensor<T, 2, true> distances_tensor(distances, {rows, k});
    faiss::gpu::DeviceTensor<int, 2, true> indices_tensor(indices, {rows, k});
    faiss::gpu::runBlockSelect(items_tensor, distances_tensor, indices_tensor,
                               true, k, 0);
  }

  CHECK_CUDA(cudaDeviceSynchronize());
}

template <typename T>
void KnnQuery::argsort(const Matrix &items, int *indices, T *distances) {
  // We can't do this in place https://github.com/NVIDIA/cub/issues/238 ?
  // so generate temp memory for this

  rmm::cuda_stream_view stream;

  rmm::device_uvector<int> temp_indices(items.rows * items.cols, stream,
                                        mr.get());
  thrust::transform(
      thrust::make_counting_iterator<int>(0),
      thrust::make_counting_iterator<int>(items.rows * items.cols),
      thrust::make_constant_iterator<int>(items.cols),
      thrust::device_pointer_cast(temp_indices.data()), thrust::modulus<int>());

  int cols = items.cols;
  auto segment_offsets = thrust::make_transform_iterator(
      thrust::make_counting_iterator<int>(0),
      [=] __device__(int i) { return i * cols; });

  void *temp_mem = NULL;
  size_t temp_size = 0;

  // sort the values.
  const T *data = items;
  if (items.rows > 1) {
    auto err = cub::DeviceSegmentedRadixSort::SortPairsDescending(
        NULL, temp_size, data, distances, temp_indices.data(), indices,
        items.rows * items.cols, items.rows, segment_offsets,
        segment_offsets + 1);
    CHECK_CUDA(err);
    temp_mem = mr->allocate(temp_size, stream);
    err = cub::DeviceSegmentedRadixSort::SortPairsDescending(
        temp_mem, temp_size, data, distances, temp_indices.data(), indices,
        items.rows * items.cols, items.rows, segment_offsets,
        segment_offsets + 1);
    CHECK_CUDA(err);
  } else {
    size_t temp_size = 0;
    auto err = cub::DeviceRadixSort::SortPairsDescending(
        NULL, temp_size, data, distances, temp_indices.data(), indices,
        items.cols);
    CHECK_CUDA(err);
    temp_mem = mr->allocate(temp_size, stream);
    err = cub::DeviceRadixSort::SortPairsDescending(
        temp_mem, temp_size, data, distances, temp_indices.data(), indices,
        items.cols);
    CHECK_CUDA(err);
  }
  mr->deallocate(temp_mem, temp_size, stream);
}

KnnQuery::~KnnQuery() {
  // TODO: don't check this, there isn't anything we can do here anyways
  CHECK_CUBLAS(cublasDestroy(blas_handle));
}

} // namespace gpu
} // namespace implicit

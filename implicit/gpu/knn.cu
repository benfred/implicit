#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <cub/iterator/counting_input_iterator.cuh>
#include <cub/iterator/transform_input_iterator.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/block/block_radix_sort.cuh>


#include "implicit/gpu/utils.cuh"
#include "implicit/gpu/knn.h"
#include "implicit/gpu/device_buffer.h"

namespace implicit { namespace gpu {

bool is_host_memory(void * address) {
    cudaPointerAttributes attr;
    auto err = cudaPointerGetAttributes(&attr, address);
    if (err == cudaErrorInvalidValue) {
        return true;
    }

#if __CUDACC_VER_MAJOR__ >= 10
    return attr.type == cudaMemoryTypeHost || attr.type == cudaMemoryTypeUnregistered;
#else
    return attr.memoryType == cudaMemoryTypeHost || attr.memoryType == cudaMemoryTypeUnregistered;
#endif
}


class StackAllocator {
public:
    StackAllocator(size_t bytes) : memory(bytes), allocated(0) {}
    void * allocate(size_t bytes) {
        size_t padding = bytes % 128;
        if (padding) {
            bytes += 128 - padding;
        }
        if (allocated + bytes >= memory.size()) {
            throw std::invalid_argument("stack allocator: out of memory");
        }
        allocations.push_back(bytes);
        void * ret = memory.get() + allocated;
        allocated += bytes;
        return ret;
    }

    void deallocate(void * ptr) {
        size_t bytes = allocations.back();
        if (ptr != memory.get() + allocated - bytes) {
            throw std::invalid_argument("stack allocator: free called out of order");
        }
        allocations.pop_back();
        allocated -= bytes;
    }

protected:
    std::vector<size_t> allocations;
    DeviceBuffer<char> memory;
    size_t allocated;
};

template <typename T>
void copy_columns(const T * input, int rows, int cols, T * output, int output_cols) {
    auto count = thrust::make_counting_iterator<int>(0);
    thrust::for_each(count, count + (rows * output_cols),
       [=] __device__(int i) {
         int col = i % output_cols;
         int row = i / output_cols;
         output[col + row * output_cols] = input[col + row * cols];
    });
}

KnnQuery::KnnQuery(size_t temp_memory)
    : max_temp_memory(temp_memory),
    alloc(new StackAllocator(temp_memory)) {
    CHECK_CUBLAS(cublasCreate(&blas_handle));
}

const static int MAX_SELECT_K = 128;

void KnnQuery::topk(const Matrix & items, const Matrix & query, int k,
                    int * indices, float * distances, float * item_norms,
                    const COOMatrix * query_filter,
                    Vector<int> * item_filter) {
    if (query.cols != items.cols) {
        throw std::invalid_argument("Must have same number of columns in each matrix for topk");
    }

    size_t available_temp_memory = max_temp_memory;

    float * host_distances = NULL;
    size_t distances_size = query.rows * k * sizeof(float);
    if (is_host_memory(distances)) {
        host_distances = distances;
        distances = reinterpret_cast<float *>(alloc->allocate(distances_size));
        available_temp_memory -= distances_size;
    }

    int * host_indices = NULL;
    size_t indices_size = query.rows * k * sizeof(int);
    if (is_host_memory(indices)) {
        host_indices = indices;
        indices = reinterpret_cast<int *>(alloc->allocate(indices_size));
        available_temp_memory -= indices_size;
    }

    // We need 6 copies of the matrix for argsort code - and then some
    // extra memory per SM as well.
    int batch_size = available_temp_memory / (sizeof(float) * items.rows);
    if (k >= MAX_SELECT_K) {
        batch_size *= 0.15;
    } else {
        batch_size *= 0.5;
    }

    batch_size = std::min(batch_size, query.rows);
    batch_size = std::max(batch_size, 1);

    // Create temporary memory for storing results
    void * temp_mem = alloc->allocate(batch_size * items.rows * sizeof(float));
    Matrix temp_distances(batch_size, items.rows, reinterpret_cast<float *>(temp_mem), false);

    for (int start = 0; start < query.rows; start += batch_size) {
        auto end = std::min(query.rows, start + batch_size);

        Matrix batch(query, start, end);
        temp_distances.rows = batch.rows;

        // matrix multiple the items by the batch, store in distances
        float alpha = 1.0, beta = 0.;

        CHECK_CUBLAS(cublasSgemm(blas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                 items.rows, batch.rows, items.cols,
                                 &alpha,
                                 items.data, items.cols,
                                 batch.data, batch.cols,
                                 &beta,
                                 temp_distances.data, temp_distances.cols));

        // If we have norms (cosine distance etc) normalize the results here
        if (item_norms != NULL) {
            auto count = thrust::make_counting_iterator<int>(0);
            int cols = temp_distances.cols;
            float * data = temp_distances.data;
            thrust::for_each(count, count + (temp_distances.rows * temp_distances.cols),
               [=] __device__(int i) {
                 data[i] /= item_norms[i % cols];
            });
        }

        if (item_filter != NULL) {
            auto count = thrust::make_counting_iterator<int>(0);
            float * data = temp_distances.data;
            int * items = item_filter->data;
            int items_size = item_filter->size;
            int cols = temp_distances.cols;
            thrust::for_each(count, count + items_size * temp_distances.rows,
               [=] __device__(int i) {
                  int col = items[i % items_size];
                  int row = i / items_size;
                  data[row * cols + col] = -FLT_MAX;
            });
        }

        if (query_filter != NULL) {
            auto count = thrust::make_counting_iterator<int>(0);
            int * row = query_filter->row;
            int * col = query_filter->col;
            float * data = temp_distances.data;
            int items = temp_distances.cols;
            thrust::for_each(count, count + query_filter->nonzeros,
                [=] __device__(int i) {
                    if ((row[i] >= start) && (row[i] < end)) {
                        data[(row[i] -start) * items + col[i]] = -FLT_MAX;
                    }
            });
        }

        argpartition(temp_distances, k, indices + start * k, distances + start * k);

        // TODO: callback per batch (show progress etc)
    }
    alloc->deallocate(temp_mem);

    if (host_indices) {
        CHECK_CUDA(cudaMemcpy(host_indices, indices, indices_size, cudaMemcpyDeviceToHost));
        alloc->deallocate(indices);
    }

    if (host_distances) {
        CHECK_CUDA(cudaMemcpy(host_distances, distances, distances_size, cudaMemcpyDeviceToHost));
        alloc->deallocate(distances);
    }
}


static const int ARGPARTITION_BLOCK_DIM_X = 128;
static const int ARGPARTITION_ITEMS_PER_THREAD = 16;
static const int ARGPARTITION_SORT_SIZE = ARGPARTITION_BLOCK_DIM_X * ARGPARTITION_ITEMS_PER_THREAD;

__global__ void argpartition_kernel(const int * indices, const float * distances,
                                    int rows, int cols, int k,
                                    int * out_indices,
                                    float * out_distances) {
    using BlockRadixSort = cub::BlockRadixSort<float, ARGPARTITION_BLOCK_DIM_X, ARGPARTITION_ITEMS_PER_THREAD, int>;
    __shared__ typename BlockRadixSort::TempStorage shared_mem;

    float keys[ARGPARTITION_ITEMS_PER_THREAD];
    int values[ARGPARTITION_ITEMS_PER_THREAD];

    int rowid = blockIdx.y;
    for (int i = 0; i < ARGPARTITION_ITEMS_PER_THREAD; i++) {
        int colid = blockIdx.x * blockDim.x + threadIdx.x + i * (blockDim.x * gridDim.x);
        if (colid < cols) {
            keys[i] = distances[rowid * cols + colid];
            values[i] = indices == NULL ? colid : indices[rowid * cols + colid];
        } else {
            keys[i] = -FLT_MAX;
            values[i] = -1;
        }
    }

    BlockRadixSort(shared_mem).SortDescendingBlockedToStriped(keys, values);

    if (threadIdx.x < k) {
        int out_col = threadIdx.x + blockIdx.x * k;
        out_distances[out_col + rowid * k * gridDim.x] = keys[0];
        out_indices[out_col + rowid * k * gridDim.x] = values[0];
    }
}

void KnnQuery::argpartition(const Matrix & items, int k, int * indices, float * distances) {
    k = std::min(k, items.cols);

    if (k >= MAX_SELECT_K) {
        int * temp_indices = reinterpret_cast<int *>(alloc->allocate(items.rows * items.cols * sizeof(int)));
        float * temp_distances = reinterpret_cast<float *>(alloc->allocate(items.rows * items.cols * sizeof(float)));
        argsort(items, temp_indices, temp_distances);
        copy_columns(temp_distances, items.rows, items.cols, distances, k);
        copy_columns(temp_indices, items.rows, items.cols, indices, k);
        alloc->deallocate(temp_distances);
        alloc->deallocate(temp_indices);
        return;
    }

    int rows = items.rows;
    int cols = items.cols;

    int blocks_per_row = (cols + ARGPARTITION_SORT_SIZE - 1) / ARGPARTITION_SORT_SIZE;

    // maintain a double buffer of input/output indices and distances
    float * distA = reinterpret_cast<float *>(alloc->allocate(rows * k * blocks_per_row * sizeof(float)));
    int * indA = reinterpret_cast<int *>(alloc->allocate(rows * k * blocks_per_row * sizeof(int)));
    blocks_per_row = (blocks_per_row * k + ARGPARTITION_SORT_SIZE - 1) / ARGPARTITION_SORT_SIZE;
    float * distB = reinterpret_cast<float *>(alloc->allocate(rows * k * blocks_per_row * sizeof(float)));
    int * indB = reinterpret_cast<int *>(alloc->allocate(rows * k * blocks_per_row * sizeof(int)));

    const float * input_distances = items.data;
    const int * input_indices = NULL;
    float * output_distances = distA;
    int * output_indices = indA;
    bool outputA = true;

    while (true) {
        int blocks_per_row = (cols + ARGPARTITION_SORT_SIZE - 1) / ARGPARTITION_SORT_SIZE;
        dim3 block_count(blocks_per_row, items.rows, 1);

        bool final = block_count.x <= 1;
        if (final) {
            output_distances = distances;
            output_indices = indices;
        }

        argpartition_kernel<<<block_count, ARGPARTITION_BLOCK_DIM_X>>>(
            input_indices, input_distances,
            rows, cols, k,
            output_indices, output_distances);

        if (final) break;

        // reduce the number of columns we process next iteration to the output of the current
        // input
        cols = block_count.x * k;

        // set the input of the next run to the output of the current run
        // (and the output to an unused block of memory)
        input_distances = output_distances;
        input_indices = output_indices;
        output_distances = outputA ? distB : distA;
        output_indices = outputA ? indB : indA;
        outputA = !outputA;
    }

    CHECK_CUDA(cudaDeviceSynchronize());

    // Free up temp memory
    alloc->deallocate(indB);
    alloc->deallocate(distB);
    alloc->deallocate(indA);
    alloc->deallocate(distA);
}

void KnnQuery::argsort(const Matrix & items, int * indices, float * distances) {
    // We can't do this in place https://github.com/NVIDIA/cub/issues/238 ?
    // so generate temp memory for this
    auto temp_indices = reinterpret_cast<int *>(alloc->allocate(items.rows * items.cols * sizeof(int)));
    thrust::transform(
        thrust::make_counting_iterator<int>(0),
        thrust::make_counting_iterator<int>(items.rows * items.cols),
        thrust::make_constant_iterator<int>(items.cols),
        thrust::device_pointer_cast(temp_indices),
        thrust::modulus<int>());

    int cols = items.cols;
    auto segment_offsets = thrust::make_transform_iterator(thrust::make_counting_iterator<int>(0),
                                                           [=] __device__(int i) {
                                                               return i * cols;
                                                           });
    void * temp_mem = NULL;

    // sort the values.
    if (items.rows > 1) {
        size_t temp_size = 0;
        auto err = cub::DeviceSegmentedRadixSort::SortPairsDescending(NULL,
            temp_size,
            items.data,
            distances,
            temp_indices,
            indices,
            items.rows * items.cols,
            items.rows,
            segment_offsets,
            segment_offsets + 1);
        CHECK_CUDA(err);
        temp_mem = alloc->allocate(temp_size);
        err = cub::DeviceSegmentedRadixSort::SortPairsDescending(temp_mem,
            temp_size,
            items.data,
            distances,
            temp_indices,
            indices,
            items.rows * items.cols,
            items.rows,
            segment_offsets,
            segment_offsets + 1);
        CHECK_CUDA(err);
    } else {
        size_t temp_size = 0;
        auto err = cub::DeviceRadixSort::SortPairsDescending(NULL,
            temp_size,
            items.data,
            distances,
            temp_indices,
            indices,
            items.cols);
        CHECK_CUDA(err);
        temp_mem = alloc->allocate(temp_size);
        err = cub::DeviceRadixSort::SortPairsDescending(temp_mem,
            temp_size,
            items.data,
            distances,
            temp_indices,
            indices,
            items.cols);
        CHECK_CUDA(err);
    }
    alloc->deallocate(temp_mem);
    alloc->deallocate(temp_indices);
}

KnnQuery::~KnnQuery() {
    // TODO: don't check this, there isn't anything we can do here anyways
    CHECK_CUBLAS(cublasDestroy(blas_handle));
}

}}  // namespace implicit::gpu

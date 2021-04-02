#include <cuda_runtime.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>


#include "implicit/gpu/matrix.h"
#include "implicit/gpu/utils.cuh"

namespace implicit { namespace gpu {
// TODO: this class doesn't seem to offer much on top of devicebuffer itself
template <typename T>
Vector<T>::Vector(int size, const T * host_data)
    : size(size), storage(new DeviceBuffer<T>(size)), data(storage->get()) {
    if (host_data) {
        CHECK_CUDA(cudaMemcpy(data, host_data, size * sizeof(T), cudaMemcpyHostToDevice));
    }
}

template <typename T>
void Vector<T>::to_host(T * out) const {
    CHECK_CUDA(cudaMemcpy(out, data, size * sizeof(T), cudaMemcpyDeviceToHost));
}

template struct Vector<char>;
template struct Vector<int>;
template struct Vector<float>;

Matrix::Matrix(const Matrix & other, int rowid)
    : rows(1), cols(other.cols), data(other.data + rowid * other.cols), storage(other.storage) {
    if (rowid >= other.rows) {
        throw std::invalid_argument("row index out of bounds for matrix");
    }
}

Matrix::Matrix(const Matrix & other, int start_rowid, int end_rowid)
    : rows(end_rowid - start_rowid), cols(other.cols), data(other.data + start_rowid * other.cols), storage(other.storage) {
    if (end_rowid < start_rowid) {
        throw std::invalid_argument("end_rowid < start_rowid for matrix slice");
    }
    if (end_rowid > other.rows) {
        throw std::invalid_argument("row index out of bounds for matrix");
    }
}

void copy_rowids(const float * input, const int * rowids, int rows, int cols, float * output) {
    // copy rows over
    auto count = thrust::make_counting_iterator<int>(0);
    thrust::for_each(count, count + (rows * cols),
        [=] __device__(int i) {
            int col = i % cols;
            int row = rowids[i / cols];
            output[i] = input[col + row * cols];
        });
}

Matrix::Matrix(const Matrix & other, const Vector<int> & rowids)
    : rows(rowids.size), cols(other.cols) {
    storage.reset(new DeviceBuffer<float>(rows * cols));
    data = storage->get();
    copy_rowids(other.data, rowids.data, rows, cols, data);
}

Matrix::Matrix(int rows, int cols, float * host_data, bool allocate)
    : rows(rows), cols(cols) {
    if (allocate) {
        storage.reset(new DeviceBuffer<float>(rows * cols));
        data = storage->get();
        if (host_data) {
            CHECK_CUDA(cudaMemcpy(data, host_data, rows * cols * sizeof(float), cudaMemcpyHostToDevice));
        }
    } else {
        data = host_data;
    }
}

__global__ void calculate_norms_kernel(const float * input, int rows, int cols, float * output) {
    static __shared__ float shared[32];
    for (int i = blockIdx.x; i < rows; i += gridDim.x) {
        float value = input[i * cols + threadIdx.x];
	float squared_norm = dot(value, value, shared);
	if (threadIdx.x == 0) {
            output[i] = sqrt(squared_norm);
            if (output[i] == 0) {
                output[i] = 1e-10;
            }
	}
    }
}

Matrix calculate_norms(const Matrix & input) {
    int devId;
    CHECK_CUDA(cudaGetDevice(&devId));

    int multiprocessor_count;
    CHECK_CUDA(cudaDeviceGetAttribute(&multiprocessor_count,
                                      cudaDevAttrMultiProcessorCount,
                                      devId));

    int block_count = 256 * multiprocessor_count;
    int thread_count = input.cols;

    Matrix output(1, input.rows, NULL);
    calculate_norms_kernel<<<block_count, thread_count>>>(input.data, input.rows, input.cols, output.data);

    CHECK_CUDA(cudaDeviceSynchronize());
    return output;
}

void Matrix::to_host(float * out) const {
    CHECK_CUDA(cudaMemcpy(out, data, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));
}

CSRMatrix::CSRMatrix(int rows, int cols, int nonzeros,
                     const int * indptr_, const int * indices_, const float * data_)
    : rows(rows), cols(cols), nonzeros(nonzeros) {

    CHECK_CUDA(cudaMalloc(&indptr, (rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(indptr, indptr_, (rows + 1)*sizeof(int), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMalloc(&indices, nonzeros * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(indices, indices_, nonzeros * sizeof(int), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMalloc(&data, nonzeros * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(data, data_, nonzeros * sizeof(int), cudaMemcpyHostToDevice));
}

CSRMatrix::~CSRMatrix() {
    CHECK_CUDA(cudaFree(indices));
    CHECK_CUDA(cudaFree(indptr));
    CHECK_CUDA(cudaFree(data));
}

COOMatrix::COOMatrix(int rows, int cols, int nonzeros,
                     const int * row_, const int * col_, const float * data_)
    : rows(rows), cols(cols), nonzeros(nonzeros) {

    CHECK_CUDA(cudaMalloc(&row, nonzeros * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(row, row_, nonzeros * sizeof(int), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMalloc(&col, nonzeros * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(col, col_, nonzeros * sizeof(int), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMalloc(&data, nonzeros * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(data, data_, nonzeros * sizeof(int), cudaMemcpyHostToDevice));
}

COOMatrix::~COOMatrix() {
    CHECK_CUDA(cudaFree(row));
    CHECK_CUDA(cudaFree(col));
    CHECK_CUDA(cudaFree(data));
}
}}  // namespace implicit::gpu

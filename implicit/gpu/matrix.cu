
#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "implicit/gpu/matrix.h"
#include "implicit/gpu/utils.cuh"

namespace implicit { namespace gpu {
template <typename T>
Vector<T>::Vector(int size, const T * host_data)
    : size(size) {
    CHECK_CUDA(cudaMalloc(&data, size * sizeof(T)));
    if (host_data) {
        CHECK_CUDA(cudaMemcpy(data, host_data, size * sizeof(T), cudaMemcpyHostToDevice));
    }
}


template <typename T>
Vector<T>::~Vector() {
    CHECK_CUDA(cudaFree(data));
}

template struct Vector<int>;
template struct Vector<float>;

Matrix::Matrix(int rows, int cols, float * host_data, bool cpu)
    : rows(rows), cols(cols) {
    if (cpu) {
        CHECK_CUDA(cudaMalloc(&data, rows * cols * sizeof(float)));
        if (host_data) {
            CHECK_CUDA(cudaMemcpy(data, host_data, rows * cols * sizeof(float), cudaMemcpyHostToDevice));
        }
        owns_data = true;
    } else {
        data = host_data;
        owns_data = false;
    }
}

void Matrix::to_host(float * out) const {
    CHECK_CUDA(cudaMemcpy(out, data, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));
}

Matrix::~Matrix() {
    if (owns_data) {
        CHECK_CUDA(cudaFree(data));
    }
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

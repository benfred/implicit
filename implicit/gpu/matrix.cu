#include <cuda_runtime.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include "implicit/gpu/convert.cuh"
#include "implicit/gpu/dot.cuh"
#include "implicit/gpu/matrix.h"
#include "implicit/gpu/utils.h"

namespace implicit {
namespace gpu {
template <typename T>
Vector<T>::Vector(size_t size, const T *host_data)
    : size(size),
      storage(new rmm::device_uvector<T>(size, rmm::cuda_stream_view())),
      data(storage->data()) {
  if (host_data) {
    CHECK_CUDA(
        cudaMemcpy(data, host_data, size * sizeof(T), cudaMemcpyHostToDevice));
  } else {
    CHECK_CUDA(cudaMemset(data, 0, size * sizeof(T)));
  }
}

template <typename T> void Vector<T>::to_host(T *out) const {
  CHECK_CUDA(cudaMemcpy(out, data, size * sizeof(T), cudaMemcpyDeviceToHost));
}

template struct Vector<char>;
template struct Vector<int>;
template struct Vector<float>;

Matrix::Matrix(const Matrix &other, size_t rowid)
    : rows(1), cols(other.cols), data(other.at(rowid * other.cols)),
      storage(other.storage), itemsize(other.itemsize) {
  if (rowid >= other.rows) {
    throw std::invalid_argument("row index out of bounds for matrix");
  }
}

Matrix::Matrix(const Matrix &other, size_t start_rowid, size_t end_rowid)
    : rows(end_rowid - start_rowid), cols(other.cols),
      data(other.at(start_rowid * other.cols)), storage(other.storage),
      itemsize(other.itemsize) {
  if (end_rowid < start_rowid) {
    throw std::invalid_argument("end_rowid < start_rowid for matrix slice");
  }
  if (end_rowid > other.rows) {
    throw std::invalid_argument("row index out of bounds for matrix");
  }
}

template <typename T>
void copy_rowids(const T *input, const int *rowids, size_t rows, size_t cols,
                 T *output) {
  // copy rows over
  auto count = thrust::make_counting_iterator<size_t>(0);
  thrust::for_each(count, count + (rows * cols), [=] __device__(size_t i) {
    size_t col = i % cols;
    size_t row = rowids[i / cols];
    output[i] = input[col + row * cols];
  });
}

Matrix::Matrix(const Matrix &other, const Vector<int> &rowids)
    : rows(rowids.size), cols(other.cols), itemsize(other.itemsize) {
  storage.reset(
      new rmm::device_buffer(itemsize * rows * cols, rmm::cuda_stream_view()));
  data = storage->data();
  if (itemsize == 4) {
    copy_rowids<float>(other, rowids.data, rows, cols, *this);
  } else if (itemsize == 2) {
    copy_rowids<half>(other, rowids.data, rows, cols, *this);
  } else {
    throw std::runtime_error("unknown itemsize initializing Matrix");
  }
}

Matrix::Matrix(size_t rows, size_t cols, void *host_data, bool allocate,
               size_t itemsize)
    : rows(rows), cols(cols), itemsize(itemsize) {
  if (allocate) {
    storage.reset(new rmm::device_buffer(itemsize * rows * cols,
                                         rmm::cuda_stream_view()));
    data = storage->data();
    if (host_data) {
      CHECK_CUDA(cudaMemcpy(data, host_data, rows * cols * itemsize,
                            cudaMemcpyHostToDevice));
    } else {
      CHECK_CUDA(cudaMemset(data, 0, rows * cols * itemsize));
    }
  } else {
    data = host_data;
  }
}

void Matrix::resize(size_t rows, size_t cols) {
  if (cols != this->cols) {
    throw std::logic_error(
        "changing number of columns in Matrix::resize is not implemented yet");
  }
  if (rows < this->rows) {
    throw std::logic_error(
        "reducing number of rows in Matrix::resize is not implemented yet");
  }
  auto new_storage =
      new rmm::device_buffer(itemsize * rows * cols, rmm::cuda_stream_view());
  CHECK_CUDA(cudaMemcpy(new_storage->data(), data,
                        this->rows * this->cols * itemsize,
                        cudaMemcpyDeviceToDevice));
  size_t extra_rows = rows - this->rows;
  storage.reset(new_storage);
  data = storage->data();
  CHECK_CUDA(
      cudaMemset(at(this->rows * this->cols), 0, extra_rows * cols * itemsize));

  this->rows = rows;
  this->cols = cols;
}

void Matrix::assign_rows(const Vector<int> &rowids, const Matrix &other) {
  if (other.cols != cols) {
    throw std::invalid_argument(
        "column dimensionality mismatch in Matrix::assign_rows");
  }

  auto count = thrust::make_counting_iterator<size_t>(0);
  size_t other_cols = other.cols, other_rows = other.rows;

  int *rowids_data = rowids.data;

  const float *other_data = other;
  float *self_data = *this;

  thrust::for_each(count, count + (other_rows * other_cols),
                   [=] __device__(size_t i) {
                     size_t col = i % other_cols;
                     size_t row = rowids_data[i / other_cols];
                     size_t idx = col + row * other_cols;
                     self_data[idx] = other_data[i];
                   });
}

template <typename I, typename O>
void convert_array(const I *input, size_t elements, O *output) {
  auto count = thrust::make_counting_iterator<size_t>(0);
  thrust::for_each(count, count + elements, [=] __device__(size_t i) {
    output[i] = convert<I, O>(input[i]);
  });
}

Matrix Matrix::astype(size_t itemsize) const {
  if (itemsize == this->itemsize) {
    return *this;
  }

  Matrix ret(rows, cols, NULL, true, itemsize);
  if (this->itemsize == 2 && itemsize == 4) {
    const half *input_data = *this;
    float *output_data = ret;
    convert_array<half, float>(input_data, rows * cols, output_data);
  } else if (this->itemsize == 4 && itemsize == 2) {
    const float *input_data = *this;
    half *output_data = ret;
    convert_array<float, half>(input_data, rows * cols, output_data);
  } else {
    throw std::runtime_error("Invalid dtype for Matrix::astype");
  }
  return ret;
}

template <typename T>
__global__ void calculate_norms_kernel(const T *input, size_t rows, size_t cols,
                                       float *output) {
  static __shared__ float shared[32];
  for (int i = blockIdx.x; i < rows; i += gridDim.x) {
    float value = convert<T, float>(input[i * cols + threadIdx.x]);
    float squared_norm = dot(value, value, shared);
    if (threadIdx.x == 0) {
      float norm = sqrt(squared_norm);
      if (norm == 0) {
        norm = 1e-10;
      }
      output[i] = norm;
    }
  }
}

Matrix Matrix::calculate_norms() const {
  int devId;
  CHECK_CUDA(cudaGetDevice(&devId));

  int multiprocessor_count;
  CHECK_CUDA(cudaDeviceGetAttribute(&multiprocessor_count,
                                    cudaDevAttrMultiProcessorCount, devId));

  int block_count = 256 * multiprocessor_count;
  int thread_count = cols;

  Matrix output(1, rows, NULL, true);

  if (itemsize == 4) {
    calculate_norms_kernel<float>
        <<<block_count, thread_count>>>(*this, rows, cols, output);
  } else if (itemsize == 2) {
    calculate_norms_kernel<half>
        <<<block_count, thread_count>>>(*this, rows, cols, output);
  } else {
    throw std::runtime_error("unknown itemsize in calculate_norms");
  }

  CHECK_CUDA(cudaDeviceSynchronize());
  return output;
}

void Matrix::to_host(void *out) const {
  CHECK_CUDA(
      cudaMemcpy(out, data, rows * cols * itemsize, cudaMemcpyDeviceToHost));
}

CSRMatrix::CSRMatrix(int rows, int cols, int nonzeros, const int *indptr_,
                     const int *indices_, const float *data_)
    : rows(rows), cols(cols), nonzeros(nonzeros) {

  CHECK_CUDA(cudaMallocManaged(&indptr, (rows + 1) * sizeof(int)));
  CHECK_CUDA(cudaMemcpy(indptr, indptr_, (rows + 1) * sizeof(int),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemAdvise(indptr, (rows + 1) * sizeof(int),
                           cudaMemAdviseSetReadMostly, 0));

  CHECK_CUDA(cudaMallocManaged(&indices, nonzeros * sizeof(int)));
  CHECK_CUDA(cudaMemcpy(indices, indices_, nonzeros * sizeof(int),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemAdvise(indices, nonzeros * sizeof(int),
                           cudaMemAdviseSetReadMostly, 0));

  CHECK_CUDA(cudaMallocManaged(&data, nonzeros * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(data, data_, nonzeros * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemAdvise(data, nonzeros * sizeof(float),
                           cudaMemAdviseSetReadMostly, 0));
}

CSRMatrix::~CSRMatrix() {
  CHECK_CUDA(cudaFree(indices));
  CHECK_CUDA(cudaFree(indptr));
  CHECK_CUDA(cudaFree(data));
}

COOMatrix::COOMatrix(int rows, int cols, int nonzeros, const int *row_,
                     const int *col_, const float *data_)
    : rows(rows), cols(cols), nonzeros(nonzeros) {

  CHECK_CUDA(cudaMallocManaged(&row, nonzeros * sizeof(int)));
  CHECK_CUDA(
      cudaMemcpy(row, row_, nonzeros * sizeof(int), cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMallocManaged(&col, nonzeros * sizeof(int)));
  CHECK_CUDA(
      cudaMemcpy(col, col_, nonzeros * sizeof(int), cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMallocManaged(&data, nonzeros * sizeof(float)));
  CHECK_CUDA(
      cudaMemcpy(data, data_, nonzeros * sizeof(int), cudaMemcpyHostToDevice));
}

COOMatrix::~COOMatrix() {
  CHECK_CUDA(cudaFree(row));
  CHECK_CUDA(cudaFree(col));
  CHECK_CUDA(cudaFree(data));
}
} // namespace gpu
} // namespace implicit

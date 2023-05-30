#ifndef IMPLICIT_GPU_MATRIX_H_
#define IMPLICIT_GPU_MATRIX_H_
#include <memory>

#include <cuda_fp16.h>

#include <rmm/device_uvector.hpp>

namespace implicit {
namespace gpu {
// Thin wrappers of CUDA memory: copies to from host, frees in destructor etc
template <typename T> struct Vector {
  Vector(size_t size, const T *data = NULL);
  void to_host(T *output) const;

  std::shared_ptr<rmm::device_uvector<T>> storage;
  size_t size;
  T *data;
};

struct Matrix {
  // Create a new matrix of shape (rows, cols) - copying from `data to the
  // device (if allocate=True and data != null). If allocate=false, this assumes
  // the data is preallocated on the gpu (cupy etc) and doesn't allocate any new
  // storage
  Matrix(size_t rows, size_t cols, void *data = NULL, bool allocate = true,
         size_t itemsize = 4);

  // Create a new Matrix by slicing a single row from an existing one. The
  // underlying storage buffer is shared in this case.
  Matrix(const Matrix &other, size_t rowid);

  // Slice a contiguous series of rows from this Matrix. The underlying storage
  // buffer is shared here.
  Matrix(const Matrix &other, size_t start_rowid, size_t end_rowid);

  // select a bunch of rows from this matrix. this creates a copy
  Matrix(const Matrix &other, const Vector<int> &rowids);

  void resize(size_t rows, size_t cols);
  void assign_rows(const Vector<int> &rowids, const Matrix &other);

  Matrix astype(size_t itemsize) const;

  Matrix() : rows(0), cols(0), data(NULL), itemsize(4) {}

  // Copy the Matrix to host memory.
  void to_host(void *output) const;

  // Calculates norms for each row in the matrix
  Matrix calculate_norms() const;

  size_t rows, cols;
  void *data;
  size_t itemsize;

  operator const float *() const {
    if (itemsize != 4) {
      throw std::runtime_error("can't cast Matrix to const float*");
    }
    return reinterpret_cast<const float *>(data);
  }

  operator float *() {
    if (itemsize != 4) {
      throw std::runtime_error("can't cast Matrix to float*");
    }
    return reinterpret_cast<float *>(data);
  }

  operator const half *() const {
    if (itemsize != 2) {
      throw std::runtime_error("can't cast Matrix to const half*");
    }
    return reinterpret_cast<const half *>(data);
  }

  operator half *() {
    if (itemsize != 2) {
      throw std::runtime_error("can't cast Matrix to half*");
    }
    return reinterpret_cast<half *>(data);
  }

  void *at(size_t element) const {
    char *x = reinterpret_cast<char *>(data);
    return x + itemsize * element;
  }

  std::shared_ptr<rmm::device_buffer> storage;
};

struct CSRMatrix {
  CSRMatrix(int rows, int cols, int nonzeros, const int *indptr,
            const int *indices, const float *data);
  ~CSRMatrix();
  int *indptr, *indices;
  float *data;
  int rows, cols, nonzeros;
};

struct COOMatrix {
  COOMatrix(int rows, int cols, int nonzeros, const int *row, const int *col,
            const float *data);
  ~COOMatrix();
  int *row, *col;
  float *data;
  int rows, cols, nonzeros;
};

} // namespace gpu
} // namespace implicit
#endif // IMPLICIT_GPU_MATRIX_H_

#ifndef IMPLICIT_GPU_MATRIX_H_
#define IMPLICIT_GPU_MATRIX_H_
#include <memory>

#include "implicit/gpu/device_buffer.h"

namespace implicit { namespace gpu {
// Thin wrappers of CUDA memory: copies to from host, frees in destructor etc
template <typename T>
struct Vector {
    Vector(int size, const T * data = NULL);
    void to_host(T * output) const;

    std::shared_ptr<DeviceBuffer<T> > storage;
    int size;
    T * data;
};

struct Matrix {
    // Create a new matrix of shape (rows, cols) - copying from `data to the device
    // (if allocate=True and data != null). If allocate=false, this assumes the data is
    // preallocated on the gpu (cupy etc) and doesn't allocate any new storage
    Matrix(int rows, int cols, float * data=NULL, bool allocate=true);

    // Create a new Matrix by slicing a single row from an existing one. The underlying
    // storage buffer is shared in this case.
    Matrix(const Matrix & other, int rowid);

    // Slice a contiguous series of rows from this Matrix. The underlying storge buffer
    // is shared here.
    Matrix(const Matrix & other, int start_rowid, int end_rowid);

    // select a bunch of rows from this matrix. this creates a copy
    Matrix(const Matrix & other, const Vector<int> & rowids);

    Matrix() : rows(0), cols(0), data(NULL) {}

    // Copy the Matrix to host memory.
    void to_host(float * output) const;

    int rows, cols;
    float * data;
    std::shared_ptr<DeviceBuffer<float> > storage;
};

Matrix calculate_norms(const Matrix & input);

struct CSRMatrix {
    CSRMatrix(int rows, int cols, int nonzeros,
                  const int * indptr, const int * indices, const float * data);
    ~CSRMatrix();
    int * indptr, * indices;
    float * data;
    int rows, cols, nonzeros;
};

struct COOMatrix {
    COOMatrix(int rows, int cols, int nonzeros,
                  const int * row, const int * col, const float * data);
    ~COOMatrix();
    int * row, * col;
    float * data;
    int rows, cols, nonzeros;
};

}}  // namespace implicit/gpu
#endif  // IMPLICIT_GPU_MATRIX_H_

#ifndef IMPLICIT_GPU_MATRIX_H_
#define IMPLICIT_GPU_MATRIX_H_

namespace implicit { namespace gpu {
/// Thin wrappers of CUDA memory: copies to from host, frees in destructor
template <typename T>
struct Vector {
    Vector(int size, const T * elements = NULL);
    ~Vector();

    int size;
    T * data;
};

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

struct Matrix {
    Matrix(int rows, int cols, float * data, bool host=true);
    ~Matrix();

    void to_host(float * output) const;

    int rows, cols;
    float * data;
    bool owns_data;
};
}}  // namespace implicit/gpu
#endif  // IMPLICIT_GPU_MATRIX_H_

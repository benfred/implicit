#ifndef IMPLICIT_GPU_ALS_H_
#define IMPLICIT_GPU_ALS_H_
#include "implicit/gpu/matrix.h"

// Forward ref: don't require the whole cublas definition here
struct cublasContext;

namespace implicit { namespace gpu {

struct LeastSquaresSolver {
    explicit LeastSquaresSolver(int factors);
    ~LeastSquaresSolver();

    void least_squares(const CSRMatrix & Cui,
                       Matrix * X, const Matrix & Y,
                       float regularization,
                       int cg_steps) const;

    float calculate_loss(const CSRMatrix & Cui,
                         const Matrix & X,
                         const Matrix & Y,
                         float regularization);

    Matrix YtY;
    cublasContext * blas_handle;
};
}}  // namespace implicit::gpu
#endif  // IMPLICIT_GPU_ALS_H_

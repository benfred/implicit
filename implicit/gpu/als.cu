#include <math.h>
#include <stdio.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "implicit/gpu/als.h"
#include "implicit/gpu/dot.cuh"
#include "implicit/gpu/utils.h"

namespace implicit {
namespace gpu {

using std::invalid_argument;

__global__ void least_squares_cg_kernel(int factors, size_t user_count,
                                        size_t item_count, float *X,
                                        const float *Y, const float *YtY,
                                        const int *indptr, const int *indices,
                                        const float *data, int cg_steps) {
  extern __shared__ float shared_memory[];
  float *P = &shared_memory[0];
  float *shared = &shared_memory[factors];

  float Ap = 0;
  float p = 0;
  float r = 0;

  // Stride over users in the grid:
  // https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
  for (int u = blockIdx.x; u < user_count; u += gridDim.x) {
    float *x = &X[u * factors];
    float x_value = x[threadIdx.x];

    // handle 0-sized rows
    if (indptr[u] == indptr[u + 1]) {
      x[threadIdx.x] = 0;
      continue;
    }

    // calculate residual r = YtCuPu - YtCuY Xu
    r = 0;
    for (int i = 0; i < factors; ++i) {
      r -= x[i] * YtY[i * factors + threadIdx.x];
    }
    for (int index = indptr[u]; index < indptr[u + 1]; ++index) {
      float Yi = Y[indices[index] * factors + threadIdx.x];
      float confidence = data[index];

      if (confidence > 0) {
        r += (confidence - (confidence - 1) * dot(Yi, x_value, shared)) * Yi;
      } else {
        confidence *= -1;
        r += (-(confidence - 1) * dot(Yi, x_value, shared)) * Yi;
      }
    }
    P[threadIdx.x] = p = r;
    __syncthreads();

    float rsold = dot(r, r, shared);
    if (rsold < 1e-20)
      continue;

    for (int it = 0; it < cg_steps; ++it) {
      // calculate Ap = YtCuYp - without actually calculating YtCuY
      Ap = 0;
      for (int i = 0; i < factors; ++i) {
        Ap += P[i] * YtY[i * factors + threadIdx.x];
      }
      for (int index = indptr[u]; index < indptr[u + 1]; ++index) {
        float Yi = Y[indices[index] * factors + threadIdx.x];
        float confidence = data[index];
        if (confidence < 0)
          confidence *= -1;

        Ap += (confidence - 1) * dot(Yi, p, shared) * Yi;
      }

      // standard CG update
      float alpha = rsold / dot(p, Ap, shared);
      x_value += alpha * p;
      r -= alpha * Ap;
      __syncthreads();
      float rsnew = dot(r, r, shared);
      if (rsnew < 1e-20)
        break;

      P[threadIdx.x] = p = r + (rsnew / rsold) * p;
      rsold = rsnew;
      __syncthreads();
    }

    // this shouldn't happen - but if we hit a NaN in the above code then
    // complain and don't let it perpetuate
    if (isnan(rsold)) {
      if (threadIdx.x == 0) {
        printf("Warning NaN Detected in row %i of %lu\n", u, user_count);
      }
      x[threadIdx.x] = 0;
    } else {
      x[threadIdx.x] = x_value;
    }
  }
}

__global__ void l2_regularize_kernel(size_t factors, float regularization,
                                     float *YtY) {
  YtY[threadIdx.x * factors + threadIdx.x] += regularization;
}

LeastSquaresSolver::LeastSquaresSolver() {
  CHECK_CUBLAS(cublasCreate(&blas_handle));
}

void LeastSquaresSolver::calculate_yty(const Matrix &Y, Matrix *YtY,
                                       float regularization) {
  if (YtY->cols != Y.cols)
    throw invalid_argument("YtY and Y should have the same number of columns");

  // calculate YtY: note this expects col-major (and we have row-major
  // basically) so that we're inverting the CUBLAS_OP_T/CU_BLAS_OP_N ordering to
  // overcome this (like calculate YYt instead of YtY)
  size_t factors = Y.cols, item_count = Y.rows;
  float alpha = 1.0, beta = 0.;
  CHECK_CUBLAS(cublasSgemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_T, factors,
                           factors, item_count, &alpha, Y.data, factors, Y.data,
                           factors, &beta, YtY->data, factors));
  CHECK_CUDA(cudaDeviceSynchronize());

  // regularize the matrix
  l2_regularize_kernel<<<1, factors>>>(factors, regularization, YtY->data);
  CHECK_CUDA(cudaDeviceSynchronize());
}

void LeastSquaresSolver::least_squares(const CSRMatrix &Cui, Matrix *X,
                                       const Matrix &YtY, const Matrix &Y,
                                       int cg_steps) const {
  int item_count = Y.rows, user_count = X->rows, factors = X->cols;
  if (X->cols != Y.cols)
    throw invalid_argument("X and Y should have the same number of columns");
  if (X->cols != YtY.cols)
    throw invalid_argument("Columns of X don't match number of factors");
  if (Cui.rows > X->rows)
    throw invalid_argument("Dimensionality mismatch between rows of Cui and X");
  if (Cui.cols > Y.rows)
    throw invalid_argument("Dimensionality mismatch between cols of Cui and Y");

  // TODO: multi-gpu support
  int devId;
  CHECK_CUDA(cudaGetDevice(&devId));

  int multiprocessor_count;
  CHECK_CUDA(cudaDeviceGetAttribute(&multiprocessor_count,
                                    cudaDevAttrMultiProcessorCount, devId));

  int block_count = 256 * multiprocessor_count;
  int thread_count = factors;
  int shared_memory_size = sizeof(float) * (2 * factors);

  least_squares_cg_kernel<<<block_count, thread_count, shared_memory_size>>>(
      factors, user_count, item_count, X->data, Y.data, YtY.data, Cui.indptr,
      Cui.indices, Cui.data, cg_steps);

  CHECK_CUDA(cudaDeviceSynchronize());
}

__global__ void calculate_loss_kernel(int factors, size_t user_count,
                                      size_t item_count, const float *X,
                                      const float *Y, const float *YtY,
                                      const int *indptr, const int *indices,
                                      const float *data, float regularization,
                                      float *output) {
  // https://devblogs.nvidia.com/parallelforall/using-shared-memory-cuda-cc/
  extern __shared__ float shared_memory[];
  float *shared = &shared_memory[0];

  float loss = 0, user_norm = 0, item_norm = 0, total_confidence = 0, r = 0;

  for (int u = blockIdx.x; u < user_count; u += gridDim.x) {
    const float *x = &X[u * factors];
    float x_value = x[threadIdx.x];

    // calculates r = (YtCuY.dot(Xu) - 2 * YtCuPu).dot(Xu), without calculating
    // YtCuY
    r = 0;
    for (int i = 0; i < factors; ++i) {
      r += x[i] * YtY[i * factors + threadIdx.x];
    }

    for (int index = indptr[u]; index < indptr[u + 1]; ++index) {
      float Yi = Y[indices[index] * factors + threadIdx.x];
      float confidence = data[index];
      if (confidence > 0) {
        r +=
            ((confidence - 1) * dot(Yi, x_value, shared) - 2 * confidence) * Yi;
      } else {
        confidence *= -1;
        r += ((confidence - 1) * dot(Yi, x_value, shared)) * Yi;
      }
      loss += confidence;
      total_confidence += confidence;
    }
    loss += dot(x_value, r, shared);

    user_norm += dot(x_value, x_value, shared);
  }

  for (int i = blockIdx.x; i < item_count; i += gridDim.x) {
    float y = Y[i * factors + threadIdx.x];
    item_norm += dot(y, y, shared);
  }

  loss += regularization * (item_norm + user_norm);
  if (threadIdx.x == 0) {
    atomicAdd(output, loss);
    atomicAdd(output + 1, total_confidence);
  }
}

float LeastSquaresSolver::calculate_loss(const CSRMatrix &Cui, const Matrix &X,
                                         const Matrix &Y,
                                         float regularization) {
  size_t item_count = Y.rows, factors = Y.cols, user_count = X.rows;

  Matrix YtY(factors, factors, NULL);
  calculate_yty(Y, &YtY, regularization);

  float alpha = 1.0, beta = 0.;
  CHECK_CUBLAS(cublasSgemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_T, factors,
                           factors, item_count, &alpha, Y.data, factors, Y.data,
                           factors, &beta, YtY.data, factors));
  CHECK_CUDA(cudaDeviceSynchronize());
  float temp[2] = {0, 0};
  Matrix output(2, 1, temp);
  calculate_loss_kernel<<<1024, factors, sizeof(float) * factors>>>(
      factors, user_count, item_count, X.data, Y.data, YtY.data, Cui.indptr,
      Cui.indices, Cui.data, regularization, output.data);
  CHECK_CUDA(cudaDeviceSynchronize());
  output.to_host(temp);

  return temp[0] / (temp[1] + Cui.rows * Cui.cols - Cui.nonzeros);
}

LeastSquaresSolver::~LeastSquaresSolver() {
  CHECK_CUBLAS(cublasDestroy(blas_handle));
}
} // namespace gpu
} // namespace implicit

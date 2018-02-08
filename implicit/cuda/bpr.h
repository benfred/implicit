#ifndef IMPLICIT_CUDA_BPR_H_
#define IMPLICIT_CUDA_BPR_H_
#include "implicit/cuda/matrix.h"

namespace implicit {
int bpr_update(const CudaCOOMatrix & Ciu,
               CudaDenseMatrix * X,
               CudaDenseMatrix * Y,
               float learning_rate,
               float regularization,
               long seed);
}  // namespace implicit
#endif  // IMPLICIT_CUDA_BPR_H_

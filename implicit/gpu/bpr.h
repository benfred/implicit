#ifndef IMPLICIT_GPU_BPR_H_
#define IMPLICIT_GPU_BPR_H_
#include <utility>

#include "implicit/gpu/matrix.h"

namespace implicit {
namespace gpu {
std::pair<int, int> bpr_update(const Vector<int> &userids,
                               const Vector<int> &itemids,
                               const Vector<int> &indptr, Matrix *X, Matrix *Y,
                               float learning_rate, float regularization,
                               long seed, bool verify_negative_samples);
}
} // namespace implicit
#endif // IMPLICIT_GPU_BPR_H_

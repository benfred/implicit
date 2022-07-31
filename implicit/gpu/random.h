#ifndef IMPLICIT_GPU_RANDOM_H_
#define IMPLICIT_GPU_RANDOM_H_
#include <curand.h>

#include "implicit/gpu/matrix.h"

namespace implicit {
namespace gpu {

struct RandomState {
  RandomState(long seed);
  ~RandomState();

  Matrix uniform(size_t rows, size_t cols, float low = 0.0, float high = 1.0);
  Matrix randn(size_t rows, size_t cols, float mean = 0, float stddev = 1);

  RandomState(const RandomState &) = delete;
  RandomState &operator=(const RandomState &) = delete;

  curandGenerator_t rng;
};
} // namespace gpu
} // namespace implicit
#endif // IMPLICIT_GPU_RANDOM_H_

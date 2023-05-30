#include <cuda_runtime.h>
#include <curand.h>

#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#include "implicit/gpu/random.h"
#include "implicit/gpu/utils.h"

namespace implicit {
namespace gpu {

RandomState::RandomState(long seed) {
  CHECK_CURAND(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT));
  CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(rng, seed));
}

Matrix RandomState::uniform(size_t rows, size_t cols, float low, float high) {
  Matrix ret(rows, cols, NULL);
  CHECK_CURAND(curandGenerateUniform(rng, ret, rows * cols));

  if ((low != 0.0) || (high != 1.0)) {
    float *data = ret;
    auto start = thrust::device_pointer_cast(data);
    thrust::transform(start, start + rows * cols, start,
                      thrust::placeholders::_1 =
                          thrust::placeholders::_1 * (high - low) + low);
  }

  return ret;
}

Matrix RandomState::randn(size_t rows, size_t cols, float mean, float stddev) {
  Matrix ret(rows, cols, NULL);
  CHECK_CURAND(curandGenerateNormal(rng, ret, rows * cols, mean, stddev));
  return ret;
}

RandomState::~RandomState() { curandDestroyGenerator(rng); }
} // namespace gpu
} // namespace implicit

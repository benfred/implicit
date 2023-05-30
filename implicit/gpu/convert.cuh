#ifndef IMPLICIT_GPU_CONVERT_CUH_
#define IMPLICIT_GPU_CONVERT_CUH_

namespace implicit {
namespace gpu {

template <typename I, typename O> inline __device__ O convert(I);

template <> inline __device__ float convert(half input) {
  return __half2float(input);
}

template <> inline __device__ half convert(float input) {
  return __float2half(input);
}

template <> inline __device__ float convert(float input) { return input; }
} // namespace gpu
} // namespace implicit
#endif // IMPLICIT_GPU_CONVERT_CUH_

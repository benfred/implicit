#ifndef IMPLICIT_GPU_DOT_CUH_
#define IMPLICIT_GPU_DOT_CUH_

namespace implicit {
namespace gpu {
#define WARP_SIZE 32

// https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
__inline__ __device__ float warp_reduce_sum(float val) {

#if __CUDACC_VER_MAJOR__ >= 9
  // __shfl_down is deprecated with cuda 9+. use newer variants
  unsigned int active = __activemask();
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(active, val, offset);
  }
#else
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_down(val, offset);
  }
#endif
  return val;
}

__inline__ __device__ float dot(float a, float b, float *shared) {
  // figure out the warp/ position inside the warp
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;

  // partially reduce the dot product inside each warp using a shuffle
  float val = a * b;
  val = warp_reduce_sum(val);

  // write out the partial reduction to shared memory if appropriate
  if (lane == 0) {
    shared[warp] = val;
  }
  __syncthreads();

  // if we we don't have multiple warps, we're done
  if (blockDim.x <= WARP_SIZE) {
    return shared[0];
  }

  // otherwise reduce again in the first warp
  if (warp == 0) {
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    val = (lane < num_warps) ? shared[lane] : 0;
    val = warp_reduce_sum(val);
    // broadcast back to shared memory
    if (threadIdx.x == 0) {
      shared[0] = val;
    }
  }
  __syncthreads();
  return shared[0];
}
} // namespace gpu
} // namespace implicit
#endif // IMPLICIT_GPU_DOT_CUH_

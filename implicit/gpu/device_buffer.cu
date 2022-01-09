#include <cuda_runtime.h>
#include <iostream>

#include "implicit/gpu/device_buffer.h"
#include "implicit/gpu/utils.cuh"

namespace implicit {
namespace gpu {

template <typename T> DeviceBuffer<T>::DeviceBuffer(size_t size) : size_(size) {
  // TODO: support custom allocators (rmm etc) ?
  CHECK_CUDA(cudaMalloc(&data, size * sizeof(T)));
}

template <typename T> DeviceBuffer<T>::~DeviceBuffer() {
  auto err = cudaFree(data);
  if (err != cudaSuccess) {
    std::cerr << "Failed to call cudaFree in ~DeviceBuffer:"
              << cudaGetErrorString(err) << std::endl;
  }
}

template <typename T> DeviceBuffer<T>::DeviceBuffer(DeviceBuffer<T> &&other) {
  if (this != &other) {
    data = other.data;
    size_ = other.size_;
    other.data = NULL;
    other.size_ = 0;
  }
}

template struct DeviceBuffer<int>;
template struct DeviceBuffer<float>;
template struct DeviceBuffer<char>;
} // namespace gpu
} // namespace implicit

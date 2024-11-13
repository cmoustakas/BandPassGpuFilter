#pragma once

#include <ErrChecker.hpp>
#include <cuda.h>
#include <cuda_runtime.h>

namespace gpudenoise {

template <typename T> class CudaUniquePtr {
public:
  CudaUniquePtr(unsigned int len) { allocPriv(len); };
  CudaUniquePtr() = default;
  ~CudaUniquePtr() {
    if (device_data) {
      cudaFree(device_data);
    }
  }

  void make_unique(unsigned int len = 1) {
    if (!device_data) {
      allocPriv(len);
    }
  }

  void release() {
    if (device_data) {
      cudaFree(device_data);
    }
  }

  T *get() { return device_data; }

private:
  void allocPriv(unsigned int len = 1) {
    if (device_data == nullptr) {
      CUDA_CHECK(cudaMalloc(&device_data, sizeof(T) * len));
    }
  }

  T *device_data = nullptr;
};

} // namespace gpudenoise

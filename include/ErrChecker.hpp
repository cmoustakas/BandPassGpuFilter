#pragma once

#include <iostream>

#include <cuda_runtime.h>
#include <cufft.h>
// Macro that checks a condition and throws a runtime_error if the condition is
// false
#define CHECK_THROW(condition, message)                                        \
  do {                                                                         \
    if (condition) {                                                           \
      throw std::runtime_error(message);                                       \
    }                                                                          \
  } while (0)

#define CHECK_AVERROR(condition)                                               \
  do {                                                                         \
    if (condition != 0) {                                                      \
      return false;                                                            \
    }                                                                          \
  } while (0)

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__     \
                << ": " << cudaGetErrorString(err) << "\n";                    \
      throw std::runtime_error(cudaGetErrorString(err));                       \
    }                                                                          \
  } while (0)

#define CUFFT_CHECK(call)                                                      \
  do {                                                                         \
    cufftResult err = call;                                                    \
    if (err != CUFFT_SUCCESS) {                                                \
      std::cerr << "cuFFT error in " << __FILE__ << " at line " << __LINE__    \
                << ": " << err << "\n";                                        \
      throw std::runtime_error("cuFFT error");                                 \
    }                                                                          \
  } while (0)

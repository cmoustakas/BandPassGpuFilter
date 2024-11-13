#include "CudaDenoise.hpp"

#include <cuda_runtime.h>
#include <cufft.h>

#include <cassert>

#include <CudaMemory.hpp>
#include <ErrChecker.hpp>

constexpr int kBandHigh19KHz = 19e3;
constexpr int kBandLow1KHz = 1e3;

template <typename T> T divUp(T a, T b) { return a / b + 1; }

namespace gpudenoise {

__global__ void bandPassKernel(cufftComplex *fft, int cut_off_h, int cut_off_l, int length) {

  const int tid_init = blockIdx.x * blockDim.x + threadIdx.x;
  // Utilize cache lines with stride loop
  const int stride = gridDim.x * blockDim.x;

  for (int tid = tid_init; tid < length; tid += stride) {
    const int curr_freq = (tid <= (length / 2)) ? tid : length - tid;
    const bool inside_spectrum = curr_freq < cut_off_h && curr_freq > cut_off_l;
    if (inside_spectrum) {
      continue;
    }
    // Cut the frequencies outside spectrum
    fft[tid].x = 0.0f;
    fft[tid].y = 0.0f;
  }
}

static inline int calculateCutoffFrequency(const int sample_rate,
                                           const int signal_length, const int marginal_freq) {
  const int freq_resolution = divUp(sample_rate, signal_length);
  return marginal_freq / freq_resolution;
}

static inline void cudaFftSingleDim(size_t signal_length, float *dev_signal,
                                    cufftComplex *fourier_transf) {

  cufftHandle handler_fft;
  CUFFT_CHECK(cufftPlan1d(&handler_fft, signal_length, CUFFT_R2C, 1));
  CUFFT_CHECK(cufftExecR2C(handler_fft, dev_signal, fourier_transf));
  CUFFT_CHECK(cufftDestroy(handler_fft));
}

static inline void cudaFftSingleDimInverse(size_t fft_ptr_length,
                                           float *dev_signal,
                                           cufftComplex *fourier_transf) {

  // Apply inverse Fourier to device pointer and copy back to host
  cufftHandle handler_fft_inv;
  CUFFT_CHECK(cufftPlan1d(&handler_fft_inv, fft_ptr_length, CUFFT_C2R, 1));
  CUFFT_CHECK(cufftExecC2R(handler_fft_inv, fourier_transf, dev_signal));
  CUFFT_CHECK(cufftDestroy(handler_fft_inv));
}

static void applyBandPassFilter(cufftComplex *fft, const size_t length,
                               const int cutoff_freq_h, const int cutoff_freq_l) {
  const size_t max_threads = 1024;
  const size_t workload_per_thread = 2;

  const size_t block_size = divUp<size_t>(max_threads, workload_per_thread);
  const size_t grid_size = divUp<size_t>(length, block_size);

  cudaStream_t kernel_strm;
  CUDA_CHECK(cudaStreamCreate(&kernel_strm));

  bandPassKernel<<<grid_size, block_size, 0, kernel_strm>>>(fft, cutoff_freq_h, cutoff_freq_l,
                                                           length);

  CUDA_CHECK(cudaStreamSynchronize(kernel_strm));
  CUDA_CHECK(cudaPeekAtLastError());
  CUDA_CHECK(cudaStreamDestroy(kernel_strm));
}

static inline void normalizeSignal(float *signal, const size_t signal_length) {
  CHECK_THROW((signal_length <= 0 || signal == nullptr),
              "Invalid signal for normalization \n");


  // Find the maximum absolute value in the array
  float max_magnitude = 0.0f;
  for (size_t i = 0; i < signal_length; ++i)
  {
      max_magnitude = std::max(max_magnitude, std::abs(signal[i]));
  }

  if (max_magnitude == 0.0f){
      return;
  }

  for (size_t i = 0; i < signal_length; ++i)
  {
      signal[i] /= max_magnitude;
  }
}

void gpuDenoiseSignal(float *audio_signal, const int sample_rate,
                      const size_t signal_length) {

  assert(signal_length > 0);

  // Upload host data to gpu
  cudaStream_t copy_strm;
  CudaUniquePtr<float> dev_audio_signal(signal_length);
  CUDA_CHECK(cudaStreamCreate(&copy_strm));
  CUDA_CHECK(cudaMemcpyAsync(dev_audio_signal.get(), audio_signal,
                             signal_length * sizeof(float),
                             cudaMemcpyHostToDevice, copy_strm));

  // Calculate FFT on device audio data
  const size_t fft_ptr_length = divUp<unsigned int>(signal_length, 2);

  CudaUniquePtr<cufftComplex> dev_fourier;
  dev_fourier.make_unique(fft_ptr_length);
  if (copy_strm) {
    CUDA_CHECK(cudaStreamSynchronize(copy_strm));
  }

  const int cutoff_freq_high = calculateCutoffFrequency(sample_rate, signal_length, kBandHigh19KHz );
  const int cutoff_freq_low = calculateCutoffFrequency(sample_rate, signal_length, kBandLow1KHz);

  cudaFftSingleDim(signal_length, dev_audio_signal.get(), dev_fourier.get());
  applyBandPassFilter(dev_fourier.get(), fft_ptr_length, cutoff_freq_high, cutoff_freq_low);
  cudaFftSingleDimInverse(signal_length, dev_audio_signal.get(),
                          dev_fourier.get());

  CUDA_CHECK(cudaMemcpy(audio_signal, dev_audio_signal.get(),
                        signal_length * sizeof(float), cudaMemcpyDeviceToHost));
//  normalizeSignal(audio_signal, signal_length);
}

} // namespace gpudenoise

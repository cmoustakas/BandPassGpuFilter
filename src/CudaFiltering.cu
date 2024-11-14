#include "CudaFiltering.hpp"

#include <cuda_runtime.h>
#include <cufft.h>

#include <cassert>

#include <CudaMemory.hpp>
#include <MacroHelpers.hpp>

constexpr int kBandHigh = 16e3;
constexpr int kBandLow = 2e2;

template <typename T> T divUp(T a, T b) { return a / b + 1; }

namespace gpufilter {

__global__ void bandPassKernel(cufftComplex *fft, int cut_off_h, int cut_off_l,
                               int length) {

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

static inline int calculateMarginalFrequency(const int sample_rate,
                                             const int signal_length,
                                             const int marginal_freq) {
  const int freq_resolution = divUp(sample_rate, signal_length);
  return marginal_freq / freq_resolution;
}

static void applyBandPassFilter(cufftComplex *fft, const size_t length,
                                const int cutoff_freq_h,
                                const int cutoff_freq_l) {
  const size_t max_threads = 1024;
  const size_t workload_per_thread = 2;

  const size_t block_size = divUp<size_t>(max_threads, workload_per_thread);
  const size_t grid_size = divUp<size_t>(length, block_size);

  cudaStream_t kernel_strm;
  CUDA_CHECK(cudaStreamCreate(&kernel_strm));

  bandPassKernel<<<grid_size, block_size, 0, kernel_strm>>>(
      fft, cutoff_freq_h, cutoff_freq_l, length);

  CUDA_CHECK(cudaStreamSynchronize(kernel_strm));
  CUDA_CHECK(cudaPeekAtLastError());
  CUDA_CHECK(cudaStreamDestroy(kernel_strm));
}

static inline void normalizeSignal(float *signal, const size_t signal_length) {
  CHECK_THROW((signal_length <= 0 || signal == nullptr),
              "Invalid signal for normalization \n");

  // Find the maximum absolute value in the array
  float max_magnitude = 0.0f;
  for (size_t i = 0; i < signal_length; ++i) {
    max_magnitude = std::max(max_magnitude, std::abs(signal[i]));
  }

  if (max_magnitude == 0.0f) {
    return;
  }

  for (size_t i = 0; i < signal_length; ++i) {
    signal[i] /= max_magnitude;
  }
}

class CudaFilterHandler::FFThandlerPriv {
public:
  ~FFThandlerPriv() {
    cufftDestroy(m_fft);
    cufftDestroy(m_fft_inverse);
  }

  cufftHandle m_fft;
  cufftHandle m_fft_inverse;
};

CudaFilterHandler::CudaFilterHandler(const int signal_length)
    : m_signal_length(signal_length) {
  m_fft_handler = std::make_unique<FFThandlerPriv>();
  setUp();
}

void CudaFilterHandler::setUp() {
  CUFFT_CHECK(
      cufftPlan1d(&m_fft_handler.get()->m_fft, m_signal_length, CUFFT_R2C, 1));
  CUFFT_CHECK(cufftPlan1d(&m_fft_handler.get()->m_fft_inverse, m_signal_length,
                          CUFFT_C2R, 1));

  // Calculate FFT on device audio data
  m_fft_ptr_length = divUp<unsigned int>(m_signal_length, 2);

  m_dev_audio_signal.make_unique(m_signal_length);
  m_dev_fourier.make_unique(m_fft_ptr_length);
}

void CudaFilterHandler::filterSignal(float *audio_signal,
                                     const int sample_rate) {

  assert(m_signal_length > 0);

  // Upload host data to gpu
  cudaStream_t copy_strm;

  CUDA_CHECK(cudaStreamCreate(&copy_strm));
  CUDA_CHECK(cudaMemcpyAsync(m_dev_audio_signal.get(), audio_signal,
                             m_signal_length * sizeof(float),
                             cudaMemcpyHostToDevice, copy_strm));

  // Calculate marginal frequencies
  const int cutoff_freq_high =
      calculateMarginalFrequency(sample_rate, m_signal_length, kBandHigh);
  const int cutoff_freq_low =
      calculateMarginalFrequency(sample_rate, m_signal_length, kBandLow);

  CUDA_CHECK(cudaStreamSynchronize(copy_strm));

  // Calculate FFT
  CUFFT_CHECK(cufftExecR2C(m_fft_handler.get()->m_fft, m_dev_audio_signal.get(),
                           m_dev_fourier.get()));

  // Apply band pass filter on Fourier coeffs
  applyBandPassFilter(m_dev_fourier.get(), m_fft_ptr_length, cutoff_freq_high,
                      cutoff_freq_low);

  // Apply inverse Fourier to return to time domain
  CUFFT_CHECK(cufftExecC2R(m_fft_handler.get()->m_fft_inverse,
                           m_dev_fourier.get(), m_dev_audio_signal.get()));

  CUDA_CHECK(cudaMemcpy(audio_signal, m_dev_audio_signal.get(),
                        m_signal_length * sizeof(float),
                        cudaMemcpyDeviceToHost));

  //  normalizeSignal(audio_signal, m_signal_length);
  CUDA_CHECK(cudaStreamDestroy(copy_strm));
}

} // namespace gpufilter

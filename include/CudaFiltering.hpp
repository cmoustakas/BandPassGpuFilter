#pragma once
#include <CudaMemory.hpp>
#include <memory>
#include <stddef.h>

namespace gpufilter {

class CudaFilterHandler {
public:
  CudaFilterHandler(const int signal_length);
  ~CudaFilterHandler();

  /**
   * @brief filterSignal
   * @param  packet
   */
  void filterSignal(float *audio_data, const int sample_rate);

private:
  // PIMPL to hide the implementation from the world
  class FFThandlerPriv;

  /**
   * @brief setUpFFT
   */
  void setUp();

  // The length of the signal
  size_t m_signal_length = 0;

  // Num of fourier coefficients
  size_t m_fft_ptr_length = 0;

  std::unique_ptr<FFThandlerPriv> m_fft_handler;

  // GPU pointer that points to fourier complex coefficients
  CudaUniquePtr<cufftComplex> m_dev_fourier;

  // GPU pointer for the audio signal data
  CudaUniquePtr<float> m_dev_audio_signal;
};
} // namespace gpufilter

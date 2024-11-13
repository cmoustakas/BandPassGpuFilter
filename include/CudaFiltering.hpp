#pragma once
#include <stddef.h>

namespace gpufilter {

/**
 * @brief gpuDenoiseAudio
 * @param  packet
 */
void gpuFilterSignal(float *audio_data, const int sample_rate,
                      const size_t signal_length);

} // namespace gpudenoise

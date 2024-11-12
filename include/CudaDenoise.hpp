#pragma once
#include <stddef.h>

namespace gpudenoise {

/**
 * @brief gpuDenoiseAudio
 * @param  packet
 */
void gpuDenoiseSignal(float *audio_data, const int sample_rate,
                      const size_t signal_length);

} // namespace gpudenoise

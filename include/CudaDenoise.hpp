#pragma once
#include <stddef.h>
#include <vector>

namespace gpudenoise {

void gpuDenoiseData(float *audio_data, const int sample_rate,
                    const size_t data_length);
} // namespace gpudenoise

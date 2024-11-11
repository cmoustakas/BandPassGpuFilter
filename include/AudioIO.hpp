#pragma once

#include <juce_audio_formats/juce_audio_formats.h>

#include <string>

constexpr int kInvalid = -1;

namespace gpudenoise {

// Forward declare metadata of audio signal
struct AudioMetaData {
  juce::AudioBuffer<float> signal;
  int sample_rate = kInvalid;

  bool isValid() { return sample_rate > 0 && !signal.hasBeenCleared(); }
};

/**
 * @brief loadAudioBufferFromWAV
 * @param path_to_file
 * @param out_buffer
 * @return
 */
AudioMetaData loadAudioBufferFromWAV(const std::string_view &path_to_file);

/**
 * @brief saveWAVfromAudioBuffer
 * @param path_to_file
 * @param in_buffer
 * @return
 */
bool saveWAVfromAudioBuffer(const std::string_view &path_to_file,
                            const AudioMetaData &input_audio_packet);

/**
 * @brief exportSignalToCSV
 * @param csv
 * @param audio_packet
 * @return
 */
void exportSignalToCSV(const std::string_view &csv,
                       const AudioMetaData &audio_packet);

} // namespace gpudenoise

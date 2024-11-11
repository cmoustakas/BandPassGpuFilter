#include "AudioIO.hpp"

#include <cassert>
#include <juce_core/juce_core.h>

#include <ErrChecker.hpp>

namespace gpudenoise {

AudioMetaData loadAudioBufferFromWAV(const std::string_view &path_to_file) {

  juce::File wav_file(path_to_file.data());
  // Ensure the file exists
  CHECK_THROW(wav_file.existsAsFile() == false, "WAV file does not exist");

  // Register formats (WAV is included by default)
  juce::AudioFormatManager formatManager;
  formatManager.registerBasicFormats();

  // Create a reader for the WAV file
  std::unique_ptr<juce::AudioFormatReader> reader(
      formatManager.createReaderFor(wav_file));

  CHECK_THROW(reader == nullptr, "Failed to create reader for WAV file");

  // Prepare an audio buffer
  const int num_samples = static_cast<int>(reader->lengthInSamples);
  const int num_channels = static_cast<int>(reader->numChannels);
  const int sample_rate = static_cast<int>(reader->sampleRate);

  juce::AudioBuffer<float> audio_signal(num_channels, num_samples);

  // Read the data into the buffer
  CHECK_THROW(reader->read(&audio_signal, 0, num_samples, 0, true, true) ==
                  false,
              "Failed to read signal from WAV file");

  AudioMetaData packet = {audio_signal, sample_rate};
  assert(packet.isValid());

  return packet;
}

bool saveWAVfromAudioBuffer(const std::string_view &path_to_file,
                            const AudioMetaData &input_audio_packet) {

  juce::File out_wav_file(path_to_file.data());

  const int channels = input_audio_packet.signal.getNumChannels();

  //   Create an output stream for the output file
  std::unique_ptr<juce::FileOutputStream> outputStream =
      out_wav_file.createOutputStream();

  juce::AudioFormatManager formatManager;
  formatManager.registerBasicFormats();

  std::unique_ptr<juce::AudioFormatWriter> format_writer(
      formatManager.findFormatForFileExtension(".wav")->createWriterFor(
          outputStream.get(), input_audio_packet.sample_rate, channels, 16, {},
          0));

  // Give the ownership to format_writer, otherwise double free
  outputStream.release();

  std::unique_ptr<juce::AudioFormatWriter> writer(format_writer.get());

  // Pass ownership to writer, otherwise double free
  format_writer.release();

  CHECK_THROW(writer == nullptr, "Failed to create writer for output WAV file");

  // Write the data from the audioBuffer into the new file
  writer->writeFromAudioSampleBuffer(input_audio_packet.signal, 0,
                                     input_audio_packet.signal.getNumSamples());

  return true;
}
} // namespace gpudenoise

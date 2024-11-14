#include <AudioIO.hpp>
#include <CudaFiltering.hpp>
#include <MacroHelpers.hpp>

#include <cassert>
#include <iostream>

void printUsg() { std::cout << "./gpufiltering /path/to/file.mp3 \n"; }

void runDemo(const std::string_view &audio_file) {

  // Load as packet the .wav file
  auto audio_packet = gpufilter::loadAudioBufferFromWAV(audio_file);
  assert(audio_packet.isValid());
  gpufilter::exportSignalToCSV("/tmp/clean_signal.csv", audio_packet);

  // Denoise in GPU the "noisy" signal for each channel, dependency injection is
  // not possible due to the fact the nvcc and JUCE are somehow incompatible
  const int num_of_samples = audio_packet.signal.getNumSamples();
  const int num_of_channels = audio_packet.signal.getNumChannels();
  const int sample_rate = audio_packet.sample_rate;

  gpufilter::CudaFilterHandler cuda_filter_handler(sample_rate);

  for (int channel = 0; channel < num_of_channels; ++channel) {
    float *audio_signal = audio_packet.signal.getWritePointer(channel);
    cuda_filter_handler.filterSignal(audio_signal, sample_rate);
  }

  gpufilter::exportSignalToCSV("/tmp/clean_signal.csv", audio_packet);

  // Save as .wav file the clean signal
  gpufilter::saveWAVfromAudioBuffer("/tmp/clean.wav", audio_packet);
}

int main(int argc, char *argv[]) {

  CHECK_THROW(argc == 2, "Invalid input to executable \n");
  std::string_view audio_file = argv[1];
  runDemo(audio_file);

  return 0;
}

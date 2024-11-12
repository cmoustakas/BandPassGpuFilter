#include <AudioIO.hpp>
#include <CudaDenoise.hpp>
#include <ErrChecker.hpp>

#include <cassert>
#include <iostream>

void printUsg() { std::cout << "./gpudenoiser /path/to/file.mp3 \n"; }

void runDemo(const std::string_view &audio_file) {

  // Load as packet the .wav file
  auto audio_packet = gpudenoise::loadAudioBufferFromWAV(audio_file);
  assert(audio_packet.isValid());
  gpudenoise::exportSignalToCSV("/tmp/clean_signal.csv", audio_packet);

  // Denoise in GPU the "noisy" signal for each channel, dependency injection is
  // not possible due to the fact the nvcc and JUCE are somehow incompatible
  const int num_of_samples = audio_packet.signal.getNumSamples();
  const int num_of_channels = audio_packet.signal.getNumChannels();
  const int sample_rate = audio_packet.sample_rate;

  for (int channel = 0; channel < num_of_channels; ++channel) {
    float *audio_signal = audio_packet.signal.getWritePointer(channel);
    gpudenoise::gpuDenoiseSignal(audio_signal, sample_rate, num_of_samples);
  }

  gpudenoise::exportSignalToCSV("/tmp/clean_signal.csv", audio_packet);

  // Save as .wav file the clean signal
  gpudenoise::saveWAVfromAudioBuffer("/tmp/clean.wav", audio_packet);
}

int main(int argc, char *argv[]) {

  CHECK_THROW(argc == 2, "Invalid input to executable \n");
  std::string_view audio_file = argv[1];
  runDemo(audio_file);

  return 0;
}

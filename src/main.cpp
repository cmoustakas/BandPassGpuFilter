#include <CudaDenoise.hpp>
#include <FfmpegHandler.hpp>

#include <iostream>

void printUsg() { std::cout << "./gpudenoiser /path/to/file.mp3 \n"; }

void runDemo(const std::string_view &audio_file) {

  gpudenoise::FfmpegHandler ffmpeg_handler(audio_file);

  const int cutoff_freq = 50;
  float *audio_data = (float *)ffmpeg_handler.getAudioBuffer().data();
  const size_t data_length = ffmpeg_handler.getAudioBuffer().size();
  const int sample_rate = ffmpeg_handler.getSampleRate();

  gpudenoise::gpuDenoiseData(audio_data, sample_rate, data_length);
}

int main(int argc, char *argv[]) {

  if (argc != 2) {
    printUsg();
    throw std::runtime_error("Invalid input to executable \n");
  }

  std::string_view audio_file = argv[1];
  runDemo(audio_file);

  return 0;
}

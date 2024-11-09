#include <gtest/gtest.h>

#include <CudaDenoise.hpp>
#include <FfmpegHandler.hpp>

#include <iostream>
#include <string>

TEST(FfmpegTest, throwOnDummyFilename) {
  std::string_view filename = "/tmp/jdwnkjnwnwjkndjw_nosense.mp3";
  gpudenoise::FfmpegHandler handler;
  EXPECT_THROW(handler.processAudioFile(filename), std::runtime_error);
}

TEST(FfmpegTest, noThrowFilename) {
  std::string_view filename = "/home/harrismoustakas/Development/Blog/"
                              "SoundDenoiser/test_data/noisy.mp3";
  gpudenoise::FfmpegHandler handler;
  EXPECT_NO_THROW(handler.processAudioFile(filename));
}

TEST(FfmpegTest, checkMP3Save) {
  std::string_view filename = "/home/harrismoustakas/Development/Blog/"
                              "SoundDenoiser/test_data/noisy.mp3";
  gpudenoise::FfmpegHandler handler;
  handler.processAudioFile(filename);

  std::vector<uint8_t> initial_signal(handler.getAudioBuffer());

  EXPECT_NO_THROW(gpudenoise::saveAsMP3(initial_signal, "/tmp/noisy_copy.mp3",
                                        handler.getSampleRate(),
                                        handler.getBitRate()));

  gpudenoise::FfmpegHandler copied_handler("/tmp/noisy_copy.mp3");
  std::vector<uint8_t> &expected_identical_signal =
      copied_handler.getAudioBuffer();

  EXPECT_EQ(expected_identical_signal.size(), initial_signal.size());

  for (size_t idx = 0; idx < expected_identical_signal.size(); ++idx) {
    EXPECT_EQ(expected_identical_signal[idx], initial_signal[idx]);
  }
}

TEST(DenoiseTest, differentSignals) {
  std::string_view filename = "/home/harrismoustakas/Development/Blog/"
                              "SoundDenoiser/test_data/noisy.mp3";

  gpudenoise::FfmpegHandler handler(filename);
  // Make a copy to the initial audio signal
  std::vector<uint8_t> initial_signal(handler.getAudioBuffer());

  gpudenoise::gpuDenoiseData((float *)handler.getAudioBuffer().data(),
                             handler.getSampleRate(), initial_signal.size());

  std::vector<uint8_t> &filtered_signal = handler.getAudioBuffer();

  double diff_signals = 0.0;

  EXPECT_TRUE(filtered_signal.data() != initial_signal.data());
  EXPECT_EQ(filtered_signal.size(), initial_signal.size());

  for (size_t idx = 0; idx < initial_signal.size(); ++idx) {
    diff_signals += std::abs(filtered_signal[idx] - initial_signal[idx]);
  }

  EXPECT_TRUE(diff_signals > 0);
}

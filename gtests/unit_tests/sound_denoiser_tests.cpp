#include <gtest/gtest.h>

#include <AudioIO.hpp>
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
                              "gpuDenoiser/test_data/noisy.mp3";
  gpudenoise::FfmpegHandler handler;
  EXPECT_NO_THROW(handler.processAudioFile(filename));
}

TEST(FfmpegTest, checkMP3Save) {
  std::string_view filename = "/home/harrismoustakas/Development/Blog/"
                              "gpuDenoiser/test_data/noisy.mp3";
  gpudenoise::FfmpegHandler handler;
  handler.processAudioFile(filename);

  std::vector<uint8_t> initial_signal(handler.getAudioBuffer());

  const auto sample_rate = handler.getSampleRate();
  const auto bit_rate = handler.getBitRate();
  const auto channels = handler.getChannels();
  const auto sample_fmt = handler.getSampleFmt();
  const auto file = "/tmp/noisy_copy.mp3";
  EXPECT_NO_THROW(gpudenoise::saveAsMP3(
      initial_signal.data(), (int64_t)initial_signal.size(), sample_rate,
      channels, bit_rate, sample_fmt, file));

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
                              "gpuDenoiser/test_data/noisy.mp3";

  gpudenoise::FfmpegHandler handler(filename);
  // Make a copy to the ivnitial audio signal
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

TEST(AudioIO, aduioIOCycle) {

  constexpr float kErrorTolerance = 1e-3;

  const std::string_view input =
      "/home/harrismoustakas/Development/Blog/gpuDenoiser/test_data/noisy.wav";

  const std::string_view copy = "/tmp/noisy_cpy.wav";

  gpudenoise::AudioMetaData proto_packet;
  EXPECT_NO_THROW(proto_packet = gpudenoise::loadAudioBufferFromWAV(input));
  EXPECT_TRUE(proto_packet.isValid());

  EXPECT_TRUE(gpudenoise::saveWAVfromAudioBuffer(copy, proto_packet));

  gpudenoise::AudioMetaData identical_packet;
  EXPECT_NO_THROW(identical_packet = gpudenoise::loadAudioBufferFromWAV(copy));

  EXPECT_EQ(identical_packet.sample_rate, proto_packet.sample_rate);
  EXPECT_EQ(identical_packet.signal.getNumChannels(),
            proto_packet.signal.getNumChannels());
  EXPECT_EQ(identical_packet.signal.getNumSamples(),
            proto_packet.signal.getNumSamples());

  const int num_of_samples = proto_packet.signal.getNumSamples();
  const int num_of_channels = proto_packet.signal.getNumChannels();

  for (int channel = 0; channel < num_of_channels; ++channel) {
    float *channel_data_proto = proto_packet.signal.getWritePointer(channel);
    float *channel_data_copy = identical_packet.signal.getWritePointer(channel);

    for (int sample = 0; sample < num_of_samples; ++sample) {
      EXPECT_NEAR(channel_data_copy[sample], channel_data_proto[sample],
                  kErrorTolerance);
    }
  }
}

TEST(AudioIO, aduioIOExportCSV) {
  const std::string_view input =
      "/home/harrismoustakas/Development/Blog/gpuDenoiser/test_data/noisy.wav";

  const std::string_view csv = "/tmp/audio_signal.csv";

  gpudenoise::AudioMetaData audio_packet;
  EXPECT_NO_THROW(audio_packet = gpudenoise::loadAudioBufferFromWAV(input));
  EXPECT_TRUE(audio_packet.isValid());

  EXPECT_NO_THROW(gpudenoise::exportSignalToCSV(csv, audio_packet));
}

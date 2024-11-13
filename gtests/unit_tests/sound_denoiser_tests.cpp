#include <gtest/gtest.h>

#include <AudioIO.hpp>
#include <CudaFiltering.hpp>
#include <ErrChecker.hpp>
#include <FfmpegHandler.hpp>

#include <iostream>
#include <string>

TEST(FfmpegTest, throwOnDummyFilename) {
  GTEST_SKIP();

  std::string_view filename = "/tmp/jdwnkjnwnwjkndjw_nosense.mp3";
  gpufilter::FfmpegHandler handler;
  EXPECT_THROW(handler.processAudioFile(filename), std::runtime_error);
}

TEST(FfmpegTest, noThrowFilename) {
  GTEST_SKIP();

  std::string_view filename = "/home/harrismoustakas/Development/Blog/"
                              "gpuDenoiser/test_data/noisy.mp3";
  gpufilter::FfmpegHandler handler;
  EXPECT_NO_THROW(handler.processAudioFile(filename));
}

TEST(FfmpegTest, checkMP3Save) {
  GTEST_SKIP();

  std::string_view filename = "/home/harrismoustakas/Development/Blog/"
                              "gpuDenoiser/test_data/noisy.mp3";
  gpufilter::FfmpegHandler handler;
  handler.processAudioFile(filename);

  std::vector<uint8_t> initial_signal(handler.getAudioBuffer());

  const auto sample_rate = handler.getSampleRate();
  const auto bit_rate = handler.getBitRate();
  const auto channels = handler.getChannels();
  const auto sample_fmt = handler.getSampleFmt();
  const auto file = "/tmp/noisy_copy.mp3";
  EXPECT_NO_THROW(gpufilter::saveAsMP3(
      initial_signal.data(), (int64_t)initial_signal.size(), sample_rate,
      channels, bit_rate, sample_fmt, file));

  gpufilter::FfmpegHandler copied_handler("/tmp/noisy_copy.mp3");
  std::vector<uint8_t> &expected_identical_signal =
      copied_handler.getAudioBuffer();

  EXPECT_EQ(expected_identical_signal.size(), initial_signal.size());
  for (size_t idx = 0; idx < expected_identical_signal.size(); ++idx) {
    EXPECT_EQ(expected_identical_signal[idx], initial_signal[idx]);
  }
}

TEST(DenoiseTest, differentSignals) {
  GTEST_SKIP();

  std::string_view filename =
      "/home/harrismoustakas/Development/Blog/gpuDenoiser/test_data/noisy.wav";

  gpufilter::FfmpegHandler handler(filename);
  // Make a copy to the ivnitial audio signal
  std::vector<uint8_t> initial_signal(handler.getAudioBuffer());

  gpufilter::gpuFilterSignal((float *)handler.getAudioBuffer().data(),
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

  gpufilter::AudioMetaData proto_packet;
  EXPECT_NO_THROW(proto_packet = gpufilter::loadAudioBufferFromWAV(input));
  EXPECT_TRUE(proto_packet.isValid());

  EXPECT_TRUE(gpufilter::saveWAVfromAudioBuffer(copy, proto_packet));

  gpufilter::AudioMetaData identical_packet;
  EXPECT_NO_THROW(identical_packet = gpufilter::loadAudioBufferFromWAV(copy));

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

  gpufilter::AudioMetaData audio_packet;
  EXPECT_NO_THROW(audio_packet = gpufilter::loadAudioBufferFromWAV(input));
  EXPECT_TRUE(audio_packet.isValid());

  EXPECT_NO_THROW(gpufilter::exportSignalToCSV(csv, audio_packet));
}

TEST(GpuDenoise, denoisePipeline) {
  const std::string_view input =
      "/home/harrismoustakas/Development/Blog/gpuDenoiser/test_data/noisy.wav";

  const std::string_view csv = "/tmp/audio_signal_test.csv";

  gpufilter::AudioMetaData audio_packet;
  EXPECT_NO_THROW(audio_packet = gpufilter::loadAudioBufferFromWAV(input));
  EXPECT_TRUE(audio_packet.isValid());

  EXPECT_NO_THROW(gpufilter::exportSignalToCSV(csv, audio_packet));

  float *copy_signal;

  const int num_of_samples = audio_packet.signal.getNumSamples();
  const int num_of_channels = audio_packet.signal.getNumChannels();
  const int sample_rate = audio_packet.sample_rate;

  for (int channel = 0; channel < num_of_channels; ++channel) {
    float *audio_signal = audio_packet.signal.getWritePointer(channel);
    EXPECT_NO_THROW(
        gpufilter::gpuFilterSignal(audio_signal, sample_rate, num_of_samples));
  }

  EXPECT_NO_THROW(
      gpufilter::exportSignalToCSV("/tmp/clean_signal_test.csv", audio_packet));

  EXPECT_NO_THROW(
      gpufilter::saveWAVfromAudioBuffer("/tmp/clean_signal.wav", audio_packet));
}

TEST(GpuDenoise, performanceMeasure) {
  const std::string_view input =
      "/home/harrismoustakas/Development/Blog/gpuDenoiser/test_data/long.wav";

  auto gpuBandPassDuration = [](gpufilter::AudioMetaData &audio_packet) {
    const int num_of_samples = audio_packet.signal.getNumSamples();
    const int num_of_channels = audio_packet.signal.getNumChannels();
    const int sample_rate = audio_packet.sample_rate;

    float duration = 0.f;
    for (int channel = 0; channel < num_of_channels; ++channel) {
      float *audio_signal = audio_packet.signal.getWritePointer(channel);
      duration += BENCHMARK(
          gpufilter::gpuFilterSignal(audio_signal, sample_rate, num_of_samples),
          1);
    }
    return duration / num_of_channels;
  };

  auto cpuBandPassDuration =
      [](std::vector<juce::AudioBuffer<float>> &audio_signals,
         int sample_rate) {
        float duration = 0.f;
        for (auto &buff : audio_signals) {
          duration +=
              BENCHMARK(gpufilter::cpuBandPassFilter(buff, sample_rate), 1);
        }
        return duration / audio_signals.size();
      };

  gpufilter::AudioMetaData audio_packet_cpu;
  EXPECT_NO_THROW(audio_packet_cpu = gpufilter::loadAudioBufferFromWAV(input));
  EXPECT_TRUE(audio_packet_cpu.isValid());

  gpufilter::AudioMetaData audio_packet_gpu;
  EXPECT_NO_THROW(audio_packet_gpu = gpufilter::loadAudioBufferFromWAV(input));
  EXPECT_TRUE(audio_packet_gpu.isValid());

  const size_t total_iterations = 1e2;

  int num_channels = audio_packet_cpu.signal.getNumChannels();
  int num_samples = audio_packet_cpu.signal.getNumSamples();

  // Create an array of AudioBuffers to hold each channel
  std::vector<juce::AudioBuffer<float>> single_channel_buffers;
  // Loop over all channels
  for (int i = 0; i < num_channels; ++i) {
    juce::AudioBuffer<float> mono_buffer(1,
                                         num_samples); // Only 1 channel
    mono_buffer.copyFrom(0, 0, audio_packet_cpu.signal, i, 0, num_samples);
    single_channel_buffers.push_back(std::move(mono_buffer));
  }

  const auto cpu_duration =
      cpuBandPassDuration(single_channel_buffers, audio_packet_cpu.sample_rate);

  const auto gpu_duration = gpuBandPassDuration(audio_packet_gpu);

  std::cout << "Gpu duration : " << gpu_duration << "\n";
  std::cout << "Cpu duration : " << cpu_duration << "\n";
}

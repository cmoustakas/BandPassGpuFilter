#pragma once

#include <memory>
#include <string>
#include <vector>

struct SwrContext;
struct AVFormatContext;
struct AVCodecContext;

constexpr int kInvalidSz = -1;
constexpr int kUninitializedIdx = -1;

namespace gpufilter {

class FfmpegHandler {

public:
  FfmpegHandler(const std::string_view &file);
  FfmpegHandler() = default;
  ~FfmpegHandler() = default;

  void processAudioFile(const std::string_view &file);

  /**
   * @brief getAudioBuffer
   * @return
   */
  std::vector<uint8_t> &getAudioBuffer() noexcept;

  /**
   * @brief getSampleRate
   * @return
   */
  int getSampleRate() const noexcept;

  /**
   * @brief getBitRate
   * @return
   */
  int getBitRate() const noexcept;

  /**
   * @brief getChannels
   * @return
   */
  int getChannels() const noexcept;

  /**
   * @brief getSampleFmt
   * @return
   */
  int getSampleFmt() const noexcept;

private:
  /**
   * @brief findAudioStream
   * @return
   */
  bool findAudioStream() noexcept;

  /**
   * @brief initializeResampler
   */
  bool calculateSegmentSize() noexcept;

  /**
   * @brief openCodec
   */
  bool openCodec() noexcept;

  /**
   * @brief extractAudioData
   */
  bool extractAudioData() noexcept;

  /**
   * @brief prepareFilePriv
   * @param file
   */
  void processAudioFilePriv(const std::string_view &file);

  /**
   * @brief updateSamplingRate
   */
  void updateSamplingNBitRate() noexcept;

  AVFormatContext *m_fmt_ctx = nullptr;
  AVCodecContext *m_codec_cxt = nullptr;
  SwrContext *m_swr_ctx = nullptr;

  int m_audio_stream_idx = kUninitializedIdx;
  int m_segment_size = kInvalidSz;
  int m_sample_rate = kInvalidSz;
  int m_bit_rate = kInvalidSz;
  int m_channels = kInvalidSz;
  int m_sample_fmt = kInvalidSz;

  std::vector<uint8_t> m_subsample_buffer;
  std::vector<uint8_t> m_audio_data;
  std::string_view m_file = "";
};

void saveAsMP3(uint8_t *audio_data, size_t data_size, int sample_rate,
               int channels, int bit_rate, int sample_fmt,
               const std::string_view &output_filename);

} // namespace gpudenoise

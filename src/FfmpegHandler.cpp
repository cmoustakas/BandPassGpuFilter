#include "FfmpegHandler.hpp"
#include "ErrChecker.hpp"

#include <cassert>
#include <iostream>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/opt.h>
#include <libswresample/swresample.h>
}

constexpr int kB = 1024;

namespace gpudenoise {
FfmpegHandler::FfmpegHandler(const std::string_view &file) {
  processAudioFilePriv(file);
}

void FfmpegHandler::processAudioFile(const std::string_view &file) {
  processAudioFilePriv(file);
}

std::vector<uint8_t> &FfmpegHandler::getAudioBuffer() noexcept {
  return m_audio_data;
}

int FfmpegHandler::getSampleRate() const noexcept { return m_sample_rate; }

void FfmpegHandler::processAudioFilePriv(const std::string_view &file) {

  CHECK_THROW(avformat_open_input(&m_fmt_ctx, file.data(), nullptr, nullptr) !=
                  0,
              "Failed to open audio file.");

  CHECK_THROW(avformat_find_stream_info(m_fmt_ctx, nullptr) < 0,
              "Failed to retrieve stream info.");

  CHECK_THROW(findAudioStream() == false,
              "Failed to find audio data in the file.");

  CHECK_THROW(openCodec() == false, "Unsupported format.");

  CHECK_THROW(calculateSegmentSize() == false,
              "Failed to initialize resampler");

  CHECK_THROW(extractAudioData() == false,
              "Invalid size of audio data vector returned");

  updateSamplingNBitRate();
  CHECK_THROW(m_sample_rate == kInvalidSz, "Unable to update sample rate");
}

bool FfmpegHandler::findAudioStream() noexcept {
  m_audio_stream_idx = kUninitializedIdx;

  for (unsigned int i = 0; i < m_fmt_ctx->nb_streams; i++) {
    if (m_fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
      m_audio_stream_idx = i;
      break;
    }
  }

  const bool found_stream = m_audio_stream_idx != kUninitializedIdx;
  return found_stream;
}

bool FfmpegHandler::openCodec() noexcept {
  AVCodec *codec = avcodec_find_decoder(
      m_fmt_ctx->streams[m_audio_stream_idx]->codecpar->codec_id);
  if (!codec) {
    return false;
  }

  m_codec_cxt = avcodec_alloc_context3(codec);

  avcodec_parameters_to_context(
      m_codec_cxt, m_fmt_ctx->streams[m_audio_stream_idx]->codecpar);

  if (avcodec_open2(m_codec_cxt, codec, nullptr) < 0) {
    return false;
  }

  return true;
}

bool FfmpegHandler::calculateSegmentSize() noexcept {
  m_swr_ctx = swr_alloc();
  if (!m_swr_ctx) {
    return false;
  }

  CHECK_AVERROR(av_opt_set_int(m_swr_ctx, "in_channel_layout",
                               m_codec_cxt->channel_layout, 0));
  CHECK_AVERROR(
      av_opt_set_int(m_swr_ctx, "in_sample_rate", m_codec_cxt->sample_rate, 0));
  CHECK_AVERROR(av_opt_set_sample_fmt(m_swr_ctx, "in_sample_fmt",
                                      m_codec_cxt->sample_fmt, 0));
  CHECK_AVERROR(av_opt_set_int(m_swr_ctx, "out_channel_layout",
                               m_codec_cxt->channel_layout, 0));
  CHECK_AVERROR(av_opt_set_int(m_swr_ctx, "out_sample_rate",
                               m_codec_cxt->sample_rate, 0));
  CHECK_AVERROR(
      av_opt_set_sample_fmt(m_swr_ctx, "out_sample_fmt", AV_SAMPLE_FMT_FLT, 0));
  CHECK_AVERROR(swr_init(m_swr_ctx));

  const int max_samples =
      m_codec_cxt->frame_size > 0 ? m_codec_cxt->frame_size : 1024;

  m_segment_size = max_samples * av_get_bytes_per_sample(AV_SAMPLE_FMT_FLT) *
                   m_codec_cxt->channels;

  return true;
}

bool FfmpegHandler::extractAudioData() noexcept {

  if (m_segment_size == kInvalidSz) {
    return false;
  }

  int64_t data_sz = 0;

  AVPacket *packet = av_packet_alloc();
  if (!packet) {
    return false;
  }

  AVFrame *frame = av_frame_alloc();
  if (!frame) {
    return false;
  }

  m_audio_data.clear();
  m_audio_data.reserve(m_segment_size);
  m_subsample_buffer.resize(m_segment_size);

  int buffer_size = 0;
  while (av_read_frame(m_fmt_ctx, packet) >= 0) {

    if (packet->stream_index != m_audio_stream_idx ||
        avcodec_send_packet(m_codec_cxt, packet) != 0) {
      continue;
    }

    while (avcodec_receive_frame(m_codec_cxt, frame) == 0) {
      // Set the ptr to avoid casts
      uint8_t *sub_sample_ptr = m_subsample_buffer.data();

      const int num_of_samples = swr_convert(
          m_swr_ctx, &sub_sample_ptr, frame->nb_samples,
          (const uint8_t **)frame->extended_data, frame->nb_samples);

      // Update buffer size
      const int buffer_sz = av_samples_get_buffer_size(
          nullptr, m_codec_cxt->channels, num_of_samples, AV_SAMPLE_FMT_FLT, 1);

      m_audio_data.insert(m_audio_data.end(), m_subsample_buffer.begin(),
                          m_subsample_buffer.begin() + buffer_sz);

      m_audio_data.shrink_to_fit();

      data_sz += buffer_sz;
    }
  }
  av_packet_unref(packet);
  av_frame_free(&frame);

  return true;
}

void FfmpegHandler::updateSamplingNBitRate() noexcept {

  m_sample_rate = kInvalidSz;
  for (int idx = 0; idx < m_fmt_ctx->nb_streams; ++idx) {
    if (m_fmt_ctx->streams[idx]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
      m_sample_rate = m_fmt_ctx->streams[idx]->codecpar->sample_rate;
      m_bit_rate = m_fmt_ctx->streams[idx]->codecpar->bit_rate;
      return;
    }
  }
}

int FfmpegHandler::getBitRate() const noexcept { return m_bit_rate; }

bool saveAsMP3(const std::vector<uint8_t> &audioData,
               const std::string_view &filename, int sample_rate, int bitrate,
               int channels) {
  AVFormatContext *fmt_ctx = nullptr;
  AVCodecContext *codec_ctx = nullptr;
  AVStream *stream = nullptr;
  AVCodec *codec = nullptr;
  AVPacket *packet = nullptr;
  AVFrame *frame = nullptr;
  int ret = 0;

  // Allocate the format context for the output file
  avformat_alloc_output_context2(&fmt_ctx, nullptr, nullptr, filename.data());
  if (!fmt_ctx) {
    std::cerr << "Could not allocate output format context\n";
    return false;
  }

  // Find the MP3 encoder
  codec = avcodec_find_encoder(AV_CODEC_ID_MP3);
  if (!codec) {
    throw std::runtime_error("MP3 codec not found\n");
  }

  // Add a new stream to the format context
  stream = avformat_new_stream(fmt_ctx, codec);
  if (!stream) {
    throw std::runtime_error("Could not create new stream\n");
  }

  // Allocate the codec context and set parameters
  codec_ctx = avcodec_alloc_context3(codec);
  if (!codec_ctx) {
    throw std::runtime_error("Could not allocate codec context\n");
  }

  codec_ctx->bit_rate = bitrate;
  codec_ctx->sample_rate = sample_rate;
  codec_ctx->channel_layout = av_get_default_channel_layout(channels);
  codec_ctx->channels = channels;
  codec_ctx->sample_fmt =
      codec->sample_fmts[0]; // Usually AV_SAMPLE_FMT_FLTP for MP3

  stream->time_base = AVRational{1, sample_rate};

  // Open the codec
  if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
    throw std::runtime_error("Could not open codec\n");
  }

  // Copy the codec parameters to the stream
  avcodec_parameters_from_context(stream->codecpar, codec_ctx);

  // Open the output file
  if (!(fmt_ctx->flags & AVFMT_NOFILE)) {
    CHECK_AVERROR(avio_open(&fmt_ctx->pb, filename.data(), AVIO_FLAG_WRITE));
  }

  // Write the file header
  CHECK_AVERROR(avformat_write_header(fmt_ctx, nullptr));

  // Allocate packet and frame
  packet = av_packet_alloc();
  frame = av_frame_alloc();
  if (!packet || !frame) {
    throw std::runtime_error("Could not allocate packet or frame\n");
  }

  // Set up frame properties
  frame->nb_samples = codec_ctx->frame_size;
  frame->format = codec_ctx->sample_fmt;
  frame->channel_layout = codec_ctx->channel_layout;
  CHECK_AVERROR(av_frame_get_buffer(frame, 0));

  int offset = 0;
  while (offset < audioData.size()) {
    // Fill the frame with data from audioData
    int dataSize =
        std::min((int)audioData.size() - offset, frame->nb_samples * channels);
    memcpy(frame->data[0], audioData.data() + offset, dataSize);
    offset += dataSize;

    CHECK_AVERROR(avcodec_send_frame(codec_ctx, frame));

    while (avcodec_receive_packet(codec_ctx, packet) == 0) {
      av_interleaved_write_frame(fmt_ctx, packet);
      av_packet_unref(packet);
    }
  }

  // Flush the encoder
  CHECK_AVERROR(avcodec_send_frame(codec_ctx, nullptr));
  while (avcodec_receive_packet(codec_ctx, packet) == 0) {
    CHECK_AVERROR(av_interleaved_write_frame(fmt_ctx, packet));
    av_packet_unref(packet);
  }

  // Write the file trailer
  CHECK_AVERROR(av_write_trailer(fmt_ctx));

  // Cleanup
  av_frame_free(&frame);
  av_packet_free(&packet);
  avcodec_free_context(&codec_ctx);
  avio_closep(&fmt_ctx->pb);
  avformat_free_context(fmt_ctx);

  return true;
}

} // namespace gpudenoise

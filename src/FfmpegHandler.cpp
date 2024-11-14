#include "FfmpegHandler.hpp"
#include <MacroHelpers.hpp>

#include <algorithm>
#include <cassert>
#include <iostream>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/opt.h>
#include <libswresample/swresample.h>
}

constexpr int kB = 1024;

namespace gpufilter {
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

int FfmpegHandler::getChannels() const noexcept { return m_channels; }

int FfmpegHandler::getSampleFmt() const noexcept { return m_sample_fmt; }

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

  // Update class attributes
  m_channels = m_codec_cxt->channels;
  m_sample_fmt = m_codec_cxt->sample_fmt;

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
  //  m_audio_data.reserve(m_segment_size);
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

void saveAsMP3(uint8_t *audio_data, size_t data_size, int sample_rate,
               int channels, int bit_rate, int sample_fmt,
               const std::string_view &output_filename) {

  assert(audio_data != nullptr);

  AVFormatContext *format_context = nullptr;
  AVStream *audio_stream = nullptr;
  AVCodecContext *codec_context = nullptr;
  AVCodec *codec = nullptr;

  // Initialize FFMPEG libraries
  avformat_alloc_output_context2(&format_context, nullptr, nullptr,
                                 output_filename.data());
  if (!format_context) {
    throw std::runtime_error("Could not allocate output format context.");
  }

  codec = avcodec_find_encoder(AV_CODEC_ID_MP3);
  if (!codec) {
    throw std::runtime_error("MP3 encoder not found.");
  }

  // Create a new audio stream
  audio_stream = avformat_new_stream(format_context, nullptr);
  if (!audio_stream) {
    throw std::runtime_error("Could not create audio stream.");
  }

  codec_context = avcodec_alloc_context3(codec);
  if (!codec_context) {
    throw std::runtime_error("Could not allocate codec context.");
  }

  // Set codec parameters
  codec_context->sample_rate = sample_rate;
  codec_context->channel_layout = av_get_default_channel_layout(channels);
  codec_context->channels = channels;
  codec_context->bit_rate = bit_rate;
  codec_context->sample_fmt = static_cast<AVSampleFormat>(sample_fmt);

  if (format_context->oformat->flags & AVFMT_GLOBALHEADER) {
    codec_context->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
  }

  // Open codec
  if (avcodec_open2(codec_context, codec, nullptr) < 0) {
    throw std::runtime_error("Could not open codec.");
  }

  // Copy the codec context parameters to the stream
  if (avcodec_parameters_from_context(audio_stream->codecpar, codec_context) <
      0) {
    throw std::runtime_error("Could not copy codec parameters to stream.");
  }

  // Open output file
  if (!(format_context->oformat->flags & AVFMT_NOFILE)) {
    if (avio_open(&format_context->pb, output_filename.data(),
                  AVIO_FLAG_WRITE) < 0) {
      throw std::runtime_error("Could not open output file.");
    }
  }

  // Write the file header
  if (avformat_write_header(format_context, nullptr) < 0) {
    throw std::runtime_error("Error occurred when writing header to file.");
  }

  // Prepare packet and frame for writing data
  AVPacket *packet = av_packet_alloc();
  if (!packet) {
    throw std::runtime_error("Could not allocate packet.");
  }

  int frame_size = codec_context->frame_size;
  int bytes_per_sample = av_get_bytes_per_sample(codec_context->sample_fmt);

  // Write audio data in frames
  for (int64_t i = 0; i < data_size;
       i += frame_size * bytes_per_sample * channels) {
    AVFrame *frame = av_frame_alloc();
    if (!frame) {
      throw std::runtime_error("Could not allocate frame.");
    }
    frame->nb_samples = frame_size;
    frame->format = codec_context->sample_fmt;
    frame->channel_layout = codec_context->channel_layout;
    frame->sample_rate = codec_context->sample_rate;

    if (av_frame_get_buffer(frame, 0) < 0) {
      throw std::runtime_error("Could not allocate audio data buffers.");
    }

    // Copy data into the frame
    int64_t len = std::min(static_cast<int64_t>(frame->linesize[0]),
                           (int64_t)data_size - i);
    memcpy(frame->data[0], audio_data + i, len);

    assert(*frame->data[0] == audio_data[i]);
    assert(frame->nb_samples == codec_context->frame_size);

    // Encode the frame
    if (avcodec_send_frame(codec_context, frame) < 0) {
      throw std::runtime_error("Error sending frame to codec.");
    }

    // Receive and write encoded packets
    while (avcodec_receive_packet(codec_context, packet) == 0) {
      av_interleaved_write_frame(format_context, packet);
      av_packet_unref(packet); // Clean up packet after writing
    }

    av_frame_free(&frame);
  }

  // Flush the encoder
  avcodec_send_frame(codec_context, nullptr);
  while (avcodec_receive_packet(codec_context, packet) == 0) {
    av_interleaved_write_frame(format_context, packet);
    av_packet_unref(packet);
  }

  // Write file trailer and cleanup
  av_write_trailer(format_context);
  avcodec_free_context(&codec_context);
  avformat_free_context(format_context);
  if (!(format_context->oformat->flags & AVFMT_NOFILE)) {
    avio_closep(&format_context->pb);
  }
  av_packet_free(&packet); // Free the packet memory
}

} // namespace gpufilter

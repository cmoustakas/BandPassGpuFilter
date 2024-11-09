# Sound Denoiser with GPU Acceleration

This project is a high-performance sound denoiser that leverages GPU computing to process audio files efficiently. By utilizing the power of the GPU, this tool applies a low-pass filter to remove high-frequency noise, producing a cleaner audio output.

## Overview

The denoising pipeline for the demo is as follows:

1. **Audio Loading**: The raw audio data is loaded using the FFMPEG API, enabling support for various audio file formats.
2. **Data Upload to GPU**: The loaded audio data is uploaded to the GPU, where accelerated processing is performed.
3. **Fourier Transformation**: The `cuFFT` library computes the Fourier transform of the audio signal, moving it to the frequency domain.
4. **Low-Pass Filtering**: A low-pass filter is applied in the frequency domain, removing high-frequency noise components.
5. **Inverse Fourier Transformation**: The inverse Fourier transform is applied to convert the data back to the time domain.
6. **Data Download and Saving**: The processed data is transferred back to the CPU and saved as a new denoised audio file through FFMPEG.

## Requirements

- **CUDA** (version 10.2 or higher)
- **FFMPEG** (development libraries for `avformat`, `avutil`, `avcodec`, and `swscale`)
- **C++17** (or higher)

## Building the Project

1. **Clone the repository**:
    ```bash
    git clone https://github.com/cmoustakas/gpuDenoiser.git
    cd sound_denoiser
    ```

2. **Build using CMake**:
    ```bash
    mkdir build
    cd build
    cmake ..
    make
    ```

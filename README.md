# Sound Denoiser with GPU Acceleration

This project is a high-performance sound denoiser that leverages GPU computing to process audio files efficiently. By utilizing the power of the GPU, this tool applies a band-pass filter to remove high-frequency noise, producing a cleaner audio output.

## Overview

The denoising pipeline for the demo is as follows:

1. **Audio Loading**: The raw audio data (wav) is loaded using the JUCE API.
2. **Data Upload to GPU**: The loaded audio data is uploaded to the GPU, where accelerated processing is performed.
3. **Fourier Transformation**: The `cuFFT` library computes the Fourier transform of the audio signal, moving it to the frequency domain.
4. **Band-Pass Filtering**: A band-pass filter is applied in the frequency domain, removing high and low frequency noise components.
5. **Inverse Fourier Transformation**: The inverse Fourier transform is applied to convert the data back to the time domain.
6. **Data Download and Saving**: The processed data is transferred back to the CPU and saved as a new denoised audio file through JUCE.

![denoise_graph](https://github.com/user-attachments/assets/c2371799-d901-40b2-9285-98bd935be744)


## Requirements

- **CUDA** (version 10.2 or higher)
- **JUCE** (version 8.0.3)
- **C++17** (or higher)

## Building the Project

1. **Clone the repository**:
    ```bash
    git clone https://github.com/cmoustakas/gpuDenoiser.git
    cd gpuDenoiser
    ```

2. **Build using CMake**:
    ```bash
    mkdir build
    cd build
    cmake ..
    make
    ```

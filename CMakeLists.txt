cmake_minimum_required(VERSION 3.10)
project(sound_denoiser)

set(CMAKE_CXX_STANDARD 17)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14")

find_package(CUDA 10.2 REQUIRED)

# Find and link CURL for JUCE
find_package(CURL REQUIRED)
include_directories(${CURL_INCLUDE_DIRS})

# Specify the path to the JUCE directory
set(JUCE_DIR /home/harrismoustakas/JUCE)

# Add JUCE to the project
add_subdirectory(${JUCE_DIR} JUCE)

enable_testing()

# Add Google Test
add_subdirectory(gtests/external/googletest)
include_directories(gtests/external/googletest/googletest/include)

find_library(CUFFT_LIBRARY cufft PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
)

# Source files
set(SOUND_FILTER_SOURCES)
list(APPEND SOUND_FILTER_SOURCES
    src/FfmpegHandler.cpp
    include/FfmpegHandler.hpp
    src/CudaFiltering.cu
    include/CudaFiltering.hpp
    src/AudioIO.cpp
    include/AudioIO.hpp
    include/CudaMemory.hpp
    include/ErrChecker.hpp
)

list(APPEND FFMPEG_LIBS
    avformat
    avutil
    avcodec
    swscale
    swresample
)

# Add main library
cuda_add_library(${PROJECT_NAME} STATIC ${SOUND_FILTER_SOURCES} )

target_include_directories(
    ${PROJECT_NAME}
    PUBLIC
    include
    ${CUDA_INCLUDE_DIRS}

)

target_link_libraries(
    ${PROJECT_NAME}
    ${CUDA_LIBS}
    ${CUFFT_LIBRARY}
    ${FFMPEG_LIBS}
    juce::juce_core
    juce::juce_audio_basics
    juce::juce_audio_formats
    ${CURL_LIBRARIES}
)

add_executable(
    gpufiltering
    src/main.cpp
)

target_link_libraries(
    gpufiltering
    ${PROJECT_NAME}
)

# Testing:
# Google Test executable for unit tests
add_executable(sound_filter_tests gtests/unit_tests/sound_denoiser_tests.cpp)
target_link_libraries(sound_filter_tests PRIVATE ${PROJECT_NAME} gtest gtest_main)

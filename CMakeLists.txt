cmake_minimum_required(VERSION 3.10)
project(sound_denoiser)

set(CMAKE_CXX_STANDARD 17)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++14")

find_package(CUDA 10.2 REQUIRED)
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
set(SOUND_DENOISE_SOURCES)
list(APPEND SOUND_DENOISE_SOURCES
    src/FfmpegHandler.cpp
    include/FfmpegHandler.hpp
    src/CudaDenoise.cu
    include/CudaDenoise.hpp
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
cuda_add_library(${PROJECT_NAME} STATIC ${SOUND_DENOISE_SOURCES} )

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
)

add_executable(
    gpudenoiser
    src/main.cpp
)

target_link_libraries(
    gpudenoiser
    ${PROJECT_NAME}
)

# Testing:
# Google Test executable for unit tests
add_executable(sound_denoiser_tests gtests/unit_tests/sound_denoiser_tests.cpp)
target_link_libraries(sound_denoiser_tests PRIVATE ${PROJECT_NAME} gtest gtest_main)

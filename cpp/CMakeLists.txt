cmake_minimum_required(VERSION 3.16)
project(celer_infer)

set(CMAKE_CXX_STANDARD 14)

# ============================================================================
# Build options
# ============================================================================
option(ENABLE_SIMD "Enable AVX2 SIMD optimizations" ON)
option(ENABLE_CUDA "Enable CUDA GPU support" OFF)

# ============================================================================
# Include directories
# ============================================================================
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/ops)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party)

# Common compiler flags
set(COMMON_FLAGS "-Wall -Wextra")

# ============================================================================
# MiniMind Baseline (no optimizations, debug-friendly)
# ============================================================================
add_executable(minimind src/models/minimind.cpp)
set_target_properties(minimind PROPERTIES
    COMPILE_FLAGS "${COMMON_FLAGS} -O2 -g"
)

# ============================================================================
# MiniMind SIMD (AVX2 vectorized)
# ============================================================================
if(ENABLE_SIMD)
    add_executable(minimind_simd src/models/minimind_simd.cpp)
    set_target_properties(minimind_simd PROPERTIES
        COMPILE_FLAGS "${COMMON_FLAGS} -O3 -mavx2 -mfma -march=native"
    )
endif()

# ============================================================================
# MiniMind Unified (all backends: baseline/simd/fusion/quant)
# Single binary that supports runtime backend selection
# ============================================================================
add_executable(minimind_unified src/models/minimind_unified.cpp)
if(ENABLE_SIMD)
    set_target_properties(minimind_unified PROPERTIES
        COMPILE_FLAGS "${COMMON_FLAGS} -O3 -mavx2 -mfma -march=native"
    )
else()
    set_target_properties(minimind_unified PROPERTIES
        COMPILE_FLAGS "${COMMON_FLAGS} -O2"
    )
endif()

# ============================================================================
# CUDA support (optional)
# ============================================================================
if(ENABLE_CUDA)
    # Set CUDA compiler path if not in PATH
    if(NOT DEFINED CMAKE_CUDA_COMPILER)
        set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
    endif()
    
    # CUDA path
    set(CUDA_PATH "/usr/local/cuda")
    
    enable_language(CUDA)
    
    # MiniMind CUDA executable (standalone .cu file)
    add_executable(minimind_cuda 
        src/models/minimind_cuda.cu
    )
    target_include_directories(minimind_cuda PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party
        ${CUDA_PATH}/include
    )
    target_link_directories(minimind_cuda PRIVATE
        ${CUDA_PATH}/lib64
    )
    target_link_libraries(minimind_cuda PRIVATE cudart cublas)
    set_target_properties(minimind_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_ARCHITECTURES "89"
    )
endif()

# ============================================================================
# Summary message
# ============================================================================
message(STATUS "CelerInfer build configuration:")
message(STATUS "  SIMD (AVX2): ${ENABLE_SIMD}")
message(STATUS "  CUDA: ${ENABLE_CUDA}")
message(STATUS "  Targets: minimind, minimind_unified" )
if(ENABLE_SIMD)
    message(STATUS "           minimind_simd")
endif()
if(ENABLE_CUDA)
    message(STATUS "           minimind_cuda")
endif()

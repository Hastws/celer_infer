cmake_minimum_required(VERSION 3.16)
project(celer_infer)

set(CMAKE_CXX_STANDARD 17)

# ============================================================================
# Build options
# ============================================================================
option(ENABLE_SIMD "Enable AVX2 SIMD optimizations" ON)
option(ENABLE_CUDA "Enable CUDA GPU support" OFF)
option(ENABLE_OPENMP "Enable OpenMP parallelization" ON)

# ============================================================================
# Find OpenMP
# ============================================================================
if(ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
    endif()
endif()

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
# MiniMind Extreme - All optimizations: OpenMP + AVX2/VNNI + Tiling + Prefetch
# ============================================================================
add_executable(minimind_extreme src/models/minimind_extreme.cpp)
set_target_properties(minimind_extreme PROPERTIES
    COMPILE_FLAGS "${COMMON_FLAGS} -O3 -mavx2 -mfma -march=native -funroll-loops -ftree-vectorize"
)
if(ENABLE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(minimind_extreme PRIVATE OpenMP::OpenMP_CXX)
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
    
    # MiniMind CUDA FP32 executable
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
    
    # MiniMind CUDA FP16 + Tensor Core executable
    add_executable(minimind_cuda_fp16 
        src/models/minimind_cuda_fp16.cu
    )
    target_include_directories(minimind_cuda_fp16 PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party
        ${CUDA_PATH}/include
    )
    target_link_directories(minimind_cuda_fp16 PRIVATE
        ${CUDA_PATH}/lib64
    )
    target_link_libraries(minimind_cuda_fp16 PRIVATE cudart cublas)
    set_target_properties(minimind_cuda_fp16 PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_ARCHITECTURES "89"
    )
endif()

# ============================================================================
# DiT (Diffusion Transformer) - Autonomous Driving Planning
# ============================================================================
add_executable(dit src/models/dit.cpp)
set_target_properties(dit PROPERTIES
    COMPILE_FLAGS "${COMMON_FLAGS} -O2 -g"
)

# DiT SIMD (AVX2 vectorized)
if(ENABLE_SIMD)
    add_executable(dit_simd src/models/dit_simd.cpp)
    set_target_properties(dit_simd PROPERTIES
        COMPILE_FLAGS "${COMMON_FLAGS} -O3 -mavx2 -mfma -march=native"
    )
endif()

# DiT Extreme - OpenMP + AVX2 + Tiling
add_executable(dit_extreme src/models/dit_extreme.cpp)
set_target_properties(dit_extreme PROPERTIES
    COMPILE_FLAGS "${COMMON_FLAGS} -O3 -mavx2 -mfma -march=native -funroll-loops -ftree-vectorize"
)
if(ENABLE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(dit_extreme PRIVATE OpenMP::OpenMP_CXX)
endif()

# DiT CUDA
if(ENABLE_CUDA)
    add_executable(dit_cuda src/models/dit_cuda.cu)
    target_include_directories(dit_cuda PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party
        ${CUDA_PATH}/include
    )
    target_link_directories(dit_cuda PRIVATE ${CUDA_PATH}/lib64)
    target_link_libraries(dit_cuda PRIVATE cudart cublas)
    set_target_properties(dit_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_ARCHITECTURES "89"
    )
    
    add_executable(dit_cuda_fp16 src/models/dit_cuda_fp16.cu)
    target_include_directories(dit_cuda_fp16 PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party
        ${CUDA_PATH}/include
    )
    target_link_directories(dit_cuda_fp16 PRIVATE ${CUDA_PATH}/lib64)
    target_link_libraries(dit_cuda_fp16 PRIVATE cudart cublas)
    set_target_properties(dit_cuda_fp16 PROPERTIES
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
message(STATUS "  OpenMP: ${ENABLE_OPENMP}")
message(STATUS "  MiniMind targets: minimind, minimind_unified, minimind_extreme" )
message(STATUS "  DiT targets: dit, dit_extreme")
if(ENABLE_SIMD)
    message(STATUS "           minimind_simd, dit_simd")
endif()
if(ENABLE_CUDA)
    message(STATUS "           minimind_cuda, minimind_cuda_fp16")
    message(STATUS "           dit_cuda, dit_cuda_fp16")
endif()

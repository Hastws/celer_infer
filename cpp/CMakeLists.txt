cmake_minimum_required(VERSION 3.16)
project(celer_infer)

set(CMAKE_CXX_STANDARD 14)

# Option to enable/disable SIMD (AVX2)
option(ENABLE_SIMD "Enable AVX2 SIMD optimizations" ON)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/ops)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party)

# Common flags
set(COMMON_FLAGS "-Wall -Wextra")

# MiniMind baseline executable (no SIMD, debug-friendly)
add_executable(minimind src/models/minimind.cpp)
set_target_properties(minimind PROPERTIES
    COMPILE_FLAGS "${COMMON_FLAGS} -O2 -g"
)

# MiniMind SIMD-optimized executable (AVX2)
if(ENABLE_SIMD)
    add_executable(minimind_simd src/models/minimind_simd.cpp)
    set_target_properties(minimind_simd PROPERTIES
        COMPILE_FLAGS "${COMMON_FLAGS} -O3 -mavx2 -mfma -march=native"
    )
endif()

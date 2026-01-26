cmake_minimum_required(VERSION 3.16)
project(celer_infer)

set(CMAKE_CXX_STANDARD 14)

# Option to enable/disable SIMD (AVX2)
option(ENABLE_SIMD "Enable AVX2 SIMD optimizations" ON)

# Common flags
set(COMMON_FLAGS "-Wall -Wextra")

# Baseline executable (no SIMD, debug-friendly)
add_executable(base_line_micro base_line_micro.cpp)
set_target_properties(base_line_micro PROPERTIES
    COMPILE_FLAGS "${COMMON_FLAGS} -O2 -g"
)

# SIMD-optimized executable (AVX2)
if(ENABLE_SIMD)
    add_executable(base_line_simd base_line_simd.cpp)
    set_target_properties(base_line_simd PROPERTIES
        COMPILE_FLAGS "${COMMON_FLAGS} -O3 -mavx2 -mfma -march=native"
    )
endif()

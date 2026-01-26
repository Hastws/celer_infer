cmake_minimum_required(VERSION 3.16)
project(base_line_micro)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -g")

add_executable(base_line_micro base_line_micro.cpp)

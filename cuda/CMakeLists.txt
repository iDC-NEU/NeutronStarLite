project(cuda_propagate)
cmake_minimum_required(VERSION 2.8)
find_package(CUDA REQUIRED)
 
file(GLOB_RECURSE CURRENT_HEADERS &.hpp *.cuh *.h)
file(GLOB CURRENT_SOURCES *.cu *.cpp)

cuda_select_nvcc_arch_flags(ARCH_FLAGS)
set(CUDA_NVCC_FLAGS "${ARCH_FLAGS} -std=c++14 -O3 -G -g")
message(STATUS "CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}" )

source_group("Include" FILES ${CURRENT_HEADERS})
source_group("Source" FILES ${CURRENT_SOURCES})
 
cuda_add_library(cuda_propagate STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})
 
#not used in this demo

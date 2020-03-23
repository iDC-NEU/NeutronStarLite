cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(GNNmini)
set(CMAKE_PREFIX_PATH "/home/wangqg/zg/GNNmini/GPU_libtorch/libtorch/")
set(ENV{LANG} "C")
find_package(MPI REQUIRED)
find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)

#exec_program("mpic++ -showme:compile" OUTPUT_VARIABLE MPI_COMPILE_FLAGS)
#exec_program("mpic++ -showme:incdirs" OUTPUT_VARIABLE MPI_INCDIRS)
#exec_program("mpic++ -showme:link" OUTPUT_VARIABLE MPI_LINK_FLAGS)
#exec_program("mpic++ -showme:libdirs" OUTPUT_VARIABLE MPI_LIBDIRS)
#exec_program("mpic++ -showme:libs" OUTPUT_VARIABLE MPI_LIBS)

#string(REPLACE " " ";" MPI_INCDIRS "${MPI_INCDIRS}")
#string(REPLACE " " ";" MPI_LINK_FLAGS "${MPI_LINK_FLAGS}")
#string(REPLACE " " ";" MPI_LIBDIRS "${MPI_LIBDIRS}")
#string(REPLACE " " ";" MPI_LIBS "${MPI_LIBS}")


add_subdirectory(cuda)
set(EXTRA_LIBS ${EXTRA_LIBS} cuda_propagate)

include_directories(
${CMAKE_CURRENT_SOURCE_DIR}
/home/wangqg/zg/GNNmini/core
/home/wangqg/zg/GNNmini/comm
/home/wangqg/zg/GNNmini/cuda
${MPI_INCLUDE_PATH}
)
set(CMAKE_CXX_FLAGS "-O3 -Wall -std=c++14 -g -fopenmp -march=native")
add_subdirectory(core)

#add_executable(myfistapp myfistapp.cpp)
add_executable(myfistapp toolkits/pagerank.cpp)
target_link_libraries(myfistapp  ${TORCH_LIBRARIES} ${MPI_LIBRARIES} ${EXTRA_LIBS} core numa) # 加入libTorch的库文件路径



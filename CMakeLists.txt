cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
add_definitions(-w)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
# please change this to the path where you put libtorch
# TODO: torch path should not hard coded in cmake file
list(APPEND CMAKE_PREFIX_PATH "./GPU_libtorch/libtorch-1-9/libtorch/")

project(GNNmini)
set(ENV{LANG} "C")

find_package(PythonInterp REQUIRED)
find_package(MPI REQUIRED)
# because torch also depends on cuda, so we find cuda first
find_package(CUDA REQUIRED)
find_package(Torch REQUIRED)
#exec_program("mpic++ -showme:compile" OUTPUT_VARIABLE MPI_COMPILE_FLAGS)
#exec_program("mpic++ -showme:incdirs" OUTPUT_VARIABLE MPI_INCDIRS)
#exec_program("mpic++ -showme:link" OUTPUT_VARIABLE MPI_LINK_FLAGS)
#exec_program("mpic++ -showme:libdirs" OUTPUT_VARIABLE MPI_LIBDIRS)
#exec_program("mpic++ -showme:libs" OUTPUT_VARIABLE MPI_LIBS)

#string(REPLACE " " ";" MPI_INCDIRS "${MPI_INCDIRS}")
#string(REPLACE " " ";" MPI_LINK_FLAGS "${MPI_LINK_FLAGS}")
#string(REPLACE " " ";" MPI_LIBDIRS "${MPI_LIBDIRS}")
#string(REPLACE " " ";" MPI_LIBS "${MPI_LIBS}")

# Expected directory structure.
set(NEUTRONSTAR_BUILD_SUPPORT_DIR "${CMAKE_SOURCE_DIR}/build_support")
set(NEUTRONSTAR_CLANG_SEARCH_PATH "/usr/local/bin" "/usr/bin" "/usr/local/opt/llvm/bin" "/usr/local/opt/llvm@8/bin"
        "/usr/local/Cellar/llvm/8.0.1/bin")

# clang-format
find_program(CLANG_FORMAT_BIN
        NAMES clang-format clang-format-8
        HINTS ${NEUTRONSTAR_CLANG_SEARCH_PATH})
if ("${CLANG_FORMAT_BIN}" STREQUAL "CLANG_FORMAT_BIN-NOTFOUND")
    message(WARNING " NeutronStar/main couldn't find clang-format.")
else()
    message(STATUS " NeutronStar/main found clang-format at ${CLANG_FORMAT_BIN}")
endif()

# set lint files
# TODO: add lint
file(GLOB_RECURSE NEUTRONSTAR_LINT_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h,"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp,"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp,"
        "${CMAKE_CURRENT_SOURCE_DIR}/core/*.h,"
        "${CMAKE_CURRENT_SOURCE_DIR}/core/*.hpp,"
        "${CMAKE_CURRENT_SOURCE_DIR}/core/*.cpp,"
        "${CMAKE_CURRENT_SOURCE_DIR}/toolkits/*.h,"
        "${CMAKE_CURRENT_SOURCE_DIR}/toolkits/*.hpp,"
        "${CMAKE_CURRENT_SOURCE_DIR}/toolkits/*.cpp,"
        "${CMAKE_CURRENT_SOURCE_DIR}/cuda/*.h,"
        "${CMAKE_CURRENT_SOURCE_DIR}/cuda/*.hpp,"
        "${CMAKE_CURRENT_SOURCE_DIR}/cuda/*.cpp,"
        )

# set format dirs
string(CONCAT NEUTRONSTAR_FORMAT_DIRS
        "${CMAKE_CURRENT_SOURCE_DIR}/src,"
        "${CMAKE_CURRENT_SOURCE_DIR}/core,"
        "${CMAKE_CURRENT_SOURCE_DIR}/comm,"
        "${CMAKE_CURRENT_SOURCE_DIR}/cuda,"
        "${CMAKE_CURRENT_SOURCE_DIR}/toolkits,"
        )

# runs clang format and updates files in place.
add_custom_target(format ${NEUTRONSTAR_BUILD_SUPPORT_DIR}/run_clang_format.py
        ${CLANG_FORMAT_BIN}
        ${NEUTRONSTAR_BUILD_SUPPORT_DIR}/clang_format_exclusions.txt
        --source_dirs
        ${NEUTRONSTAR_FORMAT_DIRS}
        --fix
        # --quiet
        )


add_subdirectory(cuda)
set(EXTRA_LIBS ${EXTRA_LIBS} cuda_propagate)

include_directories(
${CMAKE_CURRENT_SOURCE_DIR}
./core
./comm
./cuda
${MPI_INCLUDE_PATH}
)
set(CMAKE_CXX_FLAGS "-O3 -Wall -std=c++14 -g -fopenmp -march=native")
#add_subdirectory(core)
add_executable(nts toolkits/main.cpp)
target_link_libraries(nts  ${TORCH_LIBRARIES} ${MPI_LIBRARIES} ${EXTRA_LIBS} ${CUDA_LIBRARIES} cuda_propagate numa) # 加入libTorch的库文件路径

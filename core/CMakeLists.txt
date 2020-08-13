set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(HEADERS atomic.hpp 
            bitmap.hpp 
            constants.hpp
            filesystem.hpp
            graph.hpp
            math.hpp
            mpi.hpp
            operator.hpp
            time.hpp
            input.hpp
            )
set(SOURCES ${HEADERS} )
add_library(core ${SOURCES})
set_property(TARGET core PROPERTY LINKER_LANGUAGE CXX)

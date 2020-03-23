add_executable(gnn test.cpp)
target_link_libraries(gnn core numa)
set_property(TARGET gnn PROPERTY CXX_STANDARD 11)


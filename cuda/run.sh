nvcc -gencode arch=compute_61,code=sm_61 -rdc=true -lcudadevrt -std=c++14  -O3 -o test_propagate main.cpp test_propagate.cu&&./test_propagate 



## Quick Start
Gemini uses **MPI** for inter-process communication and **libnuma** for NUMA-aware memory allocation.
A compiler supporting **OpenMP** and **C++11** features (e.g. lambda expressions, multi-threading, etc.) is required.

GPU based libtorch  is required




To build:
```
1.  Install all package the Gemini REQUIRED.


2. Download the *libtorch*  with cuda support. 

3. modify the *include_dictionary*  in CMakefiles.txt according to your configuration.

4.change the *CMAKE_PREFIX_PATH* to the root dir to your *libtorch*

5.  cmake .

make
```

make

run
``` 

./run.sh $vertices  $epoch   $engine  

1. $engine: TEST is the single GPU engine.

2. $engine: GPU is a distributed engine

avaliable example:

./run.sh 2208  200  TEST 
run cora dataset with 2208 training point and 500 evaluate point. run 200 epoch with pure GPU engine.
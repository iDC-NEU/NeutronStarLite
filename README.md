This Repo contains part of the codes of NeutronStar, A Distributed GNN training system.

## Software dependency
NeutronStar uses **MPI** for inter-process communication.

A compiler supporting **OpenMP** and **C++11** features (e.g. lambda expressions, multi-threading, etc.) is required.

LibTorch 1.5 with GPU support is required.


To build:
```
1.  Install all package the NeutronStar Required.

2. Download the *libtorch*  with cuda support. 

3. modify the *include_dictionary*  in CMakefiles.txt according to your configuration.

4.change the *CMAKE_PREFIX_PATH* to the root dir to your *libtorch*

5.  cmake .

6.make
```

To run:

run example:



``` 

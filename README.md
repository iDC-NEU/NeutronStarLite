

## Requirement


A compiler supporting **OpenMP** and **C++11** features (e.g. lambda expressions, multi-threading, etc.) is required.

**MPI** for inter-process communication 

**cuda** > 9.0 for GPU based graph operation.

**libnuma** for NUMA-aware memory allocation.

```
sudo apt install libnuma-dev"
```

**libtorch** version > 1.7 with gpu support for nn computation

unzip the **libtorch** package in the root dir of **NeutronStar** and change CMAKE_PREFIX_PATH in "CMakeList.txt"to your own path


configure PATH and LD_LIBRARY_PATH for **cuda** and **mpi**
```
export CUDA_HOME=/usr/local/cuda-10.2
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH
```

To build:
```
mkdir build

cd build

cmake ..

make -j4
```


To run:

List all nodes in ./NeutronStar/hostfile for MPI communication
```
ip1
ip2
ip3
```
copy NeutronStar to all your machines

single-machine multi-slots:(strongly recommand use one slot, except for debugging)
```
./run_nts.sh #nodes_number #configure_file
./run_nts.sh 1 gcn_cora.cfg
```
distributed:

```
./run_nts_dist.sh #nodes_number #configure_file
./run_nts_dist.sh 2 gcn_cora.cfg
```

We list serveral example in the root dir for your reference
GCN:
gcn_cora.cfg
gcn_pubmed.cfg
gcn_citeseer.cfg
gcn_reddit.cfg
gcn_reddit_full.cfg




## Requirement



 **MPI** for inter-process communication 

 **libnuma** for NUMA-aware memory allocation.
"sudo apt install libnuma-dev"

A compiler supporting **OpenMP** and **C++11** features (e.g. lambda expressions, multi-threading, etc.) is required.

**libtorch** version > 1.7 with gpu support for nn computation

**cuda** > 9.0 for GPU based graph operation.




To build:
```

configure PATH and LD_LIBRARY_PATH for **cuda** and **mpi**

unzip the **libtorch** package in the root dir of **NeutronStar** and change CMAKE_PREFIX_PATH to your own version  


"cmake ."

"make -j4"
```


To run:
List all nodes in ./NeutronStar/hostfile

copy NeutronStar to all your nodes


run any program with the following command:


single-machine multi-slots:(strongly recommand use one slot, except for debugging)

./run_nts.sh #nodes_number #configure_file

distributed:
./run_nts_dist.sh #nodes_number #configure_file

We list serveral example in the root dir for your reference


GCN:
gcn_cora.cfg
gcn_pubmed.cfg
gcn_citeseer.cfg
gcn_reddit.cfg
gcn_reddit_full.cfg

Example:
./run_nts.sh 1 gcn_cora.cfg



Install CUDA MPI and NVCC:


Install Libnuma-dev:

$sudo apt install libnuma-dev

Configure the Env Variable:

$vim ~/.bashrc
and write the following lines to the tail

export CUDA_HOME=/usr/local/cuda-10.2
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

$source ~/.bashrc


Install NeutronStar:

$cd NeutronStar
$cmake .
$make -j4





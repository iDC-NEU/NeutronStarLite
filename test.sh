#i!/bin/bash
make -j4

mpiexec -np 2 ./nts./edge 875713 10 GCN 32-32-16 0 overlap
#mpiexec -hostfile hostfile -np $1 ./myfistapp ./edge 875713 $2 $3 $4 $5 $6
#mpiexec -hostfile hostfile -np $1 ./myfistapp ../LargeData/pokec_edge_binary 1632803 $2 $3 $4 $5 $6
#mpiexec -hostfile hostfile -np $1 ./myfistapp ./skitter_edge_binary 1696415 $2 $3 $4 $5 $6
#mpiexec -hostfile hostfile -np $1 ./myfistapp ../LargeData/live_edge_binary 4847572 $2 $3 $4 $5 $6
#mpiexec -hostfile hostfile -np $1 ./myfistapp ../LargeData/wiki_edge_binary 12150976 $2 $3 $4 $5 $6
#mpiexec -hostfile hostfile -np $1 ./myfistapp ../LargeData/orkut_edge_binary 3072626 $2 $3 $4 $5 $6
#mpiexec -hostfile hostfile -np $1 ./myfistapp ../LargeData/reddit_edge_binary 232966 $2 $3 $4 $5 $6



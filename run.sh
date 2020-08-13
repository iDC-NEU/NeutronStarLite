#i!/bin/bash
make

#mpiexec -np 1 ./myfistapp ./edge_live $1 $2 $3
#mpiexec -np 1 ./myfistapp ./edge $1 $2 $3
#mpiexec -np 1 ./myfistapp ./edge $1 $2 $3
#$1  training vertices      epoch      engine
mpiexec -np 1 ./myfistapp ./data/cora.2708.edge.self_write2208 $1 $2 $3
#mpiexec -np 1 ./myfistapp ./pubmed_data/pubmed.19717.edge_write15717 $1 $2 $3
#mpiexec -np 1 ./myfistapp ./citeseer_data/citeseer.3327.edge_write2627 $1 $2 $3


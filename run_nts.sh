#i!/bin/bash
#mpiexec -np $1 ./nts $2
mpiexec -np 1 ./nts cora_config.cfg




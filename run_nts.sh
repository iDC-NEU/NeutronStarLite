#i!/bin/bash
mpiexec -np $1 ./build/nts $2
#mpiexec -np 1 ./nts NTS_cora_data.cfg




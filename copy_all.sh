#i!/bin/bash

cur_dir=$(cd "$(dirname "$0")"; pwd)
cat hostfile| while read line; do
ssh ${USER}@${WORKERS[$i]} "mkdir -p ${cur_dir}"
#echo ${line}
scp -r ./*  ${USER}@${line}:${cur_dir}/
done




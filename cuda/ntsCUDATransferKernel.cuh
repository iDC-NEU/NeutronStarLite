/*
 * propagate.h
 *
 *  Created on: Dec 3, 2019
 *      Author: wangqg
 * TODO :cub support and shared memory optimization
 */

#ifndef NTSCUDATRANSFERKERNEL_CUH
#define NTSCUDATRANSFERKERNEL_CUH

#include"cuda_type.h"
#include<stdlib.h>
#include<stdio.h>
#include<cstdio>
#include<assert.h>
#include <sys/time.h>
#include<cuda.h>
#include"cub/cub.cuh"
#include"math.h"
inline double get_time() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + (tv.tv_usec / 1e6);
  }



//util for fused graph op
__global__ void aggregate_data_buffer(float *result_buffer,float *comm_buffer,
 		size_t data_size,int feature_size,int partition_offset,bool debug=false){
			
	size_t threadId = blockIdx.x *blockDim.x + threadIdx.x;
     size_t record_size=feature_size+1;//with key attached;
	for(long i=threadId;i<(long)feature_size*data_size;i+=blockDim.x*gridDim.x){
            size_t rank=i%(feature_size);
            unsigned int key=i/(feature_size);
            unsigned int *v_id=NULL;
            unsigned int id=0;
            v_id=(unsigned int*)(comm_buffer+(key*record_size));
			atomicAdd(&comm_buffer[key*record_size+rank+1],comm_buffer[key*record_size+rank+1]);
			//atomicAdd(&result_buffer[feature_size*((*v_id)-partition_offset)+rank],result_buffer[feature_size*((*v_id)-partition_offset)+rank]);
			//atomicAdd(&result_buffer[feature_size*((*v_id)-partition_offset)+rank],comm_buffer[key*record_size+rank+1]);
		
	}
	if(threadId==0)printf("partition_offset %d\n",partition_offset);
}

__global__ void aggregate_data_buffer_debug(float *result_buffer,float *comm_buffer,
	size_t data_size,size_t feature_size,size_t partition_start,size_t partition_end,bool debug=false){
	   
size_t threadId = blockIdx.x *blockDim.x + threadIdx.x;
size_t record_size=feature_size+1;//with key attached;
for(long i=threadId;i<(long)feature_size*data_size;i+=blockDim.x*gridDim.x){
	   size_t rank=i%(feature_size);
	   long  key=i/(feature_size);
	   unsigned int *v_id=NULL;
	   v_id=(unsigned int*)(comm_buffer+(key*record_size));
           
	   //if((partition_start>(*v_id)||partition_end<=(*v_id))&&i==0)
	   //printf("something wrong %d\n",(*v_id));
	  // atomicAdd(&comm_buffer[key*record_size+rank+1],comm_buffer[key*record_size+rank+1]);
	   //atomicAdd(&result_buffer[feature_size*((*v_id)-partition_start)+rank],result_buffer[feature_size*((*v_id)-partition_offset)+rank]);
	   atomicAdd(&result_buffer[feature_size*((*v_id)-partition_start)+rank],comm_buffer[key*record_size+rank+1]);
   
}
//if(threadId==0)printf("partition_start %d partition_end %d\n",partition_start,partition_end);
}

__global__ void deSerializeToGPUkernel(float *input_gpu_buffer,float *comm_buffer,
	size_t data_size,size_t feature_size,size_t partition_start,size_t partition_end,bool debug=false){
	   
size_t threadId = blockIdx.x *blockDim.x + threadIdx.x;
size_t record_size=feature_size+1;//with key attached;
for(long i=threadId;i<(long)feature_size*data_size;i+=blockDim.x*gridDim.x){
	   size_t rank=i%(feature_size);
	   long  key=i/(feature_size);
	   unsigned int *v_id=NULL;
	   v_id=(unsigned int*)(comm_buffer+(key*record_size));
	   //if((partition_start>(*v_id)||partition_end<=(*v_id))&&rank==0)
	   //printf("something wrong1 %d\n",(*v_id));
	   //atomicAdd(&comm_buffer[key*record_size+rank+1],comm_buffer[key*record_size+rank+1]);
	   //atomicAdd(&input_gpu_buffer[feature_size*((*v_id)%10-partition_start)+rank],input_gpu_buffer[feature_size*((*v_id)%10-partition_start)+rank]);
           if((partition_start<=(*v_id)&&partition_end>(*v_id))){
              // if((*v_id)==875712&&rank==0)printf("data %d %f %d,\n",(*v_id), comm_buffer[key*record_size+rank+1],partition_end);
           //atomicAdd(&input_gpu_buffer[feature_size*((*v_id)-partition_start)+rank],input_gpu_buffer[feature_size*((*v_id)-partition_start)+rank]);
               input_gpu_buffer[feature_size*((*v_id)-partition_start)+rank]=comm_buffer[key*record_size+rank+1];
           }
	   //atomicAdd(&input_gpu_buffer[feature_size*((*v_id)-partition_start)+rank],comm_buffer[key*record_size+rank+1]);
   
}
//if(threadId==0)printf("partition_start %d partition_end %d\n",partition_start,partition_end);
}


















#endif /* PROPAGATE_H_ */

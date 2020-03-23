/*
 * propagate.h
 *
 *  Created on: Dec 3, 2019
 *      Author: wangqg
 */

#ifndef PROPAGATE_H_
#define PROPAGATE_H_


#include<stdlib.h>
#include<stdio.h>
#include<cstdio>

template <typename T_v,typename T_l>
class graphOnGPUBlock{
public:
	graphOnGPUBlock(int batch_size,int edge_size, int feature_size){
	cudaMalloc(&src,sizeof(T_l)*(edge_size+1));
	cudaMalloc(&dst,sizeof(T_l)*(batch_size+1));
	cudaMalloc(&edge_data,sizeof(T_v)*edge_size*feature_size);

	cudaMalloc(&metaInfo,sizeof(int)*8);

	cudaMalloc(&old_feature,sizeof(T_v)*feature_size*batch_size);
	cudaMalloc(&new_feature,sizeof(T_v)*feature_size*batch_size);

	meta=new int [8];
	}
	void setMetaInfo(
			int src_s,int src_e,
			int dst_s,int dst_e,
			int graph_size,int partition_s,
			int partition_e,int feature_size){
		meta[0]=src_s;meta[1]=src_e;
		meta[2]=dst_s;meta[3]=dst_e;
		meta[4]=graph_size;meta[5]=partition_s;
		meta[6]=partition_e;meta[7]=feature_size;
		cudaMemcpy(metaInfo,meta,sizeof(int)*8, cudaMemcpyHostToDevice);

	}
 int *src;	// neighboors  inside a batch
 int *dst;  //neighboor index  inside a batch
 int *metaInfo;// contatins all meta information . stores on GPU
 int * meta;// contatins all meta information . stores on CPU
 T_v* old_feature;
 T_v* new_feature;
 T_v* edge_data;
/*
 int source_s;
 int source_e;
 int dst_s;
 int dst_e;
 int  graph_size;
 int partition_start;
 int partition_from;
 int featuresize
*/

 __host__ int src_s(){
 	return meta[0];
 }
 __host__ int src_e(){
 	return meta[1];
 }
 __host__ int dst_s(){
 	return meta[2];
 }
 __host__ int dst_e(){
 	return meta[3];
 }
 __host__ int graph_size(){
 	return meta[4];
 }
 __host__ int partition_s(){
 	return meta[5];
 }
 __host__ int partition_e(){
 	return meta[6];
 }
 __host__ int feature_size(){
 	return meta[7];
 }
 __host__ int batch_size(){
 	return dst_e()-dst_s();
 }
};



__device__ int src_s(int* metaInfo){
	return metaInfo[0];
}
__device__ int src_e(int* metaInfo){
	return metaInfo[1];
}
__device__ int dst_s(int* metaInfo){
	return metaInfo[2];
}
__device__ int dst_e(int* metaInfo){
	return metaInfo[3];
}
__device__ int graph_size(int* metaInfo){
	return metaInfo[4];
}
__device__ int partition_s(int* metaInfo){
	return metaInfo[5];
}
__device__ int partition_e(int* metaInfo){
	return metaInfo[6];
}
__device__ int feature_size(int* metaInfo){
	return metaInfo[7];
}
__device__ int batch_size(int * metaInfo){
	return dst_e(metaInfo)-dst_s(metaInfo);
}
__host__ int getThreadNum(int num){
int s=32;
while(s<num){
	s+=32;
}
return s;
}


template <typename T_l>
__global__  void show2(T_l *src, T_l *dst,int *meta, float* old_feature, float* new_feature,float* edge_data){
	printf("hello");
	for(int i=0;i<batch_size(meta)+1;i++){
		printf("%d\t",dst[i]);
	}printf("\n");
	for(int i=0;i<graph_size(meta);i++){
		printf("%d\t",src[i]);
	}printf("\n");

	for(int i=0;i<feature_size(meta)*batch_size(meta);i++){
		printf("%f\t",old_feature[i]);
	}printf("\n");
	for(int i=0;i<feature_size(meta)*graph_size(meta);i++){
		printf("%f\t",edge_data[i]);
	}printf("\n");
printf("src_s: %d\n",meta[0]);
printf("src_e: %d\n",meta[1]);
printf("dst_s: %d\n",meta[2]);
printf("dst_e: %d\n",meta[3]);
printf("graphsize: %d\n",meta[4]);
printf("partition_s: %d\n",meta[5]);
printf("partition_e: %d\n",meta[6]);
printf("featuresize: %d\n",meta[7]);
printf("show2 finished\n");
}

template <typename T_l>
__global__  void show(T_l *src, T_l *dst,int *meta, float* old_feature, float* new_feature,float* edge_data){

	for(int i=0;i<feature_size(meta)*batch_size(meta);i++){
		printf("%f\t",new_feature[i]);
	}
printf("\n");
}




template <typename T_v,typename T_l>
__global__ void processing_tensor_weight(T_l *src, T_l *dst, T_v* old_feature, T_v* new_feature,T_v* edge_data,int batch_size_,int src_s_,int feature_size_){
	__shared__ int metaInfo[8];

	int large_size=blockDim.x;
	int threadId = blockIdx.x *blockDim.x + threadIdx.x;
	int rank=threadIdx.x;
	// for(int i=threadId;i<large_size*batch_size_;i+=blockDim.x*gridDim.x){
	// 	if(rank<feature_size_){
	// 	T_l dstid=i/blockDim.x;
	// 	T_l local_dst=dstid;
	// 	for(int i_i=dst[dstid];i_i<dst[dstid+1];i_i++){
	// 		int local_src=src[i_i]-src_s_;
	// 		new_feature[feature_size_*local_dst+rank]+=
	// 			old_feature[feature_size_*local_src+rank]
	// 			    *edge_data[feature_size_*i_i+rank];
	// 		atomicAdd(&new_feature[feature_size_*local_dst+rank],
	// 			old_feature[feature_size_*local_src+rank]*edge_data[feature_size_*i_i+rank]);
	// 	}
	// 	}
	// }
	for(int i=threadId;i<feature_size_*batch_size_;i+=blockDim.x*gridDim.x){
		T_l local_dst=i/feature_size_;
		T_l rank=i%feature_size_;
	 	for(int i_i=dst[local_dst];i_i<dst[local_dst+1];i_i++){
			int local_src=src[i_i];
			atomicAdd(&new_feature[feature_size_*local_dst+rank],
				old_feature[feature_size_*local_src+rank]*edge_data[feature_size_*local_src+rank]);
	 	}
		
	}

}

template <typename T_v,typename T_l>
__global__ void processing_scalar_weight(T_l *src, T_l *dst, T_v* old_feature, T_v* new_feature,T_v* edge_data,int batch_size_,int src_s_,int feature_size_){

	int large_size=blockDim.x;
	int threadId = blockIdx.x *blockDim.x + threadIdx.x;
	// for(int i=threadId;i<large_size*batch_size_;i+=blockDim.x*gridDim.x){
	// 	if(rank<feature_size_){
	// 	T_l dstid=i/blockDim.x;
	// 	T_l local_dst=dstid;
	// 	for(int i_i=dst[dstid];i_i<dst[dstid+1];i_i++){
	// 		int local_src=src[i_i]-src_s_;
	// 		atomicAdd(&new_feature[feature_size_*local_dst+rank],
	// 			old_feature[feature_size_*local_src+rank] *edge_data[feature_size_*i_i+rank]);
	// 	}
	// 	}
	// }
	for(int i=threadId;i<feature_size_*batch_size_;i+=blockDim.x*gridDim.x){
		T_l local_dst=i/feature_size_;
		T_l rank=i%feature_size_;
	 	for(int i_i=dst[local_dst];i_i<dst[local_dst+1];i_i++){
			int local_src=src[i_i];
			atomicAdd(&new_feature[feature_size_*local_dst+rank],
				old_feature[feature_size_*local_src+rank]/edge_data[local_src]);
	 	}
		
	}
}
template <typename T_v,typename T_l>
__global__ void processing_no_weight(T_l *src, T_l *dst, T_v* old_feature, T_v* new_feature,T_v* edge_data,int batch_size_,int src_s_,int feature_size_){

	int large_size=blockDim.x;
	int threadId = blockIdx.x *blockDim.x + threadIdx.x;
	//int rank=threadIdx.x;
	if(threadId==0){
		printf("in");
		for(int j=0;j<10;j++){
			printf("%f\t\n",new_feature[j*feature_size_]);
		}	printf("\n%d %d %d\n",batch_size_,feature_size_,blockDim.x*gridDim.x);
	// printf("%d\n",batch_size_);
	// printf("%d\n??",src_s_);
	// for(int j=0;j<10;j++){
	// 		printf("%d\n",src[j]);
	// 	}	printf("\n");
	}

	// //			128 		128		  2208
	// for(int i=threadId;i<large_size*batch_size_;i+=blockDim.x*gridDim.x){
	// 	if(rank<feature_size_){
	// 	T_l dstid=i/blockDim.x;
	// 	T_l local_dst=dstid;
	// 	for(int i_i=dst[dstid];i_i<dst[dstid+1];i_i++){
	// 		int local_src=src[i_i];
	// 	//	printf("d%dd",feature_size_*local_src+rank);
	// 		atomicAdd(&new_feature[feature_size_*local_dst+rank],old_feature[feature_size_*local_src+rank]);
	// 	}
	// 	}
	// }
	for(int i=threadId;i<feature_size_*batch_size_;i+=blockDim.x*gridDim.x){
		T_l local_dst=i/feature_size_;
		T_l rank=i%feature_size_;
	 	for(int i_i=dst[local_dst];i_i<dst[local_dst+1];i_i++){
			int local_src=src[i_i];
			atomicAdd(&new_feature[feature_size_*local_dst+rank],
				old_feature[feature_size_*local_src+rank]);
	 	}
		
	}

}


template <typename T_v,typename T_l>
__global__ void process_edges_chunk(T_l *src, T_l *dst,int *meta, T_v* old_feature, T_v* new_feature,T_v* edge_data){
	__shared__ int metaInfo[8];

	int large_size=blockDim.x;
	int threadId = blockIdx.x *blockDim.x + threadIdx.x;
	int rank=threadIdx.x;
	int feature_size_=meta[7];//size of a vertex feature
	int batch_size_=batch_size(meta);//number of vertices in this batch
	int src_s_=src_s(meta);
	if(threadIdx.x<=8){
		metaInfo[threadIdx.x]=meta[threadIdx.x];
	}
	__syncthreads();
	for(int i=threadId;i<large_size*batch_size_;i+=blockDim.x*gridDim.x){
		if(rank<feature_size_){
		T_l dstid=i/blockDim.x;
		T_l local_dst=dstid;
		for(int i_i=dst[dstid];i_i<dst[dstid+1];i_i++){
			int local_src=src[i_i]-src_s(metaInfo);
			new_feature[feature_size_*local_dst+rank]+=
				old_feature[feature_size_*local_src+rank]
				    *edge_data[feature_size_*i_i+rank];
		}
		}
	}
}
template <typename T_v,typename T_l>
__global__ void process_edges_chunk_one_by_one(T_l *src, T_l *dst,int *meta, T_v* old_feature, T_v* new_feature,T_v* edge_data){
	__shared__ int metaInfo[8];

	int large_size=blockDim.x;
	int threadId = blockIdx.x *blockDim.x + threadIdx.x;
	int rank=threadIdx.x;
	int feature_size_=meta[7];//size of a vertex feature
	int batch_size_=graph_size(meta);//number of vertices in this batch
	int src_s_=src_s(meta);
	if(threadIdx.x<=8){
		metaInfo[threadIdx.x]=meta[threadIdx.x];
	}
	__syncthreads();
	for(int i=threadId;i<large_size*batch_size_;i+=blockDim.x*gridDim.x){
		if(rank<feature_size_){
		T_l local_dst=dst[i/blockDim.x]-dst_s(metaInfo);
		T_l local_src=src[i/blockDim.x]-src_s(metaInfo);
			atomicAdd(&new_feature[feature_size_*local_dst+rank],
				old_feature[feature_size_*local_src+rank]
				    *edge_data[feature_size_*i/blockDim.x+rank]);
		}
	}
}

__global__ void show_dot_mult_to_buffer(float* buffer, int r, int c){

	for(int i=0;i<r;i++){
		for(int j=0;j<c;j++){
			printf("%f\t",buffer[i*c+j]);
		}printf("\n");
	}
}


__global__ void dot_mult_to_buffer(float* buffer, float *src, float *dst, 
												int count, int r, int c){

	int large_size=blockDim.x;
	int threadId = blockIdx.x *blockDim.x + threadIdx.x;
	int rank=threadIdx.x;
	for(int i=threadId;i<count*c;i+=blockDim.x*gridDim.x){
			buffer[i]=src[i]*dst[i];
		}
}
__global__ void dot_mult_tensor_weight(float* buffer, float *src, float *dst, 
												int count, int c){

	int large_size=blockDim.x;
	int threadId = blockIdx.x *blockDim.x + threadIdx.x;
	int rank=threadIdx.x;
	for(int i=threadId;i<count*c;i+=blockDim.x*gridDim.x){
			buffer[i]=src[i]*dst[i];
		}
}
__global__ void dot_mult_scalar_weight(float* buffer, float *src, float *dst, 
												int count, int c){

	int large_size=blockDim.x;
	int threadId = blockIdx.x *blockDim.x + threadIdx.x;
	int rank=threadIdx.x;
	for(int i=threadId;i<count*c;i+=blockDim.x*gridDim.x){
			buffer[i]=src[i]*dst[i/c];
		}
}
__global__ void time_mult0(float* aggregate_grad, float *src, float *buffer,
																 int count, int c, int r){
//r: 1433
//c: 16 
//	buffer {count,c}
//  src    {count,r}
// TEST:
/*
src:{1,5} r:5
buffer:{1,6} c:6
src:{2,2} r:2
buffer:{2,2} c:2
*/
	int large_size=blockDim.x;
	int threadId = blockIdx.x *blockDim.x + threadIdx.x;
	int rank=threadIdx.x;
//		printf("%d %d\n",rank,blockDim.x*gridDim.x);
	 for(int i=threadId;i<count*r;i+=blockDim.x*gridDim.x){
	// 	printf("in%d\n",threadId);
			for(int j=0;j<c;j++){
		    	atomicAdd(&aggregate_grad[(i%r)*c+j],buffer[(i/r)*c+j]*src[i]);
		   // 	if(i==1)
		   // 	printf("%d,%d,%d,%f\t",(i%r)*c+j,(i/c)*r+j,i,buffer[(i/r)*c+j]*src[i]);
			}
	 }

}
__global__ void time_mult1(float* aggregate_grad, float *src, float *buffer,
																 int count, int c, int r){
//r: 1433
//c: 16 
//	buffer {count,c}
//  src    {count,r}
// TEST:
/*
src:{1,5} r:5
buffer:{1,6} c:6
src:{2,2} r:2
buffer:{2,2} c:2

*/
	int large_size=blockDim.x;
	int threadId = blockIdx.x *blockDim.x + threadIdx.x;
	int rank=threadIdx.x;
//		printf("%d %d\n",threadId,blockDim.x*gridDim.x);
	float  data2;
	 for(int i=threadId;i<c*r;i+=blockDim.x*gridDim.x){
	 	for(int j=0;j<count;j++){
		    	atomicAdd(&aggregate_grad[i],buffer[j*c+i%c]*src[j*r+i/c]);
		    //	if(i==1)
		    //	printf("%d,%d,%d,%f\t",(i%r)*c+j,(i/c)*r+j,i,buffer[(i/r)*c+j]*src[i]);
		}
	 }

}


__global__ void time_mult_to_combine_grad(float* aggregate_grad, float *src, float *buffer,
																 int count, int r, int c){
//r: 1433
//c: 16 
//	buffer {count,c}
//  src    {count,r}
// TEST:
/*
src:{1,5} r:5
buffer:{1,6} c:6
src:{2,2} r:2
buffer:{2,2} c:2

*/
	int large_size=blockDim.x;
	int threadId = blockIdx.x *blockDim.x + threadIdx.x;
	int rank=threadIdx.x;
//		printf("%d %d\n",rank,blockDim.x*gridDim.x);
	 for(int i=threadId;i<count*r;i+=blockDim.x*gridDim.x){
	// 	printf("in%d\n",threadId);
			for(int j=0;j<c;j++){
		    	atomicAdd(&aggregate_grad[(i%r)*c+j],buffer[(i/r)*c+j]*src[i]);
		   // 	if(i==1)
		   // 	printf("%d,%d,%d,%f\t",(i%r)*c+j,(i/c)*r+j,i,buffer[(i/r)*c+j]*src[i]);
			}
	 }

}
__global__ void time_mult_to_combine_grad1(float* aggregate_grad, float *src, float *buffer,
																 int count, int r, int c){
//r: 1433
//c: 16 
//	buffer {count,c}
//  src    {count,r}
// TEST:
/*
src:{1,5} r:5
buffer:{1,6} c:6
src:{2,2} r:2
buffer:{2,2} c:2

*/
	int large_size=blockDim.x;
	int threadId = blockIdx.x *blockDim.x + threadIdx.x;
	int rank=threadIdx.x;
//		printf("%d %d\n",threadId,blockDim.x*gridDim.x);
	float  data2;
	 for(int i=threadId;i<c*r;i+=blockDim.x*gridDim.x){
	 	for(int j=0;j<count;j++){
		    	atomicAdd(&aggregate_grad[i],buffer[j*c+i%c]*src[j*r+i/c]);
		    //	if(i==1)
		    //	printf("%d,%d,%d,%f\t",(i%r)*c+j,(i/c)*r+j,i,buffer[(i/r)*c+j]*src[i]);
		}
	 }

}


#endif /* PROPAGATE_H_ */

/*
 * propagate.h
 *
 *  Created on: Dec 3, 2019
 *      Author: wangqg
 * TODO :cub support and shared memory optimization
 */

#ifndef PROPAGATE_H_
#define PROPAGATE_H_


#include<stdlib.h>
#include<stdio.h>
#include<cstdio>
#include<assert.h>
#include <sys/time.h>
#include<cuda.h>
#include"cub/cub.cuh"
#include"cuda_type.h"
inline double get_time() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + (tv.tv_usec / 1e6);
  }
inline int max_(int a,int b){
	if (a>b)return a;
	else return b;
}
inline int min_(int a,int b){
	if (a<b)return a;
	else return b;
}

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



//UnTested
//for dimension smaller than 512;
//with low performance 
__global__ void aggregate_kernel_from_src_with_weight_optim_ROC(const VertexId_CUDA *row_indices,
                    const  VertexId_CUDA *column_offset, long *destination,
 		const float* old_feature, float* new_feature,const float* weight,
 		VertexId_CUDA src_s_,VertexId_CUDA src_e_,
                VertexId_CUDA dst_s_,VertexId_CUDA dst_e_,
                VertexId_CUDA edges,
 		VertexId_CUDA batch_size_, VertexId_CUDA feature_size_){
        int VtxPerBlock=CUDA_NUM_THREADS/feature_size_;
        typedef cub::BlockScan<VertexId_CUDA,CUDA_NUM_THREADS> BlockScan;
        __shared__ BlockScan::TempStorage temp_storage;
        __shared__ VertexId_CUDA blkRowStart;
        __shared__ float acc_h[CUDA_NUM_THREADS];
        int tidDiv=threadIdx.x/feature_size_;
        int tidMod=threadIdx.x%feature_size_;
        //block level iteration determnes
        
        for(VertexId_CUDA blkColStart=blockIdx.x*VtxPerBlock;blkColStart<batch_size_;blkColStart+=VtxPerBlock*gridDim.x){
            VertexId_CUDA myNumEdges=0,scratchOffset,totalNumEdges=0;
            VertexId_CUDA localDst=0;
            if(threadIdx.x+blkColStart<batch_size_&&threadIdx.x<VtxPerBlock){
                VertexId_CUDA curVtx_trans=threadIdx.x+blkColStart;
                VertexId_CUDA rowIdxStart=column_offset[curVtx_trans];
                VertexId_CUDA rowIdxEnd=column_offset[curVtx_trans+1];
                assert(rowIdxStart>=0&&rowIdxEnd<=edges);
                myNumEdges=rowIdxEnd-rowIdxStart;
                if(threadIdx.x==0)
                    blkRowStart=rowIdxStart;
            }
        acc_h[threadIdx.x]=0.0f;
        __syncthreads();
        BlockScan(temp_storage).ExclusiveSum(myNumEdges, scratchOffset, totalNumEdges);
        VertexId_CUDA done=0;
        while(totalNumEdges>0){
            if(tidDiv<totalNumEdges&&tidDiv<VtxPerBlock){
                VertexId_CUDA src_trans= row_indices[blkRowStart+done+tidDiv]-src_s_;//different with threads num
                VertexId_CUDA dst_trans= (VertexId_CUDA)destination[blkRowStart+done+tidDiv]-dst_s_;//different with threads num
                float w=weight[blkRowStart+done+tidDiv];
                float val=old_feature[src_trans*feature_size_+tidMod]*w;
                //assert(dst_trans>=blkColStart&&dst_trans<blkColStart+VtxPerBlock);
                int offset=(dst_trans-blkColStart)*feature_size_+tidMod;
                atomicAdd(&acc_h[offset],val);
            }
            done+=VtxPerBlock;
           totalNumEdges-=(totalNumEdges>VtxPerBlock) ? VtxPerBlock : totalNumEdges;
        }
        __syncthreads();
        if(tidDiv<VtxPerBlock&&tidDiv+blkColStart<=batch_size_){
            new_feature[(blkColStart)*feature_size_+threadIdx.x]=acc_h[threadIdx.x];
        
        }
    }
}

__global__ void aggregate_kernel_from_src_with_weight_optim_load_imbalance(const VertexId_CUDA *row_indices,
                    const  VertexId_CUDA *column_offset,long* destination,
 		const float* old_feature, float* new_feature,const float* weight,
 		VertexId_CUDA src_s_,VertexId_CUDA src_e_,
                VertexId_CUDA dst_s_,VertexId_CUDA dst_e_,
                VertexId_CUDA edges,
 		VertexId_CUDA batch_size_, VertexId_CUDA feature_size_){
        int VtxPerBlock=CUDA_NUM_THREADS/feature_size_;
        typedef cub::BlockScan<VertexId_CUDA,CUDA_NUM_THREADS> BlockScan;
        __shared__ BlockScan::TempStorage temp_storage;
        __shared__ VertexId_CUDA blkRowStart;
        __shared__ float acc_h[CUDA_NUM_THREADS];
        int tidDiv=threadIdx.x/feature_size_;
        int tidMod=threadIdx.x%feature_size_;
        //block level iteration determnes
        for(VertexId_CUDA blkColStart=blockIdx.x*VtxPerBlock;blkColStart<batch_size_;blkColStart+=VtxPerBlock*gridDim.x){
            VertexId_CUDA myNumEdges=0;
            VertexId_CUDA rowIdxStart=0;
            VertexId_CUDA rowIdxEnd=0;
            //the vertex to compute in current block, only the first several threads are busy.
            if(blkColStart+tidDiv<batch_size_&&tidDiv<VtxPerBlock){
                VertexId_CUDA curVtx_trans=blkColStart+tidDiv;
                rowIdxStart=column_offset[curVtx_trans];
                rowIdxEnd=column_offset[curVtx_trans+1];
                //assert(rowIdxStart>=0&&rowIdxEnd<=edges);
                myNumEdges=rowIdxEnd-rowIdxStart;
                if(threadIdx.x==0)
                    blkRowStart=rowIdxStart;
            }
        acc_h[threadIdx.x]=0.0f;
        __syncthreads();
        //BlockScan(temp_storage).ExclusiveSum(myNumEdges, scratchOffset, totalNumEdges);
        VertexId_CUDA done=0;
        while(myNumEdges>0){
            if(tidDiv<VtxPerBlock){
                VertexId_CUDA src_trans= row_indices[rowIdxStart+done]-src_s_;//different with threads num
                VertexId_CUDA dst_trans= blkColStart+tidDiv;//destination[blkRowStart+done+tidDiv]-dst_s_;//different with threads num
                float w=weight[rowIdxStart+done];
                //assert(src_trans<(src_e_-src_s_));
                float val=old_feature[src_trans*feature_size_+tidMod];
                //assert(dst_trans>=blkColStart&&dst_trans<blkColStart+VtxPerBlock);
                int offset=(dst_trans-blkColStart)*feature_size_+tidMod;
                //assert(offset<CUDA_NUM_THREADS);
                atomicAdd(&acc_h[offset],val);
            }
            done+=1;
           myNumEdges-=1;
        }
        __syncthreads();
        if(tidDiv<VtxPerBlock&&tidDiv+blkColStart<batch_size_){
            new_feature[(blkColStart)*feature_size_+threadIdx.x]=acc_h[threadIdx.x];
        
        }
    }
}

/*
 * The following three
FORWARD computation kernel
 * input: old feature
 * output: new feature
 * 
 * from source to destination
 * _with_weight:    scalar
 * _tensor_weight:  tensor
 * _without_weight  no_weight
 * with shared memory optimization
 * For dimension smaller than 512
*/
__global__ void aggregate_kernel_from_src_with_weight_optim_nts(const VertexId_CUDA *row_indices,
                    const  VertexId_CUDA *column_offset, long *destination,
 		const float* old_feature, float* new_feature,const float* weight,
 		VertexId_CUDA src_s_,VertexId_CUDA src_e_,
                VertexId_CUDA dst_s_,VertexId_CUDA dst_e_,
                VertexId_CUDA edges,
 		VertexId_CUDA batch_size_, VertexId_CUDA feature_size_){
        int VtxPerBlock=CUDA_NUM_THREADS/feature_size_;
        typedef cub::BlockScan<VertexId_CUDA,CUDA_NUM_THREADS> BlockScan;
        __shared__ BlockScan::TempStorage temp_storage;
        __shared__ VertexId_CUDA blkRowStart;
        __shared__ VertexId_CUDA blkRowEnd;
        __shared__ float acc_h[CUDA_NUM_THREADS];
        int tidDiv=threadIdx.x/feature_size_;
        int tidMod=threadIdx.x%feature_size_;
        
        for(VertexId_CUDA blkColStart=blockIdx.x*VtxPerBlock;blkColStart<batch_size_;blkColStart+=VtxPerBlock*gridDim.x){
            VertexId_CUDA myNumEdges=0,scratchOffset,totalNumEdges=0;
            if(threadIdx.x+blkColStart<batch_size_&&threadIdx.x<VtxPerBlock){
                VertexId_CUDA curVtx_trans=blkColStart+threadIdx.x;
                VertexId_CUDA rowIdxStart=column_offset[curVtx_trans];
                VertexId_CUDA rowIdxEnd=column_offset[curVtx_trans+1];
                assert(rowIdxStart>=0&&rowIdxEnd<=edges);
                if(threadIdx.x==0)
                    blkRowStart=rowIdxStart;
                if((threadIdx.x+blkColStart)==(batch_size_-1)||threadIdx.x==(VtxPerBlock)-1)
                    blkRowEnd=rowIdxEnd;
            }
        acc_h[threadIdx.x]=0.0f;
        __syncthreads();
        totalNumEdges=blkRowEnd-blkRowStart;
        VertexId_CUDA done=0;
        VertexId_CUDA curr_dst_offset=0;
        VertexId_CUDA curr_dst_edges=column_offset[blkColStart+1]-column_offset[blkColStart];
        
        while(totalNumEdges>0){
            if(tidDiv<totalNumEdges&&tidDiv<VtxPerBlock){        
                VertexId_CUDA e_offset=done+tidDiv;
                // COMPUTING destination ID rather than loading the large destination array in.
                while(e_offset>=curr_dst_edges){
                    curr_dst_offset++;
                    curr_dst_edges+=column_offset[blkColStart+1+curr_dst_offset]-column_offset[blkColStart+curr_dst_offset];
                }
                VertexId_CUDA dst_trans= blkColStart+curr_dst_offset;
                VertexId_CUDA src_trans= row_indices[blkRowStart+e_offset]-src_s_;//different with threads num
                float w=weight[blkRowStart+e_offset];
                float val=old_feature[src_trans*feature_size_+tidMod]*w;
//               // assert(dst_trans>=blkColStart&&dst_trans<blkColStart+VtxPerBlock);
                int offset=(dst_trans-blkColStart)*feature_size_+tidMod;
                atomicAdd(&acc_h[offset],val);
                //acc_h[offset]+=val;
            }          
           done+=VtxPerBlock;
           totalNumEdges-=(totalNumEdges>VtxPerBlock) ? VtxPerBlock : totalNumEdges;
        }
        __syncthreads();
        if(tidDiv<VtxPerBlock&&tidDiv+blkColStart<=batch_size_){
            new_feature[(blkColStart)*feature_size_+threadIdx.x]=acc_h[threadIdx.x];
        
        }
    }
}

__global__ void aggregate_kernel_from_src_without_weight_optim_nts(const VertexId_CUDA *row_indices,
                    const  VertexId_CUDA *column_offset, long *destination,
 		const float* old_feature, float* new_feature,const float* weight,
 		VertexId_CUDA src_s_,VertexId_CUDA src_e_,
                VertexId_CUDA dst_s_,VertexId_CUDA dst_e_,
                VertexId_CUDA edges,
 		VertexId_CUDA batch_size_, VertexId_CUDA feature_size_){
        int VtxPerBlock=CUDA_NUM_THREADS/feature_size_;
        typedef cub::BlockScan<VertexId_CUDA,CUDA_NUM_THREADS> BlockScan;
        __shared__ BlockScan::TempStorage temp_storage;
        __shared__ VertexId_CUDA blkRowStart;
        __shared__ VertexId_CUDA blkRowEnd;
        __shared__ float acc_h[CUDA_NUM_THREADS];
        int tidDiv=threadIdx.x/feature_size_;
        int tidMod=threadIdx.x%feature_size_;
        //block level iteration determnes
        
        for(VertexId_CUDA blkColStart=blockIdx.x*VtxPerBlock;blkColStart<batch_size_;blkColStart+=VtxPerBlock*gridDim.x){
            VertexId_CUDA totalNumEdges=0;
            if(threadIdx.x+blkColStart<batch_size_&&threadIdx.x<VtxPerBlock){
                VertexId_CUDA curVtx_trans=blkColStart+threadIdx.x;
                VertexId_CUDA rowIdxStart=column_offset[curVtx_trans];
                VertexId_CUDA rowIdxEnd=column_offset[curVtx_trans+1];
                assert(rowIdxStart>=0&&rowIdxEnd<=edges);
                if(threadIdx.x==0)
                    blkRowStart=rowIdxStart;
                if((threadIdx.x+blkColStart)==(batch_size_-1)||threadIdx.x==(VtxPerBlock)-1)
                    blkRowEnd=rowIdxEnd;
            }
        acc_h[threadIdx.x]=0.0f;
        __syncthreads();
        totalNumEdges=blkRowEnd-blkRowStart;
        VertexId_CUDA done=0;
        VertexId_CUDA curr_dst_offset=0;
        VertexId_CUDA curr_dst_edges=column_offset[blkColStart+1]-column_offset[blkColStart];
        while(totalNumEdges>0){
            if(tidDiv<totalNumEdges&&tidDiv<VtxPerBlock){
                VertexId_CUDA e_offset=done+tidDiv;
                while(e_offset>=curr_dst_edges){
                    curr_dst_offset++;
                    curr_dst_edges+=column_offset[blkColStart+1+curr_dst_offset]-column_offset[blkColStart+curr_dst_offset];
                }
                VertexId_CUDA dst_trans= blkColStart+curr_dst_offset;
                VertexId_CUDA src_trans= row_indices[blkRowStart+e_offset]-src_s_;//different with threads num
                //VertexId_CUDA dst_trans= (VertexId_CUDA)destination[blkRowStart+done+tidDiv]-dst_s_;//different with threads num
                float val=old_feature[src_trans*feature_size_+tidMod];
                assert(dst_trans>=blkColStart&&dst_trans<blkColStart+VtxPerBlock);
                int offset=(dst_trans-blkColStart)*feature_size_+tidMod;
                atomicAdd(&acc_h[offset],val);
            }
            done+=VtxPerBlock;
           totalNumEdges-=(totalNumEdges>VtxPerBlock) ? VtxPerBlock : totalNumEdges;
        }
        __syncthreads();
        if(tidDiv<VtxPerBlock&&tidDiv+blkColStart<=batch_size_){
            atomicAdd(&new_feature[(blkColStart)*feature_size_+threadIdx.x],acc_h[threadIdx.x]);
        
        }
    }
}

__global__ void aggregate_kernel_from_src_tensor_weight_optim_nts(const VertexId_CUDA *row_indices,
                    const  VertexId_CUDA *column_offset, long *destination,
 		const float* old_feature, float* new_feature,const float* weight,
 		VertexId_CUDA src_s_,VertexId_CUDA src_e_,
                VertexId_CUDA dst_s_,VertexId_CUDA dst_e_,
                VertexId_CUDA edges,
 		VertexId_CUDA batch_size_, VertexId_CUDA feature_size_){
        int VtxPerBlock=CUDA_NUM_THREADS/feature_size_;
        typedef cub::BlockScan<VertexId_CUDA,CUDA_NUM_THREADS> BlockScan;
        __shared__ BlockScan::TempStorage temp_storage;
        __shared__ VertexId_CUDA blkRowStart;
        __shared__ VertexId_CUDA blkRowEnd;
        __shared__ float acc_h[CUDA_NUM_THREADS];
        int tidDiv=threadIdx.x/feature_size_;
        int tidMod=threadIdx.x%feature_size_;
        //block level iteration determnes
        
        for(VertexId_CUDA blkColStart=blockIdx.x*VtxPerBlock;blkColStart<batch_size_;blkColStart+=VtxPerBlock*gridDim.x){
            VertexId_CUDA myNumEdges=0,scratchOffset,totalNumEdges=0;
            if(threadIdx.x+blkColStart<batch_size_&&threadIdx.x<VtxPerBlock){
                VertexId_CUDA curVtx_trans=blkColStart+threadIdx.x;
                VertexId_CUDA rowIdxStart=column_offset[curVtx_trans];
                VertexId_CUDA rowIdxEnd=column_offset[curVtx_trans+1];
                assert(rowIdxStart>=0&&rowIdxEnd<=edges);
                if(threadIdx.x==0)
                    blkRowStart=rowIdxStart;
                if((threadIdx.x+blkColStart)==(batch_size_-1)||threadIdx.x==(VtxPerBlock)-1)
                    blkRowEnd=rowIdxEnd;
            }
        acc_h[threadIdx.x]=0.0f;
        __syncthreads();
        totalNumEdges=blkRowEnd-blkRowStart;
        VertexId_CUDA done=0;
        while(totalNumEdges>0){
            if(tidDiv<totalNumEdges&&tidDiv<VtxPerBlock){
                VertexId_CUDA src_trans= row_indices[blkRowStart+done+tidDiv]-src_s_;//different with threads num
                VertexId_CUDA dst_trans= (VertexId_CUDA)destination[blkRowStart+done+tidDiv]-dst_s_;//different with threads num
                float w=weight[(blkRowStart+done+tidDiv)*feature_size_+tidMod];
                float val=old_feature[src_trans*feature_size_+tidMod]*w;
                assert(dst_trans>=blkColStart&&dst_trans<blkColStart+VtxPerBlock);
                int offset=(dst_trans-blkColStart)*feature_size_+tidMod;
                atomicAdd(&acc_h[offset],val);
            }
            done+=VtxPerBlock;
           totalNumEdges-=(totalNumEdges>VtxPerBlock) ? VtxPerBlock : totalNumEdges;
        }
        __syncthreads();
        if(tidDiv<VtxPerBlock&&tidDiv+blkColStart<=batch_size_){
            new_feature[(blkColStart)*feature_size_+threadIdx.x]=acc_h[threadIdx.x];
        
        }
    }
}

//for dimension larger than 512
template <typename T_v,typename T_l>
__global__ void aggregate_kernel_from_src_with_weight(const T_l *row_indices,const  T_l *column_offset,
 		const T_v* old_feature, T_v* new_feature,const T_v* weight,
 		T_l src_s_,T_l dst_s_,
 		T_l batch_size_, T_l feature_size_){
	int large_size=blockDim.x;
	int threadId = blockIdx.x *blockDim.x + threadIdx.x;

	for(long i=threadId;i<feature_size_*batch_size_;i+=blockDim.x*gridDim.x){
		T_l local_dst=i/feature_size_;
		T_l rank=i%feature_size_;
		for(int i_i=column_offset[local_dst];i_i<column_offset[local_dst+1];i_i++){
			int local_src=row_indices[i_i]-src_s_;
			 atomicAdd(&new_feature[feature_size_*local_dst+rank],
			 	old_feature[feature_size_*local_src+rank]*weight[i_i]);
	 	}
		
	}
}

template <typename T_v,typename T_l>
__global__ void aggregate_kernel_from_src_tensor_weight(const T_l *row_indices,const  T_l *column_offset,
 		const T_v* old_feature, T_v* new_feature,const T_v* weight,
 		T_l src_s_,T_l dst_s_,
 		T_l batch_size_, T_l feature_size_){
	int large_size=blockDim.x;
	int threadId = blockIdx.x *blockDim.x + threadIdx.x;

	for(long i=threadId;i<feature_size_*batch_size_;i+=blockDim.x*gridDim.x){
		T_l local_dst=i/feature_size_;
		T_l rank=i%feature_size_;
		for(int i_i=column_offset[local_dst];i_i<column_offset[local_dst+1];i_i++){
			int local_src=row_indices[i_i]-src_s_;
			 atomicAdd(&new_feature[feature_size_*local_dst+rank],
			 	old_feature[feature_size_*local_src+rank]*weight[i_i*feature_size_+rank]);
	 	}
		
	}
}

template <typename T_v,typename T_l>
__global__ void aggregate_kernel_from_src_without_weight(const T_l *row_indices,const  T_l *column_offset,
 		const T_v* old_feature, T_v* new_feature,const T_v* weight,
 		T_l src_s_,T_l dst_s_,
 		T_l batch_size_, T_l feature_size_){
	int large_size=blockDim.x;
	int threadId = blockIdx.x *blockDim.x + threadIdx.x;
	for(long i=threadId;i<feature_size_*batch_size_;i+=blockDim.x*gridDim.x){
		T_l local_dst=i/feature_size_;
		T_l rank=i%feature_size_;
		for(int i_i=column_offset[local_dst];i_i<column_offset[local_dst+1];i_i++){
			int local_src=row_indices[i_i]-src_s_;
			 atomicAdd(&new_feature[feature_size_*local_dst+rank],
			 	old_feature[feature_size_*local_src+rank]);
	 	}
		
	}
}




/*
 * The following three
BACKWARD computation kernel
 * input: old feature
 * output: new feature
 * 
 * from destination to source
 * _with_weight:    scale
 * _tensor_weight:  tensor
 * _without_weight  no_weight
 * with shared memory optimization
 * For dimension smaller than 512
*/
__global__ void aggregate_kernel_from_dst_with_weight_optim_nts(const VertexId_CUDA *row_offset,
                    const  VertexId_CUDA *column_indices, long *source,
 		const float* old_feature, float* new_feature,const float* weight,
 		VertexId_CUDA src_s_,VertexId_CUDA src_e_,
                VertexId_CUDA dst_s_,VertexId_CUDA dst_e_,
                VertexId_CUDA edges,
 		VertexId_CUDA batch_size_, VertexId_CUDA feature_size_){
        int VtxPerBlock=CUDA_NUM_THREADS/feature_size_;
        __shared__ VertexId_CUDA blkColStart;
        __shared__ VertexId_CUDA blkColEnd;
        __shared__ float acc_h[CUDA_NUM_THREADS];
        int tidDiv=threadIdx.x/feature_size_;
        int tidMod=threadIdx.x%feature_size_;
        //block level iteration determnes
        
        for(VertexId_CUDA blkRowStart=blockIdx.x*VtxPerBlock;blkRowStart<batch_size_;blkRowStart+=VtxPerBlock*gridDim.x){
            VertexId_CUDA totalNumEdges=0;
            if(threadIdx.x+blkRowStart<batch_size_&&threadIdx.x<VtxPerBlock){
                VertexId_CUDA curVtx_trans=blkRowStart+threadIdx.x;
                VertexId_CUDA colIdxStart=row_offset[curVtx_trans];
                VertexId_CUDA colIdxEnd=row_offset[curVtx_trans+1];
                assert(colIdxStart>=0&&colIdxEnd<=edges);
                if(threadIdx.x==0)
                    blkColStart=colIdxStart;
                if((threadIdx.x+blkRowStart)==(batch_size_-1)||threadIdx.x==(VtxPerBlock)-1)
                    blkColEnd=colIdxEnd;
            }
        acc_h[threadIdx.x]=0.0f;
        __syncthreads();
        totalNumEdges=blkColEnd-blkColStart;
        VertexId_CUDA done=0;
        VertexId_CUDA curr_src_offset=0;
        VertexId_CUDA curr_src_edges=row_offset[blkRowStart+1]-row_offset[blkRowStart];
        while(totalNumEdges>0){
            if(tidDiv<totalNumEdges&&tidDiv<VtxPerBlock){
                VertexId_CUDA e_offset=done+tidDiv;
                
                // COMPUTING source ID rather than loading the large source array in.
                while(e_offset>=curr_src_edges){
                    curr_src_offset++;
                    curr_src_edges+=row_offset[blkRowStart+1+curr_src_offset]-row_offset[blkRowStart+curr_src_offset];
                }
                VertexId_CUDA src_trans= blkRowStart+curr_src_offset;
                VertexId_CUDA dst_trans= column_indices[blkColStart+e_offset]-dst_s_;
                float w=weight[blkColStart+e_offset];
                float val=old_feature[dst_trans*feature_size_+tidMod]*w;
//                assert(src_trans>=blkRowStart&&src_trans<blkRowStart+VtxPerBlock);
                int offset=(src_trans-blkRowStart)*feature_size_+tidMod;
                atomicAdd(&acc_h[offset],val);
            }
            done+=VtxPerBlock;
           totalNumEdges-=(totalNumEdges>VtxPerBlock) ? VtxPerBlock : totalNumEdges;
        }
        __syncthreads();
        if(tidDiv<VtxPerBlock&&tidDiv+blkRowStart<=batch_size_){
             new_feature[(blkRowStart)*feature_size_+threadIdx.x]=acc_h[threadIdx.x];
        
        }
    }
}

__global__ void aggregate_kernel_from_dst_tensor_weight_optim_nts(const VertexId_CUDA *row_offset,
                    const  VertexId_CUDA *column_indices, long *source,
 		const float* old_feature, float* new_feature,const float* weight,
 		VertexId_CUDA src_s_,VertexId_CUDA src_e_,
                VertexId_CUDA dst_s_,VertexId_CUDA dst_e_,
                VertexId_CUDA edges,
 		VertexId_CUDA batch_size_, VertexId_CUDA feature_size_){
        int VtxPerBlock=CUDA_NUM_THREADS/feature_size_;
        __shared__ VertexId_CUDA blkColStart;
        __shared__ VertexId_CUDA blkColEnd;
        __shared__ float acc_h[CUDA_NUM_THREADS];
        int tidDiv=threadIdx.x/feature_size_;
        int tidMod=threadIdx.x%feature_size_;
        //block level iteration determnes
        
        for(VertexId_CUDA blkRowStart=blockIdx.x*VtxPerBlock;blkRowStart<batch_size_;blkRowStart+=VtxPerBlock*gridDim.x){
            VertexId_CUDA totalNumEdges=0;
            if(threadIdx.x+blkRowStart<batch_size_&&threadIdx.x<VtxPerBlock){
                VertexId_CUDA curVtx_trans=blkRowStart+threadIdx.x;
                VertexId_CUDA colIdxStart=row_offset[curVtx_trans];
                VertexId_CUDA colIdxEnd=row_offset[curVtx_trans+1];
                assert(colIdxStart>=0&&colIdxEnd<=edges);
                if(threadIdx.x==0)
                    blkColStart=colIdxStart;
                if((threadIdx.x+blkRowStart)==(batch_size_-1)||threadIdx.x==(VtxPerBlock)-1)
                    blkColEnd=colIdxEnd;
            }
        acc_h[threadIdx.x]=0.0f;
        __syncthreads();
        totalNumEdges=blkColEnd-blkColStart;
        VertexId_CUDA done=0;
        while(totalNumEdges>0){
            if(tidDiv<totalNumEdges&&tidDiv<VtxPerBlock){
                VertexId_CUDA dst_trans= column_indices[blkColStart+done+tidDiv]-dst_s_;//different with threads num
                VertexId_CUDA src_trans= (VertexId_CUDA)source[blkColStart+done+tidDiv]-src_s_;//different with threads num
                float w=weight[(blkColStart+done+tidDiv)*tidDiv+tidMod];
                float val=old_feature[dst_trans*feature_size_+tidMod]*w;
                assert(src_trans>=blkRowStart&&src_trans<blkRowStart+VtxPerBlock);
                int offset=(src_trans-blkRowStart)*feature_size_+tidMod;
                atomicAdd(&acc_h[offset],val);
            }
            done+=VtxPerBlock;
           totalNumEdges-=(totalNumEdges>VtxPerBlock) ? VtxPerBlock : totalNumEdges;
        }
        __syncthreads();
        if(tidDiv<VtxPerBlock&&tidDiv+blkRowStart<=batch_size_){
             new_feature[(blkRowStart)*feature_size_+threadIdx.x]=acc_h[threadIdx.x];
        
        }
    }
}

__global__ void aggregate_kernel_from_dst_without_weight_optim_nts(const VertexId_CUDA *row_offset,
                    const  VertexId_CUDA *column_indices, long *source,
 		const float* old_feature, float* new_feature,const float* weight,
 		VertexId_CUDA src_s_,VertexId_CUDA src_e_,
                VertexId_CUDA dst_s_,VertexId_CUDA dst_e_,
                VertexId_CUDA edges,
 		VertexId_CUDA batch_size_, VertexId_CUDA feature_size_){
        int VtxPerBlock=CUDA_NUM_THREADS/feature_size_;
        __shared__ VertexId_CUDA blkColStart;
        __shared__ VertexId_CUDA blkColEnd;
        __shared__ float acc_h[CUDA_NUM_THREADS];
        int tidDiv=threadIdx.x/feature_size_;
        int tidMod=threadIdx.x%feature_size_;
        //block level iteration determnes
        for(VertexId_CUDA blkRowStart=blockIdx.x*VtxPerBlock;blkRowStart<batch_size_;blkRowStart+=VtxPerBlock*gridDim.x){
            VertexId_CUDA totalNumEdges=0;
            if(threadIdx.x+blkRowStart<batch_size_&&threadIdx.x<VtxPerBlock){
                VertexId_CUDA curVtx_trans=blkRowStart+threadIdx.x;
                VertexId_CUDA colIdxStart=row_offset[curVtx_trans];
                VertexId_CUDA colIdxEnd=row_offset[curVtx_trans+1];
                assert(colIdxStart>=0&&colIdxEnd<=edges);
                if(threadIdx.x==0)
                    blkColStart=colIdxStart;
                if((threadIdx.x+blkRowStart)==(batch_size_-1)||threadIdx.x==(VtxPerBlock)-1)
                    blkColEnd=colIdxEnd;
            }
        acc_h[threadIdx.x]=0.0f;
        __syncthreads();
        totalNumEdges=blkColEnd-blkColStart;
        VertexId_CUDA done=0;
        while(totalNumEdges>0){
            if(tidDiv<totalNumEdges&&tidDiv<VtxPerBlock){
                VertexId_CUDA dst_trans= column_indices[blkColStart+done+tidDiv]-dst_s_;//different with threads num
                VertexId_CUDA src_trans= (VertexId_CUDA)source[blkColStart+done+tidDiv]-src_s_;//different with threads num
                float val=old_feature[dst_trans*feature_size_+tidMod];
                assert(src_trans>=blkRowStart&&src_trans<blkRowStart+VtxPerBlock);
                int offset=(src_trans-blkRowStart)*feature_size_+tidMod;
                atomicAdd(&acc_h[offset],val);
            }
            done+=VtxPerBlock;
           totalNumEdges-=(totalNumEdges>VtxPerBlock) ? VtxPerBlock : totalNumEdges;
        }
        __syncthreads();
        if(tidDiv<VtxPerBlock&&tidDiv+blkRowStart<=batch_size_){
            new_feature[(blkRowStart)*feature_size_+threadIdx.x]=acc_h[threadIdx.x];
        
        }
    }
}

//larger than 512
//backward aggregate from destination
template <typename T_v,typename T_l>
__global__ void aggregate_kernel_from_dst_with_weight(const T_l *row_offset,const  T_l *column_indices,
 		const T_v* old_feature, T_v* new_feature,const T_v* weight,
 		T_l src_s_,T_l dst_s_,
 		T_l batch_size_, T_l feature_size_){
	int large_size=blockDim.x;
	int threadId = blockIdx.x *blockDim.x + threadIdx.x;
       
	for(long i=threadId;i<feature_size_*batch_size_;i+=blockDim.x*gridDim.x){
		T_l local_src=i/feature_size_;
		T_l rank=i%feature_size_;
		for(int i_i=row_offset[local_src];i_i<row_offset[local_src+1];i_i++){
			int local_dst=column_indices[i_i]-dst_s_;
			 atomicAdd(&new_feature[feature_size_*local_src+rank],
			 	old_feature[feature_size_*local_dst+rank]*weight[i_i]);
	 	}
		
	}
}

template <typename T_v,typename T_l>
__global__ void aggregate_kernel_from_dst_without_weight(const T_l *row_offset,const  T_l *column_indices,
 		const T_v* old_feature, T_v* new_feature,const T_v* weight,
 		T_l src_s_,T_l dst_s_,
 		T_l batch_size_, T_l feature_size_){
	int large_size=blockDim.x;
	int threadId = blockIdx.x *blockDim.x + threadIdx.x;
        
	for(long i=threadId;i<feature_size_*batch_size_;i+=blockDim.x*gridDim.x){
		T_l local_src=i/feature_size_;
		T_l rank=i%feature_size_;
		for(int i_i=row_offset[local_src];i_i<row_offset[local_src+1];i_i++){
			int local_dst=column_indices[i_i]-dst_s_;
                         atomicAdd(&new_feature[feature_size_*local_src+rank],
			 	old_feature[feature_size_*local_dst+rank]);
	 	}
	}
}

template <typename T_v,typename T_l>
__global__ void aggregate_kernel_from_dst_tensor_weight(const T_l *row_offset,const  T_l *column_indices,
 		const T_v* old_feature, T_v* new_feature,const T_v* weight,
 		T_l src_s_,T_l dst_s_,
 		T_l batch_size_, T_l feature_size_){
	int large_size=blockDim.x;
	int threadId = blockIdx.x *blockDim.x + threadIdx.x;
        
	for(long i=threadId;i<feature_size_*batch_size_;i+=blockDim.x*gridDim.x){
		T_l local_src=i/feature_size_;
		T_l rank=i%feature_size_;
		for(int i_i=row_offset[local_src];i_i<row_offset[local_src+1];i_i++){
			int local_dst=column_indices[i_i]-dst_s_;
                         atomicAdd(&new_feature[feature_size_*local_src+rank],
			 	old_feature[feature_size_*local_dst+rank]*weight[i_i*feature_size_+rank]);
	 	}
	}
}


//larger than 512 
template <typename T_v,typename T_l>
__global__ void scatter_grad_back_to_weight(const T_l *src, const T_l *dst,
 		const T_v* input, const T_v* output_grad, T_v* weight_grad,
 		T_l src_s_,T_l dst_s_,
 		T_l batch_size_, T_l feature_size_){
       int threadId = blockIdx.x *blockDim.x + threadIdx.x;
       for(long i=threadId;i<feature_size_*batch_size_;i+=blockDim.x*gridDim.x){
            T_l edge_id=i%batch_size_;
            T_l local_src=src[edge_id]-src_s_;
            T_l local_dst=dst[edge_id]-dst_s_;
//            if(local_src<0||local_dst<0)
//                printf("ERROR\n");
            T_l column_id=i%feature_size_;
            atomicAdd(&weight_grad[edge_id],
	        input[local_src*feature_size_+column_id]*
                    output_grad[local_dst*feature_size_+column_id]);
	}
}

template <typename T_v,typename T_l>
__global__ void scatter_grad_back_to_tensor_weight(const T_l *src, const T_l *dst,
 		const T_v* input, const T_v* output_grad, T_v* weight_grad,
 		T_l src_s_,T_l dst_s_,
 		T_l batch_size_, T_l feature_size_){
       int threadId = blockIdx.x *blockDim.x + threadIdx.x;
       for(long i=threadId;i<feature_size_*batch_size_;i+=blockDim.x*gridDim.x){
            T_l edge_id=i%batch_size_;
            T_l local_src=src[edge_id]-src_s_;
            T_l local_dst=dst[edge_id]-dst_s_;
//            if(local_src<0||local_dst<0)
//                printf("ERROR\n");
            T_l column_id=i%feature_size_;
            atomicAdd(&weight_grad[edge_id*feature_size_+column_id],
	        input[local_src*feature_size_+column_id]*
                    output_grad[local_dst*feature_size_+column_id]);
	}
}


template <typename T_v,typename T_l>
__global__ void scatter_grad_back_to_messaage(const T_l *row_indices, const T_l *column_offset,
 		const T_v* input_grad, T_v* message_grad,
 		T_l src_s_,T_l dst_s_,
 		T_l batch_size_, T_l feature_size_){
       int threadId = blockIdx.x *blockDim.x + threadIdx.x;
       for(long i=threadId;i<feature_size_*batch_size_;i+=blockDim.x*gridDim.x){
                T_l local_dst=i/feature_size_;
		T_l rank=i%feature_size_;
		for(T_l i_i=column_offset[local_dst];i_i<column_offset[local_dst+1];i_i++){
                    atomicAdd(&message_grad[feature_size_*i_i+rank],
	            input_grad[feature_size_*local_dst+rank]);
	 	}	
	}
}


__global__ void scatter_grad_back_to_messaage_nts(const VertexId_CUDA *row_indices,
                    const  VertexId_CUDA *column_offset, long *destination,
 		const float* input_grad, float* message_grad,
 		VertexId_CUDA src_s_,VertexId_CUDA src_e_,
                VertexId_CUDA dst_s_,VertexId_CUDA dst_e_,
                VertexId_CUDA edges,
 		VertexId_CUDA batch_size_, VertexId_CUDA feature_size_){
        int VtxPerBlock=CUDA_NUM_THREADS/feature_size_;
        typedef cub::BlockScan<VertexId_CUDA,CUDA_NUM_THREADS> BlockScan;
        __shared__ BlockScan::TempStorage temp_storage;
        __shared__ VertexId_CUDA blkRowStart;
        __shared__ VertexId_CUDA blkRowEnd;
        __shared__ float acc_h[CUDA_NUM_THREADS];
        int tidDiv=threadIdx.x/feature_size_;
        int tidMod=threadIdx.x%feature_size_;
        
        for(VertexId_CUDA blkColStart=blockIdx.x*VtxPerBlock;blkColStart<batch_size_;blkColStart+=VtxPerBlock*gridDim.x){
            VertexId_CUDA myNumEdges=0,scratchOffset,totalNumEdges=0;
            if(threadIdx.x+blkColStart<batch_size_&&threadIdx.x<VtxPerBlock){
                VertexId_CUDA curVtx_trans=blkColStart+threadIdx.x;
                VertexId_CUDA rowIdxStart=column_offset[curVtx_trans];
                VertexId_CUDA rowIdxEnd=column_offset[curVtx_trans+1];
                assert(rowIdxStart>=0&&rowIdxEnd<=edges);
                if(threadIdx.x==0)
                    blkRowStart=rowIdxStart;
                if((threadIdx.x+blkColStart)==(batch_size_-1)||threadIdx.x==(VtxPerBlock)-1)
                    blkRowEnd=rowIdxEnd;
            }
        acc_h[threadIdx.x]=0.0f;
        if(tidDiv<VtxPerBlock&&tidDiv+blkColStart<=batch_size_){
            acc_h[threadIdx.x]=input_grad[(blkColStart)*feature_size_+threadIdx.x];
        }
        __syncthreads();
        totalNumEdges=blkRowEnd-blkRowStart;
        VertexId_CUDA done=0;
        while(totalNumEdges>0){
            if(tidDiv<totalNumEdges&&tidDiv<VtxPerBlock){
                //VertexId_CUDA src_trans= row_indices[blkRowStart+done+tidDiv]-src_s_;//different with threads num
                //VertexId_CUDA dst_trans= (VertexId_CUDA)destination[blkRowStart+done+tidDiv]-dst_s_;//different with threads num
                float val=acc_h[threadIdx.x];
                int offset=(blkRowStart+done+tidDiv)*feature_size_+tidMod;
                atomicAdd(&message_grad[offset],val);
            }
            done+=VtxPerBlock;
           totalNumEdges-=(totalNumEdges>VtxPerBlock) ? VtxPerBlock : totalNumEdges;
        }
    }
}

//smaller than 512

__global__ void aggregate_kernel_from_src_tensor_weight_optim_nts(const VertexId_CUDA *row_indices,
                    const  VertexId_CUDA *column_offset,long*destination,
 		const float* message, float* new_feature,
 		VertexId_CUDA src_s_,VertexId_CUDA src_e_,
                VertexId_CUDA dst_s_,VertexId_CUDA dst_e_,
                VertexId_CUDA edges,
 		VertexId_CUDA batch_size_, VertexId_CUDA feature_size_){
        int VtxPerBlock=CUDA_NUM_THREADS/feature_size_;
        typedef cub::BlockScan<VertexId_CUDA,CUDA_NUM_THREADS> BlockScan;
        __shared__ BlockScan::TempStorage temp_storage;
        __shared__ VertexId_CUDA blkRowStart;
        __shared__ VertexId_CUDA blkRowEnd;
        __shared__ float acc_h[CUDA_NUM_THREADS];
        int tidDiv=threadIdx.x/feature_size_;
        int tidMod=threadIdx.x%feature_size_;
        //block level iteration determnes
        
        for(VertexId_CUDA blkColStart=blockIdx.x*VtxPerBlock;blkColStart<batch_size_;blkColStart+=VtxPerBlock*gridDim.x){
            VertexId_CUDA totalNumEdges=0;
            if(threadIdx.x+blkColStart<batch_size_&&threadIdx.x<VtxPerBlock){
                VertexId_CUDA curVtx_trans=blkColStart+threadIdx.x;
                VertexId_CUDA rowIdxStart=column_offset[curVtx_trans];
                VertexId_CUDA rowIdxEnd=column_offset[curVtx_trans+1];
                assert(rowIdxStart>=0&&rowIdxEnd<=edges);
                if(threadIdx.x==0)
                    blkRowStart=rowIdxStart;
                if((threadIdx.x+blkColStart)==(batch_size_-1)||threadIdx.x==(VtxPerBlock)-1)
                    blkRowEnd=rowIdxEnd;
            }
        acc_h[threadIdx.x]=0.0f;
        __syncthreads();
        totalNumEdges=blkRowEnd-blkRowStart;
        VertexId_CUDA done=0;
        while(totalNumEdges>0){
            if(tidDiv<totalNumEdges&&tidDiv<VtxPerBlock){
                VertexId_CUDA dst_trans= (VertexId_CUDA)destination[blkRowStart+done+tidDiv]-dst_s_;//different with threads num
                float val=message[(blkRowStart+done+tidDiv)*feature_size_+tidMod];
                assert(dst_trans>=blkColStart&&dst_trans<blkColStart+VtxPerBlock);
                int offset=(dst_trans-blkColStart)*feature_size_+tidMod;
                atomicAdd(&acc_h[offset],val);
            }
            done+=VtxPerBlock;
           totalNumEdges-=(totalNumEdges>VtxPerBlock) ? VtxPerBlock : totalNumEdges;
        }
        __syncthreads();
        if(tidDiv<VtxPerBlock&&tidDiv+blkColStart<=batch_size_){
            new_feature[(blkColStart)*feature_size_+threadIdx.x]=acc_h[threadIdx.x];
        
        }
    }
}

//larger than 512
template <typename T_v,typename T_l>
__global__ void aggregate_kernel_from_message_without_weight_sum(const T_l *row_indices,const  T_l *column_offset,
 		const T_v* message, T_v* new_feature,const T_v* weight,
 		T_l src_s_,T_l dst_s_,
 		T_l batch_size_, T_l feature_size_){
	int threadId = blockIdx.x *blockDim.x + threadIdx.x;
        
	for(long i=threadId;i<feature_size_*batch_size_;i+=blockDim.x*gridDim.x){
		T_l local_dst=i/feature_size_;
		T_l rank=i%feature_size_;
		for(int i_i=column_offset[local_dst];i_i<column_offset[local_dst+1];i_i++){
			 atomicAdd(&new_feature[feature_size_*local_dst+rank],
			 	message[feature_size_*i_i+rank]);
	 	}	
	}
}








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

__global__ void aggregate_remote2local(float *result_buffer,float *remote_buffer, int *remote_index, int* src,int* dst,int edge_size,
 		int feature_size,int partition_offset,bool debug=false){
			
	int large_size=blockDim.x;
	int threadId = blockIdx.x *blockDim.x + threadIdx.x;
	for(int i=threadId;i<feature_size*edge_size;i+=blockDim.x*gridDim.x){
            int rank=i%(feature_size);
            int edge_to_process=i/feature_size;
            unsigned int srcV=src[edge_to_process];
            unsigned int dstV=dst[edge_to_process];
            float* actual_data=NULL;
            actual_data=remote_buffer+(feature_size*remote_index[srcV]);
            atomicAdd(&result_buffer[feature_size*(dstV-partition_offset)+rank],actual_data[rank]);		
	}
}

template <typename T_v,typename T_l>
__global__ void process_local_kernel(T_v *result_buffer,T_v *remote_buffer, 
                        T_l* src,T_l* dst, 
                        T_l *remote_index, T_v* weight_buffer, 
                        int partition_offset,int partition_offset_end,int feature_size,int edge_size, bool debug=false){
			
	int large_size=blockDim.x;
 	int threadId = blockIdx.x *blockDim.x + threadIdx.x;
	for(long i=threadId;i<(long)feature_size*edge_size;i+=blockDim.x*gridDim.x){
		unsigned int rank=i%(feature_size);
        unsigned int edge_to_process=i/feature_size;
			// if(threadId==0&&edge_to_process%1000000==0)
			// printf("edge_to_process %d\n",edge_to_process);

             unsigned int srcV=src[edge_to_process];
             unsigned int dstV=dst[edge_to_process];
             float* actual_data=NULL;
			 actual_data=remote_buffer+(feature_size*remote_index[srcV]);
			 if(partition_offset<dstV&&dstV<partition_offset_end)
             atomicAdd(&result_buffer[feature_size*(dstV-partition_offset)+rank],actual_data[rank]*weight_buffer[edge_to_process]);
			 else{
			   float* actual_data_output=remote_buffer+(feature_size*remote_index[srcV]);
			   atomicAdd(&actual_data_output[rank],actual_data[rank]*weight_buffer[edge_to_process]);				
		     } 			
	}
}

template <typename T_v,typename T_l>
__global__ void process_local_kernel_inter(T_v *result_buffer,T_v *remote_buffer, 
                        T_l* src,T_l* dst, 
                        T_l *remote_index, T_l* output_index,  T_v* weight_buffer, 
                        int partition_offset,int partition_offset_end,int feature_size,int edge_size, int out_put_buffer_size,bool debug=false){
			
	int large_size=blockDim.x;
 	int threadId = blockIdx.x *blockDim.x + threadIdx.x;
	for(long i=threadId;i<(long)feature_size*edge_size;i+=blockDim.x*gridDim.x){
		unsigned int rank=i%(feature_size);
        unsigned int edge_to_process=i/feature_size;
			// if(threadId==0&&edge_to_process%1000000==0)
			// printf("edge_to_process %d\n",edge_to_process);

             unsigned int srcV=src[edge_to_process];
             unsigned int dstV=dst[edge_to_process];
             float* actual_data=NULL;
			 actual_data=remote_buffer+(feature_size*remote_index[srcV]);
			 float* actual_data_output=result_buffer+(feature_size*output_index[dstV]);
			//atomicAdd(&actual_data[rank],actual_data[rank]*weight_buffer[edge_to_process]);				
		    atomicAdd(&actual_data_output[rank],actual_data[rank]*weight_buffer[edge_to_process]);		
			 //atomicAdd(&result_buffer[feature_size*(dstV-partition_offset)+rank],actual_data[rank]*weight_buffer[edge_to_process]);	
	}
}


template <typename T_v,typename T_l>
__global__ void process_local_kernel_inter_para(T_v *result_buffer,T_v *remote_buffer, 
                        T_l* src,T_l* dst, 
                        T_l *remote_index, T_l* output_index,  T_v* para, 
                        int partition_offset,int partition_offset_end,int feature_size,int edge_size, int out_put_buffer_size,bool debug=false){
			
	int large_size=blockDim.x;
 	int threadId = blockIdx.x *blockDim.x + threadIdx.x;
	for(long i=threadId;i<(long)feature_size*edge_size;i+=blockDim.x*gridDim.x){
		unsigned int rank=i%(feature_size);
        unsigned int edge_to_process=i/feature_size;
			// if(threadId==0&&edge_to_process%1000000==0)
			// printf("edge_to_process %d\n",edge_to_process);

             unsigned int srcV=src[edge_to_process];
             unsigned int dstV=dst[edge_to_process];
             float* actual_data=NULL;
			 actual_data=remote_buffer+(feature_size*remote_index[srcV]);
			 float* actual_data_output=result_buffer+(feature_size*output_index[dstV]);
			//atomicAdd(&actual_data[rank],actual_data[rank]*weight_buffer[edge_to_process]);
			float result=0;	
			for(int i_f=0;i_f<feature_size;i_f++){			
		    	//atomicAdd(&actual_data_output[rank],actual_data[i_f]*para[i_f*feature_size+rank]);
				atomicAdd(&result,actual_data[i_f]*para[i_f*feature_size+rank]);
			}
			atomicAdd(&actual_data_output[rank],result);		
			 //atomicAdd(&result_buffer[feature_size*(dstV-partition_offset)+rank],actual_data[rank]*weight_buffer[edge_to_process]);	
	}
}


__global__ void aggregate_remote2remote(float *inter_result_buffer,int* inter_result_index, float *remote_buffer, int *remote_index, int* src,int* dst,int edge_size,
 		int feature_size,bool debug=false){
			
	int large_size=blockDim.x;
	int threadId = blockIdx.x *blockDim.x + threadIdx.x;
	for(int i=threadId;i<feature_size*edge_size;i+=blockDim.x*gridDim.x){
            int rank=i%(feature_size);
            int edge_to_process=i/feature_size;
            unsigned int srcV=src[edge_to_process];
            unsigned int dstV=dst[edge_to_process];
            float* actual_data_from=NULL;
            float* actual_data_to=NULL;
            actual_data_from=remote_buffer+(feature_size*remote_index[srcV]);
            actual_data_to=inter_result_buffer+(feature_size*inter_result_index[dstV]);
            atomicAdd(&actual_data_to[rank],actual_data_from[rank]);		
	}
}












template <typename T_v,typename T_l>
__global__ void processing_scalar_weight_shard(const T_l *src,const  T_l *dst,
 		const T_v* old_feature, T_v* new_feature,const T_v* weight,
 		T_l src_s_,T_l dst_s_,
 		T_l edge_size_, T_l feature_size_){
			
//printf("run one batch in GPU\n");
	int large_size=blockDim.x;
	int threadId = blockIdx.x *blockDim.x + threadIdx.x;
	for(int i=threadId;i<feature_size_*edge_size_;i+=blockDim.x*gridDim.x){
		T_l local_dst=dst[i/feature_size_]-dst_s_;
		T_l rank=i%feature_size_;
		T_l local_src=src[i/feature_size_]-src_s_;
		atomicAdd(&new_feature[feature_size_*local_dst+rank],
			old_feature[feature_size_*local_src+rank]*weight[i/feature_size_]);
		
	}
	//printf("finish one batch in GPU\n");
}

template <typename T_v,typename T_l>
__global__ void processing_scalar_weight_one_by_one(T_l *src, T_l *dst, T_v* old_feature, T_v* new_feature,
	T_v* edge_data,int batch_size_,int src_s_,int dst_s_, int feature_size_){
	int large_size=blockDim.x;
	int threadId = blockIdx.x *blockDim.x + threadIdx.x;
	for(int i=threadId;i<feature_size_*batch_size_;i+=blockDim.x*gridDim.x){
		T_l local_dst=dst[i/feature_size_]-dst_s_;
		T_l rank=i%feature_size_;
		T_l local_src=src[i/feature_size_]-src_s_;
		atomicAdd(&new_feature[feature_size_*local_dst+rank],
			old_feature[feature_size_*local_src+rank]);///edge_data[i/feature_size_]);
			//int c=edge_data[i/feature_size_];
		
	}
}

template <typename T_v,typename T_l>
__global__ void processing_no_weight(T_l *src, T_l *dst, T_v* old_feature, T_v* new_feature,T_v* edge_data,
	int batch_size_,int src_s_,int feature_size_){

	int large_size=blockDim.x;
	int threadId = blockIdx.x *blockDim.x + threadIdx.x;
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
__global__ void processing_no_weight_one_by_one(T_l *src, T_l *dst, T_v* old_feature, T_v* new_feature,
	T_v* edge_data,int batch_size_,int src_s_,int dst_s_, int feature_size_){

	int large_size=blockDim.x;
	int threadId = blockIdx.x *blockDim.x + threadIdx.x;
	for(int i=threadId;i<feature_size_*batch_size_;i+=blockDim.x*gridDim.x){
		T_l local_dst=dst[i/feature_size_]-dst_s_;
		T_l rank=i%feature_size_;
		T_l local_src=src[i/feature_size_]-src_s_;
			atomicAdd(&new_feature[feature_size_*local_dst+rank],
				old_feature[feature_size_*local_src+rank]/edge_data[i/feature_size_]);
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
		// if(threadId==0){
		// 	printf("!!!!finished dot_mult\n");
		// }
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
if(threadId==0){
	printf("finished time_mult1\n");
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

template <typename T_v,typename T_l>
__global__ void processing_tensor_weight(T_l *src, T_l *dst, T_v* old_feature, T_v* new_feature,
	T_v* edge_data,int batch_size_,int src_s_,int feature_size_){
	__shared__ int metaInfo[8];

	int large_size=blockDim.x;
	int threadId = blockIdx.x *blockDim.x + threadIdx.x;
	int rank=threadIdx.x;
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
__global__ void processing_parameter_weight_shard_CSC(const T_l *src,const  T_l *dst,
 		const T_v* old_feature, T_v* new_feature,const T_v* para,
 		T_l src_s_,T_l dst_s_,
 		T_l batch_size_, T_l feature_size_){
	int warp_size=blockDim.x;
	int threadId = blockIdx.x *blockDim.x + threadIdx.x;
	for(long i=threadId;i<feature_size_*batch_size_;i+=blockDim.x*gridDim.x){
		T_l local_dst=i/feature_size_;
		T_l rank=i%feature_size_;
		for(int i_i=dst[local_dst];i_i<dst[local_dst+1];i_i++){
			int local_src=src[i_i]-src_s_;
			float sum=0;
			for(int i_f=0;i_f<feature_size_;i_f++){
				atomicAdd(&sum,
			 		old_feature[feature_size_*local_src+rank]*para[i_f*feature_size_+rank]);
				}
			atomicAdd(&new_feature[feature_size_*local_dst+rank],sum);
	 	}	
	}
}

template <typename T_v,typename T_l>
__global__ void processing_scalar_weight(T_l *src, T_l *dst, T_v* old_feature, T_v* new_feature,
	T_v* edge_data,int batch_size_,int src_s_,int feature_size_){
	 int large_size=blockDim.x;
	 int threadId = blockIdx.x *blockDim.x + threadIdx.x;
	for(int i=threadId;i<feature_size_*batch_size_;i+=blockDim.x*gridDim.x){
		T_l local_dst=i/feature_size_;
		T_l rank=i%feature_size_;
	 	for(int i_i=dst[local_dst];i_i<dst[local_dst+1];i_i++){
			int local_src=src[i_i]-src_s_;
			 atomicAdd(&new_feature[feature_size_*local_dst+rank],
			 	old_feature[feature_size_*local_src+rank]/edge_data[i_i]);
	 	}
		
	}
}

template <typename T_v,typename T_l>
__global__ void processing_scalar_weight3(T_l *src, T_l *dst, T_v* old_feature, T_v* new_feature,
	T_v* edge_data,int batch_size_,int src_s_,int feature_size_){
	 int large_size=blockDim.x;
	 int threadId = blockIdx.x *blockDim.x + threadIdx.x;
	for(int i=threadId;i<feature_size_*batch_size_;i+=blockDim.x*gridDim.x){
		T_l local_dst=i/feature_size_;
		T_l rank=i%feature_size_;
		__shared__ T_v  tmp_feature[2048];
		
	 	for(int i_i=dst[local_dst];i_i<dst[local_dst+1];i_i++){
			int local_src=src[i_i]-src_s_;
			 atomicAdd(&tmp_feature[rank],
			 	old_feature[feature_size_*local_src+rank]/edge_data[i_i]);
	 	}__syncthreads();
	 	new_feature[feature_size_*local_dst+rank]=tmp_feature[rank];
	}
}

template <typename T_v,typename T_l>
__global__ void processing_scalar_weight_shard_CSC(const T_l *src,const  T_l *dst,
 		const T_v* old_feature, T_v* new_feature,const T_v* weight,
 		T_l src_s_,T_l dst_s_,
 		T_l batch_size_, T_l feature_size_){
	int large_size=blockDim.x;
	int threadId = blockIdx.x *blockDim.x + threadIdx.x;
        
//        if(threadId==0)
// printf("run one batch in GPU %d\n",feature_size_);

	for(long i=threadId;i<feature_size_*batch_size_;i+=blockDim.x*gridDim.x){
		T_l local_dst=i/feature_size_;
		T_l rank=i%feature_size_;
		for(int i_i=dst[local_dst];i_i<dst[local_dst+1];i_i++){
			int local_src=src[i_i]-src_s_;
			 atomicAdd(&new_feature[feature_size_*local_dst+rank],
			 	old_feature[feature_size_*local_src+rank]*weight[i_i]);
	 	}
		
	}
}

#endif /* PROPAGATE_H_ */

/*
 * propagate.h
 *
 *  Created on: Dec 3, 2019
 *      Author: wangqg
 * TODO :cub support and shared memory optimization
 */

#ifndef NTSCUDADISTKERNEL_CUH
#define NTSCUDADISTKERNEL_CUH

#include"cuda_type.h"
#include<stdlib.h>
#include<stdio.h>
#include<cstdio>
#include<assert.h>
#include <sys/time.h>
#include<cuda.h>
#include"cub/cub.cuh"
#include"math.h"


template <typename T_v,typename T_l>
__global__ void scatter_src_mirror_to_msg(const T_v* message, T_v* src_mirror_feature, 
                const T_l *row_indices, const  T_l *column_offset,
 		const T_l* mirror_index,
// 		T_l src_s_,T_l dst_s_,
 		T_l batch_size_, T_l feature_size_){
	int threadId = blockIdx.x *blockDim.x + threadIdx.x;        
	for(long i=threadId;i<feature_size_*batch_size_;i+=blockDim.x*gridDim.x){
		T_l local_dst=i/feature_size_;
		T_l rank=i%feature_size_;
		for(int i_i=column_offset[local_dst];i_i<column_offset[local_dst+1];i_i++){
                    T_l src=row_indices[i_i];
                    T_l src_index=mirror_index[src];
                    message[feature_size_*i_i+rank]=
                            src_mirror_feature[feature_size_*src_index+rank];
	 	}	
	}
}

template <typename T_v,typename T_l>
__global__ void gather_msg_to_src_mirror(const T_v* src_mirror_feature, T_v* message,
                const T_l *row_indices,const  T_l *column_offset,
 		const T_l* mirror_index,
// 		T_l src_s_,T_l dst_s_,
 		T_l batch_size_, T_l feature_size_){
	int threadId = blockIdx.x *blockDim.x + threadIdx.x;        
	for(long i=threadId;i<feature_size_*batch_size_;i+=blockDim.x*gridDim.x){
            T_l local_dst=i/feature_size_;
            T_l rank=i%feature_size_;
            for(int i_i=column_offset[local_dst];i_i<column_offset[local_dst+1];i_i++){
                T_l src=row_indices[i_i];
                T_l src_index=mirror_index[src];
                atomicAdd(&src_mirror_feature[feature_size_*src_index+rank],
                        message[feature_size_*i_i+rank]);
            }	
	}
}

template <typename T_v,typename T_l>
__global__ void scatter_dst_to_msg(const T_v* message, T_v* dst_feature,
                const T_l *row_indices,const  T_l *column_offset,
// 		T_l src_s_,T_l dst_s_,
 		T_l batch_size_, T_l feature_size_){
	int threadId = blockIdx.x *blockDim.x + threadIdx.x;        
	for(long i=threadId;i<feature_size_*batch_size_;i+=blockDim.x*gridDim.x){
            T_l local_dst=i/feature_size_;
            T_l rank=i%feature_size_;
            for(int i_i=column_offset[local_dst];i_i<column_offset[local_dst+1];i_i++){
                message[feature_size_*i_i+rank]=
                        dst_feature[feature_size_*local_dst+rank];
            }	
	}
}

template <typename T_v,typename T_l>
__global__ void gather_msg_to_dst(const T_v* dst_feature, T_v* message,
                const T_l *row_indices,const  T_l *column_offset,
 		const T_l* mirror_index,
// 		T_l src_s_,T_l dst_s_,
 		T_l batch_size_, T_l feature_size_){
	int threadId = blockIdx.x *blockDim.x + threadIdx.x;        
	for(long i=threadId;i<feature_size_*batch_size_;i+=blockDim.x*gridDim.x){
            T_l local_dst=i/feature_size_;
            T_l rank=i%feature_size_;
            for(int i_i=column_offset[local_dst];i_i<column_offset[local_dst+1];i_i++){
                atomicAdd(&dst_feature[feature_size_*local_dst+rank],
                        message[feature_size_*i_i+rank]);
            }	
	}
}




template <typename T_v,typename T_l>
__global__ void edge_softmax_forward_block(const T_v* msg_output, const T_v* msg_input,
                const T_v* msg_cached, const T_l *row_indices,const  T_l *column_offset,
 		const T_l* mirror_index,
// 		T_l src_s_,T_l dst_s_,
 		T_l batch_size_, T_l feature_size_){
        int VtxPerBlock=1;
//        const int WARP_SIZE=32;
//        typedef cub::BlockScan<VertexId_CUDA,CUDA_NUM_THREADS> BlockScan;
        //typedef cub::WarpReduce<T_v> WarpReduce;
        //__shared__ typename WarpReduce::TempStorage temp_storage;
        typedef cub::BlockReduce<T_v, CUDA_NUM_THREADS> BlockReduce;
        __shared__ typename BlockReduce::TempStorage temp_storage;
//        __shared__ BlockScan::TempStorage temp_storage;
        __shared__ VertexId_CUDA blkRowStart;
        __shared__ VertexId_CUDA blkRowEnd;
        int tidDiv=threadIdx.x/feature_size_;
        int tidMod=threadIdx.x%feature_size_;
        
        for(VertexId_CUDA blkColStart=blockIdx.x*VtxPerBlock;blkColStart<batch_size_;blkColStart+=VtxPerBlock*gridDim.x){
            VertexId_CUDA myNumEdges=0,scratchOffset,totalNumEdges=0;
            VertexId_CUDA curVtx_trans=blkColStart+threadIdx.x/CUDA_NUM_THREADS;
            VertexId_CUDA rowIdxStart=column_offset[curVtx_trans];
            VertexId_CUDA rowIdxEnd=column_offset[curVtx_trans+1];               
            __syncthreads();
            T_v thread_data=0.0;
            int rest=0;
            T_v aggregate=0;
            for(VertexId_CUDA eid=rowIdxStart+threadIdx.x;eid<rowIdxEnd;eid+=CUDA_NUM_THREADS){       
                int valid_items=rowIdxEnd-rowIdxStart-CUDA_NUM_THREADS*rest;
                thread_data=exp(msg_input[eid]);
                aggregate+=BlockReduce(temp_storage).Sum(thread_data,valid_items);
//                msg_output[rowIdxStart+threadIdx.x%WARP_SIZE]
//                        =msg_input[rowIdxStart+threadIdx.x%WARP_SIZE];
                rest+=1;
            }
            __syncthreads();
            for(VertexId_CUDA eid=rowIdxStart+threadIdx.x;eid<rowIdxEnd;eid+=CUDA_NUM_THREADS){       
                msg_output[eid]=exp(msg_input[eid])/aggregate;
                msg_cached[eid]=msg_output;
            }
    }      
}


template <typename T_v,typename T_l>
__global__ void edge_softmax_backward_block(const T_v* msg_input_grad, const T_v* msg_output_grad,
                const T_v* msg_cached, const T_l *row_indices,const  T_l *column_offset,
 		const T_l* mirror_index,
// 		T_l src_s_,T_l dst_s_,
 		T_l batch_size_, T_l feature_size_){
        int VtxPerBlock=1;
//        const int WARP_SIZE=32;
//        typedef cub::BlockScan<VertexId_CUDA,CUDA_NUM_THREADS> BlockScan;
        //typedef cub::WarpReduce<T_v> WarpReduce;
        //__shared__ typename WarpReduce::TempStorage temp_storage;
        typedef cub::BlockReduce<T_v, CUDA_NUM_THREADS> BlockReduce;
        __shared__ typename BlockReduce::TempStorage temp_storage;
//        __shared__ BlockScan::TempStorage temp_storage;
        __shared__ VertexId_CUDA blkRowStart;
        __shared__ VertexId_CUDA blkRowEnd;
        int tidDiv=threadIdx.x/feature_size_;
        int tidMod=threadIdx.x%feature_size_;
        
        for(VertexId_CUDA blkColStart=blockIdx.x*VtxPerBlock;blkColStart<batch_size_;blkColStart+=VtxPerBlock*gridDim.x){
            VertexId_CUDA myNumEdges=0,scratchOffset,totalNumEdges=0;
            VertexId_CUDA curVtx_trans=blkColStart+threadIdx.x/CUDA_NUM_THREADS;
            VertexId_CUDA rowIdxStart=column_offset[curVtx_trans];
            VertexId_CUDA rowIdxEnd=column_offset[curVtx_trans+1];               
            __syncthreads();
            T_v thread_data=0.0;
            int rest=0;
            T_v aggregate=0;
            for(VertexId_CUDA eid=rowIdxStart+threadIdx.x;eid<rowIdxEnd;eid+=CUDA_NUM_THREADS){       
                int valid_items=rowIdxEnd-rowIdxStart-CUDA_NUM_THREADS*rest;
                thread_data=msg_output_grad[eid]*msg_cached[eid];
                aggregate+=BlockReduce(temp_storage).Sum(thread_data,valid_items);
//                msg_output[rowIdxStart+threadIdx.x%WARP_SIZE]
//                        =msg_input[rowIdxStart+threadIdx.x%WARP_SIZE];
                rest+=1;
            }
            __syncthreads();
            for(VertexId_CUDA eid=rowIdxStart+threadIdx.x;eid<rowIdxEnd;eid+=CUDA_NUM_THREADS){       
                msg_input_grad[eid]=msg_output_grad[eid]*msg_cached[eid]+aggregate*msg_cached[eid];
            }
    }      
}


















#endif /* PROPAGATE_H_ */

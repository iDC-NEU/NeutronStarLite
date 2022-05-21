/*
 * propagate.h
 *
 *  Created on: Dec 3, 2019
 *      Author: wangqg
 * TODO :cub support and shared memory optimization
 */

#ifndef PROPAGATE_H_
#define PROPAGATE_H_

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


//UnTested
//for dimension smaller than 512;
//with low performance 
//__global__ void aggregate_kernel_from_src_with_weight_optim_ROC(const VertexId_CUDA *row_indices,
//                    const  VertexId_CUDA *column_offset,long destination,
// 		const float* old_feature, float* new_feature,const float* weight,
// 		VertexId_CUDA src_s_,VertexId_CUDA src_e_,
//                VertexId_CUDA dst_s_,VertexId_CUDA dst_e_,
//                VertexId_CUDA edges,
// 		VertexId_CUDA batch_size_, VertexId_CUDA feature_size_){
//        int VtxPerBlock=CUDA_NUM_THREADS/feature_size_;
//        typedef cub::BlockScan<VertexId_CUDA,CUDA_NUM_THREADS> BlockScan;
//        __shared__ BlockScan::TempStorage temp_storage;
//        __shared__ VertexId_CUDA blkRowStart;
//        __shared__ float acc_h[CUDA_NUM_THREADS];
//        int tidDiv=threadIdx.x/feature_size_;
//        int tidMod=threadIdx.x%feature_size_;
//        //block level iteration determnes
//        
//        for(VertexId_CUDA blkColStart=blockIdx.x*VtxPerBlock;blkColStart<batch_size_;blkColStart+=VtxPerBlock*gridDim.x){
//            VertexId_CUDA myNumEdges=0,scratchOffset,totalNumEdges=0;
//            VertexId_CUDA localDst=0;
//            if(threadIdx.x+blkColStart<batch_size_&&threadIdx.x<VtxPerBlock){
//                VertexId_CUDA curVtx_trans=threadIdx.x+blkColStart;
//                VertexId_CUDA rowIdxStart=column_offset[curVtx_trans];
//                VertexId_CUDA rowIdxEnd=column_offset[curVtx_trans+1];
//                assert(rowIdxStart>=0&&rowIdxEnd<=edges);
//                myNumEdges=rowIdxEnd-rowIdxStart;
//                if(threadIdx.x==0)
//                    blkRowStart=rowIdxStart;
//            }
//        acc_h[threadIdx.x]=0.0f;
//        __syncthreads();
//        BlockScan(temp_storage).ExclusiveSum(myNumEdges, scratchOffset, totalNumEdges);
//        VertexId_CUDA done=0;
//        while(totalNumEdges>0){
//            if(tidDiv<totalNumEdges&&tidDiv<VtxPerBlock){
//                VertexId_CUDA src_trans= row_indices[blkRowStart+done+tidDiv]-src_s_;//different with threads num
//                VertexId_CUDA dst_trans= (VertexId_CUDA)destination[blkRowStart+done+tidDiv]-dst_s_;//different with threads num
//                float w=weight[blkRowStart+done+tidDiv];
//                float val=old_feature[src_trans*feature_size_+tidMod]*w;
//                //assert(dst_trans>=blkColStart&&dst_trans<blkColStart+VtxPerBlock);
//                int offset=(dst_trans-blkColStart)*feature_size_+tidMod;
//                atomicAdd(&acc_h[offset],val);
//            }
//            done+=VtxPerBlock;
//           totalNumEdges-=(totalNumEdges>VtxPerBlock) ? VtxPerBlock : totalNumEdges;
//        }
//        __syncthreads();
//        if(tidDiv<VtxPerBlock&&tidDiv+blkColStart<=batch_size_){
//            new_feature[(blkColStart)*feature_size_+threadIdx.x]=acc_h[threadIdx.x];
//        
//        }
//    }
//}

__global__ void aggregate_kernel_from_src_with_weight_optim_load_imbalance(const VertexId_CUDA *row_indices,
                    const  VertexId_CUDA *column_offset,
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
                    const  VertexId_CUDA *column_offset,
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
                    const  VertexId_CUDA *column_offset, 
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
                    const  VertexId_CUDA *column_indices,
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

__global__ void aggregate_kernel_from_dst_without_weight_optim_nts(const VertexId_CUDA *row_offset,
                    const  VertexId_CUDA *column_indices,
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
                float val=old_feature[dst_trans*feature_size_+tidMod];
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


//larger than 512
template <typename T_v,typename T_l>
__global__ void aggregate_kernel_from_message_without_weight_sum(const T_l *row_indices,const  T_l *column_offset,
 		const T_v* message, T_v* new_feature,
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



template <typename T_v,typename T_l>
__global__ void scatter_src_mirror_to_msg(const T_v* message, T_v* src_mirror_feature, 
                const T_l *row_indices, const  T_l *column_offset,
 		const T_l* mirror_index,
 		T_l src_s_,T_l dst_s_,
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
 		T_l src_s_,T_l dst_s_,
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
 		T_l src_s_,T_l dst_s_,
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
__global__ void edge_softmax_forward_block(const T_v* msg_output, const T_v* msg_input,
                const T_v* msg_cached, const T_l *row_indices,const  T_l *column_offset,
 		const T_l* mirror_index,
 		T_l src_s_,T_l dst_s_,
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
 		T_l src_s_,T_l dst_s_,
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

//template <typename T_v,typename T_l>
//__global__ void edge_softmax_backward(const T_v* msg_output, const T_v* msg_input,
//                const T_v* msg_intermediate, const T_l *row_indices,const  T_l *column_offset,
// 		const T_l* mirror_index,
// 		T_l src_s_,T_l dst_s_,
// 		T_l batch_size_, T_l feature_size_){
//	int threadId = blockIdx.x *blockDim.x + threadIdx.x;        
//	for(long i=threadId;i<feature_size_*batch_size_;i+=blockDim.x*gridDim.x){
//            T_l local_dst=i/feature_size_;
//            T_l rank=i%feature_size_;
//            for(int i_i=column_offset[local_dst];i_i<column_offset[local_dst+1];i_i++){
//                atomicAdd(&dst_feature[feature_size_*local_dst+rank],
//                        message[feature_size_*i_i+rank]);
//            }	
//	}
//}




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

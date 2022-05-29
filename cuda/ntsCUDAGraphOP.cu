
#include"cuda_type.h"
#include "ntsCUDA.hpp"

#if CUDA_ENABLE
#include "ntsCUDAFuseKernel.cuh"
#include "ntsCUDADistKernel.cuh"
#include "ntsCUDATransferKernel.cuh"

#endif

#if CUDA_ENABLE
#define CHECK_CUDA_RESULT(N) {											\
	cudaError_t result = N;												\
	if (result != 0) {													\
		printf("CUDA call on line %d returned error %d\n", __LINE__,	\
			result);													\
		exit(1);														\
	} }
#endif


void* getDevicePointer(void* host_data_to_device){
#if CUDA_ENABLE
    void* dev_host_data_to_device;
    CHECK_CUDA_RESULT(cudaHostGetDevicePointer(&dev_host_data_to_device,host_data_to_device,0));
    return dev_host_data_to_device;
#else
    printf("CUDA DISABLED getDevicePointer\n");
    exit(0);   
#endif 

}

void* cudaMallocPinned(long size_of_bytes){

#if CUDA_ENABLE       
    void *data=NULL;
   CHECK_CUDA_RESULT(cudaHostAlloc(&data,size_of_bytes, cudaHostAllocMapped));
    return data;
#else
    printf("CUDA DISABLED cudaMallocPinned\n");
    exit(0);   
#endif
}

void* cudaMallocGPU(long size_of_bytes){
#if CUDA_ENABLE
       void *data=NULL;
       CHECK_CUDA_RESULT(cudaMalloc(&data,size_of_bytes));
//       printf("malloc finished\n");
       return data;
#else
       printf("CUDA DISABLED cudaMallocGPU\n");
       exit(0);   
#endif  
}


Cuda_Stream::Cuda_Stream(){
#if CUDA_ENABLE
       CHECK_CUDA_RESULT(cudaStreamCreate(&stream));
#else
       printf("CUDA DISABLED Cuda_Stream::Cuda_Stream\n");
       exit(0);  
#endif  
}

void Cuda_Stream::destory_Stream(){
#if CUDA_ENABLE
    CHECK_CUDA_RESULT(cudaStreamDestroy(stream));
#else
       printf("CUDA DISABLED Cuda_Stream::Cuda_Stream\n");
       exit(0);   
#endif     

}
inline cudaStream_t Cuda_Stream::getStream(){
    
#if CUDA_ENABLE
        return stream;
#else
       printf("CUDA DISABLED Cuda_Stream::getStream\n");
       exit(0);   
#endif   
}

void ResetDevice(){
#if CUDA_ENABLE
   cudaDeviceReset();
#else
       printf("CUDA DISABLED ResetDevice\n");
       exit(0);   
#endif   
 
}
void Cuda_Stream::CUDA_DEVICE_SYNCHRONIZE(){
#if CUDA_ENABLE
       cudaStreamSynchronize(stream);
#else
       printf("CUDA DISABLED Cuda_Stream::CUDA_DEVICE_SYNCHRONIZE\n");
       exit(0);   
#endif   
}

void Cuda_Stream::move_result_out(float* output,float* input, VertexId_CUDA src,VertexId_CUDA dst, int feature_size,bool sync){
#if CUDA_ENABLE
    CHECK_CUDA_RESULT(cudaMemcpyAsync(output,input,((long)(dst-src))*feature_size*(sizeof(int)), cudaMemcpyDeviceToHost,stream));
#else
       printf("CUDA DISABLED Cuda_Stream::move_result_out\n");
       exit(0);   
#endif   
}
void Cuda_Stream::move_data_in(float* d_pointer,float* h_pointer, VertexId_CUDA start, VertexId_CUDA end, int feature_size,bool sync){
#if CUDA_ENABLE
    CHECK_CUDA_RESULT(cudaMemcpyAsync(d_pointer,h_pointer,((long)(end-start))*feature_size*(sizeof(float)), cudaMemcpyHostToDevice,stream));
#else
       printf("CUDA DISABLED Cuda_Stream::move_data_in\n");
       exit(0);   
#endif   
  
}
void Cuda_Stream::move_edge_in(VertexId_CUDA* d_pointer,VertexId_CUDA* h_pointer, VertexId_CUDA start, VertexId_CUDA end, int feature_size,bool sync){
#if CUDA_ENABLE
    CHECK_CUDA_RESULT(cudaMemcpyAsync(d_pointer,h_pointer,((long)(end-start))*feature_size*(sizeof(VertexId_CUDA)), cudaMemcpyHostToDevice,stream));
#else
       printf("CUDA DISABLED Cuda_Stream::move_edge_in\n");
       exit(0);   
#endif       
}
void Cuda_Stream::aggregate_comm_result(float* aggregate_buffer,float *input_buffer,VertexId_CUDA data_size,int feature_size,int partition_offset, bool sync){
#if CUDA_ENABLE
    aggregate_data_buffer<<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>(aggregate_buffer,input_buffer,data_size,feature_size,partition_offset,sync);
#else
       printf("CUDA DISABLED Cuda_Stream::aggregate_comm_result\n");
       exit(0);   
#endif     
}

void Cuda_Stream::aggregate_comm_result_debug(float* aggregate_buffer,float *input_buffer,VertexId_CUDA data_size,VertexId_CUDA feature_size,VertexId_CUDA partition_start,VertexId_CUDA partition_end, bool sync){
#if CUDA_ENABLE
    aggregate_data_buffer_debug<<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>(aggregate_buffer,input_buffer,data_size,feature_size,partition_start,partition_end,sync);
#else
       printf("CUDA DISABLED Cuda_Stream::aggregate_comm_result_debug\n");
       exit(0);   
#endif 
}

void Cuda_Stream::deSerializeToGPU(float* input_gpu_buffer,float *input_buffer,VertexId_CUDA data_size,VertexId_CUDA feature_size,VertexId_CUDA partition_start,VertexId_CUDA partition_end, bool sync){
#if CUDA_ENABLE
    deSerializeToGPUkernel<<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>(input_gpu_buffer,input_buffer,data_size,feature_size,partition_start,partition_end,sync);
#else
       printf("CUDA DISABLED Cuda_Stream::deSerializeToGPU\n");
       exit(0);   
#endif  
}
void Cuda_Stream::Gather_By_Dst_From_Src(float* input,float* output,float* weight_forward,//data 
        VertexId_CUDA* row_indices,VertexId_CUDA *column_offset,//graph
        VertexId_CUDA src_start, VertexId_CUDA src_end,
        VertexId_CUDA dst_start, VertexId_CUDA dst_end,
	VertexId_CUDA edges,VertexId_CUDA batch_size,
        VertexId_CUDA feature_size,bool with_weight,bool tensor_weight){
#if CUDA_ENABLE
        if(with_weight){
            if(tensor_weight){
//		aggregate_kernel_from_src_tensor_weight<float,VertexId_CUDA><<<BLOCK_SIZE,THREAD_SIZE,0,stream>>>(
//			row_indices, column_offset, input, output, weight_forward, 
//				src_start, dst_start, batch_size, feature_size);
                printf("aggregate_kernel_from_src_tensor_weight_optim_nts");
            }else{
                aggregate_kernel_from_src_with_weight<float,VertexId_CUDA><<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>(
			row_indices, column_offset, input, output, weight_forward, 
				src_start, dst_start, batch_size, feature_size);
            }
        }
        else{
                aggregate_kernel_from_src_without_weight<float,VertexId_CUDA><<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>(
                        row_indices, column_offset, input, output, weight_forward, 
                                src_start, dst_start, batch_size, feature_size);
        }
#else
       printf("CUDA DISABLED Cuda_Stream::Gather_By_Dst_From_Src\n");
       exit(0);   
#endif  
        
}
void Cuda_Stream::Gather_By_Dst_From_Src_Optim(float* input,float* output,float* weight_forward,//data 
        VertexId_CUDA* row_indices,VertexId_CUDA *column_offset,
        VertexId_CUDA src_start, VertexId_CUDA src_end,
        VertexId_CUDA dst_start, VertexId_CUDA dst_end,
	VertexId_CUDA edges,VertexId_CUDA batch_size,
        VertexId_CUDA feature_size,bool with_weight,bool tensor_weight){
#if CUDA_ENABLE
            if(with_weight){
            if(tensor_weight){
//		aggregate_kernel_from_src_tensor_weight_optim_nts<<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>(
//			row_indices, column_offset, input, output, weight_forward, 
//				src_start,src_end, dst_start,dst_end,edges, batch_size, feature_size);
                printf("aggregate_kernel_from_src_tensor_weight_optim_nts is a legacy implementation\n");
                exit(0);  
            }else{
                aggregate_kernel_from_src_with_weight_optim_nts<<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>(
			row_indices, column_offset, input, output, weight_forward, 
				src_start,src_end, dst_start,dst_end,edges, batch_size, feature_size);
            }
        }
        else{
                aggregate_kernel_from_src_without_weight_optim_nts<<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>(
			row_indices, column_offset, input, output, weight_forward, 
				src_start,src_end, dst_start,dst_end,edges, batch_size, feature_size);
        }
#else
       printf("CUDA DISABLED Cuda_Stream::Gather_By_Dst_From_Src_Optim\n");
       exit(0);   
#endif      

    
}

void Cuda_Stream::Gather_By_Src_From_Dst_Optim(float* input,float* output,float* weight_forward,//data  
        VertexId_CUDA* row_offset,VertexId_CUDA *column_indices,
        VertexId_CUDA src_start, VertexId_CUDA src_end,
        VertexId_CUDA dst_start, VertexId_CUDA dst_end,
	VertexId_CUDA edges,VertexId_CUDA batch_size,
        VertexId_CUDA feature_size,bool with_weight,bool tensor_weight){
#if CUDA_ENABLE
        if(with_weight){
            if(tensor_weight){
//		aggregate_kernel_from_dst_tensor_weight_optim_nts<<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>(
//			row_offset, column_indices, input, output, weight_forward, 
//				src_start,src_end, dst_start,dst_end,edges, batch_size, feature_size);
                printf("aggregate_kernel_from_dst_tensor_weight_optim_nts is a legacy implementation\n");
                exit(0);  
            }else{
                aggregate_kernel_from_dst_with_weight_optim_nts<<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>(
			row_offset, column_indices, input, output, weight_forward, 
				src_start,src_end, dst_start,dst_end,edges, batch_size, feature_size);
            }
        }
        else{
                aggregate_kernel_from_dst_without_weight_optim_nts<<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>(
			row_offset, column_indices, input, output, weight_forward, 
				src_start,src_end, dst_start,dst_end,edges, batch_size, feature_size);
        }
#else
       printf("CUDA DISABLED Gather_By_Src_From_Dst_Optim\n");
       exit(0);   
#endif     
}


void Cuda_Stream::Gather_By_Src_From_Dst(float* input,float* output,float* weight_forward,//data 
        VertexId_CUDA* row_offset,VertexId_CUDA *column_indices,//graph
        VertexId_CUDA src_start, VertexId_CUDA src_end,
        VertexId_CUDA dst_start, VertexId_CUDA dst_end,
	VertexId_CUDA edges,VertexId_CUDA batch_size,
        VertexId_CUDA feature_size,bool with_weight,bool tensor_weight){
#if CUDA_ENABLE
	//printf("CUDA_DEBUGE_INFO:FORWARD RUN_SYNC with \t BLOCK_SIZE:%d\tTHREAD_SIZE:%d\n",BLOCK_SIZE,THREAD_SIZE); 
        if(with_weight){
            if(tensor_weight){
             printf("aggregate_kernel_from_dst_tensor_weight is a legacy implementation\n");
                exit(0);   
            }else{
                
		aggregate_kernel_from_dst_with_weight<float,VertexId_CUDA><<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>(
			row_offset, column_indices, input, output, weight_forward, 
				src_start, dst_start, batch_size, feature_size);   
            }
        }
        else{
                aggregate_kernel_from_dst_without_weight<float,VertexId_CUDA><<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>(
			row_offset, column_indices, input, output, weight_forward, 
				src_start, dst_start, batch_size, feature_size);
        }
#else
       printf("CUDA DISABLED Cuda_Stream::Gather_By_Src_From_Dst\n");
       exit(0);   
#endif 

}

void Cuda_Stream::Scatter_Grad_Back_To_Message(float* input,float* message_grad,//data 
        VertexId_CUDA* row_indices,VertexId_CUDA *column_offset,
        VertexId_CUDA src_start, VertexId_CUDA src_end,
        VertexId_CUDA dst_start, VertexId_CUDA dst_end,
	VertexId_CUDA edges,VertexId_CUDA batch_size,
        VertexId_CUDA feature_size,bool with_weight){
#if CUDA_ENABLE
	//printf("CUDA_DEBUGE_INFO:FORWARD RUN_SYNC with \t BLOCK_SIZE:%d\tTHREAD_SIZE:%d\n",BLOCK_SIZE,THREAD_SIZE); 
        if(with_weight){
            printf("tensor_weight Scatter_Grad_Back_To_Weight not implemented\n");
            exit(0);
        }else{
            scatter_grad_back_to_messaage<float,VertexId_CUDA><<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>(
			row_indices, column_offset, input, message_grad, 
				src_start, dst_start, batch_size, feature_size);
        }
#else
       printf("CUDA DISABLED Cuda_Stream::Scatter_Grad_Back_To_Message\n");
       exit(0);   
#endif 


}

void Cuda_Stream::Scatter_Src_Mirror_to_Msg(float* message,float* src_mirror_feature,//data 
        VertexId_CUDA* row_indices,VertexId_CUDA *column_offset,
        VertexId_CUDA* mirror_index, VertexId_CUDA batch_size,
        VertexId_CUDA feature_size){
#if CUDA_ENABLE
        scatter_src_mirror_to_msg<float,VertexId_CUDA><<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>(
            message, src_mirror_feature, row_indices, column_offset, mirror_index,
                batch_size, feature_size); 
#else
       printf("CUDA DISABLED Cuda_Stream::Scatter_Src_Mirror_to_Msg\n");
       exit(0);   
#endif
        
}

void Cuda_Stream::Gather_Msg_To_Src_Mirror(float* src_mirror_feature,float* message,//data 
        VertexId_CUDA* row_indices,VertexId_CUDA *column_offset,
        VertexId_CUDA* mirror_index, VertexId_CUDA batch_size,
        VertexId_CUDA feature_size){
#if CUDA_ENABLE
	//printf("CUDA_DEBUGE_INFO:FORWARD RUN_SYNC with \t BLOCK_SIZE:%d\tfeature_size:%d\n",BLOCK_SIZE,feature_size); 
        gather_msg_to_src_mirror<float,VertexId_CUDA><<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>(
            src_mirror_feature, message, row_indices, column_offset, mirror_index,
                batch_size, feature_size); 
#else
       printf("CUDA DISABLED Cuda_Stream::Gather_Msg_To_Src_Mirror\n");
       exit(0);   
#endif
        
}

void Cuda_Stream::Scatter_Dst_to_Msg(float* message,float* dst_feature,//data 
        VertexId_CUDA* row_indices, VertexId_CUDA *column_offset,
        VertexId_CUDA batch_size, VertexId_CUDA feature_size){
#if CUDA_ENABLE
	//printf("CUDA_DEBUGE_INFO:FORWARD RUN_SYNC with \t BLOCK_SIZE:%d\tfeature_size:%d\n",BLOCK_SIZE,feature_size); 
        scatter_dst_to_msg<float,VertexId_CUDA><<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>(
            message, dst_feature, row_indices, column_offset,
            batch_size, feature_size); 
#else
       printf("CUDA DISABLED Cuda_Stream::Scatter_Dst_to_Msg\n");
       exit(0);   
#endif      
}

void Cuda_Stream::Gather_Msg_to_Dst(float* dst_feature,float* message,//data 
        VertexId_CUDA* row_indices, VertexId_CUDA *column_offset,
        VertexId_CUDA batch_size, VertexId_CUDA feature_size){
#if CUDA_ENABLE
	//printf("CUDA_DEBUGE_INFO:FORWARD RUN_SYNC with \t BLOCK_SIZE:%d\tfeature_size:%d\n",BLOCK_SIZE,feature_size); 
        gather_msg_to_dst<float,VertexId_CUDA><<<CUDA_NUM_BLOCKS,CUDA_NUM_THREADS,0,stream>>>(
            dst_feature, message, row_indices, column_offset,
            batch_size, feature_size); 
#else
       printf("CUDA DISABLED Cuda_Stream::Gather_Msg_to_Dst\n");
       exit(0);   
#endif      
}

void Cuda_Stream::Edge_Softmax_Forward_Block(float* msg_output,float* msg_input,//data 
        float* msg_cached,
        VertexId_CUDA* row_indices, VertexId_CUDA *column_offset,
        VertexId_CUDA batch_size, VertexId_CUDA feature_size){
#if CUDA_ENABLE 
        edge_softmax_forward_block<float,VertexId_CUDA><<<CUDA_NUM_BLOCKS_SOFTMAX,CUDA_NUM_THREADS_SOFTMAX,0,stream>>>(
            msg_output, msg_input, msg_cached, row_indices, column_offset,
            batch_size, feature_size); 
#else
       printf("CUDA DISABLED Cuda_Stream::Edge_Softmax_Forward_Block\n");
       exit(0);   
#endif      
}

void Cuda_Stream::Edge_Softmax_Backward_Block(float* msg_input_grad,float* msg_output_grad,//data 
        float* msg_cached,
        VertexId_CUDA* row_indices, VertexId_CUDA *column_offset,
        VertexId_CUDA batch_size, VertexId_CUDA feature_size){
#if CUDA_ENABLE
	//printf("CUDA_DEBUGE_INFO:FORWARD RUN_SYNC with \t BLOCK_SIZE:%d\tfeature_size:%d\n",BLOCK_SIZE,feature_size); 
        edge_softmax_backward_block<float,VertexId_CUDA><<<CUDA_NUM_BLOCKS_SOFTMAX,CUDA_NUM_THREADS_SOFTMAX,0,stream>>>(
            msg_input_grad, msg_output_grad, msg_cached, row_indices, column_offset,
            batch_size, feature_size); 
#else
       printf("CUDA DISABLED Cuda_Stream::Edge_Softmax_Backward_Block\n");
       exit(0);   
#endif      
}



















void move_result_out(float* output,float* input, int src,int dst, int feature_size, bool sync){
#if CUDA_ENABLE
    CHECK_CUDA_RESULT(cudaMemcpy(output,input,((long)(dst-src))*feature_size*(sizeof(int)), cudaMemcpyDeviceToHost));
    if(sync)
    cudaDeviceSynchronize();
#else
       printf("CUDA DISABLED Cuda_Stream::Gather_By_Dst_From_Message\n");
       exit(0);   
#endif 


}

void move_data_in(float* d_pointer,float* h_pointer, int start, int end, int feature_size, bool sync){
#if CUDA_ENABLE    
    CHECK_CUDA_RESULT(cudaMemcpy(d_pointer,h_pointer,((long)(end-start))*feature_size*(sizeof(float)), cudaMemcpyHostToDevice));
    if(sync)
    cudaDeviceSynchronize();
#else
       printf("CUDA DISABLED Cuda_Stream::Gather_By_Dst_From_Message\n");
       exit(0);   
#endif 
}

void move_edge_in(VertexId_CUDA * d_pointer,VertexId_CUDA* h_pointer, VertexId_CUDA start, VertexId_CUDA end, int feature_size, bool sync){
#if CUDA_ENABLE    
    CHECK_CUDA_RESULT(cudaMemcpy(d_pointer,h_pointer,((long)(end-start))*feature_size*(sizeof(VertexId_CUDA)), cudaMemcpyHostToDevice));
    if(sync)
    cudaDeviceSynchronize();
#else
       printf("CUDA DISABLED move_edge_in\n");
       exit(0);   
#endif 
}
void move_bytes_in(void * d_pointer,void* h_pointer, long bytes, bool sync){
#if CUDA_ENABLE
    CHECK_CUDA_RESULT(cudaMemcpy(d_pointer,h_pointer,bytes, cudaMemcpyHostToDevice));
    if(sync)
    cudaDeviceSynchronize();
#else
       printf("CUDA DISABLED move_bytes_in\n");
       exit(0);   
#endif 
}


//void aggregate_comm_result(float* aggregate_buffer,float *input_buffer,int data_size,int feature_size,int partition_offset,bool sync){
//#if CUDA_ENABLE
//    const int THREAD_SIZE=512;//getThreadNum(_meta->get_feature_size());
//    const int BLOCK_SIZE=32;
//    aggregate_data_buffer<<<THREAD_SIZE,BLOCK_SIZE>>>(aggregate_buffer,input_buffer,data_size,feature_size,partition_offset,sync);
//    if(sync)
//    	cudaDeviceSynchronize();
//#else
//       printf("CUDA DISABLED aggregate_comm_result\n");
//       exit(0);   
//#endif 
//
//}

void ntsFreeHost(void *buffer){
#if CUDA_ENABLE    
    cudaFreeHost(buffer);
#else
       printf("CUDA DISABLED FreeBuffer\n");
       exit(0);   
#endif 
}


void FreeBuffer(float *buffer){
#if CUDA_ENABLE    
    cudaFree(buffer);
#else
       printf("CUDA DISABLED FreeBuffer\n");
       exit(0);   
#endif 
}

void FreeEdge(VertexId_CUDA *buffer){
#if CUDA_ENABLE
     cudaFree(buffer);
#else
       printf("CUDA DISABLED FreeEdge\n");
       exit(0);   
#endif 
}
void zero_buffer(float* buffer,int size){
#if CUDA_ENABLE
    CHECK_CUDA_RESULT(cudaMemset(buffer,0,sizeof(float)*size));
    cudaDeviceSynchronize();
#else
       printf("CUDA DISABLED zero_buffer\n");
       exit(0);   
#endif 
}


void allocate_gpu_buffer(float** input, int size){
#if CUDA_ENABLE
        CHECK_CUDA_RESULT(cudaMalloc(input,sizeof(float)*(size)));
#else
       printf("CUDA DISABLED Cuda_Stream::Gather_By_Dst_From_Message\n");
       exit(0);   
#endif 

}
void allocate_gpu_edge(VertexId_CUDA** input, int size){
#if CUDA_ENABLE
     CHECK_CUDA_RESULT(cudaMalloc(input,sizeof(VertexId_CUDA)*(size)));
#else 
     printf("CUDA DISABLED Cuda_Stream::Gather_By_Dst_From_Message\n");
     exit(0);   
   
#endif 
}

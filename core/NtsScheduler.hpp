/*
Copyright (c) 2015-2016 Xiaowei Zhu, Tsinghua University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#ifndef NTSSCHEDULER_HPP
#define NTSSCHEDULER_HPP
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <malloc.h>
#include <sys/mman.h>
#include <numa.h>
#include <omp.h>
#include <math.h>
#include <algorithm>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <functional>
#include <sstream>

#include "core/atomic.hpp"
#include "core/bitmap.hpp"
#include "core/constants.hpp"
#include "core/filesystem.hpp"
#include "core/mpi.hpp"
#include "core/time.hpp"
#include "core/type.hpp"
#include "cuda/test.hpp"
#include "comm/Network.hpp"
#include "torch/torch.h"
#include "torch/csrc/autograd/generated/variable_factories.h"
#include "torch/nn/module.h"
#include "torch/csrc/api/include/torch/cuda.h"
#include "ATen/ATen.h"
#include "GraphSegment.hpp"

typedef torch::Tensor NtsVar;
typedef torch::nn::Module NtsMudule;
typedef torch::DeviceType NtsDevide;

enum AGGTYPE{
/* S: from Source
 * M: from Message
 * D: to Destination
 * P: edge weight require parameter
 * W: edge weight require no parameter
 * s: scalar type edge weight/parameter
 * t: tensor type edge weight/parameter   
 */
    SD,  
    SPsD,
    SPtD,
    SWD,
    MD,
    MPsD,
    MPtD,
    MWD,
};
class NtsScheduler{
public:
    NtsScheduler(){
         ;
    }
    void InitBlock(CSC_segment_pinned* graph_partition,runtimeinfo *rtminfo_, VertexId feature_size_, 
                    VertexId output_size_,VertexId current_process_partition_id_,
                    VertexId current_process_layer_,Cuda_Stream * cuda_stream_){//for DEBUG
        src=graph_partition->source;
        dst=graph_partition->destination;
        E=graph_partition->edge_size;
        feature_size=feature_size_;
        output_size=output_size_;
        src_start=graph_partition->src_range[0];
        dst_start=graph_partition->dst_range[0];        
        srcT=(torch::from_blob(src, {E, 1},torch::kLong)-(long)src_start).cuda();
        dstT=(torch::from_blob(dst, {E, 1},torch::kLong)-(long)dst_start).cuda();
        cuda_stream=cuda_stream_;
        subgraph=graph_partition;
        current_process_layer=current_process_layer_;
        current_process_partition_id=current_process_partition_id_;
        rtminfo=rtminfo_;
        aggtype=MD;
    }
    
    void InitBlockSimple(CSC_segment_pinned* graph_partition,runtimeinfo *rtminfo_, VertexId feature_size_, 
                    VertexId output_size_,VertexId current_process_partition_id_,
                    VertexId current_process_layer_,Cuda_Stream * cuda_stream_){//for DEBUG
        src=graph_partition->source;
        dst=graph_partition->destination;
        E=graph_partition->edge_size;
        feature_size=feature_size_;
        output_size=output_size_;
        src_start=graph_partition->src_range[0];
        dst_start=graph_partition->dst_range[0];
        cuda_stream=cuda_stream_;
        subgraph=graph_partition;
        current_process_layer=current_process_layer_;
        current_process_partition_id=current_process_partition_id_;
        rtminfo=rtminfo_;
        aggtype=MD;
    }
    
    
    inline torch::Tensor ScatterSrc(torch::Tensor &src_input){
        //srcT=torch::from_blob(src, {E, 1},torch::kInt64).cuda();
        return src_input.gather(0,(srcT).expand({srcT.size(0),src_input.size(1)}));
    }
    inline torch::Tensor ScatterDst(torch::Tensor &dst_input){
        return dst_input.gather(0,(dstT).expand({dstT.size(0),dst_input.size(1)}));
    }
    inline torch::Tensor PrepareMessage(torch::Tensor &message){
        return torch::sparse_coo_tensor(torch::cat({srcT,dstT},1).t(),message,
                at::TensorOptions().device_index(0).dtype(torch::kFloat).requires_grad(true));
    }
    inline torch::Tensor PrepareMessage(torch::Tensor index, torch::Tensor &message){
        return torch::sparse_coo_tensor(index,message,
                at::TensorOptions().device_index(0).dtype(torch::kFloat).requires_grad(true));
    }
    
    inline void AggregateForward(torch::Tensor& output, torch::Tensor &input_src,torch::Tensor &weight,torch::Tensor &message){
         with_weight=true;
        switch(aggtype){
            case SD://GIN COMMNET
                with_weight=false;
                GatherByDstFromSrc(output, input_src, weight);   
                break;
            case SPsD://GAT
                GatherByDstFromSrc(output, input_src, weight);   
               break;
            case SPtD:
                GatherByDstFromSrcTensorWeight(output, input_src, weight);
                break;
            case SWD://GCN
                GatherByDstFromSrc(output, input_src, weight);
                break;
            case MD://GGNN
                with_weight=false;
                GatherByDstFromMessage(output, message, weight);
                break;
            case MPsD:
                printf("MPsD not implemented\n");//It can be implemented with pytorch
                exit(0);
                break;
            case MPtD:
                printf("MPtD not implemented\n");//It can be implemented with pytorch
                break;
            case MWD:
                printf("MWD not implemented\n");//It can be implemented with pytorch
                break;
            default:
                printf("Unknown implemented\n");
                exit(0);
        }

    }
    
    //BackwardScatterGradBackToWeight(torch::Tensor &input_src,torch::Tensor &grad_output, torch::Tensor &message_grad){
    inline void AggregateBackward(torch::Tensor& output, torch::Tensor &input_src,torch::Tensor &weight, 
                                                   torch::Tensor &grad_output, torch::Tensor &weight_grad,torch::Tensor &message_grad){
        with_weight=true;
        switch(aggtype){
            case SD://GIN COMMNET
                with_weight=false;
                GatherBySrcFromDst(output,input_src,weight);
                break;
            case SPsD://GAT
                GatherBySrcFromDst(output,input_src,weight);
                BackwardScatterGradBackToWeight(input_src, grad_output,weight_grad);
               break;
            case SPtD:   
                GatherBySrcFromDstTensorWeight(output,input_src,weight);
                BackwardScatterGradBackToTensorWeight(input_src, grad_output,weight_grad);
                break;
            case SWD://GCN
               GatherBySrcFromDst(output,input_src,weight); 
                break;
            case MD://GGNN
                with_weight=false;
                BackwardScatterGradBackToMessage(grad_output,message_grad);
                break;
            case MPsD:
                printf("MPsD backward not implemented\n");//It can be implemented with pytorch
                exit(0);
                break;
            case MPtD:
                printf("MPtD backward not implemented\n");//It can be implemented with pytorch
                break;
            case MWD:
                printf("MWD backward not implemented\n");//It can be implemented with pytorch
                break;
            default:
                printf("Unknown backward aggregate implemented\n");
                exit(0);
        }
        
        GatherBySrcFromDst(output,input_src,weight);
        BackwardScatterGradBackToWeight(input_src, grad_output,message_grad);
    }
    
    inline void GatherByDstFromSrc(torch::Tensor& output, torch::Tensor &input_src,torch::Tensor weight){//TODO
        float *input_buffer=getWritableBuffer(input_src);//.packed_accessor<float,2>().data();
        float *weight_buffer=getWritableBuffer(weight);//.packed_accessor<float,2>().data();
        float *output_buffer=getWritableBuffer(output);//.packed_accessor<float,2>().data();
        VertexId *column_offset_from_pinned=subgraph->column_offset_gpu;
        VertexId *row_indices_from_pinned=subgraph->row_indices_gpu;
        ValueType *forward_weight_from_pinned=subgraph->edge_weight_forward_gpu;
        //printf("output size %d\n",output_size);
        
    VertexId src_start = subgraph->src_range[0];
    VertexId src_end = subgraph->src_range[1];
    VertexId dst_start = subgraph->dst_range[0];
    VertexId dst_end = subgraph->dst_range[1];
//    if(feature_size>512){
        cuda_stream->Gather_By_Dst_From_Src(input_buffer,
                               output_buffer,
                               //weight_buffer, //data
                               forward_weight_from_pinned,
                               row_indices_from_pinned,
                               column_offset_from_pinned, //graph
                               (VertexId)src_start, (VertexId)src_end, 
                               (VertexId)dst_start, (VertexId)dst_end,
                               (VertexId)subgraph->edge_size,
                               (VertexId)subgraph->batch_size_forward,
                               (VertexId)output_size,
                               rtminfo->with_weight);
        
//    }else{
//        cuda_stream->Gather_By_Dst_From_Src_Optim(input_buffer,
//                               output_buffer,
//                               //weight_buffer, //data
//                               forward_weight_from_pinned,
//                               row_indices_from_pinned,
//                               column_offset_from_pinned, //graph
//                                subgraph->destination_gpu,
//                               (VertexId)src_start, (VertexId)src_end, 
//                               (VertexId)dst_start, (VertexId)dst_end,
//                               (VertexId)subgraph->edge_size,
//                               (VertexId)subgraph->batch_size_forward,
//                               (VertexId)output_size,
//                               rtminfo->with_weight);
//    }
    //cuda_stream->CUDA_DEVICE_SYNCHRONIZE(); 
    }
    
    inline void GatherByDstFromSrcTensorWeight(torch::Tensor& output, torch::Tensor &input_src,torch::Tensor &weight){//TODO
        float *input_buffer=getWritableBuffer(input_src);//.packed_accessor<float,2>().data();
        float *weight_buffer=getWritableBuffer(weight);//.packed_accessor<float,2>().data();
        float *output_buffer=getWritableBuffer(output);//.packed_accessor<float,2>().data();
        VertexId *column_offset_from_pinned=subgraph->column_offset_gpu;
        VertexId *row_indices_from_pinned=subgraph->row_indices_gpu;
        
    VertexId src_start = subgraph->src_range[0];
    VertexId src_end = subgraph->src_range[1];
    VertexId dst_start = subgraph->dst_range[0];
    VertexId dst_end = subgraph->dst_range[1];
    
    cuda_stream->Gather_By_Dst_From_Src(input_buffer,
                               output_buffer,
                               weight_buffer, //data
                               row_indices_from_pinned,
                               column_offset_from_pinned, //graph
                               (VertexId)src_start, (VertexId)src_end, 
                               (VertexId)dst_start, (VertexId)dst_end,
                               (VertexId)subgraph->edge_size,
                               (VertexId)subgraph->batch_size_forward,
                               (VertexId)output_size,
                               with_weight,true);
    cuda_stream->CUDA_DEVICE_SYNCHRONIZE(); 
    }
    inline void GatherByDstFromMessage(torch::Tensor& output, torch::Tensor &message,torch::Tensor &weight){
        float *message_buffer=getWritableBuffer(message);
        float *weight_buffer=getWritableBuffer(weight);
        float *output_buffer=getWritableBuffer(output);
        VertexId *column_offset_from_pinned=subgraph->column_offset_gpu;
        VertexId *row_indices_from_pinned=subgraph->row_indices_gpu;
        
        VertexId src_start = subgraph->src_range[0];
        VertexId src_end = subgraph->src_range[1];
        VertexId dst_start = subgraph->dst_range[0];
        VertexId dst_end = subgraph->dst_range[1];

        cuda_stream->Gather_By_Dst_From_Message(message_buffer,
                               output_buffer,
                               weight_buffer, //data
                               row_indices_from_pinned,
                               column_offset_from_pinned, //graph
                               src_start, src_end, dst_start, dst_end,
                               E,
                               subgraph->batch_size_forward,
                               output_size, with_weight);
    cuda_stream->CUDA_DEVICE_SYNCHRONIZE(); 
    }
    
    inline void BackwardScatterGradBackToWeight(torch::Tensor &input_src,torch::Tensor &grad_output, torch::Tensor &weight_grad){
        float *input_src_buffer=getWritableBuffer(input_src);
        float *grad_output_buffer=getWritableBuffer(grad_output);//.packed_accessor<float,2>().data();
        float *weight_grad_buffer=getWritableBuffer(weight_grad);//.packed_accessor<float,2>().data();
        VertexId src_start = subgraph->src_range[0];
        VertexId src_end = subgraph->src_range[1];
        VertexId dst_start = subgraph->dst_range[0];
        VertexId dst_end = subgraph->dst_range[1];
        cuda_stream->Scatter_Grad_Back_To_Weight(input_src_buffer,
                               grad_output_buffer,
                               weight_grad_buffer, //data
                               subgraph->source_gpu,
                               subgraph->destination_gpu, //graph
                               (VertexId)src_start, (VertexId)src_end, 
                               (VertexId)dst_start, (VertexId)dst_end,
                               (VertexId)subgraph->edge_size,
                               (VertexId)subgraph->batch_size_forward,
                               (VertexId)output_size,
                               false);
        cuda_stream->CUDA_DEVICE_SYNCHRONIZE();   
    }
    inline void BackwardScatterGradBackToTensorWeight(torch::Tensor &input_src,torch::Tensor &grad_output, torch::Tensor &weight_grad){
        float *input_src_buffer=getWritableBuffer(input_src);
        float *grad_output_buffer=getWritableBuffer(grad_output);//.packed_accessor<float,2>().data();
        float *weight_grad_buffer=getWritableBuffer(weight_grad);//.packed_accessor<float,2>().data();
        VertexId src_start = subgraph->src_range[0];
        VertexId src_end = subgraph->src_range[1];
        VertexId dst_start = subgraph->dst_range[0];
        VertexId dst_end = subgraph->dst_range[1];
        cuda_stream->Scatter_Grad_Back_To_Weight(input_src_buffer,
                               grad_output_buffer,
                               weight_grad_buffer, //data
                               subgraph->source_gpu,
                               subgraph->destination_gpu, //graph
                               (VertexId)src_start, (VertexId)src_end, 
                               (VertexId)dst_start, (VertexId)dst_end,
                               (VertexId)subgraph->edge_size,
                               (VertexId)subgraph->batch_size_forward,
                               (VertexId)output_size,
                               true);
        cuda_stream->CUDA_DEVICE_SYNCHRONIZE();   
    }
        
    inline void BackwardScatterGradBackToMessage(torch::Tensor &grad_dst, torch::Tensor &message_grad){
        float *grad_dst_buffer=getWritableBuffer(grad_dst);
        float *message_grad_buffer=getWritableBuffer(message_grad);//.packed_accessor<float,2>().data();
        VertexId src_start = subgraph->src_range[0];
        VertexId src_end = subgraph->src_range[1];
        VertexId dst_start = subgraph->dst_range[0];
        VertexId dst_end = subgraph->dst_range[1];
        VertexId *column_offset_from_pinned=subgraph->column_offset_gpu;
        VertexId *row_indices_from_pinned=subgraph->row_indices_gpu;
        cuda_stream->Scatter_Grad_Back_To_Message(grad_dst_buffer,
                               message_grad_buffer, //data
                               row_indices_from_pinned,
                               column_offset_from_pinned, //graph
                               (VertexId)src_start, (VertexId)src_end, 
                               (VertexId)dst_start, (VertexId)dst_end,
                               (VertexId)subgraph->edge_size,
                               (VertexId)subgraph->batch_size_forward,
                               (VertexId)output_size,false);
        cuda_stream->CUDA_DEVICE_SYNCHRONIZE();   
    }
    
 
    inline void GatherBySrcFromDst(torch::Tensor& output, torch::Tensor &input_src,torch::Tensor &weight){//TO DO
        float *input_buffer=getWritableBuffer(input_src);
        float *weight_buffer=getWritableBuffer(weight);
        float *output_buffer=getWritableBuffer(output);
        VertexId *row_offset_from_pinned=subgraph->row_offset_gpu;
        VertexId *column_indices_from_pinned=subgraph->column_indices_gpu;
        ValueType *backward_weight_from_pinned=subgraph->edge_weight_backward_gpu;
        
        VertexId src_start = subgraph->src_range[0];
        VertexId src_end = subgraph->src_range[1];
        VertexId dst_start = subgraph->dst_range[0];
        VertexId dst_end = subgraph->dst_range[1];
//        if(feature_size>512){      
            cuda_stream->Gather_By_Src_From_Dst(input_buffer,
                               output_buffer,
                               //weight_buffer, //data
                               backward_weight_from_pinned,
                               row_offset_from_pinned, //graph
                               column_indices_from_pinned,
                               (VertexId)src_start, (VertexId)src_end, 
                               (VertexId)dst_start, (VertexId)dst_end,
                               (VertexId)subgraph->edge_size,
                               (VertexId)subgraph->batch_size_backward,
                               (VertexId)output_size,
                               rtminfo->with_weight);
//        }else{
//            cuda_stream->Gather_By_Src_From_Dst_Optim(input_buffer,
//                               output_buffer,
//                               //weight_buffer, //data
//                               backward_weight_from_pinned,
//                               row_offset_from_pinned, //graph
//                               column_indices_from_pinned,
//                               subgraph->source_backward_gpu,
//                               (VertexId)src_start, (VertexId)src_end, 
//                               (VertexId)dst_start, (VertexId)dst_end,
//                               (VertexId)subgraph->edge_size,
//                               (VertexId)subgraph->batch_size_backward,
//                               (VertexId)output_size,
//                               rtminfo->with_weight);
//        }
    }

    
    inline void GatherBySrcFromDstTensorWeight(torch::Tensor& output, torch::Tensor &input_src,torch::Tensor &weight){//TO DO
        float *input_buffer=getWritableBuffer(input_src);
        float *weight_buffer=getWritableBuffer(weight);
        float *output_buffer=getWritableBuffer(output);
        VertexId *row_offset_from_pinned=subgraph->row_offset_gpu;
        VertexId *column_indices_from_pinned=subgraph->column_indices_gpu;
        
    VertexId src_start = subgraph->src_range[0];
    VertexId src_end = subgraph->src_range[1];
    VertexId dst_start = subgraph->dst_range[0];
    VertexId dst_end = subgraph->dst_range[1];
    
    //std::cout<<output_size<<"  "<<src_end-src_start<<" "<<subgraph->batch_size_backward<<std::endl;
    cuda_stream->Gather_By_Src_From_Dst(input_buffer,
                               output_buffer,
                               weight_buffer, //data
                               row_offset_from_pinned, //graph
                               column_indices_from_pinned,
                               (VertexId)src_start, (VertexId)src_end, 
                               (VertexId)dst_start, (VertexId)dst_end,
                               (VertexId)subgraph->edge_size,
                               (VertexId)subgraph->batch_size_backward,
                               (VertexId)output_size,
                               with_weight,true);
    cuda_stream->CUDA_DEVICE_SYNCHRONIZE(); 
    }
    
    
    inline torch::Tensor DeSerializeTensorToGPU(torch::Tensor &var_cpu){
        
         torch::Tensor DeSe_data=torch::zeros_like(var_cpu.cuda(),at::TensorOptions().device_index(0).requires_grad(true));
         DeSe_data.set_data(var_cpu.cuda());
         return DeSe_data; 
    }
    
    inline void SerializeToCPU(std::string name,torch::Tensor &var_gpu){
        //assert(var_cpu.device()==torch::Device::Type::GPU);
         CacheVar[VarEncode(name)]=var_gpu.cpu();
         return; 
    }
    inline torch::Tensor DeSerializeFromCPU(std::string name,torch::DeviceType location=torch::DeviceType::CUDA,int device_id=0){
        torch::Tensor var_cpu=CacheVar[VarEncode(name)];
        if(torch::DeviceType::CUDA==location){
            //assert(var_cpu.device()==torch::Device::Type::CPU);
            torch::Tensor DeSe_data=torch::zeros_like(var_cpu.cuda(),
                        at::TensorOptions().device_index(device_id).requires_grad(true));
            DeSe_data.set_data(var_cpu.cuda());
            return DeSe_data;
        }
        else{
            torch::Tensor DeSe_data=torch::zeros_like(var_cpu.cuda(),
                        at::TensorOptions().requires_grad(true));
            DeSe_data.set_data(var_cpu.cuda());
            return DeSe_data;
        }
    }

    void ZeroVar(NtsVar& t){
        t.zero_();
    }
    //zero_buffer(local_data_buffer, (partition_offset[partition_id + 1] - partition_offset[partition_id]) * (feature_size));  
    void ZeroVarMem(NtsVar& t,DeviceLocation dl=GPU_T){
        if(dl==GPU_T)
            zero_buffer(t.packed_accessor<float,2>().data(), t.size(0) * t.size(1));
        else if (dl=CPU_T)
            memset(t.accessor<float,2>().data(), 0, t.size(0) * t.size(1));
        else{
            printf("ZeroVarMem Error\n");
        }
    }
    char* getPinnedDevicePointer(char* h_ptr){
        return (char *)getDevicePointer(h_ptr);
    }
    inline void DeserializeMsgToMirror(NtsVar& mirror_input, char* msg,VertexId msg_count,bool sync=false){
        if(msg_count<=0)
            return;
        ZeroVarMem(mirror_input);
        float*gmb=(float*)getPinnedDevicePointer(msg);
        cuda_stream->deSerializeToGPU(mirror_input.packed_accessor<float,2>().data(), gmb, msg_count, feature_size, subgraph->src_range[0], subgraph->src_range[1], false);    
    }
    inline void AggMsgToMaster(NtsVar& master_output, char* msg,VertexId msg_count,bool sync=false){
        if(msg_count<=0)
            return;
        float*gmb=(float*)getPinnedDevicePointer(msg);
        cuda_stream->aggregate_comm_result_debug(master_output.packed_accessor<float,2>().data(), gmb, msg_count, feature_size, subgraph->dst_range[0], subgraph->dst_range[1], false);   
    }
    inline void DeviceSynchronize(){
        cuda_stream->CUDA_DEVICE_SYNCHRONIZE();
    }
        inline torch::Tensor NewKeyTensor(torch::Tensor &mould,torch::DeviceType location=torch::DeviceType::CUDA, int device_id=0){
        
        if(torch::DeviceType::CUDA==location){
           return torch::zeros_like(mould,at::TensorOptions().device_index(device_id).requires_grad(true).dtype(torch::kFloat));  
        }else{
           return torch::zeros_like(mould,at::TensorOptions().requires_grad(true).dtype(torch::kFloat));  
        }
    }
     inline torch::Tensor NewKeyTensor(at::IntArrayRef size, torch::DeviceType location=torch::DeviceType::CUDA, int device_id=0){
        if(torch::DeviceType::CUDA==location){
           return torch::zeros(size,at::TensorOptions().device_index(device_id).requires_grad(true).dtype(torch::kFloat));
        }else{
           return torch::zeros(size,at::TensorOptions().requires_grad(true).dtype(torch::kFloat));
        }
    }
     
     inline torch::Tensor NewLeafTensor(torch::Tensor &mould, torch::DeviceType location=torch::DeviceType::CUDA, int device_id=0){
        
        if(torch::DeviceType::CUDA==location){
             return torch::zeros_like(mould,at::TensorOptions().device_index(device_id).dtype(torch::kFloat));  
        }else{
             return torch::zeros_like(mould,at::TensorOptions().dtype(torch::kFloat));  
        }
    }
     inline torch::Tensor NewLeafTensor(at::IntArrayRef size, torch::DeviceType location=torch::DeviceType::CUDA, int device_id=0){
        if(torch::DeviceType::CUDA==location){
           return torch::zeros(size,at::TensorOptions().device_index(device_id).dtype(torch::kFloat));
        }else{
           return torch::zeros(size,at::TensorOptions().dtype(torch::kFloat));
        }
    }
    inline torch::Tensor NewKeyTensor(float* data,at::IntArrayRef size,torch::DeviceType location=torch::DeviceType::CUDA, int device_id=0){
        if(torch::DeviceType::CUDA==location){
            return torch::from_blob(data,size,at::TensorOptions().requires_grad(true).device_index(device_id).dtype(torch::kFloat));
        }else{
            return torch::from_blob(data,size,at::TensorOptions().requires_grad(true).dtype(torch::kFloat));    
        }

    }
    inline torch::Tensor NewLeafTensor(float* data,at::IntArrayRef size,torch::DeviceType location=torch::DeviceType::CUDA, int device_id=0){
        if(torch::DeviceType::CUDA==location){
        return torch::from_blob(data,size,at::TensorOptions().device_index(device_id).dtype(torch::kFloat));
        }else{
        return torch::from_blob(data,size,at::TensorOptions().dtype(torch::kFloat));    
        }
    } 
    inline torch::Tensor NewLeafKLongTensor(long* data,at::IntArrayRef size){
        return torch::from_blob(data,size,at::TensorOptions().dtype(torch::kLong));    
    } 
    inline torch::Tensor NewLeafKIntTensor(int* data,at::IntArrayRef size){
        return torch::from_blob(data,size,at::TensorOptions().dtype(torch::kInt32));    
    } 
    inline torch::Tensor NewOnesTensor(at::IntArrayRef size, torch::DeviceType location=torch::DeviceType::CUDA, int device_id=0){
        if(torch::DeviceType::CUDA==location){
           return torch::ones(size,at::TensorOptions().device_index(device_id).dtype(torch::kFloat));
        }else{
           return torch::ones(size,at::TensorOptions().dtype(torch::kFloat));
        }
    }
    inline float* getWritableBuffer(torch::Tensor &T_var,torch::DeviceType location=torch::DeviceType::CUDA){
        if(torch::DeviceType::CUDA==location){
        return T_var.packed_accessor<float,2>().data();
        }else{
        return T_var.accessor<float,2>().data();    
        }
    }
    
   std::string Encode(std::string name, int layer){
      return name.append("_").append(std::to_string(layer));
   }
    
   std::string VarEncode(std::string name){
      return name.append("_").append(std::to_string(current_process_layer)).append("_").append(std::to_string(current_process_partition_id));
   }
    void SerializeMirrorToMsg(float* th, torch::Tensor &td, bool sync=false){
            cuda_stream->move_result_out(th + (subgraph->src_range[0] * feature_size),
                                 td.packed_accessor<float, 2>().data(),
                                 subgraph->src_range[0],
                                 subgraph->src_range[1],
                                 feature_size, sync);
    }
    
    inline void MoveDataInGPU(float* th, torch::Tensor &td, bool sync=false){
            cuda_stream->move_result_out(th + (subgraph->src_range[0] * feature_size),
                                 td.packed_accessor<float, 2>().data(),
                                 subgraph->src_range[0],
                                 subgraph->src_range[1],
                                 feature_size, sync);
    }
    
    inline int BYSRC(){
        return 0;
    }
    inline int BYDST(){
        return 1;
    }
    ValueType* recv_cached_buffer;
    ValueType* input_tensor_buffer;
    long* src;
    long* dst;
    VertexId E;
    VertexId feature_size;
    VertexId output_size;
    torch::Tensor srcT;
    torch::Tensor dstT;
    int src_start;
    int dst_start;
    bool with_weight;
    VertexId current_process_partition_id;
    VertexId current_process_layer;
    Cuda_Stream * cuda_stream;
    CSC_segment_pinned*  subgraph;
    std::map<std::string,torch::Tensor>KeyVar;//key roles in the compute graph
    std::map<std::string,torch::Tensor>InterVar;//key roles in the compute graph
    std::map<std::string,torch::Tensor>CacheVar;//used for caching data;
    runtimeinfo *rtminfo;
    AGGTYPE aggtype;
    //src_input.cpu() dst_input.cpu()
};
struct CachedData{
public:
    CachedData(){
        ;
    }
    CachedData(int partitions,int layers, bool scale=false){
        data_scale=scale;
        NtsVar s;
        for(int i=0;i<layers;i++){
            std::vector<NtsVar> tmp;
            for( int j=0;j<partitions;j++){
                tmp.push_back(s);
            }
            if(0==scale){
                mirror_input.push_back(tmp);
                message.push_back(tmp);
            }else{
                mirror_input_cpu.push_back(tmp);
                message_cpu.push_back(tmp);
            }
        }
    }
 //handling small dataset
 std::vector<std::vector<NtsVar> >mirror_input;
 std::vector<std::vector<NtsVar> >message;
 
 //large dataset
 std::vector<std::vector<NtsVar> >mirror_input_cpu;
 std::vector<std::vector<NtsVar> >message_cpu;
 bool data_scale;

};


struct Parameter : torch::nn::Module
{
    NtsVar W;
    NtsVar M;
    NtsVar V;
    NtsVar M_GPU;
    NtsVar V_GPU;
    ValueType *W_from;
    ValueType *w_gradient_buffer;
    Network_simple<float> *network_simple;
    int row, col;
    NtsVar W_gradient;
    NtsVar W_g;
    //gpiu_processor *gp;
    ValueType alpha;
    ValueType beta1;
    ValueType beta2;
    ValueType epsilon;
    ValueType alpha_t;
    ValueType beta1_t;
    ValueType beta2_t;
    ValueType epsilon_t;
    ValueType l_r;
    ValueType weight_decay;
    int curr_epoch;
    
    int decay_rate;
    int decay_epoch;
    Parameter(size_t w, size_t h, ValueType alpha_,ValueType beta1_,ValueType beta2_, ValueType epsilon_,ValueType weight_decay_)
    {
        row = w;
        col = h;
	ValueType scale=sqrt(6.0/(w+h));
        W = register_parameter("W", (2*scale)*torch::rand({w, h}, torch::kFloat)-scale*torch::ones({w,h},torch::kFloat));
//	ValueType scale=sqrt(6.0/(w+h));
//	W=(2*scale)*W-scale;
        W_from = new ValueType[w * h];
        w_gradient_buffer = new ValueType[w * h];
        memset(w_gradient_buffer, 0, sizeof(float) * w * h);
        W_gradient = torch::from_blob(w_gradient_buffer, {w, h}, torch::kFloat);
        network_simple = new Network_simple<float>(row, col);
	M=torch::zeros({w,h}, torch::kFloat);
        V=torch::zeros({w,h}, torch::kFloat);
	alpha=alpha_;
        beta1=beta1_;
        beta2=beta2_;
        epsilon=epsilon_;
	alpha_t=alpha_;
        beta1_t=beta1_;
        beta2_t=beta2_;
        epsilon_t=epsilon_;
        weight_decay=weight_decay_;
        curr_epoch=0;
        decay_epoch=-1;
    }
    Parameter(size_t w, size_t h,ValueType l_r_=0.01,ValueType weight_decay_=0.05)
    {
        alpha=0.0;
        row = w;
        col = h;
	ValueType scale=sqrt(6.0/(w+h));
        W = register_parameter("W", (2*scale)*torch::rand({w, h}, torch::kFloat)-scale*torch::ones({w,h},torch::kFloat));
//	ValueType scale=sqrt(6.0/(w+h));
//	W=(2*scale)*W-scale;
        W_from = new ValueType[w * h];
        w_gradient_buffer = new ValueType[w * h];
        memset(w_gradient_buffer, 0, sizeof(float) * w * h);
        W_gradient = torch::from_blob(w_gradient_buffer, {w, h}, torch::kFloat);
        network_simple = new Network_simple<float>(row, col);
        weight_decay=weight_decay;
        l_r=l_r_;
        curr_epoch=0;
        decay_epoch=-1;
//	M=torch::zeros({w,h}, torch::kFloat);
//        V=torch::zeros({w,h}, torch::kFloat);
//	alpha=0.01;
//        beta1=0.9;
//        beta2=0.999;
//        epsilon=1e-8;
//	alpha_t=0.01;
//        beta1_t=0.9;
//        beta2_t=0.999;
//        epsilon_t=1e-8;
    }
    void Adam_to_GPU(){
        M_GPU=M.cuda();
        V_GPU=V.cuda();
        W_g=W_gradient.cuda();
        
    }
    void set_decay(ValueType decay_rate_,ValueType decay_epoch_){
        decay_rate=decay_rate_;
        decay_epoch=decay_epoch_;
    }
    void init_parameter()
    {
        network_simple->broadcast(W.accessor<ValueType, 2>().data());
    }
    void all_reduce_to_gradient(NtsVar from)
    {
        W_gradient.set_data(from);
        network_simple->all_reduce_sum(W_gradient.accessor<ValueType, 2>().data());
    }
    void resetW(size_t w, size_t h, ValueType *buffer)
    {
        memcpy(W_from, buffer, sizeof(ValueType) * w * h);
        NtsVar new_weight_tensor = torch::from_blob(W_from, {w, h});
        W.set_data(new_weight_tensor);
    }

    void learnC2G(ValueType learning_rate)
    {
        NtsVar tmp = W_gradient.cuda();
        NtsVar a = (W - (tmp * learning_rate));
        W.set_data(a);
    }
    
    void learnC2G_with_decay(ValueType learning_rate,ValueType weight_decay)
    {
        NtsVar tmp = W_gradient.cuda();
        NtsVar a = (W - (tmp * learning_rate))*(1-weight_decay);
        W.set_data(a);
    }
     void learnC2C_with_decay_SGD(ValueType learning_rate,ValueType weight_decay)
    {
        NtsVar tmp = W_gradient;
        NtsVar a = (W - (tmp * learning_rate))*(1-weight_decay);
        W.set_data(a);
    }
    void learnC2C_with_decay_Adam()
    {
//        assert(alpha>0.0);
        W_g=W_gradient+weight_decay*W;	
        M=beta1*M+(1-beta1)*W_g;
        V=beta2*V+(1-beta2)*W_g*W_g;
        //NtsVar a = W - alpha*M/(torch::sqrt(V)+epsilon);
        W.set_data(W - alpha*M/(torch::sqrt(V)+epsilon));
    }
    void learnC2G_with_decay_Adam()
    {
//        assert(alpha>0.0);
//        W_g.set_data(W_gradient.cuda());
//        NtsVar s=W;
        W_g.set_data(W);
        W_g=W_g*weight_decay;
        W_g=W_g+W_gradient.cuda();//+weight_decay;
        M_GPU=beta1*M_GPU+(1-beta1)*W_g;
        V_GPU=beta2*V_GPU+(1-beta2)*W_g*W_g;
        W.set_data(W - alpha*M_GPU/(torch::sqrt(V_GPU)+epsilon));
    }
    void next(){
        if(decay_epoch!=-1&&(curr_epoch!=0&&curr_epoch%decay_epoch==0)){
            alpha_t*=decay_rate;
           // printf("123123123123123123123131221313123123\n");
        }
        alpha=alpha_t*sqrt(1-beta2)/(1-beta1);
        beta1*=beta1_t;
        beta2*=beta2_t;
        curr_epoch++;
    }

    void learn(NtsVar from, ValueType learning_rate)
    {
        NtsVar a = (W - (from * learning_rate));

        W.set_data(a);
    }
    void learn_gpu(NtsVar from, ValueType learning_rate)
    {
        NtsVar a = (W - (from * learning_rate));
        W.set_data(a);
        //W=a;
    }
    NtsVar forward(NtsVar x)
    {

        NtsVar x1 = x.mm(W);
        return x1;
    }
};

//struct Intermediate : torch::nn::Module
//{
//    NtsVar W;
//    ValueType *W_from;
//
//    Intermediate(size_t w, size_t h)
//    {
//        //        at::TensorOptions *opt=new at::TensorOptions;
//        //       opt->requires_grad(true);
//        //  torch::randn
//        //     A=torch::randn(torch::randn({w,h},opt));
//        W = register_parameter("W", torch::randn({w, h}));
//        W_from = new ValueType[w * h];
//    }
//    void resetW(size_t w, size_t h, ValueType *buffer)
//    {
//        memcpy(W_from, buffer, sizeof(ValueType) * w * h);
//        NtsVar new_weight_tensor = torch::from_blob(W_from, {w, h});
//        W.set_data(new_weight_tensor);
//    }
//
//    NtsVar forward(NtsVar x)
//    {
//        x = x.mm(W);
//        return x;
//    }
//};


#endif

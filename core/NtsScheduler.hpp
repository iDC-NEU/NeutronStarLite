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

typedef torch::Tensor NtsVar;
typedef torch::nn::Module NtsMudule;
typedef torch::DeviceType NtsDevide;

namespace AGGTYPE{    
enum{
    S2D,
    S2DP,
    S2DW,
    M2D,
    M2DP,
    M2DW
};
}
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
                               feature_size, rtminfo->with_weight);
    cuda_stream->CUDA_DEVICE_SYNCHRONIZE(); 
    }
    inline void AggregateForward(torch::Tensor& output, torch::Tensor &input_src,torch::Tensor &weight){
        GatherByDstFromMessage(output, input_src, weight);
    }
    
    //BackwardScatterGradBackToWeight(torch::Tensor &input_src,torch::Tensor &grad_output, torch::Tensor &message_grad){
    inline void AggregateBackward(torch::Tensor& output, torch::Tensor &input_src,torch::Tensor &weight, 
                                                   torch::Tensor &grad_output, torch::Tensor &message_grad){
        GatherBySrcFromDst(output,input_src,weight);
        BackwardScatterGradBackToWeight(input_src, grad_output,message_grad);
    }
    
    
    inline void GatherBySrcFromDst(torch::Tensor& output, torch::Tensor &input_src,torch::Tensor &weight){
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
                               rtminfo->with_weight);
    cuda_stream->CUDA_DEVICE_SYNCHRONIZE(); 
    }
    
    inline void GatherByDstFromSrc(torch::Tensor& output, torch::Tensor &input_src,torch::Tensor &weight){
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
                               rtminfo->with_weight);
    cuda_stream->CUDA_DEVICE_SYNCHRONIZE(); 
    }
    
    inline void BackwardScatterGradBackToWeight(torch::Tensor &input_src,torch::Tensor &grad_output, torch::Tensor &message_grad){
        float *input_src_buffer=getWritableBuffer(input_src);
        float *grad_output_buffer=getWritableBuffer(grad_output);//.packed_accessor<float,2>().data();
        float *message_grad_buffer=getWritableBuffer(message_grad);//.packed_accessor<float,2>().data();
        VertexId src_start = subgraph->src_range[0];
        VertexId src_end = subgraph->src_range[1];
        VertexId dst_start = subgraph->dst_range[0];
        VertexId dst_end = subgraph->dst_range[1];
        cuda_stream->Scatter_Grad_Back_To_Weight(input_src_buffer,
                               grad_output_buffer,
                               message_grad_buffer, //data
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
    void ZeroVar(NtsVar& t){
        t.zero_();
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
    void MoveResultOut(float* th, torch::Tensor &td, bool sync=false){
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
    //src_input_trans dst_input_trans, message,
    std::map<std::string,torch::Tensor>CacheVar;//used for caching data;
    runtimeinfo *rtminfo;
    //src_input.cpu() dst_input.cpu()
};



struct GnnUnit : torch::nn::Module
{
    NtsVar W;
    ValueType *W_from;
    ValueType *w_gradient_buffer;
    Network_simple<float> *network_simple;
    int row, col;
    NtsVar W_gradient;
    //gpu_processor *gp;
    GnnUnit(size_t w, size_t h)
    {
        row = w;
        col = h;
        W = register_parameter("W", torch::randn({w, h}, torch::kFloat));
        W_from = new ValueType[w * h];
        w_gradient_buffer = new ValueType[w * h];
        memset(w_gradient_buffer, 0, sizeof(float) * w * h);
        W_gradient = torch::from_blob(w_gradient_buffer, {w, h}, torch::kFloat);
        network_simple = new Network_simple<float>(row, col);
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
     void learnC2C_with_decay(ValueType learning_rate,ValueType weight_decay)
    {
        NtsVar tmp = W_gradient;
        NtsVar a = (W - (tmp * learning_rate))*(1-weight_decay);
        W.set_data(a);
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

    NtsVar forward2(NtsVar x)
    {
        return torch::sigmoid(x);
    }
    NtsVar forward3(NtsVar x)
    {

        x = x.mm(W);
        return x.log_softmax(1);
    }
};

struct Intermediate : torch::nn::Module
{
    NtsVar W;
    ValueType *W_from;

    Intermediate(size_t w, size_t h)
    {
        //        at::TensorOptions *opt=new at::TensorOptions;
        //       opt->requires_grad(true);
        //  torch::randn
        //     A=torch::randn(torch::randn({w,h},opt));
        W = register_parameter("W", torch::randn({w, h}));
        W_from = new ValueType[w * h];
    }
    void resetW(size_t w, size_t h, ValueType *buffer)
    {
        memcpy(W_from, buffer, sizeof(ValueType) * w * h);
        NtsVar new_weight_tensor = torch::from_blob(W_from, {w, h});
        W.set_data(new_weight_tensor);
    }

    NtsVar forward(NtsVar x)
    {
        x = x.mm(W);
        return x;
    }
};


#endif

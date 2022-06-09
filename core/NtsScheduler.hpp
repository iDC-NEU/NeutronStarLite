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
#include <algorithm>
#include <fcntl.h>
#include <functional>
#include <malloc.h>
#include <math.h>
#include <mutex>
#include <numa.h>
#include <omp.h>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "dep/gemini/atomic.hpp"
#include "dep/gemini/bitmap.hpp"
#include "dep/gemini/constants.hpp"
#include "dep/gemini/filesystem.hpp"
#include "dep/gemini/mpi.hpp"
#include "dep/gemini/time.hpp"
#include "dep/gemini/type.hpp"

#include "ATen/ATen.h"
#include "core/GraphSegment.h"
#include "comm/network.h"
#include "torch/csrc/autograd/generated/variable_factories.h"
#include "torch/nn/module.h"
#include "torch/torch.h"
//#define CUDA_ENABLE 1
typedef torch::Tensor NtsVar;
typedef torch::nn::Module NtsMudule;
typedef torch::DeviceType NtsDevide;

enum AGGTYPE {
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
class NtsScheduler {
public:
  NtsScheduler() { ; }
  void InitBlock(CSC_segment_pinned *graph_partition, RuntimeInfo *rtminfo_,
                 VertexId feature_size_, VertexId output_size_,
                 VertexId current_process_partition_id_,
                 VertexId current_process_layer_) { // for DEBUG
    src = graph_partition->source;
    dst = graph_partition->destination;
    E = graph_partition->edge_size;
    feature_size = feature_size_;
    output_size = output_size_;
    src_start = graph_partition->src_range[0];
    dst_start = graph_partition->dst_range[0];
#if CUDA_ENABLE
    srcT =
        (torch::from_blob(src, {E, 1}, torch::kLong) - (long)src_start).cuda();
    dstT =
        (torch::from_blob(dst, {E, 1}, torch::kLong) - (long)dst_start).cuda();
    cuda_stream = rtminfo_->cuda_stream_public;
#else
    srcT = (torch::from_blob(src, {E, 1}, torch::kLong) - (long)src_start);
    dstT = (torch::from_blob(dst, {E, 1}, torch::kLong) - (long)dst_start);
#endif
    subgraph = graph_partition;
    current_process_layer = current_process_layer_;
    current_process_partition_id = current_process_partition_id_;
    rtminfo = rtminfo_;
    aggtype = MD;
  }

  void InitBlockSimple(CSC_segment_pinned *graph_partition,
                       RuntimeInfo *rtminfo_, VertexId feature_size_,
                       VertexId output_size_,
                       VertexId current_process_partition_id_,
                       VertexId current_process_layer_) { // for DEBUG
    src = graph_partition->source;
    dst = graph_partition->destination;
    E = graph_partition->edge_size;
    feature_size = feature_size_;
    output_size = output_size_;
    src_start = graph_partition->src_range[0];
    dst_start = graph_partition->dst_range[0];
#if CUDA_ENABLE
    cuda_stream = rtminfo_->cuda_stream_public;
#endif
    subgraph = graph_partition;
    current_process_layer = current_process_layer_;
    current_process_partition_id = current_process_partition_id_;
    rtminfo = rtminfo_;
    aggtype = MD;
  }

//  inline torch::Tensor ScatterSrc(torch::Tensor &src_input) {
//    return src_input.gather(0,
//                            (srcT).expand({srcT.size(0), src_input.size(1)}));
//  }
//  inline torch::Tensor ScatterDst(torch::Tensor &dst_input) {
//    return dst_input.gather(0,
//                            (dstT).expand({dstT.size(0), dst_input.size(1)}));
//  }
#if CUDA_ENABLE
//  inline torch::Tensor PrepareMessage(torch::Tensor &message) {
//    return torch::sparse_coo_tensor(torch::cat({srcT, dstT}, 1).t(), message,
//                                    at::TensorOptions()
//                                        .device_index(0)
//                                        .dtype(torch::kFloat)
//                                        .requires_grad(true));
//  }
//  inline torch::Tensor PrepareMessage(torch::Tensor index,
//                                      torch::Tensor &message) {
//    return torch::sparse_coo_tensor(index, message,
//                                    at::TensorOptions()
//                                        .device_index(0)
//                                        .dtype(torch::kFloat)
//                                        .requires_grad(true));
//  }
  inline void GatherByDstFromSrc(torch::Tensor &output,
                                 torch::Tensor &input_src,
                                 torch::Tensor weight) { // TODO
    ValueType *input_buffer =
        getWritableBuffer(input_src); //.packed_accessor<float,2>().data();
    ValueType *weight_buffer =
        getWritableBuffer(weight); //.packed_accessor<float,2>().data();
    ValueType *output_buffer =
        getWritableBuffer(output); //.packed_accessor<float,2>().data();
    VertexId *column_offset_from_pinned = subgraph->column_offset_gpu;
    VertexId *row_indices_from_pinned = subgraph->row_indices_gpu;
    ValueType *forward_weight_from_pinned = subgraph->edge_weight_forward_gpu;
    // printf("output size %d\n",output_size);

    VertexId src_start = subgraph->src_range[0];
    VertexId src_end = subgraph->src_range[1];
    VertexId dst_start = subgraph->dst_range[0];
    VertexId dst_end = subgraph->dst_range[1];
    if (feature_size > 512 || !rtminfo->optim_kernel_enable) {
      cuda_stream->Gather_By_Dst_From_Src(
          input_buffer, output_buffer,
          // weight_buffer, //data
          forward_weight_from_pinned, row_indices_from_pinned,
          column_offset_from_pinned, // graph
          (VertexId)src_start, (VertexId)src_end, (VertexId)dst_start,
          (VertexId)dst_end, (VertexId)subgraph->edge_size,
          (VertexId)subgraph->batch_size_forward, (VertexId)output_size,
          rtminfo->with_weight);
    } else {
      cuda_stream->Gather_By_Dst_From_Src_Optim(
          input_buffer, output_buffer,
          // weight_buffer, //data
          forward_weight_from_pinned, row_indices_from_pinned,
          column_offset_from_pinned, // graph
          (VertexId)src_start, (VertexId)src_end,
          (VertexId)dst_start, (VertexId)dst_end, (VertexId)subgraph->edge_size,
          (VertexId)subgraph->batch_size_forward, (VertexId)output_size,
          rtminfo->with_weight);
    }
    // cuda_stream->CUDA_DEVICE_SYNCHRONIZE();
  }
  inline void GatherByDstFromMessage(torch::Tensor &output,
                                     torch::Tensor &message) {
    ValueType *message_buffer = getWritableBuffer(message);
    ValueType *output_buffer = getWritableBuffer(output);
    VertexId *column_offset_from_pinned = subgraph->column_offset_gpu;
    VertexId *row_indices_from_pinned = subgraph->row_indices_gpu;

    VertexId src_start = subgraph->src_range[0];
    VertexId src_end = subgraph->src_range[1];
    VertexId dst_start = subgraph->dst_range[0];
    VertexId dst_end = subgraph->dst_range[1];

    cuda_stream->Gather_By_Dst_From_Message(
        message_buffer, output_buffer, row_indices_from_pinned,
        column_offset_from_pinned, // graph
        (VertexId)src_start, (VertexId)src_end, (VertexId)dst_start,
        (VertexId)dst_end, (VertexId)subgraph->edge_size,
        subgraph->batch_size_forward, output_size, with_weight);
    // cuda_stream->CUDA_DEVICE_SYNCHRONIZE();
  }

//  inline void BackwardScatterGradBackToWeight(torch::Tensor &input_src,
//                                              torch::Tensor &grad_output,
//                                              torch::Tensor &weight_grad) {
//    ValueType *input_src_buffer = getWritableBuffer(input_src);
//    ValueType *grad_output_buffer =
//        getWritableBuffer(grad_output); //.packed_accessor<float,2>().data();
//    ValueType *weight_grad_buffer =
//        getWritableBuffer(weight_grad); //.packed_accessor<float,2>().data();
//    VertexId src_start = subgraph->src_range[0];
//    VertexId src_end = subgraph->src_range[1];
//    VertexId dst_start = subgraph->dst_range[0];
//    VertexId dst_end = subgraph->dst_range[1];
//    cuda_stream->Scatter_Grad_Back_To_Weight(
//        input_src_buffer, grad_output_buffer,
//        weight_grad_buffer, // data
//        subgraph->source_gpu,
//        subgraph->destination_gpu, // graph
//        (VertexId)src_start, (VertexId)src_end, (VertexId)dst_start,
//        (VertexId)dst_end, (VertexId)subgraph->edge_size,
//        (VertexId)subgraph->batch_size_forward, (VertexId)output_size, false);
//    cuda_stream->CUDA_DEVICE_SYNCHRONIZE();
//  }
  inline void BackwardScatterGradBackToMessage(torch::Tensor &grad_dst,
                                               torch::Tensor &message_grad) {
    ValueType *grad_dst_buffer = getWritableBuffer(grad_dst);
    ValueType *message_grad_buffer =
        getWritableBuffer(message_grad); //.packed_accessor<float,2>().data();
    VertexId src_start = subgraph->src_range[0];
    VertexId src_end = subgraph->src_range[1];
    VertexId dst_start = subgraph->dst_range[0];
    VertexId dst_end = subgraph->dst_range[1];
    VertexId *column_offset_from_pinned = subgraph->column_offset_gpu;
    VertexId *row_indices_from_pinned = subgraph->row_indices_gpu;
    cuda_stream->Scatter_Grad_Back_To_Message(
        grad_dst_buffer,
        message_grad_buffer, // data
        row_indices_from_pinned,
        column_offset_from_pinned, // graph
        (VertexId)src_start, (VertexId)src_end, (VertexId)dst_start,
        (VertexId)dst_end, (VertexId)subgraph->edge_size,
        (VertexId)subgraph->batch_size_forward, (VertexId)output_size, false);
    cuda_stream->CUDA_DEVICE_SYNCHRONIZE();
  }

  inline void GatherBySrcFromDst(torch::Tensor &output,
                                 torch::Tensor &input_src,
                                 torch::Tensor &weight) { // TO DO
    ValueType *input_buffer = getWritableBuffer(input_src);
    ValueType *weight_buffer = getWritableBuffer(weight);
    ValueType *output_buffer = getWritableBuffer(output);
    VertexId *row_offset_from_pinned = subgraph->row_offset_gpu;
    VertexId *column_indices_from_pinned = subgraph->column_indices_gpu;
    ValueType *backward_weight_from_pinned = subgraph->edge_weight_backward_gpu;

    VertexId src_start = subgraph->src_range[0];
    VertexId src_end = subgraph->src_range[1];
    VertexId dst_start = subgraph->dst_range[0];
    VertexId dst_end = subgraph->dst_range[1];
    if (feature_size > 512 || !rtminfo->optim_kernel_enable) {
      cuda_stream->Gather_By_Src_From_Dst(
          input_buffer, output_buffer,
          // weight_buffer, //data
          backward_weight_from_pinned,
          row_offset_from_pinned, // graph
          column_indices_from_pinned, (VertexId)src_start, (VertexId)src_end,
          (VertexId)dst_start, (VertexId)dst_end, (VertexId)subgraph->edge_size,
          (VertexId)subgraph->batch_size_backward, (VertexId)output_size,
          rtminfo->with_weight);
    } else {
      cuda_stream->Gather_By_Src_From_Dst_Optim(
          input_buffer, output_buffer,
          // weight_buffer, //data
          backward_weight_from_pinned,
          row_offset_from_pinned, // graph
          column_indices_from_pinned,
          (VertexId)src_start, (VertexId)src_end, (VertexId)dst_start,
          (VertexId)dst_end, (VertexId)subgraph->edge_size,
          (VertexId)subgraph->batch_size_backward, (VertexId)output_size,
          rtminfo->with_weight);
    }
  }

  inline torch::Tensor DeSerializeTensorToGPU(torch::Tensor &var_cpu) {

    torch::Tensor DeSe_data = torch::zeros_like(
        var_cpu.cuda(),
        at::TensorOptions().device_index(0).requires_grad(true));
    DeSe_data.set_data(var_cpu.cuda());
    return DeSe_data;
  }

  inline void SerializeToCPU(std::string name, torch::Tensor &var_gpu) {
    // assert(var_cpu.device()==torch::Device::Type::GPU);
    CacheVar[VarEncode(name)] = var_gpu.cpu();
    return;
  }
  inline torch::Tensor
  DeSerializeFromCPU(std::string name,
                     torch::DeviceType location = torch::DeviceType::CUDA,
                     int device_id = 0) {
    torch::Tensor var_cpu = CacheVar[VarEncode(name)];
    if (torch::DeviceType::CUDA == location) {
      // assert(var_cpu.device()==torch::Device::Type::CPU);
      torch::Tensor DeSe_data = torch::zeros_like(
          var_cpu.cuda(),
          at::TensorOptions().device_index(device_id).requires_grad(true));
      DeSe_data.set_data(var_cpu.cuda());
      return DeSe_data;
    } else {
      torch::Tensor DeSe_data = torch::zeros_like(
          var_cpu.cuda(), at::TensorOptions().requires_grad(true));
      DeSe_data.set_data(var_cpu.cuda());
      return DeSe_data;
    }
  }
  char *getPinnedDevicePointer(char *h_ptr) {
    return (char *)getDevicePointer(h_ptr);
  }
  inline void DeserializeMsgToMirror(NtsVar &mirror_input, char *msg,
                                     VertexId msg_count, bool sync = false) {
    if (msg_count <= 0)
      return;
    ZeroVarMem(mirror_input);
    ValueType *gmb = (ValueType *)getPinnedDevicePointer(msg);
    cuda_stream->deSerializeToGPU(
        mirror_input.packed_accessor<ValueType, 2>().data(), gmb, msg_count,
        feature_size, subgraph->src_range[0], subgraph->src_range[1], false);
  }
  inline void AggMsgToMaster(NtsVar &master_output, char *msg,
                             VertexId msg_count, bool sync = false) {
    if (msg_count <= 0)
      return;
    ValueType *gmb = (ValueType *)getPinnedDevicePointer(msg);
    cuda_stream->aggregate_comm_result_debug(
        master_output.packed_accessor<ValueType, 2>().data(), gmb, msg_count,
        feature_size, subgraph->dst_range[0], subgraph->dst_range[1], false);
  }
  inline void DeviceSynchronize() { cuda_stream->CUDA_DEVICE_SYNCHRONIZE(); }
  void SerializeMirrorToMsg(ValueType *th, torch::Tensor &td,
                            bool sync = false) {
    cuda_stream->move_result_out(th + (subgraph->src_range[0] * feature_size),
                                 td.packed_accessor<ValueType, 2>().data(),
                                 subgraph->src_range[0], subgraph->src_range[1],
                                 feature_size, sync);
  }
#endif

  void ZeroVar(NtsVar &t) { t.zero_(); }
  // zero_buffer(local_data_buffer, (partition_offset[partition_id + 1] -
  // partition_offset[partition_id]) * (feature_size));
  void ZeroVarMem(NtsVar &t, DeviceLocation dl = GPU_T) {
#if CUDA_ENABLE
    if (dl == GPU_T)
      zero_buffer(t.packed_accessor<ValueType, 2>().data(),
                  t.size(0) * t.size(1));
    else
#endif
        if (dl = CPU_T)
      memset(t.accessor<ValueType, 2>().data(), 0, t.size(0) * t.size(1));
    else {
      printf("ZeroVarMem Error\n");
    }
  }

  inline torch::Tensor
  NewKeyTensor(torch::Tensor &mould,
               torch::DeviceType location = torch::DeviceType::CUDA,
               int device_id = 0) {
#if CUDA_ENABLE
    if (torch::DeviceType::CUDA == location) {
      return torch::zeros_like(
          mould,
          at::TensorOptions().device_index(device_id).requires_grad(true).dtype(
              torch::kFloat));
    } else
#endif
    {
      // assert(!torch::DeviceType location==torch::DeviceType::CUDA);
      return torch::zeros_like(
          mould, at::TensorOptions().requires_grad(true).dtype(torch::kFloat));
    }
  }
  inline torch::Tensor
  NewKeyTensor(at::IntArrayRef size,
               torch::DeviceType location = torch::DeviceType::CUDA,
               int device_id = 0) {
#if CUDA_ENABLE
    if (torch::DeviceType::CUDA == location) {
      return torch::zeros(
          size,
          at::TensorOptions().device_index(device_id).requires_grad(true).dtype(
              torch::kFloat));
    } else
#endif
    {
      return torch::zeros(
          size, at::TensorOptions().requires_grad(true).dtype(torch::kFloat));
    }
  }

  inline torch::Tensor
  NewLeafTensor(torch::Tensor &mould,
                torch::DeviceType location = torch::DeviceType::CUDA,
                int device_id = 0) {
#if CUDA_ENABLE
    if (torch::DeviceType::CUDA == location) {
      return torch::zeros_like(
          mould,
          at::TensorOptions().device_index(device_id).dtype(torch::kFloat));
    } else
#endif
    {
      return torch::zeros_like(mould, at::TensorOptions().dtype(torch::kFloat));
    }
  }
  inline torch::Tensor
  NewLeafTensor(at::IntArrayRef size,
                torch::DeviceType location = torch::DeviceType::CUDA,
                int device_id = 0) {
#if CUDA_ENABLE
    if (torch::DeviceType::CUDA == location) {
      return torch::zeros(
          size,
          at::TensorOptions().device_index(device_id).dtype(torch::kFloat));
    } else
#endif
    {
      return torch::zeros(size, at::TensorOptions().dtype(torch::kFloat));
    }
  }
  inline torch::Tensor
  NewKeyTensor(ValueType *data, at::IntArrayRef size,
               torch::DeviceType location = torch::DeviceType::CUDA,
               int device_id = 0) {
#if CUDA_ENABLE
    if (torch::DeviceType::CUDA == location) {
      return torch::from_blob(
          data, size,
          at::TensorOptions().requires_grad(true).device_index(device_id).dtype(
              torch::kFloat));
    } else
#endif
    {
      return torch::from_blob(
          data, size,
          at::TensorOptions().requires_grad(true).dtype(torch::kFloat));
    }
  }
  inline torch::Tensor
  NewLeafTensor(ValueType *data, at::IntArrayRef size,
                torch::DeviceType location = torch::DeviceType::CUDA,
                int device_id = 0) {
#if CUDA_ENABLE
    if (torch::DeviceType::CUDA == location) {
      return torch::from_blob(
          data, size,
          at::TensorOptions().device_index(device_id).dtype(torch::kFloat));
    } else
#endif
    {
      return torch::from_blob(data, size,
                              at::TensorOptions().dtype(torch::kFloat));
    }
  }
  inline torch::Tensor NewLeafKLongTensor(long *data, at::IntArrayRef size) {
    return torch::from_blob(data, size,
                            at::TensorOptions().dtype(torch::kLong));
  }
  inline torch::Tensor NewLeafKLongTensor(at::IntArrayRef size) {
    return torch::zeros(size,
                            at::TensorOptions().dtype(torch::kLong));
  }
  inline torch::Tensor NewLeafKIntTensor(int *data, at::IntArrayRef size) {
    return torch::from_blob(data, size,
                            at::TensorOptions().dtype(torch::kInt32));
  }
  inline torch::Tensor
  NewOnesTensor(at::IntArrayRef size,
                torch::DeviceType location = torch::DeviceType::CUDA,
                int device_id = 0) {
#if CUDA_ENABLE
    if (torch::DeviceType::CUDA == location) {
      return torch::ones(
          size,
          at::TensorOptions().device_index(device_id).dtype(torch::kFloat));
    } else
#endif
    {
      return torch::ones(size, at::TensorOptions().dtype(torch::kFloat));
    }
  }
  inline ValueType *
  getWritableBuffer(torch::Tensor &T_var,
                    torch::DeviceType location = torch::DeviceType::CUDA) {
#if CUDA_ENABLE
    if (torch::DeviceType::CUDA == location) {
      return T_var.packed_accessor<ValueType, 2>().data();
    } else
#endif
    {
      return T_var.accessor<ValueType, 2>().data();
    }
  }

  std::string Encode(std::string name, int layer) {
    return name.append("_").append(std::to_string(layer));
  }

  std::string VarEncode(std::string name) {
    return name.append("_")
        .append(std::to_string(current_process_layer))
        .append("_")
        .append(std::to_string(current_process_partition_id));
  }

  inline int BYSRC() { return 0; }
  inline int BYDST() { return 1; }
  ValueType *recv_cached_buffer;
  ValueType *input_tensor_buffer;
  long *src;
  long *dst;
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
#if CUDA_ENABLE
  Cuda_Stream *cuda_stream;
#endif
  CSC_segment_pinned *subgraph;
  std::map<std::string, torch::Tensor> KeyVar; // key roles in the compute graph
  std::map<std::string, torch::Tensor>
      InterVar; // key roles in the compute graph
  std::map<std::string, torch::Tensor> CacheVar; // used for caching data;
  RuntimeInfo *rtminfo;
  AGGTYPE aggtype;
  // src_input.cpu() dst_input.cpu()
};
class FeatureCache{
public:
  
  //std::string caching_policy;
  VertexId policy_number;
  //0 simple_chunk
  //1 simple_full
  //2 standard_chunk
  //3 standard_chunk
  FeatureCache(VertexId policy,VertexId partitions_){
      policy_number=policy;
      partitions=partitions_;
      if(policy_number==0){
          cached_feature_for_chunks.resize(partitions,nullptr);
          cached_count.resize(partitions,0);
      }else if(policy_number==1){
          ;
      }
  }
#if CUDA_ENABLE
  void push_chunk_for_cuda(VertexId partition_id, VertexId count,VertexId f_size, char *message_ptr){
      cached_count[partition_id]=count;
      cached_feature_for_chunks[partition_id]=(char*)cudaMallocPinned(count*(f_size*sizeof(ValueType)+sizeof(VertexId)));
      memcpy(cached_feature_for_chunks[partition_id],message_ptr,count*(f_size*sizeof(ValueType)+sizeof(VertexId)));
  }
#endif
  void push_chunk(VertexId partition_id, VertexId count,VertexId f_size, char *message_ptr){
      cached_count[partition_id]=count;
      cached_feature_for_chunks[partition_id]=(char*)malloc(count*(f_size*sizeof(ValueType)+sizeof(VertexId)));
      memcpy(cached_feature_for_chunks[partition_id],message_ptr,count*(f_size*sizeof(ValueType)+sizeof(VertexId)));
  }
  void push_mirror(NtsVar &mirrors){
      cached_feature_for_mirrors=mirrors;
  }
  char* get_chunk_data(VertexId partition_id){
      return cached_feature_for_chunks[partition_id];
  }
  VertexId get_chunk_count(VertexId partition_id){
      return cached_count[partition_id];
  }
  NtsVar get_mirror(){
      return cached_feature_for_mirrors;
  }
  //for chunk-based processing
      std::vector<char*>cached_feature_for_chunks;
      std::vector<VertexId>cached_count;
  VertexId partitions;
  
  //for whole graph processing
  NtsVar cached_feature_for_mirrors;
};

struct CachedData {
public:
  CachedData() { ; }
  CachedData(int partitions, int layers, bool scale = false) {
    data_scale = scale;
    NtsVar s;
    for (int i = 0; i < layers; i++) {
      std::vector<NtsVar> tmp;
      for (int j = 0; j < partitions; j++) {
        tmp.push_back(s);
      }
      if (0 == scale) {
        mirror_input.push_back(tmp);
        message.push_back(tmp);
      } else {
        mirror_input_cpu.push_back(tmp);
        message_cpu.push_back(tmp);
      }
    }
  }
  // handling small dataset
  // NtsVar [layer][partitions]
  std::vector<std::vector<NtsVar>> mirror_input;
  std::vector<std::vector<NtsVar>> message;

  // large dataset
  std::vector<std::vector<NtsVar>> mirror_input_cpu;
  std::vector<std::vector<NtsVar>> message_cpu;
  bool data_scale;
};

struct Parameter : torch::nn::Module {
  NtsVar W;
  NtsVar M;
  NtsVar V;
  NtsVar M_GPU;
  NtsVar V_GPU;
  ValueType *W_from;
  ValueType *w_gradient_buffer;
  Network_simple<ValueType> *network_simple;
  int row, col;
  NtsVar W_gradient;
  NtsVar W_g;
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
  Parameter(size_t w, size_t h, ValueType alpha_, ValueType beta1_,
            ValueType beta2_, ValueType epsilon_, ValueType weight_decay_) {
    row = w;
    col = h;
    ValueType scale = sqrt(6.0 / (w + h));
    W = register_parameter("W",
                           (2 * scale) * torch::rand({w, h}, torch::kFloat) -
                               scale * torch::ones({w, h}, torch::kFloat));
    //	ValueType scale=sqrt(6.0/(w+h));
    //	W=(2*scale)*W-scale;
    W_from = new ValueType[w * h];
    w_gradient_buffer = new ValueType[w * h];
    memset(w_gradient_buffer, 0, sizeof(ValueType) * w * h);
    W_gradient = torch::from_blob(w_gradient_buffer, {w, h}, torch::kFloat);
    network_simple = new Network_simple<ValueType>(row, col);
    M = torch::zeros({w, h}, torch::kFloat);
    V = torch::zeros({w, h}, torch::kFloat);
    alpha = alpha_;
    beta1 = beta1_;
    beta2 = beta2_;
    epsilon = epsilon_;
    alpha_t = alpha_;
    beta1_t = beta1_;
    beta2_t = beta2_;
    epsilon_t = epsilon_;
    weight_decay = weight_decay_;
    curr_epoch = 0;
    decay_epoch = -1;
  }
  Parameter(size_t w, size_t h, ValueType l_r_ = 0.01,
            ValueType weight_decay_ = 0.05) {
    alpha = 0.0;
    row = w;
    col = h;
    ValueType scale = sqrt(6.0 / (w + h));
    W = register_parameter("W",
                           (2 * scale) * torch::rand({w, h}, torch::kFloat) -
                               scale * torch::ones({w, h}, torch::kFloat));
    W_from = new ValueType[w * h];
    w_gradient_buffer = new ValueType[w * h];
    memset(w_gradient_buffer, 0, sizeof(ValueType) * w * h);
    W_gradient = torch::from_blob(w_gradient_buffer, {w, h}, torch::kFloat);
    network_simple = new Network_simple<ValueType>(row, col);
    weight_decay = weight_decay;
    l_r = l_r_;
    curr_epoch = 0;
    decay_epoch = -1;
  }

  // root will broadcast it's parameter to other process
  // to synchronize the model
  void init_parameter() {
    network_simple->broadcast(W.accessor<ValueType, 2>().data());
  }
  void all_reduce_to_gradient(NtsVar from) {
    W_gradient.set_data(from);
    network_simple->all_reduce_sum(W_gradient.accessor<ValueType, 2>().data());
  }
  void set_decay(ValueType decay_rate_, ValueType decay_epoch_) {
    decay_rate = decay_rate_;
    decay_epoch = decay_epoch_;
  }
  void next() {
    if (decay_epoch != -1 &&
        (curr_epoch != 0 && curr_epoch % decay_epoch == 0)) {
      alpha_t *= decay_rate;
    }
    alpha = alpha_t * sqrt(1 - beta2) / (1 - beta1);
    beta1 *= beta1_t;
    beta2 *= beta2_t;
    curr_epoch++;
  }
  NtsVar forward(NtsVar x) {

    NtsVar x1 = x.mm(W);
    return x1;
  }
  void learnC2C_with_decay_Adam() {
    NtsVar S = W.detach();
    W_g = W_gradient + weight_decay * S;
    M = beta1 * M + (1 - beta1) * W_g;
    V = beta2 * V + (1 - beta2) * W_g * W_g;
    // NtsVar a = W - alpha*M/(torch::sqrt(V)+epsilon);
    W.set_data(W - alpha * M / (torch::sqrt(V) + epsilon));
  }
  void learnC2C_with_decay_SGD(ValueType learning_rate,
                               ValueType weight_decay) {
    NtsVar tmp = W_gradient;
    NtsVar a = (W - (tmp * learning_rate)) * (1 - weight_decay);
    W.set_data(a);
  }

#if CUDA_ENABLE
  void Adam_to_GPU() {
    M_GPU = M.cuda();
    V_GPU = V.cuda();
    W_g = W_gradient.cuda();
  }
  void learnC2G(ValueType learning_rate) {
    NtsVar tmp = W_gradient.cuda();
    NtsVar a = (W - (tmp * learning_rate));
    W.set_data(a);
  }

  void learnC2G_with_decay(ValueType learning_rate, ValueType weight_decay) {
    NtsVar tmp = W_gradient.cuda();
    NtsVar a = (W - (tmp * learning_rate)) * (1 - weight_decay);
    W.set_data(a);
  }
  void learnC2G_with_decay_Adam() {
    W_g.set_data(W);
    W_g = W_g * weight_decay;
    W_g = W_g + W_gradient.cuda(); //+weight_decay;
    M_GPU = beta1 * M_GPU + (1 - beta1) * W_g;
    V_GPU = beta2 * V_GPU + (1 - beta2) * W_g * W_g;
    W.set_data(W - alpha * M_GPU / (torch::sqrt(V_GPU) + epsilon));
  }
  void learn_local_with_decay_Adam() {
    W_g.set_data(W);
    W_g = W_g * weight_decay;
    W_g = W_g + W.grad(); //+weight_decay;
    M_GPU = beta1 * M_GPU + (1 - beta1) * W_g;
    V_GPU = beta2 * V_GPU + (1 - beta2) * W_g * W_g;
    W.set_data(W - alpha * M_GPU / (torch::sqrt(V_GPU) + epsilon));
  }
#endif
};

#endif

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

#ifndef GRAPH_HPP
#define GRAPH_HPP
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

#include "torch/torch.h"
#include "torch/csrc/autograd/generated/variable_factories.h"
#include "torch/nn/module.h"
#include "torch/csrc/api/include/torch/cuda.h"
#include "ATen/ATen.h"

#define CHUNK_LENGTH 32768 //32768
#define REPLICATE_THRESHOLD 40

typedef struct dataaa
{
  VertexId data[REPLICATE_THRESHOLD];
} datum;

enum ThreadStatus
{
  WORKING,
  STEALING
};

enum MessageTag
{
  ShuffleGraph,
  PassMessage,
  GatherVertexArray
};

struct ThreadState
{
  VertexId curr;
  VertexId end;
  ThreadStatus status;
};

template <typename MsgData>
struct MsgUnit
{
  VertexId vertex;
  MsgData msg_data;
} __attribute__((packed));

struct MsgUnit_buffer
{
  VertexId vertex;
  int msg_data;
} __attribute__((packed));

struct MessageBuffer
{
  size_t capacity;
  int count; // the actual size (i.e. bytes) should be sizeof(element) * count
  char *data;
  bool pinned;
  MessageBuffer()
  {
    capacity = 0;
    count = 0;
    data = NULL;
  }
  void init(int socket_id)
  {
    capacity = 4096;
    count = 0;
    data = (char *)numa_alloc_onnode(capacity, socket_id);
  }
  void resize(size_t new_capacity)
  {
    if (new_capacity > capacity)
    {
      char *new_data = NULL;
      new_data = (char *)numa_realloc(data, capacity, new_capacity);
      //printf("alloc success%d  %d\n",new_capacity, new_data != NULL);
      assert(new_data != NULL);
      data = new_data;
      capacity = new_capacity; //**********************************************************************8
      pinned = false;
    }
  }
  void resize_pinned(long new_capacity)
  {
    if ((new_capacity > capacity))
    {
      if (!pinned)
        numa_free(data, capacity);
      else
        cudaFreeHost(data);
      char *new_data = NULL;
      new_data = (char *)cudaMallocPinned(new_capacity);
      //Wassert(new_data!=NULL);
      data = new_data;
      capacity = new_capacity; //**********************************************************************8
      pinned = true;
    }
  }
  //template <typename MsgData>
  int *getMsgUnit(int i, int msg_unit_size)
  {
    (int *)this->data + i *(msg_unit_size + sizeof(MsgUnit_buffer));
  }
  template <typename t_v>
  t_v *getMsg_Data(int i, int msg_unit_size)
  {
    (t_v *)this->data + i *(msg_unit_size + sizeof(MsgUnit_buffer)) + sizeof(MsgUnit_buffer);
  }
  template <typename t_v>
  void set_Msg_Data(int i, int msg_unit_size, t_v *buffer)
  {
    memcpy(this->data + i * (msg_unit_size + sizeof(MsgUnit_buffer)) + sizeof(MsgUnit_buffer), buffer, msg_unit_size);
  }
};

typedef struct graph_Tensor_Segment
{
  torch::Tensor column_offset; //VertexNumber
  torch::Tensor row_indices;   //edge_size
  torch::Tensor edge_weight;   //edge_size
  int edge_size;
  int batch_size;
  int input_size;
  int output_size;
  int feature_size;
  int src_range[2];
  int dst_range[2];
  float *weight_buffer;
} CSC_segment;

typedef struct graph_Tensor_Segment1
{
  torch::Tensor row_offset; //VertexNumber
  torch::Tensor column_indices;   //edge_size
  torch::Tensor edge_weight;   //edge_size
  int edge_size;
  int batch_size;
  int input_size;
  int output_size;
  int feature_size;
  int src_range[2];
  int dst_range[2];
  float *weight_buffer;
} CSR_segment;

typedef struct graph_Tensor_Segment_pinned
{
  VertexId *column_offset;     //VertexNumber
  VertexId *row_indices;       //edge_size also the source nodes
  VertexId *row_offset;     //VertexNumber
  VertexId *column_indices;  
  long *source;
  long *source_gpu;
  long *destination;
  long *destination_gpu;
  float *edge_weight;          //edge_size
  float *edge_weight_backward;
  VertexId *column_offset_gpu; //VertexNumber
  VertexId *row_indices_gpu;   //edge_size
  float *edge_weight_gpu;      //edge_size
  VertexId *row_offset_gpu; //VertexNumber
  VertexId *column_indices_gpu;   //edge_size
  float *edge_weight_backward_gpu;      //edge_size
  int edge_size;
  int batch_size;
  int batch_size_forward;
  int batch_size_backward;
  int input_size;
  int output_size;
  int feature_size;
  int src_range[2];
  int dst_range[2];
} CSC_segment_pinned;




typedef struct rep_graph
{
  VertexId *src_rep;                 //rep_edge_size
  VertexId *dst_rep;                 //rep_edge_size
  std::vector<VertexId> src_rep_vec; //rep_edge_size
  std::vector<VertexId> dst_rep_vec; //rep_edge_size
  VertexId *src_map;                 //rep_node_size*2
  VertexId *dst_map;
  float *weight_rep; //rep_edge_size
  VertexId rep_edge_size;
  VertexId rep_node_size;
  VertexId feature_size;
  float *rep_feature;
  float *rep_feature_gpu_buffer;
  float *output_buffer_gpu;
  VertexId output_size;
  VertexId *src_rep_gpu;
  VertexId *dst_rep_gpu;
  VertexId *src_map_gpu;
  float *weight_rep_gpu;

} graph_replication;
typedef struct InputInfo
{
  size_t vertices;
  bool overlap;
  bool process_local;
  bool with_weight;
  size_t repthreshold;
  std::string layer_string;
  std::string feature_file;
  std::string edge_file;
  bool with_cuda;
} inputinfo;


typedef struct runtimeInfo
{
  bool process_local;
  bool with_cuda;
  bool process_overlap;
  bool with_weight;
  bool reduce_comm;
  size_t epoch;
  size_t curr_layer;
  size_t embedding_size;
  bool copy_data;
  bool forward;

} runtimeinfo;
typedef struct GNNContext
{
  std::vector<int> layer_size;
  size_t max_layer;
  size_t label_num;
  size_t p_id;
  size_t p_v_s;
  size_t p_v_e;
  size_t w_num;   //workernum
  size_t l_v_num; //local |V|
  size_t l_e_num; //local |E|
} gnncontext;
typedef struct BlockInfomation
{
  std::vector<VertexId> vertex;
  std::vector<VertexId> global_index;
  std::vector<VertexId> local_index;
  std::vector<VertexId> block_index; //num of blocks+1
  VertexId *vertex_gpu_buffer;
  VertexId *index_gpu_buffer;
  VertexId max_buffer_size; //for alloc

} BlockInfo;

class EdgeNNModule{
public:
    EdgeNNModule(){
         ;
    }
    void InitBlock(CSC_segment_pinned* graph_partition, VertexId feature_size_, 
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
                               subgraph->batch_size,
                               feature_size, false);
    cuda_stream->CUDA_DEVICE_SYNCHRONIZE(); 
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
                               false);
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
                               (VertexId)subgraph->batch_size,
                               (VertexId)output_size,
                               false);
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
                               (VertexId)subgraph->batch_size,
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
    inline torch::Tensor DeSerializeFromCPU(std::string name,bool to_gpu=true,int device_id=0){
        torch::Tensor var_cpu=CacheVar[VarEncode(name)];
        if(to_gpu){
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
    inline torch::Tensor NewKeyTensor(torch::Tensor &mould, bool GPU=true, int device_id=0){
        
        if(GPU){
           return torch::zeros_like(mould,at::TensorOptions().device_index(device_id).requires_grad(true).dtype(torch::kFloat));  
        }else{
           return torch::zeros_like(mould,at::TensorOptions().requires_grad(true).dtype(torch::kFloat));  
        }
    }
     inline torch::Tensor NewKeyTensor(at::IntArrayRef size, bool GPU=true, int device_id=0){
        if(GPU){
           return torch::zeros(size,at::TensorOptions().device_index(device_id).requires_grad(true).dtype(torch::kFloat));
        }else{
           return torch::zeros(size,at::TensorOptions().requires_grad(true).dtype(torch::kFloat));
        }
    }
     
     inline torch::Tensor NewLeafTensor(torch::Tensor &mould, bool GPU=true, int device_id=0){
        
        if(GPU){
             return torch::zeros_like(mould,at::TensorOptions().device_index(device_id).dtype(torch::kFloat));  
        }else{
             return torch::zeros_like(mould,at::TensorOptions().dtype(torch::kFloat));  
        }
    }
     inline torch::Tensor NewLeafTensor(at::IntArrayRef size, bool GPU=true, int device_id=0){
        if(GPU){
           return torch::zeros(size,at::TensorOptions().device_index(device_id).dtype(torch::kFloat));
        }else{
           return torch::zeros(size,at::TensorOptions().dtype(torch::kFloat));
        }
    }
    
    inline float* getWritableBuffer(torch::Tensor &T_var,bool GPU=true){
        if(GPU){
        return T_var.packed_accessor<float,2>().data();
        }else{
        return T_var.packed_accessor<float,2>().data();    
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
    //src_input.cpu() dst_input.cpu()
};


template <typename EdgeData = Empty>
class Graph
{
public:
  /*partitions for streaming GPU processing*/
  gnncontext *gnnctx;
  runtimeinfo *rtminfo;
  inputinfo *config;
  BlockInfo *blockinfo;
  int featureSize;

  //graph reorganization
  COOChunk *_graph_cpu;
  COOChunk *_graph_cpu_backward;
  std::vector<COOChunk *> graph_shard;
  std::vector<COOChunk *> graph_shard_backward;
  std::string filename;

  int partition_id;
  int partitions;

  size_t alpha;

  int threads;
  int sockets;
  int threads_per_socket;

  size_t edge_data_size;
  size_t unit_size;
  size_t edge_unit_size;
  int message_unit_size;

  bool symmetric;
  VertexId vertices; //
  EdgeId edges;      //
  EdgeId local_edges;
  VertexId *out_degree;         // VertexId [vertices]; numa-aware
  VertexId *in_degree;          // VertexId [vertices]; numa-aware
  VertexId *in_degree_backward; //_new_0

  VertexId *out_degree_for_backward; // VertexId [vertices]; numa-aware
  VertexId *in_degree_for_backward;  // VertexId [vertices]; numa-aware

  VertexId *partition_offset;       // VertexId [partitions+1]
  VertexId *local_partition_offset; // VertexId [sockets+1]

  VertexId owned_vertices;
  EdgeId *outgoing_edges;          // EdgeId [sockets]
  EdgeId *incoming_edges;          // EdgeId [sockets]
  EdgeId *incoming_edges_backward; //EdgeId[sockets];_new_1

  Bitmap **incoming_adj_bitmap;
  Bitmap **incoming_adj_bitmap_backward;          //_new_2
  EdgeId **incoming_adj_index;                    // EdgeId [sockets] [vertices+1]; numa-aware
  EdgeId **incoming_adj_index_backward;           //_new_3
  AdjUnit<EdgeData> **incoming_adj_list;          // AdjUnit<EdgeData> [sockets] [vertices+1]; numa-aware
  AdjUnit<EdgeData> **incoming_adj_list_backward; //_new_4
  Bitmap **outgoing_adj_bitmap;
  EdgeId **outgoing_adj_index;           // EdgeId [sockets] [vertices+1]; numa-aware
  AdjUnit<EdgeData> **outgoing_adj_list; // AdjUnit<EdgeData> [sockets] [vertices+1]; numa-aware

  VertexId *compressed_incoming_adj_vertices;
  VertexId *compressed_incoming_adj_vertices_backward;             //_new_5
  CompressedAdjIndexUnit **compressed_incoming_adj_index;          // CompressedAdjIndexUnit [sockets] [...+1]; numa-aware
  CompressedAdjIndexUnit **compressed_incoming_adj_index_backward; //_new_6

  VertexId *compressed_outgoing_adj_vertices;
  CompressedAdjIndexUnit **compressed_outgoing_adj_index; // CompressedAdjIndexUnit [sockets] [...+1]; numa-aware

  ThreadState **thread_state;       // ThreadState* [threads]; numa-aware
  ThreadState **tuned_chunks_dense; // ThreadState [partitions][threads];
  ThreadState **tuned_chunks_dense_backward;
  ThreadState **tuned_chunks_sparse; // ThreadState [partitions][threads];

  size_t local_send_buffer_limit;
  MessageBuffer **local_send_buffer; // MessageBuffer* [threads]; numa-aware

  int current_send_part_id;
  MessageBuffer ***send_buffer; // MessageBuffer* [partitions] [sockets]; numa-aware
  MessageBuffer ***recv_buffer; // MessageBuffer* [partitions] [sockets]; numa-aware
  
  //Edgefunction
  EdgeNNModule *EdgeOp;

  //replication
  int replication_threshold;
  graph_replication *graph_rep;
  bool *out_going_index;
  bool *out_going_index_gpu;
  std::vector<Bitmap *> HasRepVtx;
  std::vector<Bitmap *> RepVtx;
  Bitmap *outGoing;
  std::vector<std::vector<VertexId>> EdgeRemote2Local;
  std::vector<std::vector<VertexId>> EdgeRemote2Remote;
  std::vector<std::vector<VertexId>> RemoteVteIndex;
  std::vector<ValueType> RepFeatures;
  float *rep_output_buffer;
  VertexId *output_map;
  VertexId rep_output_size;
  VertexId **message_write_offset;
  VertexId **message_amount;
  VertexId encode_partition;

  //overlap
  VertexId *column_offset_intergate;
  VertexId *row_indices_intergate;

  float *weight_gpu_intergate;
  torch::Tensor output_gpu_buffered;

  //timer
  double all_wait_time;
  double all_overlap_time;
  double all_compute_time;
  double all_movein_time;
  double all_moveout_time;
  double all_kernel_time;
  double all_recv_copy_time;
  double all_recv_kernel_time;
  double all_recv_wait_time;
  double all_recv_thread_join_time;
  double all_cuda_sync_time;
  double all_replication_time;
  double local_replication_time;

  Graph()
  {
    threads = numa_num_configured_cpus();
    sockets = numa_num_configured_nodes();
    threads_per_socket = threads / sockets;
    all_wait_time = 0.0;
    all_overlap_time = 0.0;
    all_compute_time = 0.0;
    all_movein_time = 0.0;
    all_kernel_time = 0.0;
    all_moveout_time = 0.0;
    all_recv_copy_time = 0.0;
    all_recv_kernel_time = 0.0;
    all_recv_wait_time = 0.0;
    all_recv_thread_join_time = 0.0;
    all_cuda_sync_time = 0;
    all_replication_time = 0.0;
    local_replication_time = 0.0;
    replication_threshold = 0;
    init();
    config = new inputinfo;
    EdgeOp=new EdgeNNModule();
    encode_partition=-1;
  }
  std::string VarEncode(std::string name){
      return name.append("_").append(std::to_string(rtminfo->curr_layer)).append("_").append(std::to_string(encode_partition));
  }
  void init_blockinfo()
  {
    blockinfo = new BlockInfo[2];

    for (int l = 0; l < 2; l++)
    {
      VertexId offset = 0;
      VertexId current_partition = 0;

      blockinfo[l].block_index.clear();
      blockinfo[l].local_index.clear();
      blockinfo[l].global_index.clear();
      blockinfo[l].vertex.clear();
      blockinfo[l].block_index.resize(partitions + 1, 0);
      blockinfo[l].max_buffer_size = 0;
      //printf("%d\n", blockinfo[l].block_index.size());

      for (VertexId i = 0; i < vertices; i++)
      {
        if (i > partition_offset[current_partition + 1])
        {
          current_partition++;
          blockinfo[l].block_index[current_partition] = offset;
        }
        if (!RepVtx[l]->get_bit(i))
        {
          blockinfo[l].global_index.push_back(offset);
          blockinfo[l].vertex.push_back(i);
          offset++;
        }
      }
      blockinfo[l].block_index[partitions] = blockinfo[l].global_index.size();

      for (int i = 0; i < partitions; i++)
      {
        blockinfo[l].max_buffer_size = std::max(blockinfo[l].max_buffer_size, blockinfo[l].block_index[i + 1] - blockinfo[l].block_index[i]);
      }
      current_partition = 0;
      for (VertexId i = 0; i < blockinfo[l].global_index.size(); i++)
      {
        if (i > blockinfo[l].block_index[current_partition + 1])
          current_partition++;
        blockinfo[l].local_index.push_back(blockinfo[l].global_index[i] - blockinfo[l].block_index[current_partition]);
      }

      blockinfo[l].index_gpu_buffer = (VertexId *)cudaMallocGPU(sizeof(VertexId) * blockinfo[l].local_index.size());
      blockinfo[l].vertex_gpu_buffer = (VertexId *)cudaMallocGPU(sizeof(VertexId) * blockinfo[l].local_index.size());
      move_edge_in(blockinfo[l].index_gpu_buffer,
                   &(blockinfo[l].local_index[0]),
                   0,
                   blockinfo[l].local_index.size(),
                   1,
                   false);
      move_edge_in(blockinfo[l].vertex_gpu_buffer,
                   &(blockinfo[l].vertex[0]),
                   0,
                   blockinfo[l].vertex.size(),
                   1,
                   false);
      // for (int i = 0; i < blockinfo[l].global_index.size(); i++)
      // {
      //   if (blockinfo[l].local_index[i] > blockinfo[l].max_buffer_size)
      //     printf("what hell%d\n", blockinfo[l].local_index[i]);
      // }
    }
  }
  void init_message_map_amount()
  {
    if (partition_id == 0)
      printf("GNNmini::Init Message\n");
    for (int layer = 0; layer < gnnctx->layer_size.size() - 1; layer++)
    {

      process_edges_simple<int, int>(                            // For EACH Vertex Processing
          [&](VertexId src, VertexAdjList<Empty> outgoing_adj) { //pull
            if (!RepVtx[layer]->get_bit(src))
            {
              message_amount[layer][get_partition_id(src)] += 1;
              message_write_offset[layer][src] = 1;
            }

          },
          [&](VertexId src, int msg) {
            return 0;
          });

      for (int p = 0; p < partitions; p++)
      {

        message_write_offset[layer][partition_offset[p]] = 0;
        for (VertexId src = partition_offset[p] + 1; src < partition_offset[p + 1]; src++)
        {
          message_write_offset[layer][src] += message_write_offset[layer][src - 1];
        }
      }
    }
  }
  void init_rtminfo()
  {
    rtminfo = new runtimeinfo;
    rtminfo->process_local = false;
    rtminfo->process_overlap = false;
    rtminfo->epoch = -1;
    rtminfo->curr_layer = -1;
    rtminfo->embedding_size = -1;
    rtminfo->copy_data = false;
    rtminfo->with_cuda = false;
  }
  void init_gnnctx(std::string layer_string)
  {
    gnnctx = new gnncontext;
    std::stringstream ss(layer_string);
    std::string number;
    gnnctx->layer_size.clear();
    gnnctx->max_layer = 0;
    while (std::getline(ss, number, '-'))
    {
      gnnctx->layer_size.push_back(std::stoi(number));
      gnnctx->max_layer = std::max(gnnctx->max_layer, (size_t)std::stoi(number));
      //printf("layers %d\n", std::stoi(number));
    }
    gnnctx->label_num = gnnctx->layer_size[gnnctx->layer_size.size() - 1];

    gnnctx->p_id = partition_id;
    gnnctx->p_v_e = partition_offset[partition_id + 1];
    gnnctx->p_v_s = partition_offset[partition_id];
    gnnctx->w_num = partitions;
    gnnctx->l_v_num = gnnctx->p_v_e - gnnctx->p_v_s;
    message_write_offset = new VertexId *[gnnctx->layer_size.size()];
    message_amount = new VertexId *[gnnctx->layer_size.size()];
    if (partition_id == 0)
      printf("layer_size %d\n", gnnctx->layer_size.size());
    for (int i = 0; i < gnnctx->layer_size.size(); i++)
    {
      message_write_offset[i] = new VertexId[vertices];
      message_amount[i] = new VertexId[partitions];
      memset(message_write_offset[i], 0, sizeof(VertexId) * vertices);
      memset(message_amount[i], 0, sizeof(VertexId) * partitions);
    }
  }

  inline int get_socket_id(int thread_id)
  {
    return thread_id / threads_per_socket;
  }

  inline int get_socket_offset(int thread_id)
  {
    return thread_id % threads_per_socket;
  }

  void init()
  {
    edge_data_size = std::is_same<EdgeData, Empty>::value ? 0 : sizeof(EdgeData);
    unit_size = sizeof(VertexId) + edge_data_size;
    edge_unit_size = sizeof(VertexId) + unit_size;

    assert(numa_available() != -1);
    assert(sizeof(unsigned long) == 8); // assume unsigned long is 64-bit

    char nodestring[sockets * 2 + 1];
    for (int i = 0; i < sockets * 2 + 1; i++)
      nodestring[i] = '\0';
    nodestring[0] = '0';
    for (int s_i = 1; s_i < sockets; s_i++)
    {
      nodestring[s_i * 2 - 1] = ',';
      nodestring[s_i * 2] = '0' + s_i;
    }
    struct bitmask *nodemask = numa_parse_nodestring(nodestring);
    numa_set_interleave_mask(nodemask);

    omp_set_dynamic(0);
    omp_set_num_threads(threads);
    thread_state = new ThreadState *[threads];
    local_send_buffer_limit = 16;
    local_send_buffer = new MessageBuffer *[threads];
    for (int t_i = 0; t_i < threads; t_i++)
    {
      thread_state[t_i] = (ThreadState *)numa_alloc_onnode(sizeof(ThreadState), get_socket_id(t_i));
      local_send_buffer[t_i] = (MessageBuffer *)numa_alloc_onnode(sizeof(MessageBuffer), get_socket_id(t_i));
      local_send_buffer[t_i]->init(get_socket_id(t_i));
    }
#pragma omp parallel for
    for (int t_i = 0; t_i < threads; t_i++)
    {
      int s_i = get_socket_id(t_i);
      assert(numa_run_on_node(s_i) == 0);
#ifdef PRINT_DEBUG_MESSAGES
// printf("thread-%d bound to socket-%d\n", t_i, s_i);
#endif
    }
#ifdef PRINT_DEBUG_MESSAGES
// printf("threads=%d*%d\n", sockets, threads_per_socket);
// printf("interleave on %s\n", nodestring);
#endif

    MPI_Comm_rank(MPI_COMM_WORLD, &partition_id); 
    MPI_Comm_size(MPI_COMM_WORLD, &partitions);  
    send_buffer = new MessageBuffer **[partitions];
    recv_buffer = new MessageBuffer **[partitions];
    for (int i = 0; i < partitions; i++)
    {
      send_buffer[i] = new MessageBuffer *[sockets];
      recv_buffer[i] = new MessageBuffer *[sockets];
      for (int s_i = 0; s_i < sockets; s_i++)
      {
        send_buffer[i][s_i] = (MessageBuffer *)numa_alloc_onnode(sizeof(MessageBuffer), s_i);
        send_buffer[i][s_i]->init(s_i);
        recv_buffer[i][s_i] = (MessageBuffer *)numa_alloc_onnode(sizeof(MessageBuffer), s_i);
        recv_buffer[i][s_i]->init(s_i);
      }
    }

    alpha = 12 * (partitions + 1);
    //alpha = 4;
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // fill a vertex array with a specific value
  template <typename T>
  void fill_vertex_array(T *array, T value)
  {
#pragma omp parallel for
    for (VertexId v_i = partition_offset[partition_id]; v_i < partition_offset[partition_id + 1]; v_i++)
    {
      array[v_i] = value;
    }
  }

  /////////////////////////add by wangqg
  template <typename T>
  void fill_vertex_array_long(T *array, T value, size_t length)
  { //fill_vertex_array
#pragma omp parallel for
    for (VertexId v_i = partition_offset[partition_id]; v_i < partition_offset[partition_id + 1]; v_i++)
    {
      memcpy(array[v_i].data, value.data, length * sizeof(T));
    }
  }

  template <typename T>
  T *alloc_vertex_array(int size)
  { //create a
    char *array = (char *)mmap(NULL, size * sizeof(T) * vertices, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    assert(array != NULL);
    for (int s_i = 0; s_i < sockets; s_i++)
    {
      numa_tonode_memory(array + size * sizeof(T) * local_partition_offset[s_i], size * sizeof(T) * (local_partition_offset[s_i + 1] - local_partition_offset[s_i]), s_i);
    }
    return (T *)array;
  }
  template <typename T>
  T *alloc_vertex_array_local(int size)
  { //create a
    char *array = (char *)mmap(NULL, size * sizeof(T) * (partition_offset[partition_id + 1] - partition_offset[partition_id]),
                               PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    assert(array != NULL);
    for (int s_i = 0; s_i < sockets; s_i++)
    {
      numa_tonode_memory(array + size * sizeof(T) * (local_partition_offset[s_i] - local_partition_offset[0]), size * sizeof(T) * (local_partition_offset[s_i + 1] - local_partition_offset[s_i]), s_i);
    }
    return (T *)array;
  }

  // allocate a numa-aware vertex array
  template <typename T>
  T *alloc_vertex_array()
  { //create a
    char *array = (char *)mmap(NULL, sizeof(T) * vertices, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    assert(array != NULL);
    for (int s_i = 0; s_i < sockets; s_i++)
    {
      numa_tonode_memory(array + sizeof(T) * local_partition_offset[s_i], sizeof(T) * (local_partition_offset[s_i + 1] - local_partition_offset[s_i]), s_i);
    }
    return (T *)array;
  }

  template <typename T>
  T *alloc_vertex_array_local()
  { //create a
    char *array = (char *)mmap(NULL, sizeof(T) * (partition_offset[partition_id + 1] - partition_offset[partition_id]),
                               PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    assert(array != NULL);
    for (int s_i = 0; s_i < sockets; s_i++)
    {
      numa_tonode_memory(array + sizeof(T) * (local_partition_offset[s_i] - local_partition_offset[0]), sizeof(T) * (local_partition_offset[s_i + 1] - local_partition_offset[s_i]), s_i);
    }
    return (T *)array;
  }

  template <typename T>
  T *alloc_pointer_vertex_array()
  { //create a
    char *array = (char *)mmap(NULL, sizeof(T) * vertices, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    assert(array != NULL);
    for (int s_i = 0; s_i < sockets; s_i++)
    {
      numa_tonode_memory(array + sizeof(T) * local_partition_offset[s_i], sizeof(T) * (local_partition_offset[s_i + 1] - local_partition_offset[s_i]), s_i);
    }
    return (T *)array;
  }

  // deallocate a vertex array
  template <typename T>
  T *dealloc_vertex_array(T *array)
  {
    numa_free(array, sizeof(T) * vertices);
  }

  // allocate a numa-oblivious vertex array
  template <typename T>
  T *alloc_interleaved_vertex_array()
  {
    T *array = (T *)numa_alloc_interleaved(sizeof(T) * vertices);
    assert(array != NULL);
    return array;
  }

  // dump a vertex array to path
  template <typename T>
  void dump_vertex_array(T *array, std::string path)
  { //persistent array to path
    long file_length = sizeof(T) * vertices;
    if (!file_exists(path) || file_size(path) != file_length)
    {
      if (partition_id == 0)
      {
        FILE *fout = fopen(path.c_str(), "wb");
        char *buffer = new char[PAGESIZE];
        for (long offset = 0; offset < file_length;)
        {
          if (file_length - offset >= PAGESIZE)
          {
            fwrite(buffer, 1, PAGESIZE, fout);
            offset += PAGESIZE;
          }
          else
          {
            fwrite(buffer, 1, file_length - offset, fout);
            offset += file_length - offset;
          }
        }
        fclose(fout);
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
    int fd = open(path.c_str(), O_RDWR);
    assert(fd != -1);
    long offset = sizeof(T) * partition_offset[partition_id];
    long end_offset = sizeof(T) * partition_offset[partition_id + 1];
    void *data = (void *)array;
    assert(lseek(fd, offset, SEEK_SET) != -1);
    while (offset < end_offset)
    {
      long bytes = write(fd, data + offset, end_offset - offset);
      assert(bytes != -1);
      offset += bytes;
    }
    assert(close(fd) == 0);
  }

  // restore a vertex array from path
  template <typename T>
  void restore_vertex_array(T *array, std::string path)
  {
    long file_length = sizeof(T) * vertices;
    if (!file_exists(path) || file_size(path) != file_length)
    {
      assert(false);
    }
    int fd = open(path.c_str(), O_RDWR);
    assert(fd != -1);
    long offset = sizeof(T) * partition_offset[partition_id];
    long end_offset = sizeof(T) * partition_offset[partition_id + 1];
    void *data = (void *)array;
    assert(lseek(fd, offset, SEEK_SET) != -1);
    while (offset < end_offset)
    {
      long bytes = read(fd, data + offset, end_offset - offset);
      assert(bytes != -1);
      offset += bytes;
    }
    assert(close(fd) == 0);
  }

  // gather a vertex array
  template <typename T>
  void gather_vertex_array(T *array, int root)
  {
    if (partition_id != root)
    {
      MPI_Send(array + partition_offset[partition_id], sizeof(T) * owned_vertices, MPI_CHAR, root, GatherVertexArray, MPI_COMM_WORLD);
    }
    else
    {
      for (int i = 0; i < partitions; i++)
      {
        if (i == partition_id)
          continue;
        MPI_Status recv_status;
        MPI_Recv(array + partition_offset[i], sizeof(T) * (partition_offset[i + 1] - partition_offset[i]), MPI_CHAR, i, GatherVertexArray, MPI_COMM_WORLD, &recv_status);
        int length;
        MPI_Get_count(&recv_status, MPI_CHAR, &length);
        assert(length == sizeof(T) * (partition_offset[i + 1] - partition_offset[i]));
      }
    }
  }

  // allocate a vertex subset
  VertexSubset *alloc_vertex_subset()
  {
    return new VertexSubset(vertices);
  }

  int get_partition_id(VertexId v_i)
  {
    for (int i = 0; i < partitions; i++)
    {
      if (v_i >= partition_offset[i] && v_i < partition_offset[i + 1])
      {
        return i;
      }
    }
    printf("wrong vertex%d\n", v_i);
    exit(0);
    assert(false);
  }

  int get_local_partition_id(VertexId v_i)
  {
    for (int s_i = 0; s_i < sockets; s_i++)
    {
      if (v_i >= local_partition_offset[s_i] && v_i < local_partition_offset[s_i + 1])
      {
        return s_i;
      }
    }
    std::cout << v_i << std::endl;
    assert(false);
  }

  // load a directed graph and make it undirected
  void load_undirected_from_directed(std::string path, VertexId vertices)
  {
    double prep_time = 0;
    prep_time -= MPI_Wtime();

    symmetric = true;

    MPI_Datatype vid_t = get_mpi_data_type<VertexId>();

    this->vertices = vertices;
    long total_bytes = file_size(path.c_str());
    this->edges = total_bytes / edge_unit_size;
#ifdef PRINT_DEBUG_MESSAGES
    if (partition_id == 0)
    {
      printf("|V| = %u, |E| = %lu\n", vertices, edges);
    }
#endif

    EdgeId read_edges = edges / partitions;
    if (partition_id == partitions - 1)
    {
      read_edges += edges % partitions;
    }
    long bytes_to_read = edge_unit_size * read_edges;
    long read_offset = edge_unit_size * (edges / partitions * partition_id);
    long read_bytes;
    int fin = open(path.c_str(), O_RDONLY);
    EdgeUnit<EdgeData> *read_edge_buffer = new EdgeUnit<EdgeData>[CHUNKSIZE];

    out_degree = alloc_interleaved_vertex_array<VertexId>();
    for (VertexId v_i = 0; v_i < vertices; v_i++)
    {
      out_degree[v_i] = 0;
    }
    assert(lseek(fin, read_offset, SEEK_SET) == read_offset);
    read_bytes = 0;
    while (read_bytes < bytes_to_read)
    {
      long curr_read_bytes;
      if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE)
      {
        curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
      }
      else
      {
        curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
      }
      assert(curr_read_bytes >= 0);
      read_bytes += curr_read_bytes;
      EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
      // #pragma omp parallel for
      for (EdgeId e_i = 0; e_i < curr_read_edges; e_i++)
      {
        VertexId src = read_edge_buffer[e_i].src;
        VertexId dst = read_edge_buffer[e_i].dst;
        __sync_fetch_and_add(&out_degree[src], 1);
        __sync_fetch_and_add(&out_degree[dst], 1);
      }
    }
    MPI_Allreduce(MPI_IN_PLACE, out_degree, vertices, vid_t, MPI_SUM, MPI_COMM_WORLD);

    // locality-aware chunking
    partition_offset = new VertexId[partitions + 1];
    partition_offset[0] = 0;
    EdgeId remained_amount = edges * 2 + EdgeId(vertices) * alpha;
    for (int i = 0; i < partitions; i++)
    {
      VertexId remained_partitions = partitions - i;
      EdgeId expected_chunk_size = remained_amount / remained_partitions;
      if (remained_partitions == 1)
      {
        partition_offset[i + 1] = vertices;
      }
      else
      {
        EdgeId got_edges = 0;
        for (VertexId v_i = partition_offset[i]; v_i < vertices; v_i++)
        {
          got_edges += out_degree[v_i] + alpha;
          if (got_edges > expected_chunk_size)
          {
            partition_offset[i + 1] = v_i;
            break;
          }
        }
        partition_offset[i + 1] = (partition_offset[i + 1]) / PAGESIZE * PAGESIZE; // aligned with pages
      }
      for (VertexId v_i = partition_offset[i]; v_i < partition_offset[i + 1]; v_i++)
      {
        remained_amount -= out_degree[v_i] + alpha;
      }
    }
    assert(partition_offset[partitions] == vertices);
    owned_vertices = partition_offset[partition_id + 1] - partition_offset[partition_id];
    // check consistency of partition boundaries
    VertexId *global_partition_offset = new VertexId[partitions + 1];
    MPI_Allreduce(partition_offset, global_partition_offset, partitions + 1, vid_t, MPI_MAX, MPI_COMM_WORLD);
    for (int i = 0; i <= partitions; i++)
    {
      assert(partition_offset[i] == global_partition_offset[i]);
    }
    MPI_Allreduce(partition_offset, global_partition_offset, partitions + 1, vid_t, MPI_MIN, MPI_COMM_WORLD);
    for (int i = 0; i <= partitions; i++)
    {
      assert(partition_offset[i] == global_partition_offset[i]);
    }
#ifdef PRINT_DEBUG_MESSAGES
    if (partition_id == 0)
    {
      for (int i = 0; i < partitions; i++)
      {
        EdgeId part_out_edges = 0;
        for (VertexId v_i = partition_offset[i]; v_i < partition_offset[i + 1]; v_i++)
        {
          part_out_edges += out_degree[v_i];
        }
        printf("|V'_%d| = %u |E_%d| = %lu\n", i, partition_offset[i + 1] - partition_offset[i], i, part_out_edges);
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    delete[] global_partition_offset;
    {
      // NUMA-aware sub-chunking
      local_partition_offset = new VertexId[sockets + 1];
      EdgeId part_out_edges = 0;
      for (VertexId v_i = partition_offset[partition_id]; v_i < partition_offset[partition_id + 1]; v_i++)
      {
        part_out_edges += out_degree[v_i];
      }
      local_partition_offset[0] = partition_offset[partition_id];
      EdgeId remained_amount = part_out_edges + EdgeId(owned_vertices) * alpha;
      for (int s_i = 0; s_i < sockets; s_i++)
      {
        VertexId remained_partitions = sockets - s_i;
        EdgeId expected_chunk_size = remained_amount / remained_partitions;
        if (remained_partitions == 1)
        {
          local_partition_offset[s_i + 1] = partition_offset[partition_id + 1];
        }
        else
        {
          EdgeId got_edges = 0;
          for (VertexId v_i = local_partition_offset[s_i]; v_i < partition_offset[partition_id + 1]; v_i++)
          {
            got_edges += out_degree[v_i] + alpha;
            if (got_edges > expected_chunk_size)
            {
              local_partition_offset[s_i + 1] = v_i;
              break;
            }
          }
          local_partition_offset[s_i + 1] = (local_partition_offset[s_i + 1]) / PAGESIZE * PAGESIZE; // aligned with pages
        }
        EdgeId sub_part_out_edges = 0;
        for (VertexId v_i = local_partition_offset[s_i]; v_i < local_partition_offset[s_i + 1]; v_i++)
        {
          remained_amount -= out_degree[v_i] + alpha;
          sub_part_out_edges += out_degree[v_i];
        }
#ifdef PRINT_DEBUG_MESSAGES
        printf("|V'_%d_%d| = %u |E_%d| = %lu\n", partition_id, s_i, local_partition_offset[s_i + 1] - local_partition_offset[s_i], partition_id, sub_part_out_edges);
#endif
      }
    }

    VertexId *filtered_out_degree = alloc_vertex_array<VertexId>();
    for (VertexId v_i = partition_offset[partition_id]; v_i < partition_offset[partition_id + 1]; v_i++)
    {
      filtered_out_degree[v_i] = out_degree[v_i];
    }
    numa_free(out_degree, sizeof(VertexId) * vertices);
    out_degree = filtered_out_degree;
    in_degree = out_degree;

    int *buffered_edges = new int[partitions];
    std::vector<char> *send_buffer = new std::vector<char>[partitions];
    for (int i = 0; i < partitions; i++)
    {
      send_buffer[i].resize(edge_unit_size * CHUNKSIZE);
    }
    EdgeUnit<EdgeData> *recv_buffer = new EdgeUnit<EdgeData>[CHUNKSIZE];

    // constructing symmetric edges
    EdgeId recv_outgoing_edges = 0;
    outgoing_edges = new EdgeId[sockets];
    outgoing_adj_index = new EdgeId *[sockets];
    outgoing_adj_list = new AdjUnit<EdgeData> *[sockets];
    outgoing_adj_bitmap = new Bitmap *[sockets];
    for (int s_i = 0; s_i < sockets; s_i++)
    {
      outgoing_adj_bitmap[s_i] = new Bitmap(vertices);
      outgoing_adj_bitmap[s_i]->clear();
      outgoing_adj_index[s_i] = (EdgeId *)numa_alloc_onnode(sizeof(EdgeId) * (vertices + 1), s_i);
    }
    {
      std::thread recv_thread_dst([&]() {
        int finished_count = 0;
        MPI_Status recv_status;
        while (finished_count < partitions)
        {
          MPI_Probe(MPI_ANY_SOURCE, ShuffleGraph, MPI_COMM_WORLD, &recv_status);
          int i = recv_status.MPI_SOURCE;
          assert(recv_status.MPI_TAG == ShuffleGraph && i >= 0 && i < partitions);
          int recv_bytes;
          MPI_Get_count(&recv_status, MPI_CHAR, &recv_bytes);
          if (recv_bytes == 1)
          {
            finished_count += 1;
            char c;
            MPI_Recv(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            continue;
          }
          assert(recv_bytes % edge_unit_size == 0);
          int recv_edges = recv_bytes / edge_unit_size;
          MPI_Recv(recv_buffer, edge_unit_size * recv_edges, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          // #pragma omp parallel for
          for (EdgeId e_i = 0; e_i < recv_edges; e_i++)
          {
            VertexId src = recv_buffer[e_i].src;
            VertexId dst = recv_buffer[e_i].dst;
            assert(dst >= partition_offset[partition_id] && dst < partition_offset[partition_id + 1]);
            int dst_part = get_local_partition_id(dst);
            if (!outgoing_adj_bitmap[dst_part]->get_bit(src))
            {
              outgoing_adj_bitmap[dst_part]->set_bit(src);
              outgoing_adj_index[dst_part][src] = 0;
            }
            __sync_fetch_and_add(&outgoing_adj_index[dst_part][src], 1);
          }
          recv_outgoing_edges += recv_edges;
        }
      });
      for (int i = 0; i < partitions; i++)
      {
        buffered_edges[i] = 0;
      }
      assert(lseek(fin, read_offset, SEEK_SET) == read_offset);
      read_bytes = 0;
      while (read_bytes < bytes_to_read)
      {
        long curr_read_bytes;
        if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE)
        {
          curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
        }
        else
        {
          curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
        }
        assert(curr_read_bytes >= 0);
        read_bytes += curr_read_bytes;
        EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
        for (EdgeId e_i = 0; e_i < curr_read_edges; e_i++)
        {
          VertexId dst = read_edge_buffer[e_i].dst;
          int i = get_partition_id(dst);
          memcpy(send_buffer[i].data() + edge_unit_size * buffered_edges[i], &read_edge_buffer[e_i], edge_unit_size);
          buffered_edges[i] += 1;
          if (buffered_edges[i] == CHUNKSIZE)
          {
            MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
            buffered_edges[i] = 0;
          }
        }
        for (EdgeId e_i = 0; e_i < curr_read_edges; e_i++)
        {
          // std::swap(read_edge_buffer[e_i].src, read_edge_buffer[e_i].dst);
          VertexId tmp = read_edge_buffer[e_i].src;
          read_edge_buffer[e_i].src = read_edge_buffer[e_i].dst;
          read_edge_buffer[e_i].dst = tmp;
        }
        for (EdgeId e_i = 0; e_i < curr_read_edges; e_i++)
        {
          VertexId dst = read_edge_buffer[e_i].dst;
          int i = get_partition_id(dst);
          memcpy(send_buffer[i].data() + edge_unit_size * buffered_edges[i], &read_edge_buffer[e_i], edge_unit_size);
          buffered_edges[i] += 1;
          if (buffered_edges[i] == CHUNKSIZE)
          {
            MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
            buffered_edges[i] = 0;
          }
        }
      }
      for (int i = 0; i < partitions; i++)
      {
        if (buffered_edges[i] == 0)
          continue;
        MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
        buffered_edges[i] = 0;
      }
      for (int i = 0; i < partitions; i++)
      {
        char c = 0;
        MPI_Send(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
      }
      recv_thread_dst.join();
#ifdef PRINT_DEBUG_MESSAGES
      printf("machine(%d) got %lu symmetric edges\n", partition_id, recv_outgoing_edges);
#endif
    }
    compressed_outgoing_adj_vertices = new VertexId[sockets];
    compressed_outgoing_adj_index = new CompressedAdjIndexUnit *[sockets];
    for (int s_i = 0; s_i < sockets; s_i++)
    {
      outgoing_edges[s_i] = 0;
      compressed_outgoing_adj_vertices[s_i] = 0;
      for (VertexId v_i = 0; v_i < vertices; v_i++)
      {
        if (outgoing_adj_bitmap[s_i]->get_bit(v_i))
        {
          outgoing_edges[s_i] += outgoing_adj_index[s_i][v_i];
          compressed_outgoing_adj_vertices[s_i] += 1;
        }
      }
      compressed_outgoing_adj_index[s_i] = (CompressedAdjIndexUnit *)numa_alloc_onnode(sizeof(CompressedAdjIndexUnit) * (compressed_outgoing_adj_vertices[s_i] + 1), s_i);
      compressed_outgoing_adj_index[s_i][0].index = 0;
      EdgeId last_e_i = 0;
      compressed_outgoing_adj_vertices[s_i] = 0;
      for (VertexId v_i = 0; v_i < vertices; v_i++)
      {
        if (outgoing_adj_bitmap[s_i]->get_bit(v_i))
        {
          outgoing_adj_index[s_i][v_i] = last_e_i + outgoing_adj_index[s_i][v_i];
          last_e_i = outgoing_adj_index[s_i][v_i];
          compressed_outgoing_adj_index[s_i][compressed_outgoing_adj_vertices[s_i]].vertex = v_i;
          compressed_outgoing_adj_vertices[s_i] += 1;
          compressed_outgoing_adj_index[s_i][compressed_outgoing_adj_vertices[s_i]].index = last_e_i;
        }
      }
      for (VertexId p_v_i = 0; p_v_i < compressed_outgoing_adj_vertices[s_i]; p_v_i++)
      {
        VertexId v_i = compressed_outgoing_adj_index[s_i][p_v_i].vertex;
        outgoing_adj_index[s_i][v_i] = compressed_outgoing_adj_index[s_i][p_v_i].index;
        outgoing_adj_index[s_i][v_i + 1] = compressed_outgoing_adj_index[s_i][p_v_i + 1].index;
      }
#ifdef PRINT_DEBUG_MESSAGES
      printf("part(%d) E_%d has %lu symmetric edges\n", partition_id, s_i, outgoing_edges[s_i]);
#endif
      outgoing_adj_list[s_i] = (AdjUnit<EdgeData> *)numa_alloc_onnode(unit_size * outgoing_edges[s_i], s_i);
    }
    {
      std::thread recv_thread_dst([&]() {
        int finished_count = 0;
        MPI_Status recv_status;
        while (finished_count < partitions)
        {
          MPI_Probe(MPI_ANY_SOURCE, ShuffleGraph, MPI_COMM_WORLD, &recv_status);
          int i = recv_status.MPI_SOURCE;
          assert(recv_status.MPI_TAG == ShuffleGraph && i >= 0 && i < partitions);
          int recv_bytes;
          MPI_Get_count(&recv_status, MPI_CHAR, &recv_bytes);
          if (recv_bytes == 1)
          {
            finished_count += 1;
            char c;
            MPI_Recv(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            continue;
          }
          assert(recv_bytes % edge_unit_size == 0);
          int recv_edges = recv_bytes / edge_unit_size;
          MPI_Recv(recv_buffer, edge_unit_size * recv_edges, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#pragma omp parallel for
          for (EdgeId e_i = 0; e_i < recv_edges; e_i++)
          {
            VertexId src = recv_buffer[e_i].src;
            VertexId dst = recv_buffer[e_i].dst;
            assert(dst >= partition_offset[partition_id] && dst < partition_offset[partition_id + 1]);
            int dst_part = get_local_partition_id(dst);
            EdgeId pos = __sync_fetch_and_add(&outgoing_adj_index[dst_part][src], 1);
            outgoing_adj_list[dst_part][pos].neighbour = dst;
            if (!std::is_same<EdgeData, Empty>::value)
            {
              outgoing_adj_list[dst_part][pos].edge_data = recv_buffer[e_i].edge_data;
            }
          }
        }
      });
      for (int i = 0; i < partitions; i++)
      {
        buffered_edges[i] = 0;
      }
      assert(lseek(fin, read_offset, SEEK_SET) == read_offset);
      read_bytes = 0;
      while (read_bytes < bytes_to_read)
      {
        long curr_read_bytes;
        if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE)
        {
          curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
        }
        else
        {
          curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
        }
        assert(curr_read_bytes >= 0);
        read_bytes += curr_read_bytes;
        EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
        for (EdgeId e_i = 0; e_i < curr_read_edges; e_i++)
        {
          VertexId dst = read_edge_buffer[e_i].dst;
          int i = get_partition_id(dst);
          memcpy(send_buffer[i].data() + edge_unit_size * buffered_edges[i], &read_edge_buffer[e_i], edge_unit_size);
          buffered_edges[i] += 1;
          if (buffered_edges[i] == CHUNKSIZE)
          {
            MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
            buffered_edges[i] = 0;
          }
        }
        for (EdgeId e_i = 0; e_i < curr_read_edges; e_i++)
        {
          // std::swap(read_edge_buffer[e_i].src, read_edge_buffer[e_i].dst);
          VertexId tmp = read_edge_buffer[e_i].src;
          read_edge_buffer[e_i].src = read_edge_buffer[e_i].dst;
          read_edge_buffer[e_i].dst = tmp;
        }
        for (EdgeId e_i = 0; e_i < curr_read_edges; e_i++)
        {
          VertexId dst = read_edge_buffer[e_i].dst;
          int i = get_partition_id(dst);
          memcpy(send_buffer[i].data() + edge_unit_size * buffered_edges[i], &read_edge_buffer[e_i], edge_unit_size);
          buffered_edges[i] += 1;
          if (buffered_edges[i] == CHUNKSIZE)
          {
            MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
            buffered_edges[i] = 0;
          }
        }
      }
      for (int i = 0; i < partitions; i++)
      {
        if (buffered_edges[i] == 0)
          continue;
        MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
        buffered_edges[i] = 0;
      }
      for (int i = 0; i < partitions; i++)
      {
        char c = 0;
        MPI_Send(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
      }
      recv_thread_dst.join();
    }
    for (int s_i = 0; s_i < sockets; s_i++)
    {
      for (VertexId p_v_i = 0; p_v_i < compressed_outgoing_adj_vertices[s_i]; p_v_i++)
      {
        VertexId v_i = compressed_outgoing_adj_index[s_i][p_v_i].vertex;
        outgoing_adj_index[s_i][v_i] = compressed_outgoing_adj_index[s_i][p_v_i].index;
        outgoing_adj_index[s_i][v_i + 1] = compressed_outgoing_adj_index[s_i][p_v_i + 1].index;
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    incoming_edges = outgoing_edges;
    incoming_adj_index = outgoing_adj_index;
    incoming_adj_list = outgoing_adj_list;
    incoming_adj_bitmap = outgoing_adj_bitmap;
    compressed_incoming_adj_vertices = compressed_outgoing_adj_vertices;
    compressed_incoming_adj_index = compressed_outgoing_adj_index;
    MPI_Barrier(MPI_COMM_WORLD);

    delete[] buffered_edges;
    delete[] send_buffer;
    delete[] read_edge_buffer;
    delete[] recv_buffer;
    close(fin);

    tune_chunks();
    tuned_chunks_sparse = tuned_chunks_dense;

    prep_time += MPI_Wtime();

#ifdef PRINT_DEBUG_MESSAGES
    if (partition_id == 0)
    {
      printf("preprocessing cost: %.2lf (s)\n", prep_time);
    }
#endif
  }

  // transpose the graph
  void transpose()
  {
    std::swap(out_degree, in_degree);
    std::swap(outgoing_edges, incoming_edges);
    std::swap(outgoing_adj_index, incoming_adj_index);
    std::swap(outgoing_adj_bitmap, incoming_adj_bitmap);
    std::swap(outgoing_adj_list, incoming_adj_list);
    std::swap(tuned_chunks_dense, tuned_chunks_sparse);
    std::swap(compressed_outgoing_adj_vertices, compressed_incoming_adj_vertices);
    std::swap(compressed_outgoing_adj_index, compressed_incoming_adj_index);
  }

  // load a directed graph from path
  void load_directed(std::string path, VertexId vertices)
  {
    double prep_time = 0;
    prep_time -= MPI_Wtime();
    filename = path;
    symmetric = false;

    MPI_Datatype vid_t = get_mpi_data_type<VertexId>();

    this->vertices = vertices;
    long total_bytes = file_size(path.c_str());
    this->edges = total_bytes / edge_unit_size;
#ifdef PRINT_DEBUG_MESSAGES
    if (partition_id == 0)
    {
      printf("|V| = %u, |E| = %lu\n", vertices, edges);
    }
#endif

    EdgeId read_edges = edges / partitions;
    if (partition_id == partitions - 1)
    {
      read_edges += edges % partitions;
    } // the final partition has to gather all the vertices.|numberof edges|
    long bytes_to_read = edge_unit_size * read_edges;
    long read_offset = edge_unit_size * (edges / partitions * partition_id);
    long read_bytes;
    int fin = open(path.c_str(), O_RDONLY);
    EdgeUnit<EdgeData> *read_edge_buffer = new EdgeUnit<EdgeData>[CHUNKSIZE];

    out_degree = alloc_interleaved_vertex_array<VertexId>();
    for (VertexId v_i = 0; v_i < vertices; v_i++)
    {
      out_degree[v_i] = 0;
    }
    assert(lseek(fin, read_offset, SEEK_SET) == read_offset);
    read_bytes = 0;
    while (read_bytes < bytes_to_read)
    { // load the graph with a big chunk iteratively
      long curr_read_bytes;
      if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE)
      {
        curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
      }
      else
      {
        curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
      }
      assert(curr_read_bytes >= 0);
      read_bytes += curr_read_bytes;
      EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
#pragma omp parallel for
      for (EdgeId e_i = 0; e_i < curr_read_edges; e_i++)
      {
        VertexId src = read_edge_buffer[e_i].src;
        VertexId dst = read_edge_buffer[e_i].dst;
        __sync_fetch_and_add(&out_degree[src], 1);
      }
    }
    MPI_Allreduce(MPI_IN_PLACE, out_degree, vertices, vid_t, MPI_SUM, MPI_COMM_WORLD);
    //*************************************************************************************Gather all vertex count before this stage.

    // locality-aware chunking
    partition_offset = new VertexId[partitions + 1];
    partition_offset[0] = 0;
    EdgeId remained_amount = edges + EdgeId(vertices) * alpha;
    for (int i = 0; i < partitions; i++)
    {
      VertexId remained_partitions = partitions - i;
      EdgeId expected_chunk_size = remained_amount / remained_partitions;
      if (remained_partitions == 1)
      {
        partition_offset[i + 1] = vertices; //prefix_sum like
      }
      else
      {
        EdgeId got_edges = 0;
        for (VertexId v_i = partition_offset[i]; v_i < vertices; v_i++)
        {
          got_edges += out_degree[v_i] + alpha;
          if (got_edges > expected_chunk_size)
          {
            partition_offset[i + 1] = v_i;
            break;
          }
        }
        partition_offset[i + 1] = (partition_offset[i + 1]) / PAGESIZE * PAGESIZE; // aligned with pages
      }
      for (VertexId v_i = partition_offset[i]; v_i < partition_offset[i + 1]; v_i++)
      {
        remained_amount -= out_degree[v_i] + alpha;
      }
    }
    assert(partition_offset[partitions] == vertices);
    owned_vertices = partition_offset[partition_id + 1] - partition_offset[partition_id];
    //***************************************************************************reorganized data distribution
    // check consistency of partition boundaries
    VertexId *global_partition_offset = new VertexId[partitions + 1];
    MPI_Allreduce(partition_offset, global_partition_offset, partitions + 1, vid_t, MPI_MAX, MPI_COMM_WORLD);
    for (int i = 0; i <= partitions; i++)
    {
      assert(partition_offset[i] == global_partition_offset[i]);
    }
    MPI_Allreduce(partition_offset, global_partition_offset, partitions + 1, vid_t, MPI_MIN, MPI_COMM_WORLD);
    for (int i = 0; i <= partitions; i++)
    {
      assert(partition_offset[i] == global_partition_offset[i]);
    } //Double-check??
#ifdef PRINT_DEBUG_MESSAGES
    if (partition_id == 0)
    {
      for (int i = 0; i < partitions; i++)
      {
        EdgeId part_out_edges = 0;
        for (VertexId v_i = partition_offset[i]; v_i < partition_offset[i + 1]; v_i++)
        {
          part_out_edges += out_degree[v_i];
        }
        printf("|V'_%d| = %u |E^dense_%d| = %lu\n", i, partition_offset[i + 1] - partition_offset[i], i, part_out_edges);
      }
    }
#endif
    delete[] global_partition_offset;
    {
      // NUMA-aware sub-chunking
      local_partition_offset = new VertexId[sockets + 1];
      EdgeId part_out_edges = 0;
      for (VertexId v_i = partition_offset[partition_id]; v_i < partition_offset[partition_id + 1]; v_i++)
      {
        part_out_edges += out_degree[v_i];
      }
      local_partition_offset[0] = partition_offset[partition_id];
      EdgeId remained_amount = part_out_edges + EdgeId(owned_vertices) * alpha;
      for (int s_i = 0; s_i < sockets; s_i++)
      {
        VertexId remained_partitions = sockets - s_i;
        EdgeId expected_chunk_size = remained_amount / remained_partitions;
        if (remained_partitions == 1)
        {
          local_partition_offset[s_i + 1] = partition_offset[partition_id + 1];
        }
        else
        {
          EdgeId got_edges = 0;
          for (VertexId v_i = local_partition_offset[s_i]; v_i < partition_offset[partition_id + 1]; v_i++)
          {
            got_edges += out_degree[v_i] + alpha;
            if (got_edges > expected_chunk_size)
            {
              local_partition_offset[s_i + 1] = v_i;
              break;
            }
          }
          local_partition_offset[s_i + 1] = (local_partition_offset[s_i + 1]) / PAGESIZE * PAGESIZE; // aligned with pages
        }
        EdgeId sub_part_out_edges = 0;
        for (VertexId v_i = local_partition_offset[s_i]; v_i < local_partition_offset[s_i + 1]; v_i++)
        {
          remained_amount -= out_degree[v_i] + alpha;
          sub_part_out_edges += out_degree[v_i];
        }
#ifdef PRINT_DEBUG_MESSAGES
        printf("|V'_%d_%d| = %u |E^dense_%d_%d| = %lu\n", partition_id, s_i, local_partition_offset[s_i + 1] - local_partition_offset[s_i], partition_id, s_i, sub_part_out_edges);
#endif
      }
    }
    // **************************************************************************Partition the graph inside a node to explorit NUMA-aware.

    VertexId *filtered_out_degree = alloc_vertex_array<VertexId>();
    //    for (VertexId v_i=partition_offset[0];v_i<partition_offset[partitions-1];v_i++) {
    //      filtered_out_degree[v_i] = out_degree[v_i];
    //    }
    for (VertexId v_i = partition_offset[partition_id]; v_i < partition_offset[partition_id + 1]; v_i++)
    {
      filtered_out_degree[v_i] = out_degree[v_i];
    }
    out_degree_for_backward = out_degree;
    //numa_free(out_degree, sizeof(VertexId) * vertices);
    out_degree = filtered_out_degree;
    in_degree = alloc_vertex_array<VertexId>();
    in_degree_for_backward = alloc_interleaved_vertex_array<VertexId>();
    for (VertexId v_i = partition_offset[partition_id]; v_i < partition_offset[partition_id + 1]; v_i++)
    {
      in_degree[v_i] = 0;
    }
    for (VertexId v_i = 0; v_i < vertices; v_i++)
    {
      in_degree_for_backward[v_i] = 0;
    }
    int *buffered_edges = new int[partitions];
    std::vector<char> *send_buffer = new std::vector<char>[partitions];
    for (int i = 0; i < partitions; i++)
    {
      send_buffer[i].resize(edge_unit_size * CHUNKSIZE);
    }
    EdgeUnit<EdgeData> *recv_buffer = new EdgeUnit<EdgeData>[CHUNKSIZE];

    EdgeId recv_outgoing_edges = 0;
    outgoing_edges = new EdgeId[sockets];
    outgoing_adj_index = new EdgeId *[sockets];
    outgoing_adj_list = new AdjUnit<EdgeData> *[sockets];
    outgoing_adj_bitmap = new Bitmap *[sockets];
    for (int s_i = 0; s_i < sockets; s_i++)
    {
      outgoing_adj_bitmap[s_i] = new Bitmap(vertices);
      outgoing_adj_bitmap[s_i]->clear();
      outgoing_adj_index[s_i] = (EdgeId *)numa_alloc_onnode(sizeof(EdgeId) * (vertices + 1), s_i);
    }

    {
      std::thread recv_thread_dst([&]() {
        int finished_count = 0;
        MPI_Status recv_status;
        while (finished_count < partitions)
        {
          MPI_Probe(MPI_ANY_SOURCE, ShuffleGraph, MPI_COMM_WORLD, &recv_status);
          int i = recv_status.MPI_SOURCE;
          assert(recv_status.MPI_TAG == ShuffleGraph && i >= 0 && i < partitions);
          int recv_bytes;
          MPI_Get_count(&recv_status, MPI_CHAR, &recv_bytes);
          //printf("recv %d\n", recv_bytes);
          if (recv_bytes == 1)
          {
            finished_count += 1;
            char c;
            MPI_Recv(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            continue;
          }
          assert(recv_bytes % edge_unit_size == 0);
          int recv_edges = recv_bytes / edge_unit_size;
          MPI_Recv(recv_buffer, edge_unit_size * recv_edges, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          // #pragma omp parallel for
          for (EdgeId e_i = 0; e_i < recv_edges; e_i++)
          {
            VertexId src = recv_buffer[e_i].src;
            VertexId dst = recv_buffer[e_i].dst;
            assert(dst >= partition_offset[partition_id] && dst < partition_offset[partition_id + 1]);
            int dst_part = get_local_partition_id(dst);
            if (!outgoing_adj_bitmap[dst_part]->get_bit(src))
            {
              outgoing_adj_bitmap[dst_part]->set_bit(src);
              outgoing_adj_index[dst_part][src] = 0;
            }
            __sync_fetch_and_add(&outgoing_adj_index[dst_part][src], 1);
            __sync_fetch_and_add(&in_degree[dst], 1);
          }
          recv_outgoing_edges += recv_edges;
        }
      });
      for (int i = 0; i < partitions; i++)
      {
        buffered_edges[i] = 0;
      }
      assert(lseek(fin, read_offset, SEEK_SET) == read_offset);
      read_bytes = 0;
      while (read_bytes < bytes_to_read)
      {
        long curr_read_bytes;
        if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE)
        {
          curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
        }
        else
        {
          curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
        }
        assert(curr_read_bytes >= 0);
        read_bytes += curr_read_bytes;
        EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
        for (EdgeId e_i = 0; e_i < curr_read_edges; e_i++)
        {
          VertexId dst = read_edge_buffer[e_i].dst;
          int i = get_partition_id(dst);
          memcpy(send_buffer[i].data() + edge_unit_size * buffered_edges[i], &read_edge_buffer[e_i], edge_unit_size);
          buffered_edges[i] += 1;
          if (buffered_edges[i] == CHUNKSIZE)
          {
            //printf("send_start %d\n", edge_unit_size * buffered_edges[i]);
            MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
            //printf("send_end %d\n", recv_bytes);
            buffered_edges[i] = 0;
          }
        }
      }
      for (int i = 0; i < partitions; i++)
      {
        if (buffered_edges[i] == 0)
          continue;
        MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
        buffered_edges[i] = 0;
      }
      for (int i = 0; i < partitions; i++)
      {
        char c = 0;
        MPI_Send(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
      }
      recv_thread_dst.join();
      for (VertexId v_i = partition_offset[partition_id]; v_i < partition_offset[partition_id + 1]; v_i++)
      {
        in_degree_for_backward[v_i] = in_degree[v_i];
      }
      MPI_Allreduce(MPI_IN_PLACE, in_degree_for_backward, vertices, vid_t, MPI_SUM, MPI_COMM_WORLD);
#ifdef PRINT_DEBUG_MESSAGES
      printf("machine(%d) got %lu sparse mode edges\n", partition_id, recv_outgoing_edges);
#endif
    } //**************************************************************************I think this might be the dense model edge (partitioned by dst.)

    compressed_outgoing_adj_vertices = new VertexId[sockets];
    compressed_outgoing_adj_index = new CompressedAdjIndexUnit *[sockets];
    for (int s_i = 0; s_i < sockets; s_i++)
    {
      outgoing_edges[s_i] = 0;
      compressed_outgoing_adj_vertices[s_i] = 0;
      for (VertexId v_i = 0; v_i < vertices; v_i++)
      {
        if (outgoing_adj_bitmap[s_i]->get_bit(v_i))
        {
          outgoing_edges[s_i] += outgoing_adj_index[s_i][v_i];
          compressed_outgoing_adj_vertices[s_i] += 1;
        }
      }
      compressed_outgoing_adj_index[s_i] = (CompressedAdjIndexUnit *)numa_alloc_onnode(sizeof(CompressedAdjIndexUnit) * (compressed_outgoing_adj_vertices[s_i] + 1), s_i);
      compressed_outgoing_adj_index[s_i][0].index = 0;
      EdgeId last_e_i = 0;
      compressed_outgoing_adj_vertices[s_i] = 0;
      for (VertexId v_i = 0; v_i < vertices; v_i++)
      {
        if (outgoing_adj_bitmap[s_i]->get_bit(v_i))
        {
          outgoing_adj_index[s_i][v_i] = last_e_i + outgoing_adj_index[s_i][v_i];
          last_e_i = outgoing_adj_index[s_i][v_i];
          compressed_outgoing_adj_index[s_i][compressed_outgoing_adj_vertices[s_i]].vertex = v_i;
          compressed_outgoing_adj_vertices[s_i] += 1;
          compressed_outgoing_adj_index[s_i][compressed_outgoing_adj_vertices[s_i]].index = last_e_i;
        }
      }
      for (VertexId p_v_i = 0; p_v_i < compressed_outgoing_adj_vertices[s_i]; p_v_i++)
      {
        VertexId v_i = compressed_outgoing_adj_index[s_i][p_v_i].vertex;
        outgoing_adj_index[s_i][v_i] = compressed_outgoing_adj_index[s_i][p_v_i].index;
        outgoing_adj_index[s_i][v_i + 1] = compressed_outgoing_adj_index[s_i][p_v_i + 1].index;
      }
#ifdef PRINT_DEBUG_MESSAGES
      printf("part(%d) E_%d has %lu sparse mode edges\n", partition_id, s_i, outgoing_edges[s_i]);
#endif
      outgoing_adj_list[s_i] = (AdjUnit<EdgeData> *)numa_alloc_onnode(unit_size * outgoing_edges[s_i], s_i);
    }

    {
      std::thread recv_thread_dst([&]() {
        int finished_count = 0;
        MPI_Status recv_status;
        while (finished_count < partitions)
        {
          MPI_Probe(MPI_ANY_SOURCE, ShuffleGraph, MPI_COMM_WORLD, &recv_status);
          int i = recv_status.MPI_SOURCE;
          assert(recv_status.MPI_TAG == ShuffleGraph && i >= 0 && i < partitions);
          int recv_bytes;
          MPI_Get_count(&recv_status, MPI_CHAR, &recv_bytes);
          if (recv_bytes == 1)
          {
            finished_count += 1;
            char c;
            MPI_Recv(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            continue;
          }
          assert(recv_bytes % edge_unit_size == 0);
          int recv_edges = recv_bytes / edge_unit_size;
          MPI_Recv(recv_buffer, edge_unit_size * recv_edges, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#pragma omp parallel for
          for (EdgeId e_i = 0; e_i < recv_edges; e_i++)
          {
            VertexId src = recv_buffer[e_i].src;
            VertexId dst = recv_buffer[e_i].dst;
            assert(dst >= partition_offset[partition_id] && dst < partition_offset[partition_id + 1]);
            int dst_part = get_local_partition_id(dst);
            EdgeId pos = __sync_fetch_and_add(&outgoing_adj_index[dst_part][src], 1);
            outgoing_adj_list[dst_part][pos].neighbour = dst;
            if (!std::is_same<EdgeData, Empty>::value)
            {
              outgoing_adj_list[dst_part][pos].edge_data = recv_buffer[e_i].edge_data;
            }
          }
        }
      });
      for (int i = 0; i < partitions; i++)
      {
        buffered_edges[i] = 0;
      }
      assert(lseek(fin, read_offset, SEEK_SET) == read_offset);
      read_bytes = 0;
      while (read_bytes < bytes_to_read)
      {
        long curr_read_bytes;
        if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE)
        {
          curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
        }
        else
        {
          curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
        }
        assert(curr_read_bytes >= 0);
        read_bytes += curr_read_bytes;
        EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
        for (EdgeId e_i = 0; e_i < curr_read_edges; e_i++)
        {
          VertexId dst = read_edge_buffer[e_i].dst;
          int i = get_partition_id(dst);
          memcpy(send_buffer[i].data() + edge_unit_size * buffered_edges[i], &read_edge_buffer[e_i], edge_unit_size);
          buffered_edges[i] += 1;
          if (buffered_edges[i] == CHUNKSIZE)
          {
            MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
            buffered_edges[i] = 0;
          }
        }
      }
      for (int i = 0; i < partitions; i++)
      {
        if (buffered_edges[i] == 0)
          continue;
        MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
        buffered_edges[i] = 0;
      }
      for (int i = 0; i < partitions; i++)
      {
        char c = 0;
        MPI_Send(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
      }
      recv_thread_dst.join();
    } //maybe??*******************************************************************dense model finished
    for (int s_i = 0; s_i < sockets; s_i++)
    {
      for (VertexId p_v_i = 0; p_v_i < compressed_outgoing_adj_vertices[s_i]; p_v_i++)
      {
        VertexId v_i = compressed_outgoing_adj_index[s_i][p_v_i].vertex;
        outgoing_adj_index[s_i][v_i] = compressed_outgoing_adj_index[s_i][p_v_i].index;
        outgoing_adj_index[s_i][v_i + 1] = compressed_outgoing_adj_index[s_i][p_v_i + 1].index;
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    EdgeId recv_incoming_edges = 0;
    incoming_edges = new EdgeId[sockets];
    incoming_adj_index = new EdgeId *[sockets];
    incoming_adj_list = new AdjUnit<EdgeData> *[sockets];
    incoming_adj_bitmap = new Bitmap *[sockets];
    for (int s_i = 0; s_i < sockets; s_i++)
    {
      incoming_adj_bitmap[s_i] = new Bitmap(vertices);
      incoming_adj_bitmap[s_i]->clear();
      incoming_adj_index[s_i] = (EdgeId *)numa_alloc_onnode(sizeof(EdgeId) * (vertices + 1), s_i);
    }
    {
      std::thread recv_thread_src([&]() {
        int finished_count = 0;
        MPI_Status recv_status;
        while (finished_count < partitions)
        {
          MPI_Probe(MPI_ANY_SOURCE, ShuffleGraph, MPI_COMM_WORLD, &recv_status);
          int i = recv_status.MPI_SOURCE;
          assert(recv_status.MPI_TAG == ShuffleGraph && i >= 0 && i < partitions);
          int recv_bytes;
          MPI_Get_count(&recv_status, MPI_CHAR, &recv_bytes);
          if (recv_bytes == 1)
          {
            finished_count += 1;
            char c;
            MPI_Recv(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            continue;
          }
          assert(recv_bytes % edge_unit_size == 0);
          int recv_edges = recv_bytes / edge_unit_size;
          MPI_Recv(recv_buffer, edge_unit_size * recv_edges, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          // #pragma omp parallel for
          for (EdgeId e_i = 0; e_i < recv_edges; e_i++)
          {
            VertexId src = recv_buffer[e_i].src;
            VertexId dst = recv_buffer[e_i].dst;
            assert(src >= partition_offset[partition_id] && src < partition_offset[partition_id + 1]);
            int src_part = get_local_partition_id(src);
            if (!incoming_adj_bitmap[src_part]->get_bit(dst))
            {
              incoming_adj_bitmap[src_part]->set_bit(dst);
              incoming_adj_index[src_part][dst] = 0;
            }
            __sync_fetch_and_add(&incoming_adj_index[src_part][dst], 1);
          }
          recv_incoming_edges += recv_edges;
        }
      });
      for (int i = 0; i < partitions; i++)
      {
        buffered_edges[i] = 0;
      }
      assert(lseek(fin, read_offset, SEEK_SET) == read_offset);
      read_bytes = 0;
      while (read_bytes < bytes_to_read)
      {
        long curr_read_bytes;
        if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE)
        {
          curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
        }
        else
        {
          curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
        }
        assert(curr_read_bytes >= 0);
        read_bytes += curr_read_bytes;
        EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
        for (EdgeId e_i = 0; e_i < curr_read_edges; e_i++)
        {
          VertexId src = read_edge_buffer[e_i].src;
          int i = get_partition_id(src);
          memcpy(send_buffer[i].data() + edge_unit_size * buffered_edges[i], &read_edge_buffer[e_i], edge_unit_size);
          buffered_edges[i] += 1;
          if (buffered_edges[i] == CHUNKSIZE)
          {
            MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
            buffered_edges[i] = 0;
          }
        }
      }
      for (int i = 0; i < partitions; i++)
      {
        if (buffered_edges[i] == 0)
          continue;
        MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
        buffered_edges[i] = 0;
      }
      for (int i = 0; i < partitions; i++)
      {
        char c = 0;
        MPI_Send(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
      }
      recv_thread_src.join();
#ifdef PRINT_DEBUG_MESSAGES
      printf("machine(%d) got %lu dense mode edges\n", partition_id, recv_incoming_edges);
#endif
    }
    compressed_incoming_adj_vertices = new VertexId[sockets];
    compressed_incoming_adj_index = new CompressedAdjIndexUnit *[sockets];
    for (int s_i = 0; s_i < sockets; s_i++)
    {
      incoming_edges[s_i] = 0;
      compressed_incoming_adj_vertices[s_i] = 0;
      for (VertexId v_i = 0; v_i < vertices; v_i++)
      {
        if (incoming_adj_bitmap[s_i]->get_bit(v_i))
        {
          incoming_edges[s_i] += incoming_adj_index[s_i][v_i];
          compressed_incoming_adj_vertices[s_i] += 1;
        }
      }
      compressed_incoming_adj_index[s_i] = (CompressedAdjIndexUnit *)numa_alloc_onnode(sizeof(CompressedAdjIndexUnit) * (compressed_incoming_adj_vertices[s_i] + 1), s_i);
      compressed_incoming_adj_index[s_i][0].index = 0;
      EdgeId last_e_i = 0;
      compressed_incoming_adj_vertices[s_i] = 0;
      for (VertexId v_i = 0; v_i < vertices; v_i++)
      {
        if (incoming_adj_bitmap[s_i]->get_bit(v_i))
        {
          incoming_adj_index[s_i][v_i] = last_e_i + incoming_adj_index[s_i][v_i];
          last_e_i = incoming_adj_index[s_i][v_i];
          compressed_incoming_adj_index[s_i][compressed_incoming_adj_vertices[s_i]].vertex = v_i;
          compressed_incoming_adj_vertices[s_i] += 1;
          compressed_incoming_adj_index[s_i][compressed_incoming_adj_vertices[s_i]].index = last_e_i;
        }
      }
      for (VertexId p_v_i = 0; p_v_i < compressed_incoming_adj_vertices[s_i]; p_v_i++)
      {
        VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
        incoming_adj_index[s_i][v_i] = compressed_incoming_adj_index[s_i][p_v_i].index;
        incoming_adj_index[s_i][v_i + 1] = compressed_incoming_adj_index[s_i][p_v_i + 1].index;
      }
#ifdef PRINT_DEBUG_MESSAGES
      printf("part(%d) E_%d has %lu dense mode edges\n", partition_id, s_i, incoming_edges[s_i]);
#endif
      incoming_adj_list[s_i] = (AdjUnit<EdgeData> *)numa_alloc_onnode(unit_size * incoming_edges[s_i], s_i);
    }
    {
      std::thread recv_thread_src([&]() {
        int finished_count = 0;
        MPI_Status recv_status;
        while (finished_count < partitions)
        {
          MPI_Probe(MPI_ANY_SOURCE, ShuffleGraph, MPI_COMM_WORLD, &recv_status);
          int i = recv_status.MPI_SOURCE;
          assert(recv_status.MPI_TAG == ShuffleGraph && i >= 0 && i < partitions);
          int recv_bytes;
          MPI_Get_count(&recv_status, MPI_CHAR, &recv_bytes);
          if (recv_bytes == 1)
          {
            finished_count += 1;
            char c;
            MPI_Recv(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            continue;
          }
          assert(recv_bytes % edge_unit_size == 0);
          int recv_edges = recv_bytes / edge_unit_size;
          MPI_Recv(recv_buffer, edge_unit_size * recv_edges, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#pragma omp parallel for
          for (EdgeId e_i = 0; e_i < recv_edges; e_i++)
          {
            VertexId src = recv_buffer[e_i].src;
            VertexId dst = recv_buffer[e_i].dst;
            assert(src >= partition_offset[partition_id] && src < partition_offset[partition_id + 1]);
            int src_part = get_local_partition_id(src);
            EdgeId pos = __sync_fetch_and_add(&incoming_adj_index[src_part][dst], 1);
            incoming_adj_list[src_part][pos].neighbour = src;
            if (!std::is_same<EdgeData, Empty>::value)
            {
              incoming_adj_list[src_part][pos].edge_data = recv_buffer[e_i].edge_data;
            }
          }
        }
      });
      for (int i = 0; i < partitions; i++)
      {
        buffered_edges[i] = 0;
      }
      assert(lseek(fin, read_offset, SEEK_SET) == read_offset);
      read_bytes = 0;
      while (read_bytes < bytes_to_read)
      {
        long curr_read_bytes;
        if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE)
        {
          curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
        }
        else
        {
          curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
        }
        assert(curr_read_bytes >= 0);
        read_bytes += curr_read_bytes;
        EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
        for (EdgeId e_i = 0; e_i < curr_read_edges; e_i++)
        {
          VertexId src = read_edge_buffer[e_i].src;
          int i = get_partition_id(src);
          memcpy(send_buffer[i].data() + edge_unit_size * buffered_edges[i], &read_edge_buffer[e_i], edge_unit_size);
          buffered_edges[i] += 1;
          if (buffered_edges[i] == CHUNKSIZE)
          {
            MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
            buffered_edges[i] = 0;
          }
        }
      }
      for (int i = 0; i < partitions; i++)
      {
        if (buffered_edges[i] == 0)
          continue;
        MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
        buffered_edges[i] = 0;
      }
      for (int i = 0; i < partitions; i++)
      {
        char c = 0;
        MPI_Send(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
      }
      recv_thread_src.join();
    }
    for (int s_i = 0; s_i < sockets; s_i++)
    {
      for (VertexId p_v_i = 0; p_v_i < compressed_incoming_adj_vertices[s_i]; p_v_i++)
      {
        VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
        incoming_adj_index[s_i][v_i] = compressed_incoming_adj_index[s_i][p_v_i].index;
        incoming_adj_index[s_i][v_i + 1] = compressed_incoming_adj_index[s_i][p_v_i + 1].index;
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    delete[] buffered_edges;
    delete[] send_buffer;
    delete[] read_edge_buffer;
    delete[] recv_buffer;
    close(fin);

    transpose();
    tune_chunks();
    transpose();
    tune_chunks();

    prep_time += MPI_Wtime();

#ifdef PRINT_DEBUG_MESSAGES
    if (partition_id == 0)
    {
      printf("preprocessing cost: %.2lf (s)\n", prep_time);
    }
#endif
  }

  void tune_chunks()
  {
    tuned_chunks_dense = new ThreadState *[partitions];
    int current_send_part_id = partition_id;
    for (int step = 0; step < partitions; step++)
    {
      current_send_part_id = (current_send_part_id + 1) % partitions;
      int i = current_send_part_id;
      tuned_chunks_dense[i] = new ThreadState[threads];
      EdgeId remained_edges;
      int remained_partitions;
      VertexId last_p_v_i;
      VertexId end_p_v_i;
      for (int t_i = 0; t_i < threads; t_i++)
      {
        tuned_chunks_dense[i][t_i].status = WORKING;
        int s_i = get_socket_id(t_i);
        int s_j = get_socket_offset(t_i);
        if (s_j == 0)
        {
          VertexId p_v_i = 0;
          while (p_v_i < compressed_incoming_adj_vertices[s_i])
          {
            VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
            if (v_i >= partition_offset[i])
            {
              break;
            }
            p_v_i++;
          }
          last_p_v_i = p_v_i;
          while (p_v_i < compressed_incoming_adj_vertices[s_i])
          {
            VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
            if (v_i >= partition_offset[i + 1])
            {
              break;
            }
            p_v_i++;
          }
          end_p_v_i = p_v_i;
          remained_edges = 0;
          for (VertexId p_v_i = last_p_v_i; p_v_i < end_p_v_i; p_v_i++)
          {
            remained_edges += compressed_incoming_adj_index[s_i][p_v_i + 1].index - compressed_incoming_adj_index[s_i][p_v_i].index;
            remained_edges += alpha;
          }
        }
        tuned_chunks_dense[i][t_i].curr = last_p_v_i;
        tuned_chunks_dense[i][t_i].end = last_p_v_i;
        remained_partitions = threads_per_socket - s_j;
        EdgeId expected_chunk_size = remained_edges / remained_partitions;
        if (remained_partitions == 1)
        {
          tuned_chunks_dense[i][t_i].end = end_p_v_i;
        }
        else
        {
          EdgeId got_edges = 0;
          for (VertexId p_v_i = last_p_v_i; p_v_i < end_p_v_i; p_v_i++)
          {
            got_edges += compressed_incoming_adj_index[s_i][p_v_i + 1].index - compressed_incoming_adj_index[s_i][p_v_i].index + alpha;
            if (got_edges >= expected_chunk_size)
            {
              tuned_chunks_dense[i][t_i].end = p_v_i;
              last_p_v_i = tuned_chunks_dense[i][t_i].end;
              break;
            }
          }
          got_edges = 0;
          for (VertexId p_v_i = tuned_chunks_dense[i][t_i].curr; p_v_i < tuned_chunks_dense[i][t_i].end; p_v_i++)
          {
            got_edges += compressed_incoming_adj_index[s_i][p_v_i + 1].index - compressed_incoming_adj_index[s_i][p_v_i].index + alpha;
          }
          remained_edges -= got_edges;
        }
      }
    }
  }

  void tune_chunks_backward()
  {
    tuned_chunks_dense_backward = new ThreadState *[partitions];
    int current_send_part_id = partition_id;
    for (int step = 0; step < partitions; step++)
    {
      current_send_part_id = (current_send_part_id + 1) % partitions;
      int i = current_send_part_id;
      tuned_chunks_dense_backward[i] = new ThreadState[threads];
      EdgeId remained_edges;
      int remained_partitions;
      VertexId last_p_v_i;
      VertexId end_p_v_i;
      for (int t_i = 0; t_i < threads; t_i++)
      {
        tuned_chunks_dense_backward[i][t_i].status = WORKING;
        int s_i = get_socket_id(t_i);
        int s_j = get_socket_offset(t_i);
        if (s_j == 0)
        {
          VertexId p_v_i = 0;
          while (p_v_i < compressed_incoming_adj_vertices_backward[s_i])
          {
            VertexId v_i = compressed_incoming_adj_index_backward[s_i][p_v_i].vertex;
            if (v_i >= partition_offset[i])
            {
              break;
            }
            p_v_i++;
          }
          last_p_v_i = p_v_i;
          while (p_v_i < compressed_incoming_adj_vertices_backward[s_i])
          {
            VertexId v_i = compressed_incoming_adj_index_backward[s_i][p_v_i].vertex;
            if (v_i >= partition_offset[i + 1])
            {
              break;
            }
            p_v_i++;
          }
          end_p_v_i = p_v_i;
          remained_edges = 0;
          for (VertexId p_v_i = last_p_v_i; p_v_i < end_p_v_i; p_v_i++)
          {
            remained_edges += compressed_incoming_adj_index_backward[s_i][p_v_i + 1].index - compressed_incoming_adj_index_backward[s_i][p_v_i].index;
            remained_edges += alpha;
          }
        }
        tuned_chunks_dense_backward[i][t_i].curr = last_p_v_i;
        tuned_chunks_dense_backward[i][t_i].end = last_p_v_i;
        remained_partitions = threads_per_socket - s_j;
        EdgeId expected_chunk_size = remained_edges / remained_partitions;
        if (remained_partitions == 1)
        {
          tuned_chunks_dense_backward[i][t_i].end = end_p_v_i;
        }
        else
        {
          EdgeId got_edges = 0;
          for (VertexId p_v_i = last_p_v_i; p_v_i < end_p_v_i; p_v_i++)
          {
            got_edges += compressed_incoming_adj_index_backward[s_i][p_v_i + 1].index - compressed_incoming_adj_index_backward[s_i][p_v_i].index + alpha;
            if (got_edges >= expected_chunk_size)
            {
              tuned_chunks_dense_backward[i][t_i].end = p_v_i;
              last_p_v_i = tuned_chunks_dense_backward[i][t_i].end;
              break;
            }
          }
          got_edges = 0;
          for (VertexId p_v_i = tuned_chunks_dense_backward[i][t_i].curr; p_v_i < tuned_chunks_dense_backward[i][t_i].end; p_v_i++)
          {
            got_edges += compressed_incoming_adj_index_backward[s_i][p_v_i + 1].index - compressed_incoming_adj_index_backward[s_i][p_v_i].index + alpha;
          }
          remained_edges -= got_edges;
        }
      }
    }
  }

  // process vertices
  template <typename R>
  R process_vertices(std::function<R(VertexId)> process, Bitmap *active)
  {
    double stream_time = 0;
    stream_time -= MPI_Wtime();

    R reducer = 0;
    size_t basic_chunk = 64;
    for (int t_i = 0; t_i < threads; t_i++)
    {
      int s_i = get_socket_id(t_i);
      //   printf("p_v %d \n",t_i);
      int s_j = get_socket_offset(t_i);
      VertexId partition_size = local_partition_offset[s_i + 1] - local_partition_offset[s_i];
      thread_state[t_i]->curr = local_partition_offset[s_i] + partition_size / threads_per_socket / basic_chunk * basic_chunk * s_j;
      thread_state[t_i]->end = local_partition_offset[s_i] + partition_size / threads_per_socket / basic_chunk * basic_chunk * (s_j + 1);

      if (s_j == threads_per_socket - 1)
      {
        thread_state[t_i]->end = local_partition_offset[s_i + 1];
      }
      thread_state[t_i]->status = WORKING;
      //       if(s_i==0){
      //          printf("td %d inside %d %d\n",t_i,thread_state[t_i]->curr,thread_state[t_i]->end);
      //      }
    } //init all states.
#pragma omp parallel reduction(+ \
                               : reducer)
    {
      R local_reducer = 0;
      int thread_id = omp_get_thread_num();
      while (true)
      {
        VertexId v_i = __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
        if (v_i >= thread_state[thread_id]->end)
          break;
        unsigned long word = active->data[WORD_OFFSET(v_i)];
        while (word != 0)
        {
          if (word & 1)
          {
            local_reducer += process(v_i);
          }
          v_i++;
          word = word >> 1;
        }
      }
      thread_state[thread_id]->status = STEALING;
      for (int t_offset = 1; t_offset < threads; t_offset++)
      {
        int t_i = (thread_id + t_offset) % threads;
        while (thread_state[t_i]->status != STEALING)
        {
          VertexId v_i = __sync_fetch_and_add(&thread_state[t_i]->curr, basic_chunk);
          if (v_i >= thread_state[t_i]->end)
            continue;
          unsigned long word = active->data[WORD_OFFSET(v_i)];
          while (word != 0)
          {
            if (word & 1)
            {
              local_reducer += process(v_i);
            }
            v_i++;
            word = word >> 1;
          }
        }
      }
      // reducer += local_reducer;
    }
    R global_reducer;
    //    MPI_Datatype dt = get_mpi_data_type<R>();
    //    MPI_Allreduce(&reducer, &global_reducer, 1, dt, MPI_SUM, MPI_COMM_WORLD);
    stream_time += MPI_Wtime();
#ifdef PRINT_DEBUG_MESSAGES
    if (partition_id == 0)
    {
      printf("process_vertices took %lf (s)\n", stream_time);
    }
#endif
    return global_reducer;
  }

  template <typename M>
  inline size_t sizeofM(int f_size)
  {
    return sizeof(VertexId) + sizeof(M) * f_size;
  }

  template <typename M>
  void flush_local_send_buffer(int t_i)
  {
    int s_i = get_socket_id(t_i);
    int pos = __sync_fetch_and_add(&send_buffer[current_send_part_id][s_i]->count, local_send_buffer[t_i]->count);
    memcpy(send_buffer[current_send_part_id][s_i]->data + sizeof(MsgUnit<M>) * pos,
           local_send_buffer[t_i]->data,
           sizeof(MsgUnit<M>) * local_send_buffer[t_i]->count);
    local_send_buffer[t_i]->count = 0;
  }

  void flush_local_send_buffer_buffer(int t_i, int f_size)
  {
    int s_i = get_socket_id(t_i);
    int pos = __sync_fetch_and_add(&send_buffer[current_send_part_id][s_i]->count, local_send_buffer[t_i]->count);
    if (local_send_buffer[t_i]->count != 0)
      memcpy(send_buffer[current_send_part_id][s_i]->data + (sizeofM<float>(f_size)) * pos,
             local_send_buffer[t_i]->data,
             (sizeofM<float>(f_size)) * local_send_buffer[t_i]->count);
    local_send_buffer[t_i]->count = 0;
  }

  void flush_data_to_send_buffer_buffer_lock_free_init(VertexId message_count_partition)
  {
    int pos = __sync_fetch_and_add(&send_buffer[current_send_part_id][0]->count, message_count_partition);
    //if (pos == 0)
    //  printf("send buffer is NULL(%d)\n", pos);
  }

  void flush_data_to_send_buffer_buffer_lock_free_write(int t_i, int f_size, VertexId key, float *value, VertexId message_write_offset_key)
  {
    int s_i = get_socket_id(t_i);
    //int pos = __sync_fetch_and_add(&send_buffer[current_send_part_id][s_i]->count, local_send_buffer[t_i]->count);
    int pos = message_write_offset_key; // if (local_send_buffer[t_i]->count != 0)
    if (pos < 0 || pos >= (partition_offset[current_send_part_id + 1] - partition_offset[current_send_part_id]))
      printf("something wrong %d %d\n", key, current_send_part_id);
    // if (pos < partition_offset[current_send_part_id + 1] - partition_offset[current_send_part_id])
    //printf("POSITION %d %d %d %d\n", pos, (partition_offset[current_send_part_id + 1] - partition_offset[current_send_part_id]), current_send_part_id, partition_id);
    memcpy(send_buffer[current_send_part_id][s_i]->data + (sizeofM<float>(f_size)) * pos, &key, sizeof(VertexId));
    memcpy(send_buffer[current_send_part_id][s_i]->data + (sizeofM<float>(f_size)) * pos + sizeof(VertexId),
           value,
           f_size * sizeof(float));
  }

  // emit a message to a vertex's master (dense) / mirror (sparse)

  template <typename t_v>
  void emit_buffer(VertexId vtx, t_v *buffer, int f_size)
  {
    int t_i = omp_get_thread_num();
    char *s_buffer = NULL;
        s_buffer = (char *)local_send_buffer[t_i]->data;
    //printf("sizeofM<float>(f_size)%d %d %d %d\n",sizeofM<float>(f_size),local_send_buffer_limit,local_send_buffer[t_i]->count,s_buffer!=NULL);
    
    memcpy(s_buffer + local_send_buffer[t_i]->count * sizeofM<float>(f_size), &vtx, sizeof(VertexId));
    memcpy(s_buffer + local_send_buffer[t_i]->count * sizeofM<float>(f_size) + sizeof(VertexId), buffer, sizeof(float) * f_size);
    local_send_buffer[t_i]->count += 1;

    if (local_send_buffer[t_i]->count == local_send_buffer_limit)
    {
      ///local_send_buffer[t_i]->count = 0;
      //printf("mesage %d %d\n", sizeofM<float>(f_size), local_send_buffer_limit);
      flush_local_send_buffer_buffer(t_i, f_size);
      //flush_local_send_buffer<M>(t_i);
    }
  }

  template <typename M>
  void emit(VertexId vtx, M msg)
  {
    int t_i = omp_get_thread_num();
    MsgUnit<M> *buffer = (MsgUnit<M> *)local_send_buffer[t_i]->data;
    buffer[local_send_buffer[t_i]->count].vertex = vtx;
    buffer[local_send_buffer[t_i]->count].msg_data = msg;
    local_send_buffer[t_i]->count += 1;
    if (local_send_buffer[t_i]->count == local_send_buffer_limit)
    {
      flush_local_send_buffer<M>(t_i);
    }
  }

  template <typename R, typename M>
  R process_edges(std::function<void(VertexId, VertexAdjList<EdgeData>)> dense_signal,
                  std::function<R(VertexId, M)> dense_slot,
                  Bitmap *active,
                  Bitmap *dense_selective = nullptr)
  {
    double stream_time = 0;
    stream_time -= MPI_Wtime();
    int con = 0;

    for (int t_i = 0; t_i < threads; t_i++)
    {
      local_send_buffer[t_i]->resize(sizeof(MsgUnit<M>) * local_send_buffer_limit);
      local_send_buffer[t_i]->count = 0;
    }
    R reducer = 0;
    for (int i = 0; i < partitions; i++)
    {
      for (int s_i = 0; s_i < sockets; s_i++)
      {
        recv_buffer[i][s_i]->resize(sizeof(MsgUnit<M>) * owned_vertices * sockets);
        send_buffer[i][s_i]->resize(sizeof(MsgUnit<M>) * (partition_offset[i + 1] - partition_offset[i]) * sockets);
        send_buffer[i][s_i]->count = 0;
        recv_buffer[i][s_i]->count = 0;
      }
    } //init the send and receive buffer
    size_t basic_chunk = 64;
    {
      // dense selective bitmap
      if (dense_selective != nullptr && partitions > 1)
      {
        double sync_time = 0;
        sync_time -= get_time();
        std::thread send_thread([&]() {
          for (int step = 1; step < partitions; step++)
          {
            int recipient_id = (partition_id + step) % partitions;
            MPI_Send(dense_selective->data + WORD_OFFSET(partition_offset[partition_id]), owned_vertices / 64, MPI_UNSIGNED_LONG, recipient_id, PassMessage, MPI_COMM_WORLD);
          }
        });
        std::thread recv_thread([&]() {
          for (int step = 1; step < partitions; step++)
          {
            int sender_id = (partition_id - step + partitions) % partitions;
            MPI_Recv(dense_selective->data + WORD_OFFSET(partition_offset[sender_id]), (partition_offset[sender_id + 1] - partition_offset[sender_id]) / 64, MPI_UNSIGNED_LONG, sender_id, PassMessage, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          }
        });
        send_thread.join();
        recv_thread.join();
        MPI_Barrier(MPI_COMM_WORLD); // denseselect fill the bitmap array
        sync_time += get_time();
      }
#ifdef PRINT_DEBUG_MESSAGES
      if (partition_id == 0)
      {
        printf("dense mode\n");
      }
#endif
      int *send_queue = new int[partitions];
      int *recv_queue = new int[partitions];
      volatile int send_queue_size = 0;
      volatile int recv_queue_size = 0;
      std::mutex send_queue_mutex;
      std::mutex recv_queue_mutex;

      std::thread send_thread([&]() {
        for (int step = 0; step < partitions; step++)
        {
          if (step == partitions - 1)
          {
            break;
          }
          while (true)
          {
            send_queue_mutex.lock();
            bool condition = (send_queue_size <= step);
            send_queue_mutex.unlock();
            if (!condition)
              break;
            __asm volatile("pause" ::
                               : "memory");
          }
          int i = send_queue[step];
          for (int s_i = 0; s_i < sockets; s_i++)
          {
            MPI_Send(send_buffer[i][s_i]->data, sizeof(MsgUnit<M>) * send_buffer[i][s_i]->count, MPI_CHAR, i, PassMessage, MPI_COMM_WORLD);
          }
        }
      });
      std::thread recv_thread([&]() {
        std::vector<std::thread> threads;
        for (int step = 1; step < partitions; step++)
        {
          int i = (partition_id - step + partitions) % partitions;
          threads.emplace_back([&](int i) {
            for (int s_i = 0; s_i < sockets; s_i++)
            {
              MPI_Status recv_status;
              MPI_Probe(i, PassMessage, MPI_COMM_WORLD, &recv_status);
              MPI_Get_count(&recv_status, MPI_CHAR, &recv_buffer[i][s_i]->count);
              MPI_Recv(recv_buffer[i][s_i]->data, recv_buffer[i][s_i]->count, MPI_CHAR, i, PassMessage, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
              recv_buffer[i][s_i]->count /= sizeof(MsgUnit<M>);
            }
          },
                               i);
        }
        for (int step = 1; step < partitions; step++)
        {
          int i = (partition_id - step + partitions) % partitions;
          threads[step - 1].join();
          recv_queue[recv_queue_size] = i;
          recv_queue_mutex.lock();
          recv_queue_size += 1;
          recv_queue_mutex.unlock();
        }
        recv_queue[recv_queue_size] = partition_id;
        recv_queue_mutex.lock();
        recv_queue_size += 1;
        recv_queue_mutex.unlock();
      });
      current_send_part_id = partition_id;
      for (int step = 0; step < partitions; step++)
      {
        current_send_part_id = (current_send_part_id + 1) % partitions;
        int i = current_send_part_id;
        for (int t_i = 0; t_i < threads; t_i++)
        {
          *thread_state[t_i] = tuned_chunks_dense[i][t_i];
          //          printf("Range %d\t%d\n",tuned_chunks_dense[i][t_i].curr,tuned_chunks_dense[i][t_i].end);
        }

#pragma omp parallel
        {
          int thread_id = omp_get_thread_num();
          int s_i = get_socket_id(thread_id);
          VertexId final_p_v_i = thread_state[thread_id]->end;
          while (true)
          {
            VertexId begin_p_v_i = __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
            if (begin_p_v_i >= final_p_v_i)
              break;
            VertexId end_p_v_i = begin_p_v_i + basic_chunk;
            if (end_p_v_i > final_p_v_i)
            {
              end_p_v_i = final_p_v_i;
            }
            for (VertexId p_v_i = begin_p_v_i; p_v_i < end_p_v_i; p_v_i++)
            {
              VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
              dense_signal(v_i, VertexAdjList<EdgeData>(incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i].index, incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i + 1].index));
            }
          }
          thread_state[thread_id]->status = STEALING;
          for (int t_offset = 1; t_offset < threads; t_offset++)
          {
            int t_i = (thread_id + t_offset) % threads;
            int s_i = get_socket_id(t_i);
            while (thread_state[t_i]->status != STEALING)
            {
              VertexId begin_p_v_i = __sync_fetch_and_add(&thread_state[t_i]->curr, basic_chunk);
              if (begin_p_v_i >= thread_state[t_i]->end)
                break;
              VertexId end_p_v_i = begin_p_v_i + basic_chunk;
              if (end_p_v_i > thread_state[t_i]->end)
              {
                end_p_v_i = thread_state[t_i]->end;
              }
              for (VertexId p_v_i = begin_p_v_i; p_v_i < end_p_v_i; p_v_i++)
              {
                VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
                dense_signal(v_i, VertexAdjList<EdgeData>(incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i].index, incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i + 1].index));
              }
            }
          }
        }
#pragma omp parallel for
        for (int t_i = 0; t_i < threads; t_i++)
        {
          flush_local_send_buffer<M>(t_i);
        }
        if (i != partition_id)
        {
          send_queue[send_queue_size] = i;
          send_queue_mutex.lock();
          send_queue_size += 1;
          send_queue_mutex.unlock();
        }
      }
      for (int step = 0; step < partitions; step++)
      {
        while (true)
        {
          recv_queue_mutex.lock();
          bool condition = (recv_queue_size <= step);
          recv_queue_mutex.unlock();
          if (!condition)
            break;
          __asm volatile("pause" ::
                             : "memory");
        }
        int i = recv_queue[step];
        MessageBuffer **used_buffer;
        if (i == partition_id)
        {
          used_buffer = send_buffer[i];
        }
        else
        {
          used_buffer = recv_buffer[i];
        }
        for (int t_i = 0; t_i < threads; t_i++)
        {
          int s_i = get_socket_id(t_i);
          int s_j = get_socket_offset(t_i);
          VertexId partition_size = used_buffer[s_i]->count;
          thread_state[t_i]->curr = partition_size / threads_per_socket / basic_chunk * basic_chunk * s_j;
          thread_state[t_i]->end = partition_size / threads_per_socket / basic_chunk * basic_chunk * (s_j + 1);
          if (s_j == threads_per_socket - 1)
          {
            thread_state[t_i]->end = used_buffer[s_i]->count;
          }
          thread_state[t_i]->status = WORKING;
        }
#pragma omp parallel reduction(+ \
                               : reducer)
        {
          R local_reducer = 0;
          int thread_id = omp_get_thread_num();
          int s_i = get_socket_id(thread_id);
          MsgUnit<M> *buffer = (MsgUnit<M> *)used_buffer[s_i]->data;
          while (true)
          {
            VertexId b_i = __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
            if (b_i >= thread_state[thread_id]->end)
              break;
            VertexId begin_b_i = b_i;
            VertexId end_b_i = b_i + basic_chunk;
            if (end_b_i > thread_state[thread_id]->end)
            {
              end_b_i = thread_state[thread_id]->end;
            }
            for (b_i = begin_b_i; b_i < end_b_i; b_i++)
            {
              VertexId v_i = buffer[b_i].vertex;
              M msg_data = buffer[b_i].msg_data;
              local_reducer += dense_slot(v_i, msg_data);
            }
          }
          thread_state[thread_id]->status = STEALING;
          reducer += local_reducer;
        }
      }
      send_thread.join();
      recv_thread.join();
      delete[] send_queue;
      delete[] recv_queue;
      //      printf("con%d_%d\n",con,partition_offset[1]);
    }

    R global_reducer;
    MPI_Datatype dt = get_mpi_data_type<R>();
    MPI_Allreduce(&reducer, &global_reducer, 1, dt, MPI_SUM, MPI_COMM_WORLD);
    stream_time += MPI_Wtime();
#ifdef PRINT_DEBUG_MESSAGES
    if (partition_id == 0)
    {
      printf("process_edges took %lf (s)\n", stream_time);
    }
#endif
    return global_reducer;
  }

  template <typename R, typename M>
  R process_edges_simple(std::function<void(VertexId, VertexAdjList<EdgeData>)> dense_signal,
                         std::function<R(VertexId, M)> dense_slot,
                         Bitmap *active = nullptr,
                         Bitmap *dense_selective = nullptr)
  {
    double stream_time = 0;
    stream_time -= MPI_Wtime();

    double compute_time = 0;
    compute_time -= MPI_Wtime();
    for (int t_i = 0; t_i < threads; t_i++)
    {
      local_send_buffer[t_i]->resize(sizeof(MsgUnit<M>) * local_send_buffer_limit);
      local_send_buffer[t_i]->count = 0;
    }
    R reducer = 0;
    for (int i = 0; i < partitions; i++)
    {
      for (int s_i = 0; s_i < sockets; s_i++)
      {
        recv_buffer[i][s_i]->resize(sizeof(MsgUnit<M>) * owned_vertices * sockets);
        send_buffer[i][s_i]->resize(sizeof(MsgUnit<M>) * (partition_offset[i + 1] - partition_offset[i]) * sockets);
        send_buffer[i][s_i]->count = 0;
        recv_buffer[i][s_i]->count = 0;
      }
    } //init the send and receive buffer
    size_t basic_chunk = 64;
    {
      // dense selective bitmap
      if (dense_selective != nullptr && partitions > 1)
      {
        double sync_time = 0;
        sync_time -= get_time();
        std::thread send_thread([&]() {
          for (int step = 1; step < partitions; step++)
          {
            int recipient_id = (partition_id + step) % partitions;
            MPI_Send(dense_selective->data + WORD_OFFSET(partition_offset[partition_id]), owned_vertices / 64, MPI_UNSIGNED_LONG, recipient_id, PassMessage, MPI_COMM_WORLD);
          }
        });
        std::thread recv_thread([&]() {
          for (int step = 1; step < partitions; step++)
          {
            int sender_id = (partition_id - step + partitions) % partitions;
            MPI_Recv(dense_selective->data + WORD_OFFSET(partition_offset[sender_id]), (partition_offset[sender_id + 1] - partition_offset[sender_id]) / 64, MPI_UNSIGNED_LONG, sender_id, PassMessage, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          }
        });
        send_thread.join();
        recv_thread.join();
        MPI_Barrier(MPI_COMM_WORLD); // denseselect fill the bitmap array
        sync_time += get_time();
      }
#ifdef PRINT_DEBUG_MESSAGES
      if (partition_id == 0)
      {
        printf("dense mode\n");
      }
#endif
      int *send_queue = new int[partitions];
      int *recv_queue = new int[partitions];
      volatile int send_queue_size = 0;
      volatile int recv_queue_size = 0;
      std::mutex send_queue_mutex;
      std::mutex recv_queue_mutex;

      std::thread send_thread([&]() {
        for (int step = 0; step < partitions; step++)
        {
          if (step == partitions - 1)
          {
            break;
          }
          while (true)
          {
            send_queue_mutex.lock();
            bool condition = (send_queue_size <= step);
            send_queue_mutex.unlock();
            if (!condition)
              break;
            __asm volatile("pause" ::
                               : "memory");
          }
          int i = send_queue[step];
          for (int s_i = 0; s_i < sockets; s_i++)
          {
            MPI_Send(send_buffer[i][s_i]->data, sizeof(MsgUnit<M>) * send_buffer[i][s_i]->count, MPI_CHAR, i, PassMessage, MPI_COMM_WORLD);
          }
        }
      });
      std::thread recv_thread([&]() {
        std::vector<std::thread> threads;
        for (int step = 1; step < partitions; step++)
        {
          int i = (partition_id - step + partitions) % partitions;
          threads.emplace_back([&](int i) {
            for (int s_i = 0; s_i < sockets; s_i++)
            {
              MPI_Status recv_status;
              MPI_Probe(i, PassMessage, MPI_COMM_WORLD, &recv_status);
              MPI_Get_count(&recv_status, MPI_CHAR, &recv_buffer[i][s_i]->count);
              MPI_Recv(recv_buffer[i][s_i]->data, recv_buffer[i][s_i]->count, MPI_CHAR, i, PassMessage, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
              recv_buffer[i][s_i]->count /= sizeof(MsgUnit<M>);
            }
          },
                               i);
        }
        for (int step = 1; step < partitions; step++)
        {
          int i = (partition_id - step + partitions) % partitions;
          threads[step - 1].join();
          recv_queue[recv_queue_size] = i;
          recv_queue_mutex.lock();
          recv_queue_size += 1;
          recv_queue_mutex.unlock();
        }
        recv_queue[recv_queue_size] = partition_id;
        recv_queue_mutex.lock();
        recv_queue_size += 1;
        recv_queue_mutex.unlock();
      });
      current_send_part_id = partition_id;
      for (int step = 0; step < partitions; step++)
      {
        current_send_part_id = (current_send_part_id + 1) % partitions;
        int i = current_send_part_id;
        for (int t_i = 0; t_i < threads; t_i++)
        {
          *thread_state[t_i] = tuned_chunks_dense[i][t_i];
          //          printf("Range %d\t%d\n",tuned_chunks_dense[i][t_i].curr,tuned_chunks_dense[i][t_i].end);
        }

#pragma omp parallel
        {
          int thread_id = omp_get_thread_num();
          int s_i = get_socket_id(thread_id);
          VertexId final_p_v_i = thread_state[thread_id]->end;
          //          if(partition_id==1){
          //              printf("%d\n",thread_state[thread_id]->end);
          //          }
          while (true)
          {
            VertexId begin_p_v_i = __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
            if (begin_p_v_i >= final_p_v_i)
              break;
            VertexId end_p_v_i = begin_p_v_i + basic_chunk;
            if (end_p_v_i > final_p_v_i)
            {
              end_p_v_i = final_p_v_i;
            }
            for (VertexId p_v_i = begin_p_v_i; p_v_i < end_p_v_i; p_v_i++)
            {
              VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
              dense_signal(v_i, VertexAdjList<EdgeData>(incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i].index, incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i + 1].index));
            }
          }
          thread_state[thread_id]->status = STEALING;
          for (int t_offset = 1; t_offset < threads; t_offset++)
          {
            int t_i = (thread_id + t_offset) % threads;
            int s_i = get_socket_id(t_i);
            while (thread_state[t_i]->status != STEALING)
            {
              VertexId begin_p_v_i = __sync_fetch_and_add(&thread_state[t_i]->curr, basic_chunk);
              if (begin_p_v_i >= thread_state[t_i]->end)
                break;
              VertexId end_p_v_i = begin_p_v_i + basic_chunk;
              if (end_p_v_i > thread_state[t_i]->end)
              {
                end_p_v_i = thread_state[t_i]->end;
              }
              for (VertexId p_v_i = begin_p_v_i; p_v_i < end_p_v_i; p_v_i++)
              {
                VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
                dense_signal(v_i, VertexAdjList<EdgeData>(incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i].index, incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i + 1].index));
              }
            }
          }
        }
#pragma omp parallel for
        for (int t_i = 0; t_i < threads; t_i++)
        {
          flush_local_send_buffer<M>(t_i);
        }
        if (i != partition_id)
        {
          send_queue[send_queue_size] = i;
          send_queue_mutex.lock();
          send_queue_size += 1;
          send_queue_mutex.unlock();
        }
      }
      compute_time += MPI_Wtime();
      all_compute_time += compute_time;
      double overlap_time = 0;
      overlap_time -= MPI_Wtime();
      for (int step = 0; step < partitions; step++)
      {
        while (true)
        {
          recv_queue_mutex.lock();
          bool condition = (recv_queue_size <= step);
          recv_queue_mutex.unlock();
          if (!condition)
            break;
          __asm volatile("pause" ::
                             : "memory");
        }
        int i = recv_queue[step];
        MessageBuffer **used_buffer;
        if (i == partition_id)
        {
          used_buffer = send_buffer[i];
        }
        else
        {
          used_buffer = recv_buffer[i];
        }
        for (int t_i = 0; t_i < threads; t_i++)
        {
          int s_i = get_socket_id(t_i);
          int s_j = get_socket_offset(t_i);
          VertexId partition_size = used_buffer[s_i]->count;
          thread_state[t_i]->curr = partition_size / threads_per_socket / basic_chunk * basic_chunk * s_j;
          thread_state[t_i]->end = partition_size / threads_per_socket / basic_chunk * basic_chunk * (s_j + 1);
          if (s_j == threads_per_socket - 1)
          {
            thread_state[t_i]->end = used_buffer[s_i]->count;
          }
          thread_state[t_i]->status = WORKING;
        }
#pragma omp parallel reduction(+ \
                               : reducer)
        {
          R local_reducer = 0;
          int thread_id = omp_get_thread_num();
          int s_i = get_socket_id(thread_id);
          MsgUnit<M> *buffer = (MsgUnit<M> *)used_buffer[s_i]->data;
          while (true)
          {
            VertexId b_i = __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
            if (b_i >= thread_state[thread_id]->end)
              break;
            VertexId begin_b_i = b_i;
            VertexId end_b_i = b_i + basic_chunk;
            if (end_b_i > thread_state[thread_id]->end)
            {
              end_b_i = thread_state[thread_id]->end;
            }
            for (b_i = begin_b_i; b_i < end_b_i; b_i++)
            {
              VertexId v_i = buffer[b_i].vertex;
              M msg_data = buffer[b_i].msg_data;
              local_reducer += dense_slot(v_i, msg_data);
            }
          }
          thread_state[thread_id]->status = STEALING;
          reducer += local_reducer;
        }
      }
      overlap_time += MPI_Wtime();
      all_overlap_time += overlap_time;
      double wait_time = 0;
      wait_time -= MPI_Wtime();
      send_thread.join();
      recv_thread.join();
      wait_time += MPI_Wtime();
      all_wait_time += wait_time;
      delete[] send_queue;
      delete[] recv_queue;
      //      printf("con%d_%d\n",con,partition_offset[1]);
    }

    R global_reducer;
    MPI_Datatype dt = get_mpi_data_type<R>();
    MPI_Allreduce(&reducer, &global_reducer, 1, dt, MPI_SUM, MPI_COMM_WORLD);
    stream_time += MPI_Wtime();
#ifdef PRINT_DEBUG_MESSAGES
    if (partition_id == 0)
    {
      printf("process_edges took %lf (s)\n", stream_time);
    }
#endif
    return global_reducer;
  }

  template <typename R, typename M>
  R process_edges_with_GPU_aggregator(std::function<void(VertexId, VertexAdjList<EdgeData>)> dense_signal,
                                      float *local_data_buffer, int feature_size,
                                      Bitmap *active,
                                      Bitmap *dense_selective = nullptr)
  {
    double stream_time = 0;
    stream_time -= MPI_Wtime();

    double compute_time = 0;
    compute_time -= MPI_Wtime();
    for (int t_i = 0; t_i < threads; t_i++)
    {
      local_send_buffer[t_i]->resize(sizeof(MsgUnit<M>) * local_send_buffer_limit);
      local_send_buffer[t_i]->count = 0;
    }
    int max_recv_buffer_size = partition_offset[1] - partition_offset[0];
    for (int i = 0; i < partitions; i++)
    {
      for (int s_i = 0; s_i < sockets; s_i++)
      {
        recv_buffer[i][s_i]->resize(sizeof(MsgUnit<M>) * owned_vertices * sockets);
        send_buffer[i][s_i]->resize(sizeof(MsgUnit<M>) * (partition_offset[i + 1] - partition_offset[i]) * sockets);
        send_buffer[i][s_i]->count = 0;
        recv_buffer[i][s_i]->count = 0;
        max_recv_buffer_size = std::max(max_recv_buffer_size, (int)(partition_offset[i + 1] - partition_offset[i]));
      }
    } //init the send and receive buffer
    size_t basic_chunk = 64;
    {
#ifdef PRINT_DEBUG_MESSAGES
      if (partition_id == 0)
      {
        printf("dense mode\n");
      }
#endif
      int *send_queue = new int[partitions];
      int *recv_queue = new int[partitions];
      volatile int send_queue_size = 0;
      volatile int recv_queue_size = 0;
      std::mutex send_queue_mutex;
      std::mutex recv_queue_mutex;

      std::thread send_thread([&]() {
        for (int step = 0; step < partitions; step++)
        {
          if (step == partitions - 1)
          {
            break;
          }
          while (true)
          {
            send_queue_mutex.lock();
            bool condition = (send_queue_size <= step);
            send_queue_mutex.unlock();
            if (!condition)
              break;
            __asm volatile("pause" ::
                               : "memory");
          }
          int i = send_queue[step];
          for (int s_i = 0; s_i < sockets; s_i++)
          {
            MPI_Send(send_buffer[i][s_i]->data, sizeof(MsgUnit<M>) * send_buffer[i][s_i]->count, MPI_CHAR, i, PassMessage, MPI_COMM_WORLD);
          }
        }
      });
      std::thread recv_thread([&]() {
        std::vector<std::thread> threads;
        for (int step = 1; step < partitions; step++)
        {
          int i = (partition_id - step + partitions) % partitions;
          threads.emplace_back([&](int i) {
            for (int s_i = 0; s_i < sockets; s_i++)
            {
              MPI_Status recv_status;
              MPI_Probe(i, PassMessage, MPI_COMM_WORLD, &recv_status);
              MPI_Get_count(&recv_status, MPI_CHAR, &recv_buffer[i][s_i]->count);
              MPI_Recv(recv_buffer[i][s_i]->data, recv_buffer[i][s_i]->count, MPI_CHAR, i, PassMessage, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
              recv_buffer[i][s_i]->count /= sizeof(MsgUnit<M>);
            }
          },
                               i);
        }
        for (int step = 1; step < partitions; step++)
        {
          int i = (partition_id - step + partitions) % partitions;

          threads[step - 1].join();
          recv_queue[recv_queue_size] = i;
          recv_queue_mutex.lock();
          recv_queue_size += 1;
          recv_queue_mutex.unlock();
        }
        recv_queue[recv_queue_size] = partition_id;
        recv_queue_mutex.lock();
        recv_queue_size += 1;
        recv_queue_mutex.unlock();
      });
      current_send_part_id = partition_id;
      // 这先计算
      for (int step = 0; step < partitions; step++)
      {
        current_send_part_id = (current_send_part_id + 1) % partitions;
        int i = current_send_part_id;
        //        if(partition_id==0){
        //            printf("test current_send_id %d\n",i);
        //        }
        for (int t_i = 0; t_i < threads; t_i++)
        {
          *thread_state[t_i] = tuned_chunks_dense[i][t_i];
          //          printf("Range %d\t%d\n",tuned_chunks_dense[i][t_i].curr,tuned_chunks_dense[i][t_i].end);
        }
        //这后计算.
#pragma omp parallel
        {
          int thread_id = omp_get_thread_num();
          int s_i = get_socket_id(thread_id);
          VertexId final_p_v_i = thread_state[thread_id]->end;
          //          if(partition_id==1){
          //              printf("%d\n",thread_state[thread_id]->end);
          //          }
          while (true)
          {
            VertexId begin_p_v_i = __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
            if (begin_p_v_i >= final_p_v_i)
              break;
            VertexId end_p_v_i = begin_p_v_i + basic_chunk;
            if (end_p_v_i > final_p_v_i)
            {
              end_p_v_i = final_p_v_i;
            }
            for (VertexId p_v_i = begin_p_v_i; p_v_i < end_p_v_i; p_v_i++)
            {
              VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
              dense_signal(v_i, VertexAdjList<EdgeData>(incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i].index, incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i + 1].index));
            }
          }
          thread_state[thread_id]->status = STEALING;
          for (int t_offset = 1; t_offset < threads; t_offset++)
          {
            int t_i = (thread_id + t_offset) % threads;
            int s_i = get_socket_id(t_i);
            while (thread_state[t_i]->status != STEALING)
            {
              VertexId begin_p_v_i = __sync_fetch_and_add(&thread_state[t_i]->curr, basic_chunk);
              if (begin_p_v_i >= thread_state[t_i]->end)
                break;
              VertexId end_p_v_i = begin_p_v_i + basic_chunk;
              if (end_p_v_i > thread_state[t_i]->end)
              {
                end_p_v_i = thread_state[t_i]->end;
              }
              for (VertexId p_v_i = begin_p_v_i; p_v_i < end_p_v_i; p_v_i++)
              {
                VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
                dense_signal(v_i, VertexAdjList<EdgeData>(incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i].index, incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i + 1].index));
              }
            }
          }
        }
#pragma omp parallel for
        for (int t_i = 0; t_i < threads; t_i++)
        {
          flush_local_send_buffer<M>(t_i);
        }
        if (i != partition_id)
        {
          send_queue[send_queue_size] = i;
          send_queue_mutex.lock();
          send_queue_size += 1;
          send_queue_mutex.unlock();
        }
      }
      compute_time += MPI_Wtime();
      all_compute_time += compute_time;
      double overlap_time = 0;
      overlap_time -= MPI_Wtime();
      float *gpu_memory_buffer = NULL;
      //GPU based AGGREGATOR
      //ALLOCATE MEMOPRY
      allocate_gpu_buffer(&gpu_memory_buffer, max_recv_buffer_size * (feature_size + 1));
      zero_buffer(local_data_buffer, (partition_offset[partition_id + 1] - partition_offset[partition_id]) * (1 + feature_size));

      for (int step = 0; step < partitions; step++)
      {
        int wait_time = 0;
        wait_time -= MPI_Wtime();
        while (true)
        {
          recv_queue_mutex.lock();
          bool condition = (recv_queue_size <= step);
          recv_queue_mutex.unlock();
          if (!condition)
            break;
          __asm volatile("pause" ::
                             : "memory");
        }
        int i = recv_queue[step];
        MessageBuffer **used_buffer;
        if (i == partition_id)
        {
          used_buffer = send_buffer[i];
        }
        else
        {
          used_buffer = recv_buffer[i];
        }
        wait_time += MPI_Wtime();
        all_recv_wait_time += wait_time;

        //add GPU_aggregator
        //copy data to gpu  size: dst:  used_buffer[s_i]->count*(feature_size+1)
        for (int s_i = 0; s_i < sockets; s_i++)
        {
          //    printf("call new aggregator %d\n",used_buffer[s_i]->count);
          if (used_buffer[s_i]->count > 0)
          {
            int recv_copy = 0;
            recv_copy -= MPI_Wtime();
            move_data_in(gpu_memory_buffer, (float *)used_buffer[s_i]->data, 0, used_buffer[s_i]->count, (feature_size + 1));
            recv_copy += MPI_Wtime();
            all_recv_copy_time += recv_copy;
            //printf("call new engine %d\n",used_buffer[s_i]->count);
            int recv_kernel = 0;
            recv_kernel -= MPI_Wtime();
            aggregate_comm_result(local_data_buffer, gpu_memory_buffer, used_buffer[s_i]->count, feature_size, partition_offset[partition_id]);
            //printf("call new engine %d\n",used_buffer[s_i]->count);
            recv_kernel += MPI_Wtime();
            all_recv_kernel_time += recv_kernel;
          }
        }
        //aggregate

        //sync
      }
      FreeBuffer(gpu_memory_buffer);

      overlap_time += MPI_Wtime();
      all_overlap_time += overlap_time;
      double wait_time = 0;
      wait_time -= MPI_Wtime();
      send_thread.join();
      recv_thread.join();
      wait_time += MPI_Wtime();
      all_wait_time += wait_time;
      delete[] send_queue;
      delete[] recv_queue;
      //      printf("con%d_%d\n",con,partition_offset[1]);
    }

    R global_reducer;
    MPI_Datatype dt = get_mpi_data_type<R>();
    stream_time += MPI_Wtime();

    printf("process_edges took %lf (s)\n", stream_time);
#ifdef PRINT_DEBUG_MESSAGES
    if (partition_id == 0)
    {
      printf("process_edges took %lf (s)\n", stream_time);
    }
#endif
    return global_reducer;
  }

 
  
  template <typename R, typename M>
  R compute_sync_explict(torch::Tensor &input_gpu_or_cpu,
                                                         float *output_cpu,
                                                         std::vector<CSC_segment_pinned *> &graph_partitions,
                                                         std::function<void(VertexId, VertexAdjList<EdgeData>)> dense_signal,
                                                         float *local_data_buffer)//backward
  {

    int feature_size = gnnctx->layer_size[rtminfo->curr_layer];
    bool process_local = rtminfo->process_local;
    int layer_ = rtminfo->curr_layer;
    bool process_overlap = rtminfo->process_overlap;

    process_local = process_local && (graph_rep[layer_].rep_edge_size > 0);

    if (partition_id == 0)
    {
      printf("ComputeSync:layer(%d).process_local(%d).dimension(%d).reduce_comm(%d).overlap(%d)\n", layer_, process_local ? replication_threshold : -1, graph_rep[layer_].feature_size, process_local, process_overlap);
    }
    double stream_time = 0;
    stream_time -= MPI_Wtime();
    //    printf("initialize not been finished %d\n", partition_id);
    double compute_time = 0;
    compute_time -= MPI_Wtime();
    for (int t_i = 0; t_i < threads; t_i++)
    {
      local_send_buffer[t_i]->resize(sizeofM<M>(feature_size) * local_send_buffer_limit);
      local_send_buffer[t_i]->count = 0;
    }
    long max_recv_buffer_size = owned_vertices * sockets;
    for (int i = 0; i < partitions; i++)
    {
      for (int s_i = 0; s_i < sockets; s_i++)
      {
        recv_buffer[i][s_i]->resize_pinned(sizeofM<M>(feature_size) * owned_vertices * sockets);
        send_buffer[i][s_i]->resize_pinned(sizeofM<M>(feature_size) * (partition_offset[i + 1] - partition_offset[i]) * sockets);
        send_buffer[i][s_i]->count = 0;
        recv_buffer[i][s_i]->count = 0;
        //max_recv_buffer_size = std::max(max_recv_buffer_size, (int)(partition_offset[i + 1] - partition_offset[i]));
      }
    } //init the send and receive buffer
    //  printf("initialize send buffer finished %d\n", partition_id);
    float *gpu_memory_buffer = NULL;
    VertexId *index_gpu_input = NULL;
    //index_gpu_input=(VertexId*)cudaMallocGPU(vertices*sizeof(VertexId));
    VertexId *index_gpu_output = NULL;
    //index_gpu_output=(VertexId*)cudaMallocGPU(vertices*sizeof(VertexId));
    Cuda_Stream *cuda_stream = new Cuda_Stream();
    allocate_gpu_buffer(&gpu_memory_buffer, max_recv_buffer_size * (feature_size + 1));
    // if (gpu_memory_buffer == NULL)
    // {
    //   printf("something wrong\n");
    // }
    zero_buffer(local_data_buffer, (partition_offset[partition_id + 1] - partition_offset[partition_id]) * (feature_size));

    size_t max_dstrange = 0, max_edge_size = 0;
    for (int i = 0; i < graph_partitions.size(); i++)
    {
      max_edge_size = std::max(max_edge_size, (size_t)(graph_partitions[i]->edge_size));
      if (rtminfo->forward)
        max_dstrange = std::max(max_dstrange, (size_t)(graph_partitions[i]->dst_range[1] - graph_partitions[i]->dst_range[0]));
      else
        max_dstrange = std::max(max_dstrange, (size_t)(graph_partitions[i]->src_range[1] - graph_partitions[i]->src_range[0]));
    };

    weight_gpu_intergate = (float *)cudaMallocGPU(max_edge_size * sizeof(float));
    row_indices_intergate = (VertexId *)cudaMallocGPU(max_edge_size * sizeof(VertexId));
    column_offset_intergate = (VertexId *)cudaMallocGPU((max_dstrange + 1) * sizeof(VertexId));

    VertexId *src;
    VertexId *dst;
    float *weight;

    if (process_local)
    {
      index_gpu_output = (VertexId *)cudaMallocGPU(vertices * sizeof(VertexId));
      index_gpu_input = (VertexId *)cudaMallocGPU(vertices * sizeof(VertexId));
      graph_rep[layer_].rep_feature_gpu_buffer = (float *)cudaMallocGPU(((long)sizeof(float)) * graph_rep[layer_].rep_node_size * graph_rep[layer_].feature_size + 1);
      src = (VertexId *)cudaMallocGPU(sizeof(VertexId) * graph_rep[layer_].rep_edge_size + 1);
      dst = (VertexId *)cudaMallocGPU(sizeof(VertexId) * graph_rep[layer_].rep_edge_size + 1);
      weight = (float *)cudaMallocGPU(sizeof(float) * graph_rep[layer_].rep_edge_size + 1);
      //	printf("initialize send buffer finished %d\n", partition_id);
    }
    size_t basic_chunk = 64;
    {
#ifdef PRINT_DEBUG_MESSAGES
      if (partition_id == 0)
      {
        printf("dense mode\n");
      }
#endif
      int *send_queue = new int[partitions];
      int *recv_queue = new int[partitions];
      volatile int send_queue_size = 0;
      volatile int recv_queue_size = 0;
      std::mutex send_queue_mutex;
      std::mutex recv_queue_mutex;

      std::thread send_thread([&]() {
        for (int step = 0; step < partitions; step++)
        {
          if (step == partitions - 1)
          {
            break;
          }
          while (true)
          {
            send_queue_mutex.lock();
            bool condition = (send_queue_size <= step);
            send_queue_mutex.unlock();
            if (!condition)
              break;
            __asm volatile("pause" ::
                               : "memory");
          }
          int i = send_queue[step];
          for (int s_i = 0; s_i < sockets; s_i++)
          {
            MPI_Send(send_buffer[i][s_i]->data, sizeofM<M>(feature_size) * send_buffer[i][s_i]->count, MPI_CHAR, i, PassMessage, MPI_COMM_WORLD);
          }
        }
      });
      std::thread recv_thread([&]() {
        std::vector<std::thread> threads;
        for (int step = 1; step < partitions; step++)
        {
          int i = (partition_id - step + partitions) % partitions;
          threads.emplace_back([&](int i) {
            for (int s_i = 0; s_i < sockets; s_i++)
            {
              MPI_Status recv_status;
              MPI_Probe(i, PassMessage, MPI_COMM_WORLD, &recv_status);
              MPI_Get_count(&recv_status, MPI_CHAR, &recv_buffer[i][s_i]->count);
              MPI_Recv(recv_buffer[i][s_i]->data, recv_buffer[i][s_i]->count, MPI_CHAR, i, PassMessage, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
              recv_buffer[i][s_i]->count /= sizeofM<M>(feature_size);
            }
          },
                               i);
        }
        for (int step = 1; step < partitions; step++)
        {
          int i = (partition_id - step + partitions) % partitions;

          threads[step - 1].join();
          recv_queue[recv_queue_size] = i;
          recv_queue_mutex.lock();
          recv_queue_size += 1;
          recv_queue_mutex.unlock();
        }
        recv_queue[recv_queue_size] = partition_id;
        recv_queue_mutex.lock();
        recv_queue_size += 1;
        recv_queue_mutex.unlock();
      });

      if (process_overlap)
      {

        //pipeline
        current_send_part_id = partition_id;

        double cuda_sync_time = 0;
        cuda_sync_time -= MPI_Wtime();
        if(rtminfo->forward){
        forward_gpu_CSC_partition(
            input_gpu_or_cpu,
            output_cpu,
            graph_partitions,
            feature_size,
            (current_send_part_id + 1) % partitions,
            cuda_stream);
        }else{
        backward_gpu_CSR_partition(
            input_gpu_or_cpu,
            output_cpu,
            graph_partitions,
            feature_size,
            (current_send_part_id + 1) % partitions,
            cuda_stream);
        }
        cuda_stream->CUDA_DEVICE_SYNCHRONIZE();
        cuda_sync_time += MPI_Wtime();
        this->all_cuda_sync_time += cuda_sync_time;
      }
      else
      {

        //no pipeline
        for (int step = 0; step < partitions; step++)
        {
          //pipeline
          //current_send_part_id = partition_id;
          //no pipeline
          current_send_part_id = (current_send_part_id + 1) % partitions;
          double cuda_sync_time = 0;
          cuda_sync_time -= MPI_Wtime();
          //    printf("%d %d\n",input_gpu_or_cpu.size(0),input_gpu_or_cpu.size(1));

          if(rtminfo->forward){
        forward_gpu_CSC_partition(
            input_gpu_or_cpu,
            output_cpu,
            graph_partitions,
            feature_size,
            (current_send_part_id + 1) % partitions,
            cuda_stream);
        }else{
        backward_gpu_CSR_partition(
            input_gpu_or_cpu,
            output_cpu,
            graph_partitions,
            feature_size,
            (current_send_part_id + 1) % partitions,
            cuda_stream);
        }
          cuda_stream->CUDA_DEVICE_SYNCHRONIZE();

          //no pipeline
          cuda_sync_time += MPI_Wtime();
          this->all_cuda_sync_time += cuda_sync_time;
          //no pipeline
        }
        //printf("emove data\n");
      }
      //printf("initialize send buffer finished %d\n", partition_id);

      for (int step = 0; step < partitions; step++)
      {
        current_send_part_id = (current_send_part_id + 1) % partitions;
        int i = current_send_part_id;
        for (int t_i = 0; t_i < threads; t_i++)
        {
          *thread_state[t_i] = tuned_chunks_dense[i][t_i];
          //    printf("Range %d\t%d %d\n",tuned_chunks_dense[i][t_i].curr,tuned_chunks_dense[i][t_i].end,partition_id);
        }
        //这后计算.

        cuda_stream->CUDA_DEVICE_SYNCHRONIZE();
        //free_all_tmp();
        if (current_send_part_id != partition_id)
        {
          double cuda_sync_time = 0;
          cuda_sync_time -= MPI_Wtime();
          if (process_overlap)
          {
            if(rtminfo->forward){
                forward_gpu_CSC_partition(
                    input_gpu_or_cpu,
                    output_cpu,
                    graph_partitions,
                    feature_size,
                    (current_send_part_id + 1) % partitions,
                    cuda_stream);
            }else{
                backward_gpu_CSR_partition(
                    input_gpu_or_cpu,
                    output_cpu,
                    graph_partitions,
                    feature_size,
                    (current_send_part_id + 1) % partitions,
                    cuda_stream);
        }
          }
          //cuda_stream->CUDA_DEVICE_SYNCHRONIZE();
          cuda_sync_time += MPI_Wtime();
          this->all_cuda_sync_time += cuda_sync_time;
        }
        else
        {
          if (process_overlap)
          {
            double replication_time = 0;
            replication_time -= MPI_Wtime();
            if (process_local)
            {
              // if (partition_id == 0)
              //  printf("process_replicate %d on layer %d\n", replication_threshold, layer_);
              //MPI_Barrier(MPI_COMM_WORLD);
              forward_gpu_local(graph_rep, local_data_buffer, cuda_stream, src, dst, weight, index_gpu_input, index_gpu_output, feature_size, layer_);
              //cuda_stream->CUDA_DEVICE_SYNCHRONIZE();
            }
            replication_time += MPI_Wtime();
            //MPI_Barrier(MPI_COMM_WORLD);
            all_replication_time += replication_time;
          }
        }

#pragma omp parallel
        {
          //printf("DEBUGstart%d\n",partition_id);
          int thread_id = omp_get_thread_num();
          int s_i = get_socket_id(thread_id);
          VertexId final_p_v_i = thread_state[thread_id]->end;

          //printf("DEBUG %d %d %d %d\n",thread_state[thread_id]->end,s_i,thread_id,partition_id);

          while (true)
          {
            VertexId begin_p_v_i = __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
            if (begin_p_v_i >= final_p_v_i)
              break;
            VertexId end_p_v_i = begin_p_v_i + basic_chunk;
            if (end_p_v_i > final_p_v_i)
            {
              end_p_v_i = final_p_v_i;
            }
            for (VertexId p_v_i = begin_p_v_i; p_v_i < end_p_v_i; p_v_i++)
            {
              VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
              dense_signal(v_i, VertexAdjList<EdgeData>(incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i].index, incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i + 1].index));
            }
          }
          //  printf("Send initial has been finished %d\n", partition_id);
          thread_state[thread_id]->status = STEALING;
          for (int t_offset = 1; t_offset < threads; t_offset++)
          {
            int t_i = (thread_id + t_offset) % threads;
            int s_i = get_socket_id(t_i);
            while (thread_state[t_i]->status != STEALING)
            {
              VertexId begin_p_v_i = __sync_fetch_and_add(&thread_state[t_i]->curr, basic_chunk);
              if (begin_p_v_i >= thread_state[t_i]->end)
                break;
              VertexId end_p_v_i = begin_p_v_i + basic_chunk;
              if (end_p_v_i > thread_state[t_i]->end)
              {
                end_p_v_i = thread_state[t_i]->end;
              }
              for (VertexId p_v_i = begin_p_v_i; p_v_i < end_p_v_i; p_v_i++)
              {
                VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
                dense_signal(v_i, VertexAdjList<EdgeData>(incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i].index, incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i + 1].index));
              }
            }
          }
        }
#pragma omp parallel for
        for (int t_i = 0; t_i < threads; t_i++)
        {
          flush_local_send_buffer_buffer(t_i, feature_size);
        }
        if (i != partition_id)
        {
          //           for (int s_i=0;s_i<sockets;s_i++) {
          //            send_buffer[i][s_i]->count=(partition_offset[i+1] - partition_offset[i]);
          //          }
          send_queue[send_queue_size] = i;
          send_queue_mutex.lock();
          send_queue_size += 1;
          send_queue_mutex.unlock();
        }
      }

      compute_time += MPI_Wtime();
      all_compute_time += compute_time;

      double overlap_time = 0;
      overlap_time -= MPI_Wtime();
      // //process_local
      for (int step = 0; step < partitions; step++)
      {
        int wait_time = 0;
        wait_time -= MPI_Wtime();
        while (true)
        {
          recv_queue_mutex.lock();
          bool condition = (recv_queue_size <= step);
          recv_queue_mutex.unlock();
          if (!condition)
            break;
          __asm volatile("pause" ::
                             : "memory");
        }
        int i = recv_queue[step];
        MessageBuffer **used_buffer;
        if (i == partition_id)
        {
          used_buffer = send_buffer[i];
        }
        else
        {
          used_buffer = recv_buffer[i];
        }
        wait_time += MPI_Wtime();
        all_recv_wait_time += wait_time;

        //add GPU_aggregator
        //copy data to gpu  size: dst:  used_buffer[s_i]->count*(feature_size+1)
        for (int s_i = 0; s_i < sockets; s_i++)
        {
          cuda_stream->CUDA_DEVICE_SYNCHRONIZE();
          if (used_buffer[s_i]->count > 0)
          {
            //printf("%ld %ld %ld\n", used_buffer[s_i]->count, max_recv_buffer_size, max_recv_buffer_size * (feature_size + 1));
            int recv_copy = 0;
            recv_copy -= MPI_Wtime();
            cuda_stream->move_data_in(gpu_memory_buffer, (float *)used_buffer[s_i]->data, 0, used_buffer[s_i]->count, (feature_size + 1), false);
            //cuda_stream->CUDA_DEVICE_SYNCHRONIZE();

            for (int i = 0; i < used_buffer[s_i]->count; i++)
            {
              VertexId *S = (VertexId *)used_buffer[s_i]->data;
              if (S[i * (feature_size + 1)] < partition_offset[partition_id] || S[i * (feature_size + 1)] >= partition_offset[partition_id + 1])
                printf("something wroing %d %d\n", S[i * feature_size + 1], partition_id);
            }

            recv_copy += MPI_Wtime();
            all_recv_copy_time += recv_copy;
            int recv_kernel = 0;
            recv_kernel -= MPI_Wtime();
            cuda_stream->aggregate_comm_result_debug(local_data_buffer, gpu_memory_buffer, used_buffer[s_i]->count, feature_size, partition_offset[partition_id], partition_offset[partition_id + 1], false);
            //cuda_stream->CUDA_DEVICE_SYNCHRONIZE();
            recv_kernel += MPI_Wtime();
            all_recv_kernel_time += recv_kernel;

            //printf("remove comm\n");
          }
        }
        //aggregate

        //sync
      }
      //printf("Send has been finished %d\n", partition_id);

      overlap_time += MPI_Wtime();
      all_overlap_time += overlap_time;

      // send_thread.join();
      // recv_thread.join();
      cuda_stream->CUDA_DEVICE_SYNCHRONIZE();

      if (!process_overlap)
      {
        double replication_time = 0;
        replication_time -= MPI_Wtime();
        if (process_local)
        {
          //if (partition_id == 0)
          //printf("process_replicate %d on layer %d\n", replication_threshold, layer_);
          //MPI_Barrier(MPI_COMM_WORLD);
          forward_gpu_local(graph_rep, local_data_buffer, cuda_stream, src, dst, weight, index_gpu_input, index_gpu_output, feature_size, layer_);
          cuda_stream->CUDA_DEVICE_SYNCHRONIZE();
        }
        replication_time += MPI_Wtime();
        //MPI_Barrier(MPI_COMM_WORLD);
        all_replication_time += replication_time;
      }

      //process_local
      //  double wait_time=0;
      //  wait_time-=MPI_Wtime();
      FreeBuffer(gpu_memory_buffer);
      cuda_stream->destory_Stream();
      // wait_time+=MPI_Wtime();
      // all_wait_time+=wait_time;
      send_thread.join();
      recv_thread.join();

      delete[] send_queue;
      delete[] recv_queue;
      if (process_local)
      {
        FreeEdge(src);
        FreeEdge(dst);
        FreeBuffer(weight);
        FreeEdge(index_gpu_input);
        FreeEdge(index_gpu_output);
        FreeBuffer(graph_rep[layer_].rep_feature_gpu_buffer);
      }
      free_all_tmp();
      //      printf("con%d_%d\n",con,partition_offset[1]);
    }

    R global_reducer;
    stream_time += MPI_Wtime();
#ifdef PRINT_DEBUG_MESSAGES
    if (partition_id == 0)
    {
      printf("process_edges took %lf (s)\n", stream_time);
    }
#endif
    return global_reducer;
  }
  

  template <typename R, typename M>
  R compute_sync_lite(torch::Tensor &input_gpu_or_cpu,
                                                      float *output_cpu,
                                                      std::vector<CSC_segment_pinned *> &graph_partitions,
                                                      std::function<void(VertexId, VertexAdjList<EdgeData>)> dense_signal,
                                                      float *local_data_buffer)
  {

    int feature_size = gnnctx->layer_size[rtminfo->curr_layer];
    bool process_local = rtminfo->process_local;
    int layer_ = rtminfo->curr_layer;
    bool process_overlap = rtminfo->process_overlap;

    process_local = process_local && (graph_rep[layer_].rep_edge_size > 0);

    if (partition_id == 0)
    {
      printf("ComputeSyncLite:layer(%d).process_local(%d).dimension(%d).reduce_comm(%d).overlap(%d)\n", layer_, process_local ? replication_threshold : -1, graph_rep[layer_].feature_size, process_local, process_overlap);
    }
    double stream_time = 0;
    stream_time -= MPI_Wtime();
    //    printf("initialize not been finished %d\n", partition_id);
    double compute_time = 0;
    compute_time -= MPI_Wtime();
    for (int t_i = 0; t_i < threads; t_i++)
    {
      local_send_buffer[t_i]->resize(sizeofM<M>(feature_size) * local_send_buffer_limit);
      local_send_buffer[t_i]->count = 0;
    }
    long max_recv_buffer_size = owned_vertices * sockets;
    for (int i = 0; i < partitions; i++)
    {
      for (int s_i = 0; s_i < sockets; s_i++)
      {
        recv_buffer[i][s_i]->resize_pinned(sizeofM<M>(feature_size) * owned_vertices * sockets);
        send_buffer[i][s_i]->resize_pinned(sizeofM<M>(feature_size) * (partition_offset[i + 1] - partition_offset[i]) * sockets);
        send_buffer[i][s_i]->count = 0;
        recv_buffer[i][s_i]->count = 0;
        //max_recv_buffer_size = std::max(max_recv_buffer_size, (int)(partition_offset[i + 1] - partition_offset[i]));
      }
    } //init the send and receive buffer
    //  printf("initialize send buffer finished %d\n", partition_id);
    float *gpu_memory_buffer = NULL;
    VertexId *index_gpu_input = NULL;
    //index_gpu_input=(VertexId*)cudaMallocGPU(vertices*sizeof(VertexId));
    VertexId *index_gpu_output = NULL;
    //index_gpu_output=(VertexId*)cudaMallocGPU(vertices*sizeof(VertexId));
    Cuda_Stream *cuda_stream = new Cuda_Stream();
    allocate_gpu_buffer(&gpu_memory_buffer, max_recv_buffer_size * (feature_size + 1));
    // if (gpu_memory_buffer == NULL)
    // {
    //   printf("something wrong\n");
    // }
    zero_buffer(local_data_buffer, (partition_offset[partition_id + 1] - partition_offset[partition_id]) * (feature_size));

    size_t max_dstrange = 0, max_edge_size = 0;
    for (int i = 0; i < graph_partitions.size(); i++)
    {
      max_edge_size = std::max(max_edge_size, (size_t)(graph_partitions[i]->edge_size));
      if(rtminfo->forward)
          max_dstrange = std::max(max_dstrange, (size_t)(graph_partitions[i]->dst_range[1] - graph_partitions[i]->dst_range[0]));
      else
          max_dstrange = std::max(max_dstrange, (size_t)(graph_partitions[i]->src_range[1] - graph_partitions[i]->src_range[0]));
          
    };

    weight_gpu_intergate = (float *)cudaMallocGPU(max_edge_size * sizeof(float));
    row_indices_intergate = (VertexId *)cudaMallocGPU(max_edge_size * sizeof(VertexId));
    column_offset_intergate = (VertexId *)cudaMallocGPU((max_dstrange + 1) * sizeof(VertexId));

    VertexId *src;
    VertexId *dst;
    float *weight;

    if (process_local)
    {
      index_gpu_output = (VertexId *)cudaMallocGPU(vertices * sizeof(VertexId));
      index_gpu_input = (VertexId *)cudaMallocGPU(vertices * sizeof(VertexId));
      graph_rep[layer_].rep_feature_gpu_buffer = (float *)cudaMallocGPU(((long)sizeof(float)) * graph_rep[layer_].rep_node_size * graph_rep[layer_].feature_size + 1);
      src = (VertexId *)cudaMallocGPU(sizeof(VertexId) * graph_rep[layer_].rep_edge_size + 1);
      dst = (VertexId *)cudaMallocGPU(sizeof(VertexId) * graph_rep[layer_].rep_edge_size + 1);
      weight = (float *)cudaMallocGPU(sizeof(float) * graph_rep[layer_].rep_edge_size + 1);
      //	printf("initialize send buffer finished %d\n", partition_id);
    }
    size_t basic_chunk = 64;
    {
#ifdef PRINT_DEBUG_MESSAGES
      if (partition_id == 0)
      {
        printf("dense mode\n");
      }
#endif
      int *send_queue = new int[partitions];
      int *recv_queue = new int[partitions];
      volatile int send_queue_size = 0;
      volatile int recv_queue_size = 0;
      std::mutex send_queue_mutex;
      std::mutex recv_queue_mutex;

      std::thread send_thread([&]() {
        for (int step = 0; step < partitions; step++)
        {
          if (step == partitions - 1)
          {
            break;
          }
          while (true)
          {
            send_queue_mutex.lock();
            bool condition = (send_queue_size <= step);
            send_queue_mutex.unlock();
            if (!condition)
              break;
            __asm volatile("pause" ::
                               : "memory");
          }
          int i = send_queue[step];
          for (int s_i = 0; s_i < sockets; s_i++)
          {
            MPI_Send(send_buffer[i][s_i]->data, sizeofM<M>(feature_size) * send_buffer[i][s_i]->count, MPI_CHAR, i, PassMessage, MPI_COMM_WORLD);
          }
        }
      });
      std::thread recv_thread([&]() {
        std::vector<std::thread> threads;
        for (int step = 1; step < partitions; step++)
        {
          int i = (partition_id - step + partitions) % partitions;
          threads.emplace_back([&](int i) {
            for (int s_i = 0; s_i < sockets; s_i++)
            {
              MPI_Status recv_status;
              MPI_Probe(i, PassMessage, MPI_COMM_WORLD, &recv_status);
              MPI_Get_count(&recv_status, MPI_CHAR, &recv_buffer[i][s_i]->count);
              MPI_Recv(recv_buffer[i][s_i]->data, recv_buffer[i][s_i]->count, MPI_CHAR, i, PassMessage, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
              recv_buffer[i][s_i]->count /= sizeofM<M>(feature_size);
            }
          },
                               i);
        }
        for (int step = 1; step < partitions; step++)
        {
          int i = (partition_id - step + partitions) % partitions;

          threads[step - 1].join();
          recv_queue[recv_queue_size] = i;
          recv_queue_mutex.lock();
          recv_queue_size += 1;
          recv_queue_mutex.unlock();
        }
        recv_queue[recv_queue_size] = partition_id;
        recv_queue_mutex.lock();
        recv_queue_size += 1;
        recv_queue_mutex.unlock();
      });

      if (process_overlap)
      {

        //pipeline
        current_send_part_id = partition_id;

        double cuda_sync_time = 0;
        cuda_sync_time -= MPI_Wtime();
        if(rtminfo->forward){
        forward_gpu_CSC_partition(
            input_gpu_or_cpu,
            output_cpu,
            graph_partitions,
            feature_size,
            (current_send_part_id + 1) % partitions,
            cuda_stream);
        }else{
        backward_gpu_CSR_partition(
            input_gpu_or_cpu,
            output_cpu,
            graph_partitions,
            feature_size,
            (current_send_part_id + 1) % partitions,
            cuda_stream);
        }
        cuda_stream->CUDA_DEVICE_SYNCHRONIZE();
        cuda_sync_time += MPI_Wtime();
        this->all_cuda_sync_time += cuda_sync_time;
      }
      else
      {

        for (int step = 0; step < partitions; step++)
        {

          current_send_part_id = (current_send_part_id + 1) % partitions;
          double cuda_sync_time = 0;
          cuda_sync_time -= MPI_Wtime();

          if(rtminfo->forward){
            forward_gpu_CSC_partition(
                input_gpu_or_cpu,
                output_cpu,
                graph_partitions,
                feature_size,
                (current_send_part_id + 1) % partitions,
                cuda_stream);
            }else{
            backward_gpu_CSR_partition(
                input_gpu_or_cpu,
                output_cpu,
                graph_partitions,
                feature_size,
                (current_send_part_id + 1) % partitions,
                cuda_stream);
            }
          cuda_stream->CUDA_DEVICE_SYNCHRONIZE();

          //no pipeline
          cuda_sync_time += MPI_Wtime();
          this->all_cuda_sync_time += cuda_sync_time;
          //no pipeline
        }
        //printf("emove data\n");
      }
      //printf("initialize send buffer finished %d\n", partition_id);

      for (int step = 0; step < partitions; step++)
      {
        current_send_part_id = (current_send_part_id + 1) % partitions;
        int i = current_send_part_id;
        for (int t_i = 0; t_i < threads; t_i++)
        {
          *thread_state[t_i] = tuned_chunks_dense[i][t_i];
          //    printf("Range %d\t%d %d\n",tuned_chunks_dense[i][t_i].curr,tuned_chunks_dense[i][t_i].end,partition_id);
        }
        //这后计算.

        cuda_stream->CUDA_DEVICE_SYNCHRONIZE();
        //free_all_tmp();
        if (current_send_part_id != partition_id)
        {
          double cuda_sync_time = 0;
          cuda_sync_time -= MPI_Wtime();
          if (process_overlap)
          {
            if(rtminfo->forward){
                forward_gpu_CSC_partition(
                    input_gpu_or_cpu,
                    output_cpu,
                    graph_partitions,
                    feature_size,
                    (current_send_part_id + 1) % partitions,
                    cuda_stream);
            }else{
                backward_gpu_CSR_partition(
                input_gpu_or_cpu,
                output_cpu,
                graph_partitions,
                feature_size,
                (current_send_part_id + 1) % partitions,
                cuda_stream);
            }
          }
          //cuda_stream->CUDA_DEVICE_SYNCHRONIZE();
          cuda_sync_time += MPI_Wtime();
          this->all_cuda_sync_time += cuda_sync_time;
        }
        else
        {
          if (process_overlap)
          {
            double replication_time = 0;
            replication_time -= MPI_Wtime();
            if (process_local)
            {
              // if (partition_id == 0)
              //  printf("process_replicate %d on layer %d\n", replication_threshold, layer_);
              //MPI_Barrier(MPI_COMM_WORLD);
              forward_gpu_local(graph_rep, local_data_buffer, cuda_stream, src, dst, weight, index_gpu_input, index_gpu_output, feature_size, layer_);
              //cuda_stream->CUDA_DEVICE_SYNCHRONIZE();
            }
            replication_time += MPI_Wtime();
            //MPI_Barrier(MPI_COMM_WORLD);
            all_replication_time += replication_time;
          }
        }

#pragma omp parallel
        {
          //printf("DEBUGstart%d\n",partition_id);
          int thread_id = omp_get_thread_num();
          int s_i = get_socket_id(thread_id);
          VertexId final_p_v_i = thread_state[thread_id]->end;

          //printf("DEBUG %d %d %d %d\n",thread_state[thread_id]->end,s_i,thread_id,partition_id);

          while (true)
          {
            VertexId begin_p_v_i = __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
            if (begin_p_v_i >= final_p_v_i)
              break;
            VertexId end_p_v_i = begin_p_v_i + basic_chunk;
            if (end_p_v_i > final_p_v_i)
            {
              end_p_v_i = final_p_v_i;
            }
            for (VertexId p_v_i = begin_p_v_i; p_v_i < end_p_v_i; p_v_i++)
            {
              VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
              //dense_signal(v_i, VertexAdjList<EdgeData>(incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i].index, incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i + 1].index));
              flush_data_to_send_buffer_buffer_lock_free_write(thread_id, feature_size, v_i, output_cpu + v_i * feature_size, message_write_offset[layer_][v_i]);
            }
          }
          //  printf("Send initial has been finished %d\n", partition_id);
          thread_state[thread_id]->status = STEALING;
          for (int t_offset = 1; t_offset < threads; t_offset++)
          {
            int t_i = (thread_id + t_offset) % threads;
            int s_i = get_socket_id(t_i);
            while (thread_state[t_i]->status != STEALING)
            {
              VertexId begin_p_v_i = __sync_fetch_and_add(&thread_state[t_i]->curr, basic_chunk);
              if (begin_p_v_i >= thread_state[t_i]->end)
                break;
              VertexId end_p_v_i = begin_p_v_i + basic_chunk;
              if (end_p_v_i > thread_state[t_i]->end)
              {
                end_p_v_i = thread_state[t_i]->end;
              }
              for (VertexId p_v_i = begin_p_v_i; p_v_i < end_p_v_i; p_v_i++)
              {
                VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
                flush_data_to_send_buffer_buffer_lock_free_write(t_i, feature_size, v_i, output_cpu + v_i * feature_size, message_write_offset[layer_][v_i]);
                //dense_signal(v_i, VertexAdjList<EdgeData>(incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i].index, incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i + 1].index));
              }
            }
          }
        }
        flush_data_to_send_buffer_buffer_lock_free_init(message_amount[layer_][current_send_part_id]);
        if (message_amount[layer_][current_send_part_id] > (partition_offset[current_send_part_id + 1] - partition_offset[current_send_part_id]))
          printf("DEASTER\n");
        if (i != partition_id)
        {
          //           for (int s_i=0;s_i<sockets;s_i++) {
          //            send_buffer[i][s_i]->count=(partition_offset[i+1] - partition_offset[i]);
          //          }
          send_queue[send_queue_size] = i;
          send_queue_mutex.lock();
          send_queue_size += 1;
          send_queue_mutex.unlock();
        }
      }

      compute_time += MPI_Wtime();
      all_compute_time += compute_time;

      double overlap_time = 0;
      overlap_time -= MPI_Wtime();
      // //process_local
      for (int step = 0; step < partitions; step++)
      {
        int wait_time = 0;
        wait_time -= MPI_Wtime();
        while (true)
        {
          recv_queue_mutex.lock();
          bool condition = (recv_queue_size <= step);
          recv_queue_mutex.unlock();
          if (!condition)
            break;
          __asm volatile("pause" ::
                             : "memory");
        }
        int i = recv_queue[step];
        MessageBuffer **used_buffer;
        if (i == partition_id)
        {
          used_buffer = send_buffer[i];
        }
        else
        {
          used_buffer = recv_buffer[i];
        }
        wait_time += MPI_Wtime();
        all_recv_wait_time += wait_time;

        //add GPU_aggregator
        //copy data to gpu  size: dst:  used_buffer[s_i]->count*(feature_size+1)
        for (int s_i = 0; s_i < sockets; s_i++)
        {
          cuda_stream->CUDA_DEVICE_SYNCHRONIZE();
          if (used_buffer[s_i]->count > 0)
          {
            //printf("%ld %ld %ld\n", used_buffer[s_i]->count, max_recv_buffer_size, max_recv_buffer_size * (feature_size + 1));
            int recv_copy = 0;
            recv_copy -= MPI_Wtime();
            cuda_stream->move_data_in(gpu_memory_buffer, (float *)used_buffer[s_i]->data, 0, used_buffer[s_i]->count, (feature_size + 1), false);
            //cuda_stream->CUDA_DEVICE_SYNCHRONIZE();
            if (partition_id == 0)
              for (int i = 0; i < used_buffer[s_i]->count; i++)
              {
                VertexId *S = (VertexId *)used_buffer[s_i]->data;
                if (S[i * (feature_size + 1)] < partition_offset[partition_id] || S[i * (feature_size + 1)] >= partition_offset[partition_id + 1])
                  printf("something wroing %d %d\n", S[i * feature_size + 1], partition_id);
              }

            recv_copy += MPI_Wtime();
            all_recv_copy_time += recv_copy;
            int recv_kernel = 0;
            recv_kernel -= MPI_Wtime();
            //printf("%d\n", partition_offset[partition_id]);
            cuda_stream->aggregate_comm_result_debug(local_data_buffer, gpu_memory_buffer, used_buffer[s_i]->count, feature_size, partition_offset[partition_id], partition_offset[partition_id + 1], false);
            //cuda_stream->CUDA_DEVICE_SYNCHRONIZE();
            recv_kernel += MPI_Wtime();
            all_recv_kernel_time += recv_kernel;

            //printf("remove comm\n");
          }
        }
        //aggregate

        //sync
      }
      //printf("Send has been finished %d\n", partition_id);

      overlap_time += MPI_Wtime();
      all_overlap_time += overlap_time;

      // send_thread.join();
      // recv_thread.join();
      cuda_stream->CUDA_DEVICE_SYNCHRONIZE();

      if (!process_overlap)
      {
        double replication_time = 0;
        replication_time -= MPI_Wtime();
        if (process_local)
        {
          //if (partition_id == 0)
          //printf("process_replicate %d on layer %d\n", replication_threshold, layer_);
          //MPI_Barrier(MPI_COMM_WORLD);
          forward_gpu_local(graph_rep, local_data_buffer, cuda_stream, src, dst, weight, index_gpu_input, index_gpu_output, feature_size, layer_);
          cuda_stream->CUDA_DEVICE_SYNCHRONIZE();
        }
        replication_time += MPI_Wtime();
        //MPI_Barrier(MPI_COMM_WORLD);
        all_replication_time += replication_time;
      }

      //process_local
      //  double wait_time=0;
      //  wait_time-=MPI_Wtime();
      FreeBuffer(gpu_memory_buffer);
      cuda_stream->destory_Stream();
      // wait_time+=MPI_Wtime();
      // all_wait_time+=wait_time;
      send_thread.join();
      recv_thread.join();

      delete[] send_queue;
      delete[] recv_queue;
      if (process_local)
      {
        FreeEdge(src);
        FreeEdge(dst);
        FreeBuffer(weight);
        FreeEdge(index_gpu_input);
        FreeEdge(index_gpu_output);
        FreeBuffer(graph_rep[layer_].rep_feature_gpu_buffer);
      }
      free_all_tmp();
      //      printf("con%d_%d\n",con,partition_offset[1]);
    }

    R global_reducer;
    stream_time += MPI_Wtime();
#ifdef PRINT_DEBUG_MESSAGES
    if (partition_id == 0)
    {
      printf("process_edges took %lf (s)\n", stream_time);
    }
#endif
    return global_reducer;
  }

    
    // process edges
  template <typename R, typename M>
  R sync_compute(torch::Tensor &input_gpu_or_cpu,
                 float *output_cpu,
                 std::vector<CSC_segment_pinned *> &graph_partitions,
                 std::function<void(VertexId)> sparse_signal,
                 float *local_data_buffer)
  {
    
      
    Bitmap* active = alloc_vertex_subset();
    active->fill();
    int feature_size = gnnctx->layer_size[rtminfo->curr_layer];
    bool process_local = rtminfo->process_local;
    int layer_ = rtminfo->curr_layer;
    bool process_overlap = rtminfo->process_overlap;

    process_local = process_local && (graph_rep[layer_].rep_edge_size > 0);

    if (partition_id == 0)
    {
      printf("SyncComputeLite:layer(%d).process_local(%d).dimension(%d).reduce_comm(%d).overlap(%d)\n", layer_, process_local ? replication_threshold : -1, graph_rep[layer_].feature_size, process_local, process_overlap);
    }
    double stream_time = 0;
    stream_time -= MPI_Wtime();
    size_t basic_chunk = 64;

    for (int t_i = 0; t_i < threads; t_i++)
    {
      //  printf("alloc local send buffer %d\n",sizeofM<M>(feature_size)*local_send_buffer_limit);
      local_send_buffer[t_i]->resize(sizeofM<M>(feature_size) * local_send_buffer_limit);
      local_send_buffer[t_i]->count = 0;
    }

    long max_recv_buffer_size = owned_vertices * sockets;
    long max_partition_size=0;
    
    for (int i = 0; i < partitions; i++)
    {
      for (int s_i = 0; s_i < sockets; s_i++)
      {
        recv_buffer[i][s_i]->resize_pinned(sizeofM<M>(feature_size) * (partition_offset[i + 1] - partition_offset[i]) * sockets);
        send_buffer[i][s_i]->resize_pinned(sizeofM<M>(feature_size) * owned_vertices * sockets);
        send_buffer[i][s_i]->count = 0;
        recv_buffer[i][s_i]->count = 0;
        max_recv_buffer_size = std::max(max_recv_buffer_size, (long)(partition_offset[i + 1] - partition_offset[i])*sockets);
        max_partition_size=std::max(max_partition_size,(long)(partition_offset[i + 1] - partition_offset[i]));
      }
    }
    //  printf("initialize send buffer finished %d\n", partition_id);
    float *gpu_memory_buffer = NULL;
    float *gpu_input_buffer = NULL;
    VertexId *index_gpu_input = NULL;
    //index_gpu_input=(VertexId*)cudaMallocGPU(vertices*sizeof(VertexId));
    VertexId *index_gpu_output = NULL;
    //index_gpu_output=(VertexId*)cudaMallocGPU(vertices*sizeof(VertexId));
    Cuda_Stream *cuda_stream = new Cuda_Stream();
    allocate_gpu_buffer(&gpu_memory_buffer, max_recv_buffer_size * (feature_size + 1));
    allocate_gpu_buffer(&gpu_input_buffer, max_partition_size * (feature_size+1));
    // if (gpu_memory_buffer == NULL)
    // {
    //   printf("something wrong\n");
    // }
    
    zero_buffer(local_data_buffer, (partition_offset[partition_id + 1] - partition_offset[partition_id]) * (feature_size));

    size_t max_dstrange = 0, max_edge_size = 0;
    for (int i = 0; i < graph_partitions.size(); i++)
    {
      max_edge_size = std::max(max_edge_size, (size_t)(graph_partitions[i]->edge_size));
      max_dstrange = std::max(max_dstrange, (size_t)(graph_partitions[i]->dst_range[1] - graph_partitions[i]->dst_range[0]));
    };

    weight_gpu_intergate = (float *)cudaMallocGPU(max_edge_size * sizeof(float));
    row_indices_intergate = (VertexId *)cudaMallocGPU(max_edge_size * sizeof(VertexId));
    column_offset_intergate = (VertexId *)cudaMallocGPU((max_dstrange + 1) * sizeof(VertexId));

    VertexId *src;
    VertexId *dst;
    float *weight;

    if (process_local)
    {
      index_gpu_output = (VertexId *)cudaMallocGPU(vertices * sizeof(VertexId));
      index_gpu_input = (VertexId *)cudaMallocGPU(vertices * sizeof(VertexId));
      graph_rep[layer_].rep_feature_gpu_buffer = (float *)cudaMallocGPU(((long)sizeof(float)) * graph_rep[layer_].rep_node_size * graph_rep[layer_].feature_size + 1);
      src = (VertexId *)cudaMallocGPU(sizeof(VertexId) * graph_rep[layer_].rep_edge_size + 1);
      dst = (VertexId *)cudaMallocGPU(sizeof(VertexId) * graph_rep[layer_].rep_edge_size + 1);
      weight = (float *)cudaMallocGPU(sizeof(float) * graph_rep[layer_].rep_edge_size + 1);
      //	printf("initialize send buffer finished %d\n", partition_id);
    }
 
    {

      if (process_local)
      {
        forward_gpu_local(graph_rep, local_data_buffer, cuda_stream, src, dst, weight, index_gpu_input, index_gpu_output, feature_size, layer_); 
      }
      
      int *send_queue = new int[partitions];
      int *recv_queue = new int[partitions];
      volatile int send_queue_size = 0;
      volatile int recv_queue_size = 0;
      std::mutex send_queue_mutex;
      std::mutex recv_queue_mutex;

      current_send_part_id = partition_id;
#pragma omp parallel for
      for (VertexId begin_v_i = partition_offset[partition_id]; begin_v_i < partition_offset[partition_id + 1]; begin_v_i += basic_chunk)
      {
        VertexId v_i = begin_v_i;
        unsigned long word = active->data[WORD_OFFSET(v_i)];
        while (word != 0)
        {
          if (word & 1)
          {
            sparse_signal(v_i);
          }
          v_i++;
          word = word >> 1;
        }
      }
#pragma omp parallel for
      for (int t_i = 0; t_i < threads; t_i++)
      {
        flush_local_send_buffer<M>(t_i);
      }
      recv_queue[recv_queue_size] = partition_id;
      recv_queue_mutex.lock();
      recv_queue_size += 1;
      recv_queue_mutex.unlock();
      std::thread send_thread([&]() {
        for (int step = 1; step < partitions; step++)
        {
          int i = (partition_id - step + partitions) % partitions;
          for (int s_i = 0; s_i < sockets; s_i++)
          {
            MPI_Send(send_buffer[partition_id][s_i]->data, sizeofM<M>(feature_size) * send_buffer[partition_id][s_i]->count, MPI_CHAR, i, PassMessage, MPI_COMM_WORLD);
          }
        }
      });
      std::thread recv_thread([&]() {
        for (int step = 1; step < partitions; step++)
        {
          int i = (partition_id + step) % partitions;
          for (int s_i = 0; s_i < sockets; s_i++)
          {
            MPI_Status recv_status;
            MPI_Probe(i, PassMessage, MPI_COMM_WORLD, &recv_status);
            MPI_Get_count(&recv_status, MPI_CHAR, &recv_buffer[i][s_i]->count);
            MPI_Recv(recv_buffer[i][s_i]->data, recv_buffer[i][s_i]->count, MPI_CHAR, i, PassMessage, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            recv_buffer[i][s_i]->count /= sizeofM<M>(feature_size);
          }
          recv_queue[recv_queue_size] = i;
          recv_queue_mutex.lock();
          recv_queue_size += 1;
          recv_queue_mutex.unlock();
        }
      });
      int current_recv_part_id = partition_id;
      for (int step = 0; step < partitions; step++)
      {
        while (true)
        {
          recv_queue_mutex.lock();
          bool condition = (recv_queue_size <= step);
          recv_queue_mutex.unlock();
          if (!condition)
            break;
          __asm volatile("pause" ::
                             : "memory");
        }
        int i = recv_queue[step];
        MessageBuffer **used_buffer;
        if (i == partition_id)
        {
          used_buffer = send_buffer[i];
        }
        else
        {
          used_buffer = recv_buffer[i];
        }

        for (int s_i = 0; s_i < sockets; s_i++)
        {
            //if(partition_id==1&&i==1){
            int current_recv_part_id = i;
            cuda_stream->move_data_in(gpu_memory_buffer, (float *)used_buffer[s_i]->data, 0, used_buffer[s_i]->count, (feature_size + 1), false);
            zero_buffer(gpu_input_buffer, (partition_offset[i + 1] - partition_offset[i]) * (feature_size));
            //printf("partition offset %d %d\n",partition_offset[i],partition_offset[i+1]);
            cuda_stream->deSerializeToGPU(gpu_input_buffer, gpu_memory_buffer, used_buffer[s_i]->count, feature_size, partition_offset[i], partition_offset[i + 1], false);    
            //zero_buffer(local_data_buffer, (partition_offset[partition_id + 1] - partition_offset[partition_id]) * (feature_size));
//            if(partition_id==0)
//            printf("DEBUG:: %d %d %d %d\n", graph_partitions[i]->src_range[0],
//                                          graph_partitions[i]->src_range[1],
//                                          graph_partitions[i]->dst_range[0],
//                                          graph_partitions[i]->dst_range[1]);
              forward_gpu_CSC_partition_from_buffer(
              gpu_input_buffer,
              local_data_buffer,
              graph_partitions,
              feature_size,
              i,
              cuda_stream);
           //   cuda_stream->CUDA_DEVICE_SYNCHRONIZE();
           //}
        }
      }

      cuda_stream->CUDA_DEVICE_SYNCHRONIZE();

      
      FreeBuffer(gpu_memory_buffer);
      FreeBuffer(gpu_input_buffer);
      cuda_stream->destory_Stream();
      send_thread.join();
      recv_thread.join();
      delete[] recv_queue;
      delete[] send_queue;
   
      if (process_local)
      {
        FreeEdge(src);
        FreeEdge(dst);
        FreeBuffer(weight);
        FreeEdge(index_gpu_input);
        FreeEdge(index_gpu_output);
        FreeBuffer(graph_rep[layer_].rep_feature_gpu_buffer);
      }
      free_all_tmp();
      //      printf("con%d_%d\n",con,partition_offset[1]);
    }

    R global_reducer;
    //MPI_Datatype dt = get_mpi_data_type<R>();
    //MPI_Allreduce(&reducer, &global_reducer, 1, dt, MPI_SUM, MPI_COMM_WORLD);
    stream_time += MPI_Wtime();
#ifdef PRINT_DEBUG_MESSAGES
    if (partition_id == 0)
    {
      printf("process_edges took %lf (s)\n", stream_time);
    }
#endif
    return global_reducer;
  }
  
  
  
      // process edges
  template <typename R, typename M>
  R sync_compute_edge_computation(torch::Tensor &input_origin,//forward computation
                                  torch::Tensor &input_transferred,
                 std::vector<CSC_segment_pinned *> &graph_partitions,
                 std::function<void(VertexId)> sparse_signal,
                 std::function<torch::Tensor(torch::Tensor&)> PreComputation,
                 std::function<torch::Tensor(torch::Tensor&,torch::Tensor&,torch::Tensor&,torch::Tensor&,EdgeNNModule* edgeop)> EdgeComputation,
                 torch::Tensor &output)
  {
    
      
    Bitmap* active = alloc_vertex_subset();
    active->fill();
    int feature_size = gnnctx->layer_size[rtminfo->curr_layer];
    bool process_local = rtminfo->process_local;
    int layer_ = rtminfo->curr_layer;
    bool process_overlap = rtminfo->process_overlap;

    process_local = process_local && (graph_rep[layer_].rep_edge_size > 0);

    if (partition_id == 0)
    {
      printf("SyncComputeLite:layer(%d).process_local(%d).dimension(%d).reduce_comm(%d).overlap(%d)\n", layer_, process_local ? replication_threshold : -1, graph_rep[layer_].feature_size, process_local, process_overlap);
    }
    double stream_time = 0;
    stream_time -= MPI_Wtime();
    size_t basic_chunk = 64;

    for (int t_i = 0; t_i < threads; t_i++)
    {
      //  printf("alloc local send buffer %d\n",sizeofM<M>(feature_size)*local_send_buffer_limit);
      local_send_buffer[t_i]->resize(sizeofM<M>(feature_size) * local_send_buffer_limit);
      local_send_buffer[t_i]->count = 0;
    }

    long max_recv_buffer_size = owned_vertices * sockets;
    long max_partition_size=0;
    
    for (int i = 0; i < partitions; i++)
    {
      for (int s_i = 0; s_i < sockets; s_i++)
      {
        recv_buffer[i][s_i]->resize_pinned(sizeofM<M>(feature_size) * (partition_offset[i + 1] - partition_offset[i]) * sockets);
        send_buffer[i][s_i]->resize_pinned(sizeofM<M>(feature_size) * owned_vertices * sockets);
        send_buffer[i][s_i]->count = 0;
        recv_buffer[i][s_i]->count = 0;
        max_recv_buffer_size = std::max(max_recv_buffer_size, (long)(partition_offset[i + 1] - partition_offset[i])*sockets);
        max_partition_size=std::max(max_partition_size,(long)(partition_offset[i + 1] - partition_offset[i]));
      }
    }
    //  printf("initialize send buffer finished %d\n", partition_id);
    float *gpu_memory_buffer = NULL;
    float *gpu_input_buffer = NULL;
    Cuda_Stream *cuda_stream = new Cuda_Stream();
    allocate_gpu_buffer(&gpu_memory_buffer, max_recv_buffer_size * (feature_size + 1));
    allocate_gpu_buffer(&gpu_input_buffer, max_partition_size * (feature_size+1));
    input_transferred=PreComputation(input_origin);
    
    {  
      int *send_queue = new int[partitions];
      int *recv_queue = new int[partitions];
      volatile int send_queue_size = 0;
      volatile int recv_queue_size = 0;
      std::mutex send_queue_mutex;
      std::mutex recv_queue_mutex;

      current_send_part_id = partition_id;
#pragma omp parallel for
      for (VertexId begin_v_i = partition_offset[partition_id]; begin_v_i < partition_offset[partition_id + 1]; begin_v_i += basic_chunk)
      {
        VertexId v_i = begin_v_i;
        unsigned long word = active->data[WORD_OFFSET(v_i)];
        while (word != 0)
        {
          if (word & 1)
          {
            sparse_signal(v_i);
          }
          v_i++;
          word = word >> 1;
        }
      }
#pragma omp parallel for
      for (int t_i = 0; t_i < threads; t_i++)
      {
        flush_local_send_buffer<M>(t_i);
      }
      recv_queue[recv_queue_size] = partition_id;
      recv_queue_mutex.lock();
      recv_queue_size += 1;
      recv_queue_mutex.unlock();
      std::thread send_thread([&]() {
        for (int step = 1; step < partitions; step++)
        {
          int i = (partition_id - step + partitions) % partitions;
          for (int s_i = 0; s_i < sockets; s_i++)
          {
            MPI_Send(send_buffer[partition_id][s_i]->data, sizeofM<M>(feature_size) * send_buffer[partition_id][s_i]->count, MPI_CHAR, i, PassMessage, MPI_COMM_WORLD);
          }
        }
      });
      std::thread recv_thread([&]() {
        for (int step = 1; step < partitions; step++)
        {
          int i = (partition_id + step) % partitions;
          for (int s_i = 0; s_i < sockets; s_i++)
          {
            MPI_Status recv_status;
            MPI_Probe(i, PassMessage, MPI_COMM_WORLD, &recv_status);
            MPI_Get_count(&recv_status, MPI_CHAR, &recv_buffer[i][s_i]->count);
            MPI_Recv(recv_buffer[i][s_i]->data, recv_buffer[i][s_i]->count, MPI_CHAR, i, PassMessage, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            recv_buffer[i][s_i]->count /= sizeofM<M>(feature_size);
          }
          recv_queue[recv_queue_size] = i;
          recv_queue_mutex.lock();
          recv_queue_size += 1;
          recv_queue_mutex.unlock();
        }
      });
      int current_recv_part_id = partition_id;
      for (int step = 0; step < partitions; step++)
      {
        while (true)
        {
          recv_queue_mutex.lock();
          bool condition = (recv_queue_size <= step);
          recv_queue_mutex.unlock();
          if (!condition)
            break;
          __asm volatile("pause" ::
                             : "memory");
        }
        int i = recv_queue[step];
        MessageBuffer **used_buffer;
        if (i == partition_id)
        {
          used_buffer = send_buffer[i];
        }
        else
        {
          used_buffer = recv_buffer[i];
        }
        
        for (int s_i = 0; s_i < sockets; s_i++)
        {
            //if(partition_id==1&&i==1){
            int current_recv_part_id = i;
            cuda_stream->move_data_in(gpu_memory_buffer, (float *)used_buffer[s_i]->data, 0, used_buffer[s_i]->count, (feature_size + 1), false);
            zero_buffer(gpu_input_buffer, (partition_offset[i + 1] - partition_offset[i]) * (feature_size));
            //printf("partition offset %d %d\n",partition_offset[i],partition_offset[i+1]);
            cuda_stream->deSerializeToGPU(gpu_input_buffer, gpu_memory_buffer, used_buffer[s_i]->count, feature_size, partition_offset[i], partition_offset[i + 1], false);
            // torch::from_blob(graph->in_degree + graph->partition_offset[graph->partition_id], {embedding->rownum, 1});
            torch::Tensor mirror_inputs=torch::from_blob(gpu_input_buffer,{partition_offset[i + 1] - partition_offset[i],feature_size},at::TensorOptions().requires_grad(true).device_index(0).dtype(torch::kFloat));
            
            EdgeOp->InitBlock(graph_partitions[i],
                              gnnctx->layer_size[rtminfo->curr_layer],
                              gnnctx->layer_size[rtminfo->curr_layer+1],
                              current_recv_part_id,
                              layer_,
                              cuda_stream
                             );
            encode_partition=i;// must specify the encode_partition, or the message will be flushed even though call encode function.
            torch::Tensor mirror_inputs_transferred;
            torch::Tensor message=  EdgeComputation(mirror_inputs,mirror_inputs_transferred,input_origin,input_transferred,EdgeOp);
            EdgeOp->GatherByDstFromSrc(output, mirror_inputs_transferred, message);
            
        }
      }

      cuda_stream->CUDA_DEVICE_SYNCHRONIZE();

      
      FreeBuffer(gpu_memory_buffer);
      FreeBuffer(gpu_input_buffer);
      cuda_stream->destory_Stream();
      send_thread.join();
      recv_thread.join();
      delete[] recv_queue;
      delete[] send_queue;
   
      free_all_tmp();
    }

    R global_reducer;
    stream_time += MPI_Wtime();
#ifdef PRINT_DEBUG_MESSAGES
    if (partition_id == 0)
    {
      printf("process_edges took %lf (s)\n", stream_time);
    }
#endif
    return global_reducer;
  }

  
    template <typename R, typename M>
  R compute_sync_edge_computation(torch::Tensor &input_grad,
                                          torch::Tensor &dst_input,
                                          float *output_cpu,
                                          std::vector<CSC_segment_pinned *> &graph_partitions,
                                          std::function<void(VertexId, VertexAdjList<EdgeData>)> dense_signal,
                                          std::function<torch::Tensor(torch::Tensor&)> PreComputation,
                                          std::function<torch::Tensor(torch::Tensor&,torch::Tensor&,torch::Tensor&,torch::Tensor&,EdgeNNModule* edgeop)> EdgeForward,
                                          std::function<torch::Tensor(torch::Tensor&,torch::Tensor&,EdgeNNModule* edgeop)> EdgeBackward,
                                          torch::Tensor &output_grad)//backward
  {

    int feature_size = gnnctx->layer_size[rtminfo->curr_layer];
    bool process_local = rtminfo->process_local;
    int layer_ = rtminfo->curr_layer;
    bool process_overlap = rtminfo->process_overlap;

    process_local = process_local && (graph_rep[layer_].rep_edge_size > 0);

    if (partition_id == 0)
    {
      printf("ComputeSyncLite:layer(%d).process_local(%d).dimension(%d).reduce_comm(%d).overlap(%d)\n", layer_, process_local ? replication_threshold : -1, graph_rep[layer_].feature_size, process_local, process_overlap);
    }
    double stream_time = 0;
    stream_time -= MPI_Wtime();
    double compute_time = 0;
    compute_time -= MPI_Wtime();
    for (int t_i = 0; t_i < threads; t_i++)
    {
      local_send_buffer[t_i]->resize(sizeofM<M>(feature_size) * local_send_buffer_limit);
      local_send_buffer[t_i]->count = 0;
    }
    long max_recv_buffer_size = owned_vertices * sockets;
    for (int i = 0; i < partitions; i++)
    {
      for (int s_i = 0; s_i < sockets; s_i++)
      {
        recv_buffer[i][s_i]->resize_pinned(sizeofM<M>(feature_size) * owned_vertices * sockets);
        send_buffer[i][s_i]->resize_pinned(sizeofM<M>(feature_size) * (partition_offset[i + 1] - partition_offset[i]) * sockets);
        send_buffer[i][s_i]->count = 0;
        recv_buffer[i][s_i]->count = 0;
      }
    }
    
    
    
    float *local_data_buffer=output_grad.packed_accessor<float,2>().data();
            
    float *gpu_memory_buffer = NULL;
    Cuda_Stream *cuda_stream = new Cuda_Stream();
    allocate_gpu_buffer(&gpu_memory_buffer, max_recv_buffer_size * (feature_size + 1));

    
    zero_buffer(local_data_buffer, (partition_offset[partition_id + 1] - partition_offset[partition_id]) * (feature_size));
        
    torch::Tensor dst_input_transferred =PreComputation(dst_input);
    
    
    size_t basic_chunk = 64;
    {
#ifdef PRINT_DEBUG_MESSAGES
      if (partition_id == 0)
      {
        printf("dense mode\n");
      }
#endif
      int *send_queue = new int[partitions];
      int *recv_queue = new int[partitions];
      volatile int send_queue_size = 0;
      volatile int recv_queue_size = 0;
      std::mutex send_queue_mutex;
      std::mutex recv_queue_mutex;

      std::thread send_thread([&]() {
        for (int step = 0; step < partitions; step++)
        {
          if (step == partitions - 1)
          {
            break;
          }
          while (true)
          {
            send_queue_mutex.lock();
            bool condition = (send_queue_size <= step);
            send_queue_mutex.unlock();
            if (!condition)
              break;
            __asm volatile("pause" ::
                               : "memory");
          }
          int i = send_queue[step];
          for (int s_i = 0; s_i < sockets; s_i++)
          {
            MPI_Send(send_buffer[i][s_i]->data, sizeofM<M>(feature_size) * send_buffer[i][s_i]->count, MPI_CHAR, i, PassMessage, MPI_COMM_WORLD);
          }
        }
      });
      std::thread recv_thread([&]() {
        std::vector<std::thread> threads;
        for (int step = 1; step < partitions; step++)
        {
          int i = (partition_id - step + partitions) % partitions;
          threads.emplace_back([&](int i) {
            for (int s_i = 0; s_i < sockets; s_i++)
            {
              MPI_Status recv_status;
              MPI_Probe(i, PassMessage, MPI_COMM_WORLD, &recv_status);
              MPI_Get_count(&recv_status, MPI_CHAR, &recv_buffer[i][s_i]->count);
              MPI_Recv(recv_buffer[i][s_i]->data, recv_buffer[i][s_i]->count, MPI_CHAR, i, PassMessage, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
              recv_buffer[i][s_i]->count /= sizeofM<M>(feature_size);
            }
          },
                               i);
        }
        for (int step = 1; step < partitions; step++)
        {
          int i = (partition_id - step + partitions) % partitions;

          threads[step - 1].join();
          recv_queue[recv_queue_size] = i;
          recv_queue_mutex.lock();
          recv_queue_size += 1;
          recv_queue_mutex.unlock();
        }
        recv_queue[recv_queue_size] = partition_id;
        recv_queue_mutex.lock();
        recv_queue_size += 1;
        recv_queue_mutex.unlock();
      });

      if (process_overlap)//pipeline
      {
        current_send_part_id = partition_id;
        //COMPUTE CODE
   
      }
      else
      {
        //no pipeline
        for (int step = 0; step < partitions; step++)
        {
            current_send_part_id = (current_send_part_id + 1) % partitions;
            encode_partition=current_send_part_id;
            EdgeOp->InitBlock(graph_partitions[current_send_part_id],
                            gnnctx->layer_size[rtminfo->curr_layer],
                              gnnctx->layer_size[rtminfo->curr_layer+1],
                                current_send_part_id,
                                 layer_,
                                  cuda_stream); 
            torch::Tensor src_input_transferred;//mark
            torch::Tensor src_input;
            torch::Tensor message=EdgeForward(src_input,src_input_transferred,dst_input,dst_input_transferred,EdgeOp);
            
            torch::Tensor src_inter_grad=EdgeOp->NewLeafTensor(src_input_transferred);
            torch::Tensor message_grad=EdgeOp->NewLeafTensor(message);
            
            EdgeOp->GatherBySrcFromDst(src_inter_grad,input_grad,message);//4->3
            EdgeOp->BackwardScatterGradBackToWeight(src_input_transferred, input_grad, message_grad);//4-2
            torch::Tensor src_grad=EdgeBackward(message_grad,src_inter_grad,EdgeOp); //(2,3)->1
            EdgeOp->MoveResultOut(output_cpu,src_grad);
            
            cuda_stream->CUDA_DEVICE_SYNCHRONIZE();
            
        }
      }

      for (int step = 0; step < partitions; step++)
      {
        current_send_part_id = (current_send_part_id + 1) % partitions;
        int i = current_send_part_id;
        for (int t_i = 0; t_i < threads; t_i++)
        {
          *thread_state[t_i] = tuned_chunks_dense[i][t_i];
        }

        cuda_stream->CUDA_DEVICE_SYNCHRONIZE();
        if (current_send_part_id != partition_id)
        {
          if (process_overlap)
          {
            //COMPUTE CODE
        
          }
          //cuda_stream->CUDA_DEVICE_SYNCHRONIZE();
        }

#pragma omp parallel
        {
          //printf("DEBUGstart%d\n",partition_id);
          int thread_id = omp_get_thread_num();
          int s_i = get_socket_id(thread_id);
          VertexId final_p_v_i = thread_state[thread_id]->end;

          while (true)
          {
            VertexId begin_p_v_i = __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
            if (begin_p_v_i >= final_p_v_i)
              break;
            VertexId end_p_v_i = begin_p_v_i + basic_chunk;
            if (end_p_v_i > final_p_v_i)
            {
              end_p_v_i = final_p_v_i;
            }
            for (VertexId p_v_i = begin_p_v_i; p_v_i < end_p_v_i; p_v_i++)
            {
              VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
              dense_signal(v_i, VertexAdjList<EdgeData>(incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i].index, incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i + 1].index));
            }
          }
          //  printf("Send initial has been finished %d\n", partition_id);
          thread_state[thread_id]->status = STEALING;
          for (int t_offset = 1; t_offset < threads; t_offset++)
          {
            int t_i = (thread_id + t_offset) % threads;
            int s_i = get_socket_id(t_i);
            while (thread_state[t_i]->status != STEALING)
            {
              VertexId begin_p_v_i = __sync_fetch_and_add(&thread_state[t_i]->curr, basic_chunk);
              if (begin_p_v_i >= thread_state[t_i]->end)
                break;
              VertexId end_p_v_i = begin_p_v_i + basic_chunk;
              if (end_p_v_i > thread_state[t_i]->end)
              {
                end_p_v_i = thread_state[t_i]->end;
              }
              for (VertexId p_v_i = begin_p_v_i; p_v_i < end_p_v_i; p_v_i++)
              {
                VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
                dense_signal(v_i, VertexAdjList<EdgeData>(incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i].index, incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i + 1].index));
              }
            }
          }
        }
#pragma omp parallel for
        for (int t_i = 0; t_i < threads; t_i++)
        {
          flush_local_send_buffer_buffer(t_i, feature_size);
        }
        if (i != partition_id)
        {
          send_queue[send_queue_size] = i;
          send_queue_mutex.lock();
          send_queue_size += 1;
          send_queue_mutex.unlock();
        }
      }

      compute_time += MPI_Wtime();
      all_compute_time += compute_time;

      double overlap_time = 0;
      overlap_time -= MPI_Wtime();
      // //process_local
      for (int step = 0; step < partitions; step++)
      {
        int wait_time = 0;
        wait_time -= MPI_Wtime();
        while (true)
        {
          recv_queue_mutex.lock();
          bool condition = (recv_queue_size <= step);
          recv_queue_mutex.unlock();
          if (!condition)
            break;
          __asm volatile("pause" ::
                             : "memory");
        }
        int i = recv_queue[step];
        MessageBuffer **used_buffer;
        if (i == partition_id)
        {
          used_buffer = send_buffer[i];
        }
        else
        {
          used_buffer = recv_buffer[i];
        }
        wait_time += MPI_Wtime();
        all_recv_wait_time += wait_time;

        //add GPU_aggregator
        //copy data to gpu  size: dst:  used_buffer[s_i]->count*(feature_size+1)
        for (int s_i = 0; s_i < sockets; s_i++)
        {
          cuda_stream->CUDA_DEVICE_SYNCHRONIZE();
          if (used_buffer[s_i]->count > 0)
          {
   
            cuda_stream->move_data_in(gpu_memory_buffer, (float *)used_buffer[s_i]->data, 0, used_buffer[s_i]->count, (feature_size + 1), false);
            cuda_stream->aggregate_comm_result_debug(local_data_buffer, gpu_memory_buffer, used_buffer[s_i]->count, feature_size, partition_offset[partition_id], partition_offset[partition_id + 1], false);
            
          }
        }

      }
      
      //COMBINE GRAD
      output_grad=output_grad+dst_input.grad();
      
      overlap_time += MPI_Wtime();
      all_overlap_time += overlap_time;

      cuda_stream->CUDA_DEVICE_SYNCHRONIZE();

      FreeBuffer(gpu_memory_buffer);
      cuda_stream->destory_Stream();
      send_thread.join();
      recv_thread.join();

      delete[] send_queue;
      delete[] recv_queue;
      free_all_tmp();
    }

    R global_reducer;
    stream_time += MPI_Wtime();
#ifdef PRINT_DEBUG_MESSAGES
    if (partition_id == 0)
    {
      printf("process_edges took %lf (s)\n", stream_time);
    }
#endif
    return global_reducer;
  }


    
    
  
 
  template <typename R, typename M>
  R compute_and_sync_with_GPU_aggregator_overlap(torch::Tensor &input_gpu_or_cpu,
                                                 float *output_cpu,
                                                 std::vector<CSC_segment_pinned *> &graph_partitions,
                                                 std::function<void(VertexId, VertexAdjList<EdgeData>)> dense_signal,
                                                 float *local_data_buffer,
                                                 int feature_size, bool process_local = false, int layer_ = 0, bool process_overlap = false)
  {

    process_local = process_local && (graph_rep[layer_].rep_edge_size > 0);

    if (partition_id == 0)
    {
      printf("layer(%d).process_local(%d).dimension(%d).reduce_comm(%d).overlap(%d)\n", layer_, process_local ? replication_threshold : -1, graph_rep[layer_].feature_size, process_local, process_overlap);
    }
    double stream_time = 0;
    stream_time -= MPI_Wtime();
    //    printf("initialize not been finished %d\n", partition_id);
    double compute_time = 0;
    compute_time -= MPI_Wtime();
    for (int t_i = 0; t_i < threads; t_i++)
    {
      local_send_buffer[t_i]->resize(sizeof(MsgUnit<M>) * local_send_buffer_limit);
      local_send_buffer[t_i]->count = 0;
    }
    size_t max_recv_buffer_size = owned_vertices * sockets;
    for (int i = 0; i < partitions; i++)
    {
      for (int s_i = 0; s_i < sockets; s_i++)
      {
        recv_buffer[i][s_i]->resize_pinned(sizeof(MsgUnit<M>) * owned_vertices * sockets);
        send_buffer[i][s_i]->resize_pinned(sizeof(MsgUnit<M>) * (partition_offset[i + 1] - partition_offset[i]) * sockets);
        send_buffer[i][s_i]->count = 0;
        recv_buffer[i][s_i]->count = 0;
        //max_recv_buffer_size = std::max(max_recv_buffer_size, (int)(partition_offset[i + 1] - partition_offset[i]));
      }
    } //init the send and receive buffer
    //  printf("initialize send buffer finished %d\n", partition_id);
    float *gpu_memory_buffer = NULL;
    VertexId *index_gpu_input = NULL;
    //index_gpu_input=(VertexId*)cudaMallocGPU(vertices*sizeof(VertexId));
    VertexId *index_gpu_output = NULL;
    //index_gpu_output=(VertexId*)cudaMallocGPU(vertices*sizeof(VertexId));
    Cuda_Stream *cuda_stream = new Cuda_Stream();
    allocate_gpu_buffer(&gpu_memory_buffer, max_recv_buffer_size * (feature_size + 1));
    zero_buffer(local_data_buffer, (partition_offset[partition_id + 1] - partition_offset[partition_id]) * (feature_size));

    size_t max_dstrange = 0, max_edge_size = 0;
    for (int i = 0; i < graph_partitions.size(); i++)
    {
      max_edge_size = std::max(max_edge_size, (size_t)(graph_partitions[i]->edge_size));
      max_dstrange = std::max(max_dstrange, (size_t)(graph_partitions[i]->dst_range[1] - graph_partitions[i]->dst_range[0]));
    };

    weight_gpu_intergate = (float *)cudaMallocGPU(max_edge_size * sizeof(float));
    row_indices_intergate = (VertexId *)cudaMallocGPU(max_edge_size * sizeof(VertexId));
    column_offset_intergate = (VertexId *)cudaMallocGPU((max_dstrange + 1) * sizeof(VertexId));

    VertexId *src;
    VertexId *dst;
    float *weight;

    if (process_local)
    {
      index_gpu_output = (VertexId *)cudaMallocGPU(vertices * sizeof(VertexId));
      index_gpu_input = (VertexId *)cudaMallocGPU(vertices * sizeof(VertexId));
      graph_rep[layer_].rep_feature_gpu_buffer = (float *)cudaMallocGPU(((long)sizeof(float)) * graph_rep[layer_].rep_node_size * graph_rep[layer_].feature_size + 1);
      src = (VertexId *)cudaMallocGPU(sizeof(VertexId) * graph_rep[layer_].rep_edge_size + 1);
      dst = (VertexId *)cudaMallocGPU(sizeof(VertexId) * graph_rep[layer_].rep_edge_size + 1);
      weight = (float *)cudaMallocGPU(sizeof(float) * graph_rep[layer_].rep_edge_size + 1);
    }
    size_t basic_chunk = 64;
    {
#ifdef PRINT_DEBUG_MESSAGES
      if (partition_id == 0)
      {
        printf("dense mode\n");
      }
#endif
      int *send_queue = new int[partitions];
      int *recv_queue = new int[partitions];
      volatile int send_queue_size = 0;
      volatile int recv_queue_size = 0;
      std::mutex send_queue_mutex;
      std::mutex recv_queue_mutex;

      std::thread send_thread([&]() {
        for (int step = 0; step < partitions; step++)
        {
          if (step == partitions - 1)
          {
            break;
          }
          while (true)
          {
            send_queue_mutex.lock();
            bool condition = (send_queue_size <= step);
            send_queue_mutex.unlock();
            if (!condition)
              break;
            __asm volatile("pause" ::
                               : "memory");
          }
          int i = send_queue[step];
          for (int s_i = 0; s_i < sockets; s_i++)
          {
            MPI_Send(send_buffer[i][s_i]->data, sizeof(MsgUnit<M>) * send_buffer[i][s_i]->count, MPI_CHAR, i, PassMessage, MPI_COMM_WORLD);
          }
        }
      });
      std::thread recv_thread([&]() {
        std::vector<std::thread> threads;
        for (int step = 1; step < partitions; step++)
        {
          int i = (partition_id - step + partitions) % partitions;
          threads.emplace_back([&](int i) {
            for (int s_i = 0; s_i < sockets; s_i++)
            {
              MPI_Status recv_status;
              MPI_Probe(i, PassMessage, MPI_COMM_WORLD, &recv_status);
              MPI_Get_count(&recv_status, MPI_CHAR, &recv_buffer[i][s_i]->count);
              MPI_Recv(recv_buffer[i][s_i]->data, recv_buffer[i][s_i]->count, MPI_CHAR, i, PassMessage, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
              recv_buffer[i][s_i]->count /= sizeof(MsgUnit<M>);
            }
          },
                               i);
        }
        for (int step = 1; step < partitions; step++)
        {
          int i = (partition_id - step + partitions) % partitions;

          threads[step - 1].join();
          recv_queue[recv_queue_size] = i;
          recv_queue_mutex.lock();
          recv_queue_size += 1;
          recv_queue_mutex.unlock();
        }
        recv_queue[recv_queue_size] = partition_id;
        recv_queue_mutex.lock();
        recv_queue_size += 1;
        recv_queue_mutex.unlock();
      });

      if (process_overlap)
      {

        //pipeline
        current_send_part_id = partition_id;

        double cuda_sync_time = 0;
        cuda_sync_time -= MPI_Wtime();
        forward_gpu_CSC_partition(
            input_gpu_or_cpu,
            output_cpu,
            graph_partitions,
            feature_size,
            (current_send_part_id + 1) % partitions,
            cuda_stream);
        cuda_stream->CUDA_DEVICE_SYNCHRONIZE();
        cuda_sync_time += MPI_Wtime();
        this->all_cuda_sync_time += cuda_sync_time;
      }
      else
      {

        //no pipeline
        for (int step = 0; step < partitions; step++)
        {
          //pipeline
          //current_send_part_id = partition_id;
          //no pipeline
          current_send_part_id = (current_send_part_id + 1) % partitions;
          double cuda_sync_time = 0;
          cuda_sync_time -= MPI_Wtime();

          forward_gpu_CSC_partition(
              input_gpu_or_cpu,
              output_cpu,
              graph_partitions,
              feature_size,
              (current_send_part_id + 1) % partitions,
              cuda_stream);
          cuda_stream->CUDA_DEVICE_SYNCHRONIZE();

          //no pipeline
          cuda_sync_time += MPI_Wtime();
          this->all_cuda_sync_time += cuda_sync_time;
          //no pipeline
        }
      }
      //printf("initialize send buffer finished %d\n", partition_id);

      for (int step = 0; step < partitions; step++)
      {
        current_send_part_id = (current_send_part_id + 1) % partitions;
        int i = current_send_part_id;
        for (int t_i = 0; t_i < threads; t_i++)
        {
          *thread_state[t_i] = tuned_chunks_dense[i][t_i];
        }
        //这后计算.

        cuda_stream->CUDA_DEVICE_SYNCHRONIZE();
        //free_all_tmp();
        if (current_send_part_id != partition_id)
        {
          printf("overlap\n");
          double cuda_sync_time = 0;
          cuda_sync_time -= MPI_Wtime();
          if (process_overlap)
          {
            forward_gpu_CSC_partition(
                input_gpu_or_cpu,
                output_cpu,
                graph_partitions,
                feature_size,
                (current_send_part_id + 1) % partitions,
                cuda_stream);
          }
          //cuda_stream->CUDA_DEVICE_SYNCHRONIZE();
          cuda_sync_time += MPI_Wtime();
          this->all_cuda_sync_time += cuda_sync_time;
        }
        else
        {
          if (process_overlap)
          {
            double replication_time = 0;
            replication_time -= MPI_Wtime();
            if (process_local)
            {
              // if (partition_id == 0)
              //  printf("process_replicate %d on layer %d\n", replication_threshold, layer_);
              //MPI_Barrier(MPI_COMM_WORLD);
              forward_gpu_local(graph_rep, local_data_buffer, cuda_stream, src, dst, weight, index_gpu_input, index_gpu_output, feature_size, layer_);
              //cuda_stream->CUDA_DEVICE_SYNCHRONIZE();
            }
            replication_time += MPI_Wtime();
            //MPI_Barrier(MPI_COMM_WORLD);
            all_replication_time += replication_time;
          }
        }

#pragma omp parallel
        {
          //printf("DEBUGstart%d\n",partition_id);
          int thread_id = omp_get_thread_num();
          int s_i = get_socket_id(thread_id);
          VertexId final_p_v_i = thread_state[thread_id]->end;

          //printf("DEBUG %d %d %d %d\n",thread_state[thread_id]->end,s_i,thread_id,partition_id);

          while (true)
          {
            VertexId begin_p_v_i = __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
            if (begin_p_v_i >= final_p_v_i)
              break;
            VertexId end_p_v_i = begin_p_v_i + basic_chunk;
            if (end_p_v_i > final_p_v_i)
            {
              end_p_v_i = final_p_v_i;
            }
            for (VertexId p_v_i = begin_p_v_i; p_v_i < end_p_v_i; p_v_i++)
            {
              VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
              dense_signal(v_i, VertexAdjList<EdgeData>(incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i].index, incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i + 1].index));
            }
          }
          //  printf("Send initial has been finished %d\n", partition_id);
          thread_state[thread_id]->status = STEALING;
          for (int t_offset = 1; t_offset < threads; t_offset++)
          {
            int t_i = (thread_id + t_offset) % threads;
            int s_i = get_socket_id(t_i);
            while (thread_state[t_i]->status != STEALING)
            {
              VertexId begin_p_v_i = __sync_fetch_and_add(&thread_state[t_i]->curr, basic_chunk);
              if (begin_p_v_i >= thread_state[t_i]->end)
                break;
              VertexId end_p_v_i = begin_p_v_i + basic_chunk;
              if (end_p_v_i > thread_state[t_i]->end)
              {
                end_p_v_i = thread_state[t_i]->end;
              }
              for (VertexId p_v_i = begin_p_v_i; p_v_i < end_p_v_i; p_v_i++)
              {
                VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
                dense_signal(v_i, VertexAdjList<EdgeData>(incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i].index, incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i + 1].index));
              }
            }
          }
        }
        //  printf("Send start %d\n", partition_id);
#pragma omp parallel for
        for (int t_i = 0; t_i < threads; t_i++)
        {
          flush_local_send_buffer<M>(t_i);
        }
        if (i != partition_id)
        {
          //           for (int s_i=0;s_i<sockets;s_i++) {
          //            send_buffer[i][s_i]->count=(partition_offset[i+1] - partition_offset[i]);
          //          }
          send_queue[send_queue_size] = i;
          send_queue_mutex.lock();
          send_queue_size += 1;
          send_queue_mutex.unlock();
        }
      }

      compute_time += MPI_Wtime();
      all_compute_time += compute_time;

      double overlap_time = 0;
      overlap_time -= MPI_Wtime();
      // //process_local
      for (int step = 0; step < partitions; step++)
      {
        int wait_time = 0;
        wait_time -= MPI_Wtime();
        while (true)
        {
          recv_queue_mutex.lock();
          bool condition = (recv_queue_size <= step);
          recv_queue_mutex.unlock();
          if (!condition)
            break;
          __asm volatile("pause" ::
                             : "memory");
        }
        int i = recv_queue[step];
        MessageBuffer **used_buffer;
        if (i == partition_id)
        {
          used_buffer = send_buffer[i];
        }
        else
        {
          used_buffer = recv_buffer[i];
        }
        wait_time += MPI_Wtime();
        all_recv_wait_time += wait_time;

        //add GPU_aggregator
        //copy data to gpu  size: dst:  used_buffer[s_i]->count*(feature_size+1)
        for (int s_i = 0; s_i < sockets; s_i++)
        {
          cuda_stream->CUDA_DEVICE_SYNCHRONIZE();
          if (used_buffer[s_i]->count > 0)
          {
            int recv_copy = 0;
            recv_copy -= MPI_Wtime();
            cuda_stream->move_data_in(gpu_memory_buffer, (float *)used_buffer[s_i]->data, 0, used_buffer[s_i]->count, (feature_size + 1), false);
            //cuda_stream->CUDA_DEVICE_SYNCHRONIZE();
            recv_copy += MPI_Wtime();
            all_recv_copy_time += recv_copy;
            int recv_kernel = 0;
            recv_kernel -= MPI_Wtime();
            cuda_stream->aggregate_comm_result(local_data_buffer, gpu_memory_buffer, used_buffer[s_i]->count, feature_size, partition_offset[partition_id], false);
            //cuda_stream->CUDA_DEVICE_SYNCHRONIZE();
            recv_kernel += MPI_Wtime();
            all_recv_kernel_time += recv_kernel;
          }
        }
        //aggregate

        //sync
      }
      //printf("Send has been finished %d\n", partition_id);

      overlap_time += MPI_Wtime();
      all_overlap_time += overlap_time;

      send_thread.join();
      recv_thread.join();
      cuda_stream->CUDA_DEVICE_SYNCHRONIZE();

      if (!process_overlap)
      {
        double replication_time = 0;
        replication_time -= MPI_Wtime();
        if (process_local)
        {
          //if (partition_id == 0)
          //printf("process_replicate %d on layer %d\n", replication_threshold, layer_);
          //MPI_Barrier(MPI_COMM_WORLD);
          forward_gpu_local(graph_rep, local_data_buffer, cuda_stream, src, dst, weight, index_gpu_input, index_gpu_output, feature_size, layer_);
          cuda_stream->CUDA_DEVICE_SYNCHRONIZE();
        }
        replication_time += MPI_Wtime();
        //MPI_Barrier(MPI_COMM_WORLD);
        all_replication_time += replication_time;
      }

      //process_local
      //  double wait_time=0;
      //  wait_time-=MPI_Wtime();
      FreeBuffer(gpu_memory_buffer);
      cuda_stream->destory_Stream();
      // wait_time+=MPI_Wtime();
      // all_wait_time+=wait_time;

      delete[] send_queue;
      delete[] recv_queue;
      if (process_local)
      {
        FreeEdge(src);
        FreeEdge(dst);
        FreeBuffer(weight);
        FreeEdge(index_gpu_input);
        FreeEdge(index_gpu_output);
        FreeBuffer(graph_rep[layer_].rep_feature_gpu_buffer);
      }
      free_all_tmp();
      //      printf("con%d_%d\n",con,partition_offset[1]);
    }

    R global_reducer;
    stream_time += MPI_Wtime();
#ifdef PRINT_DEBUG_MESSAGES
    if (partition_id == 0)
    {
      printf("process_edges took %lf (s)\n", stream_time);
    }
#endif
    return global_reducer;
  }
  void forward_gpu_local(graph_replication *graph_rep, float *local_data_buffer,
                         Cuda_Stream *cuda_stream, VertexId *src, VertexId *dst, float *weight, VertexId *index, VertexId *index_output, int feature_size, int layer_)
  {
    //if(partition_id==0)

    //zero_buffer(local_data_buffer,(partition_offset[partition_id+1]-partition_offset[partition_id])*(feature_size));
    int layer = layer_;
    cuda_stream->move_data_in(graph_rep[layer].rep_feature_gpu_buffer,
                              graph_rep[layer].rep_feature,
                              0,
                              graph_rep[layer].rep_node_size,
                              graph_rep[layer].feature_size, false);

    cuda_stream->move_edge_in(index,
                              graph_rep[layer].src_map,
                              0,
                              vertices,
                              1,
                              false);
    cuda_stream->move_edge_in(index_output,
                              graph_rep[layer].dst_map,
                              0,
                              vertices,
                              1,
                              false);
    cuda_stream->move_edge_in(src,
                              graph_rep[layer].src_rep,
                              0,
                              graph_rep[layer].rep_edge_size,
                              1,
                              false);
    cuda_stream->move_edge_in(dst,
                              graph_rep[layer].dst_rep,
                              0,
                              graph_rep[layer].rep_edge_size,
                              1,
                              false);
    cuda_stream->move_data_in(weight,
                              graph_rep[layer].weight_rep,
                              0,
                              graph_rep[layer].rep_edge_size,
                              1,
                              false);

    // float * rep_output_buffer;
    // float * output_map;
    // VertexId rep_output_size;

    cuda_stream->process_local_inter(graph_rep[layer].output_buffer_gpu, graph_rep[layer].rep_feature_gpu_buffer,
                                     src, dst, index, index_output, weight, partition_offset[partition_id], partition_offset[partition_id + 1],
                                     graph_rep[layer].feature_size, graph_rep[layer].rep_edge_size, rep_output_size, false);

    //cuda_stream->CUDA_DEVICE_SYNCHRONIZE();
    //FreeBuffer(graph_rep[layer].rep_feature_gpu_buffer);
  }


  void forward_gpu_CSC_partition_from_buffer(
      float *input_gpu,
      float *output_gpu,
      std::vector<CSC_segment_pinned *> &graph_partitions,
      int feature_size,
      int current_send_partition_id,
      Cuda_Stream *cuda_stream,
      bool require_move_in = false)
  {

    double movein_time = 0;
    movein_time -= MPI_Wtime();
    int vertex_range = partition_offset[partition_id + 1] - partition_offset[partition_id];
    int dst_range = graph_partitions[current_send_partition_id]->dst_range[1] - graph_partitions[current_send_partition_id]->dst_range[0];

    cuda_stream->move_data_in(weight_gpu_intergate,
                              graph_partitions[current_send_partition_id]->edge_weight,
                              0,
                              graph_partitions[current_send_partition_id]->edge_size,
                              1, false);
    cuda_stream->move_edge_in(row_indices_intergate,
                              (VertexId *)(graph_partitions[current_send_partition_id]->row_indices),
                              0,
                              graph_partitions[current_send_partition_id]->edge_size,
                              1,
                              false);
    cuda_stream->move_edge_in(column_offset_intergate,
                              graph_partitions[current_send_partition_id]->column_offset,
                              0,
                              dst_range + 1,
                              1,
                              false);

    VertexId src_start = graph_partitions[current_send_partition_id]->src_range[0];
    VertexId src_end = graph_partitions[current_send_partition_id]->src_range[1];
    VertexId dst_start = graph_partitions[current_send_partition_id]->dst_range[0];
    VertexId dst_end = graph_partitions[current_send_partition_id]->dst_range[1];
    //cuda_stream->CUDA_DEVICE_SYNCHRONIZE();
    movein_time += MPI_Wtime();
    all_movein_time += movein_time;
    //kernal function call;
    /*
             * output_gpu
             * input_gpu
             * graph_partitions[i]->src
             * graph_partitions[i]->dst
             * graph_partitions[i]->weight;
             * graph_partitions[i]->src_range[0],[1];
             * graph_partitions[j]->dst_range[0],[1];
             * graph_partitions[i]->batch_size
             * graph_partitions[i]->edge_size
             * graph_partitions[i]->feature_size
             */
    // std::cout<<"run one batch"<<std::endl;
    double kernel_time = 0;
    kernel_time -= MPI_Wtime();

    cuda_stream->Gather_By_Dst_From_Src(input_gpu,
                                    output_gpu,
                                    weight_gpu_intergate, //data
                                    row_indices_intergate,
                                    column_offset_intergate, //graph
                                    src_start, src_end, dst_start, dst_end,
                                    graph_partitions[current_send_partition_id]->edge_size,
                                    graph_partitions[current_send_partition_id]->batch_size,
                                    feature_size, false);
    //cuda_stream->CUDA_DEVICE_SYNCHRONIZE();
    kernel_time += MPI_Wtime();
    all_kernel_time += kernel_time;
    //
    double moveout_time = 0;
    moveout_time -= MPI_Wtime();

//    cuda_stream->move_result_out(output_cpu + (graph_partitions[current_send_partition_id]->dst_range[0] * feature_size),
//                                 output_gpu_buffered.packed_accessor<float, 2>().data(),
//                                 graph_partitions[current_send_partition_id]->dst_range[0],
//                                 graph_partitions[current_send_partition_id]->dst_range[1],
//                                 feature_size, false);

    //cuda_stream->CUDA_DEVICE_SYNCHRONIZE();
    moveout_time += MPI_Wtime();
    all_moveout_time += moveout_time;
  }
  
    void forward_gpu_CSC_partition(
      torch::Tensor input_gpu_or_cpu,
      float *output_cpu,
      std::vector<CSC_segment_pinned *> &graph_partitions,
      int feature_size,
      int current_send_partition_id,
      Cuda_Stream *cuda_stream,
      bool require_move_in = false)
  {

    double movein_time = 0;
    movein_time -= MPI_Wtime();
    output_gpu_buffered.zero_();
    int vertex_range = partition_offset[partition_id + 1] - partition_offset[partition_id];
    int dst_range = graph_partitions[current_send_partition_id]->dst_range[1] - graph_partitions[current_send_partition_id]->dst_range[0];

    cuda_stream->move_data_in(weight_gpu_intergate,
                              graph_partitions[current_send_partition_id]->edge_weight,
                              0,
                              graph_partitions[current_send_partition_id]->edge_size,
                              1, false);
    cuda_stream->move_edge_in(row_indices_intergate,
                              (VertexId *)(graph_partitions[current_send_partition_id]->row_indices),
                              0,
                              graph_partitions[current_send_partition_id]->edge_size,
                              1,
                              false);
    cuda_stream->move_edge_in(column_offset_intergate,
                              graph_partitions[current_send_partition_id]->column_offset,
                              0,
                              dst_range + 1,
                              1,
                              false);

    VertexId src_start = graph_partitions[current_send_partition_id]->src_range[0];
    VertexId src_end = graph_partitions[current_send_partition_id]->src_range[1];
    VertexId dst_start = graph_partitions[current_send_partition_id]->dst_range[0];
    VertexId dst_end = graph_partitions[current_send_partition_id]->dst_range[1];
    //cuda_stream->CUDA_DEVICE_SYNCHRONIZE();
    movein_time += MPI_Wtime();
    all_movein_time += movein_time;
    //kernal function call;
    /*
             * output_gpu
             * input_gpu
             * graph_partitions[i]->src
             * graph_partitions[i]->dst
             * graph_partitions[i]->weight;
             * graph_partitions[i]->src_range[0],[1];
             * graph_partitions[j]->dst_range[0],[1];
             * graph_partitions[i]->batch_size
             * graph_partitions[i]->edge_size
             * graph_partitions[i]->feature_size
             */
    // std::cout<<"run one batch"<<std::endl;
    double kernel_time = 0;
    kernel_time -= MPI_Wtime();

    cuda_stream->Gather_By_Dst_From_Src(input_gpu_or_cpu.packed_accessor<float, 2>().data(),
                                    output_gpu_buffered.packed_accessor<float, 2>().data(),
                                    weight_gpu_intergate, //data
                                    row_indices_intergate,
                                    column_offset_intergate, //graph
                                    src_start, src_end, dst_start, dst_end,
                                    graph_partitions[current_send_partition_id]->edge_size,
                                    graph_partitions[current_send_partition_id]->batch_size,
                                    feature_size, false);
    //cuda_stream->CUDA_DEVICE_SYNCHRONIZE();
    kernel_time += MPI_Wtime();
    all_kernel_time += kernel_time;
    //
    double moveout_time = 0;
    moveout_time -= MPI_Wtime();

    cuda_stream->move_result_out(output_cpu + (graph_partitions[current_send_partition_id]->dst_range[0] * feature_size),
                                 output_gpu_buffered.packed_accessor<float, 2>().data(),
                                 graph_partitions[current_send_partition_id]->dst_range[0],
                                 graph_partitions[current_send_partition_id]->dst_range[1],
                                 feature_size, false);

    //cuda_stream->CUDA_DEVICE_SYNCHRONIZE();
    moveout_time += MPI_Wtime();
    all_moveout_time += moveout_time;
  }
    
    void backward_gpu_CSR_partition(
      torch::Tensor input_gpu_or_cpu,
      float *output_cpu,
      std::vector<CSC_segment_pinned *> &graph_partitions,
      int feature_size,
      int current_send_partition_id,
      Cuda_Stream *cuda_stream,
      bool require_move_in = false)
  {

    double movein_time = 0;
    movein_time -= MPI_Wtime();
    output_gpu_buffered.zero_();
    int vertex_range = graph_partitions[current_send_partition_id]->dst_range[1] - graph_partitions[current_send_partition_id]->dst_range[0];
    int dst_range = graph_partitions[current_send_partition_id]->src_range[1] - graph_partitions[current_send_partition_id]->src_range[0];

    cuda_stream->move_data_in(weight_gpu_intergate,
                              graph_partitions[current_send_partition_id]->edge_weight_backward,
                              0,
                              graph_partitions[current_send_partition_id]->edge_size,
                              1, false);
    cuda_stream->move_edge_in(row_indices_intergate,
                              (VertexId *)(graph_partitions[current_send_partition_id]->column_indices),
                              0,
                              graph_partitions[current_send_partition_id]->edge_size,
                              1,
                              false);
    cuda_stream->move_edge_in(column_offset_intergate,
                              graph_partitions[current_send_partition_id]->row_offset,
                              0,
                              dst_range + 1,
                              1,
                              false);

    VertexId src_start = graph_partitions[current_send_partition_id]->src_range[0];
    VertexId src_end = graph_partitions[current_send_partition_id]->src_range[1];
    VertexId dst_start = graph_partitions[current_send_partition_id]->dst_range[0];
    VertexId dst_end = graph_partitions[current_send_partition_id]->dst_range[1];
    //cuda_stream->CUDA_DEVICE_SYNCHRONIZE();
    movein_time += MPI_Wtime();
    all_movein_time += movein_time;
    //kernal function call;
    /*
             * output_gpu
             * input_gpu
             * graph_partitions[i]->src
             * graph_partitions[i]->dst
             * graph_partitions[i]->weight;
             * graph_partitions[i]->src_range[0],[1];
             * graph_partitions[j]->dst_range[0],[1];
             * graph_partitions[i]->batch_size
             * graph_partitions[i]->edge_size
             * graph_partitions[i]->feature_size
             */
    // std::cout<<"run one batch"<<std::endl;
    double kernel_time = 0;
    kernel_time -= MPI_Wtime();

    cuda_stream->Gather_By_Dst_From_Src(input_gpu_or_cpu.packed_accessor<float, 2>().data(),
                                    output_gpu_buffered.packed_accessor<float, 2>().data(),
                                    weight_gpu_intergate, //data
                                    row_indices_intergate,
                                    column_offset_intergate, //graph
                                    dst_start, dst_end, //down
                                    src_start, src_end, //up
                                    graph_partitions[current_send_partition_id]->edge_size,
                                    graph_partitions[current_send_partition_id]->batch_size_backward,
                                    feature_size, false);
    //cuda_stream->CUDA_DEVICE_SYNCHRONIZE();
    kernel_time += MPI_Wtime();
    all_kernel_time += kernel_time;
    //
    double moveout_time = 0;
    moveout_time -= MPI_Wtime();

    cuda_stream->move_result_out(output_cpu + (graph_partitions[current_send_partition_id]->src_range[0] * feature_size),
                                 output_gpu_buffered.packed_accessor<float, 2>().data(),
                                 graph_partitions[current_send_partition_id]->src_range[0],
                                 graph_partitions[current_send_partition_id]->src_range[1],
                                 feature_size, false);

    //cuda_stream->CUDA_DEVICE_SYNCHRONIZE();
    moveout_time += MPI_Wtime();
    all_moveout_time += moveout_time;
  }
    
    

  void free_all_tmp()
  {
    FreeEdge(row_indices_intergate);
    FreeEdge(column_offset_intergate);
    FreeBuffer(weight_gpu_intergate);
  }

  template <typename R, typename M>
  R process_edges_backward(
      std::function<void(VertexId, VertexAdjList<EdgeData>)> dense_signal,
      std::function<R(VertexId, M)> dense_slot,
      Bitmap *active,
      Bitmap *dense_selective = nullptr)
  {
    double stream_time = 0;
    stream_time -= MPI_Wtime();

    for (int t_i = 0; t_i < threads; t_i++)
    {
      local_send_buffer[t_i]->resize(sizeof(MsgUnit<M>) * local_send_buffer_limit);
      local_send_buffer[t_i]->count = 0;
    }
    R reducer = 0;
    //    EdgeId active_edges = process_vertices<EdgeId>(
    //      [&](VertexId vtx){
    //        return (EdgeId)out_degree[vtx];
    //      },
    //      active
    //    );
    //    bool sparse = false;//(active_edges < edges / 20);
    for (int i = 0; i < partitions; i++)
    {
      for (int s_i = 0; s_i < sockets; s_i++)
      {
        recv_buffer[i][s_i]->resize(sizeof(MsgUnit<M>) * owned_vertices * sockets);
        send_buffer[i][s_i]->resize(sizeof(MsgUnit<M>) * (partition_offset[i + 1] - partition_offset[i]) * sockets);
        send_buffer[i][s_i]->count = 0;
        recv_buffer[i][s_i]->count = 0;
      }
    } //init the send and receive buffer
    size_t basic_chunk = 64;
    {
      // dense selective bitmap
      if (dense_selective != nullptr && partitions > 1)
      {
        double sync_time = 0;
        sync_time -= get_time();
        std::thread send_thread([&]() {
          for (int step = 1; step < partitions; step++)
          {
            int recipient_id = (partition_id + step) % partitions;
            MPI_Send(dense_selective->data + WORD_OFFSET(partition_offset[partition_id]), owned_vertices / 64, MPI_UNSIGNED_LONG, recipient_id, PassMessage, MPI_COMM_WORLD);
          }
        });
        std::thread recv_thread([&]() {
          for (int step = 1; step < partitions; step++)
          {
            int sender_id = (partition_id - step + partitions) % partitions;
            MPI_Recv(dense_selective->data + WORD_OFFSET(partition_offset[sender_id]), (partition_offset[sender_id + 1] - partition_offset[sender_id]) / 64, MPI_UNSIGNED_LONG, sender_id, PassMessage, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          }
        });
        send_thread.join();
        recv_thread.join();
        MPI_Barrier(MPI_COMM_WORLD); // denseselect fill the bitmap array
        sync_time += get_time();
#ifdef PRINT_DEBUG_MESSAGES
        if (partition_id == 0)
        {
          printf("sync_time = %lf\n", sync_time);
        }
#endif
      }
#ifdef PRINT_DEBUG_MESSAGES
      if (partition_id == 0)
      {
        printf("dense mode\n");
      }
#endif
      int *send_queue = new int[partitions];
      int *recv_queue = new int[partitions];
      volatile int send_queue_size = 0;
      volatile int recv_queue_size = 0;
      std::mutex send_queue_mutex;
      std::mutex recv_queue_mutex;

      std::thread send_thread([&]() {
        for (int step = 0; step < partitions; step++)
        {
          if (step == partitions - 1)
          {
            break;
          }
          while (true)
          {
            send_queue_mutex.lock();
            bool condition = (send_queue_size <= step);
            send_queue_mutex.unlock();
            if (!condition)
              break;
            __asm volatile("pause" ::
                               : "memory");
          }
          int i = send_queue[step];
          for (int s_i = 0; s_i < sockets; s_i++)
          {
            MPI_Send(send_buffer[i][s_i]->data, sizeof(MsgUnit<M>) * send_buffer[i][s_i]->count, MPI_CHAR, i, PassMessage, MPI_COMM_WORLD);
          }
        }
      });
      std::thread recv_thread([&]() {
        std::vector<std::thread> threads;
        for (int step = 1; step < partitions; step++)
        {
          int i = (partition_id - step + partitions) % partitions;
          threads.emplace_back([&](int i) {
            for (int s_i = 0; s_i < sockets; s_i++)
            {
              MPI_Status recv_status;
              MPI_Probe(i, PassMessage, MPI_COMM_WORLD, &recv_status);
              MPI_Get_count(&recv_status, MPI_CHAR, &recv_buffer[i][s_i]->count);
              MPI_Recv(recv_buffer[i][s_i]->data, recv_buffer[i][s_i]->count, MPI_CHAR, i, PassMessage, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
              recv_buffer[i][s_i]->count /= sizeof(MsgUnit<M>);
            }
          },
                               i);
        }
        for (int step = 1; step < partitions; step++)
        {
          int i = (partition_id - step + partitions) % partitions;
          threads[step - 1].join();
          recv_queue[recv_queue_size] = i;
          recv_queue_mutex.lock();
          recv_queue_size += 1;
          recv_queue_mutex.unlock();
        }
        recv_queue[recv_queue_size] = partition_id;
        recv_queue_mutex.lock();
        recv_queue_size += 1;
        recv_queue_mutex.unlock();
      });
      current_send_part_id = partition_id;
      for (int step = 0; step < partitions; step++)
      {
        current_send_part_id = (current_send_part_id + 1) % partitions;
        int i = current_send_part_id;
        for (int t_i = 0; t_i < threads; t_i++)
        {
          *thread_state[t_i] = tuned_chunks_dense_backward[i][t_i];
        }
#pragma omp parallel
        {
          int thread_id = omp_get_thread_num();
          int s_i = get_socket_id(thread_id);
          VertexId final_p_v_i = thread_state[thread_id]->end;
          while (true)
          {
            VertexId begin_p_v_i = __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
            if (begin_p_v_i >= final_p_v_i)
              break;
            VertexId end_p_v_i = begin_p_v_i + basic_chunk;
            if (end_p_v_i > final_p_v_i)
            {
              end_p_v_i = final_p_v_i;
            }
            for (VertexId p_v_i = begin_p_v_i; p_v_i < end_p_v_i; p_v_i++)
            {
              VertexId v_i = compressed_incoming_adj_index_backward[s_i][p_v_i].vertex;
              //VertexId v_i = this->compressed_incoming_adj_index_backward[s_i][p_v_i].vertex;
              // if(v_i==0)
              //   dense_signal(v_i, VertexAdjList<EdgeData>(incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i].index, incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i+1].index));
              dense_signal(v_i, VertexAdjList<EdgeData>(incoming_adj_list_backward[s_i] + compressed_incoming_adj_index_backward[s_i][p_v_i].index, incoming_adj_list_backward[s_i] + compressed_incoming_adj_index_backward[s_i][p_v_i + 1].index));
            }
          }
          thread_state[thread_id]->status = STEALING;
          for (int t_offset = 1; t_offset < threads; t_offset++)
          {
            int t_i = (thread_id + t_offset) % threads;
            int s_i = get_socket_id(t_i);
            while (thread_state[t_i]->status != STEALING)
            {
              VertexId begin_p_v_i = __sync_fetch_and_add(&thread_state[t_i]->curr, basic_chunk);
              if (begin_p_v_i >= thread_state[t_i]->end)
                break;
              VertexId end_p_v_i = begin_p_v_i + basic_chunk;
              if (end_p_v_i > thread_state[t_i]->end)
              {
                end_p_v_i = thread_state[t_i]->end;
              }
              for (VertexId p_v_i = begin_p_v_i; p_v_i < end_p_v_i; p_v_i++)
              {
                VertexId v_i = compressed_incoming_adj_index_backward[s_i][p_v_i].vertex;
                // VertexId v_i = compressed_outgoing_adj_index[s_i][p_v_i].vertex;
                //  dense_signal(v_i, VertexAdjList<EdgeData>(incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i].index, incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i+1].index));
                dense_signal(v_i, VertexAdjList<EdgeData>(incoming_adj_list_backward[s_i] + compressed_incoming_adj_index_backward[s_i][p_v_i].index, incoming_adj_list_backward[s_i] + compressed_incoming_adj_index_backward[s_i][p_v_i + 1].index));
              }
            }
          }
        }
#pragma omp parallel for
        for (int t_i = 0; t_i < threads; t_i++)
        {
          flush_local_send_buffer<M>(t_i);
        }
        if (i != partition_id)
        {
          send_queue[send_queue_size] = i;
          send_queue_mutex.lock();
          send_queue_size += 1;
          send_queue_mutex.unlock();
        }
      }
      for (int step = 0; step < partitions; step++)
      {
        while (true)
        {
          recv_queue_mutex.lock();
          bool condition = (recv_queue_size <= step);
          recv_queue_mutex.unlock();
          if (!condition)
            break;
          __asm volatile("pause" ::
                             : "memory");
        }
        int i = recv_queue[step];
        MessageBuffer **used_buffer;
        if (i == partition_id)
        {
          used_buffer = send_buffer[i];
        }
        else
        {
          used_buffer = recv_buffer[i];
        }
        for (int t_i = 0; t_i < threads; t_i++)
        {
          int s_i = get_socket_id(t_i);
          int s_j = get_socket_offset(t_i);
          VertexId partition_size = used_buffer[s_i]->count;
          thread_state[t_i]->curr = partition_size / threads_per_socket / basic_chunk * basic_chunk * s_j;
          thread_state[t_i]->end = partition_size / threads_per_socket / basic_chunk * basic_chunk * (s_j + 1);
          if (s_j == threads_per_socket - 1)
          {
            thread_state[t_i]->end = used_buffer[s_i]->count;
          }
          thread_state[t_i]->status = WORKING;
        }
#pragma omp parallel reduction(+ \
                               : reducer)
        {
          R local_reducer = 0;
          int thread_id = omp_get_thread_num();
          int s_i = get_socket_id(thread_id);
          MsgUnit<M> *buffer = (MsgUnit<M> *)used_buffer[s_i]->data;
          while (true)
          {
            VertexId b_i = __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
            if (b_i >= thread_state[thread_id]->end)
              break;
            VertexId begin_b_i = b_i;
            VertexId end_b_i = b_i + basic_chunk;
            if (end_b_i > thread_state[thread_id]->end)
            {
              end_b_i = thread_state[thread_id]->end;
            }
            for (b_i = begin_b_i; b_i < end_b_i; b_i++)
            {
              VertexId v_i = buffer[b_i].vertex;
              M msg_data = buffer[b_i].msg_data;
              local_reducer += dense_slot(v_i, msg_data);
            }
          }
          thread_state[thread_id]->status = STEALING;
          reducer += local_reducer;
        }
      }
      send_thread.join();
      recv_thread.join();
      delete[] send_queue;
      delete[] recv_queue;
    }

    R global_reducer;
    MPI_Datatype dt = get_mpi_data_type<R>();
    MPI_Allreduce(&reducer, &global_reducer, 1, dt, MPI_SUM, MPI_COMM_WORLD);
    stream_time += MPI_Wtime();
#ifdef PRINT_DEBUG_MESSAGES
    if (partition_id == 0)
    {
      printf("process_edges took %lf (s)\n", stream_time);
    }
#endif
    return global_reducer;
  }

  void generate_backward_structure()
  {

    int fin = open(filename.c_str(), O_RDONLY);
    long bytes_to_read = lseek(fin, 0, SEEK_END);
    long read_bytes = 0;

    for (int i = 0; i < sockets; i++)
    {
      numa_free(outgoing_adj_index[i], sizeof(EdgeId) * (vertices + 1));
      outgoing_adj_bitmap[i]->~Bitmap();
      numa_free(outgoing_adj_list[i], sizeof(AdjUnit<EdgeData>) * outgoing_edges[i]);
      numa_free(compressed_outgoing_adj_index[i], sizeof(CompressedAdjIndexUnit) * compressed_outgoing_adj_vertices[i]);
    }
    free(outgoing_edges);
    free(compressed_outgoing_adj_vertices);

    int start = partition_offset[partition_id];
    int row_num = partition_offset[partition_id + 1] - start;

    //this->in_degree_backward=this->alloc_vertex_array<VertexId>();

    incoming_edges_backward = new EdgeId[sockets];
    incoming_adj_index_backward = new EdgeId *[sockets];
    incoming_adj_list_backward = new AdjUnit<EdgeData> *[sockets];
    incoming_adj_bitmap_backward = new Bitmap *[sockets];

    compressed_incoming_adj_index_backward = new CompressedAdjIndexUnit *[sockets];
    ;
    compressed_incoming_adj_vertices_backward = new VertexId[sockets];

    //VertexId** write_pos= new VertexId[sockets];
    for (int i = 0; i < sockets; i++)
    {
      incoming_edges_backward[i] = 0;
      incoming_adj_bitmap_backward[i] = new Bitmap(vertices);
      incoming_adj_bitmap_backward[i]->clear();
      incoming_adj_index_backward[i] = (EdgeId *)numa_alloc_onnode(sizeof(EdgeId) * (vertices + 1), i);
      memset(incoming_adj_index_backward[i], 0, sizeof(EdgeId) * (vertices + 1));
    }
    int local_edges = 0;
    //    printf("finish_1_%s\n",filename.c_str());
    EdgeUnit<EdgeData> *read_edge_buffer = new EdgeUnit<EdgeData>[CHUNKSIZE];
    //    printf("finish_2_%d\n",partition_id);
    assert(lseek(fin, 0, SEEK_SET) == 0);

    //    printf("%d   jello%d\n",read_edge_buffer[0].dst,local_partition_offset[1]);
    while (read_bytes < bytes_to_read)
    { // load the graph with a big chunk iteratively
      long curr_read_bytes;
      if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE)
      {
        curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
      }
      else
      {
        curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
      }
      assert(curr_read_bytes >= 0);
      //      printf("read_size %d\n",curr_read_bytes);
      read_bytes += curr_read_bytes;
      EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;

      for (EdgeId e_i = 0; e_i < curr_read_edges; e_i++)
      {
        VertexId src = read_edge_buffer[e_i].src;
        VertexId dst = read_edge_buffer[e_i].dst;
        int tmp = 0;

        for (int s_i = 0; s_i < sockets; s_i++)
        {
          if (local_partition_offset[s_i] <= dst && dst < local_partition_offset[s_i + 1])
          {
          if (!incoming_adj_bitmap_backward[s_i]->get_bit(src))
            {
                incoming_adj_bitmap_backward[s_i]->set_bit(src);
                // incoming_adj_index_backward[s_i][dst] = 0;
            }
            incoming_adj_index_backward[s_i][src] += 1;
            incoming_edges_backward[s_i]++;
          }
        }
      }
    }
    for (int s_i = 0; s_i < sockets; s_i++)
    {
      compressed_incoming_adj_vertices_backward[s_i] = 0;
      // if(partition_id==0){
      //  std::cout<<incoming_adj_index[s_i][0]<<" on "<<partition_id;
      // }
      incoming_adj_list_backward[s_i] = (AdjUnit<EdgeData> *)numa_alloc_onnode(unit_size * incoming_edges_backward[s_i], s_i);

      for (int i = 0; i < vertices; i++)
      {
        if (this->incoming_adj_bitmap_backward[s_i]->get_bit(i))
        {
          compressed_incoming_adj_vertices_backward[s_i] += 1;
        }
        incoming_adj_index_backward[s_i][i + 1] += incoming_adj_index_backward[s_i][i];
      }
      this->compressed_incoming_adj_index_backward[s_i] = (CompressedAdjIndexUnit *)
          numa_alloc_onnode(sizeof(CompressedAdjIndexUnit) * (compressed_incoming_adj_vertices_backward[s_i] + 1), s_i);

      compressed_incoming_adj_vertices_backward[s_i] = 0;
      EdgeId last_e_i = 0;

      for (int i = 0; i < vertices; i++)
      {
        if (this->incoming_adj_bitmap_backward[s_i]->get_bit(i))
        {
          last_e_i = incoming_adj_index_backward[s_i][i];
          compressed_incoming_adj_index_backward[s_i][compressed_incoming_adj_vertices_backward[s_i]].vertex = i;
          compressed_incoming_adj_vertices_backward[s_i] += 1;
          compressed_incoming_adj_index_backward[s_i][compressed_incoming_adj_vertices_backward[s_i]].index = last_e_i;
        }
      }

      for (VertexId p_v_i = 0; p_v_i < compressed_incoming_adj_vertices_backward[s_i]; p_v_i++)
      {
        VertexId v_i = compressed_incoming_adj_index_backward[s_i][p_v_i].vertex;
        incoming_adj_index_backward[s_i][v_i] = compressed_incoming_adj_index_backward[s_i][p_v_i].index;
        incoming_adj_index_backward[s_i][v_i + 1] = compressed_incoming_adj_index_backward[s_i][p_v_i + 1].index;
      }

      // std::cout<<incoming_adj_index[s_i][0]<<" off "<<partition_id;
    }
    read_bytes = 0;
    assert(lseek(fin, 0, SEEK_SET) == 0);

    while (read_bytes < bytes_to_read)
    { // load the graph with a big chunk iteratively
      long curr_read_bytes;
      if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE)
      {
        curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
      }
      else
      {
        curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
      }
      assert(curr_read_bytes >= 0);
      read_bytes += curr_read_bytes;
      EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
      //  #pragma omp parallel for
      for (EdgeId e_i = 0; e_i < curr_read_edges; e_i++)
      {
        VertexId src = read_edge_buffer[e_i].src;
        VertexId dst = read_edge_buffer[e_i].dst;
        int tmp = 0;

        for (int s_i = 0; s_i < sockets; s_i++)
        {
          if (local_partition_offset[s_i] <= dst && dst < local_partition_offset[s_i + 1])
          {
            EdgeId pos = __sync_fetch_and_add(&incoming_adj_index_backward[s_i][src], 1);
            incoming_adj_list_backward[s_i][pos].neighbour = dst;
            if (!std::is_same<EdgeData, Empty>::value)
            {
              incoming_adj_list_backward[s_i][pos].edge_data = read_edge_buffer[e_i].edge_data;
            }

            //    printf("%d %d\t",pos,dst);
          }
        }
      }
    }
    for (int s_i = 0; s_i < sockets; s_i++)
    {
      for (VertexId p_v_i = 0; p_v_i < this->compressed_incoming_adj_vertices_backward[s_i]; p_v_i++)
      {
        VertexId v_i = this->compressed_incoming_adj_index_backward[s_i][p_v_i].vertex;
        incoming_adj_index_backward[s_i][v_i] = compressed_incoming_adj_index_backward[s_i][p_v_i].index;
        incoming_adj_index_backward[s_i][v_i + 1] = compressed_incoming_adj_index_backward[s_i][p_v_i + 1].index;
      }
    }

    close(fin);
    tune_chunks_backward();
  }

  void generate_COO(VertexSubset *active)
  {
    //fill incoming_adj_index
    for (int i_s; i_s < sockets; i_s++)
    {
      for (VertexId vtx = 0; vtx < vertices; vtx++)
      {
        if (incoming_adj_index[i_s][vtx + 1] == 0)
          incoming_adj_index[i_s][vtx + 1] = incoming_adj_index[i_s][vtx];
        if (incoming_adj_index_backward[i_s][vtx + 1] == 0)
          incoming_adj_index_backward[i_s][vtx + 1] = incoming_adj_index_backward[i_s][vtx];
      }
    }
    for (VertexId vtx = 0; vtx < vertices; vtx++)
    {
      if (in_degree_for_backward[vtx] < 1)
        in_degree_for_backward[vtx] = 1; //local
      if (out_degree_for_backward[vtx] < 1)
        out_degree_for_backward[vtx] = 1; //local
    }

    _graph_cpu = new COOChunk();
    VertexId edge_size = 0; //(VertexId)incoming_adj_index[sockets-1][vertices];
    for (int i = 0; i < sockets; i++)
    {
      edge_size += (VertexId)incoming_adj_index[i][vertices];
    }
    _graph_cpu->dstList = new VertexId[edge_size];
    _graph_cpu->srcList = new VertexId[edge_size];
    _graph_cpu->numofedges = edge_size;

    int write_position = 0;
    for (int k = 0; k < sockets; k++)
      for (VertexId vtx = 0; vtx < vertices; vtx++)
      {
        for (int i = incoming_adj_index[k][vtx]; i < incoming_adj_index[k][vtx + 1]; i++)
        {
          _graph_cpu->dstList[write_position] = vtx;
          _graph_cpu->srcList[write_position++] = incoming_adj_list[k][i].neighbour;
        }
      }
    if (partition_id == 0)
      printf("GNNmini::Preprocessing[Generate Edges]\n");
  }
  void reorder_COO(int batch_size)
  { //replication
    graph_shard.clear();

    VertexId edge_size = (VertexId)incoming_adj_index[0][vertices];
    int blocks_v_num = batch_size;
    int src_blocks = owned_vertices / blocks_v_num + (int)((owned_vertices % blocks_v_num) > 0);
    int dst_blocks = vertices / blocks_v_num + (int)((vertices % blocks_v_num) > 0);
    for (int i = 0; i < src_blocks * dst_blocks; i++)
    {
      graph_shard.push_back(new COOChunk());
      graph_shard[i]->src_range[0] = i / dst_blocks * blocks_v_num + partition_offset[partition_id];
      graph_shard[i]->src_range[1] = std::min((1 + i / dst_blocks) * blocks_v_num, (int)owned_vertices) + partition_offset[partition_id];
      graph_shard[i]->dst_range[0] = i % dst_blocks * blocks_v_num;
      graph_shard[i]->dst_range[1] = std::min((1 + i % dst_blocks) * blocks_v_num, (int)vertices);
      //if(partition_id==1)std::cout<<"test in reorder "<<" "<<batch_size<<graph_shard[i]->src_range[0]<<" "<<graph_shard[i]->src_range[1]<<std::endl;
    }
    //std::cout<<this->partition_id<<" "<<batch_size<<" "<<src_blocks<<" "<<dst_blocks<<std::endl;
    // std::cout<<_graph_cpu->srcList[0]<<"????"<<partition_offset[partition_id+1]<<" "<<this->edges<<std::endl;

    for (int i = 0; i < edge_size; i++)
    {
      // if(_graph_cpu->src()[i]>=2208){
      //   std::cout<<"overflow id:"<<_graph_cpu->src()[i]<<" "<<i<<std::endl;
      // }
      int src_bucket = (_graph_cpu->srcList[i] - partition_offset[partition_id]) / blocks_v_num;
      int dst_bucket = (_graph_cpu->dstList[i]) / blocks_v_num;
      // if(partition_id==1)
      // std::cout<<_graph_cpu->srcList[i]<<" "<<partition_offset[partition_id]<<std::endl;
      graph_shard[src_bucket * dst_blocks + dst_bucket]->numofedges += 1;
    }
    for (int i = 0; i < src_blocks * dst_blocks; i++)
    {
      graph_shard[i]->src_delta = new VertexId[graph_shard[i]->numofedges];
      graph_shard[i]->dst_delta = new VertexId[graph_shard[i]->numofedges];
      //  std::cout<<graph_shard[i]->numofedges<<" ";
    }
    std::cout << std::endl;

    for (int i = 0; i < edge_size; i++)
    {
      int source = _graph_cpu->src()[i];
      int destination = _graph_cpu->dst()[i];
      int bucket_s = (source - partition_offset[partition_id]) / blocks_v_num;
      int bucket_d = (destination) / blocks_v_num;
      int offset = graph_shard[bucket_s * dst_blocks + bucket_d]->counter++;
      graph_shard[bucket_s * dst_blocks + bucket_d]->src_delta[offset] = source;
      graph_shard[bucket_s * dst_blocks + bucket_d]->dst_delta[offset] = destination;
    }
  }

  void reorder_COO_W2W()
  { //replication
    graph_shard.clear();

    VertexId edge_size = (VertexId)incoming_adj_index[0][vertices];
    int dst_blocks = partitions;
    for (int i = 0; i < dst_blocks; i++)
    {
      graph_shard.push_back(new COOChunk());
      graph_shard[i]->src_range[0] = partition_offset[partition_id];
      graph_shard[i]->src_range[1] = partition_offset[partition_id + 1];
      graph_shard[i]->dst_range[0] = partition_offset[i];
      graph_shard[i]->dst_range[1] = partition_offset[i + 1];
    }
    for (int i = 0; i < edge_size; i++)
    {
      int dst_bucket = this->get_partition_id(_graph_cpu->dstList[i]);
      graph_shard[dst_bucket]->numofedges += 1;
    }
    for (int i = 0; i < dst_blocks; i++)
    {
      graph_shard[i]->src_delta = new VertexId[graph_shard[i]->numofedges];
      graph_shard[i]->dst_delta = new VertexId[graph_shard[i]->numofedges];
      //  std::cout<<graph_shard[i]->numofedges<<" ";
    }
    //std::cout << std::endl;

    for (int i = 0; i < edge_size; i++)
    {
      int source = _graph_cpu->src()[i];
      int destination = _graph_cpu->dst()[i];
      int bucket_d = this->get_partition_id(_graph_cpu->dstList[i]);
      int offset = graph_shard[bucket_d]->counter++;
      graph_shard[bucket_d]->src_delta[offset] = source;
      graph_shard[bucket_d]->dst_delta[offset] = destination;
    }
    if (partition_id == 0)
      printf("GNNmini::Preprocessing[Reorganize Edges]\n");
  }

  void reorder_COO_W2partition(int batch_size)
  { //replication
    graph_shard.clear();

    VertexId edge_size = (VertexId)incoming_adj_index[0][vertices];

    int dst_blocks = vertices / (batch_size) + 1;
    //printf("here is the problem %d %d\n",batch_size, vertices);
    for (int i = 0; i < (dst_blocks); i++)
    {
      graph_shard.push_back(new COOChunk());
      graph_shard[i]->src_range[0] = partition_offset[partition_id];
      graph_shard[i]->src_range[1] = partition_offset[partition_id + 1];
      graph_shard[i]->dst_range[0] = i * batch_size;
      graph_shard[i]->dst_range[1] = std::min((i + 1) * batch_size, (int)(vertices));
      //      if(partition_id==0)
      //      printf("here is the problem %d %d\n",graph_shard[i]->dst_range[0], graph_shard[i]->dst_range[1]);
    }
    for (int i = 0; i < edge_size; i++)
    {
      int dst_bucket = _graph_cpu->dstList[i] / batch_size;
      graph_shard[dst_bucket]->numofedges += 1;
    }
    for (int i = 0; i < dst_blocks; i++)
    {
      graph_shard[i]->src_delta = new VertexId[graph_shard[i]->numofedges];
      graph_shard[i]->dst_delta = new VertexId[graph_shard[i]->numofedges];
      //  std::cout<<graph_shard[i]->numofedges<<" ";
    }
    std::cout << std::endl;

    for (int i = 0; i < edge_size; i++)
    {
      int source = _graph_cpu->src()[i];
      int destination = _graph_cpu->dst()[i];
      int bucket_d = _graph_cpu->dstList[i] / batch_size;
      int offset = graph_shard[bucket_d]->counter++;
      graph_shard[bucket_d]->src_delta[offset] = source;
      graph_shard[bucket_d]->dst_delta[offset] = destination;
    }
    //    printf("finish rep+++++++++++++++++++\n");
  }

  void read_whole_graph(VertexId *column_offset, VertexId *row_indices, int vertices, int edges, std::string path)
  {
    memset(column_offset, 0, sizeof(VertexId) * (vertices + 1));
    memset(row_indices, 0, sizeof(VertexId) * edges);
    VertexId *tmp_offset = new VertexId[vertices + 1];
    memset(tmp_offset, 0, sizeof(VertexId) * (vertices + 1));
    long total_bytes = file_size(path.c_str());
#ifdef PRINT_DEBUG_MESSAGES
    if (partition_id == 0)
    {
      printf("|V| = %u, |E| = %lu\n", vertices, edges);
    }
#endif
    int edge_unit_size = 8;
    EdgeId read_edges = edges;
    long bytes_to_read = edge_unit_size * read_edges;
    long read_offset = 0;
    long read_bytes;
    int fin = open(path.c_str(), O_RDONLY);
    EdgeUnit<Empty> *read_edge_buffer = new EdgeUnit<Empty>[CHUNKSIZE];

    assert(lseek(fin, read_offset, SEEK_SET) == read_offset);
    read_bytes = 0;
    while (read_bytes < bytes_to_read)
    {
      long curr_read_bytes;
      if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE)
      {
        curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
      }
      else
      {
        curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
      }
      assert(curr_read_bytes >= 0);
      read_bytes += curr_read_bytes;
      EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
      // #pragma omp parallel for
      for (EdgeId e_i = 0; e_i < curr_read_edges; e_i++)
      {
        VertexId src = read_edge_buffer[e_i].src;
        VertexId dst = read_edge_buffer[e_i].dst;
        tmp_offset[dst + 1]++;
      }
    }
    for (int i = 0; i < vertices; i++)
    {
      tmp_offset[i + 1] += tmp_offset[i];
    }

    memcpy(column_offset, tmp_offset, sizeof(VertexId) * (vertices + 1));
    //printf("%d\n", column_offset[vertices]);
    assert(lseek(fin, read_offset, SEEK_SET) == read_offset);
    read_bytes = 0;
    while (read_bytes < bytes_to_read)
    {
      long curr_read_bytes;
      if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE)
      {
        curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
      }
      else
      {
        curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
      }
      assert(curr_read_bytes >= 0);
      read_bytes += curr_read_bytes;
      EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
      // #pragma omp parallel for
      for (EdgeId e_i = 0; e_i < curr_read_edges; e_i++)
      {
        VertexId src = read_edge_buffer[e_i].src;
        VertexId dst = read_edge_buffer[e_i].dst;
        //        if(dst==875710)
        //            printf("%d",read_edge_buffer[e_i].src);
        row_indices[tmp_offset[dst]++] = src;
      }
    }
  }

  void load_replicate3(std::vector<int> layer_size)
  { 
    // if (partition_id == 0)
    //   printf("replication3\n");
    MPI_Datatype vid_t = get_mpi_data_type<int>();
    double start_time = 0;
    start_time -= get_time();
    int *indegree = new int[vertices];
    memset(indegree, 0, sizeof(int));
    VertexSubset *active = alloc_vertex_subset();
    active->fill();
    std::string path = filename;
    HasRepVtx.clear();
    RepVtx.clear();
    EdgeRemote2Local.clear();
    EdgeRemote2Remote.clear();
    Bitmap *tmpRepVtx = new Bitmap(vertices);
    Bitmap *preRepVtx = new Bitmap(vertices);
    outGoing = new Bitmap(vertices);
    tmpRepVtx->clear();
    preRepVtx->clear();
    std::vector<std::vector<VertexId>> EdgeTmpRemote2Local;
    std::vector<std::vector<VertexId>> EdgeTmpRemote2Remote;
    int beta = 3;

    for (int i = 0; i < 2; i++)
    {
      HasRepVtx.push_back(new Bitmap(vertices));
      RepVtx.push_back(new Bitmap(vertices));
      HasRepVtx[i]->clear();
      RepVtx[i]->clear();
      std::vector<VertexId> tmp;
      tmp.clear();
      EdgeRemote2Local.push_back(tmp);
      EdgeRemote2Remote.push_back(tmp);
      EdgeTmpRemote2Local.push_back(tmp);
      EdgeTmpRemote2Remote.push_back(tmp);
    }
    int count_edge = 0;
    VertexId *column_offset = new VertexId[vertices + 1];
    VertexId *row_indices = new VertexId[edges + 1];
    read_whole_graph(column_offset, row_indices, vertices, edges, path);
    //compute for first layers
    //printf("dedededd %d\n", column_offset[vertices]);
    process_edges<int, int>(                                   // For EACH Vertex Processing
        [&](VertexId dst, VertexAdjList<Empty> incoming_adj) { //pull
          //indegree[dst]=incoming_adj.end-incoming_adj.begin;

          int red = 0;
          for (AdjUnit<Empty> *ptr = incoming_adj.begin; ptr != incoming_adj.end; ptr++)
          { //pull model
            red++;
          }
          if (red > 0 && (dst < partition_offset[partition_id] || dst >= partition_offset[partition_id + 1]))
            outGoing->set_bit(dst);
          //HasRepVtx[1]->set_bit(dst);
          indegree[dst] = red;
          if (indegree[dst] <= replication_threshold && indegree[dst] > 0)
          {
            //cached_0_vertices->set_bit(dst);
            if (dst < partition_offset[partition_id] || dst >= partition_offset[partition_id + 1])
            {
              preRepVtx->set_bit(dst);
              emit(dst, 1);
            }
          }
        },
        [&](VertexId dst, int msg) {
          //tmpRepVtx->set_bit(dst);
          return 0;
        },
        active);

    int pre_reduce_com = 0;
    for (int i = 0; i < vertices; i++)
    {
      if (preRepVtx->get_bit(i))
        pre_reduce_com++;
    }
    int minnodes = 0;

    MPI_Allreduce(&pre_reduce_com, &minnodes, 1, vid_t, MPI_MIN, MPI_COMM_WORLD);
    minnodes = (int)(1.15 * minnodes);

    if (replication_threshold <= 1024)
    {
      int limit = 0;
      int *limitp = new int[partitions];
      memset(limitp, 0, sizeof(int) * partitions);
      for (int i = 0; i < partitions; i++)
      {
        limitp[i] = partition_offset[i];
      }
      while (limit < minnodes)
      { //minnodes
        for (int i = 0; i < partitions; i++)
        {
          for (int j = limitp[i]; j < partition_offset[i + 1]; j++)
          {
            limitp[i]++;
            if (preRepVtx->get_bit(j) && limit < minnodes)
            {
              RepVtx[0]->set_bit(j);
              limit++;
              break;
            }
          }
        }
        bool count = 0;
        for (int i = 0; i < partitions; i++)
        {
          count += partition_offset[i + 1] - limitp[i];
        }
        if (count == 0)
          break;
      }
    }
    else
    {
      for (int i = 0; i < vertices; i++)
      {
        if (preRepVtx->get_bit(i))
        {
          RepVtx[0]->set_bit(i);
        }
      }
    }

    process_edges<int, int>(                                   // For EACH Vertex Processing
        [&](VertexId dst, VertexAdjList<Empty> incoming_adj) { //pull
          if (RepVtx[0]->get_bit(dst))
          {
            int second_layer_count = 0;
            int current_count = 0;
            for (AdjUnit<Empty> *ptr = incoming_adj.begin; ptr != incoming_adj.end; ptr++)
            { //pull model
              VertexId src = ptr->neighbour;
              current_count++;
            }
            if (partition_id != get_partition_id(dst))
              emit(dst, current_count);
            else
              emit(dst, 0);
          }
        },
        [&](VertexId dst, int tmp) {
          tmpRepVtx->set_bit(dst);
          write_add<int>(&count_edge, tmp);
          return 0;
        },
        active);
    /* communicate and sync the communication bitmap*/
    std::vector<Bitmap *> tag;
    tag.clear();
    for (int i = 0; i < partitions; i++)
    {
      if (i != partition_id)
      {
        tag.push_back(new Bitmap(vertices));
        tag[i]->clear();
      }
      else
      {
        tag.push_back(RepVtx[0]);
      }
    }
    std::vector<Bitmap *> tag2;
    tag2.clear();
    for (int i = 0; i < partitions; i++)
    {
      if (i != partition_id)
      {
        tag2.push_back(new Bitmap(vertices));
        tag2[i]->clear();
      }
      else
      {
        tag2.push_back(RepVtx[1]);
      }
    }

    MPI_Datatype l_vid_t = get_mpi_data_type<unsigned long>();
    for (int i = 0; i < partitions; i++)
    {
      MPI_Allreduce(MPI_IN_PLACE, tag[i]->data, (WORD_OFFSET(vertices) + 1), l_vid_t, MPI_SUM, MPI_COMM_WORLD);
    }
    /* end communicate and sync the communication bitmap*/
    graph_rep = new graph_replication[3];
    graph_rep[0].src_rep_vec.clear();
    graph_rep[0].dst_rep_vec.clear();
    graph_rep[0].rep_edge_size = 0;
    graph_rep[0].rep_node_size = 0;
    graph_rep[0].feature_size = layer_size[0];
    graph_rep[1].rep_edge_size = 0;
    graph_rep[1].rep_node_size = 0;
    graph_rep[1].src_rep_vec.clear();
    graph_rep[1].dst_rep_vec.clear();
    graph_rep[1].feature_size = layer_size[1];

    VertexId count_edge2 = 0;
    VertexId count_edge3 = 0;
    Bitmap *intermediate = new Bitmap(vertices);
    intermediate->clear();
    std::vector<std::vector<int>> data(partitions, std::vector<int>(0));
    VertexId tmp_count_edge2 = 0;
    VertexId tmp_count_edge3 = 0;

    for (int i = partition_offset[partition_id]; i < partition_offset[partition_id + 1]; i++)
    {
      for (int j = column_offset[i]; j < column_offset[i + 1]; j++)
      {
        int src = row_indices[j];
        int worker_id = get_partition_id(src);
        if ((worker_id != partition_id) && tag[worker_id]->get_bit(i))
          data[worker_id].push_back(src);
      }
      for (int k = 0; k < partitions; k++)
      {
        if (data[k].size() <= replication_threshold)
        {
          tmp_count_edge2 = 0;
          for (int l = 0; l < data[k].size(); l++)
          {
            tmp_count_edge2 += column_offset[data[k][l] + 1] - column_offset[data[k][l]];
          }
          //float cost = (float)(tmp_count_edge2)*2; //layer_size[0] / layer_size[1];
          //if (cost <= replication_threshold && cost > 0)
          if ((tmp_count_edge2 * 2) <= replication_threshold && (tmp_count_edge2 * 2) > 0)
          {
            tag2[k]->set_bit(i);
            for (int l = 0; l < data[k].size(); l++)
            {
              intermediate->set_bit(data[k][l]);
            }
            count_edge3 += data[k].size();
            for (int l = 0; l < data[k].size(); l++)
            {
              graph_rep[1].src_rep_vec.push_back(data[k][l]);
              graph_rep[1].dst_rep_vec.push_back(i);
            }
          }
        }
        data[k].clear();
      }
    }

    for (int i = 0; i < vertices; i++)
    {
      if (intermediate->get_bit(i))
      {
        int tmp_count_edge2 = 0;
        for (int i_i = column_offset[i]; i_i < column_offset[i + 1]; i_i++)
        {
          VertexId src_inter = row_indices[i_i];
          if (partition_offset[partition_id] > src_inter || partition_offset[partition_id + 1] <= src_inter)
          {
            tmp_count_edge2++;
            HasRepVtx[1]->set_bit(src_inter);
          }
        }
        count_edge2 += tmp_count_edge2;
      }
    }

    for (int i = 0; i < partitions; i++)
    {
      //printf("comm tag%d\n",i);
      MPI_Allreduce(MPI_IN_PLACE, tag2[i]->data, (WORD_OFFSET(vertices) + 1), l_vid_t, MPI_SUM, MPI_COMM_WORLD);
      //  if(i!=partition_id){
      //   tag2[i]->~Bitmap();
      //   }
    }

    graph_rep[0].rep_edge_size = count_edge + count_edge2;
    //printf("DEBUG rep edges1: %d rep edges2: %d rep edges3: %d  \n", count_edge, count_edge2, count_edge3);
    graph_rep[0].dst_rep = (VertexId *)cudaMallocPinned(sizeof(VertexId) * graph_rep[0].rep_edge_size + 1);
    graph_rep[0].src_rep = (VertexId *)cudaMallocPinned(sizeof(VertexId) * graph_rep[0].rep_edge_size + 1);
    graph_rep[0].weight_rep = (float *)cudaMallocPinned(sizeof(float) * graph_rep[0].rep_edge_size + 1);
    graph_rep[0].dst_rep_gpu = (VertexId *)getDevicePointer((void *)graph_rep[0].dst_rep);
    graph_rep[0].src_rep_gpu = (VertexId *)getDevicePointer((void *)graph_rep[0].src_rep);
    graph_rep[0].weight_rep_gpu = (float *)getDevicePointer((void *)graph_rep[0].weight_rep);

    graph_rep[1].rep_edge_size = count_edge3;
    graph_rep[1].dst_rep = (VertexId *)cudaMallocPinned(sizeof(VertexId) * graph_rep[1].rep_edge_size + 1);
    graph_rep[1].src_rep = (VertexId *)cudaMallocPinned(sizeof(VertexId) * graph_rep[1].rep_edge_size + 1);
    graph_rep[1].weight_rep = (float *)cudaMallocPinned(sizeof(float) * graph_rep[1].rep_edge_size + 1);
    graph_rep[1].dst_rep_gpu = (VertexId *)getDevicePointer((void *)graph_rep[1].dst_rep);
    graph_rep[1].src_rep_gpu = (VertexId *)getDevicePointer((void *)graph_rep[1].src_rep);
    graph_rep[1].weight_rep_gpu = (float *)getDevicePointer((void *)graph_rep[1].weight_rep);
    memcpy(graph_rep[1].dst_rep, graph_rep[1].dst_rep_vec.data(), graph_rep[1].dst_rep_vec.size() * sizeof(VertexId));
    memcpy(graph_rep[1].src_rep, graph_rep[1].src_rep_vec.data(), graph_rep[1].src_rep_vec.size() * sizeof(VertexId));
    int start_index = 0;
    int start_index2 = 0;
    Bitmap *tmp1 = new Bitmap(vertices); // for out_buffer initialize
    Bitmap *tmp2 = new Bitmap(vertices); // for out_buffer initialize
    Bitmap *tmp3 = new Bitmap(vertices);
    tmp1->clear();

    for (int i = partition_offset[partition_id]; i < partition_offset[partition_id + 1]; i++)
    {
      for (int j = column_offset[i]; j < column_offset[i + 1]; j++)
      {
        int src = row_indices[j];
        int worker_id = get_partition_id(src);
        if ((worker_id != partition_id) && tag[worker_id]->get_bit(i))
          data[worker_id].push_back(src);
      }
      for (int k = 0; k < partitions; k++)
      {
        if (data[k].size() <= replication_threshold)
        {
          tmp1->set_bit(i);
          for (int l = 0; l < data[k].size(); l++)
          {
            HasRepVtx[0]->set_bit(data[k][l]);
            int indgr1 = column_offset[i + 1] - column_offset[i];
            int indgr2 = column_offset[data[k][l] + 1] - column_offset[data[k][l]];

            graph_rep[0].dst_rep[start_index] = i;
            graph_rep[0].src_rep[start_index] = data[k][l];
            graph_rep[0].weight_rep[start_index] = std::sqrt(indgr1) * std::sqrt(indgr2);
            start_index++;
          }
        }
        data[k].clear();
      }
    }
    start_index2 = start_index;
    int errorno1 = 0;
    for (int i = 0; i < start_index; i++)
    {
      if ((get_partition_id(graph_rep[0].dst_rep[i]) != partition_id) ||
          (get_partition_id(graph_rep[0].src_rep[i])) == partition_id)
        errorno1++;
    }

    for (int i = 0; i < vertices; i++)
    {
      if (intermediate->get_bit(i))
      {
        VertexId dst_inter = i;
        for (int i_i = column_offset[dst_inter]; i_i < column_offset[dst_inter + 1]; i_i++)
        {
          VertexId src_inter = row_indices[i_i];
          if (partition_offset[partition_id] > src_inter || partition_offset[partition_id + 1] <= src_inter)
          {
            graph_rep[0].dst_rep[start_index] = dst_inter;
            graph_rep[0].src_rep[start_index] = src_inter;
            int indgr1 = column_offset[dst_inter + 1] - column_offset[dst_inter];
            int indgr2 = column_offset[src_inter + 1] - column_offset[src_inter];
            graph_rep[0].weight_rep[start_index] = std::sqrt(indgr1) * std::sqrt(indgr2);
            start_index++;
            HasRepVtx[1]->set_bit(src_inter);
          }
        }
      }
    }
    int errorno2 = 0;
    for (int i = start_index2; i < start_index; i++)
    {
      if ((get_partition_id(graph_rep[0].dst_rep[i]) == partition_id) ||
          (get_partition_id(graph_rep[0].src_rep[i])) == partition_id)
        errorno2++;
    }

    for (int i = 0; i < vertices; i++)
    {
      if (HasRepVtx[0]->get_bit(i) || HasRepVtx[1]->get_bit(i))
        graph_rep[0].rep_node_size++;
      // if (HasRepVtx[0]->get_bit(i))
      //   graph_rep[1].rep_node_size++;
    }

    for (int i = 0; i < graph_rep[0].rep_edge_size; i++)
    {
      int edge_dst = graph_rep[0].dst_rep[i];
      tmp1->set_bit(edge_dst);
    }
    for (int i = 0; i < graph_rep[1].rep_edge_size; i++)
    {
      int edge_dst = graph_rep[1].dst_rep[i];
      tmp2->set_bit(edge_dst);
      int edge_src = graph_rep[1].src_rep[i];
      tmp3->set_bit(edge_src);
    }

    graph_rep[0].dst_map = (VertexId *)cudaMallocPinned(sizeof(VertexId) * vertices);
    graph_rep[1].dst_map = (VertexId *)cudaMallocPinned(sizeof(VertexId) * vertices);
    memset(graph_rep[0].dst_map, 0, (sizeof(VertexId) * vertices));
    memset(graph_rep[1].dst_map, 0, (sizeof(VertexId) * vertices));
    graph_rep[0].output_size = 0;
    graph_rep[1].output_size = 0;
    for (int i = 0; i < vertices; i++)
    {
      if (tmp1->get_bit(i))
      {

        graph_rep[0].dst_map[i] = graph_rep[1].output_size;
        graph_rep[0].output_size++;
      }
      if (tmp2->get_bit(i))
      {
        graph_rep[1].dst_map[i] = graph_rep[1].output_size;
        graph_rep[1].output_size++;
      }
    }
    graph_rep[0].output_buffer_gpu = (float *)cudaMallocGPU(((long)sizeof(float)) * graph_rep[0].output_size * graph_rep[0].feature_size + 1);
    graph_rep[1].output_buffer_gpu = (float *)cudaMallocGPU(((long)sizeof(float)) * graph_rep[1].output_size * graph_rep[1].feature_size + 1);
    tmp1->~Bitmap();
    tmp2->~Bitmap();
    graph_rep[0].src_map = (VertexId *)cudaMallocPinned(sizeof(VertexId) * vertices);

    graph_rep[1].src_map = (VertexId *)cudaMallocPinned(sizeof(VertexId) * vertices);

    memset(graph_rep[0].src_map, 0, sizeof(VertexId) * vertices);
    memset(graph_rep[1].src_map, 0, sizeof(VertexId) * vertices);

    int tmp_index = 0;
    int tmp_index2 = 0;
    int all_comm_v = 0;

    for (int i = 0; i < vertices; i++)
    {
      if (HasRepVtx[0]->get_bit(i) || HasRepVtx[1]->get_bit(i))
      {
        graph_rep[0].src_map[i] = tmp_index;
        tmp_index++;
      }
      if (tmp3->get_bit(i))
      {
        graph_rep[1].src_map[i] = tmp_index2;
        tmp_index2++;
      }
      if (outGoing->get_bit(i))
      {
        all_comm_v++;
      }
    }
    tmp3->~Bitmap();
    graph_rep[1].rep_node_size = tmp_index2;
    graph_rep[1].rep_feature = (float *)cudaMallocPinned(sizeof(float) * graph_rep[1].feature_size * graph_rep[1].rep_node_size + 1);
    graph_rep[0].rep_feature = (float *)cudaMallocPinned(sizeof(float) * graph_rep[0].feature_size * graph_rep[0].rep_node_size + 1);

    int errorno = 0;
    //printf("%d %d %d\n", partition_id, graph_rep[1].src_rep[2], graph_rep[1].dst_rep[2]);
    for (int i = 0; i < graph_rep[1].rep_edge_size; i++)
    {
      if ((get_partition_id(graph_rep[1].dst_rep_vec[i]) != partition_id) ||
          (get_partition_id(graph_rep[1].src_rep_vec[i])) == partition_id)
        errorno++;
    }

    //    printf("DEBUG Info 1:(%d %d) 2:(%d %d) 3:(%d %d) 4:(%d %d) 5:(%d %d) 6:(%d %d %d)\n",
    //           /*1*/ start_index, graph_rep[0].rep_edge_size,
    //           /*2*/ count_edge3, graph_rep[1].src_rep_vec.size(),
    //           /*3*/ tmp_index, graph_rep[0].rep_node_size,
    //           /*4*/ tmp_index2, graph_rep[1].rep_node_size,
    //           /*5*/ graph_rep[0].output_size, graph_rep[1].output_size,
    //           /*6*/ errorno, errorno1, errorno2);

    int reduce_comm[2] = {0, 0};
    for (int i = 0; i < vertices; i++)
    {
      if (RepVtx[0]->get_bit(i))
        reduce_comm[0] += 1;
      if (RepVtx[1]->get_bit(i))
        reduce_comm[1] += 1;
    }

    int global_comm = 0;
    int global_comm2 = 0;
    int global_repedge = 0;
    int global_repnode = 0;
    int global_repedge2 = 0;
    int global_repnode2 = 0;
    int global_all_comm = 0;
    MPI_Allreduce(&reduce_comm[0], &global_comm, 1, vid_t, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&reduce_comm[1], &global_comm2, 1, vid_t, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&graph_rep[0].rep_edge_size, &global_repedge, 1, vid_t, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&graph_rep[0].rep_node_size, &global_repnode, 1, vid_t, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&graph_rep[1].rep_edge_size, &global_repedge2, 1, vid_t, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&graph_rep[1].rep_node_size, &global_repnode2, 1, vid_t, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&all_comm_v, &global_all_comm, 1, vid_t, MPI_SUM, MPI_COMM_WORLD);
    start_time += get_time();
    int local_s = partition_offset[partition_id];
    int local_d = partition_offset[partition_id + 1];
    gnnctx->l_e_num = column_offset[local_d] - column_offset[local_s];
    gnnctx->l_v_num = owned_vertices;
    /*   printf("%d{|V|:%d |E|:%d}:{Layer1 rep edge:%d  noeds: %d comm %d} {Layer2 rep edge:%d  noeds: %d comm %d} {%d %d %d}{%d %d %d} {g_comm: %d local_comm: %d} ||in %f(s)\n",
           replication_threshold, owned_vertices, column_offset[local_d] - column_offset[local_s],
           global_repedge / partitions, global_repnode / partitions, global_comm / partitions,
           global_repedge2 / partitions, global_repnode2 / partitions, global_comm2 / partitions,
           graph_rep[0].rep_edge_size, graph_rep[0].rep_node_size, reduce_comm[0],
           graph_rep[1].rep_edge_size, graph_rep[1].rep_node_size, reduce_comm[1],
           global_all_comm / partitions,
           all_comm_v, start_time);
   */
    delete[] column_offset;
    delete[] row_indices;
    if (partition_id == 0)
      printf("GNNmini::Preprocessing[Finish Loading Replication]\n");
  }
  void print_info()
  {
    MPI_Datatype vid_t = get_mpi_data_type<int>();
    int *local_e_num = NULL;
    int *local_v_num = NULL;
    int *reduce_comm0 = NULL;
    int *reduce_comm1 = NULL;
    int *rep_edge_0 = NULL;
    int *rep_edge_1 = NULL;
    int *rep_node_0 = NULL;
    int *rep_node_1 = NULL;
    int *out_node = NULL;
    local_e_num = new int[partitions];
    local_v_num = new int[partitions];
    reduce_comm0 = new int[partitions];
    reduce_comm1 = new int[partitions];
    rep_edge_0 = new int[partitions];
    rep_edge_1 = new int[partitions];
    rep_node_0 = new int[partitions];
    rep_node_1 = new int[partitions];
    out_node = new int[partitions];
    memset(local_e_num, 0, sizeof(int) * partitions);
    memset(local_v_num, 0, sizeof(int) * partitions);
    memset(reduce_comm0, 0, sizeof(int) * partitions);
    memset(reduce_comm1, 0, sizeof(int) * partitions);
    memset(rep_edge_0, 0, sizeof(int) * partitions);
    memset(rep_edge_1, 0, sizeof(int) * partitions);
    memset(rep_node_0, 0, sizeof(int) * partitions);
    memset(rep_node_1, 0, sizeof(int) * partitions);
    memset(out_node, 0, sizeof(int) * partitions);
    local_e_num[partition_id] = gnnctx->l_e_num;
    local_v_num[partition_id] = gnnctx->l_v_num;
    for (int i = 0; i < vertices; i++)
    {
      if (RepVtx[0]->get_bit(i))
        reduce_comm0[partition_id] += 1;
      if (RepVtx[1]->get_bit(i))
        reduce_comm1[partition_id] += 1;
      if (outGoing->get_bit(i))
        out_node[partition_id]++;
    }
    rep_edge_0[partition_id] = graph_rep[0].rep_edge_size;
    rep_edge_1[partition_id] = graph_rep[1].rep_edge_size;
    rep_node_0[partition_id] = graph_rep[0].rep_node_size;
    rep_node_1[partition_id] = graph_rep[1].rep_node_size;

    MPI_Allreduce(MPI_IN_PLACE, local_e_num, partitions, vid_t, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, local_v_num, partitions, vid_t, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, reduce_comm0, partitions, vid_t, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, reduce_comm1, partitions, vid_t, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, rep_edge_0, partitions, vid_t, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, rep_edge_1, partitions, vid_t, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, rep_node_0, partitions, vid_t, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, rep_node_1, partitions, vid_t, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, out_node, partitions, vid_t, MPI_SUM, MPI_COMM_WORLD);

    if (partition_id == 0)
    {
      int red_comm0 = 0;
      int red_comm1 = 0;
      int rep_v_0 = 0;
      int rep_v_1 = 0;
      int rep_e_0 = 0;
      int rep_e_1 = 0;
      int avg_outgoing = 0;
      int avg_l_v = 0;
      int avg_l_e = 0;

      printf("GNNmini::Preprocessing[Print All Info:]\n");
      printf("GNNmini::Vertices:[%d],Edges:[%d], RepThreshold:[%d], Layers: ", vertices, edges, config->repthreshold);
      std::cout << config->layer_string << std::endl;
      printf("GNNmini::ProcessLocal\t[%d]\n", config->process_local);
      printf("GNNmini::ProcessOverlap\t[%d]\n", config->overlap);
      printf("GNNmini::{assigned Vertices}:\t");
      for (int i = 0; i < partitions; i++)
      {
        printf("(%d)[%d]\t", i, local_v_num[i]);
        avg_l_v += local_v_num[i];
      }
      printf("(avg)[%d]\t", avg_l_v / partitions);

      printf("\nGNNmini::{assigned Edges}:\t");
      for (int i = 0; i < partitions; i++)
      {
        printf("(%d)[%d]\t", i, local_e_num[i]);
        avg_l_e += local_e_num[i];
      }
      printf("(avg)[%d]\t", avg_l_e / partitions);

      printf("\nGNNmini::{Outgoing Msg}:\t");
      for (int i = 0; i < partitions; i++)
      {
        printf("(%d)[%d]\t", i, out_node[i]);
        avg_outgoing += out_node[i];
      }
      printf("(avg)[%d]\t", avg_outgoing / partitions);

      printf("\nGNNmini::{Reduce Comm{0}}:\t");
      for (int i = 0; i < partitions; i++)
      {
        printf("(%d)[%d]\t", i, reduce_comm0[i]);
        red_comm0 += reduce_comm0[i];
      }
      printf("(avg)[%d]\t", red_comm0 / partitions);

      printf("\nGNNmini::{Reduce Comm{1}}:\t");
      for (int i = 0; i < partitions; i++)
      {
        printf("(%d)[%d]\t", i, reduce_comm1[i]);
        red_comm1 += reduce_comm1[i];
      }
      printf("(avg)[%d]\t", red_comm1 / partitions);

      printf("\nGNNmini::{Rep Vertices{0}}:\t");
      for (int i = 0; i < partitions; i++)
      {
        printf("(%d)[%d]\t", i, rep_node_0[i]);
        rep_v_0 += rep_node_0[i];
      }
      printf("(avg)[%d]\t", rep_v_0 / partitions);

      printf("\nGNNmini::{Rep Edges{0}}:\t");
      for (int i = 0; i < partitions; i++)
      {
        printf("(%d)[%d]\t", i, rep_edge_0[i]);
        rep_e_0 += rep_edge_0[i];
      }
      printf("(avg)[%d]\t", rep_e_0 / partitions);

      printf("\nGNNmini::{Rep Vertices{1}}:\t");
      for (int i = 0; i < partitions; i++)
      {
        printf("(%d)[%d]\t", i, rep_node_1[i]);
        rep_v_1 += rep_node_1[i];
      }
      printf("(avg)[%d]\t", rep_v_1 / partitions);

      printf("\nGNNmini::{Rep Edges{1}}:\t");
      for (int i = 0; i < partitions; i++)
      {
        printf("(%d)[%d]\t", i, rep_edge_1[i]);
        rep_e_1 += rep_edge_1[i];
      }
      printf("(avg)[%d]\t", rep_e_1 / partitions);

      printf("\nGNNmini::Preprocessing[Finish Print All Info:]\n");
    }
  }
  void initialize_rep_feature()
  {
    for (int i = 0; i < graph_rep[0].rep_node_size; i++)
    {
      for (int j = 0; j < graph_rep[0].feature_size; j++)
      {
        graph_rep[0].rep_feature[i * graph_rep[0].feature_size + j] = 1.0;
      }
    }
    for (int i = 0; i < graph_rep[0].rep_edge_size; i++)
    {
      graph_rep[0].weight_rep[i] = 1;
    }
  }
};

#endif

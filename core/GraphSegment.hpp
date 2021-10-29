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

#ifndef GRAPHSEGMENT_HPP
#define GRAPHSEGMENT_HPP
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
#include "core/time.hpp"
#include "core/type.hpp"
#include "cuda/test.hpp"
const bool NOT_SUPPORT_DEVICE_TYPE=false;
//
//typedef struct graph_Tensor_Segment
//{
//  torch::Tensor column_offset; //VertexNumber
//  torch::Tensor row_indices;   //edge_size
//  torch::Tensor edge_weight;   //edge_size
//  int edge_size;
//  int batch_size;
//  int input_size;
//  int output_size;
//  int feature_size;
//  int src_range[2];
//  int dst_range[2];
//  float *weight_buffer;
//} CSC_segment;
//
//typedef struct graph_Tensor_Segment1
//{
//  torch::Tensor row_offset; //VertexNumber
//  torch::Tensor column_indices;   //edge_size
//  torch::Tensor edge_weight;   //edge_size
//  int edge_size;
//  int batch_size;
//  int input_size;
//  int output_size;
//  int feature_size;
//  int src_range[2];
//  int dst_range[2];
//  float *weight_buffer;
//} CSR_segment;
typedef struct graph_Tensor_Segment_pinned
{
    
    
  VertexId *column_offset;     //VertexNumber
  VertexId *row_indices;       //edge_size also the source nodes
  VertexId *row_offset;     //VertexNumber
  VertexId *column_indices;  
  
  long *source;
  long *destination;
  long *source_backward;
  
  float *edge_weight_forward;          //edge_size
  float *edge_weight_backward;
  
  VertexId *backward_message_index;
  VertexId *forward_message_index;
  
  VertexId *column_offset_gpu; //VertexNumber
  VertexId *row_indices_gpu;
  VertexId *row_offset_gpu; //VertexNumber
  VertexId *column_indices_gpu;   //edge_size
  
  long *source_gpu;
  long *destination_gpu;
  
  long *source_backward_gpu;
  
  float *edge_weight_forward_gpu;      //edge_size
  float *edge_weight_backward_gpu;      //edge_size
  
  int edge_size;
  int batch_size_forward;
  int batch_size_backward;
  int input_size;
  int output_size;
  int feature_size;
  int src_range[2];
  int dst_range[2];
  Bitmap* source_active;
  Bitmap* destination_active;
  Bitmap* destination_mirror_active;
  
  DeviceLocation dt;
  
  void init(VertexId src_start,VertexId src_end,VertexId dst_start,VertexId dst_end,VertexId edge_size_,DeviceLocation dt_){
      src_range[0]=src_start;
      src_range[1]=src_end;
      dst_range[0]=dst_start;
      dst_range[1]=dst_end;
      batch_size_backward=src_range[1]-src_range[0];
      batch_size_forward=dst_range[1]-dst_range[0];
      edge_size=edge_size_;
      dt=dt_;
      
  }
  void allocVertexAssociateData(){
      
    source_active=new Bitmap(batch_size_backward);
    destination_active=new Bitmap(batch_size_forward);
    destination_mirror_active=new Bitmap(batch_size_forward);
            
    source_active->clear();
    destination_active->clear();
    destination_mirror_active->clear();
    if(dt==GPU_T){  
        column_offset = (VertexId *)cudaMallocPinned((batch_size_forward+1) * sizeof(VertexId));           
        row_offset = (VertexId *)cudaMallocPinned((batch_size_backward+1) * sizeof(VertexId));///  
    }else
    if(dt==CPU_T){
        column_offset = (VertexId *)malloc((batch_size_forward+1) * sizeof(VertexId));           
        row_offset = (VertexId *)malloc((batch_size_backward+1) * sizeof(VertexId));/// 
    }else{
        assert(NOT_SUPPORT_DEVICE_TYPE);
    } 
  }
    void allocEdgeAssociateData(){
     
    if(dt==GPU_T){    
    row_indices = (VertexId *)cudaMallocPinned((edge_size + 1) * sizeof(VertexId));
    edge_weight_forward = (float *)cudaMallocPinned((edge_size + 1) * sizeof(VertexId));

    column_indices = (VertexId *)cudaMallocPinned((edge_size + 1) * sizeof(VertexId));///
    edge_weight_backward = (float *)cudaMallocPinned((edge_size + 1) * sizeof(VertexId));///

    destination = (long *)cudaMallocPinned((edge_size + 1) * sizeof(long));
    source      = (long *)cudaMallocPinned((edge_size + 1) * sizeof(long));
    source_backward  = (long *)cudaMallocPinned((edge_size + 1) * sizeof(long));
    }else
    if(dt==CPU_T){
        row_indices = (VertexId *)malloc((edge_size + 1) * sizeof(VertexId));
    edge_weight_forward = (float *)malloc((edge_size + 1) * sizeof(VertexId));

    column_indices = (VertexId *)malloc((edge_size + 1) * sizeof(VertexId));///
    edge_weight_backward = (float *)malloc((edge_size + 1) * sizeof(VertexId));///

    destination = (long *)malloc((edge_size + 1) * sizeof(long));
    source      = (long *)malloc((edge_size + 1) * sizeof(long));
    source_backward  = (long *)malloc((edge_size + 1) * sizeof(long));
    }else{
        assert(NOT_SUPPORT_DEVICE_TYPE);
    }

  }
    void getDevicePointerAll(){
    
    if(dt==GPU_T){ 
        column_offset_gpu = (VertexId *)getDevicePointer(column_offset);
        row_indices_gpu = (VertexId *)getDevicePointer(row_indices);
        edge_weight_forward_gpu = (float *)getDevicePointer(edge_weight_forward);
        
        row_offset_gpu = (VertexId *)getDevicePointer(row_offset);///
        column_indices_gpu = (VertexId *)getDevicePointer(column_indices);///
        edge_weight_backward_gpu = (float *)getDevicePointer(edge_weight_backward);/// 
       
        source_gpu = (long *)getDevicePointer(source);///
        destination_gpu = (long *)getDevicePointer(destination);///
        source_backward_gpu=(long*)getDevicePointer(source_backward);
    }else
    if(dt==CPU_T){
       ;     
    }else{
        assert(NOT_SUPPORT_DEVICE_TYPE);
    }
  }
   bool src_get_active(VertexId v_i){
       return this->source_active->get_bit(v_i-src_range[0]);
   }
   bool dst_get_active(VertexId v_i){
       return this->destination_active->get_bit(v_i-dst_range[0]);
   } 
   bool to_this_part_get_active(VertexId v_i){
       return this->destination_mirror_active->get_bit(v_i-dst_range[0]);
   }
   void src_set_active(VertexId v_i){
       this->source_active->set_bit(v_i-src_range[0]);
   }
   void dst_set_active(VertexId v_i){
       this->destination_active->set_bit(v_i-dst_range[0]);
   } 
   void to_this_part_set_active(VertexId v_i){
       this->destination_mirror_active->set_bit(v_i-dst_range[0]);
   }
   VertexId CO(VertexId v){
       return this->column_offset[v-dst_range[0]];
   }
   VertexId RI(VertexId idx){
       return row_indices[idx];
   }
   VertexId RO(VertexId v){
       return this->row_offset[v-src_range[0]];
   }
   VertexId CI(VertexId idx){
       return column_indices[idx];
   }

} CSC_segment_pinned;





typedef struct graph_Tensor_Segment_pinned1
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
} graph_seg_for_event;

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

#endif

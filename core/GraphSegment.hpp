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
#include <fstream>
#include <iostream>

#include "core/atomic.hpp"
#include "core/bitmap.hpp"
#include "core/constants.hpp"
#include "core/filesystem.hpp"
#include "core/time.hpp"
#include "core/type.hpp"
#include "cuda/test.hpp"
const bool NOT_SUPPORT_DEVICE_TYPE=false;

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
  Bitmap* forward_active;
  std::vector<Bitmap*> VertexToComm;
  
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
  void optional_init_sample(int layers){
      VertexToComm.clear();
      for(int i=0;i<layers;i++){
          VertexToComm.push_back(new Bitmap(batch_size_forward));
          VertexToComm[i]->clear();
      }
  }
  void allocVertexAssociateData(){
      
    source_active=new Bitmap(batch_size_backward);
    destination_active=new Bitmap(batch_size_forward);
    forward_active=new Bitmap(batch_size_forward);
            
    source_active->clear();
    destination_active->clear();
    forward_active->clear();
    
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
    edge_weight_forward = (float *)cudaMallocPinned((edge_size + 1) * sizeof(float));

    column_indices = (VertexId *)cudaMallocPinned((edge_size + 1) * sizeof(VertexId));///
    edge_weight_backward = (float *)cudaMallocPinned((edge_size + 1) * sizeof(float));///

    destination = (long *)cudaMallocPinned((edge_size + 1) * sizeof(long));
    source      = (long *)cudaMallocPinned((edge_size + 1) * sizeof(long));
    source_backward  = (long *)cudaMallocPinned((edge_size + 1) * sizeof(long));
    }else
    if(dt==CPU_T){
        row_indices = (VertexId *)malloc((edge_size + 1) * sizeof(VertexId));
    edge_weight_forward = (float *)malloc((edge_size + 1) * sizeof(float));

    column_indices = (VertexId *)malloc((edge_size + 1) * sizeof(VertexId));///
    edge_weight_backward = (float *)malloc((edge_size + 1) * sizeof(float));///

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
    void CopyGraphToDevice(){
    
    if(dt==GPU_T){ 
        column_offset_gpu =(VertexId*)cudaMallocGPU((batch_size_forward + 1) * sizeof(VertexId));
        row_indices_gpu = (VertexId *)cudaMallocGPU((edge_size + 1) * sizeof(VertexId));
        edge_weight_forward_gpu = (float *)cudaMallocGPU((edge_size + 1) * sizeof(float));
        
        move_bytes_in(column_offset_gpu,column_offset,(batch_size_forward + 1) * sizeof(VertexId));
        move_bytes_in(row_indices_gpu,row_indices,(edge_size + 1) * sizeof(VertexId));
        move_bytes_in(edge_weight_forward_gpu,edge_weight_forward,(edge_size + 1) * sizeof(float));
        
        row_offset_gpu = (VertexId*)cudaMallocGPU((batch_size_backward + 1) * sizeof(VertexId));
        column_indices_gpu = (VertexId *)cudaMallocGPU((edge_size + 1) * sizeof(VertexId));
        edge_weight_backward_gpu = (float *)cudaMallocGPU((edge_size + 1) * sizeof(float));
        
        move_bytes_in(row_offset_gpu,row_offset,(batch_size_backward + 1) * sizeof(VertexId));
        move_bytes_in(column_indices_gpu,column_indices,(edge_size + 1) * sizeof(VertexId));
        move_bytes_in(edge_weight_backward_gpu,edge_weight_backward,(edge_size + 1) * sizeof(float));
        
       
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
   bool get_forward_active(VertexId v_i){
       return this->forward_active->get_bit(v_i);
   }
   void set_forward_active(VertexId v_i){
       this->forward_active->set_bit(v_i);
   }
   bool get_backward_active(VertexId v_i){
       return this->source_active->get_bit(v_i-src_range[0]);
   }
   void src_set_active(VertexId v_i){
       this->source_active->set_bit(v_i-src_range[0]);
   }
   void dst_set_active(VertexId v_i){
       this->destination_active->set_bit(v_i-dst_range[0]);
   } 
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
  
  //engine related
  bool overlap;
  bool process_local;
  bool with_weight;
  size_t epochs;
  size_t repthreshold;
  bool lock_free;
  std::string algorithm;
  std::string layer_string;
  std::string feature_file;
  std::string edge_file;
  std::string label_file;
  std::string mask_file;
  bool with_cuda;
  
  //algorithm related:
  ValueType learn_rate;
  ValueType weight_decay;
  ValueType decay_rate;
  ValueType decay_epoch;
  ValueType drop_rate;
  
  
  
  
  void readFromCfgFile(std::string config_file){
      std::string cfg_oneline;
      std::ifstream inFile;
      inFile.open(config_file.c_str(),std::ios::in);
      while(getline(inFile,cfg_oneline)){
         std::string cfg_k;
         std::string cfg_v;
         int dlim= cfg_oneline.find(':');
         cfg_k=cfg_oneline.substr(0,dlim);
         cfg_v=cfg_oneline.substr(dlim+1,cfg_oneline.size()-dlim-1);
         if(0==cfg_k.compare("ALGORITHM")){
            this->algorithm=cfg_v;
         }else if(0==cfg_k.compare("VERTICES")){
            this->vertices=std::atoi(cfg_v.c_str());
         }else if(0==cfg_k.compare("EPOCHS")){
            this->epochs=std::atoi(cfg_v.c_str());
         }else if(0==cfg_k.compare("LAYERS")){
            this->layer_string=cfg_v;
         }else if(0==cfg_k.compare("EDGE_FILE")){
            this->edge_file=cfg_v.append("\0");
         }else if(0==cfg_k.compare("FEATURE_FILE")){
            this->feature_file=cfg_v; 
         }else if(0==cfg_k.compare("LABEL_FILE")){
             this->label_file=cfg_v;
         }else if(0==cfg_k.compare("MASK_FILE")){
             this->mask_file=cfg_v;
         }else if(0==cfg_k.compare("PROC_OVERLAP")){
            this->overlap=false;
            if(1==std::atoi(cfg_v.c_str()))
                this->overlap=true;
         }else if(0==cfg_k.compare("PROC_LOCAL")){
            this->process_local=false;
            if(1==std::atoi(cfg_v.c_str()))
                 this->process_local=true;
         }else if(0==cfg_k.compare("PROC_CUDA")){
            this->with_cuda=false;
            if(1==std::atoi(cfg_v.c_str()))
                 this->with_cuda=true;
         }else if(0==cfg_k.compare("PROC_REP")){
            this->repthreshold=std::atoi(cfg_v.c_str());
 
         }else if(0==cfg_k.compare("LOCK_FREE")){
            this->lock_free=false;
            if(1==std::atoi(cfg_v.c_str()))
                this->lock_free=true;
         }else if(0==cfg_k.compare("LEARN_RATE")){
            this->learn_rate=std::atof(cfg_v.c_str());
         }else if(0==cfg_k.compare("WEIGHT_DECAY")){
            this->weight_decay=std::atof(cfg_v.c_str());
         }else if(0==cfg_k.compare("DECAY_RATE")){
            this->decay_rate=std::atof(cfg_v.c_str());
         }else if(0==cfg_k.compare("DECAY_EPOCH")){
            this->decay_epoch=std::atof(cfg_v.c_str());
         }else if(0==cfg_k.compare("DROP_RATE")){
            this->drop_rate=std::atof(cfg_v.c_str());
         }
         else {
            printf("not supported configure\n");
         }             
      }
      inFile.close();  
  }
  void print(){
    
        std::cout<<"algorithm\t:\t"<<algorithm<<std::endl;
        std::cout<<"vertices\t:\t"<<vertices<<std::endl;
        std::cout<<"epochs\t\t:\t"<<epochs<<std::endl;
        std::cout<<"layers\t\t:\t"<<layer_string<<std::endl;
        std::cout<<"edge_file\t:\t"<<edge_file<<std::endl;
        std::cout<<"feature_file\t:\t"<<feature_file<<std::endl;
        std::cout<<"label_file\t:\t"<<label_file<<std::endl;
        std::cout<<"mask_file\t:\t"<<mask_file<<std::endl;
        std::cout<<"proc_overlap\t:\t"<<overlap<<std::endl;
        std::cout<<"proc_local\t:\t"<<process_local<<std::endl;
        std::cout<<"proc_cuda\t:\t"<<with_cuda<<std::endl;
        std::cout<<"proc_rep\t:\t"<<repthreshold<<std::endl;
        std::cout<<"lock_free\t:\t"<<lock_free<<std::endl;
        std::cout<<"learn_rate\t:\t"<<learn_rate<<std::endl;
        std::cout<<"weight_decay\t:\t"<<weight_decay<<std::endl;
        std::cout<<"decay_rate\t:\t"<<decay_rate<<std::endl;
        std::cout<<"decay_epoch\t:\t"<<decay_epoch<<std::endl;
        std::cout<<"drop_rate\t:\t"<<drop_rate<<std::endl;
      
  }
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
  bool lock_free;

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

typedef struct Graph_Store{
    std::vector<Bitmap*> sampled_vertices;
    VertexId* column_offset;
    VertexId* row_indices;
    
    COOChunk *_graph_cpu_in;
    COOChunk *_graph_cpu_out;
    std::vector<COOChunk *> graph_shard_in;
    std::vector<COOChunk *> graph_shard_out;
    
    void optional_generate_sample_graph(gnncontext *gnnctx,COOChunk*_graph_cpu_in){
        VertexId *tmp_column_offset;
        VertexId local_edge_size=gnnctx->l_e_num;
        column_offset= new VertexId[gnnctx->l_v_num+1];
        tmp_column_offset= new VertexId[gnnctx->l_v_num+1];
        row_indices= new VertexId[local_edge_size];
        memset(column_offset,0, sizeof(VertexId)*gnnctx->l_v_num+1);
        memset(tmp_column_offset,0, sizeof(VertexId)*gnnctx->l_v_num+1);
        memset(row_indices,0, sizeof(VertexId)*local_edge_size);
        for(int i=0;i<local_edge_size;i++){
            VertexId src=_graph_cpu_in->srcList[i];
            VertexId dst=_graph_cpu_in->dstList[i];
            VertexId dst_trans=dst-gnnctx->p_v_s;
            column_offset[dst_trans+1]+=1;
        }
        for(int i=0;i<gnnctx->l_v_num;i++){
            column_offset[i+1]+=column_offset[i];
            tmp_column_offset[i+1]=column_offset[i+1];
        }
        for(int i=0;i<local_edge_size;i++){
            VertexId src=_graph_cpu_in->srcList[i];
            VertexId dst=_graph_cpu_in->dstList[i];
            VertexId dst_trans=dst-gnnctx->p_v_s;
            VertexId r_index=tmp_column_offset[dst_trans];
            row_indices[r_index]=src;
        }
        delete [] tmp_column_offset;
    
   }
    
    
    
}Graph_Storage;

#endif

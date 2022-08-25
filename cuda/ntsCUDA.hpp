/*
 * test.h
 *
 *  Created on: Dec 3, 2019
 *      Author: wangqg
 */

#include "cuda_type.h"
// #define CUDA_ENABLE 1
#if CUDA_ENABLE
#include "cuda_runtime.h"
#endif

#ifndef TEST_HPP
#define TEST_HPP
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <vector>
//#include"../core/graph.hpp"

enum graph_type { CSR, CSC, PAIR };
enum weight_type { NULL_TYPE, SCALA_TYPE, TENSOR_TYPE };

void ntsFreeHost(void *buffer);
void *cudaMallocPinned(long size_of_bytes);
void *getDevicePointer(void *host_data_to_device);
void *cudaMallocGPU(long size_of_bytes);
void move_result_out(float *output, float *input, int src, int dst,
                     int feature_size, bool sync = true);
void move_data_in(float *d_pointer, float *h_pointer, int start, int end,
                  int feature_size, bool sync = true);
void move_edge_in(VertexId_CUDA *d_pointer, VertexId_CUDA *h_pointer,
                  VertexId_CUDA start, VertexId_CUDA end, int feature_size,
                  bool sync = true);
void move_bytes_in(void *d_pointer, void *h_pointer, long bytes,
                   bool sync = true);
void allocate_gpu_buffer(float **input, int size);
void allocate_gpu_edge(VertexId_CUDA **input, int size);
void aggregate_comm_result(float *aggregate_buffer, float *input_buffer,
                           int data_size, int feature_size,
                           int partition_offset, bool sync = true);
void FreeBuffer(float *buffer);
void FreeEdge(VertexId_CUDA *buffer);
void zero_buffer(float *buffer, int size);
void CUDA_DEVICE_SYNCHRONIZE();
void ResetDevice();

class deviceCSC{
public:
VertexId_CUDA* column_offset;
VertexId_CUDA* row_indices;
VertexId_CUDA* mirror_index;
VertexId_CUDA v_size;
VertexId_CUDA e_size;
VertexId_CUDA mirror_size;
bool require_mirror=false;

deviceCSC(){
    column_offset=NULL;
    row_indices=NULL;
}
void init(VertexId_CUDA v_size_, VertexId_CUDA e_size_,
        bool require_mirror_=false,VertexId_CUDA mirror_size_=0){
    v_size=v_size_;
    e_size=e_size_;
    require_mirror=false;
    column_offset=(VertexId_CUDA*)cudaMallocGPU((v_size_+1)*sizeof(VertexId_CUDA));
    row_indices=(VertexId_CUDA*)cudaMallocGPU((e_size_)*sizeof(VertexId_CUDA));
    if(require_mirror_){
        require_mirror=require_mirror_;
        mirror_size=mirror_size_;
        mirror_index=(VertexId_CUDA*)cudaMallocGPU((mirror_size_)*sizeof(VertexId_CUDA));
    }
}
void load_from_host(VertexId_CUDA* h_column_offset,VertexId_CUDA* h_row_indices,
            VertexId_CUDA* h_mirror_index){
   // printf("%d %d %d \n",v_size,e_size,mirror_size);
    move_bytes_in(column_offset,h_column_offset,(v_size+1)*sizeof(VertexId_CUDA));
    move_bytes_in(row_indices,h_row_indices,(e_size)*sizeof(VertexId_CUDA));
    move_bytes_in(mirror_index,h_mirror_index,(mirror_size)*sizeof(VertexId_CUDA));
}
void load_from_host(VertexId_CUDA* h_column_offset,VertexId_CUDA* h_row_indices){
    move_bytes_in(column_offset,h_column_offset,(v_size+1)*sizeof(VertexId_CUDA));
    move_bytes_in(row_indices,h_row_indices,(e_size)*sizeof(VertexId_CUDA));
}
void release(){
    FreeEdge(column_offset);
    FreeEdge(row_indices);
    if(require_mirror)
        FreeEdge(mirror_index);
}
~deviceCSC(){
}
};

class Cuda_Stream {
public:
  Cuda_Stream();
  void destory_Stream();
  cudaStream_t getStream();
  void CUDA_DEVICE_SYNCHRONIZE();
  cudaStream_t stream;
  void move_result_out(float *output, float *input, VertexId_CUDA src,
                       VertexId_CUDA dst, int feature_size, bool sync = true);
  void move_data_in(float *d_pointer, float *h_pointer, VertexId_CUDA start,
                    VertexId_CUDA end, int feature_size, bool sync = true);
  void move_edge_in(VertexId_CUDA *d_pointer, VertexId_CUDA *h_pointer,
                    VertexId_CUDA start, VertexId_CUDA end, int feature_size,
                    bool sync = true);
  void aggregate_comm_result(float *aggregate_buffer, float *input_buffer,
                             VertexId_CUDA data_size, int feature_size,
                             int partition_offset, bool sync = true);
  void deSerializeToGPU(float *input_gpu_buffer, float *input_buffer,
                        VertexId_CUDA data_size, VertexId_CUDA feature_size,
                        VertexId_CUDA partition_start,
                        VertexId_CUDA partition_end, bool sync);
  void aggregate_comm_result_debug(float *aggregate_buffer, float *input_buffer,
                                   VertexId_CUDA data_size,
                                   VertexId_CUDA feature_size,
                                   VertexId_CUDA partition_start,
                                   VertexId_CUDA partition_end, bool sync);
  
//fused op
  void Gather_By_Dst_From_Src(
      float *input, float *output, float *weight_forward,       // data
      VertexId_CUDA *row_indices, VertexId_CUDA *column_offset, // graph
      VertexId_CUDA src_start, VertexId_CUDA src_end, VertexId_CUDA dst_start,
      VertexId_CUDA dst_end, VertexId_CUDA edges, VertexId_CUDA batch_size,
      VertexId_CUDA feature_size, bool with_weight = false,
      bool tensor_weight = false);
  void Gather_By_Dst_From_Src_Optim(
      float *input, float *output, float *weight_forward, // data
      VertexId_CUDA *row_indices, VertexId_CUDA *column_offset,
      VertexId_CUDA src_start, VertexId_CUDA src_end, VertexId_CUDA dst_start,
      VertexId_CUDA dst_end, VertexId_CUDA edges, VertexId_CUDA batch_size,
      VertexId_CUDA feature_size, bool with_weight = false,
      bool tensor_weight = false);
  void Gather_By_Src_From_Dst_Optim(
      float *input, float *output, float *weight_forward, // data
      VertexId_CUDA *row_offset, VertexId_CUDA *column_indices,
      VertexId_CUDA src_start, VertexId_CUDA src_end, VertexId_CUDA dst_start,
      VertexId_CUDA dst_end, VertexId_CUDA edges, VertexId_CUDA batch_size,
      VertexId_CUDA feature_size, bool with_weight = false,
      bool tensor_weight = false);
  void Gather_By_Src_From_Dst(
      float *input, float *output, float *weight_forward,       // data
      VertexId_CUDA *row_offset, VertexId_CUDA *column_indices, // graph
      VertexId_CUDA src_start, VertexId_CUDA src_end, VertexId_CUDA dst_start,
      VertexId_CUDA dst_end, VertexId_CUDA edges, VertexId_CUDA batch_size,
      VertexId_CUDA feature_size, bool with_weight = false,
      bool tensor_weight = false);
  
  void Scatter_Src_Mirror_to_Msg(float* message,float* src_mirror_feature,//data 
        VertexId_CUDA* row_indices,VertexId_CUDA *column_offset,
        VertexId_CUDA* mirror_index, VertexId_CUDA batch_size,
        VertexId_CUDA feature_size);

  void Gather_Msg_To_Src_Mirror(float* src_mirror_feature,float* message,//data 
        VertexId_CUDA* row_indices,VertexId_CUDA *column_offset,
        VertexId_CUDA* mirror_index, VertexId_CUDA batch_size,
        VertexId_CUDA feature_size);

  void Scatter_Dst_to_Msg(float* message,float* dst_feature,//data 
        VertexId_CUDA* row_indices, VertexId_CUDA *column_offset,
        VertexId_CUDA batch_size, VertexId_CUDA feature_size);

  void Gather_Msg_to_Dst(float* dst_feature,float* message,//data 
        VertexId_CUDA* row_indices, VertexId_CUDA *column_offset,
        VertexId_CUDA batch_size, VertexId_CUDA feature_size);

  void Edge_Softmax_Forward_Block(float* msg_output,float* msg_input,//data 
        float* msg_cached,
        VertexId_CUDA* row_indices, VertexId_CUDA *column_offset,
        VertexId_CUDA batch_size, VertexId_CUDA feature_size);

  void Edge_Softmax_Backward_Block(float* msg_input_grad,float* msg_output_grad,//data 
        float* msg_cached,
        VertexId_CUDA* row_indices, VertexId_CUDA *column_offset,
        VertexId_CUDA batch_size, VertexId_CUDA feature_size);

  
  
  
  
  void Gather_By_Dst_From_Message(
      float *input, float *output,            // data
      VertexId_CUDA *src, VertexId_CUDA *dst, // graph
      VertexId_CUDA src_start, VertexId_CUDA src_end, VertexId_CUDA dst_start,
      VertexId_CUDA dst_end, VertexId_CUDA edges, VertexId_CUDA batch_size,
      VertexId_CUDA feature_size, bool with_weight = false,
      bool tensor_weight = false);
  void Scatter_Grad_Back_To_Message(
      float *input, float *message_grad, // data
      VertexId_CUDA *row_indices, VertexId_CUDA *column_offset,
      VertexId_CUDA src_start, VertexId_CUDA src_end, VertexId_CUDA dst_start,
      VertexId_CUDA dst_end, VertexId_CUDA edges, VertexId_CUDA batch_size,
      VertexId_CUDA feature_size, bool with_weight = true);
//  void process_local(float *local_buffer, float *input_tensor,
//                     VertexId_CUDA *src, VertexId_CUDA *dst,
//                     VertexId_CUDA *src_index, float *weight_buffer,
//                     int dst_offset, int dst_offset_end, int feature_size,
//                     int edge_size, bool sync = true);
//  void process_local_inter(float *local_buffer, float *input_tensor,
//                           VertexId_CUDA *src, VertexId_CUDA *dst,
//                           VertexId_CUDA *src_index, VertexId_CUDA *dst_index,
//                           float *weight_buffer, int dst_offset,
//                           int dst_offset_end, int feature_size, int edge_size,
//                           int out_put_buffer_size, bool sync = true);
//  void process_local_inter_para(float *local_buffer, float *input_tensor,
//                                VertexId_CUDA *src, VertexId_CUDA *dst,
//                                VertexId_CUDA *src_index,
//                                VertexId_CUDA *dst_index, float *para,
//                                int dst_offset, int dst_offset_end,
//                                int feature_size, int edge_size,
//                                int out_put_buffer_size, bool sync = true);
};

// int test();
#endif /* TEST_H_ */

/*
 * test.h
 *
 *  Created on: Dec 3, 2019
 *      Author: wangqg
 */

#include "cuda_type.h"
#define CUDA_ENABLE 1
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
void ResetDevice();

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
  void Gather_By_Dst_From_Message(
      float *input, float *output,            // data
      VertexId_CUDA *src, VertexId_CUDA *dst, // graph
      VertexId_CUDA src_start, VertexId_CUDA src_end, VertexId_CUDA dst_start,
      VertexId_CUDA dst_end, VertexId_CUDA edges, VertexId_CUDA batch_size,
      VertexId_CUDA feature_size, bool with_weight = false,
      bool tensor_weight = false);
  void Gather_By_Dst_From_Src_Para(
      float *input, float *output, float *para_forward, // data
      VertexId_CUDA *src, VertexId_CUDA *dst,           // graph
      VertexId_CUDA src_start, VertexId_CUDA src_end, VertexId_CUDA dst_start,
      VertexId_CUDA dst_end, VertexId_CUDA edges, VertexId_CUDA batch_size,
      VertexId_CUDA feature_size, bool sync = true);
  void Scatter_Grad_Back_To_Weight(
      float *input, float *output_grad, float *weight_grad, // data
      long *src, long *dst,                                 // graph
      VertexId_CUDA src_start, VertexId_CUDA src_end, VertexId_CUDA dst_start,
      VertexId_CUDA dst_end, VertexId_CUDA edges, VertexId_CUDA batch_size,
      VertexId_CUDA feature_size, bool tensor_weight = true);
  void Scatter_Grad_Back_To_Message(
      float *input, float *message_grad, // data
      VertexId_CUDA *row_indices, VertexId_CUDA *column_offset,
      VertexId_CUDA src_start, VertexId_CUDA src_end, VertexId_CUDA dst_start,
      VertexId_CUDA dst_end, VertexId_CUDA edges, VertexId_CUDA batch_size,
      VertexId_CUDA feature_size, bool with_weight = true);
  //	void Gather_By_Dst_From_Src_shrink(float *input, float *output, float
  //*weight_forward, //data
  // VertexId_CUDA *src, VertexId_CUDA *dst,
  ////graph VertexId_CUDA *index_gpu_buffer, VertexId_CUDA *vertex_gpu_buffer,
  //								   VertexId_CUDA
  // src_start, VertexId_CUDA src_end,
  // VertexId_CUDA dst_start, VertexId_CUDA dst_end,
  // VertexId_CUDA actual_dst_start, VertexId_CUDA batch_size,
  // VertexId_CUDA feature_size, bool sync = true);
  void process_local(float *local_buffer, float *input_tensor,
                     VertexId_CUDA *src, VertexId_CUDA *dst,
                     VertexId_CUDA *src_index, float *weight_buffer,
                     int dst_offset, int dst_offset_end, int feature_size,
                     int edge_size, bool sync = true);
  void process_local_inter(float *local_buffer, float *input_tensor,
                           VertexId_CUDA *src, VertexId_CUDA *dst,
                           VertexId_CUDA *src_index, VertexId_CUDA *dst_index,
                           float *weight_buffer, int dst_offset,
                           int dst_offset_end, int feature_size, int edge_size,
                           int out_put_buffer_size, bool sync = true);
  void process_local_inter_para(float *local_buffer, float *input_tensor,
                                VertexId_CUDA *src, VertexId_CUDA *dst,
                                VertexId_CUDA *src_index,
                                VertexId_CUDA *dst_index, float *para,
                                int dst_offset, int dst_offset_end,
                                int feature_size, int edge_size,
                                int out_put_buffer_size, bool sync = true);
};

void *cudaMallocPinned(long size_of_bytes);
void *getDevicePointer(void *host_data_to_device);
void *cudaMallocGPU(long size_of_bytes);

void forward_on_GPU(float *input, float *output, float *weight_forward, // data
                    VertexId_CUDA *src, VertexId_CUDA *dst,             // graph
                    VertexId_CUDA src_start, VertexId_CUDA src_end,
                    VertexId_CUDA dst_start, VertexId_CUDA dst_end,
                    VertexId_CUDA edges, VertexId_CUDA batch_size,
                    VertexId_CUDA feature_size);
void Gather_By_Dst_From_Src(float *input, float *output,
                            float *weight_forward,                  // data
                            VertexId_CUDA *src, VertexId_CUDA *dst, // graph
                            VertexId_CUDA src_start, VertexId_CUDA src_end,
                            VertexId_CUDA dst_start, VertexId_CUDA dst_end,
                            VertexId_CUDA edges, VertexId_CUDA batch_size,
                            VertexId_CUDA feature_size, bool sync = true);

void backward_on_GPU(float *input, float *output, float *weight_forward, // data
                     VertexId_CUDA *src, VertexId_CUDA *dst, // graph
                     VertexId_CUDA src_start, VertexId_CUDA src_end,
                     VertexId_CUDA dst_start, VertexId_CUDA dst_end,
                     VertexId_CUDA edges, VertexId_CUDA batch_size,
                     VertexId_CUDA feature_size);

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
// int test();
#endif /* TEST_H_ */

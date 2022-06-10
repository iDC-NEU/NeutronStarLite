/*
Copyright (c) 2021-2022 Qiange Wang, Northeastern University

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
#include <fcntl.h>
#include <malloc.h>
#include <numa.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "core/GraphSegment.h"
#include "dep/gemini/atomic.hpp"
#include "dep/gemini/bitmap.hpp"
#include "dep/gemini/constants.hpp"
#include "dep/gemini/filesystem.hpp"
#include "dep/gemini/time.hpp"
#include "dep/gemini/type.hpp"


void CSC_segment_pinned::init(VertexId src_start, VertexId src_end,
                              VertexId dst_start, VertexId dst_end,
                              VertexId edge_size_, DeviceLocation dt_) {
  src_range[0] = src_start;
  src_range[1] = src_end;
  dst_range[0] = dst_start;
  dst_range[1] = dst_end;
  batch_size_backward = src_range[1] - src_range[0];
  batch_size_forward = dst_range[1] - dst_range[0];
  edge_size = edge_size_;
  dt = dt_;
}

//void CSC_segment_pinned::optional_init_sample(int layers) {
//  VertexToComm.clear();
//  for (int i = 0; i < layers; i++) {
//    VertexToComm.push_back(new Bitmap(batch_size_forward));
//    VertexToComm[i]->clear();
//  }
//}

// Allocate bitmap for forward and backward vertex
// and row_offset and column_offset, for CSC/CSR format
void CSC_segment_pinned::allocVertexAssociateData() {

  source_active = new Bitmap(batch_size_backward);
  destination_active = new Bitmap(batch_size_forward);

  source_active->clear();
  destination_active->clear();
  
#if CUDA_ENABLE
  if (dt == GPU_T) {
    column_offset = (VertexId *)cudaMallocPinned((batch_size_forward + 1) *
                                                 sizeof(VertexId));
    row_offset = (VertexId *)cudaMallocPinned((batch_size_backward + 1) *
                                              sizeof(VertexId)); ///
  } else
#endif

      if (dt == CPU_T) {
    column_offset =
        (VertexId *)malloc((batch_size_forward + 1) * sizeof(VertexId));
    row_offset =
        (VertexId *)malloc((batch_size_backward + 1) * sizeof(VertexId)); ///
    memset(column_offset, 0, (batch_size_forward + 1) * sizeof(VertexId));
    memset(row_offset, 0, (batch_size_backward + 1) * sizeof(VertexId));
    //        column_offset = new VertexId[batch_size_forward+1];
    //        row_offset = new VertexId[batch_size_backward+1];///
  } else {
    assert(NOT_SUPPORT_DEVICE_TYPE);
  }
}

// allocate space for edge associated data.
// e.g. destination vertexID, edge data
void CSC_segment_pinned::allocEdgeAssociateData() {
#if CUDA_ENABLE
  if (dt == GPU_T) {
    row_indices =
        (VertexId *)cudaMallocPinned((edge_size + 1) * sizeof(VertexId));
    edge_weight_forward =
        (ValueType *)cudaMallocPinned((edge_size + 1) * sizeof(ValueType));

    column_indices =
        (VertexId *)cudaMallocPinned((edge_size + 1) * sizeof(VertexId)); ///
    edge_weight_backward =
        (ValueType *)cudaMallocPinned((edge_size + 1) * sizeof(ValueType)); ///

    destination = (long *)cudaMallocPinned((edge_size + 1) * sizeof(long));
    source = (long *)cudaMallocPinned((edge_size + 1) * sizeof(long));
  } else
#endif
      if (dt == CPU_T) {
    row_indices = (VertexId *)malloc((edge_size + 1) * sizeof(VertexId));
    memset(row_indices, 0, (edge_size + 1) * sizeof(VertexId));
    edge_weight_forward =
        (ValueType *)malloc((edge_size + 1) * sizeof(ValueType));
    memset(edge_weight_forward, 0, (edge_size + 1) * sizeof(VertexId));
    column_indices = (VertexId *)malloc((edge_size + 1) * sizeof(VertexId)); ///
    memset(column_indices, 0, (edge_size + 1) * sizeof(VertexId));
    edge_weight_backward =
        (ValueType *)malloc((edge_size + 1) * sizeof(ValueType)); ///
    memset(edge_weight_backward, 0, (edge_size + 1) * sizeof(ValueType));
    destination = (long *)malloc((edge_size + 1) * sizeof(long));
    memset(destination, 0, (edge_size + 1) * sizeof(long));
    source = (long *)malloc((edge_size + 1) * sizeof(long));
    memset(source, 0, (edge_size + 1) * sizeof(long));
//    source_backward = (long *)malloc((edge_size + 1) * sizeof(long));
//    memset(source_backward, 0, (edge_size + 1) * sizeof(long));
  } else {
    assert(NOT_SUPPORT_DEVICE_TYPE);
  }
}

void CSC_segment_pinned::freeAdditional(){
    #if CUDA_ENABLE
    if (dt == GPU_T) {
        ntsFreeHost(destination);
        ntsFreeHost(source);
    }
    #endif
    if (dt == CPU_T) {
        free(destination);
        free(source);
    }
}

void CSC_segment_pinned::getDevicePointerAll() {
#if CUDA_ENABLE
  if (dt == GPU_T) {
    column_offset_gpu = (VertexId *)getDevicePointer(column_offset);
    row_indices_gpu = (VertexId *)getDevicePointer(row_indices);
    edge_weight_forward_gpu =
        (ValueType *)getDevicePointer(edge_weight_forward);

    row_offset_gpu = (VertexId *)getDevicePointer(row_offset);         ///
    column_indices_gpu = (VertexId *)getDevicePointer(column_indices); ///
    edge_weight_backward_gpu =
        (ValueType *)getDevicePointer(edge_weight_backward); ///

    source_gpu = (long *)getDevicePointer(source);           ///
    destination_gpu = (long *)getDevicePointer(destination); ///
//    source_backward_gpu = (long *)getDevicePointer(source_backward);
  } else
#endif
      if (dt == CPU_T) {
    ;
  } else {
    assert(NOT_SUPPORT_DEVICE_TYPE);
  }
}

void CSC_segment_pinned::CopyGraphToDevice() {
#if CUDA_ENABLE
  if (dt == GPU_T) {
    column_offset_gpu =
        (VertexId *)cudaMallocGPU((batch_size_forward + 1) * sizeof(VertexId));
    row_indices_gpu =
        (VertexId *)cudaMallocGPU((edge_size + 1) * sizeof(VertexId));
    edge_weight_forward_gpu =
        (ValueType *)cudaMallocGPU((edge_size + 1) * sizeof(ValueType));

    move_bytes_in(column_offset_gpu, column_offset,
                  (batch_size_forward + 1) * sizeof(VertexId));
    move_bytes_in(row_indices_gpu, row_indices,
                  (edge_size + 1) * sizeof(VertexId));
    move_bytes_in(edge_weight_forward_gpu, edge_weight_forward,
                  (edge_size + 1) * sizeof(ValueType));

    row_offset_gpu =
        (VertexId *)cudaMallocGPU((batch_size_backward + 1) * sizeof(VertexId));
    column_indices_gpu =
        (VertexId *)cudaMallocGPU((edge_size + 1) * sizeof(VertexId));
    edge_weight_backward_gpu =
        (ValueType *)cudaMallocGPU((edge_size + 1) * sizeof(ValueType));

    move_bytes_in(row_offset_gpu, row_offset,
                  (batch_size_backward + 1) * sizeof(VertexId));
    move_bytes_in(column_indices_gpu, column_indices,
                  (edge_size + 1) * sizeof(VertexId));
    move_bytes_in(edge_weight_backward_gpu, edge_weight_backward,
                  (edge_size + 1) * sizeof(ValueType));

    source_gpu = (long *)getDevicePointer(source);           ///
    destination_gpu = (long *)getDevicePointer(destination); ///
//    source_backward_gpu = (long *)getDevicePointer(source_backward);

  } else
#endif
      if (dt == CPU_T) {
    ;
  } else {
    assert(NOT_SUPPORT_DEVICE_TYPE);
  }
}

void InputInfo::readFromCfgFile(std::string config_file) {
  std::string cfg_oneline;
  std::ifstream inFile;
  inFile.open(config_file.c_str(), std::ios::in);
  while (getline(inFile, cfg_oneline)) {
    std::string cfg_k;
    std::string cfg_v;
    int dlim = cfg_oneline.find(':');
    cfg_k = cfg_oneline.substr(0, dlim);
    cfg_v = cfg_oneline.substr(dlim + 1, cfg_oneline.size() - dlim - 1);
    if (0 == cfg_k.compare("ALGORITHM")) {
      this->algorithm = cfg_v;
    } else if (0 == cfg_k.compare("VERTICES")) {
      this->vertices = std::atoi(cfg_v.c_str());
    } else if (0 == cfg_k.compare("EPOCHS")) {
      this->epochs = std::atoi(cfg_v.c_str());
    } else if (0 == cfg_k.compare("LAYERS")) {
      this->layer_string = cfg_v;
    } else if (0 == cfg_k.compare("FANOUT")) {
      this->fanout_string = cfg_v;
    } else if (0 == cfg_k.compare("EDGE_FILE")) {
      this->edge_file = cfg_v.append("\0");
    } else if (0 == cfg_k.compare("FEATURE_FILE")) {
      this->feature_file = cfg_v;
    } else if (0 == cfg_k.compare("LABEL_FILE")) {
      this->label_file = cfg_v;
    } else if (0 == cfg_k.compare("MASK_FILE")) {
      this->mask_file = cfg_v;
    } else if (0 == cfg_k.compare("PROC_OVERLAP")) {
      this->overlap = false;
      if (1 == std::atoi(cfg_v.c_str()))
        this->overlap = true;
    } else if (0 == cfg_k.compare("PROC_LOCAL")) {
      this->process_local = false;
      if (1 == std::atoi(cfg_v.c_str()))
        this->process_local = true;
    } else if (0 == cfg_k.compare("PROC_CUDA")) {
      this->with_cuda = false;
      if (1 == std::atoi(cfg_v.c_str()))
        this->with_cuda = true;
    } else if (0 == cfg_k.compare("PROC_REP")) {
      this->repthreshold = std::atoi(cfg_v.c_str());

    } else if (0 == cfg_k.compare("LOCK_FREE")) {
      this->lock_free = false;
      if (1 == std::atoi(cfg_v.c_str()))
        this->lock_free = true;
    } else if (0 == cfg_k.compare("LEARN_RATE")) {
      this->learn_rate = std::atof(cfg_v.c_str());
    } else if (0 == cfg_k.compare("WEIGHT_DECAY")) {
      this->weight_decay = std::atof(cfg_v.c_str());
    } else if (0 == cfg_k.compare("DECAY_RATE")) {
      this->decay_rate = std::atof(cfg_v.c_str());
    } else if (0 == cfg_k.compare("DECAY_EPOCH")) {
      this->decay_epoch = std::atof(cfg_v.c_str());
    } else if (0 == cfg_k.compare("DROP_RATE")) {
      this->drop_rate = std::atof(cfg_v.c_str());
    } else if (0 == cfg_k.compare("BATCH_SIZE")) {
      this->batch_size = std::atoi(cfg_v.c_str());
    } else if (0 == cfg_k.compare("OPTIM_KERNEL")) {
      this->optim_kernel_enable = true;
      if (1 == std::atoi(cfg_v.c_str()))
        this->optim_kernel_enable = true;
    }

    else {
      printf("not supported configure\n");
    }
  }
  inFile.close();
}

void InputInfo::print() {

  std::cout << "algorithm\t:\t" << algorithm << std::endl;
  std::cout << "vertices\t:\t" << vertices << std::endl;
  std::cout << "epochs\t\t:\t" << epochs << std::endl;
  std::cout << "layers\t\t:\t" << layer_string << std::endl;
  std::cout << "fanout\t\t:\t" << fanout_string << std::endl;
  std::cout << "batch_size\t\t:\t" << batch_size << std::endl;
  std::cout << "edge_file\t:\t" << edge_file << std::endl;
  std::cout << "feature_file\t:\t" << feature_file << std::endl;
  std::cout << "label_file\t:\t" << label_file << std::endl;
  std::cout << "mask_file\t:\t" << mask_file << std::endl;
  std::cout << "proc_overlap\t:\t" << overlap << std::endl;
  std::cout << "proc_local\t:\t" << process_local << std::endl;
  std::cout << "proc_cuda\t:\t" << with_cuda << std::endl;
  std::cout << "proc_rep\t:\t" << repthreshold << std::endl;
  std::cout << "lock_free\t:\t" << lock_free << std::endl;
  std::cout << "optim_kernel\t:\t" << optim_kernel_enable << std::endl;
  std::cout << "learn_rate\t:\t" << learn_rate << std::endl;
  std::cout << "weight_decay\t:\t" << weight_decay << std::endl;
  std::cout << "decay_rate\t:\t" << decay_rate << std::endl;
  std::cout << "decay_epoch\t:\t" << decay_epoch << std::endl;
  std::cout << "drop_rate\t:\t" << drop_rate << std::endl;
  std::cout <<"------------------input info--------------"<<std::endl;
}

void RuntimeInfo::init_rtminfo() {
  process_local = false;
  process_overlap = false;
  epoch = -1;
  curr_layer = -1;
  embedding_size = -1;
  copy_data = false;
  with_cuda = false;
  lock_free = false;

#if CUDA_ENABLE
  cuda_stream_public = new Cuda_Stream();
//  printf("cuda stream avaliable\n");
#endif
}

void RuntimeInfo::set(InputInfo *gnncfg) {
  this->process_local = gnncfg->process_local;
  this->reduce_comm = gnncfg->process_local;
  this->process_overlap = gnncfg->overlap;
  this->lock_free = gnncfg->lock_free;
  this->optim_kernel_enable = gnncfg->optim_kernel_enable;
}

void GraphStorage::optional_generate_sample_graph(GNNContext *gnnctx,
                                                  COOChunk *_graph_cpu_in) {
  VertexId *tmp_column_offset;
  VertexId local_edge_size = gnnctx->l_e_num;
  column_offset = new VertexId[gnnctx->l_v_num + 1];
  tmp_column_offset = new VertexId[gnnctx->l_v_num + 1];
  row_indices = new VertexId[local_edge_size];
  memset(column_offset, 0, sizeof(VertexId) * gnnctx->l_v_num + 1);
  memset(tmp_column_offset, 0, sizeof(VertexId) * gnnctx->l_v_num + 1);
  memset(row_indices, 0, sizeof(VertexId) * local_edge_size);
  for (int i = 0; i < local_edge_size; i++) {
    VertexId src = _graph_cpu_in->srcList[i];
    VertexId dst = _graph_cpu_in->dstList[i];
    VertexId dst_trans = dst - gnnctx->p_v_s;
    column_offset[dst_trans + 1] += 1;
  }
  for (int i = 0; i < gnnctx->l_v_num; i++) {
    column_offset[i + 1] += column_offset[i];
    tmp_column_offset[i + 1] = column_offset[i + 1];
  }
  for (int i = 0; i < local_edge_size; i++) {
    VertexId src = _graph_cpu_in->srcList[i];
    VertexId dst = _graph_cpu_in->dstList[i];
    VertexId dst_trans = dst - gnnctx->p_v_s;
    VertexId r_index = tmp_column_offset[dst_trans];
    row_indices[r_index] = src;
  }
  delete[] tmp_column_offset;
}

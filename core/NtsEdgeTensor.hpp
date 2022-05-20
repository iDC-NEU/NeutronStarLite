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

#ifndef NTSTENSOR_HPP
#define NTSTENSOR_HPP
#include "core/GraphSegment.h"
#include "core/NtsScheduler.hpp"

namespace nts {
struct ntsEdgeTensor {
  ntsEdgeTensor() {
    subgraph = nullptr;
    data_buffer = nullptr;
    size_0 = -1;
    size_1 = -1;
    dim = 0;
  }
  ntsEdgeTensor(int feature_size, CSC_segment_pinned *subgraph_,
                NtsScheduler *ntsscheduler_) {
    initEdgeTensor(feature_size, subgraph_, ntsscheduler_);
  }

  void initEdgeTensorFromTensor(int feature_size, CSC_segment_pinned *subgraph_,
                                NtsScheduler *ntsscheduler_, NtsVar &e_tensor) {
    size_1 = feature_size;
    subgraph = subgraph_;
    ntsscheduler = ntsscheduler_;
    size_0 = subgraph->edge_size;
    long size = ((long)size_1) * size_0;
    edgeTensor = e_tensor;
    data_buffer =
        ntsscheduler->getWritableBuffer(e_tensor, torch::DeviceType::CPU);
    edgeTensorIndexedbyDst.clear();
    NtsVar d;
    edgeTensorIndexedbyDst.resize(subgraph->batch_size_forward, d);
    for (VertexId vtx = subgraph->dst_range[0]; vtx < subgraph->dst_range[1];
         vtx++) {
      long eid_start = subgraph->column_offset[vtx];
      long eid_end = subgraph->column_offset[vtx + 1];
      int offset = vtx - subgraph->dst_range[0];
      edgeTensorIndexedbyDst[offset] = ntsscheduler->NewLeafTensor(
          data_buffer + eid_start * size_1, {(eid_end - eid_start), size_1},
          torch::DeviceType::CPU);
    }
    dim = 2;
  }

  void initEdgeTensor(int feature_size, CSC_segment_pinned *subgraph_,
                      NtsScheduler *ntsscheduler_) {
    size_1 = feature_size;
    subgraph = subgraph_;
    ntsscheduler = ntsscheduler_;
    size_0 = subgraph->edge_size;
    long size = ((long)size_1) * size_0;
    data_buffer = new float[size];
    edgeTensorIndexedbyDst.clear();
    NtsVar d;
    edgeTensorIndexedbyDst.resize(subgraph->batch_size_forward, d);
    for (VertexId vtx = subgraph->dst_range[0]; vtx < subgraph->dst_range[1];
         vtx++) {
      long eid_start = subgraph->column_offset[vtx];
      long eid_end = subgraph->column_offset[vtx + 1];
      int offset = vtx - subgraph->dst_range[0];
      edgeTensorIndexedbyDst[offset] = ntsscheduler->NewLeafTensor(
          data_buffer + eid_start * size_1, {(eid_end - eid_start), size_1},
          torch::DeviceType::CPU);
    }
    edgeTensor = ntsscheduler->NewLeafTensor(data_buffer, {size_0, size_1},
                                             torch::DeviceType::CPU);
    dim = 2;
  }
  NtsVar &getNbrTensor(VertexId vtx) {
    return edgeTensorIndexedbyDst[vtx - subgraph->dst_range[0]];
  }

  long size(int index) {
    assert(0 <= index && index <= 1);
    if (index == 0)
      return size_0;
    else
      return size_1;
  }
  float *data_buffer;
  int dim;
  long size_0;
  long size_1;
  std::vector<NtsVar> edgeTensorIndexedbyDst;
  NtsVar edgeTensor;
  CSC_segment_pinned *subgraph;
  NtsScheduler *ntsscheduler;
};

struct ntsVertexTensor {
  ntsVertexTensor() {
    subgraph = nullptr;
    data_buffer = nullptr;
    size_0 = -1;
    size_1 = -1;
    dim = 0;
  }
  ntsVertexTensor(int feature_size, CSC_segment_pinned *subgraph_,
                  NtsScheduler *ntsscheduler_) {
    initVertexTensor(feature_size, subgraph_, ntsscheduler_);
  }

  void initVertexTensorFromTensor(int feature_size,
                                  CSC_segment_pinned *subgraph_,
                                  NtsScheduler *ntsscheduler_,
                                  NtsVar &e_tensor) {
    size_1 = feature_size;
    subgraph = subgraph_;
    ntsscheduler = ntsscheduler_;
    size_0 = subgraph->batch_size_forward;
    long size = ((long)size_1) * size_0;
    vertexTensor = e_tensor;
    data_buffer =
        ntsscheduler->getWritableBuffer(e_tensor, torch::DeviceType::CPU);
    vertexTensorIndexedbyDst.clear();
    NtsVar d;
    vertexTensorIndexedbyDst.resize(subgraph->batch_size_forward, d);
    for (VertexId vtx = subgraph->dst_range[0]; vtx < subgraph->dst_range[1];
         vtx++) {
      int offset = vtx - subgraph->dst_range[0];
      vertexTensorIndexedbyDst[offset] = ntsscheduler->NewLeafTensor(
          data_buffer + offset * size_1, {1, size_1}, torch::DeviceType::CPU);
    }
    dim = 2;
  }

  void initVertexTensor(int feature_size, CSC_segment_pinned *subgraph_,
                        NtsScheduler *ntsscheduler_) {
    size_1 = feature_size;
    subgraph = subgraph_;
    ntsscheduler = ntsscheduler_;
    size_0 = subgraph->batch_size_forward;
    long size = ((long)size_1) * size_0;
    data_buffer = new float[size];
    vertexTensorIndexedbyDst.clear();
    NtsVar d;
    vertexTensorIndexedbyDst.resize(subgraph->batch_size_forward, d);
    for (VertexId vtx = subgraph->dst_range[0]; vtx < subgraph->dst_range[1];
         vtx++) {
      int offset = vtx - subgraph->dst_range[0];
      vertexTensorIndexedbyDst[offset] = ntsscheduler->NewLeafTensor(
          data_buffer + offset * size_1, {1, size_1}, torch::DeviceType::CPU);
    }
    vertexTensor = ntsscheduler->NewLeafTensor(data_buffer, {size_0, size_1},
                                               torch::DeviceType::CPU);
    dim = 2;
  }
  NtsVar &getVtxTensor(VertexId vtx) {
    return vertexTensorIndexedbyDst[vtx - subgraph->dst_range[0]];
  }

  long size(int index) {
    assert(0 <= index && index <= 1);
    if (index == 0)
      return size_0;
    else
      return size_1;
  }
  float *data_buffer;
  int dim;
  long size_0;
  long size_1;
  std::vector<NtsVar> vertexTensorIndexedbyDst;
  NtsVar vertexTensor;
  CSC_segment_pinned *subgraph;
  NtsScheduler *ntsscheduler;
};

} // namespace nts

#endif

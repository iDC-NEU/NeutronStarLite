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
#ifndef NTSDISTCPUFUSEDGRAPHOP_HPP
#define NTSDISTCPUFUSEDGRAPHOP_HPP
#include <assert.h>
#include <map>
#include <math.h>
#include <stack>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <vector>

#include "core/graph.hpp"
#include "core/ntsBaseOp.hpp"
#include "core/PartitionedGraph.hpp"

namespace nts {
namespace op {

//class ntsGraphOp {
//public:
//  Graph<Empty> *graph_;
//  VertexSubset *active_;
//  ntsGraphOp() { ; }
//  ntsGraphOp(Graph<Empty> *graph, VertexSubset *active) {
//    graph_ = graph;
//    active_ = active;
//  }
//  virtual NtsVar &forward(NtsVar &input) = 0;
//  virtual NtsVar backward(NtsVar &output_grad) = 0;
//};
    
#if CUDA_ENABLE    
class ForwardGPUfuseOp : public ntsGraphOp{
public:
  std::vector<CSC_segment_pinned *> subgraphs;

  ForwardGPUfuseOp(PartitionedGraph *partitioned_graph,VertexSubset *active)
      : ntsGraphOp(partitioned_graph, active) {
    subgraphs = partitioned_graph->graph_chunks;
  }
  NtsVar forward(NtsVar &f_input){
        int feature_size = f_input.size(1);
  NtsVar f_input_cpu = f_input.cpu();
  NtsVar f_output=graph_->Nts->NewKeyTensor(f_input,torch::DeviceType::CUDA);
  ValueType *f_input_cpu_buffered = f_input_cpu.accessor<ValueType, 2>().data();

  { // original communication
    graph_->sync_compute_decoupled<int, ValueType>(
        f_input, subgraphs,
        [&](VertexId src) {
          graph_->NtsComm->emit_buffer(
              src, f_input_cpu_buffered + (src - graph_->gnnctx->p_v_s) * feature_size,
              feature_size);
        },
        f_output, feature_size);
  }
  return f_output;
  }
  
  NtsVar backward(NtsVar &f_output_grad){
      int feature_size = f_output_grad.size(1);
      NtsVar f_input_grad=graph_->Nts->NewKeyTensor(f_output_grad,torch::DeviceType::CUDA);
  // if (!selective)
  {
    graph_->compute_sync_decoupled<int, ValueType>(
        f_output_grad, subgraphs,
        [&](VertexId src, VertexAdjList<Empty> outgoing_adj) { // pull
          graph_->NtsComm->emit_buffer(
              src, graph_->output_cpu_buffer + (src)*feature_size,
              feature_size);
        },
        f_input_grad, feature_size);
  }
      return f_input_grad;
  }
};
#endif



} // namespace graphop
} // namespace nts

#endif

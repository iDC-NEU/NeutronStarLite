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
#ifndef NTSSINGLEGPUFUSEDGRAPHOP_HPP
#define NTSSINGLEGPUFUSEDGRAPHOP_HPP
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
class ForwardSingleGPUfuseOp : public ntsGraphOp{
public:
  std::vector<CSC_segment_pinned *> subgraphs;
  
  ForwardSingleGPUfuseOp(PartitionedGraph *partitioned_graph,VertexSubset *active)
      : ntsGraphOp(partitioned_graph, active) {
    subgraphs = partitioned_graph->graph_chunks;
  }
  NtsVar forward(NtsVar &f_input){
    int feature_size = f_input.size(1);
    NtsVar f_output=graph_->Nts->NewKeyTensor(f_input,torch::DeviceType::CUDA);
    graph_->forward_single<int, ValueType>(f_input, subgraphs, f_output, feature_size);
    return f_output;
  }
  
  NtsVar backward(NtsVar &f_output_grad){
    int feature_size = f_output_grad.size(1);
    NtsVar f_input_grad=graph_->Nts->NewKeyTensor(f_output_grad,torch::DeviceType::CUDA);
    graph_->backward_single<int, ValueType>(f_output_grad, subgraphs, 
            f_input_grad, feature_size);
      return f_input_grad;
  }    

};
#endif



} // namespace graphop
} // namespace nts

#endif

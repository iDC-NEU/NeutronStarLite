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
#ifndef NTSSINGLECPUGRAPHOP_HPP
#define NTSSINGLECPUGRAPHOP_HPP
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

class SingleCPUSrcDstScatterOp : public ntsGraphOp{
public:
  std::vector<CSC_segment_pinned *> subgraphs;
  
  SingleCPUSrcDstScatterOp(PartitionedGraph *partitioned_graph,VertexSubset *active)
      : ntsGraphOp(partitioned_graph, active) {
    subgraphs = partitioned_graph->graph_chunks;
  }
  NtsVar forward(NtsVar &f_input){
    int feature_size = f_input.size(1);
    NtsVar f_output=graph_->Nts->NewKeyTensor({graph_->gnnctx->l_e_num, 
                2*feature_size},torch::DeviceType::CPU);
    ValueType *f_input_buffer =
      graph_->Nts->getWritableBuffer(f_input, torch::DeviceType::CPU);
    ValueType *f_output_buffer =
      graph_->Nts->getWritableBuffer(f_output, torch::DeviceType::CPU);            
  graph_->local_vertex_operation<int, ValueType>( // For EACH Vertex
      [&](VertexId vtx, CSC_segment_pinned *subgraph, VertexId recv_id) {
        // iterate the incoming edge for vtx
        for (long eid = subgraph->column_offset[vtx];
             eid < subgraph->column_offset[vtx + 1]; eid++) {
          VertexId src = subgraph->row_indices[eid];
          nts_copy(f_output_buffer, eid * 2, f_input_buffer, src, feature_size,1);
          nts_copy(f_output_buffer, eid * 2 + 1, f_input_buffer, vtx, feature_size,1);
        }
      },
      subgraphs, feature_size, active_);            
    return f_output;
  }
  
  NtsVar backward(NtsVar &f_output_grad){
      int feature_size=f_output_grad.size(1);
                     assert(feature_size%2==0); 
    NtsVar f_input_grad=graph_->Nts->NewLeafTensor({graph_->gnnctx->l_v_num, 
                feature_size/2},torch::DeviceType::CPU);
              
    ValueType *f_input_grad_buffer =
      graph_->Nts->getWritableBuffer(f_input_grad, torch::DeviceType::CPU);
    ValueType *f_output_grad_buffer =
      graph_->Nts->getWritableBuffer(f_output_grad, torch::DeviceType::CPU);
    feature_size/=2;
      graph_->local_vertex_operation<int, ValueType>( // For EACH Vertex
      [&](VertexId vtx, CSC_segment_pinned *subgraph, VertexId recv_id) {
        // iterate the incoming edge for vtx
        for (long eid = subgraph->column_offset[vtx];
             eid < subgraph->column_offset[vtx + 1]; eid++) {
          VertexId src = subgraph->row_indices[eid];
            nts_acc(f_input_grad_buffer + src * feature_size,
                    f_output_grad_buffer + (feature_size * eid * 2), feature_size);
            nts_acc(f_input_grad_buffer + vtx * feature_size,
                    f_output_grad_buffer + (feature_size * (eid * 2 + 1)),
                     feature_size);
        }
      },
      subgraphs, feature_size, active_);
      return f_input_grad;
  }    

};

class SingleCPUSrcScatterOp : public ntsGraphOp{
public:
  std::vector<CSC_segment_pinned *> subgraphs;
  
  SingleCPUSrcScatterOp(PartitionedGraph *partitioned_graph,VertexSubset *active)
      : ntsGraphOp(partitioned_graph, active) {
    subgraphs = partitioned_graph->graph_chunks;
  }
  NtsVar forward(NtsVar &f_input){
    int feature_size = f_input.size(1);
    NtsVar f_output=graph_->Nts->NewKeyTensor({graph_->gnnctx->l_e_num, 
                feature_size},torch::DeviceType::CPU);
    ValueType *f_input_buffer =
      graph_->Nts->getWritableBuffer(f_input, torch::DeviceType::CPU);
    ValueType *f_output_buffer =
      graph_->Nts->getWritableBuffer(f_output, torch::DeviceType::CPU);            
    graph_->local_vertex_operation<int, ValueType>( // For EACH Vertex
      [&](VertexId vtx, CSC_segment_pinned *subgraph, VertexId recv_id) {
        // iterate the incoming edge for vtx
        for (long eid = subgraph->column_offset[vtx];
             eid < subgraph->column_offset[vtx + 1]; eid++) {
          VertexId src = subgraph->row_indices[eid];
          nts_copy(f_output_buffer, eid, f_input_buffer, src, feature_size,1);
        }
      },
      subgraphs, feature_size, active_);            
    return f_output;
  }
  
  NtsVar backward(NtsVar &f_output_grad){
      int feature_size=f_output_grad.size(1);
    NtsVar f_input_grad=graph_->Nts->NewLeafTensor({graph_->gnnctx->l_v_num, 
                feature_size},torch::DeviceType::CPU);
              
    ValueType *f_input_grad_buffer =
      graph_->Nts->getWritableBuffer(f_input_grad, torch::DeviceType::CPU);
    ValueType *f_output_grad_buffer =
      graph_->Nts->getWritableBuffer(f_output_grad, torch::DeviceType::CPU);
    graph_->local_vertex_operation<int, ValueType>( // For EACH Vertex
      [&](VertexId vtx, CSC_segment_pinned *subgraph, VertexId recv_id) {
        // iterate the incoming edge for vtx
        for (long eid = subgraph->column_offset[vtx];
             eid < subgraph->column_offset[vtx + 1]; eid++) {
          VertexId src = subgraph->row_indices[eid];
            nts_acc(f_output_grad_buffer + (feature_size * eid),
                    f_input_grad_buffer + src * feature_size,
                     feature_size);
        }
      },
      subgraphs, feature_size, active_);
      return f_input_grad;
  }    

};

class SingleCPUDstAggregateOp : public ntsGraphOp{
public:
  std::vector<CSC_segment_pinned *> subgraphs;
  
  SingleCPUDstAggregateOp(PartitionedGraph *partitioned_graph,VertexSubset *active)
      : ntsGraphOp(partitioned_graph, active) {
    subgraphs = partitioned_graph->graph_chunks;
  }
  NtsVar forward(NtsVar &f_input){// input edge  output vertex
    int feature_size = f_input.size(1);
    NtsVar f_output=graph_->Nts->NewKeyTensor({graph_->gnnctx->l_v_num, 
                feature_size},torch::DeviceType::CPU);
    ValueType *f_input_buffer =
      graph_->Nts->getWritableBuffer(f_input, torch::DeviceType::CPU);
    ValueType *f_output_buffer =
      graph_->Nts->getWritableBuffer(f_output, torch::DeviceType::CPU);  
    
    graph_->local_vertex_operation<int, ValueType>( // For EACH Vertex
      [&](VertexId vtx, CSC_segment_pinned *subgraph, VertexId recv_id) {
        // iterate the incoming edge for vtx
        for (long eid = subgraph->column_offset[vtx];
             eid < subgraph->column_offset[vtx + 1]; eid++) {
          VertexId src = subgraph->row_indices[eid];
          nts_acc(f_output_buffer + vtx * feature_size, 
                  f_input_buffer + eid * feature_size, feature_size);
        }
      },
      subgraphs, feature_size, active_);            
    return f_output;
  }
  
  NtsVar backward(NtsVar &f_output_grad){// input vtx grad; output edge grad
      int feature_size=f_output_grad.size(1);
    NtsVar f_input_grad=graph_->Nts->NewLeafTensor({graph_->gnnctx->l_e_num, 
                feature_size},torch::DeviceType::CPU);
              
    ValueType *f_input_grad_buffer =
      graph_->Nts->getWritableBuffer(f_input_grad, torch::DeviceType::CPU);
    ValueType *f_output_grad_buffer =
      graph_->Nts->getWritableBuffer(f_output_grad, torch::DeviceType::CPU);
    graph_->local_vertex_operation<int, ValueType>( // For EACH Vertex
      [&](VertexId vtx, CSC_segment_pinned *subgraph, VertexId recv_id) {
        // iterate the incoming edge for vtx
        for (long eid = subgraph->column_offset[vtx];
             eid < subgraph->column_offset[vtx + 1]; eid++) {
          VertexId src = subgraph->row_indices[eid];
            nts_acc(f_input_grad_buffer+ (feature_size * eid),
                    f_output_grad_buffer + vtx * feature_size,
                     feature_size);
        }
      },
      subgraphs, feature_size, active_);
      return f_input_grad;
  }    

};

class SingleCPUDstAggregateOpMin : public ntsGraphOp{
public:
  std::vector<CSC_segment_pinned *> subgraphs;
  VertexId* record;
  
  SingleCPUDstAggregateOpMin(PartitionedGraph *partitioned_graph,VertexSubset *active)
      : ntsGraphOp(partitioned_graph, active) {
    subgraphs = partitioned_graph->graph_chunks;
  }
  ~SingleCPUDstAggregateOpMin(){
      delete [] record;
  }
  NtsVar forward(NtsVar &f_input){// input edge  output vertex
    int feature_size = f_input.size(1);
    
    record=new VertexId(partitioned_graph_->owned_vertices*feature_size);
    NtsVar f_output=graph_->Nts->NewKeyTensor({graph_->gnnctx->l_v_num, 
                feature_size},torch::DeviceType::CPU);
    ValueType *f_input_buffer =
      graph_->Nts->getWritableBuffer(f_input, torch::DeviceType::CPU);
    ValueType *f_output_buffer =
      graph_->Nts->getWritableBuffer(f_output, torch::DeviceType::CPU);  
    
    graph_->local_vertex_operation<int, ValueType>( // For EACH Vertex
      [&](VertexId vtx, CSC_segment_pinned *subgraph, VertexId recv_id) {
        // iterate the incoming edge for vtx
        for (long eid = subgraph->column_offset[vtx];
             eid < subgraph->column_offset[vtx + 1]; eid++) {
          VertexId src = subgraph->row_indices[eid];
          nts_min(f_output_buffer+ vtx * feature_size,
                   f_input_buffer + eid * feature_size, 
                    record + vtx * feature_size,
                  feature_size,eid);
        }
      },
      subgraphs, feature_size, active_);            
    return f_output;
  }
  
  NtsVar backward(NtsVar &f_output_grad){// input vtx grad; output edge grad
      int feature_size=f_output_grad.size(1);
    NtsVar f_input_grad=graph_->Nts->NewLeafTensor({graph_->gnnctx->l_e_num, 
                feature_size},torch::DeviceType::CPU);
              
    ValueType *f_input_grad_buffer =
      graph_->Nts->getWritableBuffer(f_input_grad, torch::DeviceType::CPU);
    ValueType *f_output_grad_buffer =
      graph_->Nts->getWritableBuffer(f_output_grad, torch::DeviceType::CPU);
    graph_->local_vertex_operation<int, ValueType>( // For EACH Vertex
      [&](VertexId vtx, CSC_segment_pinned *subgraph, VertexId recv_id) {
        // iterate the incoming edge for vtx
        nts_assign(f_input_grad_buffer, f_output_grad_buffer+feature_size*vtx,
                record+feature_size*vtx, feature_size); 
//        for (long eid = subgraph->column_offset[vtx];
//             eid < subgraph->column_offset[vtx + 1]; eid++) {
//          VertexId src = subgraph->row_indices[eid];
//          
////            nts_acc(f_input_grad_buffer+ (feature_size * eid),
////                    f_output_grad_buffer + vtx * feature_size,
////                     feature_size);
//        }
      },
      subgraphs, feature_size, active_);
      return f_input_grad;
  }    

};

class SingleCPUDstAggregateOpMax : public ntsGraphOp{
public:
  std::vector<CSC_segment_pinned *> subgraphs;
  VertexId* record;
  
  SingleCPUDstAggregateOpMax(PartitionedGraph *partitioned_graph,VertexSubset *active)
      : ntsGraphOp(partitioned_graph, active) {
    subgraphs = partitioned_graph->graph_chunks;
  }
  ~SingleCPUDstAggregateOpMax(){
      delete [] record;
  }
  NtsVar forward(NtsVar &f_input){// input edge  output vertex
    int feature_size = f_input.size(1);
    
    record=new VertexId(partitioned_graph_->owned_vertices*feature_size);
    NtsVar f_output=graph_->Nts->NewKeyTensor({graph_->gnnctx->l_v_num, 
                feature_size},torch::DeviceType::CPU);
    ValueType *f_input_buffer =
      graph_->Nts->getWritableBuffer(f_input, torch::DeviceType::CPU);
    ValueType *f_output_buffer =
      graph_->Nts->getWritableBuffer(f_output, torch::DeviceType::CPU);  
    
    graph_->local_vertex_operation<int, ValueType>( // For EACH Vertex
      [&](VertexId vtx, CSC_segment_pinned *subgraph, VertexId recv_id) {
        // iterate the incoming edge for vtx
        for (long eid = subgraph->column_offset[vtx];
             eid < subgraph->column_offset[vtx + 1]; eid++) {
          VertexId src = subgraph->row_indices[eid];
          nts_max(f_output_buffer+ vtx * feature_size,
                   f_input_buffer + eid * feature_size, 
                    record + vtx * feature_size,
                  feature_size,eid);
        }
      },
      subgraphs, feature_size, active_);            
    return f_output;
  }
  
  NtsVar backward(NtsVar &f_output_grad){// input vtx grad; output edge grad
      int feature_size=f_output_grad.size(1);
    NtsVar f_input_grad=graph_->Nts->NewLeafTensor({graph_->gnnctx->l_e_num, 
                feature_size},torch::DeviceType::CPU);
              
    ValueType *f_input_grad_buffer =
      graph_->Nts->getWritableBuffer(f_input_grad, torch::DeviceType::CPU);
    ValueType *f_output_grad_buffer =
      graph_->Nts->getWritableBuffer(f_output_grad, torch::DeviceType::CPU);
    graph_->local_vertex_operation<int, ValueType>( // For EACH Vertex
      [&](VertexId vtx, CSC_segment_pinned *subgraph, VertexId recv_id) {
        // iterate the incoming edge for vtx
        nts_assign(f_input_grad_buffer, f_output_grad_buffer+feature_size*vtx,
                record+feature_size*vtx, feature_size); 
//        for (long eid = subgraph->column_offset[vtx];
//             eid < subgraph->column_offset[vtx + 1]; eid++) {
//          VertexId src = subgraph->row_indices[eid];
//          
////            nts_acc(f_input_grad_buffer+ (feature_size * eid),
////                    f_output_grad_buffer + vtx * feature_size,
////                     feature_size);
//        }
      },
      subgraphs, feature_size, active_);
      return f_input_grad;
  }    

};


class SingleEdgeSoftMax : public ntsGraphOp{
public:
  std::vector<CSC_segment_pinned *> subgraphs;
  NtsVar IntermediateResult;
  
  SingleEdgeSoftMax(PartitionedGraph *partitioned_graph,VertexSubset *active)
      : ntsGraphOp(partitioned_graph, active) {
    subgraphs = partitioned_graph->graph_chunks;
  }
  NtsVar forward(NtsVar &f_input_){// input i_msg  output o_msg
     //NtsVar f_input_=f_input.detach();
    int feature_size = f_input_.size(1);
    NtsVar f_output=graph_->Nts->NewKeyTensor({graph_->gnnctx->l_e_num, 
                feature_size},torch::DeviceType::CPU);
    ValueType *f_input_buffer =
      graph_->Nts->getWritableBuffer(f_input_, torch::DeviceType::CPU);
    ValueType *f_output_buffer =
      graph_->Nts->getWritableBuffer(f_output, torch::DeviceType::CPU);
    
        graph_->local_vertex_operation<int, ValueType>( // For EACH Vertex
        [&](VertexId vtx, CSC_segment_pinned *subgraph, VertexId recv_id) {
          long eid_start = subgraph->column_offset[vtx];
          long eid_end = subgraph->column_offset[vtx + 1];
          assert(eid_end <= graph_->edges);
          assert(eid_start >= 0);
          NtsVar d = f_input_.slice(0, eid_start, eid_end, 1).softmax(0);
          ValueType *d_buffer =
          graph_->Nts->getWritableBuffer(d, torch::DeviceType::CPU);      
          nts_copy(f_output_buffer, eid_start, d_buffer, 
                  0, feature_size,(eid_end-eid_start));   
        },
        subgraphs, f_input_.size(1), this->active_);
    
    IntermediateResult=f_output;
          
    return f_output;
  }
  
  NtsVar backward(NtsVar &f_output_grad){// input vtx grad; output edge grad
    int feature_size=f_output_grad.size(1);
    NtsVar f_input_grad=graph_->Nts->NewLeafTensor({graph_->gnnctx->l_e_num, 
                feature_size},torch::DeviceType::CPU);
    ValueType *f_input_grad_buffer =
      graph_->Nts->getWritableBuffer(f_input_grad, torch::DeviceType::CPU);
    
    graph_->local_vertex_operation<int, ValueType>( // For EACH Vertex
        [&](VertexId vtx, CSC_segment_pinned *subgraph, VertexId recv_id) {
          long eid_start = subgraph->column_offset[vtx];
          long eid_end = subgraph->column_offset[vtx + 1];
          assert(eid_end <= graph_->edges);
          assert(eid_start >= 0);
          NtsVar d   = f_output_grad.slice(0, eid_start, eid_end, 1);
          NtsVar imr =IntermediateResult.slice(0, eid_start, eid_end, 1);
          //s4=(s2*s1)-(s2)*(s2.t().mm(s1)); 
          NtsVar d_o =(imr*d)-imr*(d.t().mm(imr)); 
          ValueType *d_o_buffer =
          graph_->Nts->getWritableBuffer(d_o, torch::DeviceType::CPU);
          nts_copy(f_input_grad_buffer, eid_start, d_o_buffer, 
                  0, feature_size,(eid_end-eid_start));
        },
        subgraphs, f_output_grad.size(1), this->active_);
      return f_input_grad;
  }    

};



} // namespace graphop
} // namespace nts

#endif

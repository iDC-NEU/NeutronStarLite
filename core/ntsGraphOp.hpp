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
#ifndef NTSGRAPHOP_HPP
#define NTSGRAPHOP_HPP
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

#include "ntsCPUFusedGraphOp.hpp"
#include "ntsDistCPUGraphOp.hpp"
#include "ntsSingleCPUGraphOp.hpp"
#include "ntsMiniBatchGraphOp.hpp"

#if CUDA_ENABLE
#include "ntsDistGPUFusedGraphOp.hpp"
#include "ntsDistGPUGraphOp.hpp"
#include "ntsSingleGPUFusedGraphOp.hpp"
#endif
//#include "ntsSubLinearNNOP.hpp"
//#include "ntsNNOP.hpp"

//namespace nts {
//namespace op {
//
////class ntsGraphOp {
////public:
////  Graph<Empty> *graph_;
////  VertexSubset *active_;
////  ntsGraphOp() { ; }
////  ntsGraphOp(Graph<Empty> *graph, VertexSubset *active) {
////    graph_ = graph;
////    active_ = active;
////  }
////  virtual NtsVar &forward(NtsVar &input) = 0;
////  virtual NtsVar backward(NtsVar &output_grad) = 0;
////};
//    
//#if CUDA_ENABLE    
//class ForwardGPUfuseOp : public ntsGraphOp{
//public:
//  std::vector<CSC_segment_pinned *> subgraphs;
//
//  ForwardGPUfuseOp(PartitionedGraph *partitioned_graph,VertexSubset *active)
//      : ntsGraphOp(partitioned_graph, active) {
//    subgraphs = partitioned_graph->graph_chunks;
//  }
//  NtsVar forward(NtsVar &f_input){
//        int feature_size = f_input.size(1);
//  NtsVar f_input_cpu = f_input.cpu();
//  NtsVar f_output=graph_->Nts->NewKeyTensor(f_input,torch::DeviceType::CUDA);
//  ValueType *f_input_cpu_buffered = f_input_cpu.accessor<ValueType, 2>().data();
//
//  { // original communication
//    graph_->sync_compute_decoupled<int, ValueType>(
//        f_input, subgraphs,
//        [&](VertexId src) {
//          graph_->NtsComm->emit_buffer(
//              src, f_input_cpu_buffered + (src - graph_->gnnctx->p_v_s) * feature_size,
//              feature_size);
//        },
//        f_output, feature_size);
//  }
//  return f_output;
//  }
//  
//  NtsVar backward(NtsVar &f_output_grad){
//      int feature_size = f_output_grad.size(1);
//      NtsVar f_input_grad=graph_->Nts->NewKeyTensor(f_output_grad,torch::DeviceType::CUDA);
//  // if (!selective)
//  {
//    graph_->compute_sync_decoupled<int, ValueType>(
//        f_output_grad, subgraphs,
//        [&](VertexId src, VertexAdjList<Empty> outgoing_adj) { // pull
//          graph_->NtsComm->emit_buffer(
//              src, graph_->output_cpu_buffer + (src)*feature_size,
//              feature_size);
//        },
//        f_input_grad, feature_size);
//  }
//      return f_input_grad;
//  }
//};
//
//class ForwardSingleGPUfuseOp : public ntsGraphOp{
//public:
//  std::vector<CSC_segment_pinned *> subgraphs;
//  
//  ForwardSingleGPUfuseOp(PartitionedGraph *partitioned_graph,VertexSubset *active)
//      : ntsGraphOp(partitioned_graph, active) {
//    subgraphs = partitioned_graph->graph_chunks;
//  }
//  NtsVar forward(NtsVar &f_input){
//    int feature_size = f_input.size(1);
//    NtsVar f_output=graph_->Nts->NewKeyTensor(f_input,torch::DeviceType::CUDA);
//    graph_->forward_single<int, ValueType>(f_input, subgraphs, f_output, feature_size);
//    return f_output;
//  }
//  
//  NtsVar backward(NtsVar &f_output_grad){
//    int feature_size = f_output_grad.size(1);
//    NtsVar f_input_grad=graph_->Nts->NewKeyTensor(f_output_grad,torch::DeviceType::CUDA);
//    graph_->backward_single<int, ValueType>(f_output_grad, subgraphs, 
//            f_input_grad, feature_size);
//      return f_input_grad;
//  }    
//
//};
//#endif
//class SingleCPUSrcDstScatterOp : public ntsGraphOp{
//public:
//  std::vector<CSC_segment_pinned *> subgraphs;
//  
//  SingleCPUSrcDstScatterOp(PartitionedGraph *partitioned_graph,VertexSubset *active)
//      : ntsGraphOp(partitioned_graph, active) {
//    subgraphs = partitioned_graph->graph_chunks;
//  }
//  NtsVar forward(NtsVar &f_input){
//    int feature_size = f_input.size(1);
//    NtsVar f_output=graph_->Nts->NewKeyTensor({graph_->gnnctx->l_e_num, 
//                2*feature_size},torch::DeviceType::CPU);
//    ValueType *f_input_buffer =
//      graph_->Nts->getWritableBuffer(f_input, torch::DeviceType::CPU);
//    ValueType *f_output_buffer =
//      graph_->Nts->getWritableBuffer(f_output, torch::DeviceType::CPU);            
//  graph_->local_vertex_operation<int, ValueType>( // For EACH Vertex
//      [&](VertexId vtx, CSC_segment_pinned *subgraph, VertexId recv_id) {
//        // iterate the incoming edge for vtx
//        for (long eid = subgraph->column_offset[vtx];
//             eid < subgraph->column_offset[vtx + 1]; eid++) {
//          VertexId src = subgraph->row_indices[eid];
//          nts_copy(f_output_buffer, eid * 2, f_input_buffer, src, feature_size,1);
//          nts_copy(f_output_buffer, eid * 2 + 1, f_input_buffer, vtx, feature_size,1);
//        }
//      },
//      subgraphs, feature_size, active_);            
//    return f_output;
//  }
//  
//  NtsVar backward(NtsVar &f_output_grad){
//      int feature_size=f_output_grad.size(1);
//                     assert(feature_size%2==0); 
//    NtsVar f_input_grad=graph_->Nts->NewLeafTensor({graph_->gnnctx->l_v_num, 
//                feature_size/2},torch::DeviceType::CPU);
//              
//    ValueType *f_input_grad_buffer =
//      graph_->Nts->getWritableBuffer(f_input_grad, torch::DeviceType::CPU);
//    ValueType *f_output_grad_buffer =
//      graph_->Nts->getWritableBuffer(f_output_grad, torch::DeviceType::CPU);
//    feature_size/=2;
//      graph_->local_vertex_operation<int, ValueType>( // For EACH Vertex
//      [&](VertexId vtx, CSC_segment_pinned *subgraph, VertexId recv_id) {
//        // iterate the incoming edge for vtx
//        for (long eid = subgraph->column_offset[vtx];
//             eid < subgraph->column_offset[vtx + 1]; eid++) {
//          VertexId src = subgraph->row_indices[eid];
//            nts_acc(f_input_grad_buffer + src * feature_size,
//                    f_output_grad_buffer + (feature_size * eid * 2), feature_size);
//            nts_acc(f_input_grad_buffer + vtx * feature_size,
//                    f_output_grad_buffer + (feature_size * (eid * 2 + 1)),
//                     feature_size);
//        }
//      },
//      subgraphs, feature_size, active_);
//      return f_input_grad;
//  }    
//
//};
//
//class SingleCPUSrcScatterOp : public ntsGraphOp{
//public:
//  std::vector<CSC_segment_pinned *> subgraphs;
//  
//  SingleCPUSrcScatterOp(PartitionedGraph *partitioned_graph,VertexSubset *active)
//      : ntsGraphOp(partitioned_graph, active) {
//    subgraphs = partitioned_graph->graph_chunks;
//  }
//  NtsVar forward(NtsVar &f_input){
//    int feature_size = f_input.size(1);
//    NtsVar f_output=graph_->Nts->NewKeyTensor({graph_->gnnctx->l_e_num, 
//                feature_size},torch::DeviceType::CPU);
//    ValueType *f_input_buffer =
//      graph_->Nts->getWritableBuffer(f_input, torch::DeviceType::CPU);
//    ValueType *f_output_buffer =
//      graph_->Nts->getWritableBuffer(f_output, torch::DeviceType::CPU);            
//    graph_->local_vertex_operation<int, ValueType>( // For EACH Vertex
//      [&](VertexId vtx, CSC_segment_pinned *subgraph, VertexId recv_id) {
//        // iterate the incoming edge for vtx
//        for (long eid = subgraph->column_offset[vtx];
//             eid < subgraph->column_offset[vtx + 1]; eid++) {
//          VertexId src = subgraph->row_indices[eid];
//          nts_copy(f_output_buffer, eid, f_input_buffer, src, feature_size,1);
//        }
//      },
//      subgraphs, feature_size, active_);            
//    return f_output;
//  }
//  
//  NtsVar backward(NtsVar &f_output_grad){
//      int feature_size=f_output_grad.size(1);
//    NtsVar f_input_grad=graph_->Nts->NewLeafTensor({graph_->gnnctx->l_v_num, 
//                feature_size},torch::DeviceType::CPU);
//              
//    ValueType *f_input_grad_buffer =
//      graph_->Nts->getWritableBuffer(f_input_grad, torch::DeviceType::CPU);
//    ValueType *f_output_grad_buffer =
//      graph_->Nts->getWritableBuffer(f_output_grad, torch::DeviceType::CPU);
//    graph_->local_vertex_operation<int, ValueType>( // For EACH Vertex
//      [&](VertexId vtx, CSC_segment_pinned *subgraph, VertexId recv_id) {
//        // iterate the incoming edge for vtx
//        for (long eid = subgraph->column_offset[vtx];
//             eid < subgraph->column_offset[vtx + 1]; eid++) {
//          VertexId src = subgraph->row_indices[eid];
//            nts_acc(f_output_grad_buffer + (feature_size * eid),
//                    f_input_grad_buffer + src * feature_size,
//                     feature_size);
//        }
//      },
//      subgraphs, feature_size, active_);
//      return f_input_grad;
//  }    
//
//};
//
//class SingleCPUDstAggregateOp : public ntsGraphOp{
//public:
//  std::vector<CSC_segment_pinned *> subgraphs;
//  
//  SingleCPUDstAggregateOp(PartitionedGraph *partitioned_graph,VertexSubset *active)
//      : ntsGraphOp(partitioned_graph, active) {
//    subgraphs = partitioned_graph->graph_chunks;
//  }
//  NtsVar forward(NtsVar &f_input){// input edge  output vertex
//    int feature_size = f_input.size(1);
//    NtsVar f_output=graph_->Nts->NewKeyTensor({graph_->gnnctx->l_v_num, 
//                feature_size},torch::DeviceType::CPU);
//    ValueType *f_input_buffer =
//      graph_->Nts->getWritableBuffer(f_input, torch::DeviceType::CPU);
//    ValueType *f_output_buffer =
//      graph_->Nts->getWritableBuffer(f_output, torch::DeviceType::CPU);  
//    
//    graph_->local_vertex_operation<int, ValueType>( // For EACH Vertex
//      [&](VertexId vtx, CSC_segment_pinned *subgraph, VertexId recv_id) {
//        // iterate the incoming edge for vtx
//        for (long eid = subgraph->column_offset[vtx];
//             eid < subgraph->column_offset[vtx + 1]; eid++) {
//          VertexId src = subgraph->row_indices[eid];
//          nts_acc(f_output_buffer + vtx * feature_size, 
//                  f_input_buffer + eid * feature_size, feature_size);
//        }
//      },
//      subgraphs, feature_size, active_);            
//    return f_output;
//  }
//  
//  NtsVar backward(NtsVar &f_output_grad){// input vtx grad; output edge grad
//      int feature_size=f_output_grad.size(1);
//    NtsVar f_input_grad=graph_->Nts->NewLeafTensor({graph_->gnnctx->l_e_num, 
//                feature_size},torch::DeviceType::CPU);
//              
//    ValueType *f_input_grad_buffer =
//      graph_->Nts->getWritableBuffer(f_input_grad, torch::DeviceType::CPU);
//    ValueType *f_output_grad_buffer =
//      graph_->Nts->getWritableBuffer(f_output_grad, torch::DeviceType::CPU);
//    graph_->local_vertex_operation<int, ValueType>( // For EACH Vertex
//      [&](VertexId vtx, CSC_segment_pinned *subgraph, VertexId recv_id) {
//        // iterate the incoming edge for vtx
//        for (long eid = subgraph->column_offset[vtx];
//             eid < subgraph->column_offset[vtx + 1]; eid++) {
//          VertexId src = subgraph->row_indices[eid];
//            nts_acc(f_input_grad_buffer+ (feature_size * eid),
//                    f_output_grad_buffer + vtx * feature_size,
//                     feature_size);
//        }
//      },
//      subgraphs, feature_size, active_);
//      return f_input_grad;
//  }    
//
//};
//class SingleEdgeSoftMax : public ntsGraphOp{
//public:
//  std::vector<CSC_segment_pinned *> subgraphs;
//  NtsVar IntermediateResult;
//  
//  SingleEdgeSoftMax(PartitionedGraph *partitioned_graph,VertexSubset *active)
//      : ntsGraphOp(partitioned_graph, active) {
//    subgraphs = partitioned_graph->graph_chunks;
//  }
//  NtsVar forward(NtsVar &f_input_){// input i_msg  output o_msg
//     //NtsVar f_input_=f_input.detach();
//    int feature_size = f_input_.size(1);
//    NtsVar f_output=graph_->Nts->NewKeyTensor({graph_->gnnctx->l_e_num, 
//                feature_size},torch::DeviceType::CPU);
//    ValueType *f_input_buffer =
//      graph_->Nts->getWritableBuffer(f_input_, torch::DeviceType::CPU);
//    ValueType *f_output_buffer =
//      graph_->Nts->getWritableBuffer(f_output, torch::DeviceType::CPU);
//    
//        graph_->local_vertex_operation<int, ValueType>( // For EACH Vertex
//        [&](VertexId vtx, CSC_segment_pinned *subgraph, VertexId recv_id) {
//          long eid_start = subgraph->column_offset[vtx];
//          long eid_end = subgraph->column_offset[vtx + 1];
//          assert(eid_end <= graph_->edges);
//          assert(eid_start >= 0);
//          NtsVar d = f_input_.slice(0, eid_start, eid_end, 1).softmax(0);
//          ValueType *d_buffer =
//          graph_->Nts->getWritableBuffer(d, torch::DeviceType::CPU);      
//          nts_copy(f_output_buffer, eid_start, d_buffer, 
//                  0, feature_size,(eid_end-eid_start));   
//        },
//        subgraphs, f_input_.size(1), this->active_);
//    
//    IntermediateResult=f_output;
//          
//    return f_output;
//  }
//  
//  NtsVar backward(NtsVar &f_output_grad){// input vtx grad; output edge grad
//    int feature_size=f_output_grad.size(1);
//    NtsVar f_input_grad=graph_->Nts->NewLeafTensor({graph_->gnnctx->l_e_num, 
//                feature_size},torch::DeviceType::CPU);
//    ValueType *f_input_grad_buffer =
//      graph_->Nts->getWritableBuffer(f_input_grad, torch::DeviceType::CPU);
//    
//    graph_->local_vertex_operation<int, ValueType>( // For EACH Vertex
//        [&](VertexId vtx, CSC_segment_pinned *subgraph, VertexId recv_id) {
//          long eid_start = subgraph->column_offset[vtx];
//          long eid_end = subgraph->column_offset[vtx + 1];
//          assert(eid_end <= graph_->edges);
//          assert(eid_start >= 0);
//          NtsVar d   = f_output_grad.slice(0, eid_start, eid_end, 1);
//          NtsVar imr =IntermediateResult.slice(0, eid_start, eid_end, 1);
//          //s4=(s2*s1)-(s2)*(s2.t().mm(s1)); 
//          NtsVar d_o =(imr*d)-imr*(d.t().mm(imr)); 
//          ValueType *d_o_buffer =
//          graph_->Nts->getWritableBuffer(d_o, torch::DeviceType::CPU);
//          nts_copy(f_input_grad_buffer, eid_start, d_o_buffer, 
//                  0, feature_size,(eid_end-eid_start));
//        },
//        subgraphs, f_output_grad.size(1), this->active_);
//      return f_input_grad;
//  }    
//
//};
//
//
//class ForwardCPUfuseOp : public ntsGraphOp {
//public:
//  std::vector<CSC_segment_pinned *> subgraphs;
//  ForwardCPUfuseOp(PartitionedGraph *partitioned_graph,VertexSubset *active)
//      : ntsGraphOp(partitioned_graph, active) {
//    subgraphs = partitioned_graph->graph_chunks;
//  }
//  NtsVar forward(NtsVar &f_input) {
//    //f_input = input;
//    NtsVar f_output = graph_->Nts->NewKeyTensor(f_input, torch::DeviceType::CPU);
//    ValueType *f_input_buffer =
//        graph_->Nts->getWritableBuffer(f_input, torch::DeviceType::CPU);
//    ValueType *f_output_buffer =
//        graph_->Nts->getWritableBuffer(f_output, torch::DeviceType::CPU);
//    memset(f_output_buffer, 0,
//           sizeof(ValueType) * f_input.size(0) * f_input.size(1));
//    int feature_size = f_input.size(1);
//    graph_->process_edges_forward_decoupled_mutisockets<int, ValueType>(
//        [&](VertexId src, int current_send_partition) {
//          if (graph_->rtminfo->lock_free) {
//            VertexId src_trans = src - graph_->gnnctx->p_v_s;
//            // check whether this vertex is necessary to send to
//            // current_send_partition
//            if (partitioned_graph_->has_mirror_at(current_send_partition,src)
////              subgraphs[current_send_partition]->get_forward_active(src_trans)
//                ) {
//              // get the index where we shall place the data
//              // and invoke emit_buffer_lock_free to send messages
//              VertexId write_index =
//                  subgraphs[current_send_partition]
//                      ->forward_multisocket_message_index[src_trans];
//              graph_->NtsComm->emit_buffer_lock_free(
//                  src,
//                  f_input_buffer + (src - graph_->gnnctx->p_v_s) * feature_size,
//                  write_index, feature_size);
//            }
//          } else {
//            // send to mirror directly
//            graph_->NtsComm->emit_buffer(
//                src,
//                f_input_buffer + (src - graph_->gnnctx->p_v_s) * feature_size,
//                feature_size);
//          }
//        },
//        // sparse slot.
//        // accumulate incoming feature for dst
//        // recv_id represent the partition ID corresponding to current subgraph
//        [&](VertexId dst, CSC_segment_pinned *subgraph,
//            MessageBuffer **recv_buffer, std::vector<VertexIndex> &src_index,
//            VertexId recv_id) {
//          VertexId dst_trans =
//              dst - graph_->partition_offset[graph_->partition_id];
//          // for every vertex, accumulate the incoming feature though iterating
//          // column vertices
//          for (long idx = subgraph->column_offset[dst_trans];
//               idx < subgraph->column_offset[dst_trans + 1]; idx++) {
//            VertexId src = subgraph->row_indices[idx];
//            VertexId src_trans = src - graph_->partition_offset[recv_id];
//            // fetch input from recv buffer
//            // bufferIndex indicate which socket we've placed the data
//            // positionIndex indicate the index of the buffer of that socket
//            ValueType *local_input =
//                (ValueType *)(recv_buffer[src_index[src_trans].bufferIndex]
//                                  ->data +
//                              graph_->sizeofM<ValueType>(feature_size) *
//                                  src_index[src_trans].positionIndex +
//                              sizeof(VertexId));
//            ValueType *local_output =
//                f_output_buffer + dst_trans * feature_size;
//            nts_comp(local_output, local_input, nts_norm_degree(graph_, src, dst),
//                 feature_size);
//          }
//        },
//        subgraphs, feature_size, active_);
//    return f_output;
//  }
//  NtsVar backward(NtsVar &f_output_grad) {
//    NtsVar f_input_grad=graph_->Nts->NewLeafTensor(f_output_grad, torch::DeviceType::CPU);
//  ValueType *output_grad_buffer =
//      graph_->Nts->getWritableBuffer(f_output_grad, torch::DeviceType::CPU);
//  ValueType *input_grad_buffer =
//      graph_->Nts->getWritableBuffer(f_input_grad, torch::DeviceType::CPU);
//  memset(input_grad_buffer, 0, sizeof(ValueType) * f_output_grad.size(0) * f_output_grad.size(1));
//  // int feature_size=graph_->gnnctx->layer_size[graph_->rtminfo->curr_layer];
//  int feature_size = f_output_grad.size(1);
//  ValueType *output_buffer = new ValueType[feature_size * graph_->threads];
//  graph_->process_edges_backward_decoupled_multisockets<
//      int, ValueType>( // For EACH Vertex
//                       // Processing
//      [&](VertexId src, VertexAdjList<Empty> outgoing_adj, VertexId thread_id,
//          VertexId recv_id, VertexId socketId) { // pull
//        ValueType *local_output_buffer =
//            output_buffer + feature_size * thread_id;
//        memset(local_output_buffer, 0, sizeof(ValueType) * feature_size);
//        VertexId src_trans = src - graph_->partition_offset[recv_id];
//
//        // iterate the outgoing vertices and combine the gradients
//        for (long d_idx = subgraphs[recv_id]->row_offset[src_trans];
//             d_idx < subgraphs[recv_id]->row_offset[src_trans + 1]; d_idx++) {
//          VertexId dst = subgraphs[recv_id]->column_indices[d_idx];
//
//          // FIXME: will this work?
//          if ((dst < graph_->local_partition_offset[socketId]) ||
//              (dst >= graph_->local_partition_offset[socketId + 1])) {
//            continue;
//          }
//          VertexId dst_trans = dst - graph_->gnnctx->p_v_s;
//          ValueType *local_input_buffer =
//             output_grad_buffer + (dst_trans)*feature_size;
//          nts_comp(local_output_buffer, local_input_buffer, nts_norm_degree(graph_,src, dst),
//               feature_size);
//        }
//        if (graph_->rtminfo->lock_free) {
//          if (subgraphs[recv_id]->src_get_active(src)) {
//            VertexId write_index =
//                subgraphs[recv_id]
//                    ->backward_multisocket_message_index[src_trans]
//                    .vertexSocketPosition[socketId];
//            graph_->NtsComm->emit_buffer_lock_free(src, local_output_buffer,
//                                                   write_index, feature_size);
//          }
//        } else {
//          graph_->NtsComm->emit_buffer(src, local_output_buffer, feature_size);
//        }
//      },
//      [&](VertexId src, ValueType *msg) {
//        nts_acc(input_grad_buffer + (src - graph_->gnnctx->p_v_s) * feature_size, msg, feature_size);
//        return 1;
//      },
//      feature_size, active_);
//  delete[] output_buffer;
//  return f_input_grad;
//  
//  }
//};
//
//class DistGetDepNbrOp : public ntsGraphOp{
//public:
//  std::vector<CSC_segment_pinned *> subgraphs;
//  
//  DistGetDepNbrOp(PartitionedGraph *partitioned_graph,VertexSubset *active)
//      : ntsGraphOp(partitioned_graph, active) {
//    subgraphs = partitioned_graph->graph_chunks;
//  }
//  NtsVar forward(NtsVar &f_input){// input edge  output vertex
//    int feature_size = f_input.size(1);
//    //LOG_INFO("owned_mirrors (%d)",partitioned_graph_->owned_mirrors);
//    NtsVar f_output=graph_->Nts->NewKeyTensor({partitioned_graph_->owned_mirrors, 
//                feature_size},torch::DeviceType::CPU);
//    ValueType *f_input_buffer =
//      graph_->Nts->getWritableBuffer(f_input, torch::DeviceType::CPU);
//    ValueType *f_output_buffer =
//      graph_->Nts->getWritableBuffer(f_output, torch::DeviceType::CPU);  
//    //LOG_INFO("owned_mirrors (%d)",partitioned_graph_->owned_mirrors);    
//      graph_->get_from_dep_neighbor_mutisockets<int, ValueType>( // For EACH Vertex Processing
//      [&](VertexId src, int current_send_partition) {
//        if (graph_->rtminfo->lock_free) {
//            VertexId src_trans = src - graph_->gnnctx->p_v_s;
//            if (partitioned_graph_->has_mirror_at(current_send_partition,src)
////          subgraphs[current_send_partition]->get_forward_active(src_trans)
//                ) {
//              VertexId write_index = subgraphs[current_send_partition]
//                      ->forward_multisocket_message_index[src_trans];
//              graph_->NtsComm->emit_buffer_lock_free(
//                  src,
//                  f_input_buffer + (src - graph_->gnnctx->p_v_s) * feature_size,
//                  write_index, feature_size);
//            }
//          } else {
//            // send to mirror directly
//            graph_->NtsComm->emit_buffer(
//                src,
//                f_input_buffer + (src - graph_->gnnctx->p_v_s) * feature_size,
//                feature_size);
//          }
//        
//      },
//      [&](VertexId dst, ValueType *recv_buffer, VertexId recv_id) {
//        memcpy(f_output_buffer + partitioned_graph_->MirrorIndex[dst] 
//                * feature_size, recv_buffer, sizeof(ValueType) * feature_size);
//        return 0;
//      },
//      subgraphs, feature_size, active_);
//      
//    return f_output;
//  }
//  
//  NtsVar backward(NtsVar &f_output_grad){// input vtx grad; output edge grad
//    int feature_size=f_output_grad.size(1);
//    NtsVar f_input_grad=graph_->Nts->NewLeafTensor({graph_->gnnctx->l_v_num, 
//                feature_size},torch::DeviceType::CPU);
//    ValueType *f_input_grad_buffer =
//      graph_->Nts->getWritableBuffer(f_input_grad, torch::DeviceType::CPU);
//    ValueType *f_output_grad_buffer =
//      graph_->Nts->getWritableBuffer(f_output_grad, torch::DeviceType::CPU);
//    
//      
//    graph_->process_edges_backward_decoupled_multisockets<
//      int, ValueType>( // For EACH Vertex
//                       // Processing
//      [&](VertexId src, VertexAdjList<Empty> outgoing_adj, VertexId thread_id,
//          VertexId recv_id, VertexId socketId) { // pull
//        VertexId src_trans = src - graph_->partition_offset[recv_id];
//        //assert(src<2708);
//        ValueType *local_output_buffer =//new ValueType[2048];
//            f_output_grad_buffer + partitioned_graph_->MirrorIndex[src] * feature_size;
//        if (graph_->rtminfo->lock_free) {
//          if (subgraphs[recv_id]->src_get_active(src)) {
//            VertexId write_index =
//                subgraphs[recv_id]
//                    ->backward_multisocket_message_index[src_trans]
//                    .vertexSocketPosition[socketId];
//            graph_->NtsComm->emit_buffer_lock_free(src, local_output_buffer,
//                                                   write_index, feature_size);
//          }
//        } else {
//          graph_->NtsComm->emit_buffer(src, local_output_buffer, feature_size);
//        }
//      },
//      [&](VertexId src, ValueType *msg) {
//        nts_acc(f_input_grad_buffer + (src - graph_->gnnctx->p_v_s) 
//                * feature_size, msg, feature_size);
//        return 1;
//      },
//     feature_size, active_); 
//      return f_input_grad;
//  }    
//
//};
//class DistScatterSrc : public ntsGraphOp{
//public:
//  //std::vector<CSC_segment_pinned *> subgraphs;
//  
//  DistScatterSrc(PartitionedGraph *partitioned_graph,VertexSubset *active)
//      : ntsGraphOp(partitioned_graph, active) {
//    //subgraphs = partitioned_graph->graph_chunks;
//  }
//  NtsVar forward(NtsVar &f_input){// input edge  output vertex
//    int feature_size = f_input.size(1);
//    //LOG_INFO("owned_mirrors (%d)",partitioned_graph_->owned_mirrors);
//    NtsVar f_output=graph_->Nts->NewKeyTensor({partitioned_graph_->owned_edges, 
//                feature_size},torch::DeviceType::CPU);
//    ValueType *f_input_buffer =
//      graph_->Nts->getWritableBuffer(f_input, torch::DeviceType::CPU);
//    ValueType *f_output_buffer =
//      graph_->Nts->getWritableBuffer(f_output, torch::DeviceType::CPU);  
//    //LOG_INFO("owned_mirrors (%d)",partitioned_graph_->owned_mirrors);
//    
//      partitioned_graph_->DistSchedulingMaster(
//        [&](VertexId dst,PartitionedGraph* pg){
//            VertexId dst_trans =dst-graph_->gnnctx->p_v_s;
//            for(int eid=pg->column_offset[dst_trans];
//                    eid<pg->column_offset[dst_trans+1];eid++){
//                VertexId src=pg->row_indices[eid];
//                VertexId src_pos=pg->MirrorIndex[src];
//                nts_copy(f_output_buffer,eid,f_input_buffer,
//                        src_pos,feature_size,1);    
//            }     
//        
//        });    
//      return f_output;
//  }
//  
//  NtsVar backward(NtsVar &f_output_grad){// input vtx grad; output edge grad
//    int feature_size=f_output_grad.size(1);
//    NtsVar f_input_grad=graph_->Nts->NewLeafTensor({partitioned_graph_->owned_mirrors, 
//                feature_size},torch::DeviceType::CPU);
//    ValueType *f_input_grad_buffer =
//      graph_->Nts->getWritableBuffer(f_input_grad, torch::DeviceType::CPU);
//    ValueType *f_output_grad_buffer =
//      graph_->Nts->getWritableBuffer(f_output_grad, torch::DeviceType::CPU);
//     partitioned_graph_->DistSchedulingMaster(
//        [&](VertexId dst,PartitionedGraph* pg){
//            VertexId dst_trans =dst-graph_->gnnctx->p_v_s;
//            for(int eid=pg->column_offset[dst_trans];
//                    eid<pg->column_offset[dst_trans+1];eid++){
//                VertexId src=pg->row_indices[eid];
//                VertexId src_pos=pg->MirrorIndex[src];
//                nts_acc(f_input_grad_buffer+src_pos*feature_size,
//                            f_output_grad_buffer+eid*feature_size,
//                                feature_size);    
//            }     
//        
//        });    
//      
//      return f_input_grad;
//  }
//};
//class DistScatterDst : public ntsGraphOp{
//public:
//  //std::vector<CSC_segment_pinned *> subgraphs;
//  
//  DistScatterDst(PartitionedGraph *partitioned_graph,VertexSubset *active)
//      : ntsGraphOp(partitioned_graph, active) {
//    //subgraphs = partitioned_graph->graph_chunks;
//  }
//  NtsVar forward(NtsVar &f_input){// input edge  output vertex
//    int feature_size = f_input.size(1);
//    //LOG_INFO("owned_mirrors (%d)",partitioned_graph_->owned_mirrors);
//    NtsVar f_output=graph_->Nts->NewKeyTensor({partitioned_graph_->owned_edges, 
//                feature_size},torch::DeviceType::CPU);
//    ValueType *f_input_buffer =
//      graph_->Nts->getWritableBuffer(f_input, torch::DeviceType::CPU);
//    ValueType *f_output_buffer =
//      graph_->Nts->getWritableBuffer(f_output, torch::DeviceType::CPU);  
//    //LOG_INFO("owned_mirrors (%d)",partitioned_graph_->owned_mirrors);
//    
//      partitioned_graph_->DistSchedulingMaster(
//        [&](VertexId dst,PartitionedGraph* pg){
//            VertexId dst_trans =dst-graph_->gnnctx->p_v_s;
//            for(int eid=pg->column_offset[dst_trans];
//                    eid<pg->column_offset[dst_trans+1];eid++){
//                nts_copy(f_output_buffer,eid,f_input_buffer,
//                        dst_trans,feature_size,1);    
//            }     
//        
//        });    
//      return f_output;
//  }
//  
//  NtsVar backward(NtsVar &f_output_grad){// input vtx grad; output edge grad
//    int feature_size=f_output_grad.size(1);
//    NtsVar f_input_grad=graph_->Nts->NewLeafTensor({graph_->gnnctx->l_v_num, 
//                feature_size},torch::DeviceType::CPU);
//    ValueType *f_input_grad_buffer =
//      graph_->Nts->getWritableBuffer(f_input_grad, torch::DeviceType::CPU);
//    ValueType *f_output_grad_buffer =
//      graph_->Nts->getWritableBuffer(f_output_grad, torch::DeviceType::CPU);
//     partitioned_graph_->DistSchedulingMaster(
//        [&](VertexId dst,PartitionedGraph* pg){
//            VertexId dst_trans =dst-graph_->gnnctx->p_v_s;
//            for(int eid=pg->column_offset[dst_trans];
//                    eid<pg->column_offset[dst_trans+1];eid++){
//                VertexId src=pg->row_indices[eid];
//                VertexId src_pos=pg->MirrorIndex[src];
//                nts_acc(f_input_grad_buffer+dst_trans*feature_size,
//                            f_output_grad_buffer+eid*feature_size,
//                                feature_size);    
//            }     
//        
//        });    
//      
//      return f_input_grad;
//  }    
//};
//class DistAggregateDst : public ntsGraphOp{
//public:
//  //std::vector<CSC_segment_pinned *> subgraphs;
//  
//  DistAggregateDst(PartitionedGraph *partitioned_graph,VertexSubset *active)
//      : ntsGraphOp(partitioned_graph, active) {
//    //subgraphs = partitioned_graph->graph_chunks;
//  }
//  NtsVar forward(NtsVar &f_input){// input edge  output vertex
//    int feature_size = f_input.size(1);
//    //LOG_INFO("owned_mirrors (%d)",partitioned_graph_->owned_mirrors);
//    NtsVar f_output=graph_->Nts->NewKeyTensor({partitioned_graph_->owned_vertices, 
//                feature_size},torch::DeviceType::CPU);
//    ValueType *f_input_buffer =
//      graph_->Nts->getWritableBuffer(f_input, torch::DeviceType::CPU);
//    ValueType *f_output_buffer =
//      graph_->Nts->getWritableBuffer(f_output, torch::DeviceType::CPU);  
//    //LOG_INFO("owned_mirrors (%d)",partitioned_graph_->owned_mirrors);
//    
//      partitioned_graph_->DistSchedulingMaster(
//        [&](VertexId dst,PartitionedGraph* pg){
//            VertexId dst_trans =dst-graph_->gnnctx->p_v_s;
//            for(int eid=pg->column_offset[dst_trans];
//                    eid<pg->column_offset[dst_trans+1];eid++){
//                    nts_acc(f_output_buffer+dst_trans*feature_size,
//                            f_input_buffer+eid*feature_size,
//                                feature_size);    
////                nts_copy(f_output_buffer,eid,f_input_buffer,
////                        dst_trans,feature_size,1);    
//            }     
//        
//        });    
//      return f_output;
//  }
//  
//  NtsVar backward(NtsVar &f_output_grad){// input vtx grad; output edge grad
//    int feature_size=f_output_grad.size(1);
//    NtsVar f_input_grad=graph_->Nts->NewLeafTensor({partitioned_graph_->owned_edges, 
//                feature_size},torch::DeviceType::CPU);
//    ValueType *f_input_grad_buffer =
//      graph_->Nts->getWritableBuffer(f_input_grad, torch::DeviceType::CPU);
//    ValueType *f_output_grad_buffer =
//      graph_->Nts->getWritableBuffer(f_output_grad, torch::DeviceType::CPU);
//     partitioned_graph_->DistSchedulingMaster(
//        [&](VertexId dst,PartitionedGraph* pg){
//            VertexId dst_trans =dst-graph_->gnnctx->p_v_s;
//            for(int eid=pg->column_offset[dst_trans];
//                    eid<pg->column_offset[dst_trans+1];eid++){
//                VertexId src=pg->row_indices[eid];
//                VertexId src_pos=pg->MirrorIndex[src];
////                nts_acc(f_input_grad_buffer+dst_trans*feature_size,
////                            f_output_grad_buffer+eid*feature_size,
////                                feature_size);
//                nts_copy(f_input_grad_buffer,eid,f_output_grad_buffer,
//                        dst_trans,feature_size,1);                   
//            }     
//        
//        });    
//      
//      return f_input_grad;
//  }    
//};
//class DistEdgeSoftMax : public ntsGraphOp{
//public:
//  NtsVar IntermediateResult;
//  
//  DistEdgeSoftMax(PartitionedGraph *partitioned_graph,VertexSubset *active)
//      : ntsGraphOp(partitioned_graph, active) {
//  }
//  NtsVar forward(NtsVar &f_input_){// input i_msg  output o_msg
//     //NtsVar f_input_=f_input.detach();
//    int feature_size = f_input_.size(1);
//    NtsVar f_output=graph_->Nts->NewKeyTensor({partitioned_graph_->owned_edges, 
//                feature_size},torch::DeviceType::CPU);
//    ValueType *f_input_buffer =
//      graph_->Nts->getWritableBuffer(f_input_, torch::DeviceType::CPU);
//    ValueType *f_output_buffer =
//      graph_->Nts->getWritableBuffer(f_output, torch::DeviceType::CPU);
//    partitioned_graph_->DistSchedulingMaster(
//        [&](VertexId dst,PartitionedGraph* pg){
//          long eid_start = pg->column_offset[dst];
//          long eid_end = pg->column_offset[dst + 1];
//          NtsVar d = f_input_.slice(0, eid_start, eid_end, 1).softmax(0);
//            VertexId dst_trans =dst-graph_->gnnctx->p_v_s;
//          ValueType *d_buffer =
//          graph_->Nts->getWritableBuffer(d, torch::DeviceType::CPU);    
//          nts_copy(f_output_buffer, eid_start, d_buffer, 
//                  0, feature_size,(eid_end-eid_start));
//        });
//    IntermediateResult=f_output;        
//    return f_output;
//  }
//  
//  NtsVar backward(NtsVar &f_output_grad){// input vtx grad; output edge grad
//    int feature_size=f_output_grad.size(1);
//    NtsVar f_input_grad=graph_->Nts->NewLeafTensor({partitioned_graph_->owned_edges, 
//                feature_size},torch::DeviceType::CPU);
//    ValueType *f_input_grad_buffer =
//      graph_->Nts->getWritableBuffer(f_input_grad, torch::DeviceType::CPU);
//    partitioned_graph_->DistSchedulingMaster(
//        [&](VertexId dst,PartitionedGraph* pg){
//          long eid_start = pg->column_offset[dst];
//          long eid_end = pg->column_offset[dst + 1];
//          NtsVar d   = f_output_grad.slice(0, eid_start, eid_end, 1);
//          NtsVar imr =IntermediateResult.slice(0, eid_start, eid_end, 1);
//          NtsVar d_o =(imr*d)-imr*(d.t().mm(imr)); 
//          ValueType *d_o_buffer =
//          graph_->Nts->getWritableBuffer(d_o, torch::DeviceType::CPU);
//          nts_copy(f_input_grad_buffer, eid_start, d_o_buffer, 
//                  0, feature_size,(eid_end-eid_start));
//        
//        });
//  }    
//
//};
//
//} // namespace graphop
//} // namespace nts

#endif

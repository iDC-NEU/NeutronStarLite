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
#ifndef NTSDISTCPUGRAPHOP_HPP
#define NTSDISTCPUGRAPHOP_HPP
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

class DistGetDepNbrOp : public ntsGraphOp{
public:
  std::vector<CSC_segment_pinned *> subgraphs;
  
  DistGetDepNbrOp(PartitionedGraph *partitioned_graph,VertexSubset *active)
      : ntsGraphOp(partitioned_graph, active) {
    subgraphs = partitioned_graph->graph_chunks;
  }
  NtsVar forward(NtsVar &f_input){// input edge  output vertex
    int feature_size = f_input.size(1);
    //LOG_INFO("owned_mirrors (%d)",partitioned_graph_->owned_mirrors);
    NtsVar f_output=graph_->Nts->NewKeyTensor({partitioned_graph_->owned_mirrors, 
                feature_size},torch::DeviceType::CPU);
    ValueType *f_input_buffer =
      graph_->Nts->getWritableBuffer(f_input, torch::DeviceType::CPU);
    ValueType *f_output_buffer =
      graph_->Nts->getWritableBuffer(f_output, torch::DeviceType::CPU);  
    //LOG_INFO("owned_mirrors (%d)",partitioned_graph_->owned_mirrors);    
      graph_->get_from_dep_neighbor_mutisockets<int, ValueType>( // For EACH Vertex Processing
      [&](VertexId src, int current_send_partition) {
        if (graph_->rtminfo->lock_free) {
            VertexId src_trans = src - graph_->gnnctx->p_v_s;
            if (partitioned_graph_->has_mirror_at(current_send_partition,src)
//          subgraphs[current_send_partition]->get_forward_active(src_trans)
                ) {
              VertexId write_index = subgraphs[current_send_partition]
                      ->forward_multisocket_message_index[src_trans];
              graph_->NtsComm->emit_buffer_lock_free(
                  src,
                  f_input_buffer + (src - graph_->gnnctx->p_v_s) * feature_size,
                  write_index, feature_size);
            }
          } else {
            // send to mirror directly
            graph_->NtsComm->emit_buffer(
                src,
                f_input_buffer + (src - graph_->gnnctx->p_v_s) * feature_size,
                feature_size);
          }
        
      },
      [&](VertexId dst, ValueType *recv_buffer, VertexId recv_id) {
        memcpy(f_output_buffer + partitioned_graph_->MirrorIndex[dst] 
                * feature_size, recv_buffer, sizeof(ValueType) * feature_size);
        return 0;
      },
      subgraphs, feature_size, active_);
      
    return f_output;
  }
  
  NtsVar backward(NtsVar &f_output_grad){// input vtx grad; output edge grad
    int feature_size=f_output_grad.size(1);
    NtsVar f_input_grad=graph_->Nts->NewLeafTensor({graph_->gnnctx->l_v_num, 
                feature_size},torch::DeviceType::CPU);
    ValueType *f_input_grad_buffer =
      graph_->Nts->getWritableBuffer(f_input_grad, torch::DeviceType::CPU);
    ValueType *f_output_grad_buffer =
      graph_->Nts->getWritableBuffer(f_output_grad, torch::DeviceType::CPU);
    
      
    graph_->process_edges_backward_decoupled_multisockets<
      int, ValueType>( // For EACH Vertex
                       // Processing
      [&](VertexId src, VertexAdjList<Empty> outgoing_adj, VertexId thread_id,
          VertexId recv_id, VertexId socketId) { // pull
        VertexId src_trans = src - graph_->partition_offset[recv_id];
        //assert(src<2708);
        ValueType *local_output_buffer =//new ValueType[2048];
            f_output_grad_buffer + partitioned_graph_->MirrorIndex[src] * feature_size;
        if (graph_->rtminfo->lock_free) {
          if (subgraphs[recv_id]->src_get_active(src)) {
            VertexId write_index =
                subgraphs[recv_id]
                    ->backward_multisocket_message_index[src_trans]
                    .vertexSocketPosition[socketId];
            graph_->NtsComm->emit_buffer_lock_free(src, local_output_buffer,
                                                   write_index, feature_size);
          }
        } else {
          graph_->NtsComm->emit_buffer(src, local_output_buffer, feature_size);
        }
      },
      [&](VertexId src, ValueType *msg) {
        nts_acc(f_input_grad_buffer + (src - graph_->gnnctx->p_v_s) 
                * feature_size, msg, feature_size);
        return 1;
      },
     feature_size, active_); 
      return f_input_grad;
  }    

};
class DistScatterSrc : public ntsGraphOp{
public:
  //std::vector<CSC_segment_pinned *> subgraphs;
  
  DistScatterSrc(PartitionedGraph *partitioned_graph,VertexSubset *active)
      : ntsGraphOp(partitioned_graph, active) {
    //subgraphs = partitioned_graph->graph_chunks;
  }
  NtsVar forward(NtsVar &f_input){// input edge  output vertex
    int feature_size = f_input.size(1);
    //LOG_INFO("owned_mirrors (%d)",partitioned_graph_->owned_mirrors);
    NtsVar f_output=graph_->Nts->NewKeyTensor({partitioned_graph_->owned_edges, 
                feature_size},torch::DeviceType::CPU);
    ValueType *f_input_buffer =
      graph_->Nts->getWritableBuffer(f_input, torch::DeviceType::CPU);
    ValueType *f_output_buffer =
      graph_->Nts->getWritableBuffer(f_output, torch::DeviceType::CPU);  
    //LOG_INFO("owned_mirrors (%d)",partitioned_graph_->owned_mirrors);
    
      partitioned_graph_->DistSchedulingMaster(
        [&](VertexId dst,PartitionedGraph* pg){
            VertexId dst_trans =dst-graph_->gnnctx->p_v_s;
            for(int eid=pg->column_offset[dst_trans];
                    eid<pg->column_offset[dst_trans+1];eid++){
                VertexId src=pg->row_indices[eid];
                VertexId src_pos=pg->MirrorIndex[src];
                nts_copy(f_output_buffer,eid,f_input_buffer,
                        src_pos,feature_size,1);    
            }     
        
        });    
      return f_output;
  }
  
  NtsVar backward(NtsVar &f_output_grad){// input vtx grad; output edge grad
    int feature_size=f_output_grad.size(1);
    NtsVar f_input_grad=graph_->Nts->NewLeafTensor({partitioned_graph_->owned_mirrors, 
                feature_size},torch::DeviceType::CPU);
    ValueType *f_input_grad_buffer =
      graph_->Nts->getWritableBuffer(f_input_grad, torch::DeviceType::CPU);
    ValueType *f_output_grad_buffer =
      graph_->Nts->getWritableBuffer(f_output_grad, torch::DeviceType::CPU);
     partitioned_graph_->DistSchedulingMaster(
        [&](VertexId dst,PartitionedGraph* pg){
            VertexId dst_trans =dst-graph_->gnnctx->p_v_s;
            for(int eid=pg->column_offset[dst_trans];
                    eid<pg->column_offset[dst_trans+1];eid++){
                VertexId src=pg->row_indices[eid];
                VertexId src_pos=pg->MirrorIndex[src];
                nts_acc(f_input_grad_buffer+src_pos*feature_size,
                            f_output_grad_buffer+eid*feature_size,
                                feature_size);    
            }     
        
        });    
      
      return f_input_grad;
  }
};
class DistScatterDst : public ntsGraphOp{
public:
  //std::vector<CSC_segment_pinned *> subgraphs;
  
  DistScatterDst(PartitionedGraph *partitioned_graph,VertexSubset *active)
      : ntsGraphOp(partitioned_graph, active) {
    //subgraphs = partitioned_graph->graph_chunks;
  }
  NtsVar forward(NtsVar &f_input){// input edge  output vertex
    int feature_size = f_input.size(1);
    //LOG_INFO("owned_mirrors (%d)",partitioned_graph_->owned_mirrors);
    NtsVar f_output=graph_->Nts->NewKeyTensor({partitioned_graph_->owned_edges, 
                feature_size},torch::DeviceType::CPU);
    ValueType *f_input_buffer =
      graph_->Nts->getWritableBuffer(f_input, torch::DeviceType::CPU);
    ValueType *f_output_buffer =
      graph_->Nts->getWritableBuffer(f_output, torch::DeviceType::CPU);  
    //LOG_INFO("owned_mirrors (%d)",partitioned_graph_->owned_mirrors);
    
      partitioned_graph_->DistSchedulingMaster(
        [&](VertexId dst,PartitionedGraph* pg){
            VertexId dst_trans =dst-graph_->gnnctx->p_v_s;
            for(int eid=pg->column_offset[dst_trans];
                    eid<pg->column_offset[dst_trans+1];eid++){
                nts_copy(f_output_buffer,eid,f_input_buffer,
                        dst_trans,feature_size,1);    
            }     
        
        });    
      return f_output;
  }
  
  NtsVar backward(NtsVar &f_output_grad){// input vtx grad; output edge grad
    int feature_size=f_output_grad.size(1);
    NtsVar f_input_grad=graph_->Nts->NewLeafTensor({graph_->gnnctx->l_v_num, 
                feature_size},torch::DeviceType::CPU);
    ValueType *f_input_grad_buffer =
      graph_->Nts->getWritableBuffer(f_input_grad, torch::DeviceType::CPU);
    ValueType *f_output_grad_buffer =
      graph_->Nts->getWritableBuffer(f_output_grad, torch::DeviceType::CPU);
     partitioned_graph_->DistSchedulingMaster(
        [&](VertexId dst,PartitionedGraph* pg){
            VertexId dst_trans =dst-graph_->gnnctx->p_v_s;
            for(int eid=pg->column_offset[dst_trans];
                    eid<pg->column_offset[dst_trans+1];eid++){
                VertexId src=pg->row_indices[eid];
                VertexId src_pos=pg->MirrorIndex[src];
                nts_acc(f_input_grad_buffer+dst_trans*feature_size,
                            f_output_grad_buffer+eid*feature_size,
                                feature_size);    
            }     
        
        });    
      
      return f_input_grad;
  }    
};
class DistAggregateDst : public ntsGraphOp{
public:
  //std::vector<CSC_segment_pinned *> subgraphs;
  
  DistAggregateDst(PartitionedGraph *partitioned_graph,VertexSubset *active)
      : ntsGraphOp(partitioned_graph, active) {
    //subgraphs = partitioned_graph->graph_chunks;
  }
  NtsVar forward(NtsVar &f_input){// input edge  output vertex
    int feature_size = f_input.size(1);
    //LOG_INFO("owned_mirrors (%d)",partitioned_graph_->owned_mirrors);
    NtsVar f_output=graph_->Nts->NewKeyTensor({partitioned_graph_->owned_vertices, 
                feature_size},torch::DeviceType::CPU);
    ValueType *f_input_buffer =
      graph_->Nts->getWritableBuffer(f_input, torch::DeviceType::CPU);
    ValueType *f_output_buffer =
      graph_->Nts->getWritableBuffer(f_output, torch::DeviceType::CPU);  
    //LOG_INFO("owned_mirrors (%d)",partitioned_graph_->owned_mirrors);
    
      partitioned_graph_->DistSchedulingMaster(
        [&](VertexId dst,PartitionedGraph* pg){
            VertexId dst_trans =dst-graph_->gnnctx->p_v_s;
            for(int eid=pg->column_offset[dst_trans];
                    eid<pg->column_offset[dst_trans+1];eid++){
                    nts_acc(f_output_buffer+dst_trans*feature_size,
                            f_input_buffer+eid*feature_size,
                                feature_size);    
//                nts_copy(f_output_buffer,eid,f_input_buffer,
//                        dst_trans,feature_size,1);    
            }     
        
        });    
      return f_output;
  }
  
  NtsVar backward(NtsVar &f_output_grad){// input vtx grad; output edge grad
    int feature_size=f_output_grad.size(1);
    NtsVar f_input_grad=graph_->Nts->NewLeafTensor({partitioned_graph_->owned_edges, 
                feature_size},torch::DeviceType::CPU);
    ValueType *f_input_grad_buffer =
      graph_->Nts->getWritableBuffer(f_input_grad, torch::DeviceType::CPU);
    ValueType *f_output_grad_buffer =
      graph_->Nts->getWritableBuffer(f_output_grad, torch::DeviceType::CPU);
     partitioned_graph_->DistSchedulingMaster(
        [&](VertexId dst,PartitionedGraph* pg){
            VertexId dst_trans =dst-graph_->gnnctx->p_v_s;
            for(int eid=pg->column_offset[dst_trans];
                    eid<pg->column_offset[dst_trans+1];eid++){
                VertexId src=pg->row_indices[eid];
                VertexId src_pos=pg->MirrorIndex[src];
//                nts_acc(f_input_grad_buffer+dst_trans*feature_size,
//                            f_output_grad_buffer+eid*feature_size,
//                                feature_size);
                nts_copy(f_input_grad_buffer,eid,f_output_grad_buffer,
                        dst_trans,feature_size,1);                   
            }     
        
        });    
      
      return f_input_grad;
  }    
};

class DistAggregateDstMin : public ntsGraphOp{
public:
  //std::vector<CSC_segment_pinned *> subgraphs;
  VertexId* record;
  DistAggregateDstMin(PartitionedGraph *partitioned_graph,VertexSubset *active)
      : ntsGraphOp(partitioned_graph, active) {
    //subgraphs = partitioned_graph->graph_chunks;
  }
  ~DistAggregateDstMin(){
      delete [] record;
  }
  NtsVar forward(NtsVar &f_input){// input edge  output vertex
    int feature_size = f_input.size(1);
    //LOG_INFO("owned_mirrors (%d)",partitioned_graph_->owned_mirrors);
    record=new VertexId(partitioned_graph_->owned_vertices*feature_size);
    NtsVar f_output=graph_->Nts->NewKeyTensor({partitioned_graph_->owned_vertices, 
                feature_size},torch::DeviceType::CPU);
    ValueType *f_input_buffer =
      graph_->Nts->getWritableBuffer(f_input, torch::DeviceType::CPU);
    ValueType *f_output_buffer =
      graph_->Nts->getWritableBuffer(f_output, torch::DeviceType::CPU);  
    //LOG_INFO("owned_mirrors (%d)",partitioned_graph_->owned_mirrors);
    
      partitioned_graph_->DistSchedulingMaster(
        [&](VertexId dst,PartitionedGraph* pg){
            VertexId dst_trans =dst-graph_->gnnctx->p_v_s;
            for(int eid=pg->column_offset[dst_trans];
                    eid<pg->column_offset[dst_trans+1];eid++){
                nts_min(f_output_buffer+ dst_trans * feature_size,
                   f_input_buffer + eid * feature_size, 
                    record + dst_trans * feature_size,
                     feature_size, eid);
//                    nts_acc(f_output_buffer+dst_trans*feature_size,
//                            f_input_buffer+eid*feature_size,
//                                feature_size);     
            }     
        
        });    
      return f_output;
  }
  
  NtsVar backward(NtsVar &f_output_grad){// input vtx grad; output edge grad
    int feature_size=f_output_grad.size(1);
    NtsVar f_input_grad=graph_->Nts->NewLeafTensor({partitioned_graph_->owned_edges, 
                feature_size},torch::DeviceType::CPU);
    ValueType *f_input_grad_buffer =
      graph_->Nts->getWritableBuffer(f_input_grad, torch::DeviceType::CPU);
    ValueType *f_output_grad_buffer =
      graph_->Nts->getWritableBuffer(f_output_grad, torch::DeviceType::CPU);
     partitioned_graph_->DistSchedulingMaster(
        [&](VertexId dst,PartitionedGraph* pg){
            VertexId dst_trans =dst-graph_->gnnctx->p_v_s;
            nts_assign(f_input_grad_buffer, f_output_grad_buffer+feature_size*dst_trans,
                record+feature_size*dst_trans, feature_size);
//            for(int eid=pg->column_offset[dst_trans];
//                    eid<pg->column_offset[dst_trans+1];eid++){
//                VertexId src=pg->row_indices[eid];
//                VertexId src_pos=pg->MirrorIndex[src];
//                nts_copy(f_input_grad_buffer,eid,f_output_grad_buffer,
//                        dst_trans,feature_size,1);                   
//            }     
        
        });    
      
      return f_input_grad;
  }    
};

class DistAggregateDstMax : public ntsGraphOp{
public:
  //std::vector<CSC_segment_pinned *> subgraphs;
  VertexId* record;
  DistAggregateDstMax(PartitionedGraph *partitioned_graph,VertexSubset *active)
      : ntsGraphOp(partitioned_graph, active) {
    //subgraphs = partitioned_graph->graph_chunks;
  }
  ~DistAggregateDstMax(){
      delete [] record;
  }
  NtsVar forward(NtsVar &f_input){// input edge  output vertex
    int feature_size = f_input.size(1);
    //LOG_INFO("owned_mirrors (%d)",partitioned_graph_->owned_mirrors);
    record=new VertexId(partitioned_graph_->owned_vertices*feature_size);
    NtsVar f_output=graph_->Nts->NewKeyTensor({partitioned_graph_->owned_vertices, 
                feature_size},torch::DeviceType::CPU);
    ValueType *f_input_buffer =
      graph_->Nts->getWritableBuffer(f_input, torch::DeviceType::CPU);
    ValueType *f_output_buffer =
      graph_->Nts->getWritableBuffer(f_output, torch::DeviceType::CPU);  
    //LOG_INFO("owned_mirrors (%d)",partitioned_graph_->owned_mirrors);
    
      partitioned_graph_->DistSchedulingMaster(
        [&](VertexId dst,PartitionedGraph* pg){
            VertexId dst_trans =dst-graph_->gnnctx->p_v_s;
            for(int eid=pg->column_offset[dst_trans];
                    eid<pg->column_offset[dst_trans+1];eid++){
                nts_max(f_output_buffer+ dst_trans * feature_size,
                   f_input_buffer + eid * feature_size, 
                    record + dst_trans * feature_size,
                     feature_size, eid);
//                    nts_acc(f_output_buffer+dst_trans*feature_size,
//                            f_input_buffer+eid*feature_size,
//                                feature_size);     
            }     
        
        });    
      return f_output;
  }
  
  NtsVar backward(NtsVar &f_output_grad){// input vtx grad; output edge grad
    int feature_size=f_output_grad.size(1);
    NtsVar f_input_grad=graph_->Nts->NewLeafTensor({partitioned_graph_->owned_edges, 
                feature_size},torch::DeviceType::CPU);
    ValueType *f_input_grad_buffer =
      graph_->Nts->getWritableBuffer(f_input_grad, torch::DeviceType::CPU);
    ValueType *f_output_grad_buffer =
      graph_->Nts->getWritableBuffer(f_output_grad, torch::DeviceType::CPU);
     partitioned_graph_->DistSchedulingMaster(
        [&](VertexId dst,PartitionedGraph* pg){
            VertexId dst_trans =dst-graph_->gnnctx->p_v_s;
            nts_assign(f_input_grad_buffer, f_output_grad_buffer+feature_size*dst_trans,
                record+feature_size*dst_trans, feature_size);
//            for(int eid=pg->column_offset[dst_trans];
//                    eid<pg->column_offset[dst_trans+1];eid++){
//                VertexId src=pg->row_indices[eid];
//                VertexId src_pos=pg->MirrorIndex[src];
//                nts_copy(f_input_grad_buffer,eid,f_output_grad_buffer,
//                        dst_trans,feature_size,1);                   
//            }     
        
        });    
      
      return f_input_grad;
  }    
};

class DistEdgeSoftMax : public ntsGraphOp{
public:
  NtsVar IntermediateResult;
  
  DistEdgeSoftMax(PartitionedGraph *partitioned_graph,VertexSubset *active)
      : ntsGraphOp(partitioned_graph, active) {
  }
  NtsVar forward(NtsVar &f_input_){// input i_msg  output o_msg
     //NtsVar f_input_=f_input.detach();
    int feature_size = f_input_.size(1);
    NtsVar f_output=graph_->Nts->NewKeyTensor({partitioned_graph_->owned_edges, 
                feature_size},torch::DeviceType::CPU);
    ValueType *f_input_buffer =
      graph_->Nts->getWritableBuffer(f_input_, torch::DeviceType::CPU);
    ValueType *f_output_buffer =
      graph_->Nts->getWritableBuffer(f_output, torch::DeviceType::CPU);
    partitioned_graph_->DistSchedulingMaster(
        [&](VertexId dst,PartitionedGraph* pg){
          VertexId dst_trans =dst-graph_->gnnctx->p_v_s;
          long eid_start = pg->column_offset[dst_trans];
          long eid_end = pg->column_offset[dst_trans + 1];
          NtsVar d = f_input_.slice(0, eid_start, eid_end, 1).softmax(0);
          //NtsVar d= f_input_.slice(0, eid_start, eid_end, 1)*torch::exp(f_input_.slice(0, eid_start, eid_end, 1)).sum(0);/// torch::exp(f_input_.slice(0, eid_start, eid_end, 1)).sum(0);
          ValueType *d_buffer =
          graph_->Nts->getWritableBuffer(d, torch::DeviceType::CPU);    
          nts_copy(f_output_buffer, eid_start, d_buffer, 
                  0, feature_size,(eid_end-eid_start));
        });
    IntermediateResult=f_output;        
    return f_output;
  }
  
  NtsVar backward(NtsVar &f_output_grad){// input vtx grad; output edge grad
    int feature_size=f_output_grad.size(1);
    NtsVar f_input_grad=graph_->Nts->NewLeafTensor({partitioned_graph_->owned_edges, 
                feature_size},torch::DeviceType::CPU);
    ValueType *f_input_grad_buffer =
      graph_->Nts->getWritableBuffer(f_input_grad, torch::DeviceType::CPU);
    partitioned_graph_->DistSchedulingMaster(
        [&](VertexId dst,PartitionedGraph* pg){
          VertexId dst_trans =dst-graph_->gnnctx->p_v_s;
          long eid_start = pg->column_offset[dst_trans];
          long eid_end = pg->column_offset[dst_trans + 1];
          NtsVar d   = f_output_grad.slice(0, eid_start, eid_end, 1);
          NtsVar imr =IntermediateResult.slice(0, eid_start, eid_end, 1);
          NtsVar d_o =(imr*d)-imr*(d.t().mm(imr)); 
          ValueType *d_o_buffer =
          graph_->Nts->getWritableBuffer(d_o, torch::DeviceType::CPU);
          nts_copy(f_input_grad_buffer, eid_start, d_o_buffer, 
                  0, feature_size,(eid_end-eid_start));
        
        });
        return f_input_grad;
  }    

};

class DistAggregateDstFuseWeight : public ntsGraphOp{
public:
  //std::vector<CSC_segment_pinned *> subgraphs;
    NtsVar e_weight_grad;
    ValueType *input_ptr;
    NtsVar e_weight_cache;
  DistAggregateDstFuseWeight(PartitionedGraph *partitioned_graph,VertexSubset *active)
      : ntsGraphOp(partitioned_graph, active) {
    //subgraphs = partitioned_graph->graph_chunks;
          
  }
   NtsVar forward(NtsVar &f_input){
       ;
   }
  NtsVar forward(NtsVar &f_input, NtsVar &e_weight){// input edge  output vertex
    int feature_size = f_input.size(1);
    //LOG_INFO("owned_mirrors (%d)",partitioned_graph_->owned_mirrors);
    NtsVar f_output=graph_->Nts->NewKeyTensor({partitioned_graph_->owned_vertices, 
                feature_size},torch::DeviceType::CPU);
    ValueType *f_input_buffer =
      graph_->Nts->getWritableBuffer(f_input, torch::DeviceType::CPU);
    input_ptr=f_input_buffer;
    ValueType *e_weight_buffer =
      graph_->Nts->getWritableBuffer(e_weight, torch::DeviceType::CPU);
    ValueType *f_output_buffer =
      graph_->Nts->getWritableBuffer(f_output, torch::DeviceType::CPU);  
    //LOG_INFO("owned_mirrors (%d)",partitioned_graph_->owned_mirrors);
     e_weight_cache=e_weight;
      partitioned_graph_->DistSchedulingMaster(
        [&](VertexId dst,PartitionedGraph* pg){
            VertexId dst_trans =dst-graph_->gnnctx->p_v_s;
            for(long eid=pg->column_offset[dst_trans];
                    eid<pg->column_offset[dst_trans+1];eid++){
                VertexId src=pg->row_indices[eid];
                VertexId src_pos=pg->MirrorIndex[src];
//                    nts_acc(f_output_buffer+dst_trans*feature_size,
//                            f_input_buffer+src_pos*feature_size,
//                                feature_size);    
//                nts_copy(f_output_buffer,eid,f_input_buffer,
//                        dst_trans,feature_size,1);   
                ValueType* local_input_buffer=f_input_buffer+src_pos*feature_size;
                ValueType* local_output_buffer=f_output_buffer+dst_trans*feature_size;  
                nts_comp(local_output_buffer, local_input_buffer, e_weight_buffer[eid],
               feature_size);
            }     
        
        });    
      return f_output;
  }
  
  NtsVar backward(NtsVar &f_output_grad){// input vtx grad; output edge grad
    int feature_size=f_output_grad.size(1);
    NtsVar f_input_grad=graph_->Nts->NewLeafTensor({partitioned_graph_->owned_mirrors, 
                feature_size},torch::DeviceType::CPU);
    e_weight_grad=graph_->Nts->NewLeafTensor(e_weight_cache,torch::DeviceType::CPU); 
    ValueType *f_input_grad_buffer =
      graph_->Nts->getWritableBuffer(f_input_grad, torch::DeviceType::CPU);
    ValueType *e_weight_buffer =
      graph_->Nts->getWritableBuffer(e_weight_cache, torch::DeviceType::CPU);
    ValueType *e_weight_grad_buffer =
      graph_->Nts->getWritableBuffer(e_weight_grad, torch::DeviceType::CPU);
    ValueType *f_output_grad_buffer =
      graph_->Nts->getWritableBuffer(f_output_grad, torch::DeviceType::CPU);
     partitioned_graph_->DistSchedulingMaster(
        [&](VertexId dst,PartitionedGraph* pg){
            VertexId dst_trans =dst-graph_->gnnctx->p_v_s;
            for(int eid=pg->column_offset[dst_trans];
                    eid<pg->column_offset[dst_trans+1];eid++){
                VertexId src=pg->row_indices[eid];
                VertexId src_pos=pg->MirrorIndex[src];
//                nts_acc(f_input_grad_buffer+dst_trans*feature_size,
//                            f_output_grad_buffer+eid*feature_size,
//                                feature_size);
                nts_acc(f_input_grad_buffer+src_pos*feature_size,
                            f_output_grad_buffer+dst_trans*feature_size,
                                feature_size);   
                ValueType* local_input_buffer=f_output_grad_buffer+dst_trans*feature_size;
                ValueType* local_output_buffer=f_input_grad_buffer+src_pos*feature_size;  
                nts_comp(local_output_buffer, local_input_buffer, e_weight_buffer[eid],
                    feature_size);
                ValueType* i_f_buffer=input_ptr+src_pos*feature_size;
                ValueType* i_f_g_buffer=f_output_grad_buffer+dst_trans*feature_size;
                e_weight_grad_buffer[eid]=
                        dot_product(i_f_buffer,i_f_g_buffer,feature_size);
                
            }     
        
        });    
      
      return f_input_grad;
  }
  NtsVar get_additional_grad(){
      return this->e_weight_grad;
  }
  
};


} // namespace graphop
} // namespace nts

#endif

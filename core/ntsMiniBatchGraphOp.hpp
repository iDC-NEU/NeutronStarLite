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
#ifndef NTSMINIBATCHGRAPHOP_HPP
#define NTSMINIBATCHGRAPHOP_HPP
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
#include "core/ntsPeerRPC.hpp"
#include "ntsSampler.hpp"

namespace nts {
namespace op {
    
    

NtsVar get_label(std::vector<VertexId>& dst, NtsVar &whole, Graph<Empty> *graph){
    NtsVar f_output=graph->Nts->NewLeafKLongTensor({dst.size()});
#pragma omp parallel for
    for(int i=0;i<dst.size();i++){
          f_output[i]=whole[dst[i] - graph->partition_offset[graph->partition_id]];
      }
    return f_output;
  }
    
NtsVar get_feature(std::vector<VertexId>& src, NtsVar &whole, Graph<Empty> *graph){
    int feature_size = whole.size(1);
    NtsVar f_output=graph->Nts->NewKeyTensor({src.size(), 
                feature_size},torch::DeviceType::CPU);
    ValueType *f_input_buffer =
        graph->Nts->getWritableBuffer(whole, torch::DeviceType::CPU);
    ValueType *f_output_buffer =
        graph->Nts->getWritableBuffer(f_output, torch::DeviceType::CPU);
#pragma omp parallel for
      for(int i=0;i<src.size();i++){
          memcpy(f_output_buffer+i*feature_size,
                    f_input_buffer+src[i]*feature_size,
                        feature_size*sizeof(ValueType));
      }
    return f_output;
  }

  NtsVar get_feature_from_global(ntsPeerRPC<ValueType, VertexId>&rpc, std::vector<VertexId>& src, NtsVar& X, Graph<Empty>* graph){
    int feature_size = X.size(1);
    NtsVar f_output = graph->Nts->NewKeyTensor({static_cast<long>(src.size()), feature_size}, torch::DeviceType::CPU);
    ValueType * f_output_buffer = graph->Nts->getWritableBuffer(f_output, torch::DeviceType::CPU);
    std::vector<std::vector<VertexId>> partition_ids;

    // 将节点按照partition分类
    partition_ids.resize(graph->partitions);
    for(auto id : src){
        partition_ids[graph->get_partition_id(id)].push_back(id);
    }
    std::vector<std::vector<std::vector<ValueType>>> resultVector;
    resultVector.resize(graph->partitions);
    for(int i = 0; i < partition_ids.size(); i++) {
        int target = (i + graph->partition_id) % graph->partitions;
        resultVector[target] = rpc.call_function("get_feature", partition_ids[target], target);

    }
    std::vector<int> partition_index(graph->partitions, 0);

    for(int i = 0; i < src.size(); i++) {

        int partition_id = graph->get_partition_id(src[i]);
//        int feature_index = __sync_fetch_and_add(&partition_index[partition_id], 1);
        int feature_index = partition_index[partition_id]++;
        assert(feature_index < resultVector[partition_id].size());
        memcpy(f_output_buffer+i*feature_size, resultVector[partition_id][feature_index].data(),
               feature_size*sizeof(ValueType));

//        float sum = 0.0;
//        for(int j = 0; j < resultVector[partition_id][feature_index].size(); j++) {
//            sum += resultVector[partition_id][feature_index][j];
//        }
//        std::printf("%u sum %f\n", src[i], sum);
    }
//    for(int i = 0; i < resultVector.size(); i++) {
//        assert(partition_index[i] == resultVector[i].size());
//    }
//    for(int i = 0; i < src.size(); i++) {
//        if(graph->get_partition_id(src[i]) == graph->partition_id) {
//            continue;
//        }
//        auto sum = f_output[i].sum().item<float>();
//        std::printf("%u sum %f\n", src[i], sum);
//    }
//    if(graph->vertices != 0) {
//        MPI_Barrier(MPI_COMM_WORLD);
//        exit(3);
//    }
    return f_output;

  }
class MiniBatchFuseOp : public ntsGraphOp{
public:
  SampledSubgraph* subgraphs;
  int layer=0;
  
  MiniBatchFuseOp(SampledSubgraph *subgraphs_,Graph<Empty> *graph_,int layer_)
      : ntsGraphOp(graph_) {
    subgraphs = subgraphs_;
    layer=layer_;
  }
  NtsVar forward(NtsVar &f_input){
    int feature_size = f_input.size(1);
    NtsVar f_output=graph_->Nts->NewKeyTensor({subgraphs->sampled_sgs[layer]->dst().size(), 
                feature_size},torch::DeviceType::CPU);
    ValueType *f_input_buffer =
      graph_->Nts->getWritableBuffer(f_input, torch::DeviceType::CPU);
    ValueType *f_output_buffer =
      graph_->Nts->getWritableBuffer(f_output, torch::DeviceType::CPU);   
    this->subgraphs->compute_one_layer(
            [&](VertexId local_dst, std::vector<VertexId>&column_offset, 
                std::vector<VertexId>&row_indices){
                VertexId src_start=column_offset[local_dst];
                VertexId src_end=column_offset[local_dst+1];
                VertexId dst=subgraphs->sampled_sgs[layer]->dst()[local_dst];
                ValueType *local_output=f_output_buffer+local_dst*feature_size;
                for(VertexId src_offset=src_start;
                        src_offset<src_end;src_offset++){
                    VertexId local_src=subgraphs->sampled_sgs[layer]->r_i(src_offset);
                    VertexId src=subgraphs->sampled_sgs[layer]->src()[local_src];
                    ValueType *local_input=f_input_buffer+local_src*feature_size;
                    nts_comp(local_output, local_input,
                            nts_norm_degree(graph_,src, dst), feature_size);
                    
                }
              },
            layer,
            12//compute thread num;
            );   
      return f_output;
   }
   NtsVar backward(NtsVar &f_output_grad){
        int feature_size=f_output_grad.size(1);
//        std::printf("output size: (%lu, %lu)\n", f_output_grad.size(0), f_output_grad.size(1));
        NtsVar f_input_grad=graph_->Nts->NewLeafTensor({subgraphs->sampled_sgs[layer]->src().size(), 
                feature_size},torch::DeviceType::CPU);
        ValueType *f_input_grad_buffer =
          graph_->Nts->getWritableBuffer(f_input_grad, torch::DeviceType::CPU);
        ValueType *f_output_grad_buffer =
          graph_->Nts->getWritableBuffer(f_output_grad, torch::DeviceType::CPU);

        // 下面是从dst传回src
        this->subgraphs->compute_one_layer(
                // 实际调用中local_dst是从0开始
            [&](VertexId local_dst, std::vector<VertexId>&column_offset, 
                std::vector<VertexId>&row_indices){
                VertexId src_start=column_offset[local_dst];
                VertexId src_end=column_offset[local_dst+1];
                VertexId dst=subgraphs->sampled_sgs[layer]->dst()[local_dst];
                ValueType *local_input=f_output_grad_buffer+local_dst*feature_size;
//                if(graph_->partition_id == 0) {
//                    for(VertexId src_offset=src_start;
//                        src_offset<src_end;src_offset++) {
//                        VertexId local_src = subgraphs->sampled_sgs[layer]->r_i(src_offset);
//                        VertexId src=subgraphs->sampled_sgs[layer]->src()[local_src];
//                        std::printf("%d ", graph_->get_partition_id(src));
//                    }
//                    std::printf("\n");
//                }
                for(VertexId src_offset=src_start;
                        src_offset<src_end;src_offset++){
                    VertexId local_src=subgraphs->sampled_sgs[layer]->r_i(src_offset);

//                    auto p_id = graph_->partition_id;
////                    assert(local_src >= graph_->partition_offset[p_id] && local_src < graph_->partition_offset[p_id + 1]);
//                    if(p_id == 2) {
//                        auto actual_id = graph_->get_partition_id(local_src);
//                        std::printf("src: %u, actual_id: %u\n", local_src, actual_id);
//                    }



                    VertexId src=subgraphs->sampled_sgs[layer]->src()[local_src];
//                    assert(graph_->get_partition_id(src) == graph_->partition_id);
//                    if(graph_->get_partition_id(src) == graph_->partition_id){
                        ValueType *local_output=f_input_grad_buffer+local_src*feature_size;
                        nts_acc(local_output, local_input,nts_norm_degree(graph_,src, dst), feature_size);
//                    }
                }
              },
            layer,
            12//compute thread num;
        );
       return f_input_grad;
   }

};

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
//
//class SingleCPUDstAggregateOpMin : public ntsGraphOp{
//public:
//  std::vector<CSC_segment_pinned *> subgraphs;
//  VertexId* record;
//  
//  SingleCPUDstAggregateOpMin(PartitionedGraph *partitioned_graph,VertexSubset *active)
//      : ntsGraphOp(partitioned_graph, active) {
//    subgraphs = partitioned_graph->graph_chunks;
//  }
//  ~SingleCPUDstAggregateOpMin(){
//      delete [] record;
//  }
//  NtsVar forward(NtsVar &f_input){// input edge  output vertex
//    int feature_size = f_input.size(1);
//    
//    record=new VertexId(partitioned_graph_->owned_vertices*feature_size);
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
//          nts_min(f_output_buffer+ vtx * feature_size,
//                   f_input_buffer + eid * feature_size, 
//                    record + vtx * feature_size,
//                  feature_size,eid);
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
//        nts_assign(f_input_grad_buffer, f_output_grad_buffer+feature_size*vtx,
//                record+feature_size*vtx, feature_size); 
////        for (long eid = subgraph->column_offset[vtx];
////             eid < subgraph->column_offset[vtx + 1]; eid++) {
////          VertexId src = subgraph->row_indices[eid];
////          
//////            nts_acc(f_input_grad_buffer+ (feature_size * eid),
//////                    f_output_grad_buffer + vtx * feature_size,
//////                     feature_size);
////        }
//      },
//      subgraphs, feature_size, active_);
//      return f_input_grad;
//  }    
//
//};
//
//class SingleCPUDstAggregateOpMax : public ntsGraphOp{
//public:
//  std::vector<CSC_segment_pinned *> subgraphs;
//  VertexId* record;
//  
//  SingleCPUDstAggregateOpMax(PartitionedGraph *partitioned_graph,VertexSubset *active)
//      : ntsGraphOp(partitioned_graph, active) {
//    subgraphs = partitioned_graph->graph_chunks;
//  }
//  ~SingleCPUDstAggregateOpMax(){
//      delete [] record;
//  }
//  NtsVar forward(NtsVar &f_input){// input edge  output vertex
//    int feature_size = f_input.size(1);
//    
//    record=new VertexId(partitioned_graph_->owned_vertices*feature_size);
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
//          nts_max(f_output_buffer+ vtx * feature_size,
//                   f_input_buffer + eid * feature_size, 
//                    record + vtx * feature_size,
//                  feature_size,eid);
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
//        nts_assign(f_input_grad_buffer, f_output_grad_buffer+feature_size*vtx,
//                record+feature_size*vtx, feature_size); 
////        for (long eid = subgraph->column_offset[vtx];
////             eid < subgraph->column_offset[vtx + 1]; eid++) {
////          VertexId src = subgraph->row_indices[eid];
////          
//////            nts_acc(f_input_grad_buffer+ (feature_size * eid),
//////                    f_output_grad_buffer + vtx * feature_size,
//////                     feature_size);
////        }
//      },
//      subgraphs, feature_size, active_);
//      return f_input_grad;
//  }    
//
//};
//
//
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



} // namespace graphop
} // namespace nts

#endif

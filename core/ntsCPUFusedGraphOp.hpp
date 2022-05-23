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
#ifndef NTSCPUFUSEDGRAPHOP_HPP
#define NTSCPUFUSEDGRAPHOP_HPP
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

class ForwardCPUfuseOp : public ntsGraphOp {
public:
  std::vector<CSC_segment_pinned *> subgraphs;
  ForwardCPUfuseOp(PartitionedGraph *partitioned_graph,VertexSubset *active)
      : ntsGraphOp(partitioned_graph, active) {
    subgraphs = partitioned_graph->graph_chunks;
  }
  NtsVar forward(NtsVar &f_input) {
    //f_input = input;
    NtsVar f_output = graph_->Nts->NewKeyTensor(f_input, torch::DeviceType::CPU);
    ValueType *f_input_buffer =
        graph_->Nts->getWritableBuffer(f_input, torch::DeviceType::CPU);
    ValueType *f_output_buffer =
        graph_->Nts->getWritableBuffer(f_output, torch::DeviceType::CPU);
    memset(f_output_buffer, 0,
           sizeof(ValueType) * f_input.size(0) * f_input.size(1));
    int feature_size = f_input.size(1);
    graph_->process_edges_forward_decoupled_mutisockets<int, ValueType>(
        [&](VertexId src, int current_send_partition) {
          if (graph_->rtminfo->lock_free) {
            VertexId src_trans = src - graph_->gnnctx->p_v_s;
            // check whether this vertex is necessary to send to
            // current_send_partition
            if (partitioned_graph_->has_mirror_at(current_send_partition,src)
//              subgraphs[current_send_partition]->get_forward_active(src_trans)
                ) {
              // get the index where we shall place the data
              // and invoke emit_buffer_lock_free to send messages
              VertexId write_index =
                  subgraphs[current_send_partition]
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
        // sparse slot.
        // accumulate incoming feature for dst
        // recv_id represent the partition ID corresponding to current subgraph
        [&](VertexId dst, CSC_segment_pinned *subgraph,
            MessageBuffer **recv_buffer, std::vector<VertexIndex> &src_index,
            VertexId recv_id) {
          VertexId dst_trans =
              dst - graph_->partition_offset[graph_->partition_id];
          // for every vertex, accumulate the incoming feature though iterating
          // column vertices
          for (long idx = subgraph->column_offset[dst_trans];
               idx < subgraph->column_offset[dst_trans + 1]; idx++) {
            VertexId src = subgraph->row_indices[idx];
            VertexId src_trans = src - graph_->partition_offset[recv_id];
            // fetch input from recv buffer
            // bufferIndex indicate which socket we've placed the data
            // positionIndex indicate the index of the buffer of that socket
            ValueType *local_input =
                (ValueType *)(recv_buffer[src_index[src_trans].bufferIndex]
                                  ->data +
                              graph_->sizeofM<ValueType>(feature_size) *
                                  src_index[src_trans].positionIndex +
                              sizeof(VertexId));
            ValueType *local_output =
                f_output_buffer + dst_trans * feature_size;
            nts_comp(local_output, local_input, nts_norm_degree(graph_, src, dst),
                 feature_size);
          }
        },
        subgraphs, feature_size, active_);
    return f_output;
  }
  NtsVar backward(NtsVar &f_output_grad) {
    NtsVar f_input_grad=graph_->Nts->NewLeafTensor(f_output_grad, torch::DeviceType::CPU);
  ValueType *output_grad_buffer =
      graph_->Nts->getWritableBuffer(f_output_grad, torch::DeviceType::CPU);
  ValueType *input_grad_buffer =
      graph_->Nts->getWritableBuffer(f_input_grad, torch::DeviceType::CPU);
  memset(input_grad_buffer, 0, sizeof(ValueType) * f_output_grad.size(0) * f_output_grad.size(1));
  // int feature_size=graph_->gnnctx->layer_size[graph_->rtminfo->curr_layer];
  int feature_size = f_output_grad.size(1);
  ValueType *output_buffer = new ValueType[feature_size * graph_->threads];
  graph_->process_edges_backward_decoupled_multisockets<
      int, ValueType>( // For EACH Vertex
                       // Processing
      [&](VertexId src, VertexAdjList<Empty> outgoing_adj, VertexId thread_id,
          VertexId recv_id, VertexId socketId) { // pull
        ValueType *local_output_buffer =
            output_buffer + feature_size * thread_id;
        memset(local_output_buffer, 0, sizeof(ValueType) * feature_size);
        VertexId src_trans = src - graph_->partition_offset[recv_id];

        // iterate the outgoing vertices and combine the gradients
        for (long d_idx = subgraphs[recv_id]->row_offset[src_trans];
             d_idx < subgraphs[recv_id]->row_offset[src_trans + 1]; d_idx++) {
          VertexId dst = subgraphs[recv_id]->column_indices[d_idx];

          // FIXME: will this work?
          if ((dst < graph_->local_partition_offset[socketId]) ||
              (dst >= graph_->local_partition_offset[socketId + 1])) {
            continue;
          }
          VertexId dst_trans = dst - graph_->gnnctx->p_v_s;
          ValueType *local_input_buffer =
             output_grad_buffer + (dst_trans)*feature_size;
          nts_comp(local_output_buffer, local_input_buffer, nts_norm_degree(graph_,src, dst),
               feature_size);
        }
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
        nts_acc(input_grad_buffer + (src - graph_->gnnctx->p_v_s) * feature_size, msg, feature_size);
        return 1;
      },
      feature_size, active_);
  delete[] output_buffer;
  return f_input_grad;
  
  }
};
} // namespace graphop
} // namespace nts

#endif

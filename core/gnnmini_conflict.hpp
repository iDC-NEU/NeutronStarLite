/*
Copyright (c) 2014-2015 Xiaowei Zhu, Tsinghua University

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
#ifndef GNNMINI_HPP
#define GNNMINI_HPP
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "graph.hpp"
#include <map>
#include <math.h>
#include <unistd.h>
//#include "comm/Network.hpp"
//#include "cuda/test.hpp"
#include "core/input.hpp"
#include <stack>

class GNNDatum {
public:
  gnncontext *gnnctx;
  ValueType *local_feature;
  long *local_label;
  int *local_mask;
  Graph<Empty> *graph;

  // train:    0
  // eval:     1
  // test:     2
  // initialize GNN Data using GNNContext and Graph
  // allocating space to saving data. e.g. local feature, local label
  GNNDatum(gnncontext *_gnnctx, Graph<Empty> *graph_) {
    gnnctx = _gnnctx;
    local_feature = new ValueType[gnnctx->l_v_num * gnnctx->layer_size[0]];
    local_label = new long[gnnctx->l_v_num];
    local_mask = new int[gnnctx->l_v_num];
    memset(local_mask, 0, sizeof(int) * gnnctx->l_v_num);
    graph = graph_;
  }
  void random_generate() {
    for (int i = 0; i < gnnctx->l_v_num; i++) {
      for (int j = 0; j < gnnctx->layer_size[0]; j++) {
        local_feature[i * gnnctx->layer_size[0] + j] = 1.0;
      }
      local_label[i] = rand() % gnnctx->label_num;
      local_mask[i] = i % 3;
    }
  }
  void registLabel(NtsVar &target) {
    target = graph->Nts->NewLeafKLongTensor(local_label, {gnnctx->l_v_num});
    // torch::from_blob(local_label, gnnctx->l_v_num, torch::kLong);
  }
  void registMask(NtsVar &mask) {
    mask = graph->Nts->NewLeafKIntTensor(local_mask, {gnnctx->l_v_num, 1});
    // torch::from_blob(local_mask, {gnnctx->l_v_num,1}, torch::kInt32);
  }
  void readFtrFrom1(std::string inputF, std::string inputL) {

    std::string str;
    std::ifstream input_ftr(inputF.c_str(), std::ios::in);
    std::ifstream input_lbl(inputL.c_str(), std::ios::in);
    // std::ofstream outputl("cora.labeltable",std::ios::out);
    // ID    F   F   F   F   F   F   F   L
    if (!input_ftr.is_open()) {
      std::cout << "open feature file fail!" << std::endl;
      return;
    }
    if (!input_lbl.is_open()) {
      std::cout << "open label file fail!" << std::endl;
      return;
    }
    ValueType *con_tmp = new ValueType[gnnctx->layer_size[0]];
    std::string la;
    // std::cout<<"finish1"<<std::endl;
    VertexId id = 0;
    while (input_ftr >> id) {
      VertexId size_0 = gnnctx->layer_size[0];
      VertexId id_trans = id - gnnctx->p_v_s;
      if ((gnnctx->p_v_s <= id) && (gnnctx->p_v_e > id)) {
        for (int i = 0; i < size_0; i++) {
          input_ftr >> local_feature[size_0 * id_trans + i];
        }
        input_lbl >> la;
        input_lbl >> local_label[id_trans];
        local_mask[id_trans] = id % 3;
      } else {
        for (int i = 0; i < size_0; i++) {
          input_ftr >> con_tmp[i];
        }
        input_lbl >> la;
        input_lbl >> la;
      }
    }
    free(con_tmp);
    input_ftr.close();
    input_lbl.close();
  }
  void readFeature_Label_Mask(std::string inputF, std::string inputL,
                              std::string inputM) {

    std::string str;
    std::ifstream input_ftr(inputF.c_str(), std::ios::in);
    std::ifstream input_lbl(inputL.c_str(), std::ios::in);
    std::ifstream input_msk(inputM.c_str(), std::ios::in);
    // std::ofstream outputl("cora.labeltable",std::ios::out);
    // ID    F   F   F   F   F   F   F   L
    if (!input_ftr.is_open()) {
      std::cout << "open feature file fail!" << std::endl;
      return;
    }
    if (!input_lbl.is_open()) {
      std::cout << "open label file fail!" << std::endl;
      return;
    }
    if (!input_msk.is_open()) {
      std::cout << "open mask file fail!" << std::endl;
      return;
    }
    ValueType *con_tmp = new ValueType[gnnctx->layer_size[0]];
    std::string la;
    // std::cout<<"finish1"<<std::endl;
    VertexId id = 0;
    while (input_ftr >> id) {
      VertexId size_0 = gnnctx->layer_size[0];
      VertexId id_trans = id - gnnctx->p_v_s;
      if ((gnnctx->p_v_s <= id) && (gnnctx->p_v_e > id)) {
        for (int i = 0; i < size_0; i++) {
          input_ftr >> local_feature[size_0 * id_trans + i];
        }
        input_lbl >> la;
        input_lbl >> local_label[id_trans];

        input_msk >> la;
        std::string msk;
        input_msk >> msk;
        // std::cout<<la<<" "<<msk<<std::endl;
        if (msk.compare("train") == 0) {
          local_mask[id_trans] = 0;
        } else if (msk.compare("eval") == 0 || msk.compare("val") == 0) {
          local_mask[id_trans] = 1;
        } else if (msk.compare("test") == 0) {
          local_mask[id_trans] = 2;
        } else {
          local_mask[id_trans] = 3;
        }

      } else {
        for (int i = 0; i < size_0; i++) {
          input_ftr >> con_tmp[i];
        }

        input_lbl >> la;
        input_lbl >> la;

        input_msk >> la;
        input_msk >> la;
      }
    }
    delete[] con_tmp;
    input_ftr.close();
    input_lbl.close();
  }
};

class GraphOperation {

public:
  Graph<Empty> *graph_;
  VertexSubset *active_;
  VertexId start_, end_, range_;

  int *size_at_layer;

  GraphOperation(Graph<Empty> *graph, VertexSubset *active) {
    graph_ = graph;
    active_ = active;
    start_ = graph->gnnctx->p_v_s;
    end_ = graph->gnnctx->p_v_e;
    range_ = end_ - start_;
  }
  void comp(ValueType *input, ValueType *output, ValueType weight,
            int feat_size) {
    for (int i = 0; i < feat_size; i++) {
      output[i] += input[i] * weight;
    }
  }
  void acc(ValueType *input, ValueType *output, int feat_size) {
    for (int i = 0; i < feat_size; i++) {
      write_add(&output[i], input[i]);
    }
  }
  void copy(ValueType *b_dst, long d_offset, ValueType *b_src,
            VertexId s_offset, int feat_size) {
    VertexId length = sizeof(ValueType) * feat_size;
    //      LOG_INFO("length %d feat_size %d d_offset %d s_offset
    //      %d\n",length,feat_size,d_offset,s_offset);
    memcpy((char *)b_dst + d_offset * length, (char *)b_src + s_offset * length,
           length);
  }
  ValueType norm_degree(VertexId src, VertexId dst) {
    return 1 / ((ValueType)std::sqrt(graph_->out_degree_for_backward[src]) *
                (ValueType)std::sqrt(graph_->in_degree_for_backward[dst]));
  }
  ValueType out_degree(VertexId v) {
    return (ValueType)(graph_->out_degree_for_backward[v]);
  }
  ValueType in_degree(VertexId v) {
    return (ValueType)(graph_->in_degree_for_backward[v]);
  }

  void ProcessForwardCPU(
      NtsVar &X, NtsVar &Y, std::vector<CSC_segment_pinned *> &subgraphs,
      std::function<ValueType(VertexId &, VertexId &)> weight_fun) {
    ValueType *X_buffer =
        graph_->Nts->getWritableBuffer(X, torch::DeviceType::CPU);
    ValueType *Y_buffer =
        graph_->Nts->getWritableBuffer(Y, torch::DeviceType::CPU);
    memset(Y_buffer, 0, sizeof(ValueType) * X.size(0) * X.size(1));
    int feature_size = graph_->gnnctx->layer_size[graph_->rtminfo->curr_layer];

    // graph_->process_edges_forward_debug<int,float>( // For EACH Vertex
    // Processing
    graph_->process_edges_forward_decoupled_dynamic_length<
        int, ValueType>( // For EACH Vertex Processing
        [&](VertexId src) {
          // graph_->emit_buffer(src,
          // X_buffer+(src-graph_->gnnctx->p_v_s)*feature_size, feature_size);
          graph_->NtsComm->emit_buffer(
              src, X_buffer + (src - graph_->gnnctx->p_v_s) * feature_size,
              feature_size);
        },
        [&](VertexId dst, CSC_segment_pinned *subgraph, char *recv_buffer,
            std::vector<VertexId> &src_index, VertexId recv_id) {
          VertexId dst_trans =
              dst - graph_->partition_offset[graph_->partition_id];
          for (long idx = subgraph->column_offset[dst_trans];
               idx < subgraph->column_offset[dst_trans + 1]; idx++) {
            VertexId src = subgraph->row_indices[idx];
            VertexId src_trans = src - graph_->partition_offset[recv_id];
            ValueType *local_input =
                (ValueType *)(recv_buffer +
                              graph_->sizeofM<ValueType>(feature_size) *
                                  src_index[src_trans] +
                              sizeof(VertexId));
            ValueType *local_output = Y_buffer + dst_trans * feature_size;
            //                    if(dst==0&&recv_id==0){
            //                        printf("DEBUGGGG%d :%d
            //                        %f\n",feature_size,subgraph->column_offset[dst_trans+1]-subgraph->column_offset[dst_trans],local_input[7]);
            //                    }
            comp(local_input, local_output, weight_fun(src, dst), feature_size);
          }
        },
        subgraphs, feature_size, active_);
  } // graph propagation engine

  void LocalScatter(NtsVar &X, NtsVar &Ei,
                    std::vector<CSC_segment_pinned *> &subgraphs,
                    bool bi_scatter = false) {
    ValueType *X_buffer =
        graph_->Nts->getWritableBuffer(X, torch::DeviceType::CPU);
    ValueType *Ei_buffer =
        graph_->Nts->getWritableBuffer(Ei, torch::DeviceType::CPU);

    if (!bi_scatter && X.size(1) == Ei.size(1))
      assert(false);
    memset(Ei_buffer, 0, sizeof(ValueType) * Ei.size(0) * Ei.size(1));
    if (bi_scatter && ((2 * X.size(1)) != Ei.size(1))) {
      // LOG_INFO("bi_scatter ERROR: X size: %d, Ei
      // size:%d",X.size(1),Ei.size(1));
      assert(false);
    }
    int feature_size = X.size(1);
    graph_->local_vertex_operation<int, ValueType>( // For EACH Vertex
        [&](VertexId vtx, CSC_segment_pinned *subgraph, VertexId recv_id) {
          for (long eid = subgraph->column_offset[vtx];
               eid < subgraph->column_offset[vtx + 1]; eid++) {
            VertexId src = subgraph->row_indices[eid];
            assert(0 <= src && src < graph_->vertices);
            assert(0 <= eid && eid < graph_->edges);
            //            LOG_INFO("src:%d dst%d, e offset %d",src,vtx,eid);
            if (bi_scatter) {
              copy(Ei_buffer, eid * 2, X_buffer, src, feature_size);
              copy(Ei_buffer, eid * 2 + 1, X_buffer, vtx, feature_size);
            } else {
              copy(Ei_buffer, eid, X_buffer, src, feature_size);
            }
          }
        },
        subgraphs, feature_size, active_);
  }

  void LocalAggregate(NtsVar &Ei, NtsVar &Y,
                      std::vector<CSC_segment_pinned *> &subgraphs,
                      bool bi_scatter = false) {
    ValueType *Y_buffer =
        graph_->Nts->getWritableBuffer(Y, torch::DeviceType::CPU);
    ValueType *Ei_buffer =
        graph_->Nts->getWritableBuffer(Ei, torch::DeviceType::CPU);
    memset(Y_buffer, 0, sizeof(ValueType) * Y.size(0) * Y.size(1));
    // int feature_size=graph_->gnnctx->layer_size[graph_->rtminfo->curr_layer];
    int feature_size = Y.size(1);
    graph_->local_vertex_operation<int, ValueType>( // For EACH Vertex
        [&](VertexId vtx, CSC_segment_pinned *subgraph, VertexId recv_id) {
          for (long eid = subgraph->column_offset[vtx];
               eid < subgraph->column_offset[vtx + 1]; eid++) {
            VertexId src = subgraph->row_indices[eid];
            assert(0 <= vtx && vtx < graph_->vertices);
            assert(0 <= src && src < graph_->vertices);
            assert(0 <= eid && eid < graph_->edges);
            acc(Ei_buffer + eid * feature_size, Y_buffer + vtx * feature_size,
                feature_size);
          }
        },
        subgraphs, feature_size, active_);
  }

  //
  void
  PropagateForwardCPU_Lockfree(NtsVar &X, NtsVar &Y,
                               std::vector<CSC_segment_pinned *> &subgraphs) {
    ValueType *X_buffer =
        graph_->Nts->getWritableBuffer(X, torch::DeviceType::CPU);
    ValueType *Y_buffer =
        graph_->Nts->getWritableBuffer(Y, torch::DeviceType::CPU);
    memset(Y_buffer, 0, sizeof(ValueType) * X.size(0) * X.size(1));
    // int feature_size=graph_->gnnctx->layer_size[graph_->rtminfo->curr_layer];
    int feature_size = X.size(1);
    graph_->process_edges_forward_decoupled<int, ValueType>( // For EACH Vertex
                                                             // Processing
        [&](VertexId src, int current_send_partition) {
          if (graph_->rtminfo->lock_free) {
            VertexId src_trans = src - graph_->gnnctx->p_v_s;
            if (subgraphs[current_send_partition]->get_forward_active(
                    src_trans)) {
              VertexId write_index = subgraphs[current_send_partition]
                                         ->forward_message_index[src_trans];
              graph_->NtsComm->emit_buffer_lock_free(
                  src, X_buffer + (src - graph_->gnnctx->p_v_s) * feature_size,
                  write_index, feature_size);
            }
          } else {
            graph_->NtsComm->emit_buffer(
                src, X_buffer + (src - graph_->gnnctx->p_v_s) * feature_size,
                feature_size);
          }
        },
        [&](VertexId dst, CSC_segment_pinned *subgraph, char *recv_buffer,
            std::vector<VertexId> &src_index, VertexId recv_id) {
          VertexId dst_trans =
              dst - graph_->partition_offset[graph_->partition_id];
          for (long idx = subgraph->column_offset[dst_trans];
               idx < subgraph->column_offset[dst_trans + 1]; idx++) {
            VertexId src = subgraph->row_indices[idx];
            VertexId src_trans = src - graph_->partition_offset[recv_id];
            ValueType *local_input =
                (ValueType *)(recv_buffer +
                              graph_->sizeofM<ValueType>(feature_size) *
                                  src_index[src_trans] +
                              sizeof(VertexId));
            ValueType *local_output = Y_buffer + dst_trans * feature_size;
            comp(local_input, local_output, norm_degree(src, dst),
                 feature_size);
          }
        },
        subgraphs, feature_size, active_);
  }

  void PropagateForwardCPU_Lockfree_multisockets(
      NtsVar &X, NtsVar &Y, std::vector<CSC_segment_pinned *> &subgraphs) {
    // get access to the raw data
    ValueType *X_buffer =
        graph_->Nts->getWritableBuffer(X, torch::DeviceType::CPU);
    ValueType *Y_buffer =
        graph_->Nts->getWritableBuffer(Y, torch::DeviceType::CPU);
    memset(Y_buffer, 0, sizeof(ValueType) * X.size(0) * X.size(1));
    // int feature_size=graph_->gnnctx->layer_size[graph_->rtminfo->curr_layer];
    int feature_size = X.size(1);
    graph_->process_edges_forward_decoupled_mutisockets<
        int, ValueType>( // For EACH Vertex
                         // Processing
        // sparse signal
        // for every vertex, send it's data to corresponding mirror node at
        // current_send_partition
        [&](VertexId src, int current_send_partition) {
          if (graph_->rtminfo->lock_free) {
            VertexId src_trans = src - graph_->gnnctx->p_v_s;
            // first check does this node still active.
            // i.e. does it need to send messages
            if (subgraphs[current_send_partition]->get_forward_active(
                    src_trans)) {
              // get the index where we shall place the data
              // and invoke emit_buffer_lock_free to send messages
              VertexId write_index =
                  subgraphs[current_send_partition]
                      ->forward_multisocket_message_index[src_trans];
              graph_->NtsComm->emit_buffer_lock_free(
                  src, X_buffer + (src - graph_->gnnctx->p_v_s) * feature_size,
                  write_index, feature_size);
            }
          } else {
            // send to mirror directly
            graph_->NtsComm->emit_buffer(
                src, X_buffer + (src - graph_->gnnctx->p_v_s) * feature_size,
                feature_size);
          }
        },
        // sparse slot.
        // accumulate incoming feature for dst
        [&](VertexId dst, CSC_segment_pinned *subgraph,
            MessageBuffer **recv_buffer, std::vector<VertexIndex> &src_index,
            VertexId recv_id) {
          VertexId dst_trans =
              dst - graph_->partition_offset[graph_->partition_id];
          // for every vertex, accumulate the incoming feature though iterating
          // column vertices
          for (long idx = subgraph->column_offset[dst_trans];
               idx < subgraph->column_offset[dst_trans + 1]; idx++) {
            // Question here, i think recv_id represent the partition ID while
            // the corresponding partition own the src. But we are subtracting
            // this partition_offset with arbitrary src Won't this cause
            // segmentation fault?
            VertexId src = subgraph->row_indices[idx];
            VertexId src_trans = src - graph_->partition_offset[recv_id];
            // fetch input from recv buffer
            ValueType *local_input =
                (ValueType *)(recv_buffer[src_index[src_trans].bufferIndex]
                                  ->data +
                              graph_->sizeofM<ValueType>(feature_size) *
                                  src_index[src_trans].positionIndex +
                              sizeof(VertexId));
            ValueType *local_output = Y_buffer + dst_trans * feature_size;
            // should we use edge_weight instead of norm_degree?
            comp(local_input, local_output, norm_degree(src, dst),
                 feature_size);
          }
        },
        subgraphs, feature_size, active_);
  }

  void
  PropagateBackwardCPU_Lockfree(NtsVar &X_grad, NtsVar &Y_grad,
                                std::vector<CSC_segment_pinned *> &subgraphs) {
    ValueType *X_grad_buffer =
        graph_->Nts->getWritableBuffer(X_grad, torch::DeviceType::CPU);
    ValueType *Y_grad_buffer =
        graph_->Nts->getWritableBuffer(Y_grad, torch::DeviceType::CPU);
    memset(Y_grad_buffer, 0,
           sizeof(ValueType) * X_grad.size(0) * X_grad.size(1));
    // int feature_size=graph_->gnnctx->layer_size[graph_->rtminfo->curr_layer];
    int feature_size = X_grad.size(1);
    ValueType *output_buffer = new ValueType[feature_size * graph_->threads];
    graph_->process_edges_backward_decoupled<int, ValueType>( // For EACH Vertex
                                                              // Processing
        [&](VertexId src, VertexAdjList<Empty> outgoing_adj, VertexId thread_id,
            VertexId recv_id) { // pull
          ValueType *local_output_buffer =
              output_buffer + feature_size * thread_id;
          memset(local_output_buffer, 0, sizeof(ValueType) * feature_size);
          VertexId src_trans = src - graph_->partition_offset[recv_id];
          for (long d_idx = subgraphs[recv_id]->row_offset[src_trans];
               d_idx < subgraphs[recv_id]->row_offset[src_trans + 1]; d_idx++) {
            VertexId dst = subgraphs[recv_id]->column_indices[d_idx];
            VertexId dst_trans = dst - start_;
            ValueType *local_input_buffer =
                X_grad_buffer + (dst_trans)*feature_size;
            comp(local_input_buffer, local_output_buffer, norm_degree(src, dst),
                 feature_size);
          }
          if (graph_->rtminfo->lock_free) {
            if (subgraphs[recv_id]->source_active->get_bit(src_trans)) {
              VertexId write_index =
                  subgraphs[recv_id]->backward_message_index[src_trans];
              graph_->NtsComm->emit_buffer_lock_free(src, local_output_buffer,
                                                     write_index, feature_size);
            }
          } else {
            graph_->NtsComm->emit_buffer(src, local_output_buffer,
                                         feature_size);
          }
        },
        [&](VertexId src, ValueType *msg) {
          acc(msg, Y_grad_buffer + (src - start_) * feature_size, feature_size);
          return 1;
        },
        feature_size, active_);
    delete[] output_buffer;
  }

  void PropagateBackwardCPU_Lockfree_multisockets(
      NtsVar &X_grad, NtsVar &Y_grad,
      std::vector<CSC_segment_pinned *> &subgraphs) {
    ValueType *X_grad_buffer =
        graph_->Nts->getWritableBuffer(X_grad, torch::DeviceType::CPU);
    ValueType *Y_grad_buffer =
        graph_->Nts->getWritableBuffer(Y_grad, torch::DeviceType::CPU);
    memset(Y_grad_buffer, 0,
           sizeof(ValueType) * X_grad.size(0) * X_grad.size(1));
    // int feature_size=graph_->gnnctx->layer_size[graph_->rtminfo->curr_layer];
    int feature_size = X_grad.size(1);
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
          for (long d_idx = subgraphs[recv_id]->row_offset[src_trans];
               d_idx < subgraphs[recv_id]->row_offset[src_trans + 1]; d_idx++) {
            VertexId dst = subgraphs[recv_id]->column_indices[d_idx];
            if ((dst < graph_->local_partition_offset[socketId]) ||
                (dst >= graph_->local_partition_offset[socketId + 1]))
              continue;
            VertexId dst_trans = dst - start_;
            ValueType *local_input_buffer =
                X_grad_buffer + (dst_trans)*feature_size;
            comp(local_input_buffer, local_output_buffer, norm_degree(src, dst),
                 feature_size);
          }
          if (graph_->rtminfo->lock_free) {
            if (subgraphs[recv_id]->source_active->get_bit(src_trans)) {
              VertexId write_index =
                  subgraphs[recv_id]
                      ->backward_multisocket_message_index[src_trans]
                      .vertexSocketPosition[socketId];
              graph_->NtsComm->emit_buffer_lock_free(src, local_output_buffer,
                                                     write_index, feature_size);
            }
          } else {
            graph_->NtsComm->emit_buffer(src, local_output_buffer,
                                         feature_size);
          }
        },
        [&](VertexId src, ValueType *msg) {
          acc(msg, Y_grad_buffer + (src - start_) * feature_size, feature_size);
          return 1;
        },
        feature_size, active_);
    delete[] output_buffer;
  }

  void GetFromDepNeighbor(NtsVar &X, std::vector<NtsVar> &Y_list,
                          std::vector<CSC_segment_pinned *> &subgraphs) {
    ValueType *X_buffer =
        graph_->Nts->getWritableBuffer(X, torch::DeviceType::CPU);
    // float* Y_buffer=graph_->Nts->getWritableBuffer(Y,torch::DeviceType::CPU);
    // memset(Y_buffer,0,sizeof(float)*X.size(0)*X.size(1));
    // int feature_size=graph_->gnnctx->layer_size[graph_->rtminfo->curr_layer];
    ValueType **Y_buffer = new ValueType *[graph_->partitions];
    for (int i = 0; i < graph_->partitions; i++) {
      Y_buffer[i] =
          graph_->Nts->getWritableBuffer(Y_list[i], torch::DeviceType::CPU);
    }
    int feature_size = X.size(1);
    graph_->get_from_dep_neighbor<int, ValueType>( // For EACH Vertex Processing
        [&](VertexId src, int current_send_partition) {
          if (graph_->rtminfo->lock_free) {
            VertexId src_trans = src - graph_->gnnctx->p_v_s;
            if (subgraphs[current_send_partition]->get_forward_active(
                    src_trans)) {
              VertexId write_index = subgraphs[current_send_partition]
                                         ->forward_message_index[src_trans];
              graph_->NtsComm->emit_buffer_lock_free(
                  src, X_buffer + (src - graph_->gnnctx->p_v_s) * feature_size,
                  write_index, feature_size);
            }
          } else {
            graph_->NtsComm->emit_buffer(
                src, X_buffer + (src - graph_->gnnctx->p_v_s) * feature_size,
                feature_size);
          }
        },
        [&](VertexId dst, ValueType *recv_buffer, VertexId recv_id) {
          VertexId dst_trans = dst - graph_->partition_offset[recv_id];
          // Y_list[recv_id]
          memcpy(Y_buffer[recv_id] + dst_trans * feature_size, recv_buffer,
                 sizeof(ValueType) * feature_size);
          return 0;
        },
        subgraphs, feature_size, active_);
    delete[] Y_buffer;
  }

  void PostToDepNeighbor(std::vector<NtsVar> &X_grad_list, NtsVar &Y_grad,
                         std::vector<CSC_segment_pinned *> &subgraphs) {
    // float*
    // X_grad_buffer=graph_->Nts->getWritableBuffer(X_grad,torch::DeviceType::CPU);
    ValueType **X_grad_buffer = new ValueType *[graph_->partitions];
    for (int i = 0; i < graph_->partitions; i++) {
      X_grad_buffer[i] = graph_->Nts->getWritableBuffer(X_grad_list[i],
                                                        torch::DeviceType::CPU);
    }
    ValueType *Y_grad_buffer =
        graph_->Nts->getWritableBuffer(Y_grad, torch::DeviceType::CPU);
    memset(Y_grad_buffer, 0,
           sizeof(ValueType) * Y_grad.size(0) * Y_grad.size(1));
    // int feature_size=graph_->gnnctx->layer_size[graph_->rtminfo->curr_layer];
    int feature_size = Y_grad.size(1);
    graph_->process_edges_backward_decoupled<int, ValueType>( // For EACH Vertex
                                                              // Processing
        [&](VertexId src, VertexAdjList<Empty> outgoing_adj, VertexId thread_id,
            VertexId recv_id) { // pull
          VertexId src_trans = src - graph_->partition_offset[recv_id];
          ValueType *local_output_buffer =
              X_grad_buffer[recv_id] + src_trans * feature_size;

          if (graph_->rtminfo->lock_free) {
            if (subgraphs[recv_id]->source_active->get_bit(src_trans)) {
              VertexId write_index =
                  subgraphs[recv_id]->backward_message_index[src_trans];
              graph_->NtsComm->emit_buffer_lock_free(src, local_output_buffer,
                                                     write_index, feature_size);
            }
          } else {
            graph_->NtsComm->emit_buffer(src, local_output_buffer,
                                         feature_size);
          }
        },
        [&](VertexId src, ValueType *msg) {
          acc(msg, Y_grad_buffer + (src - start_) * feature_size, feature_size);
          return 1;
        },
        feature_size, active_);
  }

#if CUDA_ENABLE
  inline void
  ForwardSingle(NtsVar &X, NtsVar &Y,
                std::vector<CSC_segment_pinned *> &graph_partitions) {
    int feature_size = X.size(1);
    graph_->forward_single<int, ValueType>(X, graph_partitions, Y,
                                           feature_size);
  }
  inline void
  BackwardSingle(NtsVar &X, NtsVar &Y,
                 std::vector<CSC_segment_pinned *> &graph_partitions) {

    int feature_size = X.size(1);
    graph_->backward_single<int, ValueType>(X, graph_partitions, Y,
                                            feature_size);
  }
  void ForwardAggMessage(NtsVar &src_input_transferred, NtsVar &dst_output,
                         std::vector<CSC_segment_pinned *> &graph_partitions) {
    int feature_size = src_input_transferred.size(1);
    graph_->forward_single_edge<int, ValueType>(
        src_input_transferred, graph_partitions, dst_output, feature_size);
  }
  void
  BackwardScatterMessage(NtsVar &dst_grad_input, NtsVar &msg_grad_output,
                         std::vector<CSC_segment_pinned *> &graph_partitions) {
    int feature_size = dst_grad_input.size(
        1); //= graph_->gnnctx->layer_size[graph_->rtminfo->curr_layer];
    graph_->backward_single_edge<int, ValueType>(
        dst_grad_input, graph_partitions, msg_grad_output, feature_size);
  }

  inline void
  GraphPropagateForward(NtsVar &X, NtsVar &Y,
                        std::vector<CSC_segment_pinned *> &graph_partitions) {
    int feature_size = X.size(1);
    NtsVar X_cpu = X.cpu();
    ValueType *X_buffered = X_cpu.accessor<ValueType, 2>().data();

    { // original communication
      graph_->sync_compute_decoupled<int, ValueType>(
          X, graph_partitions,
          [&](VertexId src) {
            graph_->NtsComm->emit_buffer(
                src, X_buffered + (src - graph_->gnnctx->p_v_s) * feature_size,
                feature_size);
          },
          Y, feature_size);
    }
  }

  inline void
  GraphPropagateBackward(NtsVar &X, NtsVar &Y,
                         std::vector<CSC_segment_pinned *> &graph_partitions) {
    int feature_size = X.size(1);
    // if (!selective)
    {
      graph_->compute_sync_decoupled<int, ValueType>(
          X, graph_partitions,
          [&](VertexId src, VertexAdjList<Empty> outgoing_adj) { // pull
            graph_->NtsComm->emit_buffer(
                src, graph_->output_cpu_buffer + (src)*feature_size,
                feature_size);
          },
          Y, feature_size);
    }
  }

  void PropagateForwardEdgeGPU(
      NtsVar &src_input_transferred, NtsVar &dst_output,
      std::vector<CSC_segment_pinned *> &graph_partitions,
      std::function<NtsVar(NtsVar &, NtsVar &, NtsScheduler *nts)>
          EdgeComputation) {
    int feature_size = src_input_transferred.size(1);
    NtsVar X_cpu = src_input_transferred.cpu();
    ValueType *X_buffered = X_cpu.accessor<ValueType, 2>().data();
    graph_->sync_compute_decoupled_edge<int, ValueType>(
        src_input_transferred, graph_partitions,
        [&](VertexId src) {
          graph_->NtsComm->emit_buffer(
              src, X_buffered + (src - graph_->gnnctx->p_v_s) * feature_size,
              feature_size);
        },
        [&](NtsVar &s_i_t, NtsVar &d_i_t, NtsScheduler *nts) {
          return EdgeComputation(s_i_t, d_i_t, nts);
        },
        dst_output, feature_size);
  }
  void PropagateBackwardEdgeGPU(
      NtsVar &src_input_origin, NtsVar &dst_grad_input, NtsVar &dst_grad_output,
      std::vector<CSC_segment_pinned *> &graph_partitions,
      std::function<NtsVar(NtsVar &, NtsVar &, NtsScheduler *nts)>
          EdgeComputation,
      std::function<NtsVar(NtsVar &, NtsVar &, NtsScheduler *nts)>
          EdgeBackward) {
    int feature_size = src_input_origin.size(
        1); //= graph_->gnnctx->layer_size[graph_->rtminfo->curr_layer];
    graph_->compute_sync_decoupled_edge<int, ValueType>(
        dst_grad_input, src_input_origin, graph_partitions,
        [&](VertexId src, VertexAdjList<Empty> outgoing_adj) {
          graph_->NtsComm->emit_buffer(
              src, graph_->output_cpu_buffer + (src)*feature_size,
              feature_size);
        },
        [&](NtsVar &s_i_t, NtsVar &d_i_t, NtsScheduler *nts) {
          return EdgeComputation(s_i_t, d_i_t, nts);
        },
        [&](NtsVar &b_i, NtsVar &c_i, NtsScheduler *nts) {
          return EdgeBackward(b_i, c_i, nts);
        },
        dst_grad_output, feature_size);
    // printf("done!\n");
  }
#endif

  void GenerateGraphSegment(
      std::vector<CSC_segment_pinned *> &graph_partitions, DeviceLocation dt,
      std::function<ValueType(VertexId, VertexId)> weight_compute) {
    graph_partitions.clear();
    int *tmp_column_offset = new int[graph_->vertices + 1];
    int *tmp_row_offset = new int[graph_->vertices + 1];
    memset(tmp_column_offset, 0, sizeof(int) * (graph_->vertices + 1)); //
    memset(tmp_row_offset, 0, sizeof(int) * (graph_->vertices + 1));    //
    for (int i = 0; i < graph_->graph_shard_in.size(); i++) {
      graph_partitions.push_back(new CSC_segment_pinned);
      graph_partitions[i]->init(graph_->graph_shard_in[i]->src_range[0],
                                graph_->graph_shard_in[i]->src_range[1],
                                graph_->graph_shard_in[i]->dst_range[0],
                                graph_->graph_shard_in[i]->dst_range[1],
                                graph_->graph_shard_in[i]->numofedges, dt);
      graph_partitions[i]->allocVertexAssociateData();
      graph_partitions[i]->allocEdgeAssociateData();
      memset(tmp_column_offset, 0, sizeof(int) * (graph_->vertices + 1));
      memset(tmp_row_offset, 0, sizeof(int) * (graph_->vertices + 1));
      for (int j = 0; j < graph_partitions[i]->edge_size; j++) {
        // note that the vertex in the same partition has the contiguous
        // vertexID so we can minus the start index to get the offset
        VertexId v_src_m = graph_->graph_shard_in[i]->src_delta[j];
        VertexId v_dst_m = graph_->graph_shard_in[i]->dst_delta[j];
        VertexId v_dst = v_dst_m - graph_partitions[i]->dst_range[0];
        VertexId v_src = v_src_m - graph_partitions[i]->src_range[0];

        // count of edges which has dst to v_dst plus one
        tmp_column_offset[v_dst + 1] += 1;
        // count of edges which has src from v_src plus one
        tmp_row_offset[v_src + 1] += 1; ///
        // graph_partitions[i]->weight_buffer[j]=(ValueType)std::sqrt(graph->out_degree_for_backward[v_src])*(ValueType)std::sqrt(graph->in_degree_for_backward[v_dst]);
      }
      // accumulate those offset, calc the partial sum
      for (int j = 0; j < graph_partitions[i]->batch_size_forward; j++) {
        tmp_column_offset[j + 1] += tmp_column_offset[j];
        graph_partitions[i]->column_offset[j + 1] = tmp_column_offset[j + 1];
      }

      for (int j = 0; j < graph_partitions[i]->batch_size_backward; j++) ///
      {
        tmp_row_offset[j + 1] += tmp_row_offset[j];
        graph_partitions[i]->row_offset[j + 1] = tmp_row_offset[j + 1];
      }

      // after calc the offset, we should place those edges now
      for (int j = 0; j < graph_partitions[i]->edge_size; j++) {
        // if(graph->partition_id==0)std::cout<<"After j edges: "<<j<<std::endl;
        VertexId v_src_m = graph_->graph_shard_in[i]->src_delta[j];
        VertexId v_dst_m = graph_->graph_shard_in[i]->dst_delta[j];
        VertexId v_dst = v_dst_m - graph_partitions[i]->dst_range[0];
        VertexId v_src = v_src_m - graph_partitions[i]->src_range[0];

        // save the src and dst in the column format
        graph_partitions[i]->source[tmp_column_offset[v_dst]] = (long)(v_src_m);
        graph_partitions[i]->destination[tmp_column_offset[v_dst]] =
            (long)(v_dst_m);
        // save the src in the row format
        graph_partitions[i]->source_backward[tmp_row_offset[v_src]] =
            (long)(v_src_m);

        graph_partitions[i]->src_set_active(
            v_src_m); // source_active->set_bit(v_src);
        graph_partitions[i]->dst_set_active(
            v_dst_m); // destination_active->set_bit(v_dst);

        // save src in CSC format, used in forward computation
        graph_partitions[i]->row_indices[tmp_column_offset[v_dst]] = v_src_m;
        graph_partitions[i]->edge_weight_forward[tmp_column_offset[v_dst]++] =
            weight_compute(v_src_m, v_dst_m);

        // save dst in CSR format, used in backward computation
        graph_partitions[i]->column_indices[tmp_row_offset[v_src]] =
            v_dst_m; ///
        graph_partitions[i]->edge_weight_backward[tmp_row_offset[v_src]++] =
            weight_compute(v_src_m, v_dst_m);
      }
      // graph_partitions[i]->getDevicePointerAll();
      graph_partitions[i]->CopyGraphToDevice();
    }
#if CUDA_ENABLE
    if (GPU_T == dt) {
      int max_batch_size = 0;
      for (int i = 0; i < graph_partitions.size(); i++) {
        max_batch_size =
            std::max(max_batch_size, graph_partitions[i]->batch_size_backward);
      }
      graph_->output_gpu_buffered = graph_->Nts->NewLeafTensor(
          {max_batch_size, graph_->gnnctx->max_layer}, torch::DeviceType::CUDA);
    }
#endif
    delete[] tmp_column_offset;
    delete[] tmp_row_offset;
    if (graph_->partition_id == 0)
      printf("GNNmini::Preprocessing[Graph Segments Prepared]\n");
  }

  void GenerateMessageBitmap_multisokects(
      std::vector<CSC_segment_pinned *>
          &graph_partitions) { // local partition offset
    int feature_size = 1;
    graph_->process_edges_backward<int, VertexId>( // For EACH Vertex Processing
        [&](VertexId src, VertexAdjList<Empty> outgoing_adj, VertexId thread_id,
            VertexId recv_id) { // pull
          VertexId src_trans = src - graph_->partition_offset[recv_id];
          if (graph_partitions[recv_id]->source_active->get_bit(src_trans)) {
            VertexId part = (VertexId)graph_->partition_id;
            graph_->emit_buffer(src, &part, feature_size);
          }
        },
        [&](VertexId master, VertexId *msg) {
          VertexId part = *msg;
          graph_partitions[part]->set_forward_active(
              master -
              graph_->gnnctx
                  ->p_v_s); // destination_mirror_active->set_bit(master-start_);
          return 0;
        },
        feature_size, active_);

    size_t basic_chunk = 64;
    for (int i = 0; i < graph_partitions.size(); i++) {
      graph_partitions[i]->forward_multisocket_message_index =
          new VertexId[graph_partitions[i]->batch_size_forward];
      memset(graph_partitions[i]->forward_multisocket_message_index, 0,
             sizeof(VertexId) * graph_partitions[i]->batch_size_forward);

      graph_partitions[i]->backward_multisocket_message_index =
          new BackVertexIndex[graph_partitions[i]->batch_size_backward];
      int socketNum = numa_num_configured_nodes();
      for (int bck = 0; bck < graph_partitions[i]->batch_size_backward; bck++) {
        graph_partitions[i]->backward_multisocket_message_index[bck].setSocket(
            socketNum);
      }

      std::vector<VertexId> socket_backward_write_offset;
      socket_backward_write_offset.resize(numa_num_configured_nodes());
      memset(socket_backward_write_offset.data(), 0,
             sizeof(VertexId) * socket_backward_write_offset.size());
#pragma omp parallel
      {
        int thread_id = omp_get_thread_num();
        int s_i = graph_->get_socket_id(thread_id);
        VertexId begin_p_v_i =
            graph_->tuned_chunks_dense_backward[i][thread_id].curr;
        VertexId final_p_v_i =
            graph_->tuned_chunks_dense_backward[i][thread_id].end;
        for (VertexId p_v_i = begin_p_v_i; p_v_i < final_p_v_i; p_v_i++) {
          VertexId v_i =
              graph_->compressed_incoming_adj_index_backward[s_i][p_v_i].vertex;
          VertexId v_trans = v_i - graph_partitions[i]->src_range[0];
          if (graph_partitions[i]->src_get_active(v_i)) {
            int position =
                __sync_fetch_and_add(&socket_backward_write_offset[s_i], 1);
            graph_partitions[i]
                ->backward_multisocket_message_index[v_trans]
                .vertexSocketPosition[s_i] = position;
          }
        }
      }

      std::vector<VertexId> socket_forward_write_offset;
      socket_forward_write_offset.resize(numa_num_configured_nodes());
      memset(socket_forward_write_offset.data(), 0,
             sizeof(VertexId) * socket_forward_write_offset.size());
#pragma omp parallel for schedule(static, basic_chunk)
      // pragma omp parallel for
      for (VertexId begin_v_i = graph_partitions[i]->dst_range[0];
           begin_v_i < graph_partitions[i]->dst_range[1]; begin_v_i++) {
        int thread_id = omp_get_thread_num();
        int s_i = graph_->get_socket_id(thread_id);
        VertexId v_i = begin_v_i;
        VertexId v_trans = v_i - graph_partitions[i]->dst_range[0];
        if (graph_partitions[i]->get_forward_active(v_trans)) {
          int position =
              __sync_fetch_and_add(&socket_forward_write_offset[s_i], 1);
          graph_partitions[i]->forward_multisocket_message_index[v_trans] =
              position;
        }
      }
      // printf("forward_write_offset %d\n",forward_write_offset);
    }
    if (graph_->partition_id == 0)
      printf("GNNmini::Preprocessing[Compressed Message Prepared]\n");
  }

  void GenerateMessageBitmap(std::vector<CSC_segment_pinned *>
                                 &graph_partitions) { // local partition offset
    int feature_size = 1;
    graph_->process_edges_backward<int, VertexId>( // For EACH Vertex Processing
        [&](VertexId src, VertexAdjList<Empty> outgoing_adj, VertexId thread_id,
            VertexId recv_id) { // pull
          VertexId src_trans = src - graph_->partition_offset[recv_id];
          if (graph_partitions[recv_id]->source_active->get_bit(src_trans)) {
            VertexId part = (VertexId)graph_->partition_id;
            graph_->emit_buffer(src, &part, feature_size);
          }
        },
        [&](VertexId master, VertexId *msg) {
          VertexId part = *msg;
          graph_partitions[part]->set_forward_active(
              master -
              graph_->gnnctx
                  ->p_v_s); // destination_mirror_active->set_bit(master-start_);
          return 0;
        },
        feature_size, active_);

    size_t basic_chunk = 64;
    for (int i = 0; i < graph_partitions.size(); i++) {
      graph_partitions[i]->backward_message_index =
          new VertexId[graph_partitions[i]->batch_size_backward];
      graph_partitions[i]->forward_message_index =
          new VertexId[graph_partitions[i]->batch_size_forward];
      memset(graph_partitions[i]->backward_message_index, 0,
             sizeof(VertexId) * graph_partitions[i]->batch_size_backward);
      memset(graph_partitions[i]->forward_message_index, 0,
             sizeof(VertexId) * graph_partitions[i]->batch_size_forward);
      int backward_write_offset = 0;

      for (VertexId begin_v_i = graph_partitions[i]->src_range[0];
           begin_v_i < graph_partitions[i]->src_range[1]; begin_v_i += 1) {
        VertexId v_i = begin_v_i;
        VertexId v_trans = v_i - graph_partitions[i]->src_range[0];
        if (graph_partitions[i]->src_get_active(v_i))
          graph_partitions[i]->backward_message_index[v_trans] =
              backward_write_offset++;
      }

      int forward_write_offset = 0;
      for (VertexId begin_v_i = graph_partitions[i]->dst_range[0];
           begin_v_i < graph_partitions[i]->dst_range[1]; begin_v_i += 1) {
        VertexId v_i = begin_v_i;
        VertexId v_trans = v_i - graph_partitions[i]->dst_range[0];
        if (graph_partitions[i]->get_forward_active(v_trans))
          graph_partitions[i]->forward_message_index[v_trans] =
              forward_write_offset++;
      }
      // printf("forward_write_offset %d\n",forward_write_offset);
    }
    if (graph_->partition_id == 0)
      printf("GNNmini::Preprocessing[Compressed Message Prepared]\n");
  }

  void TestGeneratedBitmap(std::vector<CSC_segment_pinned *> &subgraphs) {
    for (int i = 0; i < subgraphs.size(); i++) {
      int count_act_src = 0;
      int count_act_dst = 0;
      int count_act_master = 0;
      for (int j = subgraphs[i]->dst_range[0]; j < subgraphs[i]->dst_range[1];
           j++) {
        if (subgraphs[i]->dst_get_active(j)) {
          count_act_dst++;
        }
      }
      for (int j = subgraphs[i]->src_range[0]; j < subgraphs[i]->src_range[1];
           j++) {
        if (subgraphs[i]->src_get_active(j)) {
          count_act_src++;
        }
      }
      printf("PARTITION:%d CHUNK %d ACTIVE_SRC %d ACTIVE_DST %d ACTIVE_MIRROR "
             "%d\n",
             graph_->partition_id, i, count_act_src, count_act_dst,
             count_act_master);
    }
  }
};

#endif

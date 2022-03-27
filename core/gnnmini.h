#ifndef GNNMINI_H
#define GNNMINI_H
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <map>
#include <math.h>
#include <unistd.h>
#include <stack>

#include "input.h"
#include "graph.hpp"

class GNNDatum {
public:
  gnncontext *gnnctx;
  ValueType *local_feature;
  long *local_label;
  int *local_mask;
  Graph<Empty> *graph;

  GNNDatum(gnncontext *_gnnctx, Graph<Empty> *graph_);
  void random_generate();
  void registLabel(NtsVar &target);
  void registMask(NtsVar &mask);
  void readFtrFrom1(std::string inputF, std::string inputL);
  void readFeature_Label_Mask(std::string inputF, std::string inputL,
                              std::string inputM);
};

class GraphOperation {

public:
  Graph<Empty> *graph_;
  VertexSubset *active_;
  VertexId start_, end_, range_;

  int *size_at_layer;

  GraphOperation(Graph<Empty> *graph, VertexSubset *active);
  void comp(ValueType *input, ValueType *output, ValueType weight,
            int feat_size);
  void acc(ValueType *input, ValueType *output, int feat_size);
  void copy(ValueType* b_dst, long d_offset,ValueType* b_src,
         VertexId s_offset,int feat_size);
  ValueType norm_degree(VertexId src, VertexId dst);
  ValueType out_degree(VertexId v);
  ValueType in_degree(VertexId v);

  void ProcessForwardCPU(
      NtsVar &X, NtsVar &Y, std::vector<CSC_segment_pinned *> &subgraphs,
      std::function<ValueType(VertexId &, VertexId &)> weight_fun);

  // graph propagation engine

  void LocalScatter(NtsVar &X, NtsVar &Ei,
                               std::vector<CSC_segment_pinned *> &subgraphs);
  
  void LocalAggregate(NtsVar &Ei, NtsVar &Y,
                               std::vector<CSC_segment_pinned *> &subgraphs);
  
  void PropagateForwardCPU_Lockfree(NtsVar &X, NtsVar &Y,
                               std::vector<CSC_segment_pinned *> &subgraphs);

  void PropagateForwardCPU_Lockfree_multisockets(NtsVar &X, NtsVar &Y,
                               std::vector<CSC_segment_pinned *> &subgraphs);

  void PropagateBackwardCPU_Lockfree(NtsVar &X_grad, NtsVar &Y_grad,
                                std::vector<CSC_segment_pinned *> &subgraphs);

  void PropagateBackwardCPU_Lockfree_multisockets(NtsVar &X_grad, NtsVar &Y_grad,
                                std::vector<CSC_segment_pinned *> &subgraphs);

  void GetFromDepNeighbor(NtsVar &X, std::vector<NtsVar> &Y_list,
                          std::vector<CSC_segment_pinned *> &subgraphs);

  void PostToDepNeighbor(std::vector<NtsVar> &X_grad_list, NtsVar &Y_grad,
                         std::vector<CSC_segment_pinned *> &subgraphs);

#if CUDA_ENABLE
  void
  ForwardSingle(NtsVar &X, NtsVar &Y,
                std::vector<CSC_segment_pinned *> &graph_partitions);
  void
  BackwardSingle(NtsVar &X, NtsVar &Y,
                 std::vector<CSC_segment_pinned *> &graph_partitions);
  void ForwardAggMessage(NtsVar &src_input_transferred, NtsVar &dst_output,
                         std::vector<CSC_segment_pinned *> &graph_partitions);
  void
  BackwardScatterMessage(NtsVar &dst_grad_input, NtsVar &msg_grad_output,
                         std::vector<CSC_segment_pinned *> &graph_partitions);

  void
  GraphPropagateForward(NtsVar &X, NtsVar &Y,
                        std::vector<CSC_segment_pinned *> &graph_partitions);

  void
  GraphPropagateBackward(NtsVar &X, NtsVar &Y,
                         std::vector<CSC_segment_pinned *> &graph_partitions);

  void PropagateForwardEdgeGPU(
      NtsVar &src_input_transferred, NtsVar &dst_output,
      std::vector<CSC_segment_pinned *> &graph_partitions,
      std::function<NtsVar(NtsVar &, NtsVar &, NtsScheduler *nts)>
          EdgeComputation);
  void PropagateBackwardEdgeGPU(
      NtsVar &src_input_origin, NtsVar &dst_grad_input, NtsVar &dst_grad_output,
      std::vector<CSC_segment_pinned *> &graph_partitions,
      std::function<NtsVar(NtsVar &, NtsVar &, NtsScheduler *nts)>
          EdgeComputation,
      std::function<NtsVar(NtsVar &, NtsVar &, NtsScheduler *nts)>
          EdgeBackward);
#endif

  void GenerateGraphSegment(
      std::vector<CSC_segment_pinned *> &graph_partitions, DeviceLocation dt,
      std::function<ValueType(VertexId, VertexId)> weight_compute);

  void
  GenerateMessageBitmap_multisokects(std::vector<CSC_segment_pinned *> &graph_partitions);

  void
  GenerateMessageBitmap(std::vector<CSC_segment_pinned *> &graph_partitions);

  void TestGeneratedBitmap(std::vector<CSC_segment_pinned *> &subgraphs);
};

#endif

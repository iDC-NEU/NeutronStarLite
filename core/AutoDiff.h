#ifndef AUTODIFF_H
#define AUTODIFF_H
#include "gnnmini.h"

namespace nts {

namespace autodiff {

const VertexId NNOP = 0;

const VertexId DIST_CPU = 1;
const VertexId DIST_GPU = 2;

const VertexId SINGLE_CPU = 3;
const VertexId SINGLE_GPU = 4;

const VertexId SINGLE_CPU_EDGE_SCATTER =5;
const VertexId SINGLE_CPU_EDGE_GATHER =6;

const VertexId SINGLE_GPU_EDGE_SCATTER = 7;
const VertexId SINGLE_GPU_EDGE_GATHER = 8;


const VertexId DIST_CPU_EDGE = 9;
const VertexId DIST_GPU_EDGE = 10;




class ComputionPath {
public:
  ComputionPath(GraphOperation *gt_,
                std::vector<CSC_segment_pinned *> subgraphs_);
  void op_push(NtsVar &input_t, NtsVar &output_t, VertexId op_type);
  void reset();
  void pop_one_op();
  void self_backward(bool retain_graph = true);
  void debug();

private:
  std::stack<VertexId> op;
  std::stack<NtsVar> output;
  std::stack<NtsVar> input;
  std::vector<NtsVar> output_grad;
  std::vector<NtsVar> input_grad;
  int count;
  GraphOperation *gt;
  std::vector<CSC_segment_pinned *> subgraphs;
};

} // namespace autodiff
} // namespace nts

#endif

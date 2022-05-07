#ifndef AUTODIFF_H
#define AUTODIFF_H
#include "gnnmini.h"

namespace nts {

namespace autodiff {

typedef uint32_t OpType;

const OpType NNOP = 0;

const OpType DIST_CPU = 1;
const OpType DIST_GPU = 2;

const OpType SINGLE_CPU = 3;
const OpType SINGLE_GPU = 4;

const OpType SINGLE_CPU_EDGE_SCATTER = 5;
const OpType SINGLE_CPU_EDGE_GATHER = 6;

const OpType SINGLE_GPU_EDGE_SCATTER = 7;
const OpType SINGLE_GPU_EDGE_GATHER = 8;

const OpType DIST_CPU_EDGE = 9;
const OpType DIST_GPU_EDGE = 10;

/**
 * @brief
 * since GNN operation is just iteration of graph operation and NN operation.
 * so we can simply use a chain to represent GNN operation, which can reduce
 * system complexity greatly.
 * you can also regard it as the NN operation splited by graph operation.
 * And as the extention of auto diff library, we will provide backward
 * computation for graph operation. And thus, the computation path of GNN is
 * constructed.
 */
class ComputionPath {
public:
  ComputionPath(GraphOperation *gt_,
                std::vector<CSC_segment_pinned *> subgraphs_,
                bool bi_direction = false);
  void op_push(NtsVar &input_t, NtsVar &output_t, OpType op_type);
  void reset();
  void pop_one_op();
  void self_backward(bool retain_graph = true);
  void debug();
  int top_idx();

private:
  std::stack<OpType> op;
  std::stack<NtsVar> output;
  std::stack<NtsVar> input;
  std::vector<NtsVar> output_grad;
  int count;
  GraphOperation *gt;
  std::vector<CSC_segment_pinned *> subgraphs;
  bool bi_direction;
};

} // namespace autodiff
} // namespace nts

#endif

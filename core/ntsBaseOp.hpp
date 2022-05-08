#ifndef NTSBASEOP_HPP
#define NTSBASEOP_HPP
#include "graph.hpp"

namespace nts {
namespace op {

class ntsGraphOp {
public:
  Graph<Empty> *graph_;
  VertexSubset *active_;
  ntsGraphOp() {}
  ntsGraphOp(Graph<Empty> *graph, VertexSubset *active) {
    graph_ = graph;
    active_ = active;
  }
  virtual NtsVar forward(NtsVar &f_input)=0;
  virtual NtsVar backward(NtsVar &output_grad)=0;
};

} // namespace graphop
} // namespace nts

void nts_comp(ValueType *output, ValueType *input, ValueType weight,
          int feat_size) {
  for (int i = 0; i < feat_size; i++) {
    output[i] += input[i] * weight;
  }
}

/**
 * @brief
 * do output += input at feature(array) level
 * @param input input feature
 * @param output output feature
 * @param feat_size feature size
 */
void nts_acc(ValueType *output, ValueType *input, int feat_size) {
  for (int i = 0; i < feat_size; i++) {
    // atomic add
    write_add(&output[i], input[i]);
  }
}

/**
 * @brief
 * copy feature_size elements from b_src[d_offset * feature_size]
 * to d_dst[s_offset * feature_size]
 * @param b_dst dst buffer
 * @param d_offset dst offset, should be a vertex id
 * @param b_src src buffer
 * @param s_offset src offset, should be a vertex id
 * @param feat_size feature size that every vertex have
 */
void nts_copy(ValueType *b_dst, long d_offset, ValueType *b_src, VertexId s_offset,
          int feat_size) {
  // length is the byte level space cost for a vertex feature data
  VertexId length = sizeof(ValueType) * feat_size;
  // LOG_INFO("length %d feat_size %d d_offset %d s_offset
  // %d\n",length,feat_size,d_offset,s_offset);
  memcpy((char *)b_dst + d_offset * length, (char *)b_src + s_offset * length,
         length);
}

/**
 * @brief
 * return 1 / sqrt(out_degree[src] * in_degree[dst]).
 * normally used as edge weight
 * @param src src id
 * @param dst dst id
 * @return ValueType
 */
ValueType nts_norm_degree(Graph<Empty> *graph_, VertexId src, VertexId dst) {
  return 1 / ((ValueType)std::sqrt(graph_->out_degree_for_backward[src]) *
              (ValueType)std::sqrt(graph_->in_degree_for_backward[dst]));
}

/**
 * @brief
 * get out degree for v
 * @param v vertex id
 * @return ValueType
 */
ValueType nts_out_degree(Graph<Empty> *graph_, VertexId v) {
  return (ValueType)(graph_->out_degree_for_backward[v]);
}

/**
 * @brief
 * get in degree for v
 * @param v vertex id
 * @return ValueType
 */
ValueType nts_in_degree(Graph<Empty> *graph_, VertexId v) {
  return (ValueType)(graph_->in_degree_for_backward[v]);
}

//class ntsNNOp {
//public:
//
//  ntsNNOp() {}
//  virtual NtsVar &forward(NtsVar &input) = 0;
//  virtual NtsVar backward(NtsVar &output_grad) = 0;
//};

#endif

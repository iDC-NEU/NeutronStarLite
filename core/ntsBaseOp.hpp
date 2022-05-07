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

//class ntsNNOp {
//public:
//
//  ntsNNOp() {}
//  virtual NtsVar &forward(NtsVar &input) = 0;
//  virtual NtsVar backward(NtsVar &output_grad) = 0;
//};

#endif

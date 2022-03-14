/*
Copyright (c) 2015-2016 Xiaowei Zhu, Tsinghua University

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

#ifndef AUTODIFF_HPP
#define AUTODIFF_HPP
#include "gnnmini.hpp"

namespace nts {

namespace autodiff {

const VertexId NNOP = 0;

const VertexId DIST_CPU = 1;
const VertexId DIST_CPU_EDGE = 2;

const VertexId DIST_GPU = 3;
const VertexId DIST_GPU_EDGE = 4;

const VertexId SINGLE_CPU = 5;
const VertexId SINGLE_CPU_EDGE = 6;

const VertexId SINGLE_GPU = 7;
const VertexId SINGLE_GPU_EDGE = 8;

class ComputionPath {
public:
  ComputionPath(GraphOperation *gt_,
                std::vector<CSC_segment_pinned *> subgraphs_) {
    op.empty();
    output.empty();
    input.empty();
    output_grad.clear();
    input_grad.clear();
    count = 0;
    gt = gt_;
    subgraphs = subgraphs_;
  }
  void op_push(NtsVar &input_t, NtsVar &output_t, VertexId op_type) {
    // LOG_INFO("OP PUSH%d",op_type);
    NtsVar ig;
    NtsVar og;

    assert(op_type < 9);

    if (count > 0 && NNOP == op_type && op.top() == NNOP) {
      output.pop();
      output.push(output_t);
      //     LOG_INFO("TRIG");
    } else {
      count++;
      op.push(op_type);
      output.push(output_t);
      input.push(input_t);
      output_grad.push_back(ig);
      input_grad.push_back(og);
    }
  }
  void reset() {
    assert(0 == count);
    op.empty();
    output.empty();
    input.empty();
    output_grad.empty();
    input_grad.empty();
  }
  void pop_one_op() {
    op.pop();
    output.pop();
    input.pop();
    count--;
  }
  void self_backward(bool retain_graph = true) {
    count--;
    NtsVar final_output = output.top();
    NtsVar final_input = input.top();
    final_output.backward(torch::ones_like(final_output), retain_graph);
    NtsVar grad_to_previous_op = final_input.grad();

    input_grad[count] = grad_to_previous_op;
    // LOG_INFO("grad_to_previous_op %d",grad_to_previous_op.dim());
    pop_one_op();
    output_grad[count] = grad_to_previous_op;
    while (count > 0 || (count == 0 && NNOP == op.top())) {
      if (NNOP != op.top()) { // test
        // LOG_INFO("GOP");
        input_grad[count] = torch::zeros_like(input.top());
        // LOG_INFO("GOP DEBUG %d
        // %d\n",output_grad[count].dim(),input_grad[count].dim());
        switch (op.top()) {
        case DIST_CPU:
          gt->PropagateBackwardCPU_Lockfree(output_grad[count],
                                            input_grad[count], subgraphs);
          break; // TODO : choose engine
        case DIST_CPU_EDGE:
          LOG_INFO("DIST_CPU_EDGE not implement");
          break;
        case DIST_GPU:
          gt->GraphPropagateBackward(output_grad[count], input_grad[count],
                                     subgraphs);
          break;
        case DIST_GPU_EDGE:
          LOG_INFO("DIST_GPU_EDGE not implement");
          break;
        case SINGLE_CPU:
          gt->PropagateBackwardCPU_Lockfree(output_grad[count],
                                            input_grad[count], subgraphs);
          break;
        case SINGLE_CPU_EDGE:
          LOG_INFO("SINGLE_CPU_EDGE not implement");
          ;
          break;
        case SINGLE_GPU:
          gt->BackwardSingle(output_grad[count], input_grad[count], subgraphs);
          break;
        case SINGLE_GPU_EDGE:
          LOG_INFO("SINGLE_GPU_EDGE not implement");
          break;
          ;
          break;
        default:
          LOG_INFO("error_engine_type");
          assert(true);
          break;
        }
        pop_one_op();
        output_grad[count] = input_grad[count + 1];

      } else if (NNOP == op.top()) {
        // LOG_INFO("NOP %d",output_grad[count].dim());
        NtsVar inter_output = output.top();
        NtsVar inter_input = input.top();
        inter_output.backward(output_grad[count], retain_graph);
        NtsVar grad_to_previous_op = inter_input.grad();
        input_grad[count] = grad_to_previous_op;
        pop_one_op();
        output_grad[count] = grad_to_previous_op;
      } else {
        LOG_INFO("NOT SUPPORT OP");
        assert(true);
      }
    }
    reset();
  }
  void debug() {
    printf("ADDEBUG input.size()%d\n", input.size());
    // for(int i=0;i<count;i++){
    while (!input.empty()) {
      LOG_INFO("input dim %d", input.top().dim());
      LOG_INFO("output dim %d", output.top().dim());
      input.pop();
      output.pop();
    }
  }

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

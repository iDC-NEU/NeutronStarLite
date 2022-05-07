#ifndef AUTODIFF_CPP
#define AUTODIFF_CPP

#include "AutoDiff.h"
#include "gnnmini.h"
#include "comm/logger.h"

namespace nts {

namespace autodiff {

/**
 * @brief Construct a new Compution Path:: Compution Path object.
 * @param gt_ 
 * @param subgraphs_ 
 */
ComputionPath::ComputionPath(GraphOperation *gt_,
            std::vector<CSC_segment_pinned *> subgraphs_,bool bi_direction_) {
  std::stack<OpType>().swap(op);
  std::stack<NtsVar>().swap(output);
  std::stack<NtsVar>().swap(input);
  output_grad.clear();
  count = 0;
  gt = gt_;
  subgraphs = subgraphs_;
  bi_direction=bi_direction_;
}

void ComputionPath::op_push(NtsVar &input_t, NtsVar &output_t, OpType op_type) {
//    if(output_t.dim()>1&&input_t.dim()>1)
//  LOG_INFO("input dim %d \t output dim %d \t OP type %d", input_t.size(1),output_t.size(1),op_type);
  NtsVar ig;
  NtsVar og;

  assert(op_type < 9);

  // we will chain the NNOP together, because torch lib will handle the backward propagation
  // when there is no graph operation
  if (count > 0 && NNOP == op_type && op.top() == NNOP) {
    output.pop();
    output.push(output_t);
    //     LOG_INFO("TRIG");
  } else {
    count++;
    op.push(op_type);
    output.push(output_t);
    input.push(input_t);
    // pre-alloc space to save graident
    output_grad.push_back(ig);
  }
}

void ComputionPath::reset() {
  assert(count<=1);
  count = 0;
  std::stack<OpType>().swap(op);
  std::stack<NtsVar>().swap(output);
  std::stack<NtsVar>().swap(input);
  output_grad.clear();
}

int ComputionPath::top_idx(){
  return count - 1;
}

/**
 * @brief 
 * pop one operation in computation path.
 * used in backward propagation
 */
void ComputionPath::pop_one_op() {
  op.pop();
  output.pop();
  input.pop();
  count--;
}

/**
 * @brief 
 * do the backward propagation using the value that we stored while doing forward
 * computation.
 * @param retain_graph 
 */
void ComputionPath::self_backward(bool retain_graph) {
  NtsVar final_output = output.top();
  NtsVar final_input = input.top();
  // compute the gradient of loss
  // the gradient of top-most result is 1.
  final_output.backward(torch::ones_like(final_output), retain_graph);
  // store the grad
  NtsVar grad_to_previous_op = final_input.grad();

  //input_grad[count] = grad_to_previous_op;
  // LOG_INFO("grad_to_previous_op %d",grad_to_previous_op.dim());
  
  output_grad[top_idx()-1] = grad_to_previous_op;
  pop_one_op();
//  LOG_INFO("finish loss op\n");
  while (count > 1 || (count == 1 && NNOP == op.top())) {
    // NNOP means we are using torch lib to do the forward computation
    // thus we can use auto diff framework in libtorch
    if (NNOP != op.top()) { // test
      output_grad[top_idx()-1] = torch::zeros_like(input.top());
   //   LOG_INFO("output_grad[count] %d \t input_grad[count] %d \t OP type %d", output_grad[count-1].size(1),input_grad[count-1].size(1),op.top());
      switch (op.top()) {
      case DIST_CPU:
//          LOG_INFO("start graph op");
//        LOG_INFO("start graph op%d %d \n",output_grad[top_idx()].size(1),output_grad[top_idx()-1].size(1));
        gt->PropagateBackwardCPU_Lockfree_multisockets(output_grad[top_idx()],
                                            output_grad[top_idx()-1], subgraphs); 
//        LOG_INFO("finish graph op\n");
        break; // TODO : choose engine
      case DIST_CPU_EDGE:
        LOG_INFO("DIST_CPU_EDGE not implement");
        break;
#if CUDA_ENABLE          
      case DIST_GPU:
        gt->GraphPropagateBackward(output_grad[top_idx()], output_grad[top_idx()-1],
                                    subgraphs);
        break;
      case DIST_GPU_EDGE:
        LOG_INFO("DIST_GPU_EDGE not implement");
        break;
#endif          
      case SINGLE_CPU:
        gt->PropagateBackwardCPU_Lockfree(output_grad[top_idx()],
                                          output_grad[top_idx()-1], subgraphs);
        break;
      case SINGLE_CPU_EDGE_SCATTER:
          gt->LocalScatterBackward(output_grad[top_idx()],
                                          output_grad[top_idx()-1], subgraphs,bi_direction);
        //LOG_INFO("SINGLE_CPU_EDGE not implement");
        break;
      case SINGLE_CPU_EDGE_GATHER:
          gt->LocalScatter(output_grad[top_idx()],
                                          output_grad[top_idx()-1], subgraphs);
        //LOG_INFO("SINGLE_CPU_EDGE not implement");
        break;
#if CUDA_ENABLE          
      case SINGLE_GPU:
        gt->BackwardSingle(output_grad[top_idx()], output_grad[top_idx()-1], subgraphs);
        break;
      case SINGLE_GPU_EDGE_SCATTER:
        LOG_INFO("SINGLE_GPU_EDGE not implement");
        break;
      case SINGLE_GPU_EDGE_GATHER:
        LOG_INFO("SINGLE_GPU_EDGE not implement");
        break;
#endif          
      default:
        LOG_INFO("error_engine_type");
        assert(true);
        break;
      }
      pop_one_op();

    } else if (NNOP == op.top()) {
      // LOG_INFO("NOP %d",output_grad[count].dim());
      NtsVar inter_output = output.top();
      NtsVar inter_input = input.top();
      // backward will compute the bottom_diff for inter_output
      // the top_diff is output_grad[count]
      // and the bottom_diff for inter_output, also is top_diff for inter_input
      // will store in inter_input.grad()
      // then we retrieve it for future use
      inter_output.backward(output_grad[top_idx()], retain_graph);
      NtsVar grad_to_previous_op = inter_input.grad();
      if(count>1)
      output_grad[top_idx()-1] = grad_to_previous_op;
      pop_one_op();
//       LOG_INFO("finish nn op\n");
  //    LOG_INFO("output_grad[count] %d \t input_grad[count] %d \t OP type %d", output_grad[count-1].size(1),input_grad[count-1].size(1),op.top());
    } else {
      LOG_INFO("NOT SUPPORT OP");
      assert(true);
    }
  }
  reset();
}

void ComputionPath::debug() {
  printf("ADDEBUG input.size()%d\n", input.size());
  // for(int i=0;i<count;i++){
  while (!input.empty()) {
    LOG_INFO("input dim %d \t output dim %d \t OP type %d", input.top().dim(),output.top().dim(),op.top());
    input.pop();
    output.pop();
    op.pop();
  }
}

} // namespace autodiff
} // namespace nts

#endif

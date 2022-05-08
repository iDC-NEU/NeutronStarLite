#ifndef NTSOPS_HPP
#define NTSOPS_HPP
#include <stack>
#include "core/ntsGraphOp.hpp"
#include<type_traits>
namespace nts {

namespace ctx {

typedef uint32_t OpType;

const OpType NNOP = 0;
const OpType GRAPHOP = 1;
const OpType SEGMENTOP = 2;

class ntsOperator{
public:
    ntsOperator(){
        
    }
    ntsOperator(nts::op::ntsGraphOp* op_,OpType op_t_){
        assert(GRAPHOP==op_t_);
        op=op_;
        op_t=op_t_;
    }
    ntsOperator(OpType op_t_){
        assert(NNOP==op_t_);
        op_t=op_t_;
    }
    nts::op::ntsGraphOp* op;
    OpType op_t;
};
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
class NtsContext {
public:
  NtsContext(){
  std::stack<OpType>().swap(op);
  std::stack<NtsVar>().swap(output);
  std::stack<NtsVar>().swap(input);
  std::stack<ntsOperator>().swap(ntsOp);
  output_grad.clear();
  count = 0;
}
  template <typename GOPT>
  NtsVar runGraphOp(Graph<Empty> *graph, VertexSubset *active,
                   std::vector<CSC_segment_pinned *> &subgraphs_,NtsVar &f_input){//graph op
      
    static_assert(std::is_base_of<nts::op::ntsGraphOp,GOPT>::value,
                "template must be a type of graph op!");
    
    nts::op::ntsGraphOp * curr=new GOPT(graph,active,subgraphs_);
    NtsVar f_output=curr->forward(f_input); 
    NtsVar ig;
    op.push(GRAPHOP);
    output.push(f_output);
    input.push(f_input);
    ntsOp.push(ntsOperator(curr,GRAPHOP));
    // pre-alloc space to save graident
    output_grad.push_back(ig);
    count++;
    return f_output;
} 
 NtsVar runVertexForward(std::function<NtsVar(NtsVar &, NtsVar &)> vertexforward,
            NtsVar &nbr_input,NtsVar &vtx_input){//NNOP
//     LOG_INFO("call run vertex forward");
    NtsVar f_output=vertexforward(nbr_input,vtx_input); 
    NtsVar ig;
    if (count > 0 && op.top() == NNOP) {
        output.pop();
        output.push(f_output);
        //     LOG_INFO("TRIG");
    }else{
        op.push(NNOP);
        output.push(f_output);
        input.push(nbr_input);
        ntsOp.push(ntsOperator(NNOP));
        // pre-alloc space to save graident
        output_grad.push_back(ig);
        count++;
        return f_output;
    }
}
 NtsVar runVertexForward(std::function<NtsVar(NtsVar &)> vertexforward,
            NtsVar &nbr_input){//NNOP
//     LOG_INFO("call run vertex forward");
    NtsVar f_output=vertexforward(nbr_input); 
    NtsVar ig;
    if (count > 0 && op.top() == NNOP) {
        output.pop();
        output.push(f_output);
        //     LOG_INFO("TRIG");
    }else{
        op.push(NNOP);
        output.push(f_output);
        input.push(nbr_input);
        ntsOp.push(ntsOperator(NNOP));
        // pre-alloc space to save graident
        output_grad.push_back(ig);
        count++;
        return f_output;
    }
}
 
 NtsVar runEdgeForward(std::function<NtsVar(NtsVar &)> edgeforward,
            NtsVar &edge_input){//NNOP
//     LOG_INFO("call run vertex forward");
    NtsVar f_output=edgeforward(edge_input); 
    NtsVar ig;
    if (count > 0 && op.top() == NNOP) {
        output.pop();
        output.push(f_output);
        //     LOG_INFO("TRIG");
    }else{
        op.push(NNOP);
        output.push(f_output);
        input.push(edge_input);
        ntsOp.push(ntsOperator(NNOP));
        // pre-alloc space to save graident
        output_grad.push_back(ig);
        count++;
        return f_output;
    }
} 
  
  void appendNNOp(NtsVar &input_t, NtsVar &output_t){
    NtsVar ig;

    // we will chain the NNOP together, because torch lib will handle the backward propagation
    // when there is no graph operation
    if (count > 0 && op.top() == NNOP) {
        output.pop();
        output.push(output_t);
        //     LOG_INFO("TRIG");
    } else {
        count++;
        op.push(NNOP);
        output.push(output_t);
        input.push(input_t);
        ntsOp.push(ntsOperator(NNOP));
        // pre-alloc space to save graident
        output_grad.push_back(ig);
    }
  }
 
  void reset(){
    assert(count<=1);
    if(count==1&&ntsOp.top().op_t==GRAPHOP){
        delete ntsOp.top().op;
    }
    count = 0;
    std::stack<OpType>().swap(op);
    std::stack<NtsVar>().swap(output);
    std::stack<NtsVar>().swap(input);
    std::stack<ntsOperator>().swap(ntsOp);
    output_grad.clear();
}
  void pop_one_op(){
    if(ntsOp.top().op_t==GRAPHOP){
        delete ntsOp.top().op;
    }
    op.pop();
    output.pop();
    input.pop();
    ntsOp.pop();
    count--;
  }
  void self_backward(bool retain_graph = true){
    output.top().backward(torch::ones_like(output.top()), retain_graph);
    output_grad[top_idx()-1]= input.top().grad();// grad of loss
    pop_one_op();
//    LOG_INFO("FINISH LOSS");
      while (count > 1 || (count == 1 && NNOP == op.top())) {
    // NNOP means we are using torch lib to do the forward computation
    // thus we can use auto diff framework in libtorch
         
    if (GRAPHOP == op.top()) { // test
//         LOG_INFO("START GRAPH op%d",op.top());
      output_grad[top_idx()-1]=ntsOp.top().op->backward(output_grad[top_idx()]);
      pop_one_op();

    } else if (NNOP == op.top()) {// directly use pytorch
//        LOG_INFO("START nn op%d",op.top());
      assert(output_grad[top_idx()].size(1)==output.top().size(1));
      assert(output_grad[top_idx()].size(0)==output.top().size(0));  
      output.top().backward(output_grad[top_idx()], retain_graph);
      if(count>1)
          output_grad[top_idx()-1] = input.top().grad();
      pop_one_op();

    } else {
      LOG_INFO("NOT SUPPORT OP");
      assert(true);
    }
  }
    reset();  
  }
  void debug(){
    printf("ADDEBUG input.size()%d\n", input.size());
    // for(int i=0;i<count;i++){
    int i=0;
    while (!input.empty()) {
        if(i==0){
          LOG_INFO("input dim %d %d\t output dim %d \t OP type %d", input.top().size(0),input.top().size(1),output.top().dim(),op.top());
        }else{
          LOG_INFO("input dim %d %d \t output dim %d %d\t OP type %d", input.top().size(0),
                  input.top().size(1),output.top().size(0),output.top().size(1),op.top());  
        }
        input.pop();
        output.pop();
        op.pop();
        ntsOp.pop();
        i++;
    }
    this->output_grad.clear();
    count=0;
  }
  
  int top_idx(){
    return count - 1;
  }

private:
  std::stack<OpType> op;
  std::stack<NtsVar> output;
  std::stack<NtsVar> input;
  std::stack<ntsOperator> ntsOp;
  std::vector<NtsVar> output_grad;
  int count;
//  GraphOperation *gt;
//  std::vector<CSC_segment_pinned *> subgraphs;
//  bool bi_direction;
};

} // namespace autodiff
} // namespace nts

#endif

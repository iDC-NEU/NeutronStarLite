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
#ifndef SUBLINEARNNOP_HPP
#define SUBLINEARNNOP_HPP
#include <assert.h>
#include <map>
#include <math.h>
#include <stack>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <vector>

#include "NtsScheduler.hpp"

namespace nts {
namespace op {

class SubLinearMemCostNNOP{
public:
  NtsVar *f_input;
  std::function<NtsVar(NtsVar &)>* forward_function;
  
  SubLinearMemCostNNOP(std::function<NtsVar(NtsVar &)> vertexforward){
      forward_function=(&vertexforward);
  }
  NtsVar forward(NtsVar &f_input_msg){// input i_msg  output o_msg
     NtsVar f_input_=f_input_msg.detach();
    f_input=&f_input_msg;
    return (*forward_function)(f_input_);
  }
  NtsVar backward(NtsVar &f_output_grad){// input i_msg  output o_msg
     //NtsVar f_input_=f_input.detach();
    NtsVar f_output=(*forward_function)(*f_input);
    f_output.backward(f_output_grad);
    return f_input->grad();
    
  }
};

} // namespace graphop
} // namespace nts

#endif

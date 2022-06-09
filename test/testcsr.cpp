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

#include "core/neutronstar.hpp"

int main(int argc, char **argv) {
  MPI_Instance mpi(&argc, &argv);
  if (argc < 2) {
    printf("configuration file missed \n");
    exit(-1);
  }

  double exec_time = 0;
  exec_time -= get_time();

  Graph<Empty> *graph;
  graph = new Graph<Empty>();
  graph->config->readFromCfgFile(argv[1]);
  graph->load_directed(graph->config->edge_file, graph->config->vertices);
  graph->generate_backward_structure();
  graph->init_gnnctx(graph->config->layer_string);
    // rtminfo initialize
    graph->init_rtminfo();
  FullyRepGraph* fully_rep_graph=new FullyRepGraph(graph);
  fully_rep_graph->GenerateAll();
  fully_rep_graph->SyncAndLog("read_finish");
  for(int i=graph->partition_offset[graph->partition_id];
          i<graph->partition_offset[graph->partition_id+1];i++){
      //printf("%d %d %d\n",graph->partition_id,fully_rep_graph->column_offset[i+1]-fully_rep_graph->column_offset[i],graph->in_degree_for_backward[i]);
      assert(graph->in_degree_for_backward[i]==(fully_rep_graph->column_offset[i+1]-fully_rep_graph->column_offset[i]));
  }
  fully_rep_graph->SyncAndLog("Graph column_offset OK");
  
 Sampler* sampler=new Sampler(fully_rep_graph,graph->partition_offset[graph->partition_id],graph->partition_offset[graph->partition_id+1]);
 //Sampler* sampler=new Sampler(fully_rep_graph,0,graph->vertices);
//  sampler->sampled_a_new_graph(4,2,{2,4,5,6});
// sampler->work_queue[0]->sampled_sgs[0]->debug();
// sampler->work_queue[0]->sampled_sgs[1]->debug();
  int number=0;
  while(sampler->sample_not_finished()){
      sampler->reservoir_sample(2,128,{5,10});
   //   printf("number %d\n",number+=128);
  }
 fully_rep_graph->SyncAndLog("sample_one_finish");
 SampledSubgraph *sg;
 //if(graph->partition_id==0)
 //while(sampler->has_rest()){
 sg=sampler->get_one();
 sg->sampled_sgs[0]->debug();
 sg->sampled_sgs[1]->debug();
 //} 
 LOG_INFO("CORRECT1");
 nts::op::MiniBatchFuseOp* miniBatchFuseOp=new nts::op::MiniBatchFuseOp(sg,graph,0);
 LOG_INFO("CORRECT2%d %d",sg->sampled_sgs[0]->src().size(),128);
 NtsVar f_input=graph->Nts->NewOnesTensor({sg->sampled_sgs[0]->src().size(),4}, torch::DeviceType::CPU);
 
 NtsVar f_output=miniBatchFuseOp->forward(f_input);
 std::cout<<f_output<<std::endl;
 NtsVar s=torch::ones_like(f_output);
 NtsVar f_input_grad=miniBatchFuseOp->backward(s);
  std::cout<<f_input_grad<<std::endl;
 fully_rep_graph->SyncAndLog("finish debug");
  sampler->clear_queue();
  fully_rep_graph->SyncAndLog("finish clear");
  

// sampler->work_queue[0]->sampled_sgs[2]->debug();
// sampler->work_queue[0]->sampled_sgs[3]->debug();
 
 
 //sampler->work_queue[0]->sampled_sgs[1]->debug();
 //sampler->work_queue[0]->sampled_sgs[2]->debug();  
  
  
//  sampCSC *csc=new sampCSC(3,6);
//  csc->allocate_all();
//  csc->c_o()[0]=0;
//  csc->c_o()[1]=2;
//  csc->c_o()[2]=6;
//  for(int i=0;i<csc->r_i().size();i++){
//    csc->r_i()[i]=i%4;  
//  }
//  for(int i=0;i<3;i++){
//      LOG_INFO("c_o %d",csc->c_o(i));
//  }
//  for(int i=0;i<csc->r_i().size();i++){
//      LOG_INFO("r_i %d",csc->r_i(i));
//  }
//  csc->postprocessing();
//  for(int i=0;i<csc->r_i().size();i++){
//      LOG_INFO("r_i_local %d",csc->r_i(i));
//  }
//  for(int i=0;i<csc->src().size();i++){
//      LOG_INFO("src %d",csc->src()[i]);
//  }
  exec_time += get_time();
  if (graph->partition_id == 0) {
    printf("exec_time=%lf(s)\n", exec_time);
  }

  delete graph;

  //    ResetDevice();

  return 0;
}

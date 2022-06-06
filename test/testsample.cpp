/*
Copyright (c) 2014-2015 Xiaowei Zhu, Tsinghua University

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

#include "GCN_CPU.hpp"
#include "GCN_CPU_EAGER.hpp"
#include "GIN_CPU.hpp"
#include "GGNN_CPU.hpp"
#include "GAT_CPU.hpp"
#if CUDA_ENABLE
#include "COMMNET_GPU.hpp"
//#include "GAT_GPU.hpp"
//#include "GAT_GPU_SINGLE.hpp"
#include "GCN.hpp"
#include "GCN_EAGER.hpp"
#include "GCN_EAGER_single.hpp"
#include "GIN_GPU.hpp"
#endif

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
  if (graph->partition_id == 0)
    graph->config->print();

  int iterations = graph->config->epochs;
  graph->replication_threshold = graph->config->repthreshold;

  if (graph->config->algorithm == std::string("GCNCPU")) {
    graph->load_directed(graph->config->edge_file, graph->config->vertices);
    graph->generate_backward_structure();
    GCN_CPU_impl *ntsGCN = new GCN_CPU_impl(graph, iterations);
    ntsGCN->init_graph();
    ntsGCN->init_nn();
    ntsGCN->run();
  } else if (graph->config->algorithm == std::string("GINCPU")) {
    graph->load_directed(graph->config->edge_file, graph->config->vertices);
    graph->generate_backward_structure();
    GIN_CPU_impl *ntsGIN = new GIN_CPU_impl(graph, iterations);
    ntsGIN->init_graph();
    ntsGIN->init_nn();
    ntsGIN->run();
  } else if (graph->config->algorithm == std::string("GCNCPUEAGER")) {
    graph->load_directed(graph->config->edge_file, graph->config->vertices);
    graph->generate_backward_structure();
    GCN_CPU_EAGER_impl *ntsGCN = new GCN_CPU_EAGER_impl(graph, iterations);
    ntsGCN->init_graph();
    ntsGCN->init_nn();
    ntsGCN->run();
  } else if (graph->config->algorithm == std::string("GGNNCPU")) {
    graph->load_directed(graph->config->edge_file, graph->config->vertices);
    graph->generate_backward_structure();
    GGNN_CPU_impl *ntsGGNN = new GGNN_CPU_impl(graph, iterations);
    ntsGGNN->init_graph();
    ntsGGNN->init_nn();
    ntsGGNN->run();
  } else if (graph->config->algorithm == std::string("GATCPU")) {
    graph->load_directed(graph->config->edge_file, graph->config->vertices);
    graph->generate_backward_structure();
    GAT_CPU_impl *ntsGAT = new GAT_CPU_impl(graph, iterations);
    ntsGAT->init_graph();
    ntsGAT->init_nn();
    ntsGAT->run();
  } 
 

#if CUDA_ENABLE
  else if (graph->config->algorithm == std::string("COMMNETGPU")) {
    graph->load_directed(graph->config->edge_file, graph->config->vertices);
    graph->generate_backward_structure();
    COMMNET_impl *ntsCOMM = new COMMNET_impl(graph, iterations);
    ntsCOMM->init_graph();
    ntsCOMM->init_nn();
    ntsCOMM->run();
  } else if (graph->config->algorithm == std::string("GINGPU")) {
    graph->load_directed(graph->config->edge_file, graph->config->vertices);
    graph->generate_backward_structure();
    GIN_impl *ntsGIN = new GIN_impl(graph, iterations);
    ntsGIN->init_graph();
    ntsGIN->init_nn();
    ntsGIN->run();
  } else if (graph->config->algorithm == std::string("GCN")) {
    graph->load_directed(graph->config->edge_file, graph->config->vertices);
    graph->generate_backward_structure();
    GCN_impl *ntsGCN = new GCN_impl(graph, iterations);
    ntsGCN->init_graph();
    ntsGCN->init_nn();
    ntsGCN->run();
    // GCN(graph, iterations);
  } else if (graph->config->algorithm == std::string("GCNEAGER")) {
    graph->load_directed(graph->config->edge_file, graph->config->vertices);
    graph->generate_backward_structure();
    GCN_EAGER_impl *ntsGCN = new GCN_EAGER_impl(graph, iterations);
    ntsGCN->init_graph();
    ntsGCN->init_nn();
    ntsGCN->run();
    // GCN(graph, iterations);
  } else if (graph->config->algorithm == std::string("GCNEAGERSINGLE")) {
    graph->load_directed(graph->config->edge_file, graph->config->vertices);
    graph->generate_backward_structure();
    GCN_EAGER_single_impl *ntsGCN =
        new GCN_EAGER_single_impl(graph, iterations);
    ntsGCN->init_graph();
    ntsGCN->init_nn();
    ntsGCN->run();
    // GCN(graph, iterations);
//  } else if (graph->config->algorithm == std::string("GAT")) {
//    graph->load_directed(graph->config->edge_file, graph->config->vertices);
//    graph->generate_backward_structure();
//    GAT_GPU_impl *ntsGAT = new GAT_GPU_impl(graph, iterations);
//    ntsGAT->init_graph();
//    ntsGAT->init_nn();
//    ntsGAT->run();
//  } else if (graph->config->algorithm == std::string("GATSINGLE")) {
//    graph->load_directed(graph->config->edge_file, graph->config->vertices);
//    graph->generate_backward_structure();
//    GAT_GPU_SINGLE_impl *ntsGAT = new GAT_GPU_SINGLE_impl(graph, iterations);
//    ntsGAT->init_graph();
//    ntsGAT->init_nn();
//    ntsGAT->run();
  } else if (graph->config->algorithm == std::string("COMMNETGPU")) {
    graph->load_directed(graph->config->edge_file, graph->config->vertices);
    graph->generate_backward_structure();
    COMMNET_impl *ntsCOMM = new COMMNET_impl(graph, iterations);
    ntsCOMM->init_graph();
    ntsCOMM->init_nn();
    ntsCOMM->run();
  } else if (graph->config->algorithm == std::string("GINGPU")) {
    graph->load_directed(graph->config->edge_file, graph->config->vertices);
    graph->generate_backward_structure();
    GIN_impl *ntsGIN = new GIN_impl(graph, iterations);
    ntsGIN->init_graph();
    ntsGIN->init_nn();
    ntsGIN->run();
  } 
#endif
  exec_time += get_time();
  if (graph->partition_id == 0) {
    printf("exec_time=%lf(s)\n", exec_time);
  }

  delete graph;

  //    ResetDevice();

  return 0;
}

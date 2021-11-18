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
/* CPU single experiment*/
//#include "cpuengine.hpp"
//#include "gpuengine.hpp"
//#include "gpuclusterengine.hpp"
#include "GCN.hpp"
#include "GCN_CPU.hpp"
#include "GIN.hpp"
#include "GAT.hpp"
//#include "pr.hpp"
#include "pr_cpu.hpp"
//#include"testengine.hpp"

void statistic(Graph<Empty> *graph, int workers)
{
    
    int worker_num = workers;
    int *offset = new int[worker_num + 1];
    long *first_layer = new long[worker_num + 1];
    int *first_layer_unique = new int[worker_num + 1];
    long *second_layer = new long[worker_num + 1];
    int *second_layer_unique = new int[worker_num + 1];
    char *bitmap_second = new char[graph->vertices + 1];
    char *bitmap_first = new char[graph->vertices + 1];
    char *bitmap_third = new char[graph->vertices + 1];
    char *bitmap_zero = new char[graph->vertices + 1];

    long first_layer_vertices = 0, first_layer_vertice_dump = 0;
    long long first_layer_edges = 0, first_layer_edges_dump = 0;
    long second_layer_vertices = 0, second_layer_vertice_dump = 0;
    long long second_layer_edges = 0, second_layer_edges_dump = 0;

    memset(offset, 0, sizeof(int) * (worker_num + 1));
    memset(first_layer, 0, sizeof(long) * (worker_num + 1));
    memset(first_layer_unique, 0, sizeof(int) * (worker_num + 1));
    memset(second_layer, 0, sizeof(long) * (worker_num + 1));
    memset(second_layer_unique, 0, sizeof(int) * (worker_num + 1));
    int worker_vertices = graph->vertices / worker_num;
    for (int i = 0; i < worker_num; i++)
    {
        offset[i + 1] = offset[i] + worker_vertices;
    }

    double cnt = 0;
    int id = 0;
    for (int i = 0; i < graph->vertices; i++)
    {
        cnt = graph->in_degree[i] + cnt;
        if (cnt > (graph->edges / worker_num) * (id + 1))
        {
            offset[id + 1] = i;
            id++;
        }
    }

    std::cout << std::endl
              << "++++++++++status++++++++++" << std::endl;
    std::cout << std::endl
              << "worker_num: " << worker_num << "++++" << std::endl;
    std::cout << "graph->vertices: " << graph->vertices << std::endl;
    std::cout << "graph->edges   : " << graph->edges << std::endl;
    std::cout << "partition: ";
    offset[worker_num] = graph->vertices;
    for (int i = 0; i < worker_num; i++)
    {
        std::cout << offset[i + 1] - offset[i] << " ";
    }
    std::cout << std::endl;

    memset(bitmap_zero, 0, graph->vertices);
    memset(bitmap_first, 0, graph->vertices);
    memset(bitmap_second, 0, graph->vertices);
    memset(bitmap_third, 0, graph->vertices);
    int cache_0_edge_count = 0;
    int cache_00_vertice_count = 0;
    int cache_vertices = 0;

    int cache_1_edge_count = 0;
    int cache_0_vertice_count = 0;
    int cache_1_vertice_count = 0;
    int cache_2_edge_count = 0;
    int cache_2_vertice_count = 0;
    int anchor = 5;
    for (int j = 0; j < graph->vertices; j++)
    {
        if (j % 10000 == 0)
        {
            std::cout << "processed " << j << std::endl;
        }
        if (graph->in_degree[j] <= anchor)
        {
            int signature = true;
            cache_0_edge_count += graph->in_degree[j];
            cache_vertices += 1;
            for (int k = graph->incoming_adj_index[0][j]; k < graph->incoming_adj_index[0][j + 1]; k++)
            {
                int neighbour = graph->incoming_adj_list[0][k].neighbour;
                bitmap_zero[neighbour] = 1;
                if (graph->in_degree[neighbour] > anchor)
                {
                    signature = false;
                }
            }
            if (signature == true)
            {
                bitmap_first[j] = 1;
                cache_1_edge_count += graph->in_degree[j];
                for (int k = graph->incoming_adj_index[0][j]; k < graph->incoming_adj_index[0][j + 1]; k++)
                {
                    int neighbour = graph->incoming_adj_list[0][k].neighbour;
                    cache_2_edge_count += graph->in_degree[neighbour];
                    bitmap_second[neighbour] = 1;
                    for (int l = graph->incoming_adj_index[0][neighbour]; l < graph->incoming_adj_index[0][neighbour + 1]; l++)
                    {
                        int nbr_second = graph->incoming_adj_list[0][l].neighbour;
                        bitmap_third[nbr_second] = 1;
                    }
                }
            }
        }
    }
    for (int i = 0; i < graph->vertices; i++)
    {
        if (bitmap_first[i])
            cache_0_vertice_count++;
        if (bitmap_second[i])
            cache_1_vertice_count++;
        if (bitmap_third[i])
            cache_2_vertice_count++;
        if (bitmap_zero[i])
            cache_00_vertice_count++;
    }
    std::cout << "SECOND_LAYER#####################" << std::endl;
    std::cout << "vertices" << graph->vertices << std::endl;
    std::cout << "edges" << graph->edges << std::endl;
    std::cout << "cache_1_edge_count " << cache_1_edge_count << std::endl;
    std::cout << "cache_2_edge_count " << cache_2_edge_count << std::endl;
    std::cout << "cache_0_vertice_count " << cache_0_vertice_count << std::endl;
    std::cout << "cache_1_vertice_count " << cache_1_vertice_count << std::endl;
    std::cout << "cache_2_vertice_count " << cache_2_vertice_count << std::endl;
    std::cout << "FIRST_LAYER#####################" << std::endl;
    std::cout << "cache_vertice " << cache_vertices << std::endl;
    std::cout << "cache_00_vertice_count " << cache_00_vertice_count << std::endl;
    std::cout << "cache_0_edge_count " << cache_0_edge_count << std::endl;
    std::cout << "SECOND_LAYER#####################" << std::endl;
}

int main(int argc, char **argv)
{
    MPI_Instance mpi(&argc, &argv);
        if (argc < 2)
    {
        printf("configuration file missed \n");
        exit(-1);
    }
//    if (argc < 6)
//    {
//        printf("pagerank [(1)file] [(2)vertices] [(3)iterations] [(4)CPU/GPU] [(5)layers] [(6.opt)rep threshold] [(7.opt)overlap] \n");
//        exit(-1);
//    }

    Graph<Empty> *graph;
    graph = new Graph<Empty>();
    graph->config->readFromCfgFile(argv[1]);
    if(graph->partition_id==0)
        graph->config->print();
    graph->load_directed(graph->config->edge_file, graph->config->vertices);
    graph->generate_backward_structure();
    int iterations = graph->config->epochs;
//    graph->config->layer_string = std::string(argv[5]);
//    if (argc > 6)
//    {
        graph->replication_threshold = graph->config->repthreshold ;
//        graph->config->repthreshold = std::atoi(argv[6]);
//        if (graph->config->repthreshold > 0)
//            graph->config->process_local = true;
//        else
//            graph->config->process_local = false;
//    }
//    else
//    {
//        graph->config->repthreshold = 0;
//        graph->config->process_local = false;
//    }
//    graph->config->overlap = false;
//    if (argc > 7)
//    {
//        if (std::string("overlap") == std::string(argv[7]))
//            graph->config->overlap = true;
//        else
//            graph->config->overlap = false;
//    }

    double exec_time = 0;
    exec_time -= get_time();
    if (graph->config->algorithm == std::string("GCNCPU"))
    {
        GCN_CPU_impl *ntsGCN=new GCN_CPU_impl(graph,iterations);
        ntsGCN->init_graph();
        ntsGCN->init_nn();
        ntsGCN->run();
    }
    else if (graph->config->algorithm == std::string("GCN"))
    {
        GCN_impl *ntsGCN=new GCN_impl(graph,iterations);
        ntsGCN->init_graph();
        ntsGCN->init_nn();
        ntsGCN->run();
        //GCN(graph, iterations);
    }
    else if (graph->config->algorithm == std::string("GIN"))
    {
        GIN_impl *ntsGIN=new GIN_impl(graph,iterations);
        ntsGIN->init_graph();
        ntsGIN->init_nn();
        ntsGIN->run();
    }
    else if (graph->config->algorithm == std::string("GAT"))
    {
        GAT_impl *ntsGAT=new GAT_impl(graph,iterations);
        ntsGAT->init_graph();
        ntsGAT->init_nn();
        ntsGAT->run();
    }
    else if (graph->config->algorithm == std::string("PR_CPU"))
    {
        pr_cpu_impl *ntsPR=new pr_cpu_impl(graph,iterations);
        ntsPR->init_graph();
        ntsPR->init_nn();
        ntsPR->run();
    }
    exec_time += get_time();
    if (graph->partition_id == 0)
    {
        printf("exec_time=%lf(s)\n", exec_time);
    }

    //ResetDevice();

    delete graph;

    return 0;
}

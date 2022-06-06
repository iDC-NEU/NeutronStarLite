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
#ifndef FULLLYREPGRAPH_HPP
#define FULLLYREPGRAPH_HPP
#include <assert.h>
#include <map>
#include <math.h>
#include <stack>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <vector>
#include "core/graph.hpp"
#include "core/coocsc.hpp"
class SampledSubgraph{
public:    
    SampledSubgraph(){
        ;
    }
    SampledSubgraph(int layers_, int batch_size_,std::vector<int>& fanout_){
        layers=layers_;
        batch_size=batch_size_;
        fanout=fanout_;
        sampled_sgs.clear();
        curr_layer=0;
        curr_dst_size=batch_size;
    }
    ~SampledSubgraph(){
        fanout.clear();
        for(int i=0;i<sampled_sgs.size();i++){
            delete sampled_sgs[i];
        }
        sampled_sgs.clear();
    }
    void sample_preprocessing(VertexId layer){
        curr_layer=layer;
        sampCSC* sampled_sg=new sampCSC(curr_dst_size);
        //sampled_sg->allocate_all();
        sampled_sg->allocate_vertex();
        sampled_sgs.push_back(sampled_sg);
       // assert(layer==sampled_sgs.size()-1);
    }
    void sample_load_destination(std::function<void(std::vector<VertexId> &destination)> dst_select,VertexId layer){
        dst_select(sampled_sgs[layer]->dst());//init destination;
    }
    
    void init_co(std::function<VertexId(VertexId dst)> get_nbr_size,VertexId layer){
        VertexId offset=0;
        for(VertexId i=0;i<curr_dst_size;i++){
            sampled_sgs[layer]->c_o()[i]=offset;
            offset+=get_nbr_size(sampled_sgs[layer]->dst()[i]);//init destination;
        }
        sampled_sgs[layer]->c_o()[curr_dst_size]=offset;  
        sampled_sgs[layer]->allocate_edge(offset);
    }
    
    void sample_load_destination(VertexId layer){
        assert(layer>0);
        for(VertexId i_id=0;i_id<curr_dst_size;i_id++){
            sampled_sgs[layer]->dst()[i_id]=sampled_sgs[layer-1]->src()[i_id];
        }
    }
    void sample_processing(std::function<void(VertexId fanout_i,
                std::vector<VertexId> &destination,
                    std::vector<VertexId> &column_offset,
                        std::vector<VertexId> &row_indices,VertexId id)> vertex_sample){
        {
#pragma omp parallel for
            for (VertexId begin_v_i = 0;begin_v_i < curr_dst_size;begin_v_i += 1) {
            // for every vertex, apply the sparse_slot at the partition
            // corresponding to the step
             vertex_sample(fanout[curr_layer],
                    sampled_sgs[curr_layer]->dst(),
                     sampled_sgs[curr_layer]->c_o(),
                      sampled_sgs[curr_layer]->r_i(),
                        begin_v_i);
            }
        }
        
    }
    void sample_postprocessing(){
        sampled_sgs[sampled_sgs.size()-1]->postprocessing();
        curr_dst_size=sampled_sgs[sampled_sgs.size()-1]->get_distinct_src_size();
        curr_layer++;
    }
    
    std::vector<sampCSC*> sampled_sgs;
    int layers;
    int batch_size;
    std::vector<int> fanout;
    int curr_layer;
    int curr_dst_size;
    
};
class FullyRepGraph{
public:
    //topo:
  VertexId *dstList;
  VertexId *srcList;
  //meta info
  Graph<Empty> *graph_;
  VertexId *partition_offset;
  VertexId partitions;
  VertexId partition_id;
  VertexId global_vertices;
  VertexId global_edges;
  // vertex range for this chunk
  VertexId owned_vertices;
  VertexId owned_edges;
  VertexId owned_mirrors;
  
  //global graph;
  VertexId* column_offset;
  VertexId* row_indices;
  
  FullyRepGraph(){
  }
  FullyRepGraph(Graph<Empty> *graph){
        global_vertices=graph->vertices;
        global_edges=graph->edges;
        owned_vertices=graph->owned_vertices;
        partitions=graph->partitions;
        partition_id=graph->partition_id;
        partition_offset=graph->partition_offset;
        graph_=graph;
  }
  void SyncAndLog(const char* data){
      MPI_Barrier(MPI_COMM_WORLD);
      if(partition_id==0)
      std::cout<<data<<std::endl;
  }
  void GenerateAll(){
     
        ReadRepGraphFromRawFile();
        SyncAndLog("NeutronStar::Preprocessing[Generate Full Replicated Graph Topo]");
     SyncAndLog("------------------finish graph preprocessing--------------\n");
  }
   void ReadRepGraphFromRawFile() {
    column_offset=new VertexId[global_vertices+1];
    row_indices=new VertexId[global_edges];   
    memset(column_offset, 0, sizeof(VertexId) * (global_vertices + 1));
    memset(row_indices, 0, sizeof(VertexId) * global_edges);
    VertexId *tmp_offset = new VertexId[global_vertices + 1];
    memset(tmp_offset, 0, sizeof(VertexId) * (global_vertices + 1));
    long total_bytes = file_size(graph_->filename.c_str());
#ifdef PRINT_DEBUG_MESSAGES
    if (partition_id == 0) {
      printf("|V| = %u, |E| = %lu\n", vertices, edges);
    }
#endif
    int edge_unit_size = sizeof(VertexId)*2;
    EdgeId read_edges = global_edges;
    long bytes_to_read = edge_unit_size * read_edges;
    long read_offset = 0;
    long read_bytes;
    int fin = open(graph_->filename.c_str(), O_RDONLY);
    EdgeUnit<Empty> *read_edge_buffer = new EdgeUnit<Empty>[CHUNKSIZE];

    assert(lseek(fin, read_offset, SEEK_SET) == read_offset);
    read_bytes = 0;
    while (read_bytes < bytes_to_read) {
      long curr_read_bytes;
      if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE) {
        curr_read_bytes =
            read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
      } else {
        curr_read_bytes =
            read(fin, read_edge_buffer, bytes_to_read - read_bytes);
      }
      assert(curr_read_bytes >= 0);
      read_bytes += curr_read_bytes;
      EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
      // #pragma omp parallel for
      for (EdgeId e_i = 0; e_i < curr_read_edges; e_i++) {
        VertexId src = read_edge_buffer[e_i].src;
        VertexId dst = read_edge_buffer[e_i].dst;
        tmp_offset[dst + 1]++;
      }
    }
    for (int i = 0; i < global_vertices; i++) {
      tmp_offset[i + 1] += tmp_offset[i];
    }

    memcpy(column_offset, tmp_offset, sizeof(VertexId) * (global_vertices + 1));
    // printf("%d\n", column_offset[vertices]);
    assert(lseek(fin, read_offset, SEEK_SET) == read_offset);
    read_bytes = 0;
    while (read_bytes < bytes_to_read) {
      long curr_read_bytes;
      if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE) {
        curr_read_bytes =
            read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
      } else {
        curr_read_bytes =
            read(fin, read_edge_buffer, bytes_to_read - read_bytes);
      }
      assert(curr_read_bytes >= 0);
      read_bytes += curr_read_bytes;
      EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
      // #pragma omp parallel for
      for (EdgeId e_i = 0; e_i < curr_read_edges; e_i++) {
        VertexId src = read_edge_buffer[e_i].src;
        VertexId dst = read_edge_buffer[e_i].dst;
        //        if(dst==875710)
        //            printf("%d",read_edge_buffer[e_i].src);
        row_indices[tmp_offset[dst]++] = src;
      }
    }
    delete []read_edge_buffer;
    delete []tmp_offset; 
  } 
};



#endif
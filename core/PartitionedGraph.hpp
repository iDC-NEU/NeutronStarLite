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
#ifndef PARTITIONEDGRAPH_HPP
#define PARTITIONEDGRAPH_HPP
#include <assert.h>
#include <map>
#include <math.h>
#include <stack>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <vector>
#include "core/graph.hpp"
class PartitionedGraph{
public:
    //topo:
  VertexId *dstList;
  VertexId *srcList;
  VertexId *MirrorIndex;
  //meta info
  Graph<Empty> *graph_;
  VertexSubset *active_;
  VertexId *partition_offset;
  VertexId partitions;
  VertexId partition_id;
  VertexId global_vertices;
  VertexId global_edges;
  // vertex range for this chunk
  VertexId owned_vertices;
  VertexId owned_edges;
  VertexId owned_mirrors;
  
  //communication
  std::vector<Bitmap*>hasMirrorAtPartition;
  
  //global graph;
  VertexId* column_offset;
  VertexId* row_indices;
  VertexId* compressed_row_offset;
  VertexId* column_indices;
  
  std::vector<SampleGraph> sample_graph;
  //graph segment;
  std::vector<CSC_segment_pinned*>graph_chunks;
  PartitionedGraph(){
  }
  PartitionedGraph(Graph<Empty> *graph, VertexSubset *active){
        global_vertices=graph->vertices;
        global_edges=graph->edges;
        owned_vertices=graph->owned_vertices;
        partitions=graph->partitions;
        partition_id=graph->partition_id;
        partition_offset=graph->partition_offset;
        graph_=graph;
        active_=active;
        graph_chunks.clear();
        hasMirrorAtPartition.clear();
  }
  void SyncAndLog(const char* data){
      MPI_Barrier(MPI_COMM_WORLD);
      if(partition_id==0)
      std::cout<<data<<std::endl;
  }
  inline bool has_mirror_at(VertexId p_i,VertexId v_i){
      return hasMirrorAtPartition[p_i]->get_bit(v_i-graph_->gnnctx->p_v_s);
  }
  void GenerateAll(std::function<ValueType(VertexId, VertexId)> weight_compute,
                            DeviceLocation dt_,bool dist=false,bool chunk=false){
      generatePartitionedSubgraph();
      SyncAndLog("NeutronStar::Preprocessing[Generate Partitioned Subgraph]");
      PartitionToChunks(weight_compute, dt_);
      SyncAndLog("NeutronStar::Preprocessing[Generate Graph Chunks]");
      if(dt_==CPU_T){
        DetermineMirror();
        SyncAndLog("NeutronStar::Preprocessing[Determine Mirrors]");
        GenerateMessageBitmap_multisokects();
        SyncAndLog("NeutronStar::Preprocessing[Compressed Message Prepared]");
      }

      if(dt_==GPU_T){
//        GenerateMessageBitmap();  
        GenerateTmpMsg(dt_);
      }
      if(dist){
        generateMirrorIndex();
        SyncAndLog("NeutronStar::Preprocessing[Generate Mirror Indices]");
        GenerateWholeGraphTopo();
        SyncAndLog("NeutronStar::Preprocessing[Generate Full Graph Topo]");
      }
     SyncAndLog("------------------finish graph preprocessing--------------\n");
  }
  void GenerateWholeGraphTopo(){
      column_offset=new VertexId[owned_vertices+1];
      compressed_row_offset=new VertexId[owned_mirrors+1];
      column_indices=new VertexId[owned_edges];
      row_indices=new VertexId[owned_edges];
      memset(column_offset,0,(owned_vertices+1)*sizeof(VertexId));
      memset(compressed_row_offset,0,(owned_mirrors+1)*sizeof(VertexId));
      memset(column_indices,0,owned_edges*sizeof(VertexId));
      memset(row_indices,0,owned_edges*sizeof(VertexId));
      VertexId* tmp_column_offset=new VertexId[owned_vertices+1]; 
      VertexId* tmp_row_offset=new VertexId[owned_mirrors+1];
      memset(tmp_column_offset,0,(owned_vertices+1)*sizeof(VertexId));
      memset(tmp_row_offset,0,(owned_mirrors+1)*sizeof(VertexId));
      for(VertexId i=0;i<this->owned_edges;i++){
          VertexId src_trans=this->MirrorIndex[this->srcList[i]];
          VertexId dst_trans=this->dstList[i]-graph_->gnnctx->p_v_s;
          tmp_column_offset[dst_trans+1]++;
          tmp_row_offset[src_trans+1]++;
      }
      for(int i=0;i<owned_vertices;i++){
          tmp_column_offset[i+1]+=tmp_column_offset[i];
      }
      for(int i=0;i<owned_mirrors;i++){
          tmp_row_offset[i+1]+=tmp_row_offset[i];
      }
      memcpy(column_offset,tmp_column_offset,sizeof(VertexId)*(owned_vertices+1));
      memcpy(compressed_row_offset,tmp_row_offset,sizeof(VertexId)*(owned_mirrors+1));
      for(VertexId i=0;i<this->owned_edges;i++){
          VertexId src=this->srcList[i];
          VertexId dst=this->dstList[i];
          VertexId src_trans=this->MirrorIndex[src];
          VertexId dst_trans=dst-graph_->gnnctx->p_v_s;
         row_indices[tmp_column_offset[dst_trans]++]=src;
         column_indices[tmp_row_offset[src_trans]++]=dst;
      }
      delete []tmp_column_offset;
      delete []tmp_row_offset;

  }
  void DetermineMirror_(){
      // this codes is rely on Gemini engine
      //and will not be used in the future version
            for(int i=0;i<this->partitions;i++){
          hasMirrorAtPartition.push_back(new Bitmap(owned_vertices));
      }
      int feature_size = 1;
        graph_->process_edges_backward<int, VertexId>( // For EACH Vertex Processing
      [&](VertexId src, VertexAdjList<Empty> outgoing_adj, VertexId thread_id,
          VertexId recv_id) { // pull
        VertexId src_trans = src - graph_->partition_offset[recv_id];
        // send id of local partition to src
        // indicate that src will have contribution to local partition
        // while doing forward propagation
        if (graph_chunks[recv_id]->source_active->get_bit(src_trans)) {
          VertexId part = (VertexId)graph_->partition_id;
          graph_->emit_buffer(src, &part, feature_size);
        }
      },
      [&](VertexId master, VertexId *msg) {
        // vertex master in local partition will have contribution to partition
        // part
        VertexId part = *msg;
        hasMirrorAtPartition[part]->set_bit(master - graph_->gnnctx->p_v_s); 
        // destination_mirror_active->set_bit(master-start_);
        return 0;
      },
      feature_size, active_);
      
  }
  void DetermineMirror(){
      //the sec_active of each chunk definitely indicate the active mirrors on 
      //the its master node. so we synchronize them through a ring based mpi 
      //allreduce.   
      for(int i=0;i<this->partitions;i++){
          hasMirrorAtPartition.push_back(new Bitmap(owned_vertices));
      }
      int feature_size = 1;
      std::thread send_thread([&]() {
          for (int step = 1; step < partitions; step++) {
            int recipient_id = (partition_id + step) % partitions;
            MPI_Send(graph_chunks[recipient_id]->source_active->data,
                     WORD_OFFSET(graph_chunks[recipient_id]->source_active->size)+1,
                    MPI_UNSIGNED_LONG, recipient_id,
                     PassMessage, MPI_COMM_WORLD);
          }
        });
        std::thread recv_thread([&]() {
          for (int step = 1; step < partitions; step++) {
            int sender_id = (partition_id - step + partitions) % partitions;
            MPI_Recv(this->hasMirrorAtPartition[sender_id]->data,
                     WORD_OFFSET(this->owned_vertices)+1,
                     MPI_UNSIGNED_LONG, sender_id, PassMessage, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
          }
        });
        send_thread.join();
        recv_thread.join();
        MPI_Barrier(MPI_COMM_WORLD); // denseselect fill the bitmap array
        memcpy(this->hasMirrorAtPartition[partition_id]->data,
                graph_chunks[partition_id]->source_active->data,
                 sizeof(unsigned long)*(WORD_OFFSET(this->owned_vertices)+1));
//      if (graph_->partition_id == 0)

      
  }
  void GenerateMessageBitmap_multisokects(){      
  size_t basic_chunk = 64;
  // for every partition
  for (int i = 0; i < graph_chunks.size(); i++) {
    // allocate the data structure
    graph_chunks[i]->forward_multisocket_message_index =
        new VertexId[graph_chunks[i]->batch_size_forward];
    memset(graph_chunks[i]->forward_multisocket_message_index, 0,
           sizeof(VertexId) * graph_chunks[i]->batch_size_forward);

    graph_chunks[i]->backward_multisocket_message_index =
        new BackVertexIndex[graph_chunks[i]->batch_size_backward];
    int socketNum = numa_num_configured_nodes();
    // for every vertex in partition i, set socket num
    for (int bck = 0; bck < graph_chunks[i]->batch_size_backward; bck++) {
      graph_chunks[i]->backward_multisocket_message_index[bck].setSocket(
          socketNum);
    }

    std::vector<VertexId> socket_backward_write_offset;
    socket_backward_write_offset.resize(numa_num_configured_nodes());
    memset(socket_backward_write_offset.data(), 0,
           sizeof(VertexId) * socket_backward_write_offset.size());
#pragma omp parallel
    {
      int thread_id = omp_get_thread_num();
      int s_i = graph_->get_socket_id(thread_id);
      VertexId begin_p_v_i =
          graph_->tuned_chunks_dense_backward[i][thread_id].curr;
      VertexId final_p_v_i =
          graph_->tuned_chunks_dense_backward[i][thread_id].end;
      // for every vertex, we calc the message_index. This is used in backward
      // propagation in lockfree style and every vertex has chance to gain
      // gradient from every socket in local partition so we need to calculate
      // write index for every socket
      for (VertexId p_v_i = begin_p_v_i; p_v_i < final_p_v_i; p_v_i++) {
        VertexId v_i =
            graph_->compressed_incoming_adj_index_backward[s_i][p_v_i].vertex;
        VertexId v_trans = v_i - graph_chunks[i]->src_range[0];
        if (graph_chunks[i]->src_get_active(v_i)) {
          int position =
              __sync_fetch_and_add(&socket_backward_write_offset[s_i], 1);
          graph_chunks[i]
              ->backward_multisocket_message_index[v_trans]
              .vertexSocketPosition[s_i] = position;
        }
      }
    }

    std::vector<VertexId> socket_forward_write_offset;
    socket_forward_write_offset.resize(numa_num_configured_nodes());
    memset(socket_forward_write_offset.data(), 0,
           sizeof(VertexId) * socket_forward_write_offset.size());
#pragma omp parallel for schedule(static, basic_chunk)
    // pragma omp parallel for
    for (VertexId begin_v_i = graph_chunks[i]->dst_range[0];
         begin_v_i < graph_chunks[i]->dst_range[1]; begin_v_i++) {
      int thread_id = omp_get_thread_num();
      int s_i = graph_->get_socket_id(thread_id);
      VertexId v_i = begin_v_i;
      VertexId v_trans = v_i - graph_chunks[i]->dst_range[0];
      // if v_trans has contribution to partition i
      // then we calculate the write_index for it
      // Looks like vertex id will bind to the socket id
      // i.e. the same vertex will always be processed by the same thread
      if (hasMirrorAtPartition[i]->get_bit(v_trans)) {
        int position =
            __sync_fetch_and_add(&socket_forward_write_offset[s_i], 1);
        graph_chunks[i]->forward_multisocket_message_index[v_trans] =
            position;
      }
    }
    // printf("forward_write_offset %d\n",forward_write_offset);
  }
      
  }
  void postProcessing(bool chunk=false){
      delete [] dstList;
      delete [] srcList;
      for (VertexId i = 0; i < partitions; i++) {
          graph_chunks[i]->freeAdditional();
      }
      
      
  }
  void generateMirrorIndex(){
      MirrorIndex=new VertexId[global_vertices+1];
      memset(MirrorIndex,0,sizeof(VertexId)*(global_vertices+1));
      for(VertexId i=0;i<this->owned_edges;i++){
          MirrorIndex[srcList[i]+1]=1;
      }
      for(VertexId i=0;i<global_vertices;i++){
          MirrorIndex[i+1]+=MirrorIndex[i];
      }
      owned_mirrors=MirrorIndex[global_vertices];
  }
  void generatePartitionedSubgraph(){
      owned_edges=0;
      for (int i = 0; i < graph_->sockets; i++) {
        owned_edges += (VertexId)(graph_->outgoing_adj_index[i])[graph_->vertices];
      }
      this->dstList = new VertexId[owned_edges];
      this->srcList = new VertexId[owned_edges];
      int write_position=0; 
      for (int k = 0; k < graph_->sockets; k++) {
        for (VertexId vtx = 0; vtx < graph_->vertices; vtx++) {
            for (VertexId i = graph_->outgoing_adj_index[k][vtx];
                i < graph_->outgoing_adj_index[k][vtx + 1]; i++) {
                srcList[write_position] = vtx;
                dstList[write_position++] =graph_->outgoing_adj_list[k][i].neighbour;
            }
        }
      }
  }
  void PartitionToChunks(std::function<ValueType(VertexId, VertexId)> weight_compute,
                            DeviceLocation dt_){
      graph_chunks.clear();
      std::vector<VertexId>edgecount;
      edgecount.resize(partitions,0);
      std::vector<VertexId>edgenumber;
      edgenumber.resize(partitions,0);
            
      for(VertexId i=0;i<this->owned_edges;i++){
        VertexId src_partition=graph_->get_partition_id(srcList[i]);
        edgenumber[src_partition]+=1;
      }
      
      // assign all edges to partitions
      for (VertexId i = 0; i < partitions; i++) {
        graph_chunks.push_back(new CSC_segment_pinned);
        graph_chunks[i]->init(partition_offset[i],
                              partition_offset[i + 1],
                              partition_offset[partition_id],
                              partition_offset[partition_id + 1],
                              edgenumber[i], dt_);
        graph_chunks[i]->allocVertexAssociateData();
        graph_chunks[i]->allocEdgeAssociateData();
      }
      for (VertexId i = 0; i < owned_edges; i++) {
        int source = srcList[i];
        int destination = dstList[i];
        int src_partition = graph_->get_partition_id(source);
        int offset = edgecount[src_partition]++;
        graph_chunks[src_partition]->source[offset] = source;
        graph_chunks[src_partition]->destination[offset] = destination;
      }
      VertexId *tmp_column_offset = new VertexId[global_vertices + 1];
      VertexId *tmp_row_offset = new VertexId[global_vertices + 1];
      for (VertexId i = 0; i <partitions; i++) {
        memset(tmp_column_offset, 0, sizeof(VertexId) * (global_vertices+ 1));
        memset(tmp_row_offset, 0, sizeof(VertexId) * (global_vertices + 1));
        for (VertexId j = 0; j < graph_chunks[i]->edge_size; j++) {
            // note that the vertex in the same partition has the contiguous vertexID
            // so we can minus the start index to get the offset
            // v_src_m and v_dst_m is the real vertex id
            // v_dst and v_src is local vertex id
            VertexId v_src_m = graph_chunks[i]->source[j];
            VertexId v_dst_m = graph_chunks[i]->destination[j];
            VertexId v_dst = v_dst_m - graph_chunks[i]->dst_range[0];
            VertexId v_src = v_src_m - graph_chunks[i]->src_range[0];

            // count of edges which has dst to v_dst plus one
            tmp_column_offset[v_dst + 1] += 1;
            // count of edges which has src from v_src plus one
            tmp_row_offset[v_src + 1] += 1; ///
            // graph_partitions[i]->weight_buffer[j]=(ValueType)std::sqrt(graph->out_degree_for_backward[v_src])*(ValueType)std::sqrt(graph->in_degree_for_backward[v_dst]);
        }
        // accumulate those offset, calc the partial sum
        graph_chunks[i]->column_offset[0]=0;
        for (VertexId j = 0; j < graph_chunks[i]->batch_size_forward; j++) {
            tmp_column_offset[j + 1] += tmp_column_offset[j];
            graph_chunks[i]->column_offset[j + 1] = tmp_column_offset[j + 1];
        }
        graph_chunks[i]->row_offset[0]=0;
        for (VertexId j = 0; j < graph_chunks[i]->batch_size_backward; j++){
            tmp_row_offset[j + 1] += tmp_row_offset[j];
            graph_chunks[i]->row_offset[j + 1] = tmp_row_offset[j + 1];
        }
        // after calc the offset, we should place those edges now
        for (VertexId j = 0; j < graph_chunks[i]->edge_size; j++) {
            // if(graph->partition_id==0)std::cout<<"After j edges: "<<j<<std::endl;
            // v_src is from partition i
            // v_dst is from local partition
            VertexId v_src_m = graph_chunks[i]->source[j];
            VertexId v_dst_m = graph_chunks[i]->destination[j];
            VertexId v_dst = v_dst_m - graph_chunks[i]->dst_range[0];
            VertexId v_src = v_src_m - graph_chunks[i]->src_range[0];
            graph_chunks[i]->src_set_active(v_src_m);
            graph_chunks[i]->dst_set_active(v_dst_m);
            graph_chunks[i]->row_indices[tmp_column_offset[v_dst]] = v_src_m;
            graph_chunks[i]->edge_weight_forward[tmp_column_offset[v_dst]++] =
                weight_compute(v_src_m, v_dst_m);
            graph_chunks[i]->column_indices[tmp_row_offset[v_src]] = v_dst_m; ///
            graph_chunks[i]->edge_weight_backward[tmp_row_offset[v_src]++] =
                weight_compute(v_src_m, v_dst_m);
            
        }
        for (VertexId j = 0; j < graph_chunks[i]->batch_size_forward; j++) {        
            // save the src and dst in the column format
            VertexId v_dst_m = j+ graph_chunks[i]->dst_range[0];
            for(VertexId e_idx=graph_chunks[i]->column_offset[j];e_idx<graph_chunks[i]->column_offset[j+1];e_idx++){
                VertexId v_src_m = graph_chunks[i]->row_indices[e_idx];
                 graph_chunks[i]->source[e_idx] = (long)(v_src_m);
                 graph_chunks[i]->destination[e_idx]=(long)(v_dst_m);
            }
        }
        graph_chunks[i]->CopyGraphToDevice();
    }
  delete[] tmp_column_offset;
  delete[] tmp_row_offset;
  }
  void DistSchedulingMaster(
      std::function<void(VertexId, PartitionedGraph*)> sparse_slot) {
    omp_set_num_threads(graph_->threads);
    double stream_time = 0;
    stream_time -= MPI_Wtime();
    
    size_t basic_chunk = 64;
    {
      // since we have only one partition. i think this loop should be discarded
#pragma omp parallel for
        for (VertexId begin_v_i = partition_offset[partition_id];
             begin_v_i < partition_offset[partition_id + 1];
             begin_v_i += basic_chunk) {
          // for every vertex, apply the sparse_slot at the partition
          // corresponding to the step
          VertexId v_i = begin_v_i;
          unsigned long word = active_->data[WORD_OFFSET(v_i)];
          while (word != 0) {
            if (word & 1) {
              sparse_slot(v_i,this);
            }
            v_i++;
            word = word >> 1;
          }
        }
      //      NtsComm->release_communicator();
    }
    stream_time += MPI_Wtime();
#ifdef PRINT_DEBUG_MESSAGES
    if (partition_id == 0) {
      printf("process_edges took %lf (s)\n", stream_time);
    }
#endif
  }
  
void DistSchedulingMirror(
      std::function<void(VertexId, PartitionedGraph*)> sparse_slot) {
    omp_set_num_threads(graph_->threads);
    double stream_time = 0;
    stream_time -= MPI_Wtime();
    
    size_t basic_chunk = 64;
    {
      // since we have only one partition. i think this loop should be discarded
#pragma omp parallel for
        for (VertexId begin_v_i = 0;
             begin_v_i < global_vertices;
             begin_v_i += basic_chunk) {
          // for every vertex, apply the sparse_slot at the partition
          // corresponding to the step
          VertexId v_i = begin_v_i;
          unsigned long word = 0xffffffff;
          while (word != 0) {
            if (this->MirrorIndex[v_i+1]!=this->MirrorIndex[v_i]) {
              sparse_slot(v_i,this);
            }
            v_i++;
            word = word >> 1;
          }
        }
      //      NtsComm->release_communicator();
    }
    stream_time += MPI_Wtime();
#ifdef PRINT_DEBUG_MESSAGES
    if (partition_id == 0) {
      printf("process_edges took %lf (s)\n", stream_time);
    }
#endif
  }  

void TestGeneratedBitmap() {
  for (int i = 0; i < graph_chunks.size(); i++) {
    int count_act_src = 0;
    int count_act_dst = 0;
    int count_act_master = 0;
    for (int j = graph_chunks[i]->dst_range[0]; j < graph_chunks[i]->dst_range[1];
         j++) {
      if (graph_chunks[i]->dst_get_active(j)) {
        count_act_dst++;
      }
    }
    for (int j = graph_chunks[i]->src_range[0]; j < graph_chunks[i]->src_range[1];
         j++) {
      if (graph_chunks[i]->src_get_active(j)) {
        count_act_src++;
      }
    }
    printf("PARTITION:%d CHUNK %d ACTIVE_SRC %d ACTIVE_DST %d ACTIVE_MIRROR "
           "%d\n",
           graph_->partition_id, i, count_act_src, count_act_dst,
           count_act_master);
  }
}

void GenerateTmpMsg(DeviceLocation dt_){
#if CUDA_ENABLE
  if (GPU_T == dt_) {
    int max_batch_size = 0;
    for (int i = 0; i < graph_chunks.size(); i++) {
      max_batch_size =
          std::max(max_batch_size, graph_chunks[i]->batch_size_backward);
    }
    graph_->output_gpu_buffered = graph_->Nts->NewLeafTensor(
        {max_batch_size, graph_->gnnctx->max_layer}, torch::DeviceType::CUDA);
  }
#endif 
}






};



#endif
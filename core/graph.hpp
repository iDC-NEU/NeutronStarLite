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

#ifndef GRAPH_HPP
#define GRAPH_HPP
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <malloc.h>
#include <sys/mman.h>
#include <numa.h>
#include <omp.h>

#include <algorithm>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <functional>
#include <sstream>

#include "core/atomic.hpp"
#include "core/bitmap.hpp"
#include "core/constants.hpp"
#include "core/filesystem.hpp"
#include "core/mpi.hpp"
#include "core/time.hpp"
#include "core/type.hpp"
#include "core/NtsScheduler.hpp"
//#include "torch/torch.h"
//#include "torch/csrc/autograd/generated/variable_factories.h"
//#include "torch/nn/module.h"
#include "ATen/ATen.h"

#define CHUNK_LENGTH 32768 //32768
#define REPLICATE_THRESHOLD 40



typedef struct dataaa
{
  VertexId data[REPLICATE_THRESHOLD];
} datum;

enum ThreadStatus
{
  WORKING,
  STEALING
};



struct ThreadState
{
  VertexId curr;
  VertexId end;
  ThreadStatus status;
};


template <typename EdgeData = Empty>
class Graph
{
public:
  /*partitions for streaming GPU processing*/
  gnncontext *gnnctx;
  runtimeinfo *rtminfo;
  inputinfo *config;

  //graph reorganization
  COOChunk *_graph_cpu_in;
  std::vector<COOChunk *> graph_shard_in;
  
  
  
  std::string filename;

  int partition_id;
  int partitions;

  size_t alpha;

  int threads;
  int sockets;
  int threads_per_socket;

  size_t edge_data_size;
  size_t unit_size;
  size_t edge_unit_size;
  int message_unit_size;

  bool symmetric;
  VertexId vertices; //
  EdgeId edges;      //
  EdgeId local_edges;
  VertexId *out_degree;         // VertexId [vertices]; numa-aware
  VertexId *in_degree;          // VertexId [vertices]; numa-aware
  VertexId *in_degree_backward; //_new_0

  VertexId *out_degree_for_backward; // VertexId [vertices]; numa-aware
  VertexId *in_degree_for_backward;  // VertexId [vertices]; numa-aware

  VertexId *partition_offset;       // VertexId [partitions+1]
  VertexId *local_partition_offset; // VertexId [sockets+1]

  VertexId owned_vertices;
  EdgeId *outgoing_edges;          // EdgeId [sockets]
  EdgeId *incoming_edges;          // EdgeId [sockets]
  EdgeId *incoming_edges_backward; //EdgeId[sockets];_new_1

  Bitmap **incoming_adj_bitmap;
  Bitmap **incoming_adj_bitmap_backward;          //_new_2
  EdgeId **incoming_adj_index;                    // EdgeId [sockets] [vertices+1]; numa-aware
  EdgeId **incoming_adj_index_backward;           //_new_3
  AdjUnit<EdgeData> **incoming_adj_list;          // AdjUnit<EdgeData> [sockets] [vertices+1]; numa-aware
  AdjUnit<EdgeData> **incoming_adj_list_backward; //_new_4
  Bitmap **outgoing_adj_bitmap;
  EdgeId **outgoing_adj_index;           // EdgeId [sockets] [vertices+1]; numa-aware
  AdjUnit<EdgeData> **outgoing_adj_list; // AdjUnit<EdgeData> [sockets] [vertices+1]; numa-aware

  VertexId *compressed_incoming_adj_vertices;
  VertexId *compressed_incoming_adj_vertices_backward;             //_new_5
  CompressedAdjIndexUnit **compressed_incoming_adj_index;          // CompressedAdjIndexUnit [sockets] [...+1]; numa-aware
  CompressedAdjIndexUnit **compressed_incoming_adj_index_backward; //_new_6

  VertexId *compressed_outgoing_adj_vertices;
  CompressedAdjIndexUnit **compressed_outgoing_adj_index; // CompressedAdjIndexUnit [sockets] [...+1]; numa-aware

  ThreadState **thread_state;       // ThreadState* [threads]; numa-aware
  ThreadState **tuned_chunks_dense; // ThreadState [partitions][threads];
  ThreadState **tuned_chunks_dense_backward;
  ThreadState **tuned_chunks_sparse; // ThreadState [partitions][threads];

  size_t local_send_buffer_limit;
  MessageBuffer **local_send_buffer; // MessageBuffer* [threads]; numa-aware

  int current_send_part_id;
  MessageBuffer ***send_buffer; // MessageBuffer* [partitions] [sockets]; numa-aware
  MessageBuffer ***recv_buffer; // MessageBuffer* [partitions] [sockets]; numa-aware
  

  
  //Edgefunction
  NtsScheduler *Nts;
  //comm tool
  NtsGraphCommunicator* NtsComm;
  //Nts GRAPH STORE
  Graph_Storage *NtsGraphStore; 
  
  CachedData* cachedData;
  //replication
  int replication_threshold;
  
  std::vector<Bitmap *> HasRepVtx;
  std::vector<Bitmap *> RepVtx;
  Bitmap *outGoing;
  std::vector<std::vector<VertexId>> EdgeRemote2Local;
  std::vector<std::vector<VertexId>> EdgeRemote2Remote;
  std::vector<std::vector<VertexId>> RemoteVteIndex;
  std::vector<ValueType> RepFeatures;
  float *rep_output_buffer;
  VertexId *output_map;
  VertexId rep_output_size;
  VertexId **message_write_offset;
  VertexId **message_amount;
  //Cuda_Stream *cuda_stream_public;
  
  
  
  

  //overlap
  VertexId *column_offset_intergate;
  VertexId *row_indices_intergate;

  float *weight_gpu_intergate;
  NtsVar output_gpu_buffered;
  
  //cpu data;
  std::vector<VertexId>cpu_recv_message_index;
  //gpu cache
  float *output_cpu_buffer;
  VertexId cpp;//Current Porcessing Partition

  //timer
  double all_wait_time;
  double all_overlap_time;
  double all_compute_time;
  double all_movein_time;
  double all_moveout_time;
  double all_kernel_time;
  double all_recv_copy_time;
  double all_recv_kernel_time;
  double all_recv_wait_time;
  double all_recv_thread_join_time;
  double all_cuda_sync_time;
  double all_replication_time;
  double local_replication_time;

  Graph()
  {
    threads = numa_num_configured_cpus();
    //  threads=6;
    sockets = numa_num_configured_nodes();
    threads_per_socket = threads / sockets;
    all_wait_time = 0.0;
    all_overlap_time = 0.0;
    all_compute_time = 0.0;
    all_movein_time = 0.0;
    all_kernel_time = 0.0;
    all_moveout_time = 0.0;
    all_recv_copy_time = 0.0;
    all_recv_kernel_time = 0.0;
    all_recv_wait_time = 0.0;
    all_recv_thread_join_time = 0.0;
    all_cuda_sync_time = 0;
    all_replication_time = 0.0;
    local_replication_time = 0.0;
    replication_threshold = 0;
    init();
    config = new inputinfo;
    Nts=new NtsScheduler();
    NtsComm=new NtsGraphCommunicator();
    cpp=-1;
  }
  void init_message_map_amount()
  {
    if (partition_id == 0)
      printf("GNNmini::Init Message\n");
    for (int layer = 0; layer < gnnctx->layer_size.size() - 1; layer++)
    {

      process_edges_simple<int, int>(                            // For EACH Vertex Processing
          [&](VertexId src, VertexAdjList<Empty> outgoing_adj) { //pull
            if (!RepVtx[layer]->get_bit(src))
            {
              message_amount[layer][get_partition_id(src)] += 1;
              message_write_offset[layer][src] = 1;
            }

          },
          [&](VertexId src, int msg) {
            return 0;
          });

      for (int p = 0; p < partitions; p++)
      {

        message_write_offset[layer][partition_offset[p]] = 0;
        for (VertexId src = partition_offset[p] + 1; src < partition_offset[p + 1]; src++)
        {
          message_write_offset[layer][src] += message_write_offset[layer][src - 1];
        }
      }
    }
  }
#if CUDA_ENABLE
  void init_message_buffer(){
        output_cpu_buffer = (float *)cudaMallocPinned(((long)vertices) * gnnctx->max_layer * sizeof(float));
    if (output_cpu_buffer == NULL)
        printf("allocate fail\n");
  }
#endif
  void init_communicatior(){
        NtsComm->init(partition_offset,owned_vertices,partitions,
        sockets,threads,partition_id, local_send_buffer_limit);
  }
  void init_rtminfo()
  {
    rtminfo = new runtimeinfo();
    rtminfo->init_rtminfo();
  }
  void init_gnnctx(std::string layer_string)
  {
    gnnctx = new gnncontext;
    std::stringstream ss(layer_string);
    std::string number;
    gnnctx->layer_size.clear();
    gnnctx->max_layer = 0;
    while (std::getline(ss, number, '-'))
    {
      gnnctx->layer_size.push_back(std::stoi(number));
      gnnctx->max_layer = std::max(gnnctx->max_layer, (size_t)std::stoi(number));
      //printf("layers %d\n", std::stoi(number));
    }
    gnnctx->label_num = gnnctx->layer_size[gnnctx->layer_size.size() - 1];

    gnnctx->p_id = partition_id;
    gnnctx->p_v_e = partition_offset[partition_id + 1];
    gnnctx->p_v_s = partition_offset[partition_id];
    gnnctx->w_num = partitions;
    gnnctx->l_v_num = gnnctx->p_v_e - gnnctx->p_v_s;
    gnnctx->l_e_num=0;
    for (int i = 0; i < sockets; i++){
      gnnctx->l_e_num += (VertexId)incoming_adj_index_backward[i][vertices];
    } 
    message_write_offset = new VertexId *[gnnctx->layer_size.size()];
    message_amount = new VertexId *[gnnctx->layer_size.size()];
    for (int i = 0; i < gnnctx->layer_size.size(); i++)
    {
      message_write_offset[i] = new VertexId[vertices];
      message_amount[i] = new VertexId[partitions];
      memset(message_write_offset[i], 0, sizeof(VertexId) * vertices);
      memset(message_amount[i], 0, sizeof(VertexId) * partitions);
    }
  }

  inline int get_socket_id(int thread_id)
  {
    return thread_id / threads_per_socket;
  }

  inline int get_socket_offset(int thread_id)
  {
    return thread_id % threads_per_socket;
  }

  void init()
  {
    edge_data_size = std::is_same<EdgeData, Empty>::value ? 0 : sizeof(EdgeData);
    unit_size = sizeof(VertexId) + edge_data_size;
    edge_unit_size = sizeof(VertexId) + unit_size;

    assert(numa_available() != -1);
    assert(sizeof(unsigned long) == 8); // assume unsigned long is 64-bit

    char nodestring[sockets * 2 + 1];
    for (int i = 0; i < sockets * 2 + 1; i++)
      nodestring[i] = '\0';
    nodestring[0] = '0';
    for (int s_i = 1; s_i < sockets; s_i++)
    {
      nodestring[s_i * 2 - 1] = ',';
      nodestring[s_i * 2] = '0' + s_i;
    }
    struct bitmask *nodemask = numa_parse_nodestring(nodestring);
    numa_set_interleave_mask(nodemask);

    omp_set_dynamic(0);
    omp_set_num_threads(threads);
    thread_state = new ThreadState *[threads];
    local_send_buffer_limit = 16;
    local_send_buffer = new MessageBuffer *[threads];
    for (int t_i = 0; t_i < threads; t_i++)
    {
      thread_state[t_i] = (ThreadState *)numa_alloc_onnode(sizeof(ThreadState), get_socket_id(t_i));
      local_send_buffer[t_i] = (MessageBuffer *)numa_alloc_onnode(sizeof(MessageBuffer), get_socket_id(t_i));
      local_send_buffer[t_i]->init(get_socket_id(t_i));
    }
#pragma omp parallel for
    for (int t_i = 0; t_i < threads; t_i++)
    {
      int s_i = get_socket_id(t_i);
      assert(numa_run_on_node(s_i) == 0);
#ifdef PRINT_DEBUG_MESSAGES
// printf("thread-%d bound to socket-%d\n", t_i, s_i);
#endif
    }
#ifdef PRINT_DEBUG_MESSAGES
// printf("threads=%d*%d\n", sockets, threads_per_socket);
// printf("interleave on %s\n", nodestring);
#endif

    MPI_Comm_rank(MPI_COMM_WORLD, &partition_id); 
    MPI_Comm_size(MPI_COMM_WORLD, &partitions);  
    send_buffer = new MessageBuffer **[partitions];
    recv_buffer = new MessageBuffer **[partitions];
    for (int i = 0; i < partitions; i++)
    {
      send_buffer[i] = new MessageBuffer *[sockets];
      recv_buffer[i] = new MessageBuffer *[sockets];
      for (int s_i = 0; s_i < sockets; s_i++)
      {
        send_buffer[i][s_i] = (MessageBuffer *)numa_alloc_onnode(sizeof(MessageBuffer), s_i);
        send_buffer[i][s_i]->init(s_i);
        recv_buffer[i][s_i] = (MessageBuffer *)numa_alloc_onnode(sizeof(MessageBuffer), s_i);
        recv_buffer[i][s_i]->init(s_i);
      }
    }

    alpha = 12 * (partitions + 1);
    //alpha = 4;
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // fill a vertex array with a specific value
  template <typename T>
  void fill_vertex_array(T *array, T value)
  {
//      printf("%d %d\n",partition_offset[partition_id],partition_offset[partition_id+1]);
#pragma omp parallel for
    for (VertexId v_i = partition_offset[partition_id]; v_i < partition_offset[partition_id + 1]; v_i++)
    {
      array[v_i] = value;
    }
  }

  /////////////////////////add by 
  template <typename T>
  void fill_vertex_array_long(T *array, T value, size_t length)
  { //fill_vertex_array
#pragma omp parallel for
    for (VertexId v_i = partition_offset[partition_id]; v_i < partition_offset[partition_id + 1]; v_i++)
    {
      memcpy(array[v_i].data, value.data, length * sizeof(T));
    }
  }

  template <typename T>
  T *alloc_vertex_array(int size)
  { //create a
    char *array = (char *)mmap(NULL, size * sizeof(T) * vertices, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    assert(array != NULL);
    for (int s_i = 0; s_i < sockets; s_i++)
    {
      numa_tonode_memory(array + size * sizeof(T) * local_partition_offset[s_i], size * sizeof(T) * (local_partition_offset[s_i + 1] - local_partition_offset[s_i]), s_i);
    }
    return (T *)array;
  }
  template <typename T>
  T *alloc_vertex_array_local(int size)
  { //create a
    char *array = (char *)mmap(NULL, size * sizeof(T) * (partition_offset[partition_id + 1] - partition_offset[partition_id]),
                               PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    assert(array != NULL);
    for (int s_i = 0; s_i < sockets; s_i++)
    {
      numa_tonode_memory(array + size * sizeof(T) * (local_partition_offset[s_i] - local_partition_offset[0]), size * sizeof(T) * (local_partition_offset[s_i + 1] - local_partition_offset[s_i]), s_i);
    }
    return (T *)array;
  }

  // allocate a numa-aware vertex array
  template <typename T>
  T *alloc_vertex_array()
  { //create a
    char *array = (char *)mmap(NULL, sizeof(T) * vertices, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    assert(array != NULL);
    for (int s_i = 0; s_i < sockets; s_i++)
    {
      numa_tonode_memory(array + sizeof(T) * local_partition_offset[s_i], sizeof(T) * (local_partition_offset[s_i + 1] - local_partition_offset[s_i]), s_i);
    }
    return (T *)array;
  }

  template <typename T>
  T *alloc_vertex_array_local()
  { //create a
    char *array = (char *)mmap(NULL, sizeof(T) * (partition_offset[partition_id + 1] - partition_offset[partition_id]),
                               PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    assert(array != NULL);
    for (int s_i = 0; s_i < sockets; s_i++)
    {
      numa_tonode_memory(array + sizeof(T) * (local_partition_offset[s_i] - local_partition_offset[0]), sizeof(T) * (local_partition_offset[s_i + 1] - local_partition_offset[s_i]), s_i);
    }
    return (T *)array;
  }

  template <typename T>
  T *alloc_pointer_vertex_array()
  { //create a
    char *array = (char *)mmap(NULL, sizeof(T) * vertices, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    assert(array != NULL);
    for (int s_i = 0; s_i < sockets; s_i++)
    {
      numa_tonode_memory(array + sizeof(T) * local_partition_offset[s_i], sizeof(T) * (local_partition_offset[s_i + 1] - local_partition_offset[s_i]), s_i);
    }
    return (T *)array;
  }

  // deallocate a vertex array
  template <typename T>
  T *dealloc_vertex_array(T *array)
  {
    numa_free(array, sizeof(T) * vertices);
  }

  // allocate a numa-oblivious vertex array
  template <typename T>
  T *alloc_interleaved_vertex_array()
  {
    T *array = (T *)numa_alloc_interleaved(sizeof(T) * vertices);
    assert(array != NULL);
    return array;
  }

  // dump a vertex array to path
  template <typename T>
  void dump_vertex_array(T *array, std::string path)
  { //persistent array to path
    long file_length = sizeof(T) * vertices;
    if (!file_exists(path) || file_size(path) != file_length)
    {
      if (partition_id == 0)
      {
        FILE *fout = fopen(path.c_str(), "wb");
        char *buffer = new char[PAGESIZE];
        for (long offset = 0; offset < file_length;)
        {
          if (file_length - offset >= PAGESIZE)
          {
            fwrite(buffer, 1, PAGESIZE, fout);
            offset += PAGESIZE;
          }
          else
          {
            fwrite(buffer, 1, file_length - offset, fout);
            offset += file_length - offset;
          }
        }
        fclose(fout);
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
    int fd = open(path.c_str(), O_RDWR);
    assert(fd != -1);
    long offset = sizeof(T) * partition_offset[partition_id];
    long end_offset = sizeof(T) * partition_offset[partition_id + 1];
    void *data = (void *)array;
    assert(lseek(fd, offset, SEEK_SET) != -1);
    while (offset < end_offset)
    {
      long bytes = write(fd, data + offset, end_offset - offset);
      assert(bytes != -1);
      offset += bytes;
    }
    assert(close(fd) == 0);
  }

  // restore a vertex array from path
  template <typename T>
  void restore_vertex_array(T *array, std::string path)
  {
    long file_length = sizeof(T) * vertices;
    if (!file_exists(path) || file_size(path) != file_length)
    {
      assert(false);
    }
    int fd = open(path.c_str(), O_RDWR);
    assert(fd != -1);
    long offset = sizeof(T) * partition_offset[partition_id];
    long end_offset = sizeof(T) * partition_offset[partition_id + 1];
    void *data = (void *)array;
    assert(lseek(fd, offset, SEEK_SET) != -1);
    while (offset < end_offset)
    {
      long bytes = read(fd, data + offset, end_offset - offset);
      assert(bytes != -1);
      offset += bytes;
    }
    assert(close(fd) == 0);
  }

  // gather a vertex array
  template <typename T>
  void gather_vertex_array(T *array, int root)
  {
    if (partition_id != root)
    {
      MPI_Send(array + partition_offset[partition_id], sizeof(T) * owned_vertices, MPI_CHAR, root, GatherVertexArray, MPI_COMM_WORLD);
    }
    else
    {
      for (int i = 0; i < partitions; i++)
      {
        if (i == partition_id)
          continue;
        MPI_Status recv_status;
        MPI_Recv(array + partition_offset[i], sizeof(T) * (partition_offset[i + 1] - partition_offset[i]), MPI_CHAR, i, GatherVertexArray, MPI_COMM_WORLD, &recv_status);
        int length;
        MPI_Get_count(&recv_status, MPI_CHAR, &length);
        assert(length == sizeof(T) * (partition_offset[i + 1] - partition_offset[i]));
      }
    }
  }

  // allocate a vertex subset
  VertexSubset *alloc_vertex_subset()
  {
    return new VertexSubset(vertices);
  }

  int get_partition_id(VertexId v_i)
  {
    for (int i = 0; i < partitions; i++)
    {
      if (v_i >= partition_offset[i] && v_i < partition_offset[i + 1])
      {
        return i;
      }
    }
    printf("wrong vertex%d\n", v_i);
    assert(false);
  }
  long get_max_partition_size(){
      long max_partition_size=0;
    for (int i = 0; i < partitions; i++)
    {
      for (int s_i = 0; s_i < sockets; s_i++)
      {
        max_partition_size=std::max(max_partition_size,(long)(partition_offset[i + 1] - partition_offset[i]));
      }
    }
      return max_partition_size;
  }

  int get_local_partition_id(VertexId v_i)
  {
    for (int s_i = 0; s_i < sockets; s_i++)
    {
      if (v_i >= local_partition_offset[s_i] && v_i < local_partition_offset[s_i + 1])
      {
        return s_i;
      }
    }
    std::cout << v_i << std::endl;
    assert(false);
  }

  // load a directed graph and make it undirected
  void load_undirected_from_directed(std::string path, VertexId vertices)
  {
    double prep_time = 0;
    prep_time -= MPI_Wtime();

    symmetric = true;

    MPI_Datatype vid_t = get_mpi_data_type<VertexId>();

    this->vertices = vertices;
    long total_bytes = file_size(path.c_str());
    this->edges = total_bytes / edge_unit_size;
#ifdef PRINT_DEBUG_MESSAGES
    if (partition_id == 0)
    {
      printf("|V| = %u, |E| = %lu\n", vertices, edges);
    }
#endif

    EdgeId read_edges = edges / partitions;
    if (partition_id == partitions - 1)
    {
      read_edges += edges % partitions;
    }
    long bytes_to_read = edge_unit_size * read_edges;
    long read_offset = edge_unit_size * (edges / partitions * partition_id);
    long read_bytes;
    int fin = open(path.c_str(), O_RDONLY);
    EdgeUnit<EdgeData> *read_edge_buffer = new EdgeUnit<EdgeData>[CHUNKSIZE];

    out_degree = alloc_interleaved_vertex_array<VertexId>();
    for (VertexId v_i = 0; v_i < vertices; v_i++)
    {
      out_degree[v_i] = 0;
    }
    assert(lseek(fin, read_offset, SEEK_SET) == read_offset);
    read_bytes = 0;
    while (read_bytes < bytes_to_read)
    {
      long curr_read_bytes;
      if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE)
      {
        curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
      }
      else
      {
        curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
      }
      assert(curr_read_bytes >= 0);
      read_bytes += curr_read_bytes;
      EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
      // #pragma omp parallel for
      for (EdgeId e_i = 0; e_i < curr_read_edges; e_i++)
      {
        VertexId src = read_edge_buffer[e_i].src;
        VertexId dst = read_edge_buffer[e_i].dst;
        __sync_fetch_and_add(&out_degree[src], 1);
        __sync_fetch_and_add(&out_degree[dst], 1);
      }
    }
    MPI_Allreduce(MPI_IN_PLACE, out_degree, vertices, vid_t, MPI_SUM, MPI_COMM_WORLD);

    // locality-aware chunking
    partition_offset = new VertexId[partitions + 1];
    partition_offset[0] = 0;
    EdgeId remained_amount = edges * 2 + EdgeId(vertices) * alpha;
    for (int i = 0; i < partitions; i++)
    {
      VertexId remained_partitions = partitions - i;
      EdgeId expected_chunk_size = remained_amount / remained_partitions;
      if (remained_partitions == 1)
      {
        partition_offset[i + 1] = vertices;
      }
      else
      {
        EdgeId got_edges = 0;
        for (VertexId v_i = partition_offset[i]; v_i < vertices; v_i++)
        {
          got_edges += out_degree[v_i] + alpha;
          if (got_edges > expected_chunk_size)
          {
            partition_offset[i + 1] = v_i;
            break;
          }
        }
        partition_offset[i + 1] = (partition_offset[i + 1]) / PAGESIZE * PAGESIZE; // aligned with pages
      }
      for (VertexId v_i = partition_offset[i]; v_i < partition_offset[i + 1]; v_i++)
      {
        remained_amount -= out_degree[v_i] + alpha;
      }
    }
    assert(partition_offset[partitions] == vertices);
    owned_vertices = partition_offset[partition_id + 1] - partition_offset[partition_id];
    // check consistency of partition boundaries
    VertexId *global_partition_offset = new VertexId[partitions + 1];
    MPI_Allreduce(partition_offset, global_partition_offset, partitions + 1, vid_t, MPI_MAX, MPI_COMM_WORLD);
    for (int i = 0; i <= partitions; i++)
    {
      assert(partition_offset[i] == global_partition_offset[i]);
    }
    MPI_Allreduce(partition_offset, global_partition_offset, partitions + 1, vid_t, MPI_MIN, MPI_COMM_WORLD);
    for (int i = 0; i <= partitions; i++)
    {
      assert(partition_offset[i] == global_partition_offset[i]);
    }
#ifdef PRINT_DEBUG_MESSAGES
    if (partition_id == 0)
    {
      for (int i = 0; i < partitions; i++)
      {
        EdgeId part_out_edges = 0;
        for (VertexId v_i = partition_offset[i]; v_i < partition_offset[i + 1]; v_i++)
        {
          part_out_edges += out_degree[v_i];
        }
        printf("|V'_%d| = %u |E_%d| = %lu\n", i, partition_offset[i + 1] - partition_offset[i], i, part_out_edges);
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    delete[] global_partition_offset;
    {
      // NUMA-aware sub-chunking
      local_partition_offset = new VertexId[sockets + 1];
      EdgeId part_out_edges = 0;
      for (VertexId v_i = partition_offset[partition_id]; v_i < partition_offset[partition_id + 1]; v_i++)
      {
        part_out_edges += out_degree[v_i];
      }
      local_partition_offset[0] = partition_offset[partition_id];
      EdgeId remained_amount = part_out_edges + EdgeId(owned_vertices) * alpha;
      for (int s_i = 0; s_i < sockets; s_i++)
      {
        VertexId remained_partitions = sockets - s_i;
        EdgeId expected_chunk_size = remained_amount / remained_partitions;
        if (remained_partitions == 1)
        {
          local_partition_offset[s_i + 1] = partition_offset[partition_id + 1];
        }
        else
        {
          EdgeId got_edges = 0;
          for (VertexId v_i = local_partition_offset[s_i]; v_i < partition_offset[partition_id + 1]; v_i++)
          {
            got_edges += out_degree[v_i] + alpha;
            if (got_edges > expected_chunk_size)
            {
              local_partition_offset[s_i + 1] = v_i;
              break;
            }
          }
          local_partition_offset[s_i + 1] = (local_partition_offset[s_i + 1]) / PAGESIZE * PAGESIZE; // aligned with pages
        }
        EdgeId sub_part_out_edges = 0;
        for (VertexId v_i = local_partition_offset[s_i]; v_i < local_partition_offset[s_i + 1]; v_i++)
        {
          remained_amount -= out_degree[v_i] + alpha;
          sub_part_out_edges += out_degree[v_i];
        }
#ifdef PRINT_DEBUG_MESSAGES
        printf("|V'_%d_%d| = %u |E_%d| = %lu\n", partition_id, s_i, local_partition_offset[s_i + 1] - local_partition_offset[s_i], partition_id, sub_part_out_edges);
#endif
      }
    }

    VertexId *filtered_out_degree = alloc_vertex_array<VertexId>();
    for (VertexId v_i = partition_offset[partition_id]; v_i < partition_offset[partition_id + 1]; v_i++)
    {
      filtered_out_degree[v_i] = out_degree[v_i];
    }
    numa_free(out_degree, sizeof(VertexId) * vertices);
    out_degree = filtered_out_degree;
    in_degree = out_degree;

    int *buffered_edges = new int[partitions];
    std::vector<char> *send_buffer = new std::vector<char>[partitions];
    for (int i = 0; i < partitions; i++)
    {
      send_buffer[i].resize(edge_unit_size * CHUNKSIZE);
    }
    EdgeUnit<EdgeData> *recv_buffer = new EdgeUnit<EdgeData>[CHUNKSIZE];

    // constructing symmetric edges
    EdgeId recv_outgoing_edges = 0;
    outgoing_edges = new EdgeId[sockets];
    outgoing_adj_index = new EdgeId *[sockets];
    outgoing_adj_list = new AdjUnit<EdgeData> *[sockets];
    outgoing_adj_bitmap = new Bitmap *[sockets];
    for (int s_i = 0; s_i < sockets; s_i++)
    {
      outgoing_adj_bitmap[s_i] = new Bitmap(vertices);
      outgoing_adj_bitmap[s_i]->clear();
      outgoing_adj_index[s_i] = (EdgeId *)numa_alloc_onnode(sizeof(EdgeId) * (vertices + 1), s_i);
    }
    {
      std::thread recv_thread_dst([&]() {
        int finished_count = 0;
        MPI_Status recv_status;
        while (finished_count < partitions)
        {
          MPI_Probe(MPI_ANY_SOURCE, ShuffleGraph, MPI_COMM_WORLD, &recv_status);
          int i = recv_status.MPI_SOURCE;
          assert(recv_status.MPI_TAG == ShuffleGraph && i >= 0 && i < partitions);
          int recv_bytes;
          MPI_Get_count(&recv_status, MPI_CHAR, &recv_bytes);
          if (recv_bytes == 1)
          {
            finished_count += 1;
            char c;
            MPI_Recv(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            continue;
          }
          assert(recv_bytes % edge_unit_size == 0);
          int recv_edges = recv_bytes / edge_unit_size;
          MPI_Recv(recv_buffer, edge_unit_size * recv_edges, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          // #pragma omp parallel for
          for (EdgeId e_i = 0; e_i < recv_edges; e_i++)
          {
            VertexId src = recv_buffer[e_i].src;
            VertexId dst = recv_buffer[e_i].dst;
            assert(dst >= partition_offset[partition_id] && dst < partition_offset[partition_id + 1]);
            int dst_part = get_local_partition_id(dst);
            if (!outgoing_adj_bitmap[dst_part]->get_bit(src))
            {
              outgoing_adj_bitmap[dst_part]->set_bit(src);
              outgoing_adj_index[dst_part][src] = 0;
            }
            __sync_fetch_and_add(&outgoing_adj_index[dst_part][src], 1);
          }
          recv_outgoing_edges += recv_edges;
        }
      });
      for (int i = 0; i < partitions; i++)
      {
        buffered_edges[i] = 0;
      }
      assert(lseek(fin, read_offset, SEEK_SET) == read_offset);
      read_bytes = 0;
      while (read_bytes < bytes_to_read)
      {
        long curr_read_bytes;
        if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE)
        {
          curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
        }
        else
        {
          curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
        }
        assert(curr_read_bytes >= 0);
        read_bytes += curr_read_bytes;
        EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
        for (EdgeId e_i = 0; e_i < curr_read_edges; e_i++)
        {
          VertexId dst = read_edge_buffer[e_i].dst;
          int i = get_partition_id(dst);
          memcpy(send_buffer[i].data() + edge_unit_size * buffered_edges[i], &read_edge_buffer[e_i], edge_unit_size);
          buffered_edges[i] += 1;
          if (buffered_edges[i] == CHUNKSIZE)
          {
            MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
            buffered_edges[i] = 0;
          }
        }
        for (EdgeId e_i = 0; e_i < curr_read_edges; e_i++)
        {
          // std::swap(read_edge_buffer[e_i].src, read_edge_buffer[e_i].dst);
          VertexId tmp = read_edge_buffer[e_i].src;
          read_edge_buffer[e_i].src = read_edge_buffer[e_i].dst;
          read_edge_buffer[e_i].dst = tmp;
        }
        for (EdgeId e_i = 0; e_i < curr_read_edges; e_i++)
        {
          VertexId dst = read_edge_buffer[e_i].dst;
          int i = get_partition_id(dst);
          memcpy(send_buffer[i].data() + edge_unit_size * buffered_edges[i], &read_edge_buffer[e_i], edge_unit_size);
          buffered_edges[i] += 1;
          if (buffered_edges[i] == CHUNKSIZE)
          {
            MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
            buffered_edges[i] = 0;
          }
        }
      }
      for (int i = 0; i < partitions; i++)
      {
        if (buffered_edges[i] == 0)
          continue;
        MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
        buffered_edges[i] = 0;
      }
      for (int i = 0; i < partitions; i++)
      {
        char c = 0;
        MPI_Send(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
      }
      recv_thread_dst.join();
#ifdef PRINT_DEBUG_MESSAGES
      printf("machine(%d) got %lu symmetric edges\n", partition_id, recv_outgoing_edges);
#endif
    }
    compressed_outgoing_adj_vertices = new VertexId[sockets];
    compressed_outgoing_adj_index = new CompressedAdjIndexUnit *[sockets];
    for (int s_i = 0; s_i < sockets; s_i++)
    {
      outgoing_edges[s_i] = 0;
      compressed_outgoing_adj_vertices[s_i] = 0;
      for (VertexId v_i = 0; v_i < vertices; v_i++)
      {
        if (outgoing_adj_bitmap[s_i]->get_bit(v_i))
        {
          outgoing_edges[s_i] += outgoing_adj_index[s_i][v_i];
          compressed_outgoing_adj_vertices[s_i] += 1;
        }
      }
      compressed_outgoing_adj_index[s_i] = (CompressedAdjIndexUnit *)numa_alloc_onnode(sizeof(CompressedAdjIndexUnit) * (compressed_outgoing_adj_vertices[s_i] + 1), s_i);
      compressed_outgoing_adj_index[s_i][0].index = 0;
      EdgeId last_e_i = 0;
      compressed_outgoing_adj_vertices[s_i] = 0;
      for (VertexId v_i = 0; v_i < vertices; v_i++)
      {
        if (outgoing_adj_bitmap[s_i]->get_bit(v_i))
        {
          outgoing_adj_index[s_i][v_i] = last_e_i + outgoing_adj_index[s_i][v_i];
          last_e_i = outgoing_adj_index[s_i][v_i];
          compressed_outgoing_adj_index[s_i][compressed_outgoing_adj_vertices[s_i]].vertex = v_i;
          compressed_outgoing_adj_vertices[s_i] += 1;
          compressed_outgoing_adj_index[s_i][compressed_outgoing_adj_vertices[s_i]].index = last_e_i;
        }
      }
      for (VertexId p_v_i = 0; p_v_i < compressed_outgoing_adj_vertices[s_i]; p_v_i++)
      {
        VertexId v_i = compressed_outgoing_adj_index[s_i][p_v_i].vertex;
        outgoing_adj_index[s_i][v_i] = compressed_outgoing_adj_index[s_i][p_v_i].index;
        outgoing_adj_index[s_i][v_i + 1] = compressed_outgoing_adj_index[s_i][p_v_i + 1].index;
      }
#ifdef PRINT_DEBUG_MESSAGES
      printf("part(%d) E_%d has %lu symmetric edges\n", partition_id, s_i, outgoing_edges[s_i]);
#endif
      outgoing_adj_list[s_i] = (AdjUnit<EdgeData> *)numa_alloc_onnode(unit_size * outgoing_edges[s_i], s_i);
    }
    {
      std::thread recv_thread_dst([&]() {
        int finished_count = 0;
        MPI_Status recv_status;
        while (finished_count < partitions)
        {
          MPI_Probe(MPI_ANY_SOURCE, ShuffleGraph, MPI_COMM_WORLD, &recv_status);
          int i = recv_status.MPI_SOURCE;
          assert(recv_status.MPI_TAG == ShuffleGraph && i >= 0 && i < partitions);
          int recv_bytes;
          MPI_Get_count(&recv_status, MPI_CHAR, &recv_bytes);
          if (recv_bytes == 1)
          {
            finished_count += 1;
            char c;
            MPI_Recv(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            continue;
          }
          assert(recv_bytes % edge_unit_size == 0);
          int recv_edges = recv_bytes / edge_unit_size;
          MPI_Recv(recv_buffer, edge_unit_size * recv_edges, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#pragma omp parallel for
          for (EdgeId e_i = 0; e_i < recv_edges; e_i++)
          {
            VertexId src = recv_buffer[e_i].src;
            VertexId dst = recv_buffer[e_i].dst;
            assert(dst >= partition_offset[partition_id] && dst < partition_offset[partition_id + 1]);
            int dst_part = get_local_partition_id(dst);
            EdgeId pos = __sync_fetch_and_add(&outgoing_adj_index[dst_part][src], 1);
            outgoing_adj_list[dst_part][pos].neighbour = dst;
            if (!std::is_same<EdgeData, Empty>::value)
            {
              outgoing_adj_list[dst_part][pos].edge_data = recv_buffer[e_i].edge_data;
            }
          }
        }
      });
      for (int i = 0; i < partitions; i++)
      {
        buffered_edges[i] = 0;
      }
      assert(lseek(fin, read_offset, SEEK_SET) == read_offset);
      read_bytes = 0;
      while (read_bytes < bytes_to_read)
      {
        long curr_read_bytes;
        if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE)
        {
          curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
        }
        else
        {
          curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
        }
        assert(curr_read_bytes >= 0);
        read_bytes += curr_read_bytes;
        EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
        for (EdgeId e_i = 0; e_i < curr_read_edges; e_i++)
        {
          VertexId dst = read_edge_buffer[e_i].dst;
          int i = get_partition_id(dst);
          memcpy(send_buffer[i].data() + edge_unit_size * buffered_edges[i], &read_edge_buffer[e_i], edge_unit_size);
          buffered_edges[i] += 1;
          if (buffered_edges[i] == CHUNKSIZE)
          {
            MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
            buffered_edges[i] = 0;
          }
        }
        for (EdgeId e_i = 0; e_i < curr_read_edges; e_i++)
        {
          // std::swap(read_edge_buffer[e_i].src, read_edge_buffer[e_i].dst);
          VertexId tmp = read_edge_buffer[e_i].src;
          read_edge_buffer[e_i].src = read_edge_buffer[e_i].dst;
          read_edge_buffer[e_i].dst = tmp;
        }
        for (EdgeId e_i = 0; e_i < curr_read_edges; e_i++)
        {
          VertexId dst = read_edge_buffer[e_i].dst;
          int i = get_partition_id(dst);
          memcpy(send_buffer[i].data() + edge_unit_size * buffered_edges[i], &read_edge_buffer[e_i], edge_unit_size);
          buffered_edges[i] += 1;
          if (buffered_edges[i] == CHUNKSIZE)
          {
            MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
            buffered_edges[i] = 0;
          }
        }
      }
      for (int i = 0; i < partitions; i++)
      {
        if (buffered_edges[i] == 0)
          continue;
        MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
        buffered_edges[i] = 0;
      }
      for (int i = 0; i < partitions; i++)
      {
        char c = 0;
        MPI_Send(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
      }
      recv_thread_dst.join();
    }
    for (int s_i = 0; s_i < sockets; s_i++)
    {
      for (VertexId p_v_i = 0; p_v_i < compressed_outgoing_adj_vertices[s_i]; p_v_i++)
      {
        VertexId v_i = compressed_outgoing_adj_index[s_i][p_v_i].vertex;
        outgoing_adj_index[s_i][v_i] = compressed_outgoing_adj_index[s_i][p_v_i].index;
        outgoing_adj_index[s_i][v_i + 1] = compressed_outgoing_adj_index[s_i][p_v_i + 1].index;
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    incoming_edges = outgoing_edges;
    incoming_adj_index = outgoing_adj_index;
    incoming_adj_list = outgoing_adj_list;
    incoming_adj_bitmap = outgoing_adj_bitmap;
    compressed_incoming_adj_vertices = compressed_outgoing_adj_vertices;
    compressed_incoming_adj_index = compressed_outgoing_adj_index;
    MPI_Barrier(MPI_COMM_WORLD);

    delete[] buffered_edges;
    delete[] send_buffer;
    delete[] read_edge_buffer;
    delete[] recv_buffer;
    close(fin);

    tune_chunks();
    tuned_chunks_sparse = tuned_chunks_dense;

    prep_time += MPI_Wtime();

#ifdef PRINT_DEBUG_MESSAGES
    if (partition_id == 0)
    {
      printf("preprocessing cost: %.2lf (s)\n", prep_time);
    }
#endif
  }

  // transpose the graph
  void transpose()
  {
    std::swap(out_degree, in_degree);
    std::swap(outgoing_edges, incoming_edges);
    std::swap(outgoing_adj_index, incoming_adj_index);
    std::swap(outgoing_adj_bitmap, incoming_adj_bitmap);
    std::swap(outgoing_adj_list, incoming_adj_list);
    std::swap(tuned_chunks_dense, tuned_chunks_sparse);
    std::swap(compressed_outgoing_adj_vertices, compressed_incoming_adj_vertices);
    std::swap(compressed_outgoing_adj_index, compressed_incoming_adj_index);
  }

  // load a directed graph from path
  void load_directed(std::string path, VertexId vertices)
  {
    double prep_time = 0;
    prep_time -= MPI_Wtime();
    filename = path;
    symmetric = false;

    MPI_Datatype vid_t = get_mpi_data_type<VertexId>();

    this->vertices = vertices;
    long total_bytes = file_size(path.c_str());
    this->edges = total_bytes / edge_unit_size;
#ifdef PRINT_DEBUG_MESSAGES
    if (partition_id == 0)
    {
      printf("|V| = %u, |E| = %lu\n", vertices, edges);
    }
#endif

    EdgeId read_edges = edges / partitions;
    if (partition_id == partitions - 1)
    {
      read_edges += edges % partitions;
    } // the final partition has to gather all the vertices.|numberof edges|
    long bytes_to_read = edge_unit_size * read_edges;
    long read_offset = edge_unit_size * (edges / partitions * partition_id);
    long read_bytes;
    int fin = open(path.c_str(), O_RDONLY);
    EdgeUnit<EdgeData> *read_edge_buffer = new EdgeUnit<EdgeData>[CHUNKSIZE];

    out_degree = alloc_interleaved_vertex_array<VertexId>();
    for (VertexId v_i = 0; v_i < vertices; v_i++)
    {
      out_degree[v_i] = 0;
    }
    assert(lseek(fin, read_offset, SEEK_SET) == read_offset);
    read_bytes = 0;
    while (read_bytes < bytes_to_read)
    { // load the graph with a big chunk iteratively
      long curr_read_bytes;
      if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE)
      {
        curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
      }
      else
      {
        curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
      }
      assert(curr_read_bytes >= 0);
      read_bytes += curr_read_bytes;
      EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
#pragma omp parallel for
      for (EdgeId e_i = 0; e_i < curr_read_edges; e_i++)
      {
        VertexId src = read_edge_buffer[e_i].src;
        VertexId dst = read_edge_buffer[e_i].dst;
        __sync_fetch_and_add(&out_degree[src], 1);
      }
    }
    MPI_Allreduce(MPI_IN_PLACE, out_degree, vertices, vid_t, MPI_SUM, MPI_COMM_WORLD);
    //*************************************************************************************Gather all vertex count before this stage.
 
    // locality-aware chunking
    partition_offset = new VertexId[partitions + 1];
    partition_offset[0] = 0;
    EdgeId remained_amount = edges + EdgeId(vertices) * alpha;
    for (int i = 0; i < partitions; i++)
    {
      VertexId remained_partitions = partitions - i;
      EdgeId expected_chunk_size = remained_amount / remained_partitions;
      if (remained_partitions == 1)
      {
        partition_offset[i + 1] = vertices; //prefix_sum like
      }
      else
      {
        EdgeId got_edges = 0;
        for (VertexId v_i = partition_offset[i]; v_i < vertices; v_i++)
        {
          got_edges += out_degree[v_i] + alpha;
          if (got_edges > expected_chunk_size)
          {
            partition_offset[i + 1] = v_i;
            break;
          }
        }
        partition_offset[i + 1] = (partition_offset[i + 1]) / PAGESIZE * PAGESIZE; // aligned with pages
      }
      for (VertexId v_i = partition_offset[i]; v_i < partition_offset[i + 1]; v_i++)
      {
        remained_amount -= out_degree[v_i] + alpha;
      }
    }
    assert(partition_offset[partitions] == vertices);
    owned_vertices = partition_offset[partition_id + 1] - partition_offset[partition_id];
    //***************************************************************************reorganized data distribution
    // check consistency of partition boundaries
    VertexId *global_partition_offset = new VertexId[partitions + 1];
    MPI_Allreduce(partition_offset, global_partition_offset, partitions + 1, vid_t, MPI_MAX, MPI_COMM_WORLD);
    for (int i = 0; i <= partitions; i++)
    {
      assert(partition_offset[i] == global_partition_offset[i]);
    }
    MPI_Allreduce(partition_offset, global_partition_offset, partitions + 1, vid_t, MPI_MIN, MPI_COMM_WORLD);
    for (int i = 0; i <= partitions; i++)
    {
      assert(partition_offset[i] == global_partition_offset[i]);
    } //Double-check??
#ifdef PRINT_DEBUG_MESSAGES
    if (partition_id == 0)
    {
      for (int i = 0; i < partitions; i++)
      {
        EdgeId part_out_edges = 0;
        for (VertexId v_i = partition_offset[i]; v_i < partition_offset[i + 1]; v_i++)
        {
          part_out_edges += out_degree[v_i];
        }
        printf("|V'_%d| = %u |E^dense_%d| = %lu\n", i, partition_offset[i + 1] - partition_offset[i], i, part_out_edges);
      }
    }
#endif
    delete[] global_partition_offset;
    {
      // NUMA-aware sub-chunking
      local_partition_offset = new VertexId[sockets + 1];
      EdgeId part_out_edges = 0;
      for (VertexId v_i = partition_offset[partition_id]; v_i < partition_offset[partition_id + 1]; v_i++)
      {
        part_out_edges += out_degree[v_i];
      }
      local_partition_offset[0] = partition_offset[partition_id];
      EdgeId remained_amount = part_out_edges + EdgeId(owned_vertices) * alpha;
      for (int s_i = 0; s_i < sockets; s_i++)
      {
        VertexId remained_partitions = sockets - s_i;
        EdgeId expected_chunk_size = remained_amount / remained_partitions;
        if (remained_partitions == 1)
        {
          local_partition_offset[s_i + 1] = partition_offset[partition_id + 1];
        }
        else
        {
          EdgeId got_edges = 0;
          for (VertexId v_i = local_partition_offset[s_i]; v_i < partition_offset[partition_id + 1]; v_i++)
          {
            got_edges += out_degree[v_i] + alpha;
            if (got_edges > expected_chunk_size)
            {
              local_partition_offset[s_i + 1] = v_i;
              break;
            }
          }
          local_partition_offset[s_i + 1] = (local_partition_offset[s_i + 1]) / PAGESIZE * PAGESIZE; // aligned with pages
        }
        EdgeId sub_part_out_edges = 0;
        for (VertexId v_i = local_partition_offset[s_i]; v_i < local_partition_offset[s_i + 1]; v_i++)
        {
          remained_amount -= out_degree[v_i] + alpha;
          sub_part_out_edges += out_degree[v_i];
        }
#ifdef PRINT_DEBUG_MESSAGES
        printf("|V'_%d_%d| = %u |E^dense_%d_%d| = %lu\n", partition_id, s_i, local_partition_offset[s_i + 1] - local_partition_offset[s_i], partition_id, s_i, sub_part_out_edges);
#endif
      }
    }
    // **************************************************************************Partition the graph inside a node to explorit NUMA-aware.

    VertexId *filtered_out_degree = alloc_vertex_array<VertexId>();
    //    for (VertexId v_i=partition_offset[0];v_i<partition_offset[partitions-1];v_i++) {
    //      filtered_out_degree[v_i] = out_degree[v_i];
    //    }
    for (VertexId v_i = partition_offset[partition_id]; v_i < partition_offset[partition_id + 1]; v_i++)
    {
      filtered_out_degree[v_i] = out_degree[v_i];
    }
    out_degree_for_backward = out_degree;
    //numa_free(out_degree, sizeof(VertexId) * vertices);
    out_degree = filtered_out_degree;
    in_degree = alloc_vertex_array<VertexId>();
    in_degree_for_backward = alloc_interleaved_vertex_array<VertexId>();
    for (VertexId v_i = partition_offset[partition_id]; v_i < partition_offset[partition_id + 1]; v_i++)
    {
      in_degree[v_i] = 0;
    }
    for (VertexId v_i = 0; v_i < vertices; v_i++)
    {
      in_degree_for_backward[v_i] = 0;
    }
    int *buffered_edges = new int[partitions];
    std::vector<char> *send_buffer = new std::vector<char>[partitions];
    for (int i = 0; i < partitions; i++)
    {
      send_buffer[i].resize(edge_unit_size * CHUNKSIZE);
    }
    EdgeUnit<EdgeData> *recv_buffer = new EdgeUnit<EdgeData>[CHUNKSIZE];

    EdgeId recv_outgoing_edges = 0;
    outgoing_edges = new EdgeId[sockets];
    outgoing_adj_index = new EdgeId *[sockets];
    outgoing_adj_list = new AdjUnit<EdgeData> *[sockets];
    outgoing_adj_bitmap = new Bitmap *[sockets];
    for (int s_i = 0; s_i < sockets; s_i++)
    {
      outgoing_adj_bitmap[s_i] = new Bitmap(vertices);
      outgoing_adj_bitmap[s_i]->clear();
      outgoing_adj_index[s_i] = (EdgeId *)numa_alloc_onnode(sizeof(EdgeId) * (vertices + 1), s_i);
    }

    {
      std::thread recv_thread_dst([&]() {
        int finished_count = 0;
        MPI_Status recv_status;
        while (finished_count < partitions)
        {
          MPI_Probe(MPI_ANY_SOURCE, ShuffleGraph, MPI_COMM_WORLD, &recv_status);
          int i = recv_status.MPI_SOURCE;
          assert(recv_status.MPI_TAG == ShuffleGraph && i >= 0 && i < partitions);
          int recv_bytes;
          MPI_Get_count(&recv_status, MPI_CHAR, &recv_bytes);
          //printf("recv %d\n", recv_bytes);
          if (recv_bytes == 1)
          {
            finished_count += 1;
            char c;
            MPI_Recv(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            continue;
          }
          assert(recv_bytes % edge_unit_size == 0);
          int recv_edges = recv_bytes / edge_unit_size;
          MPI_Recv(recv_buffer, edge_unit_size * recv_edges, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          // #pragma omp parallel for
          for (EdgeId e_i = 0; e_i < recv_edges; e_i++)
          {
            VertexId src = recv_buffer[e_i].src;
            VertexId dst = recv_buffer[e_i].dst;
            assert(dst >= partition_offset[partition_id] && dst < partition_offset[partition_id + 1]);
            int dst_part = get_local_partition_id(dst);
            if (!outgoing_adj_bitmap[dst_part]->get_bit(src))
            {
              outgoing_adj_bitmap[dst_part]->set_bit(src);
              outgoing_adj_index[dst_part][src] = 0;
            }
            __sync_fetch_and_add(&outgoing_adj_index[dst_part][src], 1);
            __sync_fetch_and_add(&in_degree[dst], 1);
          }
          recv_outgoing_edges += recv_edges;
        }
      });
      for (int i = 0; i < partitions; i++)
      {
        buffered_edges[i] = 0;
      }
      assert(lseek(fin, read_offset, SEEK_SET) == read_offset);
      read_bytes = 0;
      while (read_bytes < bytes_to_read)
      {
        long curr_read_bytes;
        if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE)
        {
          curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
        }
        else
        {
          curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
        }
        assert(curr_read_bytes >= 0);
        read_bytes += curr_read_bytes;
        EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
        for (EdgeId e_i = 0; e_i < curr_read_edges; e_i++)
        {
          VertexId dst = read_edge_buffer[e_i].dst;
          int i = get_partition_id(dst);
          memcpy(send_buffer[i].data() + edge_unit_size * buffered_edges[i], &read_edge_buffer[e_i], edge_unit_size);
          buffered_edges[i] += 1;
          if (buffered_edges[i] == CHUNKSIZE)
          {
            //printf("send_start %d\n", edge_unit_size * buffered_edges[i]);
            MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
            //printf("send_end %d\n", recv_bytes);
            buffered_edges[i] = 0;
          }
        }
      }
      for (int i = 0; i < partitions; i++)
      {
        if (buffered_edges[i] == 0)
          continue;
        MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
        buffered_edges[i] = 0;
      }
      for (int i = 0; i < partitions; i++)
      {
        char c = 0;
        MPI_Send(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
      }
      recv_thread_dst.join();
      for (VertexId v_i = partition_offset[partition_id]; v_i < partition_offset[partition_id + 1]; v_i++)
      {
        in_degree_for_backward[v_i] = in_degree[v_i];
      }
      MPI_Allreduce(MPI_IN_PLACE, in_degree_for_backward, vertices, vid_t, MPI_SUM, MPI_COMM_WORLD);
#ifdef PRINT_DEBUG_MESSAGES
      printf("machine(%d) got %lu sparse mode edges\n", partition_id, recv_outgoing_edges);
#endif
    } //**************************************************************************I think this might be the dense model edge (partitioned by dst.)

    compressed_outgoing_adj_vertices = new VertexId[sockets];
    compressed_outgoing_adj_index = new CompressedAdjIndexUnit *[sockets];
    for (int s_i = 0; s_i < sockets; s_i++)
    {
      outgoing_edges[s_i] = 0;
      compressed_outgoing_adj_vertices[s_i] = 0;
      for (VertexId v_i = 0; v_i < vertices; v_i++)
      {
        if (outgoing_adj_bitmap[s_i]->get_bit(v_i))
        {
          outgoing_edges[s_i] += outgoing_adj_index[s_i][v_i];
          compressed_outgoing_adj_vertices[s_i] += 1;
        }
      }
      compressed_outgoing_adj_index[s_i] = (CompressedAdjIndexUnit *)numa_alloc_onnode(sizeof(CompressedAdjIndexUnit) * (compressed_outgoing_adj_vertices[s_i] + 1), s_i);
      compressed_outgoing_adj_index[s_i][0].index = 0;
      EdgeId last_e_i = 0;
      compressed_outgoing_adj_vertices[s_i] = 0;
      for (VertexId v_i = 0; v_i < vertices; v_i++)
      {
        if (outgoing_adj_bitmap[s_i]->get_bit(v_i))
        {
          outgoing_adj_index[s_i][v_i] = last_e_i + outgoing_adj_index[s_i][v_i];
          last_e_i = outgoing_adj_index[s_i][v_i];
          compressed_outgoing_adj_index[s_i][compressed_outgoing_adj_vertices[s_i]].vertex = v_i;
          compressed_outgoing_adj_vertices[s_i] += 1;
          compressed_outgoing_adj_index[s_i][compressed_outgoing_adj_vertices[s_i]].index = last_e_i;
        }
      }
      for (VertexId p_v_i = 0; p_v_i < compressed_outgoing_adj_vertices[s_i]; p_v_i++)
      {
        VertexId v_i = compressed_outgoing_adj_index[s_i][p_v_i].vertex;
        outgoing_adj_index[s_i][v_i] = compressed_outgoing_adj_index[s_i][p_v_i].index;
        outgoing_adj_index[s_i][v_i + 1] = compressed_outgoing_adj_index[s_i][p_v_i + 1].index;
      }
#ifdef PRINT_DEBUG_MESSAGES
      printf("part(%d) E_%d has %lu sparse mode edges\n", partition_id, s_i, outgoing_edges[s_i]);
#endif
      outgoing_adj_list[s_i] = (AdjUnit<EdgeData> *)numa_alloc_onnode(unit_size * outgoing_edges[s_i], s_i);
    }

    {
      std::thread recv_thread_dst([&]() {
        int finished_count = 0;
        MPI_Status recv_status;
        while (finished_count < partitions)
        {
          MPI_Probe(MPI_ANY_SOURCE, ShuffleGraph, MPI_COMM_WORLD, &recv_status);
          int i = recv_status.MPI_SOURCE;
          assert(recv_status.MPI_TAG == ShuffleGraph && i >= 0 && i < partitions);
          int recv_bytes;
          MPI_Get_count(&recv_status, MPI_CHAR, &recv_bytes);
          if (recv_bytes == 1)
          {
            finished_count += 1;
            char c;
            MPI_Recv(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            continue;
          }
          assert(recv_bytes % edge_unit_size == 0);
          int recv_edges = recv_bytes / edge_unit_size;
          MPI_Recv(recv_buffer, edge_unit_size * recv_edges, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#pragma omp parallel for
          for (EdgeId e_i = 0; e_i < recv_edges; e_i++)
          {
            VertexId src = recv_buffer[e_i].src;
            VertexId dst = recv_buffer[e_i].dst;
            assert(dst >= partition_offset[partition_id] && dst < partition_offset[partition_id + 1]);
            int dst_part = get_local_partition_id(dst);
            EdgeId pos = __sync_fetch_and_add(&outgoing_adj_index[dst_part][src], 1);
            outgoing_adj_list[dst_part][pos].neighbour = dst;
            if (!std::is_same<EdgeData, Empty>::value)
            {
              outgoing_adj_list[dst_part][pos].edge_data = recv_buffer[e_i].edge_data;
            }
          }
        }
      });
      for (int i = 0; i < partitions; i++)
      {
        buffered_edges[i] = 0;
      }
      assert(lseek(fin, read_offset, SEEK_SET) == read_offset);
      read_bytes = 0;
      while (read_bytes < bytes_to_read)
      {
        long curr_read_bytes;
        if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE)
        {
          curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
        }
        else
        {
          curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
        }
        assert(curr_read_bytes >= 0);
        read_bytes += curr_read_bytes;
        EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
        for (EdgeId e_i = 0; e_i < curr_read_edges; e_i++)
        {
          VertexId dst = read_edge_buffer[e_i].dst;
          int i = get_partition_id(dst);
          memcpy(send_buffer[i].data() + edge_unit_size * buffered_edges[i], &read_edge_buffer[e_i], edge_unit_size);
          buffered_edges[i] += 1;
          if (buffered_edges[i] == CHUNKSIZE)
          {
            MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
            buffered_edges[i] = 0;
          }
        }
      }
      for (int i = 0; i < partitions; i++)
      {
        if (buffered_edges[i] == 0)
          continue;
        MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
        buffered_edges[i] = 0;
      }
      for (int i = 0; i < partitions; i++)
      {
        char c = 0;
        MPI_Send(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
      }
      recv_thread_dst.join();
    } //maybe??*******************************************************************dense model finished
    for (int s_i = 0; s_i < sockets; s_i++)
    {
      for (VertexId p_v_i = 0; p_v_i < compressed_outgoing_adj_vertices[s_i]; p_v_i++)
      {
        VertexId v_i = compressed_outgoing_adj_index[s_i][p_v_i].vertex;
        outgoing_adj_index[s_i][v_i] = compressed_outgoing_adj_index[s_i][p_v_i].index;
        outgoing_adj_index[s_i][v_i + 1] = compressed_outgoing_adj_index[s_i][p_v_i + 1].index;
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    EdgeId recv_incoming_edges = 0;
    incoming_edges = new EdgeId[sockets];
    incoming_adj_index = new EdgeId *[sockets];
    incoming_adj_list = new AdjUnit<EdgeData> *[sockets];
    incoming_adj_bitmap = new Bitmap *[sockets];
    for (int s_i = 0; s_i < sockets; s_i++)
    {
      incoming_adj_bitmap[s_i] = new Bitmap(vertices);
      incoming_adj_bitmap[s_i]->clear();
      incoming_adj_index[s_i] = (EdgeId *)numa_alloc_onnode(sizeof(EdgeId) * (vertices + 1), s_i);
    }
    {
      std::thread recv_thread_src([&]() {
        int finished_count = 0;
        MPI_Status recv_status;
        while (finished_count < partitions)
        {
          MPI_Probe(MPI_ANY_SOURCE, ShuffleGraph, MPI_COMM_WORLD, &recv_status);
          int i = recv_status.MPI_SOURCE;
          assert(recv_status.MPI_TAG == ShuffleGraph && i >= 0 && i < partitions);
          int recv_bytes;
          MPI_Get_count(&recv_status, MPI_CHAR, &recv_bytes);
          if (recv_bytes == 1)
          {
            finished_count += 1;
            char c;
            MPI_Recv(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            continue;
          }
          assert(recv_bytes % edge_unit_size == 0);
          int recv_edges = recv_bytes / edge_unit_size;
          MPI_Recv(recv_buffer, edge_unit_size * recv_edges, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          // #pragma omp parallel for
          for (EdgeId e_i = 0; e_i < recv_edges; e_i++)
          {
            VertexId src = recv_buffer[e_i].src;
            VertexId dst = recv_buffer[e_i].dst;
            assert(src >= partition_offset[partition_id] && src < partition_offset[partition_id + 1]);
            int src_part = get_local_partition_id(src);
            if (!incoming_adj_bitmap[src_part]->get_bit(dst))
            {
              incoming_adj_bitmap[src_part]->set_bit(dst);
              incoming_adj_index[src_part][dst] = 0;
            }
            __sync_fetch_and_add(&incoming_adj_index[src_part][dst], 1);
          }
          recv_incoming_edges += recv_edges;
        }
      });
      for (int i = 0; i < partitions; i++)
      {
        buffered_edges[i] = 0;
      }
      assert(lseek(fin, read_offset, SEEK_SET) == read_offset);
      read_bytes = 0;
      while (read_bytes < bytes_to_read)
      {
        long curr_read_bytes;
        if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE)
        {
          curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
        }
        else
        {
          curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
        }
        assert(curr_read_bytes >= 0);
        read_bytes += curr_read_bytes;
        EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
        for (EdgeId e_i = 0; e_i < curr_read_edges; e_i++)
        {
          VertexId src = read_edge_buffer[e_i].src;
          int i = get_partition_id(src);
          memcpy(send_buffer[i].data() + edge_unit_size * buffered_edges[i], &read_edge_buffer[e_i], edge_unit_size);
          buffered_edges[i] += 1;
          if (buffered_edges[i] == CHUNKSIZE)
          {
            MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
            buffered_edges[i] = 0;
          }
        }
      }
      for (int i = 0; i < partitions; i++)
      {
        if (buffered_edges[i] == 0)
          continue;
        MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
        buffered_edges[i] = 0;
      }
      for (int i = 0; i < partitions; i++)
      {
        char c = 0;
        MPI_Send(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
      }
      recv_thread_src.join(); 
#ifdef PRINT_DEBUG_MESSAGES
      printf("machine(%d) got %lu dense mode edges\n", partition_id, recv_incoming_edges);
#endif
    }
    compressed_incoming_adj_vertices = new VertexId[sockets];
    compressed_incoming_adj_index = new CompressedAdjIndexUnit *[sockets];
    for (int s_i = 0; s_i < sockets; s_i++)
    {
      incoming_edges[s_i] = 0;
      compressed_incoming_adj_vertices[s_i] = 0;
      for (VertexId v_i = 0; v_i < vertices; v_i++)
      {
        if (incoming_adj_bitmap[s_i]->get_bit(v_i))
        {
          incoming_edges[s_i] += incoming_adj_index[s_i][v_i];
          compressed_incoming_adj_vertices[s_i] += 1;
        }
      }
      compressed_incoming_adj_index[s_i] = (CompressedAdjIndexUnit *)numa_alloc_onnode(sizeof(CompressedAdjIndexUnit) * (compressed_incoming_adj_vertices[s_i] + 1), s_i);
      compressed_incoming_adj_index[s_i][0].index = 0;
      EdgeId last_e_i = 0;
      compressed_incoming_adj_vertices[s_i] = 0;
      for (VertexId v_i = 0; v_i < vertices; v_i++)
      {
        if (incoming_adj_bitmap[s_i]->get_bit(v_i))
        {
          incoming_adj_index[s_i][v_i] = last_e_i + incoming_adj_index[s_i][v_i];
          last_e_i = incoming_adj_index[s_i][v_i];
          compressed_incoming_adj_index[s_i][compressed_incoming_adj_vertices[s_i]].vertex = v_i;
          compressed_incoming_adj_vertices[s_i] += 1;
          compressed_incoming_adj_index[s_i][compressed_incoming_adj_vertices[s_i]].index = last_e_i;
        }
      }
      for (VertexId p_v_i = 0; p_v_i < compressed_incoming_adj_vertices[s_i]; p_v_i++)
      {
        VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
        incoming_adj_index[s_i][v_i] = compressed_incoming_adj_index[s_i][p_v_i].index;
        incoming_adj_index[s_i][v_i + 1] = compressed_incoming_adj_index[s_i][p_v_i + 1].index;
      }
#ifdef PRINT_DEBUG_MESSAGES
      printf("part(%d) E_%d has %lu dense mode edges\n", partition_id, s_i, incoming_edges[s_i]);
#endif
      incoming_adj_list[s_i] = (AdjUnit<EdgeData> *)numa_alloc_onnode(unit_size * incoming_edges[s_i], s_i);
    }
    {
      std::thread recv_thread_src([&]() {
        int finished_count = 0;
        MPI_Status recv_status;
        while (finished_count < partitions)
        {
          MPI_Probe(MPI_ANY_SOURCE, ShuffleGraph, MPI_COMM_WORLD, &recv_status);
          int i = recv_status.MPI_SOURCE;
          assert(recv_status.MPI_TAG == ShuffleGraph && i >= 0 && i < partitions);
          int recv_bytes;
          MPI_Get_count(&recv_status, MPI_CHAR, &recv_bytes);
          if (recv_bytes == 1)
          {
            finished_count += 1;
            char c;
            MPI_Recv(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            continue;
          }
          assert(recv_bytes % edge_unit_size == 0);
          int recv_edges = recv_bytes / edge_unit_size;
          MPI_Recv(recv_buffer, edge_unit_size * recv_edges, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#pragma omp parallel for
          for (EdgeId e_i = 0; e_i < recv_edges; e_i++)
          {
            VertexId src = recv_buffer[e_i].src;
            VertexId dst = recv_buffer[e_i].dst;
            assert(src >= partition_offset[partition_id] && src < partition_offset[partition_id + 1]);
            int src_part = get_local_partition_id(src);
            EdgeId pos = __sync_fetch_and_add(&incoming_adj_index[src_part][dst], 1);
            incoming_adj_list[src_part][pos].neighbour = src;
            if (!std::is_same<EdgeData, Empty>::value)
            {
              incoming_adj_list[src_part][pos].edge_data = recv_buffer[e_i].edge_data;
            }
          }
        }
      });
      for (int i = 0; i < partitions; i++)
      {
        buffered_edges[i] = 0;
      }
      assert(lseek(fin, read_offset, SEEK_SET) == read_offset);
      read_bytes = 0;
      while (read_bytes < bytes_to_read)
      {
        long curr_read_bytes;
        if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE)
        {
          curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
        }
        else
        {
          curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
        }
        assert(curr_read_bytes >= 0);
        read_bytes += curr_read_bytes;
        EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
        for (EdgeId e_i = 0; e_i < curr_read_edges; e_i++)
        {
          VertexId src = read_edge_buffer[e_i].src;
          int i = get_partition_id(src);
          memcpy(send_buffer[i].data() + edge_unit_size * buffered_edges[i], &read_edge_buffer[e_i], edge_unit_size);
          buffered_edges[i] += 1;
          if (buffered_edges[i] == CHUNKSIZE)
          {
            MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
            buffered_edges[i] = 0;
          }
        }
      }
      for (int i = 0; i < partitions; i++)
      {
        if (buffered_edges[i] == 0)
          continue;
        MPI_Send(send_buffer[i].data(), edge_unit_size * buffered_edges[i], MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
        buffered_edges[i] = 0;
      }
      for (int i = 0; i < partitions; i++)
      {
        char c = 0;
        MPI_Send(&c, 1, MPI_CHAR, i, ShuffleGraph, MPI_COMM_WORLD);
      }
      recv_thread_src.join();
    }
    for (int s_i = 0; s_i < sockets; s_i++)
    {
      for (VertexId p_v_i = 0; p_v_i < compressed_incoming_adj_vertices[s_i]; p_v_i++)
      {
        VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
        incoming_adj_index[s_i][v_i] = compressed_incoming_adj_index[s_i][p_v_i].index;
        incoming_adj_index[s_i][v_i + 1] = compressed_incoming_adj_index[s_i][p_v_i + 1].index;
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    delete[] buffered_edges;
    delete[] send_buffer;
    delete[] read_edge_buffer;
    delete[] recv_buffer;
    close(fin);

    transpose();
    tune_chunks();
    transpose();
    tune_chunks();

    prep_time += MPI_Wtime();

#ifdef PRINT_DEBUG_MESSAGES
    if (partition_id == 0)
    {
      printf("preprocessing cost: %.2lf (s)\n", prep_time);
    }
#endif
  }

  void tune_chunks()
  {
    tuned_chunks_dense = new ThreadState *[partitions];
    int current_send_part_id = partition_id;
    for (int step = 0; step < partitions; step++)
    {
      current_send_part_id = (current_send_part_id + 1) % partitions;
      int i = current_send_part_id;
      tuned_chunks_dense[i] = new ThreadState[threads];
      EdgeId remained_edges;
      int remained_partitions;
      VertexId last_p_v_i;
      VertexId end_p_v_i;
      for (int t_i = 0; t_i < threads; t_i++)
      {
        tuned_chunks_dense[i][t_i].status = WORKING;
        int s_i = get_socket_id(t_i);
        int s_j = get_socket_offset(t_i);
        if (s_j == 0)
        {
          VertexId p_v_i = 0;
          while (p_v_i < compressed_incoming_adj_vertices[s_i])
          {
            VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
            if (v_i >= partition_offset[i])
            {
              break;
            }
            p_v_i++;
          }
          last_p_v_i = p_v_i;
          while (p_v_i < compressed_incoming_adj_vertices[s_i])
          {
            VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
            if (v_i >= partition_offset[i + 1])
            {
              break;
            }
            p_v_i++;
          }
          end_p_v_i = p_v_i;
          remained_edges = 0;
          for (VertexId p_v_i = last_p_v_i; p_v_i < end_p_v_i; p_v_i++)
          {
            remained_edges += compressed_incoming_adj_index[s_i][p_v_i + 1].index - compressed_incoming_adj_index[s_i][p_v_i].index;
            remained_edges += alpha;
          }
        }
        tuned_chunks_dense[i][t_i].curr = last_p_v_i;
        tuned_chunks_dense[i][t_i].end = last_p_v_i;
        remained_partitions = threads_per_socket - s_j;
        EdgeId expected_chunk_size = remained_edges / remained_partitions;
        if (remained_partitions == 1)
        {
          tuned_chunks_dense[i][t_i].end = end_p_v_i;
        }
        else
        {
          EdgeId got_edges = 0;
          for (VertexId p_v_i = last_p_v_i; p_v_i < end_p_v_i; p_v_i++)
          {
            got_edges += compressed_incoming_adj_index[s_i][p_v_i + 1].index - compressed_incoming_adj_index[s_i][p_v_i].index + alpha;
            if (got_edges >= expected_chunk_size)
            {
              tuned_chunks_dense[i][t_i].end = p_v_i;
              last_p_v_i = tuned_chunks_dense[i][t_i].end;
              break;
            }
          }
          got_edges = 0;
          for (VertexId p_v_i = tuned_chunks_dense[i][t_i].curr; p_v_i < tuned_chunks_dense[i][t_i].end; p_v_i++)
          {
            got_edges += compressed_incoming_adj_index[s_i][p_v_i + 1].index - compressed_incoming_adj_index[s_i][p_v_i].index + alpha;
          }
          remained_edges -= got_edges;
        }
      }
    }
  }

  void tune_chunks_backward()
  {
    tuned_chunks_dense_backward = new ThreadState *[partitions];
    int current_send_part_id = partition_id;
    for (int step = 0; step < partitions; step++)
    {
      current_send_part_id = (current_send_part_id + 1) % partitions;
      int i = current_send_part_id;
      tuned_chunks_dense_backward[i] = new ThreadState[threads];
      EdgeId remained_edges;
      int remained_partitions;
      VertexId last_p_v_i;
      VertexId end_p_v_i;
      for (int t_i = 0; t_i < threads; t_i++)
      {
        tuned_chunks_dense_backward[i][t_i].status = WORKING;
        int s_i = get_socket_id(t_i);
        int s_j = get_socket_offset(t_i);
        if (s_j == 0)
        {
          VertexId p_v_i = 0;
          while (p_v_i < compressed_incoming_adj_vertices_backward[s_i])
          {
            VertexId v_i = compressed_incoming_adj_index_backward[s_i][p_v_i].vertex;
            if (v_i >= partition_offset[i])
            {
              break;
            }
            p_v_i++;
          }
          last_p_v_i = p_v_i;
          while (p_v_i < compressed_incoming_adj_vertices_backward[s_i])
          {
            VertexId v_i = compressed_incoming_adj_index_backward[s_i][p_v_i].vertex;
            if (v_i >= partition_offset[i + 1])
            {
              break;
            }
            p_v_i++;
          }
          end_p_v_i = p_v_i;
          remained_edges = 0;
          for (VertexId p_v_i = last_p_v_i; p_v_i < end_p_v_i; p_v_i++)
          {
            remained_edges += compressed_incoming_adj_index_backward[s_i][p_v_i + 1].index - compressed_incoming_adj_index_backward[s_i][p_v_i].index;
            remained_edges += alpha;
          }
        }
        tuned_chunks_dense_backward[i][t_i].curr = last_p_v_i;
        tuned_chunks_dense_backward[i][t_i].end = last_p_v_i;
        remained_partitions = threads_per_socket - s_j;
        EdgeId expected_chunk_size = remained_edges / remained_partitions;
        if (remained_partitions == 1)
        {
          tuned_chunks_dense_backward[i][t_i].end = end_p_v_i;
        }
        else
        {
          EdgeId got_edges = 0;
          for (VertexId p_v_i = last_p_v_i; p_v_i < end_p_v_i; p_v_i++)
          {
            got_edges += compressed_incoming_adj_index_backward[s_i][p_v_i + 1].index - compressed_incoming_adj_index_backward[s_i][p_v_i].index + alpha;
            if (got_edges >= expected_chunk_size)
            {
              tuned_chunks_dense_backward[i][t_i].end = p_v_i;
              last_p_v_i = tuned_chunks_dense_backward[i][t_i].end;
              break;
            }
          }
          got_edges = 0;
          for (VertexId p_v_i = tuned_chunks_dense_backward[i][t_i].curr; p_v_i < tuned_chunks_dense_backward[i][t_i].end; p_v_i++)
          {
            got_edges += compressed_incoming_adj_index_backward[s_i][p_v_i + 1].index - compressed_incoming_adj_index_backward[s_i][p_v_i].index + alpha;
          }
          remained_edges -= got_edges;
        }
      }
    }
  }

  // process vertices
  template <typename R>
  R process_vertices(std::function<R(VertexId)> process, Bitmap *active)
  {
    double stream_time = 0;
    stream_time -= MPI_Wtime();

    R reducer = 0;
    size_t basic_chunk = 64;
    for (int t_i = 0; t_i < threads; t_i++)
    {
      int s_i = get_socket_id(t_i);
      //   printf("p_v %d \n",t_i);
      int s_j = get_socket_offset(t_i);
      VertexId partition_size = local_partition_offset[s_i + 1] - local_partition_offset[s_i];
      thread_state[t_i]->curr = local_partition_offset[s_i] + partition_size / threads_per_socket / basic_chunk * basic_chunk * s_j;
      thread_state[t_i]->end = local_partition_offset[s_i] + partition_size / threads_per_socket / basic_chunk * basic_chunk * (s_j + 1);

      if (s_j == threads_per_socket - 1)
      {
        thread_state[t_i]->end = local_partition_offset[s_i + 1];
      }
      thread_state[t_i]->status = WORKING;
      //       if(s_i==0){
      //          printf("td %d inside %d %d\n",t_i,thread_state[t_i]->curr,thread_state[t_i]->end);
      //      }
    } //init all states.
#pragma omp parallel reduction(+ \
                               : reducer)
    {
      R local_reducer = 0;
      int thread_id = omp_get_thread_num();
      while (true)
      {
        VertexId v_i = __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
        if (v_i >= thread_state[thread_id]->end)
          break;
        unsigned long word = active->data[WORD_OFFSET(v_i)];
        while (word != 0)
        {
          if (word & 1)
          {
            local_reducer += process(v_i);
          }
          v_i++;
          word = word >> 1;
        }
      }
      thread_state[thread_id]->status = STEALING;
      for (int t_offset = 1; t_offset < threads; t_offset++)
      {
        int t_i = (thread_id + t_offset) % threads;
        while (thread_state[t_i]->status != STEALING)
        {
          VertexId v_i = __sync_fetch_and_add(&thread_state[t_i]->curr, basic_chunk);
          if (v_i >= thread_state[t_i]->end)
            continue;
          unsigned long word = active->data[WORD_OFFSET(v_i)];
          while (word != 0)
          {
            if (word & 1)
            {
              local_reducer += process(v_i);
            }
            v_i++;
            word = word >> 1;
          }
        }
      }
       reducer += local_reducer;
    }
    R global_reducer;
        MPI_Datatype dt = get_mpi_data_type<R>();
        MPI_Allreduce(&reducer, &global_reducer, 1, dt, MPI_SUM, MPI_COMM_WORLD);
    stream_time += MPI_Wtime();
#ifdef PRINT_DEBUG_MESSAGES
    if (partition_id == 0)
    {
      printf("process_vertices took %lf (s)\n", stream_time);
    }
#endif
    return global_reducer;
  }

  template <typename M>
  inline size_t sizeofM(int f_size)
  {
    return sizeof(VertexId) + sizeof(M) * f_size;
  }

  template <typename M>
  void flush_local_send_buffer(int t_i)
  {
    int s_i = get_socket_id(t_i);
    int pos = __sync_fetch_and_add(&send_buffer[current_send_part_id][s_i]->count, local_send_buffer[t_i]->count);
    memcpy(send_buffer[current_send_part_id][s_i]->data + sizeof(MsgUnit<M>) * pos,
           local_send_buffer[t_i]->data,
           sizeof(MsgUnit<M>) * local_send_buffer[t_i]->count);
    local_send_buffer[t_i]->count = 0;
  }

  void flush_local_send_buffer_buffer(int t_i, int f_size)
  {
    int s_i = get_socket_id(t_i);
    int pos = __sync_fetch_and_add(&send_buffer[current_send_part_id][s_i]->count, local_send_buffer[t_i]->count);
    if (local_send_buffer[t_i]->count != 0)
      memcpy(send_buffer[current_send_part_id][s_i]->data + (sizeofM<float>(f_size)) * pos,
             local_send_buffer[t_i]->data,
             (sizeofM<float>(f_size)) * local_send_buffer[t_i]->count);
    local_send_buffer[t_i]->count = 0;
  }

  void flush_data_to_send_buffer_buffer_lock_free_init(VertexId message_count_partition)
  {
    int pos = __sync_fetch_and_add(&send_buffer[current_send_part_id][0]->count, message_count_partition);
    //if (pos == 0)
    //  printf("send buffer is NULL(%d)\n", pos);
  }

  void flush_data_to_send_buffer_buffer_lock_free_write(int t_i, int f_size, VertexId key, float *value, VertexId message_write_offset_key)
  {
    int s_i = get_socket_id(t_i);
    //int pos = __sync_fetch_and_add(&send_buffer[current_send_part_id][s_i]->count, local_send_buffer[t_i]->count);
    int pos = message_write_offset_key; // if (local_send_buffer[t_i]->count != 0)
    if (pos < 0 || pos >= (partition_offset[current_send_part_id + 1] - partition_offset[current_send_part_id]))
      printf("something wrong %d %d\n", key, current_send_part_id);
    // if (pos < partition_offset[current_send_part_id + 1] - partition_offset[current_send_part_id])
    //printf("POSITION %d %d %d %d\n", pos, (partition_offset[current_send_part_id + 1] - partition_offset[current_send_part_id]), current_send_part_id, partition_id);
    memcpy(send_buffer[current_send_part_id][s_i]->data + (sizeofM<float>(f_size)) * pos, &key, sizeof(VertexId));
    memcpy(send_buffer[current_send_part_id][s_i]->data + (sizeofM<float>(f_size)) * pos + sizeof(VertexId),
           value,
           f_size * sizeof(float));
  }

  // emit a message to a vertex's master (dense) / mirror (sparse)

  template <typename t_v>
  void emit_buffer(VertexId vtx, t_v *buffer, int f_size)
  {
    int t_i = omp_get_thread_num();
    char *s_buffer = NULL;
        s_buffer = (char *)local_send_buffer[t_i]->data;
    //printf("sizeofM<float>(f_size)%d %d %d %d\n",sizeofM<float>(f_size),local_send_buffer_limit,local_send_buffer[t_i]->count,s_buffer!=NULL);
    
    memcpy(s_buffer + local_send_buffer[t_i]->count * sizeofM<float>(f_size), &vtx, sizeof(VertexId));
    memcpy(s_buffer + local_send_buffer[t_i]->count * sizeofM<float>(f_size) + sizeof(VertexId), buffer, sizeof(float) * f_size);
    local_send_buffer[t_i]->count += 1;

    if (local_send_buffer[t_i]->count == local_send_buffer_limit)
    {
      flush_local_send_buffer_buffer(t_i, f_size);
      //flush_local_send_buffer<M>(t_i);
    }
  }

  template <typename M>
  void emit(VertexId vtx, M msg)
  {
    int t_i = omp_get_thread_num();
    MsgUnit<M> *buffer = (MsgUnit<M> *)local_send_buffer[t_i]->data;
    buffer[local_send_buffer[t_i]->count].vertex = vtx;
    buffer[local_send_buffer[t_i]->count].msg_data = msg;
    local_send_buffer[t_i]->count += 1;
    if (local_send_buffer[t_i]->count == local_send_buffer_limit)
    {
      flush_local_send_buffer<M>(t_i);
    }
  }


  
  
  
  // process edges
  template <typename R, typename M>
  R process_edges_forward_decoupled_dynamic_length(std::function<void(VertexId)> sparse_signal, 
                                std::function<void(VertexId, CSC_segment_pinned* ,char* ,std::vector<VertexId>& ,VertexId)> sparse_slot,
                                std::vector<CSC_segment_pinned *> &graph_partitions,
                                int feature_size,
                                Bitmap *active, 
                                Bitmap *dense_selective = nullptr)
  {
    omp_set_num_threads(threads);
    double stream_time = 0;
    stream_time -= MPI_Wtime();
    NtsComm->init_layer_all(feature_size,Master2Mirror,CPU_T);
          NtsComm->run_all_master_to_mirror_no_wait();
    R reducer = 0;
    
    size_t basic_chunk = 64;
    {
      current_send_part_id = partition_id;
      NtsComm->set_current_send_partition(current_send_part_id);
#pragma omp parallel for
      for (VertexId begin_v_i = partition_offset[partition_id]; begin_v_i < partition_offset[partition_id + 1]; begin_v_i += basic_chunk)
      {
        VertexId v_i = begin_v_i;
        unsigned long word = active->data[WORD_OFFSET(v_i)];
        while (word != 0)
        {
          if (word & 1)
          {
            sparse_signal(v_i);
          }
          v_i++;
          word = word >> 1;
        }
      }
      for(int step=0;step<partitions;step++){
          int trigger_partition=(partition_id-step+partitions)%partitions;
          NtsComm->trigger_one_partition(trigger_partition,trigger_partition==current_send_part_id);
      }
     
      for (int step = 0; step < partitions; step++)
      {
        int i = -1;
        MessageBuffer **used_buffer;      
        used_buffer=NtsComm->recv_one_partition(i,step);
        
        cpu_recv_message_index.resize(partition_offset[i+1]-partition_offset[i]);
        memset(cpu_recv_message_index.data(),0,sizeof(VertexId)*cpu_recv_message_index.size());
        for (int s_i = 0; s_i < sockets; s_i++)
        {
          char *buffer = used_buffer[s_i]->data;
          size_t buffer_size = used_buffer[s_i]->count;
          for (int t_i = 0; t_i < threads; t_i++)
          {
            // int s_i = get_socket_id(t_i);
            int s_j = get_socket_offset(t_i);
            VertexId partition_size = buffer_size;
            thread_state[t_i]->curr = partition_size / threads_per_socket / basic_chunk * basic_chunk * s_j;
            thread_state[t_i]->end = partition_size / threads_per_socket / basic_chunk * basic_chunk * (s_j + 1);
            if (s_j == threads_per_socket - 1)
            {
              thread_state[t_i]->end = buffer_size;
            }
            thread_state[t_i]->status = WORKING;
          }
#pragma omp parallel reduction(+ \
                               : reducer)
          {
            R local_reducer = 0;
            int thread_id = omp_get_thread_num();
            int s_i = get_socket_id(thread_id);
            while (true)
            {
              VertexId b_i = __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
              if (b_i >= thread_state[thread_id]->end)
                break;
              VertexId begin_b_i = b_i;
              VertexId end_b_i = b_i + basic_chunk;
              if (end_b_i > thread_state[thread_id]->end)
              {
                end_b_i = thread_state[thread_id]->end;
              }
              for (b_i = begin_b_i; b_i < end_b_i; b_i++)
              {
                long index= (long)b_i*sizeofM<M>(feature_size);
                VertexId v_i = *((VertexId*)(buffer+index));
                M* msg_data = (M*)(buffer+index+sizeof(VertexId));
                if (outgoing_adj_bitmap[s_i]->get_bit(v_i))
                {
                    VertexId v_trans=v_i-partition_offset[i];
                    cpu_recv_message_index[v_trans]=b_i;
                }
              }
            }
            reducer += local_reducer;
          }
#pragma omp parallel for     
          for (VertexId begin_v_i = partition_offset[partition_id]; begin_v_i < partition_offset[partition_id + 1]; begin_v_i += basic_chunk){
                VertexId v_i = begin_v_i;
                unsigned long word = active->data[WORD_OFFSET(v_i)];
                while (word != 0){
                    if (word & 1){
                    sparse_slot(v_i,graph_partitions[i],buffer,cpu_recv_message_index,i);
                    }
                    v_i++;
                    word = word >> 1;
                }
          }          
          
        }
      }
      NtsComm->release_communicator();
    }

    R global_reducer;
    MPI_Datatype dt = get_mpi_data_type<R>();
    MPI_Allreduce(&reducer, &global_reducer, 1, dt, MPI_SUM, MPI_COMM_WORLD);
    stream_time += MPI_Wtime();
#ifdef PRINT_DEBUG_MESSAGES
    if (partition_id == 0)
    {
      printf("process_edges took %lf (s)\n", stream_time);
    }
#endif
    return global_reducer;
  }
  
  
  
    // process edges
  template <typename R, typename M>
  R process_edges_forward_decoupled(std::function<void(VertexId,int)> sparse_signal, 
                                std::function<void(VertexId, CSC_segment_pinned* ,char* ,std::vector<VertexId>& ,VertexId)> sparse_slot,
                                std::vector<CSC_segment_pinned *> &graph_partitions,
                                int feature_size,
                                Bitmap *active, 
                                Bitmap *dense_selective = nullptr)
  {
    omp_set_num_threads(threads);
    double stream_time = 0;
    stream_time -= MPI_Wtime();
    NtsComm->init_layer_all(feature_size,Master2Mirror,CPU_T);
    //printf("call lock_free forward %d\n",rtminfo->lock_free);
    if(rtminfo->lock_free){
        NtsComm->run_all_master_to_mirror_lock_free_no_wait();
    }else{
        NtsComm->run_all_master_to_mirror_no_wait();
    }
    R reducer = 0;
    
    size_t basic_chunk = 64;
    {
        
        if(rtminfo->lock_free){
            for(int step=0;step<partitions;step++){
                int trigger_partition=(partition_id-step+partitions)%partitions;
                current_send_part_id = trigger_partition;
                NtsComm->set_current_send_partition(current_send_part_id);
#pragma omp parallel for
                for (VertexId begin_v_i = partition_offset[partition_id]; begin_v_i < partition_offset[partition_id + 1]; begin_v_i += basic_chunk)
                {
                    VertexId v_i = begin_v_i;
                    unsigned long word = active->data[WORD_OFFSET(v_i)];
                    while (word != 0)
                    {
                        if (word & 1)
                        {
                            sparse_signal(v_i,current_send_part_id);
                        }
                        v_i++;
                        word = word >> 1;
                    }
                }
                NtsComm->trigger_one_partition(trigger_partition,trigger_partition==current_send_part_id);
            }
        }else{
            current_send_part_id = partition_id;
            NtsComm->set_current_send_partition(current_send_part_id);
#pragma omp parallel for
            for (VertexId begin_v_i = partition_offset[partition_id]; begin_v_i < partition_offset[partition_id + 1]; begin_v_i += basic_chunk)
            {
                VertexId v_i = begin_v_i;
                unsigned long word = active->data[WORD_OFFSET(v_i)];
                while (word != 0)
                {
                    if (word & 1)
                    {
                        sparse_signal(v_i,-1);
                    }
                    v_i++;
                    word = word >> 1;
                }
            }
            for(int step=0;step<partitions;step++){
                int trigger_partition=(partition_id-step+partitions)%partitions;
                NtsComm->trigger_one_partition(trigger_partition,trigger_partition==current_send_part_id);
            }
        }
     
      for (int step = 0; step < partitions; step++)
      {
        int i = -1;
        MessageBuffer **used_buffer;      
        used_buffer=NtsComm->recv_one_partition(i,step);
        
        cpu_recv_message_index.resize(partition_offset[i+1]-partition_offset[i]);
        memset(cpu_recv_message_index.data(),0,sizeof(VertexId)*cpu_recv_message_index.size());
        for (int s_i = 0; s_i < sockets; s_i++)
        {
          char *buffer = used_buffer[s_i]->data;
          size_t buffer_size = used_buffer[s_i]->count;
          for (int t_i = 0; t_i < threads; t_i++)
          {
            // int s_i = get_socket_id(t_i);
            int s_j = get_socket_offset(t_i);
            VertexId partition_size = buffer_size;
            thread_state[t_i]->curr = partition_size / threads_per_socket / basic_chunk * basic_chunk * s_j;
            thread_state[t_i]->end = partition_size / threads_per_socket / basic_chunk * basic_chunk * (s_j + 1);
            if (s_j == threads_per_socket - 1)
            {
              thread_state[t_i]->end = buffer_size;
            }
            thread_state[t_i]->status = WORKING;
          }
#pragma omp parallel reduction(+ \
                               : reducer)
          {
            R local_reducer = 0;
            int thread_id = omp_get_thread_num();
            int s_i = get_socket_id(thread_id);
            while (true)
            {
              VertexId b_i = __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
              if (b_i >= thread_state[thread_id]->end)
                break;
              VertexId begin_b_i = b_i;
              VertexId end_b_i = b_i + basic_chunk;
              if (end_b_i > thread_state[thread_id]->end)
              {
                end_b_i = thread_state[thread_id]->end;
              }
              for (b_i = begin_b_i; b_i < end_b_i; b_i++)
              {
                long index= (long)b_i*sizeofM<M>(feature_size);
                VertexId v_i = *((VertexId*)(buffer+index));
                M* msg_data = (M*)(buffer+index+sizeof(VertexId));
                if (outgoing_adj_bitmap[s_i]->get_bit(v_i))
                {
                    VertexId v_trans=v_i-partition_offset[i];
                    cpu_recv_message_index[v_trans]=b_i;
                }
              }
            }
            reducer += local_reducer;
          }
#pragma omp parallel for     
          for (VertexId begin_v_i = partition_offset[partition_id]; begin_v_i < partition_offset[partition_id + 1]; begin_v_i += basic_chunk){
                VertexId v_i = begin_v_i;
                unsigned long word = active->data[WORD_OFFSET(v_i)];
                while (word != 0){
                    if (word & 1){
                    sparse_slot(v_i,graph_partitions[i],buffer,cpu_recv_message_index,i);
                    }
                    v_i++;
                    word = word >> 1;
                }
          }          
          
        }
      }
      NtsComm->release_communicator();
    }

    R global_reducer;
    MPI_Datatype dt = get_mpi_data_type<R>();
    MPI_Allreduce(&reducer, &global_reducer, 1, dt, MPI_SUM, MPI_COMM_WORLD);
    stream_time += MPI_Wtime();
#ifdef PRINT_DEBUG_MESSAGES
    if (partition_id == 0)
    {
      printf("process_edges took %lf (s)\n", stream_time);
    }
#endif
    return global_reducer;
  }
  

        template <typename R, typename M>
  R process_edges_backward_decoupled(
      std::function<void(VertexId, VertexAdjList<EdgeData>,VertexId,VertexId)> dense_signal,
      std::function<R(VertexId, M*)> dense_slot,
      int feature_size,
      Bitmap *active,
      Bitmap *dense_selective = nullptr)
  {
    omp_set_num_threads(threads);
    double stream_time = 0;
    stream_time -= MPI_Wtime();

    NtsComm->init_layer_all(feature_size,Mirror2Master,CPU_T);
    NtsComm->run_all_mirror_to_master();
    
    R reducer = 0;
    size_t basic_chunk = 64;
    {
        
      current_send_part_id = partition_id;
      for (int step = 0; step < partitions; step++)
      {
        current_send_part_id = (current_send_part_id + 1) % partitions;
        int i = current_send_part_id;
        NtsComm->set_current_send_partition(i);
        
        for (int t_i = 0; t_i < threads; t_i++)
        {
          *thread_state[t_i] = tuned_chunks_dense_backward[i][t_i];
        }
#pragma omp parallel
        {
          int thread_id = omp_get_thread_num();
//          printf("PRE threadsId %d\n",thread_id);
          int s_i = get_socket_id(thread_id);
          VertexId final_p_v_i = thread_state[thread_id]->end;
          while (true)
          {
            VertexId begin_p_v_i = __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
            if (begin_p_v_i >= final_p_v_i)
              break;
            VertexId end_p_v_i = begin_p_v_i + basic_chunk;
            if (end_p_v_i > final_p_v_i)
            {
              end_p_v_i = final_p_v_i;
            }
            for (VertexId p_v_i = begin_p_v_i; p_v_i < end_p_v_i; p_v_i++)
            {
              VertexId v_i = compressed_incoming_adj_index_backward[s_i][p_v_i].vertex;
              
              dense_signal(v_i, VertexAdjList<EdgeData>(incoming_adj_list_backward[s_i] +
                      compressed_incoming_adj_index_backward[s_i][p_v_i].index, incoming_adj_list_backward[s_i]
                      + compressed_incoming_adj_index_backward[s_i][p_v_i + 1].index),thread_id,i);
            }
          }
          thread_state[thread_id]->status = STEALING;
          for (int t_offset = 1; t_offset < threads; t_offset++)
          {
            int t_i = (thread_id + t_offset) % threads;
            int s_i = get_socket_id(t_i);
            while (thread_state[t_i]->status != STEALING)
            {
              VertexId begin_p_v_i = __sync_fetch_and_add(&thread_state[t_i]->curr, basic_chunk);
              if (begin_p_v_i >= thread_state[t_i]->end)
                break;
              VertexId end_p_v_i = begin_p_v_i + basic_chunk;
              if (end_p_v_i > thread_state[t_i]->end)
              {
                end_p_v_i = thread_state[t_i]->end;
              }
              for (VertexId p_v_i = begin_p_v_i; p_v_i < end_p_v_i; p_v_i++)
              {
                VertexId v_i = compressed_incoming_adj_index_backward[s_i][p_v_i].vertex;
                dense_signal(v_i, VertexAdjList<EdgeData>(incoming_adj_list_backward[s_i] + 
                        compressed_incoming_adj_index_backward[s_i][p_v_i].index, incoming_adj_list_backward[s_i] + 
                        compressed_incoming_adj_index_backward[s_i][p_v_i + 1].index),thread_id,i);
              }
            }
          }
        }
//        NtsComm->achieve_local_message(i);
//        NtsComm->partition_is_ready_for_send(i);
        NtsComm->trigger_one_partition(i,true);
      }
      
      for (int step = 0; step < partitions; step++)
      {
        int i = -1;
        MessageBuffer **used_buffer;
        used_buffer=NtsComm->recv_one_partition(i,step);

        for (int t_i = 0; t_i < threads; t_i++)
        {
          int s_i = get_socket_id(t_i);
          int s_j = get_socket_offset(t_i);
          VertexId partition_size = used_buffer[s_i]->count;
          //printf("DEBUG %d\n",partition_size);
          thread_state[t_i]->curr = partition_size / threads_per_socket / basic_chunk * basic_chunk * s_j;
          thread_state[t_i]->end = partition_size / threads_per_socket / basic_chunk * basic_chunk * (s_j + 1);
          if (s_j == threads_per_socket - 1)
          {
            thread_state[t_i]->end = used_buffer[s_i]->count;
          }
          thread_state[t_i]->status = WORKING;
        }
        
        /*TODO openmp does not launch enough threads,
         * and Gemini dose not provide the stealing method as in dense_signal stage
         * we manually re configure enough threads to fix the bug
         */
        
#pragma omp parallel reduction(+ \
                               : reducer)
        {
          R local_reducer = 0;
          int thread_id = omp_get_thread_num();
          int s_i = get_socket_id(thread_id);
          char* buffer = used_buffer[s_i]->data;
          while (true)
          {
            VertexId b_i = __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
            if (b_i >= thread_state[thread_id]->end)
              break;
            VertexId begin_b_i = b_i;
            VertexId end_b_i = b_i + basic_chunk;
            if (end_b_i > thread_state[thread_id]->end)
            {
              end_b_i = thread_state[thread_id]->end;
            }
            for (b_i = begin_b_i; b_i < end_b_i; b_i++)
            {
              long index= (long)b_i*(sizeof(M)*(feature_size)+sizeof(VertexId));
              VertexId v_i = *((VertexId*)(buffer+index));
              M* msg_data = (M*)(buffer+index+sizeof(VertexId));
              local_reducer += 1.0;
              dense_slot(v_i, msg_data);
            }
          }
          thread_state[thread_id]->status = STEALING;
          reducer += local_reducer;
        }
       
        
      }
      NtsComm->release_communicator();
    }

    R global_reducer;
    MPI_Datatype dt = get_mpi_data_type<R>();
    MPI_Allreduce(&reducer, &global_reducer, 1, dt, MPI_SUM, MPI_COMM_WORLD);
    stream_time += MPI_Wtime();
#ifdef PRINT_DEBUG_MESSAGES
    if (partition_id == 0)
    {
      printf("process_edges took %lf (s)\n", stream_time);
    }
#endif
    return global_reducer;
  }

        
    // process edges
  template <typename R, typename M>
  R get_from_dep_neighbor(std::function<void(VertexId,int)> sparse_signal, 
                                std::function<void(VertexId, M*, VertexId)> sparse_slot,
                                std::vector<CSC_segment_pinned *> &graph_partitions,
                                int feature_size,
                                Bitmap *active, 
                                Bitmap *dense_selective = nullptr)
  {
    omp_set_num_threads(threads);
    double stream_time = 0;
    stream_time -= MPI_Wtime();
    NtsComm->init_layer_all(feature_size,Master2Mirror,CPU_T);
    //printf("call lock_free forward %d\n",rtminfo->lock_free);
    if(rtminfo->lock_free){
        NtsComm->run_all_master_to_mirror_lock_free_no_wait();
    }else{
        NtsComm->run_all_master_to_mirror_no_wait();
    }
    R reducer = 0;
    
    size_t basic_chunk = 64;
    {
        
        if(rtminfo->lock_free){
            for(int step=0;step<partitions;step++){
                int trigger_partition=(partition_id-step+partitions)%partitions;
                current_send_part_id = trigger_partition;
                NtsComm->set_current_send_partition(current_send_part_id);
#pragma omp parallel for
                for (VertexId begin_v_i = partition_offset[partition_id]; begin_v_i < partition_offset[partition_id + 1]; begin_v_i += basic_chunk)
                {
                    VertexId v_i = begin_v_i;
                    unsigned long word = active->data[WORD_OFFSET(v_i)];
                    while (word != 0)
                    {
                        if (word & 1)
                        {
                            sparse_signal(v_i,current_send_part_id);
                        }
                        v_i++;
                        word = word >> 1;
                    }
                }
                NtsComm->trigger_one_partition(trigger_partition,trigger_partition==current_send_part_id);
            }
        }else{
            current_send_part_id = partition_id;
            NtsComm->set_current_send_partition(current_send_part_id);
#pragma omp parallel for
            for (VertexId begin_v_i = partition_offset[partition_id]; begin_v_i < partition_offset[partition_id + 1]; begin_v_i += basic_chunk)
            {
                VertexId v_i = begin_v_i;
                unsigned long word = active->data[WORD_OFFSET(v_i)];
                while (word != 0)
                {
                    if (word & 1)
                    {
                        sparse_signal(v_i,-1);
                    }
                    v_i++;
                    word = word >> 1;
                }
            }
            for(int step=0;step<partitions;step++){
                int trigger_partition=(partition_id-step+partitions)%partitions;
                NtsComm->trigger_one_partition(trigger_partition,trigger_partition==current_send_part_id);
            }
        }
     
      for (int step = 0; step < partitions; step++)
      {
        int i = -1;
        MessageBuffer **used_buffer;      
        used_buffer=NtsComm->recv_one_partition(i,step);
        
        cpu_recv_message_index.resize(partition_offset[i+1]-partition_offset[i]);
        memset(cpu_recv_message_index.data(),0,sizeof(VertexId)*cpu_recv_message_index.size());
        for (int s_i = 0; s_i < sockets; s_i++)
        {
          char *buffer = used_buffer[s_i]->data;
          size_t buffer_size = used_buffer[s_i]->count;
          for (int t_i = 0; t_i < threads; t_i++)
          {
            // int s_i = get_socket_id(t_i);
            int s_j = get_socket_offset(t_i);
            VertexId partition_size = buffer_size;
            thread_state[t_i]->curr = partition_size / threads_per_socket / basic_chunk * basic_chunk * s_j;
            thread_state[t_i]->end = partition_size / threads_per_socket / basic_chunk * basic_chunk * (s_j + 1);
            if (s_j == threads_per_socket - 1)
            {
              thread_state[t_i]->end = buffer_size;
            }
            thread_state[t_i]->status = WORKING;
          }
#pragma omp parallel reduction(+ \
                               : reducer)
          {
            R local_reducer = 0;
            int thread_id = omp_get_thread_num();
            int s_i = get_socket_id(thread_id);
            while (true)
            {
              VertexId b_i = __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
              if (b_i >= thread_state[thread_id]->end)
                break;
              VertexId begin_b_i = b_i;
              VertexId end_b_i = b_i + basic_chunk;
              if (end_b_i > thread_state[thread_id]->end)
              {
                end_b_i = thread_state[thread_id]->end;
              }
              for (b_i = begin_b_i; b_i < end_b_i; b_i++)
              {
                long index= (long)b_i*sizeofM<M>(feature_size);
                VertexId v_i = *((VertexId*)(buffer+index));
                M* msg_data = (M*)(buffer+index+sizeof(VertexId));
                if (outgoing_adj_bitmap[s_i]->get_bit(v_i))
                {
                   sparse_slot(v_i,msg_data,i);
                   // VertexId v_trans=v_i-partition_offset[i];
                   // cpu_recv_message_index[v_trans]=b_i;
                }
              }
            }
            reducer += local_reducer;
          }
//#pragma omp parallel for     
//          for (VertexId begin_v_i = partition_offset[partition_id]; begin_v_i < partition_offset[partition_id + 1]; begin_v_i += basic_chunk){
//                VertexId v_i = begin_v_i;
//                unsigned long word = active->data[WORD_OFFSET(v_i)];
//                while (word != 0){
//                    if (word & 1){
//                    sparse_slot(v_i,graph_partitions[i],buffer,cpu_recv_message_index,i);
//                    }
//                    v_i++;
//                    word = word >> 1;
//                }
//          }          
          
        }
      }
      NtsComm->release_communicator();
    }

    R global_reducer;
    MPI_Datatype dt = get_mpi_data_type<R>();
    MPI_Allreduce(&reducer, &global_reducer, 1, dt, MPI_SUM, MPI_COMM_WORLD);
    stream_time += MPI_Wtime();
#ifdef PRINT_DEBUG_MESSAGES
    if (partition_id == 0)
    {
      printf("process_edges took %lf (s)\n", stream_time);
    }
#endif
    return global_reducer;
  }        
        
#if CUDA_ENABLE       
   template <typename R, typename M>
  R compute_sync_decoupled(NtsVar &input_gpu_or_cpu,
                         std::vector<CSC_segment_pinned *> &graph_partitions,
                         std::function<void(VertexId, VertexAdjList<EdgeData>)> dense_signal,
                         NtsVar& Y,int feature_size)//backward
  {
   
    //int feature_size = gnnctx->layer_size[rtminfo->curr_layer];
    bool process_local = rtminfo->process_local;
    int layer_ = rtminfo->curr_layer;
    bool process_overlap = rtminfo->process_overlap;
    
    NtsComm->init_layer_all(feature_size,Mirror2Master,GPU_T);
    NtsComm->run_all_mirror_to_master();

//    if (partition_id == 0)
//    {
//      printf("ComputeSync:layer(%d).process_local(%d).dimension(%d).reduce_comm(%d).overlap(%d)\n", layer_, process_local ? replication_threshold : -1, feature_size, process_local, process_overlap);
//    }
    double stream_time = 0;
    stream_time -= MPI_Wtime();   
    
    Nts->ZeroVarMem(Y,GPU_T);
    
    size_t basic_chunk = 64;
    {
#ifdef PRINT_DEBUG_MESSAGES
      if (partition_id == 0)
      {
        printf("dense mode\n");
      }
#endif

      if (process_overlap)
      {
        current_send_part_id = partition_id;
            Nts->InitBlockSimple(graph_partitions[(current_send_part_id + 1) % partitions],rtminfo,feature_size, 
                 feature_size, (current_send_part_id + 1) % partitions,layer_);
            Nts->ZeroVarMem(output_gpu_buffered,GPU_T);
            Nts->GatherBySrcFromDst(output_gpu_buffered,input_gpu_or_cpu,input_gpu_or_cpu);
            Nts->SerializeMirrorToMsg(output_cpu_buffer,output_gpu_buffered);
            Nts->DeviceSynchronize();
      }
      else
      {
        for (int step = 0; step < partitions; step++)
        {
            current_send_part_id = (current_send_part_id + 1) % partitions;
            Nts->InitBlockSimple(graph_partitions[(current_send_part_id + 1) % partitions],rtminfo,feature_size, 
                 feature_size, (current_send_part_id + 1) % partitions,layer_);
            Nts->ZeroVarMem(output_gpu_buffered,GPU_T);
            Nts->GatherBySrcFromDst(output_gpu_buffered,input_gpu_or_cpu,input_gpu_or_cpu);
            Nts->SerializeMirrorToMsg(output_cpu_buffer,output_gpu_buffered);
            Nts->DeviceSynchronize();
        }
      }
      
      for (int step = 0; step < partitions; step++)
      {
        current_send_part_id = (current_send_part_id + 1) % partitions;
        int i = current_send_part_id;
        NtsComm->set_current_send_partition(current_send_part_id);
        
        for (int t_i = 0; t_i < threads; t_i++)
        {
          *thread_state[t_i] = tuned_chunks_dense_backward[i][t_i];
        }
        
        rtminfo->device_sync();
        if (current_send_part_id != partition_id&&process_overlap)
        {
            Nts->InitBlockSimple(graph_partitions[(current_send_part_id + 1) % partitions],rtminfo,feature_size, 
                 feature_size, (current_send_part_id + 1) % partitions,layer_);
            Nts->ZeroVarMem(output_gpu_buffered,GPU_T);
            Nts->GatherBySrcFromDst(output_gpu_buffered,input_gpu_or_cpu,input_gpu_or_cpu);
            Nts->SerializeMirrorToMsg(output_cpu_buffer,output_gpu_buffered);
        }

#pragma omp parallel
        {
          int thread_id = omp_get_thread_num();
          int s_i = get_socket_id(thread_id);
          VertexId final_p_v_i = thread_state[thread_id]->end;
          while (true)
          {
            VertexId begin_p_v_i = __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
            if (begin_p_v_i >= final_p_v_i)
              break;
            VertexId end_p_v_i = begin_p_v_i + basic_chunk;
            if (end_p_v_i > final_p_v_i)
            {
              end_p_v_i = final_p_v_i;
            }
            for (VertexId p_v_i = begin_p_v_i; p_v_i < end_p_v_i; p_v_i++)
            {
              VertexId v_i = compressed_incoming_adj_index_backward[s_i][p_v_i].vertex;
              dense_signal(v_i, VertexAdjList<EdgeData>(incoming_adj_list_backward[s_i] +
                      compressed_incoming_adj_index_backward[s_i][p_v_i].index, 
                        incoming_adj_list_backward[s_i] + 
                          compressed_incoming_adj_index[s_i][p_v_i + 1].index));
            }
          }
          thread_state[thread_id]->status = STEALING;
          for (int t_offset = 1; t_offset < threads; t_offset++)
          {
            int t_i = (thread_id + t_offset) % threads;
            int s_i = get_socket_id(t_i);
            while (thread_state[t_i]->status != STEALING)
            {
              VertexId begin_p_v_i = __sync_fetch_and_add(&thread_state[t_i]->curr, basic_chunk);
              if (begin_p_v_i >= thread_state[t_i]->end)
                break;
              VertexId end_p_v_i = begin_p_v_i + basic_chunk;
              if (end_p_v_i > thread_state[t_i]->end)
              {
                end_p_v_i = thread_state[t_i]->end;
              }
              for (VertexId p_v_i = begin_p_v_i; p_v_i < end_p_v_i; p_v_i++)
              {
                VertexId v_i = compressed_incoming_adj_index_backward[s_i][p_v_i].vertex;
                dense_signal(v_i, VertexAdjList<EdgeData>(incoming_adj_list_backward[s_i] + 
                        compressed_incoming_adj_index_backward[s_i][p_v_i].index,
                            incoming_adj_list_backward[s_i] + 
                                compressed_incoming_adj_index_backward[s_i][p_v_i + 1].index));
              }
            }
          }
        }
        NtsComm->trigger_one_partition(i);        
      }
      
      for (int step = 0; step < partitions; step++)
      {
        int i = -1;
        MessageBuffer **used_buffer;
        used_buffer=NtsComm->recv_one_partition(i,step);
        Nts->InitBlockSimple(graph_partitions[i],rtminfo,feature_size, 
                        feature_size, i,layer_);
        for (int s_i = 0; s_i < sockets; s_i++)
        {
          Nts->DeviceSynchronize();
          Nts->AggMsgToMaster(Y, used_buffer[s_i]->data,used_buffer[s_i]->count);
        }
      }

      Nts->DeviceSynchronize(); 
      NtsComm->release_communicator();      
    }

    R global_reducer;
    stream_time += MPI_Wtime();
#ifdef PRINT_DEBUG_MESSAGES
    if (partition_id == 0)
    {
      printf("process_edges took %lf (s)\n", stream_time);
    }
#endif
    return global_reducer;
  } 
  
      // process edges
  template <typename R, typename M>
  R sync_compute_decoupled(NtsVar &input_gpu_or_cpu,
                 std::vector<CSC_segment_pinned *> &graph_partitions,
                 std::function<void(VertexId)> sparse_signal,
                 NtsVar& Y,
                 int feature_size)
  {
    //int feature_size = gnnctx->layer_size[rtminfo->curr_layer];
    bool process_local = rtminfo->process_local;
    int layer_ = rtminfo->curr_layer;
    bool process_overlap = rtminfo->process_overlap;

//    if (partition_id == 0)
//    {
//      printf("SyncComputeDecoupled:layer(%d).process_local(%d).dimension(%d).reduce_comm(%d).overlap(%d)\n", layer_, process_local ? replication_threshold : -1, feature_size, process_local, process_overlap);
//    }
    double stream_time = 0;
    stream_time -= MPI_Wtime();

    NtsComm->init_layer_all(feature_size,Master2Mirror,GPU_T);
    NtsComm->run_all_master_to_mirror_no_wait();
    
    NtsVar mirror_input=Nts->NewLeafTensor({get_max_partition_size(),(feature_size+1)});
    Nts->ZeroVarMem(Y);
    
    {//1-stage 
      current_send_part_id = partition_id;
      NtsComm->set_current_send_partition(current_send_part_id);
#pragma omp parallel for
      for (VertexId begin_v_i = partition_offset[partition_id]; begin_v_i < partition_offset[partition_id + 1]; begin_v_i += 1)
      {
        VertexId v_i = begin_v_i;
            sparse_signal(v_i);
      }
      for(int step=0;step<partitions;step++){
          int trigger_partition=(partition_id-step+partitions)%partitions;
          NtsComm->trigger_one_partition(trigger_partition,trigger_partition==current_send_part_id);
      }     
      //2-stage
      int current_recv_part_id = 0;
      for (int step = 0; step < partitions; step++)
      {
        int i = -1;
        MessageBuffer **used_buffer;
        used_buffer=NtsComm->recv_one_partition(i,step);
        Nts->InitBlockSimple(graph_partitions[i],rtminfo,feature_size, 
                    feature_size, i,layer_);
//         printf("SyncComputeDecoupled\n");
        for (int s_i = 0; s_i < sockets; s_i++)
        {
            int current_recv_part_id = i;
            Nts->DeserializeMsgToMirror(mirror_input,used_buffer[s_i]->data, used_buffer[s_i]->count);
            Nts->GatherByDstFromSrc(Y,mirror_input,mirror_input);
        }
      }

      rtminfo->device_sync();   
      NtsComm->release_communicator();
    }
    stream_time += MPI_Wtime();
    R global_reducer;
#ifdef PRINT_DEBUG_MESSAGES
    if (partition_id == 0)
    {
      printf("process_edges took %lf (s)\n", stream_time);
    }
#endif
    return global_reducer;
  }
  
    template <typename R, typename M>
  R forward_single(NtsVar &input_gpu_or_cpu,
                 std::vector<CSC_segment_pinned *> &graph_partitions,
                 NtsVar& Y, int feature_size)
  {
    int layer_ = rtminfo->curr_layer;

//    if (partition_id == 0)
//    {
//      printf("SyncComputeDecoupled:layer(%d).process_local(%d).dimension(%d).reduce_comm(%d).overlap(%d)\n", layer_, process_local ? replication_threshold : -1, feature_size, process_local, process_overlap);
//    }
    double stream_time = 0;
    stream_time -= MPI_Wtime();

    Nts->ZeroVarMem(Y);
    Nts->InitBlockSimple(graph_partitions[0],rtminfo,feature_size, 
                    feature_size, 0,layer_);
    Nts->GatherByDstFromSrc(Y,input_gpu_or_cpu,input_gpu_or_cpu);

    rtminfo->device_sync();      
    stream_time += MPI_Wtime();
    
    R global_reducer;
    return global_reducer;
  }
  
     template <typename R, typename M>
  R backward_single(NtsVar &input_gpu_or_cpu,
                         std::vector<CSC_segment_pinned *> &graph_partitions,
                         NtsVar& Y,int feature_size)//backward
  {
    int layer_ = rtminfo->curr_layer;
//    if (partition_id == 0)
//    {
//      printf("ComputeSync:layer(%d).process_local(%d).dimension(%d).reduce_comm(%d).overlap(%d)\n", layer_, process_local ? replication_threshold : -1, feature_size, process_local, process_overlap);
//    }
    double stream_time = 0;
    stream_time -= MPI_Wtime();
    Nts->ZeroVarMem(Y,GPU_T);
    Nts->InitBlockSimple(graph_partitions[0],rtminfo,feature_size, 
                 feature_size, 0,layer_);
    Nts->GatherBySrcFromDst(Y,input_gpu_or_cpu,input_gpu_or_cpu);
    Nts->DeviceSynchronize(); 

    R global_reducer;
    stream_time += MPI_Wtime();
    return global_reducer;
  }  
        // process edges
  template <typename R, typename M>
  R forward_single_edge(NtsVar &message,
                 std::vector<CSC_segment_pinned *> &graph_partitions,
                 NtsVar& Y,
                 int feature_size)
  {
    int layer_ = rtminfo->curr_layer;
//    if (partition_id == 0)
//    {
//      printf("SyncComputeDecoupledEdge:layer(%d).process_local(%d).dimension(%d).reduce_comm(%d).overlap(%d)\n", layer_, process_local ? replication_threshold : -1, feature_size, process_local, process_overlap);
//    }
    double stream_time = 0;
    stream_time -= MPI_Wtime();
    Nts->ZeroVarMem(Y);
    int current_recv_part_id = 0;
    for (int step = 0; step < partitions; step++)
    {
        int i=0;
        Nts->InitBlock(graph_partitions[i],
                            rtminfo,
                            feature_size,
                            feature_size,
                            current_recv_part_id,
                            layer_
                            );     
        Nts->GatherByDstFromMessage(Y, message);      
    }
    rtminfo->device_sync();
    stream_time += MPI_Wtime();
    R global_reducer;
    return global_reducer;
  }
 
          template <typename R, typename M>
  R backward_single_edge(NtsVar &input_grad,
                         std::vector<CSC_segment_pinned *> &graph_partitions,
//                       std::function<NtsVar(NtsVar&,NtsVar&,NtsScheduler* nts)> EdgeForward,
//                       std::function<NtsVar(NtsVar&,NtsVar&,NtsScheduler* nts)> EdgeBackward,
                         NtsVar &message_grad,
                         int feature_size)//backward
  {
    int layer_ = rtminfo->curr_layer;
//    if (partition_id == 0)
//    {
//      printf("ComputeSyncEdgeDecoupled:layer(%d).process_local(%d).dimension(%d).reduce_comm(%d).overlap(%d)\n", layer_, process_local ? replication_threshold : -1, feature_size, process_local, process_overlap);
//    }
    double stream_time = 0;
    stream_time -= MPI_Wtime();
    Nts->ZeroVarMem(message_grad,GPU_T);    

    current_send_part_id = 0;
    cpp=0;
    Nts->InitBlock(graph_partitions[current_send_part_id],rtminfo,feature_size,
                   feature_size, current_send_part_id, layer_); 
    Nts->BackwardScatterGradBackToMessage(input_grad, message_grad);//4-2
//            //NtsVar src_grad1=EdgeBackward(message_grad,src_inter_grad,Nts); //(2,3)->1
//            NtsVar src_grad1=EdgeBackward(message_grad,src_input_transferred,Nts);
//            output_grad=src_grad;//+src_grad1;
            Nts->DeviceSynchronize();

    R global_reducer;
    stream_time += MPI_Wtime();
    return global_reducer;
  } 
   
  
  
        template <typename R, typename M>
  R compute_sync_decoupled_edge(NtsVar &input_grad,
                                          NtsVar &dst_input_transferred,
                                          std::vector<CSC_segment_pinned *> &graph_partitions,
                                          std::function<void(VertexId, VertexAdjList<EdgeData>)> dense_signal,
                                          std::function<NtsVar(NtsVar&,NtsVar&,NtsScheduler* nts)> EdgeForward,
                                          std::function<NtsVar(NtsVar&,NtsVar&,NtsScheduler* nts)> EdgeBackward,
                                          NtsVar &output_grad,
                                          int feature_size)//backward
  {
    bool process_local = rtminfo->process_local;
    int layer_ = rtminfo->curr_layer;
    bool process_overlap = rtminfo->process_overlap;
    NtsComm->init_layer_all(feature_size,Mirror2Master,GPU_T);
    NtsComm->run_all_mirror_to_master();
    
    if (partition_id == 0)
    {
      printf("ComputeSyncEdgeDecoupled:layer(%d).process_local(%d).dimension(%d).reduce_comm(%d).overlap(%d)\n", layer_, process_local ? replication_threshold : -1, feature_size, process_local, process_overlap);
    }
    double stream_time = 0;
    stream_time -= MPI_Wtime();
    long max_recv_buffer_size = owned_vertices * sockets;
    Nts->ZeroVarMem(output_grad,GPU_T);    
    size_t basic_chunk = 64;
    {
#ifdef PRINT_DEBUG_MESSAGES
      if (partition_id == 0)
      {
        printf("dense mode\n");
      }
#endif
//      if (process_overlap)//pipeline
//      {
//            current_send_part_id = partition_id;
//            cpp=(current_send_part_id + 1) % partitions;
//            Nts->InitBlock(graph_partitions[(current_send_part_id + 1) % partitions],
//                            rtminfo,
//                            feature_size,
//                              feature_size,
//                                (current_send_part_id + 1) % partitions,
//                                 layer_); 
//            NtsVar src_input_transferred;//mark
//            NtsVar src_input;
//            NtsVar message=EdgeForward(src_input,src_input_transferred,dst_input,dst_input_transferred,Nts);
//            NtsVar src_inter_grad=Nts->NewLeafTensor(src_input_transferred);
//            NtsVar message_grad=Nts->NewLeafTensor(message);          
//            Nts->GatherBySrcFromDst(src_inter_grad,input_grad,message);//4->3
//            Nts->BackwardScatterGradBackToWeight(src_input_transferred, input_grad, message_grad);//4-2
//            NtsVar src_grad=EdgeBackward(message_grad,src_inter_grad,Nts); //(2,3)->1
//            Nts->SerializeMirrorToMsg(output_cpu_buffer,src_grad);
//            Nts->DeviceSynchronize();
//            rtminfo->device_sync();
//      }
//      else
      {
        //no pipeline
        for (int step = 0; step < partitions; step++)
        {
            current_send_part_id = (current_send_part_id + 1) % partitions;
            cpp=(current_send_part_id + 1) % partitions;
            Nts->InitBlock(graph_partitions[(current_send_part_id + 1) % partitions],
                            rtminfo,
                            feature_size,
                              feature_size,
                                (current_send_part_id + 1) % partitions,
                                 layer_); 
            NtsVar src_input_transferred=this->cachedData->mirror_input[layer_][(current_send_part_id + 1) % partitions];
            NtsVar message=this->cachedData->message[layer_][(current_send_part_id + 1) % partitions];
            //NtsVar message=EdgeForward(src_input_transferred,dst_input_transferred,Nts);
            NtsVar src_grad=Nts->NewLeafTensor(src_input_transferred);
            NtsVar message_grad=Nts->NewLeafTensor(message);
            
            Nts->GatherBySrcFromDst(src_grad,input_grad,message);//4->3
            Nts->BackwardScatterGradBackToWeight(src_input_transferred, input_grad, message_grad);//4-2
            //NtsVar src_grad1=EdgeBackward(message_grad,src_inter_grad,Nts); //(2,3)->1
            NtsVar src_grad1=EdgeBackward(message_grad,src_input_transferred,Nts);
                      std::cout<<"src_grad1: "<<src_grad1.dim()<<std::endl;
            src_grad=src_grad+src_grad1;
            Nts->SerializeMirrorToMsg(output_cpu_buffer,src_grad);
            Nts->DeviceSynchronize();
            
        }
      }

      for (int step = 0; step < partitions; step++)
      {
        current_send_part_id = (current_send_part_id + 1) % partitions;
        int i = current_send_part_id;
        NtsComm->set_current_send_partition(current_send_part_id);
        for (int t_i = 0; t_i < threads; t_i++)
        {
          *thread_state[t_i] = tuned_chunks_dense[i][t_i];
        }

//        cuda_stream->CUDA_DEVICE_SYNCHRONIZE();
//        if (current_send_part_id != partition_id&&process_overlap)
//        {

//        }

#pragma omp parallel
        {
          //printf("DEBUGstart%d\n",partition_id);
          int thread_id = omp_get_thread_num();
          int s_i = get_socket_id(thread_id);
          VertexId final_p_v_i = thread_state[thread_id]->end;

          while (true)
          {
            VertexId begin_p_v_i = __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
            if (begin_p_v_i >= final_p_v_i)
              break;
            VertexId end_p_v_i = begin_p_v_i + basic_chunk;
            if (end_p_v_i > final_p_v_i)
            {
              end_p_v_i = final_p_v_i;
            }
            for (VertexId p_v_i = begin_p_v_i; p_v_i < end_p_v_i; p_v_i++)
            {
              VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
              dense_signal(v_i, VertexAdjList<EdgeData>(incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i].index, incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i + 1].index));
            }
          }
          thread_state[thread_id]->status = STEALING;
          for (int t_offset = 1; t_offset < threads; t_offset++)
          {
            int t_i = (thread_id + t_offset) % threads;
            int s_i = get_socket_id(t_i);
            while (thread_state[t_i]->status != STEALING)
            {
              VertexId begin_p_v_i = __sync_fetch_and_add(&thread_state[t_i]->curr, basic_chunk);
              if (begin_p_v_i >= thread_state[t_i]->end)
                break;
              VertexId end_p_v_i = begin_p_v_i + basic_chunk;
              if (end_p_v_i > thread_state[t_i]->end)
              {
                end_p_v_i = thread_state[t_i]->end;
              }
              for (VertexId p_v_i = begin_p_v_i; p_v_i < end_p_v_i; p_v_i++)
              {
                VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
                dense_signal(v_i, VertexAdjList<EdgeData>(incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i].index, incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i + 1].index));
              }
            }
          }
        }
        NtsComm->trigger_one_partition(current_send_part_id);

      }
      for (int step = 0; step < partitions; step++)
      {
        int i = -1;
        MessageBuffer **used_buffer;          
        used_buffer=NtsComm->recv_one_partition(i,step);
        Nts->InitBlockSimple(graph_partitions[i],rtminfo,feature_size, 
                        feature_size, i,layer_);
        for (int s_i = 0; s_i < sockets; s_i++)
        {
          Nts->DeviceSynchronize();
          Nts->AggMsgToMaster(output_grad, used_buffer[s_i]->data,used_buffer[s_i]->count);
        }

      }

      Nts->DeviceSynchronize();
      NtsComm->release_communicator();
    }

    R global_reducer;
    stream_time += MPI_Wtime();
#ifdef PRINT_DEBUG_MESSAGES
    if (partition_id == 0)
    {
      printf("process_edges took %lf (s)\n", stream_time);
    }
#endif
    return global_reducer;
  } 
   
   
  
        
        // process edges
  template <typename R, typename M>
  R sync_compute_decoupled_edge(NtsVar &input_master_trans,
                 std::vector<CSC_segment_pinned *> &graph_partitions,
                 std::function<void(VertexId)> sparse_signal,
                 std::function<NtsVar(NtsVar&,NtsVar&,NtsScheduler* nts)> EdgeComputation,
                 NtsVar& Y,
                 int feature_size)
  {
    //int feature_size = gnnctx->layer_size[rtminfo->curr_layer];
    bool process_local = rtminfo->process_local;
    int layer_ = rtminfo->curr_layer;
    bool process_overlap = rtminfo->process_overlap;

//    if (partition_id == 0)
//    {
//      printf("SyncComputeDecoupledEdge:layer(%d).process_local(%d).dimension(%d).reduce_comm(%d).overlap(%d)\n", layer_, process_local ? replication_threshold : -1, feature_size, process_local, process_overlap);
//    }
    double stream_time = 0;
    stream_time -= MPI_Wtime();

    NtsComm->init_layer_all(feature_size,Master2Mirror,GPU_T);
    NtsComm->run_all_master_to_mirror_no_wait();
    
    
    NtsVar mirror_input=Nts->NewLeafTensor({get_max_partition_size(),(feature_size+1)});
    Nts->ZeroVarMem(Y);
    
    {//1-stage 
      current_send_part_id = partition_id;
      NtsComm->set_current_send_partition(current_send_part_id);
#pragma omp parallel for
      for (VertexId begin_v_i = partition_offset[partition_id]; begin_v_i < partition_offset[partition_id + 1]; begin_v_i += 1)
      {
        VertexId v_i = begin_v_i;
            sparse_signal(v_i);
      }
      for(int step=0;step<partitions;step++){
          int trigger_partition=(partition_id-step+partitions)%partitions;
          NtsComm->trigger_one_partition(trigger_partition,trigger_partition==current_send_part_id);
      }     
      //2-stage
      int current_recv_part_id = 0;
      for (int step = 0; step < partitions; step++)
      {
        int i = -1;
        MessageBuffer **used_buffer;
        used_buffer=NtsComm->recv_one_partition(i,step);
        for (int s_i = 0; s_i < sockets; s_i++)
        {
            int current_recv_part_id = i;
            NtsVar input_mirror_trans=Nts->NewLeafTensor({partition_offset[i + 1] - partition_offset[i],feature_size});
            Nts->InitBlock(graph_partitions[i],
                              rtminfo,
                              feature_size,
                              feature_size,
                              current_recv_part_id,
                              layer_);
            Nts->DeserializeMsgToMirror(input_mirror_trans,used_buffer[s_i]->data,used_buffer[s_i]->count);            
            cpp=i;// must specify the cpp, or the message will be flushed even though call encode function.
            NtsVar message=  EdgeComputation(input_mirror_trans,input_master_trans,Nts);
            Nts->GatherByDstFromSrc(Y, input_mirror_trans, message);  
            
            this->cachedData->mirror_input[layer_][i]=input_mirror_trans;
            this->cachedData->message[layer_][i]=message;
        }            
      }
      rtminfo->device_sync();
      NtsComm->release_communicator();
    }
    stream_time += MPI_Wtime();
    R global_reducer;
#ifdef PRINT_DEBUG_MESSAGES
    if (partition_id == 0)
    {
      printf("process_edges took %lf (s)\n", stream_time);
    }
#endif
    return global_reducer;
  }
    void free_all_tmp()
  {
    FreeEdge(row_indices_intergate);
    FreeEdge(column_offset_intergate);
    FreeBuffer(weight_gpu_intergate);
  }
#endif       
        
       
  void generate_backward_structure()
  {

    int fin = open(filename.c_str(), O_RDONLY);
    long bytes_to_read = lseek(fin, 0, SEEK_END);
    long read_bytes = 0;

//    for (int i = 0; i < sockets; i++)
//    {
//      numa_free(outgoing_adj_index[i], sizeof(EdgeId) * (vertices + 1));
//      outgoing_adj_bitmap[i]->~Bitmap();
//      numa_free(outgoing_adj_list[i], sizeof(AdjUnit<EdgeData>) * outgoing_edges[i]);
//      numa_free(compressed_outgoing_adj_index[i], sizeof(CompressedAdjIndexUnit) * compressed_outgoing_adj_vertices[i]);
//    }
//    free(outgoing_edges);
//    free(compressed_outgoing_adj_vertices);

    int start = partition_offset[partition_id];
    int row_num = partition_offset[partition_id + 1] - start;

    //this->in_degree_backward=this->alloc_vertex_array<VertexId>();

    incoming_edges_backward = new EdgeId[sockets];
    incoming_adj_index_backward = new EdgeId *[sockets];
    incoming_adj_list_backward = new AdjUnit<EdgeData> *[sockets];
    incoming_adj_bitmap_backward = new Bitmap *[sockets];

    compressed_incoming_adj_index_backward = new CompressedAdjIndexUnit *[sockets];
    ;
    compressed_incoming_adj_vertices_backward = new VertexId[sockets];

    //VertexId** write_pos= new VertexId[sockets];
    for (int i = 0; i < sockets; i++)
    {
      incoming_edges_backward[i] = 0;
      incoming_adj_bitmap_backward[i] = new Bitmap(vertices);
      incoming_adj_bitmap_backward[i]->clear();
      incoming_adj_index_backward[i] = (EdgeId *)numa_alloc_onnode(sizeof(EdgeId) * (vertices + 1), i);
      memset(incoming_adj_index_backward[i], 0, sizeof(EdgeId) * (vertices + 1));
    }
    int local_edges = 0;
    //    printf("finish_1_%s\n",filename.c_str());
    EdgeUnit<EdgeData> *read_edge_buffer = new EdgeUnit<EdgeData>[CHUNKSIZE];
    //    printf("finish_2_%d\n",partition_id);
    assert(lseek(fin, 0, SEEK_SET) == 0);

    //    printf("%d   jello%d\n",read_edge_buffer[0].dst,local_partition_offset[1]);
    while (read_bytes < bytes_to_read)
    { // load the graph with a big chunk iteratively
      long curr_read_bytes;
      if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE)
      {
        curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
      }
      else
      {
        curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
      }
      assert(curr_read_bytes >= 0);
      //      printf("read_size %d\n",curr_read_bytes);
      read_bytes += curr_read_bytes;
      EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;

      for (EdgeId e_i = 0; e_i < curr_read_edges; e_i++)
      {
        VertexId src = read_edge_buffer[e_i].src;
        VertexId dst = read_edge_buffer[e_i].dst;
        int tmp = 0;

        for (int s_i = 0; s_i < sockets; s_i++)
        {
          if (local_partition_offset[s_i] <= dst && dst < local_partition_offset[s_i + 1])
          {
          if (!incoming_adj_bitmap_backward[s_i]->get_bit(src))
            {
                incoming_adj_bitmap_backward[s_i]->set_bit(src);
                // incoming_adj_index_backward[s_i][dst] = 0;
            }
            incoming_adj_index_backward[s_i][src] += 1;
            incoming_edges_backward[s_i]++;
          }
        }
      }
    }
    for (int s_i = 0; s_i < sockets; s_i++)
    {
      compressed_incoming_adj_vertices_backward[s_i] = 0;
      // if(partition_id==0){
      //  std::cout<<incoming_adj_index[s_i][0]<<" on "<<partition_id;
      // }
      incoming_adj_list_backward[s_i] = (AdjUnit<EdgeData> *)numa_alloc_onnode(unit_size * incoming_edges_backward[s_i], s_i);

      for (int i = 0; i < vertices; i++)
      {
        if (this->incoming_adj_bitmap_backward[s_i]->get_bit(i))
        {
          compressed_incoming_adj_vertices_backward[s_i] += 1;
        }
        incoming_adj_index_backward[s_i][i + 1] += incoming_adj_index_backward[s_i][i];
      }
      this->compressed_incoming_adj_index_backward[s_i] = (CompressedAdjIndexUnit *)
          numa_alloc_onnode(sizeof(CompressedAdjIndexUnit) * (compressed_incoming_adj_vertices_backward[s_i] + 1), s_i);

      compressed_incoming_adj_vertices_backward[s_i] = 0;
      EdgeId last_e_i = 0;

      for (int i = 0; i < vertices; i++)
      {
        if (this->incoming_adj_bitmap_backward[s_i]->get_bit(i))
        {
          last_e_i = incoming_adj_index_backward[s_i][i];
          compressed_incoming_adj_index_backward[s_i][compressed_incoming_adj_vertices_backward[s_i]].vertex = i;
          compressed_incoming_adj_vertices_backward[s_i] += 1;
          compressed_incoming_adj_index_backward[s_i][compressed_incoming_adj_vertices_backward[s_i]].index = last_e_i;
        }
      }

      for (VertexId p_v_i = 0; p_v_i < compressed_incoming_adj_vertices_backward[s_i]; p_v_i++)
      {
        VertexId v_i = compressed_incoming_adj_index_backward[s_i][p_v_i].vertex;
        incoming_adj_index_backward[s_i][v_i] = compressed_incoming_adj_index_backward[s_i][p_v_i].index;
        incoming_adj_index_backward[s_i][v_i + 1] = compressed_incoming_adj_index_backward[s_i][p_v_i + 1].index;
      }

      // std::cout<<incoming_adj_index[s_i][0]<<" off "<<partition_id;
    }
    read_bytes = 0;
    assert(lseek(fin, 0, SEEK_SET) == 0);

    while (read_bytes < bytes_to_read)
    { // load the graph with a big chunk iteratively
      long curr_read_bytes;
      if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE)
      {
        curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
      }
      else
      {
        curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
      }
      assert(curr_read_bytes >= 0);
      read_bytes += curr_read_bytes;
      EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
      //  #pragma omp parallel for
      for (EdgeId e_i = 0; e_i < curr_read_edges; e_i++)
      {
        VertexId src = read_edge_buffer[e_i].src;
        VertexId dst = read_edge_buffer[e_i].dst;
        int tmp = 0;

        for (int s_i = 0; s_i < sockets; s_i++)
        {
          if (local_partition_offset[s_i] <= dst && dst < local_partition_offset[s_i + 1])
          {
            EdgeId pos = __sync_fetch_and_add(&incoming_adj_index_backward[s_i][src], 1);
            incoming_adj_list_backward[s_i][pos].neighbour = dst;
            if (!std::is_same<EdgeData, Empty>::value)
            {
              incoming_adj_list_backward[s_i][pos].edge_data = read_edge_buffer[e_i].edge_data;
            }
          }
        }
      }
    }
    for (int s_i = 0; s_i < sockets; s_i++)
    {
      for (VertexId p_v_i = 0; p_v_i < this->compressed_incoming_adj_vertices_backward[s_i]; p_v_i++)
      {
        VertexId v_i = this->compressed_incoming_adj_index_backward[s_i][p_v_i].vertex;
        incoming_adj_index_backward[s_i][v_i] = compressed_incoming_adj_index_backward[s_i][p_v_i].index;
        incoming_adj_index_backward[s_i][v_i + 1] = compressed_incoming_adj_index_backward[s_i][p_v_i + 1].index;
      }
    }

    close(fin);
    tune_chunks_backward();
    //fill incoming_adj_index
    for (int i_s; i_s < sockets; i_s++)
    {
      for (VertexId vtx = 0; vtx < vertices; vtx++)
      {
        if (incoming_adj_index[i_s][vtx + 1] == 0)
          incoming_adj_index[i_s][vtx + 1] = incoming_adj_index[i_s][vtx];
        if (outgoing_adj_index[i_s][vtx + 1] == 0)
          outgoing_adj_index[i_s][vtx + 1] = outgoing_adj_index[i_s][vtx];
      }
    }
    for (VertexId vtx = 0; vtx < vertices; vtx++)
    {
      if (in_degree_for_backward[vtx] < 1)
        in_degree_for_backward[vtx] = 1; //local
      if (out_degree_for_backward[vtx] < 1)
        out_degree_for_backward[vtx] = 1; //local
    }
  }

  void generate_COO()
  {
    _graph_cpu_in = new COOChunk();
    
    VertexId edge_size_in=0;
    for (int i = 0; i < sockets; i++)
    {
      edge_size_in += (VertexId)outgoing_adj_index[i][vertices];
    }
    
    _graph_cpu_in->dstList = new VertexId[edge_size_in];
    _graph_cpu_in->srcList = new VertexId[edge_size_in];
    _graph_cpu_in->numofedges = edge_size_in;

    int write_position_in = 0;
    for (int k = 0; k < sockets; k++){
      for (VertexId vtx = 0; vtx < vertices; vtx++)
      {
        for (int i = outgoing_adj_index[k][vtx]; i < outgoing_adj_index[k][vtx + 1]; i++)
        {
          _graph_cpu_in->srcList[write_position_in] = vtx;
          _graph_cpu_in->dstList[write_position_in++] = outgoing_adj_list[k][i].neighbour;
        }
        
      }
    }
    if (partition_id == 0)
      printf("GNNmini::Preprocessing[Generate Edges]\n");
  }
  
  void reorder_COO_W2W()
  { //replication
    graph_shard_in.clear();
    VertexId edge_size_out = 0; //(VertexId)incoming_adj_index[sockets-1][vertices];
    VertexId edge_size_in=0;
    for (int i = 0; i < sockets; i++)
    {
      edge_size_out += (VertexId)incoming_adj_index[i][vertices];
      edge_size_in += (VertexId)incoming_adj_index_backward[i][vertices];
    }
    
    int src_blocks = partitions;
    for (int i = 0; i < src_blocks; i++)
    {
      graph_shard_in.push_back(new COOChunk());
      graph_shard_in[i]->dst_range[0] = partition_offset[partition_id];
      graph_shard_in[i]->dst_range[1] = partition_offset[partition_id + 1];
      graph_shard_in[i]->src_range[0] = partition_offset[i];
      graph_shard_in[i]->src_range[1] = partition_offset[i + 1];
    }
    for (int i = 0; i < edge_size_in; i++)
    {
      int src_bucket = this->get_partition_id(_graph_cpu_in->srcList[i]);
      graph_shard_in[src_bucket]->numofedges += 1;
    }
    for (int i = 0; i < src_blocks; i++)
    {
      graph_shard_in[i]->src_delta = new VertexId[graph_shard_in[i]->numofedges];
      graph_shard_in[i]->dst_delta = new VertexId[graph_shard_in[i]->numofedges];
      graph_shard_in[i]->counter=0;
    }
    for (int i = 0; i < edge_size_in; i++)
    {
      int source = _graph_cpu_in->src()[i];
      int destination = _graph_cpu_in->dst()[i];
      int bucket_s = this->get_partition_id(source);
      int offset = graph_shard_in[bucket_s]->counter++;
      graph_shard_in[bucket_s]->src_delta[offset] = source;
      graph_shard_in[bucket_s]->dst_delta[offset] = destination;
    }

    if (partition_id == 0)
      printf("GNNmini::Preprocessing[Reorganize Edges]\n");
  }


  void print_info()
  {
    MPI_Datatype vid_t = get_mpi_data_type<int>();
    int *local_e_num = NULL;
    int *local_v_num = NULL;
    //int *out_node = NULL;
    local_e_num = new int[partitions];
    local_v_num = new int[partitions];
    //out_node = new int[partitions];
    
    memset(local_e_num, 0, sizeof(int) * partitions);
    memset(local_v_num, 0, sizeof(int) * partitions);
    //memset(out_node, 0, sizeof(int) * partitions);
    
    local_e_num[partition_id] = gnnctx->l_e_num;
    local_v_num[partition_id] = gnnctx->l_v_num;

    MPI_Allreduce(MPI_IN_PLACE, local_e_num, partitions, vid_t, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, local_v_num, partitions, vid_t, MPI_SUM, MPI_COMM_WORLD);

    if (partition_id == 0)
    {
      int avg_l_v = 0;
      int avg_l_e = 0;

      printf("GNNmini::Preprocessing[Print All Info:]\n");
      printf("GNNmini::Vertices:[%d],Edges:[%d], RepThreshold:[%d], Layers: ", vertices, edges, config->repthreshold);
      std::cout << config->layer_string << std::endl;
      printf("GNNmini::ProcessLocal\t[%d]\n", config->process_local);
      printf("GNNmini::ProcessOverlap\t[%d]\n", config->overlap);
      printf("GNNmini::{assigned Vertices}:\t");
      for (int i = 0; i < partitions; i++)
      {
        printf("(%d)[%d]\t", i, local_v_num[i]);
        avg_l_v += local_v_num[i];
      }
      printf("(avg)[%d]\t", avg_l_v / partitions);

      printf("\nGNNmini::{assigned Edges}:\t");
      for (int i = 0; i < partitions; i++)
      {
        printf("(%d)[%d]\t", i, local_e_num[i]);
        avg_l_e += local_e_num[i];
      }
      printf("(avg)[%d]\t", avg_l_e / partitions);
      printf("\nGNNmini::Preprocessing[Finish Print All Info:]\n");
    }
  }
  
    template <typename R, typename M>
  R process_edges(std::function<void(VertexId, VertexAdjList<EdgeData>)> dense_signal,
                  std::function<R(VertexId, M)> dense_slot,
                  Bitmap *active,
                  Bitmap *dense_selective = nullptr)
  {
    double stream_time = 0;
    stream_time -= MPI_Wtime();
    int con = 0;

    for (int t_i = 0; t_i < threads; t_i++)
    {
      local_send_buffer[t_i]->resize(sizeof(MsgUnit<M>) * local_send_buffer_limit);
      local_send_buffer[t_i]->count = 0;
    }
    R reducer = 0;
    for (int i = 0; i < partitions; i++)
    {
      for (int s_i = 0; s_i < sockets; s_i++)
      {
        recv_buffer[i][s_i]->resize(sizeof(MsgUnit<M>) * owned_vertices * sockets);
        send_buffer[i][s_i]->resize(sizeof(MsgUnit<M>) * (partition_offset[i + 1] - partition_offset[i]) * sockets);
        send_buffer[i][s_i]->count = 0;
        recv_buffer[i][s_i]->count = 0;
      }
    } //init the send and receive buffer
    size_t basic_chunk = 64;
    {
      // dense selective bitmap
      if (dense_selective != nullptr && partitions > 1)
      {
        double sync_time = 0;
        sync_time -= get_time();
        std::thread send_thread([&]() {
          for (int step = 1; step < partitions; step++)
          {
            int recipient_id = (partition_id + step) % partitions;
            MPI_Send(dense_selective->data + WORD_OFFSET(partition_offset[partition_id]), owned_vertices / 64, MPI_UNSIGNED_LONG, recipient_id, PassMessage, MPI_COMM_WORLD);
          }
        });
        std::thread recv_thread([&]() {
          for (int step = 1; step < partitions; step++)
          {
            int sender_id = (partition_id - step + partitions) % partitions;
            MPI_Recv(dense_selective->data + WORD_OFFSET(partition_offset[sender_id]), (partition_offset[sender_id + 1] - partition_offset[sender_id]) / 64, MPI_UNSIGNED_LONG, sender_id, PassMessage, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          }
        });
        send_thread.join();
        recv_thread.join();
        MPI_Barrier(MPI_COMM_WORLD); // denseselect fill the bitmap array
        sync_time += get_time();
      }
#ifdef PRINT_DEBUG_MESSAGES
      if (partition_id == 0)
      {
        printf("dense mode\n");
      }
#endif
      int *send_queue = new int[partitions];
      int *recv_queue = new int[partitions];
      volatile int send_queue_size = 0;
      volatile int recv_queue_size = 0;
      std::mutex send_queue_mutex;
      std::mutex recv_queue_mutex;

      std::thread send_thread([&]() {
        for (int step = 0; step < partitions; step++)
        {
          if (step == partitions - 1)
          {
            break;
          }
          while (true)
          {
            send_queue_mutex.lock();
            bool condition = (send_queue_size <= step);
            send_queue_mutex.unlock();
            if (!condition)
              break;
            __asm volatile("pause" ::
                               : "memory");
          }
          int i = send_queue[step];
          for (int s_i = 0; s_i < sockets; s_i++)
          {
            MPI_Send(send_buffer[i][s_i]->data, sizeof(MsgUnit<M>) * send_buffer[i][s_i]->count, MPI_CHAR, i, PassMessage, MPI_COMM_WORLD);
          }
        }
      });
      std::thread recv_thread([&]() {
        std::vector<std::thread> threads;
        for (int step = 1; step < partitions; step++)
        {
          int i = (partition_id - step + partitions) % partitions;
          threads.emplace_back([&](int i) {
            for (int s_i = 0; s_i < sockets; s_i++)
            {
              MPI_Status recv_status;
              MPI_Probe(i, PassMessage, MPI_COMM_WORLD, &recv_status);
              MPI_Get_count(&recv_status, MPI_CHAR, &recv_buffer[i][s_i]->count);
              MPI_Recv(recv_buffer[i][s_i]->data, recv_buffer[i][s_i]->count, MPI_CHAR, i, PassMessage, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
              recv_buffer[i][s_i]->count /= sizeof(MsgUnit<M>);
            }
          },
                               i);
        }
        for (int step = 1; step < partitions; step++)
        {
          int i = (partition_id - step + partitions) % partitions;
          threads[step - 1].join();
          recv_queue[recv_queue_size] = i;
          recv_queue_mutex.lock();
          recv_queue_size += 1;
          recv_queue_mutex.unlock();
        }
        recv_queue[recv_queue_size] = partition_id;
        recv_queue_mutex.lock();
        recv_queue_size += 1;
        recv_queue_mutex.unlock();
      });
      current_send_part_id = partition_id;
      for (int step = 0; step < partitions; step++)
      {
        current_send_part_id = (current_send_part_id + 1) % partitions;
        int i = current_send_part_id;
        for (int t_i = 0; t_i < threads; t_i++)
        {
          *thread_state[t_i] = tuned_chunks_dense[i][t_i];
          //          printf("Range %d\t%d\n",tuned_chunks_dense[i][t_i].curr,tuned_chunks_dense[i][t_i].end);
        }

#pragma omp parallel
        {
          int thread_id = omp_get_thread_num();
          int s_i = get_socket_id(thread_id);
          VertexId final_p_v_i = thread_state[thread_id]->end;
          while (true)
          {
            VertexId begin_p_v_i = __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
            if (begin_p_v_i >= final_p_v_i)
              break;
            VertexId end_p_v_i = begin_p_v_i + basic_chunk;
            if (end_p_v_i > final_p_v_i)
            {
              end_p_v_i = final_p_v_i;
            }
            for (VertexId p_v_i = begin_p_v_i; p_v_i < end_p_v_i; p_v_i++)
            {
              VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
              dense_signal(v_i, VertexAdjList<EdgeData>(incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i].index, incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i + 1].index));
            }
          }
          thread_state[thread_id]->status = STEALING;
          for (int t_offset = 1; t_offset < threads; t_offset++)
          {
            int t_i = (thread_id + t_offset) % threads;
            int s_i = get_socket_id(t_i);
            while (thread_state[t_i]->status != STEALING)
            {
              VertexId begin_p_v_i = __sync_fetch_and_add(&thread_state[t_i]->curr, basic_chunk);
              if (begin_p_v_i >= thread_state[t_i]->end)
                break;
              VertexId end_p_v_i = begin_p_v_i + basic_chunk;
              if (end_p_v_i > thread_state[t_i]->end)
              {
                end_p_v_i = thread_state[t_i]->end;
              }
              for (VertexId p_v_i = begin_p_v_i; p_v_i < end_p_v_i; p_v_i++)
              {
                VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
                dense_signal(v_i, VertexAdjList<EdgeData>(incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i].index, incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i + 1].index));
              }
            }
          }
        }
#pragma omp parallel for
        for (int t_i = 0; t_i < threads; t_i++)
        {
          flush_local_send_buffer<M>(t_i);
        }
        if (i != partition_id)
        {
          send_queue[send_queue_size] = i;
          send_queue_mutex.lock();
          send_queue_size += 1;
          send_queue_mutex.unlock();
        }
      }
      for (int step = 0; step < partitions; step++)
      {
        while (true)
        {
          recv_queue_mutex.lock();
          bool condition = (recv_queue_size <= step);
          recv_queue_mutex.unlock();
          if (!condition)
            break;
          __asm volatile("pause" ::
                             : "memory");
        }
        int i = recv_queue[step];
        MessageBuffer **used_buffer;
        if (i == partition_id)
        {
          used_buffer = send_buffer[i];
        }
        else
        {
          used_buffer = recv_buffer[i];
        }
        for (int t_i = 0; t_i < threads; t_i++)
        {
          int s_i = get_socket_id(t_i);
          int s_j = get_socket_offset(t_i);
          VertexId partition_size = used_buffer[s_i]->count;
          thread_state[t_i]->curr = partition_size / threads_per_socket / basic_chunk * basic_chunk * s_j;
          thread_state[t_i]->end = partition_size / threads_per_socket / basic_chunk * basic_chunk * (s_j + 1);
          if (s_j == threads_per_socket - 1)
          {
            thread_state[t_i]->end = used_buffer[s_i]->count;
          }
          thread_state[t_i]->status = WORKING;
        }
#pragma omp parallel reduction(+ \
                               : reducer)
        {
          R local_reducer = 0;
          int thread_id = omp_get_thread_num();
          int s_i = get_socket_id(thread_id);
          MsgUnit<M> *buffer = (MsgUnit<M> *)used_buffer[s_i]->data;
          while (true)
          {
            VertexId b_i = __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
            if (b_i >= thread_state[thread_id]->end)
              break;
            VertexId begin_b_i = b_i;
            VertexId end_b_i = b_i + basic_chunk;
            if (end_b_i > thread_state[thread_id]->end)
            {
              end_b_i = thread_state[thread_id]->end;
            }
            for (b_i = begin_b_i; b_i < end_b_i; b_i++)
            {
              VertexId v_i = buffer[b_i].vertex;
              M msg_data = buffer[b_i].msg_data;
              local_reducer += dense_slot(v_i, msg_data);
            }
          }
          thread_state[thread_id]->status = STEALING;
          reducer += local_reducer;
        }
      }
      send_thread.join();
      recv_thread.join();
      delete[] send_queue;
      delete[] recv_queue;
      //      printf("con%d_%d\n",con,partition_offset[1]);
    }

    R global_reducer;
    MPI_Datatype dt = get_mpi_data_type<R>();
    MPI_Allreduce(&reducer, &global_reducer, 1, dt, MPI_SUM, MPI_COMM_WORLD);
    stream_time += MPI_Wtime();
#ifdef PRINT_DEBUG_MESSAGES
    if (partition_id == 0)
    {
      printf("process_edges took %lf (s)\n", stream_time);
    }
#endif
    return global_reducer;
  }
    
    
  template <typename R, typename M>
  R process_edges_simple(std::function<void(VertexId, VertexAdjList<EdgeData>)> dense_signal,
                         std::function<R(VertexId, M)> dense_slot,
                         Bitmap *active = nullptr,
                         Bitmap *dense_selective = nullptr)
  {
    double stream_time = 0;
    stream_time -= MPI_Wtime();

    double compute_time = 0;
    compute_time -= MPI_Wtime();
    for (int t_i = 0; t_i < threads; t_i++)
    {
      local_send_buffer[t_i]->resize(sizeof(MsgUnit<M>) * local_send_buffer_limit);
      local_send_buffer[t_i]->count = 0;
    }
    R reducer = 0;
    for (int i = 0; i < partitions; i++)
    {
      for (int s_i = 0; s_i < sockets; s_i++)
      {
        recv_buffer[i][s_i]->resize(sizeof(MsgUnit<M>) * owned_vertices * sockets);
        send_buffer[i][s_i]->resize(sizeof(MsgUnit<M>) * (partition_offset[i + 1] - partition_offset[i]) * sockets);
        send_buffer[i][s_i]->count = 0;
        recv_buffer[i][s_i]->count = 0;
      }
    } //init the send and receive buffer
    size_t basic_chunk = 64;
    {
      // dense selective bitmap
      if (dense_selective != nullptr && partitions > 1)
      {
        double sync_time = 0;
        sync_time -= get_time();
        std::thread send_thread([&]() {
          for (int step = 1; step < partitions; step++)
          {
            int recipient_id = (partition_id + step) % partitions;
            MPI_Send(dense_selective->data + WORD_OFFSET(partition_offset[partition_id]), owned_vertices / 64, MPI_UNSIGNED_LONG, recipient_id, PassMessage, MPI_COMM_WORLD);
          }
        });
        std::thread recv_thread([&]() {
          for (int step = 1; step < partitions; step++)
          {
            int sender_id = (partition_id - step + partitions) % partitions;
            MPI_Recv(dense_selective->data + WORD_OFFSET(partition_offset[sender_id]), (partition_offset[sender_id + 1] - partition_offset[sender_id]) / 64, MPI_UNSIGNED_LONG, sender_id, PassMessage, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          }
        });
        send_thread.join();
        recv_thread.join();
        MPI_Barrier(MPI_COMM_WORLD); // denseselect fill the bitmap array
        sync_time += get_time();
      }
#ifdef PRINT_DEBUG_MESSAGES
      if (partition_id == 0)
      {
        printf("dense mode\n");
      }
#endif
      int *send_queue = new int[partitions];
      int *recv_queue = new int[partitions];
      volatile int send_queue_size = 0;
      volatile int recv_queue_size = 0;
      std::mutex send_queue_mutex;
      std::mutex recv_queue_mutex;

      std::thread send_thread([&]() {
        for (int step = 0; step < partitions; step++)
        {
          if (step == partitions - 1)
          {
            break;
          }
          while (true)
          {
            send_queue_mutex.lock();
            bool condition = (send_queue_size <= step);
            send_queue_mutex.unlock();
            if (!condition)
              break;
            __asm volatile("pause" ::
                               : "memory");
          }
          int i = send_queue[step];
          for (int s_i = 0; s_i < sockets; s_i++)
          {
            MPI_Send(send_buffer[i][s_i]->data, sizeof(MsgUnit<M>) * send_buffer[i][s_i]->count, MPI_CHAR, i, PassMessage, MPI_COMM_WORLD);
          }
        }
      });
      std::thread recv_thread([&]() {
        std::vector<std::thread> threads;
        for (int step = 1; step < partitions; step++)
        {
          int i = (partition_id - step + partitions) % partitions;
          threads.emplace_back([&](int i) {
            for (int s_i = 0; s_i < sockets; s_i++)
            {
              MPI_Status recv_status;
              MPI_Probe(i, PassMessage, MPI_COMM_WORLD, &recv_status);
              MPI_Get_count(&recv_status, MPI_CHAR, &recv_buffer[i][s_i]->count);
              MPI_Recv(recv_buffer[i][s_i]->data, recv_buffer[i][s_i]->count, MPI_CHAR, i, PassMessage, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
              recv_buffer[i][s_i]->count /= sizeof(MsgUnit<M>);
            }
          },
                               i);
        }
        for (int step = 1; step < partitions; step++)
        {
          int i = (partition_id - step + partitions) % partitions;
          threads[step - 1].join();
          recv_queue[recv_queue_size] = i;
          recv_queue_mutex.lock();
          recv_queue_size += 1;
          recv_queue_mutex.unlock();
        }
        recv_queue[recv_queue_size] = partition_id;
        recv_queue_mutex.lock();
        recv_queue_size += 1;
        recv_queue_mutex.unlock();
      });
      current_send_part_id = partition_id;
      for (int step = 0; step < partitions; step++)
      {
        current_send_part_id = (current_send_part_id + 1) % partitions;
        int i = current_send_part_id;
        for (int t_i = 0; t_i < threads; t_i++)
        {
          *thread_state[t_i] = tuned_chunks_dense[i][t_i];
          //          printf("Range %d\t%d\n",tuned_chunks_dense[i][t_i].curr,tuned_chunks_dense[i][t_i].end);
        }

#pragma omp parallel
        {
          int thread_id = omp_get_thread_num();
          int s_i = get_socket_id(thread_id);
          VertexId final_p_v_i = thread_state[thread_id]->end;
          //          if(partition_id==1){
          //              printf("%d\n",thread_state[thread_id]->end);
          //          }
          while (true)
          {
            VertexId begin_p_v_i = __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
            if (begin_p_v_i >= final_p_v_i)
              break;
            VertexId end_p_v_i = begin_p_v_i + basic_chunk;
            if (end_p_v_i > final_p_v_i)
            {
              end_p_v_i = final_p_v_i;
            }
            for (VertexId p_v_i = begin_p_v_i; p_v_i < end_p_v_i; p_v_i++)
            {
              VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
              dense_signal(v_i, VertexAdjList<EdgeData>(incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i].index, incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i + 1].index));
            }
          }
          thread_state[thread_id]->status = STEALING;
          for (int t_offset = 1; t_offset < threads; t_offset++)
          {
            int t_i = (thread_id + t_offset) % threads;
            int s_i = get_socket_id(t_i);
            while (thread_state[t_i]->status != STEALING)
            {
              VertexId begin_p_v_i = __sync_fetch_and_add(&thread_state[t_i]->curr, basic_chunk);
              if (begin_p_v_i >= thread_state[t_i]->end)
                break;
              VertexId end_p_v_i = begin_p_v_i + basic_chunk;
              if (end_p_v_i > thread_state[t_i]->end)
              {
                end_p_v_i = thread_state[t_i]->end;
              }
              for (VertexId p_v_i = begin_p_v_i; p_v_i < end_p_v_i; p_v_i++)
              {
                VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
                dense_signal(v_i, VertexAdjList<EdgeData>(incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i].index, incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i + 1].index));
              }
            }
          }
        }
#pragma omp parallel for
        for (int t_i = 0; t_i < threads; t_i++)
        {
          flush_local_send_buffer<M>(t_i);
        }
        if (i != partition_id)
        {
          send_queue[send_queue_size] = i;
          send_queue_mutex.lock();
          send_queue_size += 1;
          send_queue_mutex.unlock();
        }
      }
      compute_time += MPI_Wtime();
      all_compute_time += compute_time;
      double overlap_time = 0;
      overlap_time -= MPI_Wtime();
      for (int step = 0; step < partitions; step++)
      {
        while (true)
        {
          recv_queue_mutex.lock();
          bool condition = (recv_queue_size <= step);
          recv_queue_mutex.unlock();
          if (!condition)
            break;
          __asm volatile("pause" ::
                             : "memory");
        }
        int i = recv_queue[step];
        MessageBuffer **used_buffer;
        if (i == partition_id)
        {
          used_buffer = send_buffer[i];
        }
        else
        {
          used_buffer = recv_buffer[i];
        }
        for (int t_i = 0; t_i < threads; t_i++)
        {
          int s_i = get_socket_id(t_i);
          int s_j = get_socket_offset(t_i);
          VertexId partition_size = used_buffer[s_i]->count;
          thread_state[t_i]->curr = partition_size / threads_per_socket / basic_chunk * basic_chunk * s_j;
          thread_state[t_i]->end = partition_size / threads_per_socket / basic_chunk * basic_chunk * (s_j + 1);
          if (s_j == threads_per_socket - 1)
          {
            thread_state[t_i]->end = used_buffer[s_i]->count;
          }
          thread_state[t_i]->status = WORKING;
        }
#pragma omp parallel reduction(+ \
                               : reducer)
        {
          R local_reducer = 0;
          int thread_id = omp_get_thread_num();
          int s_i = get_socket_id(thread_id);
          MsgUnit<M> *buffer = (MsgUnit<M> *)used_buffer[s_i]->data;
          while (true)
          {
            VertexId b_i = __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
            if (b_i >= thread_state[thread_id]->end)
              break;
            VertexId begin_b_i = b_i;
            VertexId end_b_i = b_i + basic_chunk;
            if (end_b_i > thread_state[thread_id]->end)
            {
              end_b_i = thread_state[thread_id]->end;
            }
            for (b_i = begin_b_i; b_i < end_b_i; b_i++)
            {
              VertexId v_i = buffer[b_i].vertex;
              M msg_data = buffer[b_i].msg_data;
              local_reducer += dense_slot(v_i, msg_data);
            }
          }
          thread_state[thread_id]->status = STEALING;
          reducer += local_reducer;
        }
      }
      overlap_time += MPI_Wtime();
      all_overlap_time += overlap_time;
      double wait_time = 0;
      wait_time -= MPI_Wtime();
      send_thread.join();
      recv_thread.join();
      wait_time += MPI_Wtime();
      all_wait_time += wait_time;
      delete[] send_queue;
      delete[] recv_queue;
      //      printf("con%d_%d\n",con,partition_offset[1]);
    }

    R global_reducer;
    MPI_Datatype dt = get_mpi_data_type<R>();
    MPI_Allreduce(&reducer, &global_reducer, 1, dt, MPI_SUM, MPI_COMM_WORLD);
    stream_time += MPI_Wtime();
#ifdef PRINT_DEBUG_MESSAGES
    if (partition_id == 0)
    {
      printf("process_edges took %lf (s)\n", stream_time);
    }
#endif
    return global_reducer;
  }
 
          
 
    
      template <typename R, typename M>
  R process_edges_forward(std::function<void(VertexId, VertexAdjList<EdgeData>,VertexId)> dense_signal,
                  std::function<R(VertexId, M*)> dense_slot,
                  int feature_size,
                  Bitmap *active,
                  Bitmap *dense_selective = nullptr)
  {
    omp_set_num_threads(threads);
    double stream_time = 0;
    stream_time -= MPI_Wtime();
    int con = 0;

    for (int t_i = 0; t_i < threads; t_i++)
    {
      local_send_buffer[t_i]->resize(sizeofM<M>(feature_size) * local_send_buffer_limit);
      local_send_buffer[t_i]->count = 0;
    }
    R reducer = 0;
    for (int i = 0; i < partitions; i++)
    {
      for (int s_i = 0; s_i < sockets; s_i++)
      {
        recv_buffer[i][s_i]->resize_pinned(sizeofM<M>(feature_size) * owned_vertices * sockets);
        send_buffer[i][s_i]->resize_pinned(sizeofM<M>(feature_size) * (partition_offset[i + 1] - partition_offset[i]) * sockets);
        send_buffer[i][s_i]->count = 0;
        recv_buffer[i][s_i]->count = 0;
      }
    } //init the send and receive buffer
    size_t basic_chunk = 64;
    {
      // dense selective bitmap
      if (dense_selective != nullptr && partitions > 1)
      {
        double sync_time = 0;
        sync_time -= get_time();
        std::thread send_thread([&]() {
          for (int step = 1; step < partitions; step++)
          {
            int recipient_id = (partition_id + step) % partitions;
            MPI_Send(dense_selective->data + WORD_OFFSET(partition_offset[partition_id]), owned_vertices / 64, MPI_UNSIGNED_LONG, recipient_id, PassMessage, MPI_COMM_WORLD);
          }
        });
        std::thread recv_thread([&]() {
          for (int step = 1; step < partitions; step++)
          {
            int sender_id = (partition_id - step + partitions) % partitions;
            MPI_Recv(dense_selective->data + WORD_OFFSET(partition_offset[sender_id]), (partition_offset[sender_id + 1] - partition_offset[sender_id]) / 64, MPI_UNSIGNED_LONG, sender_id, PassMessage, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          }
        });
        send_thread.join();
        recv_thread.join();
        MPI_Barrier(MPI_COMM_WORLD); // denseselect fill the bitmap array
        sync_time += get_time();
      }
#ifdef PRINT_DEBUG_MESSAGES
      if (partition_id == 0)
      {
        printf("dense mode\n");
      }
#endif
      int *send_queue = new int[partitions];
      int *recv_queue = new int[partitions];
      volatile int send_queue_size = 0;
      volatile int recv_queue_size = 0;
      std::mutex send_queue_mutex;
      std::mutex recv_queue_mutex;

      std::thread send_thread([&]() {
        for (int step = 0; step < partitions; step++)
        {
          if (step == partitions - 1)
          {
            break;
          }
          while (true)
          {
            send_queue_mutex.lock();
            bool condition = (send_queue_size <= step);
            send_queue_mutex.unlock();
            if (!condition)
              break;
            __asm volatile("pause" ::
                               : "memory");
          }
          int i = send_queue[step];
          for (int s_i = 0; s_i < sockets; s_i++)
          {
            MPI_Send(send_buffer[i][s_i]->data, sizeofM<M>(feature_size) * send_buffer[i][s_i]->count, MPI_CHAR, i, PassMessage, MPI_COMM_WORLD);
          }
        }
      });
      std::thread recv_thread([&]() {
        std::vector<std::thread> threads;
        for (int step = 1; step < partitions; step++)
        {
          int i = (partition_id - step + partitions) % partitions;
          threads.emplace_back([&](int i) {
            for (int s_i = 0; s_i < sockets; s_i++)
            {
              MPI_Status recv_status;
              MPI_Probe(i, PassMessage, MPI_COMM_WORLD, &recv_status);
              MPI_Get_count(&recv_status, MPI_CHAR, &recv_buffer[i][s_i]->count);
              MPI_Recv(recv_buffer[i][s_i]->data, recv_buffer[i][s_i]->count, MPI_CHAR, i, PassMessage, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
              recv_buffer[i][s_i]->count /= sizeofM<M>(feature_size);
            }
          },
                               i);
        }
        for (int step = 1; step < partitions; step++)
        {
          int i = (partition_id - step + partitions) % partitions;
          threads[step - 1].join();
          recv_queue[recv_queue_size] = i;
          recv_queue_mutex.lock();
          recv_queue_size += 1;
          recv_queue_mutex.unlock();
        }
        recv_queue[recv_queue_size] = partition_id;
        recv_queue_mutex.lock();
        recv_queue_size += 1;
        recv_queue_mutex.unlock();
      });
      current_send_part_id = partition_id;
      for (int step = 0; step < partitions; step++)
      {
        current_send_part_id = (current_send_part_id + 1) % partitions;
        int i = current_send_part_id;
        for (int t_i = 0; t_i < threads; t_i++)
        {
          *thread_state[t_i] = tuned_chunks_dense[i][t_i];
          //          printf("Range %d\t%d\n",tuned_chunks_dense[i][t_i].curr,tuned_chunks_dense[i][t_i].end);
        }

#pragma omp parallel
        {
          int thread_id = omp_get_thread_num();
          int s_i = get_socket_id(thread_id);
          VertexId final_p_v_i = thread_state[thread_id]->end;
          while (true)
          {
            VertexId begin_p_v_i = __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
            if (begin_p_v_i >= final_p_v_i)
              break;
            VertexId end_p_v_i = begin_p_v_i + basic_chunk;
            if (end_p_v_i > final_p_v_i)
            {
              end_p_v_i = final_p_v_i;
            }
            for (VertexId p_v_i = begin_p_v_i; p_v_i < end_p_v_i; p_v_i++)
            {
              VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
              dense_signal(v_i, VertexAdjList<EdgeData>(incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i].index, 
                      incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i + 1].index),thread_id);
            }
          }
          thread_state[thread_id]->status = STEALING;
          for (int t_offset = 1; t_offset < threads; t_offset++)
          {
            int t_i = (thread_id + t_offset) % threads;
            int s_i = get_socket_id(t_i);
            while (thread_state[t_i]->status != STEALING)
            {
              VertexId begin_p_v_i = __sync_fetch_and_add(&thread_state[t_i]->curr, basic_chunk);
              if (begin_p_v_i >= thread_state[t_i]->end)
                break;
              VertexId end_p_v_i = begin_p_v_i + basic_chunk;
              if (end_p_v_i > thread_state[t_i]->end)
              {
                end_p_v_i = thread_state[t_i]->end;
              }
              for (VertexId p_v_i = begin_p_v_i; p_v_i < end_p_v_i; p_v_i++)
              {
                VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
                dense_signal(v_i, VertexAdjList<EdgeData>(incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i].index,
                        incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i + 1].index),thread_id);
              }
            }
          }
        }
#pragma omp parallel for
        for (int t_i = 0; t_i < threads; t_i++)
        {
           flush_local_send_buffer_buffer(t_i, feature_size);
        }
        if (i != partition_id)
        {
          send_queue[send_queue_size] = i;
          send_queue_mutex.lock();
          send_queue_size += 1;
          send_queue_mutex.unlock();
        }
      }
      for (int step = 0; step < partitions; step++)
      {
        while (true)
        {
          recv_queue_mutex.lock();
          bool condition = (recv_queue_size <= step);
          recv_queue_mutex.unlock();
          if (!condition)
            break;
          __asm volatile("pause" ::
                             : "memory");
        }
        int i = recv_queue[step];
        MessageBuffer **used_buffer;
        if (i == partition_id)
        {
          used_buffer = send_buffer[i];
        }
        else
        {
          used_buffer = recv_buffer[i];
        }
        for (int t_i = 0; t_i < threads; t_i++)
        {
          int s_i = get_socket_id(t_i);
          int s_j = get_socket_offset(t_i);
          VertexId partition_size = used_buffer[s_i]->count;
          thread_state[t_i]->curr = partition_size / threads_per_socket / basic_chunk * basic_chunk * s_j;
          thread_state[t_i]->end = partition_size / threads_per_socket / basic_chunk * basic_chunk * (s_j + 1);
          if (s_j == threads_per_socket - 1)
          {
            thread_state[t_i]->end = used_buffer[s_i]->count;
          }
          thread_state[t_i]->status = WORKING;
        }
#pragma omp parallel reduction(+ \
                               : reducer)
        {
          R local_reducer = 0;
          int thread_id = omp_get_thread_num();
          int s_i = get_socket_id(thread_id);
          char* buffer = used_buffer[s_i]->data;
          while (true)
          {
            VertexId b_i = __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
            if (b_i >= thread_state[thread_id]->end)
              break;
            VertexId begin_b_i = b_i;
            VertexId end_b_i = b_i + basic_chunk;
            if (end_b_i > thread_state[thread_id]->end)
            {
              end_b_i = thread_state[thread_id]->end;
            }
            for (b_i = begin_b_i; b_i < end_b_i; b_i++)
            {
              long index= (long)b_i*(sizeof(M)*(feature_size)+sizeof(VertexId));
              VertexId v_i = *((VertexId*)(buffer+index));
              M* msg_data = (M*)(buffer+index+sizeof(VertexId));
              local_reducer += dense_slot(v_i, msg_data);
            }
          }
          thread_state[thread_id]->status = STEALING;
          reducer += local_reducer;
        }
      }
      send_thread.join();
      recv_thread.join();
      delete[] send_queue;
      delete[] recv_queue;
      //      printf("con%d_%d\n",con,partition_offset[1]);
    }

    R global_reducer;
    MPI_Datatype dt = get_mpi_data_type<R>();
    MPI_Allreduce(&reducer, &global_reducer, 1, dt, MPI_SUM, MPI_COMM_WORLD);
    stream_time += MPI_Wtime();
#ifdef PRINT_DEBUG_MESSAGES
    if (partition_id == 0)
    {
      printf("process_edges took %lf (s)\n", stream_time);
    }
#endif
    return global_reducer;
  }  
    template <typename R, typename M>
  R process_edges_backward(
      std::function<void(VertexId, VertexAdjList<EdgeData>,VertexId,VertexId)> dense_signal,
      std::function<R(VertexId, M*)> dense_slot,
      int feature_size,
      Bitmap *active,
      Bitmap *dense_selective = nullptr)
  {
    omp_set_num_threads(threads);
    double stream_time = 0;
    stream_time -= MPI_Wtime();

    for (int t_i = 0; t_i < threads; t_i++)
    {
      local_send_buffer[t_i]->resize(sizeofM<M>(feature_size) * local_send_buffer_limit);
      local_send_buffer[t_i]->count = 0;
    }
    R reducer = 0;

    for (int i = 0; i < partitions; i++)
    {
      for (int s_i = 0; s_i < sockets; s_i++)
      {
        recv_buffer[i][s_i]->resize(sizeofM<M>(feature_size) * owned_vertices * sockets);
        send_buffer[i][s_i]->resize(sizeofM<M>(feature_size) * (partition_offset[i + 1] - partition_offset[i]) * sockets);
        send_buffer[i][s_i]->count = 0;
        recv_buffer[i][s_i]->count = 0;
      }
    } //init the send and receive buffer
    size_t basic_chunk = 64;
    {
      // dense selective bitmap
      if (dense_selective != nullptr && partitions > 1)
      {
        double sync_time = 0;
        sync_time -= get_time();
        std::thread send_thread([&]() {
          for (int step = 1; step < partitions; step++)
          {
            int recipient_id = (partition_id + step) % partitions;
            MPI_Send(dense_selective->data + WORD_OFFSET(partition_offset[partition_id]), owned_vertices / 64, MPI_UNSIGNED_LONG, recipient_id, PassMessage, MPI_COMM_WORLD);
          }
        });
        std::thread recv_thread([&]() {
          for (int step = 1; step < partitions; step++)
          {
            int sender_id = (partition_id - step + partitions) % partitions;
            MPI_Recv(dense_selective->data + WORD_OFFSET(partition_offset[sender_id]), (partition_offset[sender_id + 1] - partition_offset[sender_id]) / 64, MPI_UNSIGNED_LONG, sender_id, PassMessage, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          }
        });
        send_thread.join();
        recv_thread.join();
        MPI_Barrier(MPI_COMM_WORLD); // denseselect fill the bitmap array
        sync_time += get_time();
#ifdef PRINT_DEBUG_MESSAGES
        if (partition_id == 0)
        {
          printf("sync_time = %lf\n", sync_time);
        }
#endif
      }
#ifdef PRINT_DEBUG_MESSAGES
      if (partition_id == 0)
      {
        printf("dense mode\n");
      }
#endif
      int *send_queue = new int[partitions];
      int *recv_queue = new int[partitions];
      volatile int send_queue_size = 0;
      volatile int recv_queue_size = 0;
      std::mutex send_queue_mutex;
      std::mutex recv_queue_mutex;

      std::thread send_thread([&]() {
        for (int step = 0; step < partitions; step++)
        {
          if (step == partitions - 1)
          {
            break;
          }
          while (true)
          {
            send_queue_mutex.lock();
            bool condition = (send_queue_size <= step);
            send_queue_mutex.unlock();
            if (!condition)
              break;
            __asm volatile("pause" ::
                               : "memory");
          }
          int i = send_queue[step];
          for (int s_i = 0; s_i < sockets; s_i++)
          {
            MPI_Send(send_buffer[i][s_i]->data, sizeofM<M>(feature_size) * send_buffer[i][s_i]->count, MPI_CHAR, i, PassMessage, MPI_COMM_WORLD);
          }
        }
      });
      std::thread recv_thread([&]() {
        std::vector<std::thread> threads;
        for (int step = 1; step < partitions; step++)
        {
          int i = (partition_id - step + partitions) % partitions;
          threads.emplace_back([&](int i) {
            for (int s_i = 0; s_i < sockets; s_i++)
            {
              MPI_Status recv_status;
              MPI_Probe(i, PassMessage, MPI_COMM_WORLD, &recv_status);
              MPI_Get_count(&recv_status, MPI_CHAR, &recv_buffer[i][s_i]->count);
              MPI_Recv(recv_buffer[i][s_i]->data, recv_buffer[i][s_i]->count, MPI_CHAR, i, PassMessage, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
              recv_buffer[i][s_i]->count /= sizeofM<M>(feature_size);
            }
          },
                               i);
        }
        for (int step = 1; step < partitions; step++)
        {
          int i = (partition_id - step + partitions) % partitions;
          threads[step - 1].join();
          recv_queue[recv_queue_size] = i;
          recv_queue_mutex.lock();
          recv_queue_size += 1;
          recv_queue_mutex.unlock();
        }
        recv_queue[recv_queue_size] = partition_id;
        recv_queue_mutex.lock();
        recv_queue_size += 1;
        recv_queue_mutex.unlock();
      });
      current_send_part_id = partition_id;
      for (int step = 0; step < partitions; step++)
      {
        current_send_part_id = (current_send_part_id + 1) % partitions;
        int i = current_send_part_id;
        for (int t_i = 0; t_i < threads; t_i++)
        {
          *thread_state[t_i] = tuned_chunks_dense_backward[i][t_i];
        }
#pragma omp parallel
        {
          int thread_id = omp_get_thread_num();
//          printf("PRE threadsId %d\n",thread_id);
          int s_i = get_socket_id(thread_id);
          VertexId final_p_v_i = thread_state[thread_id]->end;
          while (true)
          {
            VertexId begin_p_v_i = __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
            if (begin_p_v_i >= final_p_v_i)
              break;
            VertexId end_p_v_i = begin_p_v_i + basic_chunk;
            if (end_p_v_i > final_p_v_i)
            {
              end_p_v_i = final_p_v_i;
            }
            for (VertexId p_v_i = begin_p_v_i; p_v_i < end_p_v_i; p_v_i++)
            {
              VertexId v_i = compressed_incoming_adj_index_backward[s_i][p_v_i].vertex;
              
              dense_signal(v_i, VertexAdjList<EdgeData>(incoming_adj_list_backward[s_i] +
                      compressed_incoming_adj_index_backward[s_i][p_v_i].index, incoming_adj_list_backward[s_i]
                      + compressed_incoming_adj_index_backward[s_i][p_v_i + 1].index),thread_id,i);
            }
          }
          thread_state[thread_id]->status = STEALING;
          for (int t_offset = 1; t_offset < threads; t_offset++)
          {
            int t_i = (thread_id + t_offset) % threads;
            int s_i = get_socket_id(t_i);
            while (thread_state[t_i]->status != STEALING)
            {
              VertexId begin_p_v_i = __sync_fetch_and_add(&thread_state[t_i]->curr, basic_chunk);
              if (begin_p_v_i >= thread_state[t_i]->end)
                break;
              VertexId end_p_v_i = begin_p_v_i + basic_chunk;
              if (end_p_v_i > thread_state[t_i]->end)
              {
                end_p_v_i = thread_state[t_i]->end;
              }
              for (VertexId p_v_i = begin_p_v_i; p_v_i < end_p_v_i; p_v_i++)
              {
                VertexId v_i = compressed_incoming_adj_index_backward[s_i][p_v_i].vertex;
                dense_signal(v_i, VertexAdjList<EdgeData>(incoming_adj_list_backward[s_i] + 
                        compressed_incoming_adj_index_backward[s_i][p_v_i].index, incoming_adj_list_backward[s_i] + 
                        compressed_incoming_adj_index_backward[s_i][p_v_i + 1].index),thread_id,i);
              }
            }
          }
        }
#pragma omp parallel for
        for (int t_i = 0; t_i < threads; t_i++)
        {
          flush_local_send_buffer_buffer(t_i,feature_size);
        }
        if (i != partition_id)
        {
          send_queue[send_queue_size] = i;
          send_queue_mutex.lock();
          send_queue_size += 1;
          send_queue_mutex.unlock();
        }
      }
      for (int step = 0; step < partitions; step++)
      {
        while (true)
        {
          recv_queue_mutex.lock();
          bool condition = (recv_queue_size <= step);
          recv_queue_mutex.unlock();
          if (!condition)
            break;
          __asm volatile("pause" ::
                             : "memory");
        }
        int i = recv_queue[step];
        MessageBuffer **used_buffer;
        if (i == partition_id)
        {
          used_buffer = send_buffer[i];
        }
        else
        {
          used_buffer = recv_buffer[i];
        }
        for (int t_i = 0; t_i < threads; t_i++)
        {
          int s_i = get_socket_id(t_i);
          int s_j = get_socket_offset(t_i);
          VertexId partition_size = used_buffer[s_i]->count;
          //printf("DEBUG %d\n",partition_size);
          thread_state[t_i]->curr = partition_size / threads_per_socket / basic_chunk * basic_chunk * s_j;
          thread_state[t_i]->end = partition_size / threads_per_socket / basic_chunk * basic_chunk * (s_j + 1);
          if (s_j == threads_per_socket - 1)
          {
            thread_state[t_i]->end = used_buffer[s_i]->count;
          }
          thread_state[t_i]->status = WORKING;
        }
        
        /*TODO openmp does not launches enough threads,
         * and Gemini dose not provides the stealing method as in dense_signal stage
         * we manually re configure enough threads to fix the bug
         */
        
#pragma omp parallel reduction(+ \
                               : reducer)
        {
//        for(int thread_id=0; thread_id<this->threads;thread_id++){//
          R local_reducer = 0;
          int thread_id = omp_get_thread_num();
//          printf("POST threadsId%d\n",thread_id);
          int s_i = get_socket_id(thread_id);
          char* buffer = used_buffer[s_i]->data;
          while (true)
          {
            VertexId b_i = __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
            if (b_i >= thread_state[thread_id]->end)
              break;
            VertexId begin_b_i = b_i;
            VertexId end_b_i = b_i + basic_chunk;
            if (end_b_i > thread_state[thread_id]->end)
            {
              end_b_i = thread_state[thread_id]->end;
            }
            for (b_i = begin_b_i; b_i < end_b_i; b_i++)
            {
              long index= (long)b_i*(sizeof(M)*(feature_size)+sizeof(VertexId));
              VertexId v_i = *((VertexId*)(buffer+index));
              M* msg_data = (M*)(buffer+index+sizeof(VertexId));
              local_reducer += 1.0;dense_slot(v_i, msg_data);
            }
          }
          thread_state[thread_id]->status = STEALING;
          reducer += local_reducer;
        }
       
        
      }
      send_thread.join();
      recv_thread.join();
      delete[] send_queue;
      delete[] recv_queue;
    }

    R global_reducer;
    MPI_Datatype dt = get_mpi_data_type<R>();
    MPI_Allreduce(&reducer, &global_reducer, 1, dt, MPI_SUM, MPI_COMM_WORLD);
    stream_time += MPI_Wtime();
#ifdef PRINT_DEBUG_MESSAGES
    if (partition_id == 0)
    {
      printf("process_edges took %lf (s)\n", stream_time);
    }
#endif
    return global_reducer;
  }

  // process vertices
  template <typename R>
  R process_vertices_G(std::function<R(VertexId)> process, Bitmap *active)
  {
    double stream_time = 0;
    stream_time -= MPI_Wtime();

    R reducer = 0;
    size_t basic_chunk = 64;
    for (int t_i = 0; t_i < threads; t_i++)
    {
      int s_i = get_socket_id(t_i);
      int s_j = get_socket_offset(t_i);
      VertexId partition_size = local_partition_offset[s_i + 1] - local_partition_offset[s_i];
      thread_state[t_i]->curr = local_partition_offset[s_i] + partition_size / threads_per_socket / basic_chunk * basic_chunk * s_j;
      thread_state[t_i]->end = local_partition_offset[s_i] + partition_size / threads_per_socket / basic_chunk * basic_chunk * (s_j + 1);
      if (s_j == threads_per_socket - 1)
      {
        thread_state[t_i]->end = local_partition_offset[s_i + 1];
      }
      thread_state[t_i]->status = WORKING;
    }
#pragma omp parallel reduction(+ \
                               : reducer)
    {
      R local_reducer = 0;
      int thread_id = omp_get_thread_num();
      while (true)
      {
        VertexId v_i = __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
        if (v_i >= thread_state[thread_id]->end)
          break;
        unsigned long word = active->data[WORD_OFFSET(v_i)];
        while (word != 0)
        {
          if (word & 1)
          {
            local_reducer += process(v_i);
          }
          v_i++;
          word = word >> 1;
        }
      }
      thread_state[thread_id]->status = STEALING;
      for (int t_offset = 1; t_offset < threads; t_offset++)
      {
        int t_i = (thread_id + t_offset) % threads;
        while (thread_state[t_i]->status != STEALING)
        {
          VertexId v_i = __sync_fetch_and_add(&thread_state[t_i]->curr, basic_chunk);
          if (v_i >= thread_state[t_i]->end)
            continue;
          unsigned long word = active->data[WORD_OFFSET(v_i)];
          while (word != 0)
          {
            if (word & 1)
            {
              local_reducer += process(v_i);
            }
            v_i++;
            word = word >> 1;
          }
        }
      }
      reducer += local_reducer;
    }
    R global_reducer;
    MPI_Datatype dt = get_mpi_data_type<R>();
    MPI_Allreduce(&reducer, &global_reducer, 1, dt, MPI_SUM, MPI_COMM_WORLD);
    stream_time += MPI_Wtime();
#ifdef PRINT_DEBUG_MESSAGES
    if (partition_id == 0)
    {
      printf("process_vertices took %lf (s)\n", stream_time);
    }
#endif
    return global_reducer;
  }

    // process edges
  template <typename R, typename M>
  R process_edges_G(std::function<void(VertexId)> sparse_signal, std::function<R(VertexId, M, VertexAdjList<EdgeData>)> sparse_slot, std::function<void(VertexId, VertexAdjList<EdgeData>)> dense_signal, std::function<R(VertexId, M)> dense_slot, Bitmap *active, Bitmap *dense_selective = nullptr)
  {
    double stream_time = 0;
    stream_time -= MPI_Wtime();

    for (int t_i = 0; t_i < threads; t_i++)
    {
      local_send_buffer[t_i]->resize(sizeof(MsgUnit<M>) * local_send_buffer_limit);
      local_send_buffer[t_i]->count = 0;
    }
    R reducer = 0;
    EdgeId active_edges = process_vertices<EdgeId>(
        [&](VertexId vtx) {
          return (EdgeId)out_degree[vtx];
        },
        active);
    bool sparse = (false); //active_edges < edges / 20
    if (sparse)
    {
      for (int i = 0; i < partitions; i++)
      {
        for (int s_i = 0; s_i < sockets; s_i++)
        {
          recv_buffer[i][s_i]->resize(sizeof(MsgUnit<M>) * (partition_offset[i + 1] - partition_offset[i]) * sockets);
          send_buffer[i][s_i]->resize(sizeof(MsgUnit<M>) * owned_vertices * sockets);
          send_buffer[i][s_i]->count = 0;
          recv_buffer[i][s_i]->count = 0;
        }
      }
    }
    else
    {
      for (int i = 0; i < partitions; i++)
      {
        for (int s_i = 0; s_i < sockets; s_i++)
        {
          recv_buffer[i][s_i]->resize(sizeof(MsgUnit<M>) * owned_vertices * sockets);
          send_buffer[i][s_i]->resize(sizeof(MsgUnit<M>) * (partition_offset[i + 1] - partition_offset[i]) * sockets);
          send_buffer[i][s_i]->count = 0;
          recv_buffer[i][s_i]->count = 0;
        }
      }
    }
    size_t basic_chunk = 64;
    if (sparse)
    {
#ifdef PRINT_DEBUG_MESSAGES
      if (partition_id == 0)
      {
        printf("sparse mode\n");
      }
#endif
      int *recv_queue = new int[partitions];
      int recv_queue_size = 0;
      std::mutex recv_queue_mutex;

      current_send_part_id = partition_id;
#pragma omp parallel for
      for (VertexId begin_v_i = partition_offset[partition_id]; begin_v_i < partition_offset[partition_id + 1]; begin_v_i += basic_chunk)
      {
        VertexId v_i = begin_v_i;
        unsigned long word = active->data[WORD_OFFSET(v_i)];
        while (word != 0)
        {
          if (word & 1)
          {
            sparse_signal(v_i);
          }
          v_i++;
          word = word >> 1;
        }
      }
#pragma omp parallel for
      for (int t_i = 0; t_i < threads; t_i++)
      {
        flush_local_send_buffer<M>(t_i);
      }
      recv_queue[recv_queue_size] = partition_id;
      recv_queue_mutex.lock();
      recv_queue_size += 1;
      recv_queue_mutex.unlock();
      std::thread send_thread([&]() {
        for (int step = 1; step < partitions; step++)
        {
          int i = (partition_id - step + partitions) % partitions;
          for (int s_i = 0; s_i < sockets; s_i++)
          {
            MPI_Send(send_buffer[partition_id][s_i]->data, sizeof(MsgUnit<M>) * send_buffer[partition_id][s_i]->count, MPI_CHAR, i, PassMessage, MPI_COMM_WORLD);
          }
        }
      });
      std::thread recv_thread([&]() {
        for (int step = 1; step < partitions; step++)
        {
          int i = (partition_id + step) % partitions;
          for (int s_i = 0; s_i < sockets; s_i++)
          {
            MPI_Status recv_status;
            MPI_Probe(i, PassMessage, MPI_COMM_WORLD, &recv_status);
            MPI_Get_count(&recv_status, MPI_CHAR, &recv_buffer[i][s_i]->count);
            MPI_Recv(recv_buffer[i][s_i]->data, recv_buffer[i][s_i]->count, MPI_CHAR, i, PassMessage, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            recv_buffer[i][s_i]->count /= sizeof(MsgUnit<M>);
          }
          recv_queue[recv_queue_size] = i;
          recv_queue_mutex.lock();
          recv_queue_size += 1;
          recv_queue_mutex.unlock();
        }
      });
      for (int step = 0; step < partitions; step++)
      {
        while (true)
        {
          recv_queue_mutex.lock();
          bool condition = (recv_queue_size <= step);
          recv_queue_mutex.unlock();
          if (!condition)
            break;
          __asm volatile("pause" ::
                             : "memory");
        }
        int i = recv_queue[step];
        MessageBuffer **used_buffer;
        if (i == partition_id)
        {
          used_buffer = send_buffer[i];
        }
        else
        {
          used_buffer = recv_buffer[i];
        }
        for (int s_i = 0; s_i < sockets; s_i++)
        {
          MsgUnit<M> *buffer = (MsgUnit<M> *)used_buffer[s_i]->data;
          size_t buffer_size = used_buffer[s_i]->count;
          for (int t_i = 0; t_i < threads; t_i++)
          {
            // int s_i = get_socket_id(t_i);
            int s_j = get_socket_offset(t_i);
            VertexId partition_size = buffer_size;
            thread_state[t_i]->curr = partition_size / threads_per_socket / basic_chunk * basic_chunk * s_j;
            thread_state[t_i]->end = partition_size / threads_per_socket / basic_chunk * basic_chunk * (s_j + 1);
            if (s_j == threads_per_socket - 1)
            {
              thread_state[t_i]->end = buffer_size;
            }
            thread_state[t_i]->status = WORKING;
          }
#pragma omp parallel reduction(+ \
                               : reducer)
          {
            R local_reducer = 0;
            int thread_id = omp_get_thread_num();
            int s_i = get_socket_id(thread_id);
            while (true)
            {
              VertexId b_i = __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
              if (b_i >= thread_state[thread_id]->end)
                break;
              VertexId begin_b_i = b_i;
              VertexId end_b_i = b_i + basic_chunk;
              if (end_b_i > thread_state[thread_id]->end)
              {
                end_b_i = thread_state[thread_id]->end;
              }
              for (b_i = begin_b_i; b_i < end_b_i; b_i++)
              {
                VertexId v_i = buffer[b_i].vertex;
                M msg_data = buffer[b_i].msg_data;
                if (outgoing_adj_bitmap[s_i]->get_bit(v_i))
                {
                  local_reducer += sparse_slot(v_i, msg_data, VertexAdjList<EdgeData>(outgoing_adj_list[s_i] + outgoing_adj_index[s_i][v_i], outgoing_adj_list[s_i] + outgoing_adj_index[s_i][v_i + 1]));
                }
              }
            }
            thread_state[thread_id]->status = STEALING;
            for (int t_offset = 1; t_offset < threads; t_offset++)
            {
              int t_i = (thread_id + t_offset) % threads;
              if (thread_state[t_i]->status == STEALING)
                continue;
              while (true)
              {
                VertexId b_i = __sync_fetch_and_add(&thread_state[t_i]->curr, basic_chunk);
                if (b_i >= thread_state[t_i]->end)
                  break;
                VertexId begin_b_i = b_i;
                VertexId end_b_i = b_i + basic_chunk;
                if (end_b_i > thread_state[t_i]->end)
                {
                  end_b_i = thread_state[t_i]->end;
                }
                int s_i = get_socket_id(t_i);
                for (b_i = begin_b_i; b_i < end_b_i; b_i++)
                {
                  VertexId v_i = buffer[b_i].vertex;
                  M msg_data = buffer[b_i].msg_data;
                  if (outgoing_adj_bitmap[s_i]->get_bit(v_i))
                  {
                    local_reducer += sparse_slot(v_i, msg_data, VertexAdjList<EdgeData>(outgoing_adj_list[s_i] + outgoing_adj_index[s_i][v_i], outgoing_adj_list[s_i] + outgoing_adj_index[s_i][v_i + 1]));
                  }
                }
              }
            }
            reducer += local_reducer;
          }
        }
      }
      send_thread.join();
      recv_thread.join();
      delete[] recv_queue;
    }
    else
    {
      // dense selective bitmap
      if (dense_selective != nullptr && partitions > 1)
      {
        double sync_time = 0;
        sync_time -= get_time();
        std::thread send_thread([&]() {
          for (int step = 1; step < partitions; step++)
          {
            int recipient_id = (partition_id + step) % partitions;
            MPI_Send(dense_selective->data + WORD_OFFSET(partition_offset[partition_id]), owned_vertices / 64, MPI_UNSIGNED_LONG, recipient_id, PassMessage, MPI_COMM_WORLD);
          }
        });
        std::thread recv_thread([&]() {
          for (int step = 1; step < partitions; step++)
          {
            int sender_id = (partition_id - step + partitions) % partitions;
            MPI_Recv(dense_selective->data + WORD_OFFSET(partition_offset[sender_id]), (partition_offset[sender_id + 1] - partition_offset[sender_id]) / 64, MPI_UNSIGNED_LONG, sender_id, PassMessage, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          }
        });
        send_thread.join();
        recv_thread.join();
        MPI_Barrier(MPI_COMM_WORLD);
        sync_time += get_time();
#ifdef PRINT_DEBUG_MESSAGES
        if (partition_id == 0)
        {
          printf("sync_time = %lf\n", sync_time);
        }
#endif
      }
#ifdef PRINT_DEBUG_MESSAGES
      if (partition_id == 0)
      {
        printf("dense mode\n");
      }
#endif
      int *send_queue = new int[partitions];
      int *recv_queue = new int[partitions];
      volatile int send_queue_size = 0;
      volatile int recv_queue_size = 0;
      std::mutex send_queue_mutex;
      std::mutex recv_queue_mutex;

      std::thread send_thread([&]() {
        for (int step = 0; step < partitions; step++)
        {
          if (step == partitions - 1)
          {
            break;
          }
          while (true)
          {
            send_queue_mutex.lock();
            bool condition = (send_queue_size <= step);
            send_queue_mutex.unlock();
            if (!condition)
              break;
            __asm volatile("pause" ::
                               : "memory");
          }
          int i = send_queue[step];
          for (int s_i = 0; s_i < sockets; s_i++)
          {
            MPI_Send(send_buffer[i][s_i]->data, sizeof(MsgUnit<M>) * send_buffer[i][s_i]->count, MPI_CHAR, i, PassMessage, MPI_COMM_WORLD);
          }
        }
      });
      std::thread recv_thread([&]() {
        std::vector<std::thread> threads;
        for (int step = 1; step < partitions; step++)
        {
          int i = (partition_id - step + partitions) % partitions;
          threads.emplace_back([&](int i) {
            for (int s_i = 0; s_i < sockets; s_i++)
            {
              MPI_Status recv_status;
              MPI_Probe(i, PassMessage, MPI_COMM_WORLD, &recv_status);
              MPI_Get_count(&recv_status, MPI_CHAR, &recv_buffer[i][s_i]->count);
              MPI_Recv(recv_buffer[i][s_i]->data, recv_buffer[i][s_i]->count, MPI_CHAR, i, PassMessage, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
              recv_buffer[i][s_i]->count /= sizeof(MsgUnit<M>);
            }
          },
                               i);
        }
        for (int step = 1; step < partitions; step++)
        {
          int i = (partition_id - step + partitions) % partitions;
          threads[step - 1].join();
          recv_queue[recv_queue_size] = i;
          recv_queue_mutex.lock();
          recv_queue_size += 1;
          recv_queue_mutex.unlock();
        }
        recv_queue[recv_queue_size] = partition_id;
        recv_queue_mutex.lock();
        recv_queue_size += 1;
        recv_queue_mutex.unlock();
      });
      current_send_part_id = partition_id;
      for (int step = 0; step < partitions; step++)
      {
        current_send_part_id = (current_send_part_id + 1) % partitions;
        int i = current_send_part_id;
        for (int t_i = 0; t_i < threads; t_i++)
        {
          *thread_state[t_i] = tuned_chunks_dense[i][t_i];
        }
#pragma omp parallel
        {
          int thread_id = omp_get_thread_num();
          int s_i = get_socket_id(thread_id);
          VertexId final_p_v_i = thread_state[thread_id]->end;
          while (true)
          {
            VertexId begin_p_v_i = __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
            if (begin_p_v_i >= final_p_v_i)
              break;
            VertexId end_p_v_i = begin_p_v_i + basic_chunk;
            if (end_p_v_i > final_p_v_i)
            {
              end_p_v_i = final_p_v_i;
            }
            for (VertexId p_v_i = begin_p_v_i; p_v_i < end_p_v_i; p_v_i++)
            {
              VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
              dense_signal(v_i, VertexAdjList<EdgeData>(incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i].index, incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i + 1].index));
            }
          }
          thread_state[thread_id]->status = STEALING;
          for (int t_offset = 1; t_offset < threads; t_offset++)
          {
            int t_i = (thread_id + t_offset) % threads;
            int s_i = get_socket_id(t_i);
            while (thread_state[t_i]->status != STEALING)
            {
              VertexId begin_p_v_i = __sync_fetch_and_add(&thread_state[t_i]->curr, basic_chunk);
              if (begin_p_v_i >= thread_state[t_i]->end)
                break;
              VertexId end_p_v_i = begin_p_v_i + basic_chunk;
              if (end_p_v_i > thread_state[t_i]->end)
              {
                end_p_v_i = thread_state[t_i]->end;
              }
              for (VertexId p_v_i = begin_p_v_i; p_v_i < end_p_v_i; p_v_i++)
              {
                VertexId v_i = compressed_incoming_adj_index[s_i][p_v_i].vertex;
                dense_signal(v_i, VertexAdjList<EdgeData>(incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i].index, incoming_adj_list[s_i] + compressed_incoming_adj_index[s_i][p_v_i + 1].index));
              }
            }
          }
        }
#pragma omp parallel for
        for (int t_i = 0; t_i < threads; t_i++)
        {
          flush_local_send_buffer<M>(t_i);
        }
        if (i != partition_id)
        {
          send_queue[send_queue_size] = i;
          send_queue_mutex.lock();
          send_queue_size += 1;
          send_queue_mutex.unlock();
        }
      }
      for (int step = 0; step < partitions; step++)
      {
        while (true)
        {
          recv_queue_mutex.lock();
          bool condition = (recv_queue_size <= step);
          recv_queue_mutex.unlock();
          if (!condition)
            break;
          __asm volatile("pause" ::
                             : "memory");
        }
        int i = recv_queue[step];
        MessageBuffer **used_buffer;
        if (i == partition_id)
        {
          used_buffer = send_buffer[i];
        }
        else
        {
          used_buffer = recv_buffer[i];
        }
        for (int t_i = 0; t_i < threads; t_i++)
        {
          int s_i = get_socket_id(t_i);
          int s_j = get_socket_offset(t_i);
          VertexId partition_size = used_buffer[s_i]->count;
          thread_state[t_i]->curr = partition_size / threads_per_socket / basic_chunk * basic_chunk * s_j;
          thread_state[t_i]->end = partition_size / threads_per_socket / basic_chunk * basic_chunk * (s_j + 1);
          if (s_j == threads_per_socket - 1)
          {
            thread_state[t_i]->end = used_buffer[s_i]->count;
          }
          thread_state[t_i]->status = WORKING;
        }
#pragma omp parallel reduction(+ \
                               : reducer)
        {
          R local_reducer = 0;
          int thread_id = omp_get_thread_num();
          int s_i = get_socket_id(thread_id);
          MsgUnit<M> *buffer = (MsgUnit<M> *)used_buffer[s_i]->data;
          while (true)
          {
            VertexId b_i = __sync_fetch_and_add(&thread_state[thread_id]->curr, basic_chunk);
            if (b_i >= thread_state[thread_id]->end)
              break;
            VertexId begin_b_i = b_i;
            VertexId end_b_i = b_i + basic_chunk;
            if (end_b_i > thread_state[thread_id]->end)
            {
              end_b_i = thread_state[thread_id]->end;
            }
            for (b_i = begin_b_i; b_i < end_b_i; b_i++)
            {
              VertexId v_i = buffer[b_i].vertex;
              M msg_data = buffer[b_i].msg_data;
              local_reducer += dense_slot(v_i, msg_data);
            }
          }
          thread_state[thread_id]->status = STEALING;
          reducer += local_reducer;
        }
      }
      send_thread.join();
      recv_thread.join();
      delete[] send_queue;
      delete[] recv_queue;
    }

    R global_reducer;
    MPI_Datatype dt = get_mpi_data_type<R>();
    MPI_Allreduce(&reducer, &global_reducer, 1, dt, MPI_SUM, MPI_COMM_WORLD);
    stream_time += MPI_Wtime();
#ifdef PRINT_DEBUG_MESSAGES
    if (partition_id == 0)
    {
      printf("process_edges took %lf (s)\n", stream_time);
    }
#endif
    return global_reducer;
  }

  
  
};

#endif

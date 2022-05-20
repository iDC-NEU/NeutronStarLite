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

#ifndef NETWORK_H
#define NETWORK_H

#define BIG_MESSAGE 1 // untested

#include "dep/gemini/mpi.hpp"
#include "dep/gemini/type.hpp"

#include <functional>
#include <malloc.h>
#include <mutex>
#include <numa.h>
#include <omp.h>
#include <stdio.h>
#include <thread>
#include <vector>

enum MessageTag { ShuffleGraph, PassMessage, GatherVertexArray };
enum CommType { Master2Mirror, Mirror2Master };

template <typename MsgData> struct MsgUnit {
  VertexId vertex;
  MsgData msg_data;
} __attribute__((packed));

struct MsgUnit_buffer {
  VertexId vertex;
  int msg_data;
} __attribute__((packed));

struct MessageBuffer {
  size_t capacity;

  // the actual size (i.e. bytes) should be sizeof(element) * count
  int count;

  // deprecated
  // Messagebuffer is orgainized like this
  // unit_1 data_1 unit_2 data_2 unit_3 data_3 ...
  char *data;

  // CUDA pinned memory is faster than the normal one
  // for study purpose, you can refer to this answer
  // https://stackoverflow.com/questions/5736968/why-is-cuda-pinned-memory-so-fast
  bool pinned;

  MessageBuffer();
  void init(int socket_id);
  void resize(size_t new_capacity);
  void resize_pinned(size_t new_capacity);
  int *getMsgUnit(int i, int msg_unit_size);

  template <typename t_v> t_v *getMsgData(int i, int msg_unit_size);

  template <typename t_v>
  void setMsgData(int i, int msg_unit_size, t_v *buffer);

  ~MessageBuffer();
};

class NtsGraphCommunicator {
public:
  void init(VertexId *partition_offset_, VertexId owned_vertices_,
            VertexId partitions_, VertexId sockets_, VertexId threads_,
            VertexId partition_id_, size_t lsbl);

  void release_communicator();

  void init_layer_all(VertexId feature_size_, CommType et, DeviceLocation dl);

  void init_communicator(VertexId feature_size_);

  void init_local_message_buffer();

  void init_message_buffer_master_to_mirror();
  void init_message_buffer_mirror_to_master();

  void init_message_buffer_master_to_mirror_pipe();
  void init_message_buffer_mirror_to_master_pipe();

  inline void set_current_send_partition(VertexId cspi) {
    current_send_part_id = cspi;
  }

  void trigger_one_partition(VertexId partition_id_,
                             bool flush_local_buffer = true);
  void partition_is_ready_for_recv(VertexId partition_id_);

  void achieve_local_message(VertexId current_send_partition_id_);

  void partition_is_ready_for_send(VertexId partition_id_);

  MessageBuffer **recv_one_partition(int &workerId, int step);

  void emit_buffer(VertexId vtx, ValueType *buffer, int f_size);
  void emit_buffer_lock_free(VertexId vtx, ValueType *buffer,
                             VertexId write_index, int f_size);

  void send_mirror_to_master();

  void send_master_to_mirror_no_wait();
  void send_master_to_mirror();

  void recv_master_to_mirror_no_wait();
  void recv_master_to_mirror();

  void recv_mirror_to_master();

  void run_all_master_to_mirror();
  void run_all_master_to_mirror_no_wait();
  void run_all_mirror_to_master();

  void send_master_to_mirror_lock_free_no_wait();
  void run_all_master_to_mirror_lock_free_no_wait();

private:
  void flush_local_send_buffer_buffer(int t_i, int f_size);

  inline int get_socket_id(int thread_id) {
    return thread_id / threads_per_socket;
  }

  inline int get_socket_offset(int thread_id) {
    return thread_id % threads_per_socket;
  }

  inline size_t size_of_msg(int f_size) {
    return sizeof(VertexId) + sizeof(ValueType) * f_size;
  }

  inline size_t elements_of_msg(int f_size) {
    return sizeof(VertexId) / sizeof(ValueType) + f_size;
  }

  VertexId current_send_part_id;
  VertexId threads_per_socket;
  VertexId threads;
  VertexId sockets;
  // number of partitions
  VertexId partitions;
  VertexId feature_size;
  // local partition id
  VertexId partition_id;
  // partition array
  VertexId *partition_offset;
  // number of owned vertices
  VertexId owned_vertices;
  // MessageBuffer* [partitions] [sockets]; numa-aware
  MessageBuffer ***send_buffer;
  // MessageBuffer* [partitions] [sockets]; numa-aware
  MessageBuffer ***recv_buffer;
  // maxium number of messages that local send buffer could cache
  size_t local_send_buffer_limit;
  // MessageBuffer *local_send_buffer[sockets]
  MessageBuffer **local_send_buffer;

  int *send_queue;
  int *recv_queue;
  volatile int send_queue_size;
  volatile int recv_queue_size;
  std::mutex send_queue_mutex;
  std::mutex recv_queue_mutex;
  std::vector<std::thread> send_threads;
  std::vector<std::thread> recv_threads;
  std::thread *Send;
  std::thread *Recv;
};
template <typename t_v> class Network_simple {
public:
  t_v *recv_buffer;
  t_v *buffer;
  int worknum;
  int workid = -1;
  int weight_row = 0;
  int weight_col = 0;

  Network_simple(int weight_row_, int weight_col_) {
    weight_row = weight_row_;
    weight_col = weight_col_;
  }

  void all_reduce_sum(t_v *buffer) {
    MPI_Datatype f_vid_t = get_mpi_data_type<float>();
    MPI_Allreduce(MPI_IN_PLACE, buffer, weight_row * weight_col, MPI_FLOAT,
                  MPI_SUM, MPI_COMM_WORLD);
    // printf("%d sd%f\n", weight_row * weight_col, buffer[3]);
  }

  // broadcast the message from root to all processes of group
  // it's called by all members of group using the same comm, root
  // our is MPI_COMM_WORLD and root
  void broadcast(t_v *buffer) {
    MPI_Datatype f_vid_t = get_mpi_data_type<t_v>();
    MPI_Bcast(buffer, weight_row * weight_col, f_vid_t, 0, MPI_COMM_WORLD);
  }
};

#endif /* NETWORK_H */

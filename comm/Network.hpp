/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   Network.hpp
 * Author: wangqg
 *
 * Created on November 18, 2019, 9:24 AM
 */

#ifndef NETWORK_HPP
#define NETWORK_HPP
#include "core/mpi.hpp"
#include "core/type.hpp"
#include <fcntl.h>
#include <functional>
#include <malloc.h>
#include <mutex>
#include <numa.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
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
  int count; // the actual size (i.e. bytes) should be sizeof(element) * count
  char *data;
  bool pinned;
  MessageBuffer() {
    capacity = 0;
    count = 0;
    data = NULL;
  }
  void init(int socket_id) {
    capacity = 4096;
    count = 0;
    data = (char *)numa_alloc_onnode(capacity, socket_id);
  }
  void resize(size_t new_capacity) {
    if (new_capacity > capacity) {
      char *new_data = NULL;
      new_data = (char *)numa_realloc(data, capacity, new_capacity);
      // printf("alloc success%d  %d\n",new_capacity, new_data != NULL);
      assert(new_data != NULL);
      data = new_data;
      capacity =
          new_capacity; //**********************************************************************8
      pinned = false;
    }
  }
  void resize_pinned(long new_capacity) {
#if CUDA_ENABLE
    if ((new_capacity > capacity)) {
      if (!pinned)
        numa_free(data, capacity);
      else
        cudaFreeHost(data);
      char *new_data = NULL;
      new_data = (char *)cudaMallocPinned(new_capacity);
      // Wassert(new_data!=NULL);
      data = new_data;
      capacity =
          new_capacity; //**********************************************************************8
      pinned = true;
    }
#endif
    if (new_capacity > capacity) {
      char *new_data = NULL;
      new_data = (char *)numa_realloc(data, capacity, new_capacity);
      // printf("alloc success%d  %d\n",new_capacity, new_data != NULL);
      assert(new_data != NULL);
      data = new_data;
      capacity =
          new_capacity; //**********************************************************************8
      pinned = false;
    }
  }
  // template <typename MsgData>
  int *getMsgUnit(int i, int msg_unit_size) {
    (int *)this->data + i *(msg_unit_size + sizeof(MsgUnit_buffer));
  }
  template <typename t_v> t_v *getMsg_Data(int i, int msg_unit_size) {
    (t_v *)this->data + i *(msg_unit_size + sizeof(MsgUnit_buffer)) +
        sizeof(MsgUnit_buffer);
  }
  template <typename t_v>
  void set_Msg_Data(int i, int msg_unit_size, t_v *buffer) {
    memcpy(this->data + i * (msg_unit_size + sizeof(MsgUnit_buffer)) +
               sizeof(MsgUnit_buffer),
           buffer, msg_unit_size);
  }
};

class NtsGraphCommunicator {
public:
  void init(VertexId *partition_offset_, VertexId owned_vertices_,
            VertexId partitions_, VertexId sockets_, VertexId threads_,
            VertexId partition_id_, size_t lsbl) {
    partition_offset = partition_offset, owned_vertices = owned_vertices_;
    partitions = partitions_;
    sockets = sockets_;
    threads = threads_;
    local_send_buffer_limit = lsbl;
    partition_id = partition_id_;
    partition_offset = partition_offset_;
    threads_per_socket = threads / sockets;

    local_send_buffer = new MessageBuffer *[threads];
    for (int t_i = 0; t_i < threads; t_i++) {
      local_send_buffer[t_i] = (MessageBuffer *)numa_alloc_onnode(
          sizeof(MessageBuffer), get_socket_id(t_i));
      local_send_buffer[t_i]->init(get_socket_id(t_i));
    }
    send_buffer = new MessageBuffer **[partitions];
    recv_buffer = new MessageBuffer **[partitions];
    for (int i = 0; i < partitions; i++) {
      send_buffer[i] = new MessageBuffer *[sockets];
      recv_buffer[i] = new MessageBuffer *[sockets];
      for (int s_i = 0; s_i < sockets; s_i++) {
        send_buffer[i][s_i] =
            (MessageBuffer *)numa_alloc_onnode(sizeof(MessageBuffer), s_i);
        send_buffer[i][s_i]->init(s_i);
        recv_buffer[i][s_i] =
            (MessageBuffer *)numa_alloc_onnode(sizeof(MessageBuffer), s_i);
        recv_buffer[i][s_i]->init(s_i);
      }
    }
  }
  void release_communicator() {
    Send->join();
    Recv->join();
    delete[] send_queue;
    delete[] recv_queue;
    delete Send;
    delete Recv;
    send_threads.clear();
    recv_threads.clear();
  }

  void init_layer_all(VertexId feature_size_, CommType et, DeviceLocation dl) {
    init_communicator(feature_size_);
    init_local_message_buffer();
    if (et == Master2Mirror) {
      if (CPU_T == dl) {
        init_message_buffer_master_to_mirror();
      }
#if CUDA_ENABLE
      else if (GPU_T == dl) {
        init_message_buffer_master_to_mirror_pipe();
      }
#endif
      else {
        printf("CUDA disable\n");
        assert(0);
      }
    } else if (Mirror2Master == et) {
      if (CPU_T == dl) {
        init_message_buffer_mirror_to_master();
      }
#if CUDA_ENABLE
      else if (GPU_T == dl) {
        init_message_buffer_mirror_to_master_pipe();
      }
#endif
      else {
        printf("CUDA disable\n");
        assert(0);
      }
    }
  }
  void init_communicator(VertexId feature_size_) {
    send_queue = new int[partitions];
    recv_queue = new int[partitions];
    send_queue_size = 0;
    recv_queue_size = 0;
    feature_size = feature_size_;
  }

  void init_local_message_buffer() {

    for (int t_i = 0; t_i < threads; t_i++) {
      local_send_buffer[t_i]->resize(size_of_msg(feature_size) *
                                     local_send_buffer_limit);
      local_send_buffer[t_i]->count = 0;
    }
  }
  void init_message_buffer_master_to_mirror() {
    for (int i = 0; i < partitions; i++) {
      for (int s_i = 0; s_i < sockets; s_i++) {
        recv_buffer[i][s_i]->resize(
            size_of_msg(feature_size) *
            (partition_offset[i + 1] - partition_offset[i]) * sockets);
        send_buffer[i][s_i]->resize(size_of_msg(feature_size) * owned_vertices *
                                    sockets);
        send_buffer[i][s_i]->count = 0;
        recv_buffer[i][s_i]->count = 0;
      }
    }
  }
  void init_message_buffer_mirror_to_master() {
    // printf("%d %d %d %d %d %d
    // %d\n",owned_vertices,sockets,feature_size,local_send_buffer_limit,partitions,partition_offset[0],partition_offset[1]);
    for (int i = 0; i < partitions; i++) {
      for (int s_i = 0; s_i < sockets; s_i++) {
        recv_buffer[i][s_i]->resize(size_of_msg(feature_size) * owned_vertices *
                                    sockets);
        send_buffer[i][s_i]->resize(
            size_of_msg(feature_size) *
            (partition_offset[i + 1] - partition_offset[i]) * sockets);
        send_buffer[i][s_i]->count = 0;
        recv_buffer[i][s_i]->count = 0;
      }
    }
  }

  void init_message_buffer_master_to_mirror_pipe() {
    for (int i = 0; i < partitions; i++) {
      for (int s_i = 0; s_i < sockets; s_i++) {
        recv_buffer[i][s_i]->resize_pinned(
            size_of_msg(feature_size) *
            (partition_offset[i + 1] - partition_offset[i]) * sockets);
        send_buffer[i][s_i]->resize_pinned(size_of_msg(feature_size) *
                                           owned_vertices * sockets);
        send_buffer[i][s_i]->count = 0;
        recv_buffer[i][s_i]->count = 0;
      }
    }
  }
  void init_message_buffer_mirror_to_master_pipe() {
    // printf("%d %d %d %d %d %d
    // %d\n",owned_vertices,sockets,feature_size,local_send_buffer_limit,partitions,partition_offset[0],partition_offset[1]);
    for (int i = 0; i < partitions; i++) {
      for (int s_i = 0; s_i < sockets; s_i++) {
        recv_buffer[i][s_i]->resize_pinned(size_of_msg(feature_size) *
                                           owned_vertices * sockets);
        send_buffer[i][s_i]->resize_pinned(
            size_of_msg(feature_size) *
            (partition_offset[i + 1] - partition_offset[i]) * sockets);
        send_buffer[i][s_i]->count = 0;
        recv_buffer[i][s_i]->count = 0;
      }
    }
  }

  inline void set_current_send_partition(VertexId cspi) {
    current_send_part_id = cspi;
  }
  void trigger_one_partition(VertexId partition_id_,
                             bool flush_local_buffer = true) {
    if (flush_local_buffer == true) {
      achieve_local_message(partition_id_);
    }
    if (partition_id_ == partition_id) {
      //  printf("local triggered %d\n",partition_id_);
      partition_is_ready_for_recv(partition_id_);
    } else if (partition_id_ != partition_id) {
      // printf("remote triggered %d\n",partition_id_);
      partition_is_ready_for_send(partition_id_);

    } else {
      printf("illegal partition_id(%d)\n", partition_id_);
      exit(0);
    }
  }
  void partition_is_ready_for_recv(VertexId partition_id_) {
    recv_queue[recv_queue_size] = partition_id_;
    recv_queue_mutex.lock();
    recv_queue_size += 1;
    recv_queue_mutex.unlock();
  }

  void achieve_local_message(VertexId current_send_partition_id_) {
    current_send_part_id = current_send_partition_id_;
#pragma omp parallel for
    for (int t_i = 0; t_i < threads; t_i++) {
      flush_local_send_buffer_buffer(t_i, feature_size);
    }
  }
  void partition_is_ready_for_send(VertexId partition_id_) {
    if (partition_id_ != partition_id) {
      send_queue[send_queue_size] = partition_id_;
      send_queue_mutex.lock();
      send_queue_size += 1;
      send_queue_mutex.unlock();
    }
  }

  MessageBuffer **recv_one_partition(int &workerId, int step) {
    while (true) {
      recv_queue_mutex.lock();
      bool condition = (recv_queue_size <= step);
      recv_queue_mutex.unlock();
      if (!condition)
        break;
      __asm volatile("pause" ::: "memory");
    }
    int i = recv_queue[step];
    workerId = i;
    //        printf("DEBUG NTSCOMM%d\n",i);
    if (i == partition_id) {
      return send_buffer[i];
    } else {
      return recv_buffer[i];
    }
  }

  void emit_buffer(VertexId vtx, ValueType *buffer, int f_size) {
    int t_i = omp_get_thread_num();
    char *s_buffer = NULL;
    s_buffer = (char *)local_send_buffer[t_i]->data;

    memcpy(s_buffer + local_send_buffer[t_i]->count * size_of_msg(f_size), &vtx,
           sizeof(VertexId));
    memcpy(s_buffer + local_send_buffer[t_i]->count * size_of_msg(f_size) +
               sizeof(VertexId),
           buffer, sizeof(float) * f_size);
    local_send_buffer[t_i]->count += 1;

    if (local_send_buffer[t_i]->count == local_send_buffer_limit) {
      flush_local_send_buffer_buffer(t_i, f_size);
    }
  }
  void emit_buffer_lock_free(VertexId vtx, ValueType *buffer,
                             VertexId write_index, int f_size) {
    int t_i = omp_get_thread_num();
    char *s_buffer = NULL;
    s_buffer = (char *)local_send_buffer[t_i]->data;

    int s_i = get_socket_id(t_i);
    int pos =
        __sync_fetch_and_add(&send_buffer[current_send_part_id][s_i]->count, 1);
    memcpy(send_buffer[current_send_part_id][s_i]->data +
               (size_of_msg(f_size)) * write_index,
           &vtx, sizeof(VertexId));
    memcpy(send_buffer[current_send_part_id][s_i]->data +
               (size_of_msg(f_size)) * write_index + sizeof(VertexId),
           buffer, sizeof(float) * f_size);
  }

  void send_mirror_to_master() {
    for (int step = 0; step < partitions; step++) {
      if (step == partitions - 1) {
        break;
      }
      while (true) {
        send_queue_mutex.lock();
        bool condition = (send_queue_size <= step);
        send_queue_mutex.unlock();
        if (!condition)
          break;
        __asm volatile("pause" ::: "memory");
      }
      int i = send_queue[step];
      for (int s_i = 0; s_i < sockets; s_i++) {
        MPI_Send(send_buffer[i][s_i]->data,
                 size_of_msg(feature_size) * send_buffer[i][s_i]->count,
                 MPI_CHAR, i, PassMessage, MPI_COMM_WORLD);
      }
    }
  }

  void send_master_to_mirror_no_wait() {
    for (int step = 0; step < partitions; step++) {
      if (step == partitions - 1) {
        break;
      }
      while (true) {
        //  printf("while %d  %d\n",send_queue_size,step);
        send_queue_mutex.lock();
        bool condition = (send_queue_size <= step);
        send_queue_mutex.unlock();
        if (!condition)
          break;
        __asm volatile("pause" ::: "memory");
      }
      int i = send_queue[step];
      //          if(i==partition_id){
      //              printf("continue\n");
      //              continue;
      //          }
      for (int s_i = 0; s_i < sockets;
           s_i++) { // printf("send_success part_id %d\n",partition_id);
        MPI_Send(send_buffer[partition_id][s_i]->data,
                 size_of_msg(feature_size) *
                     send_buffer[partition_id][s_i]->count,
                 MPI_CHAR, i, PassMessage, MPI_COMM_WORLD);
      }
    }
  }
  void send_master_to_mirror() {
    for (int step = 1; step < partitions; step++) {
      int i = (partition_id - step + partitions) % partitions;
      for (int s_i = 0; s_i < sockets; s_i++) { // printf("send_success\n");
        MPI_Send(send_buffer[partition_id][s_i]->data,
                 size_of_msg(feature_size) *
                     send_buffer[partition_id][s_i]->count,
                 MPI_CHAR, i, PassMessage, MPI_COMM_WORLD);
      }
    }
  }

  void recv_master_to_mirror_no_wait() {
    for (int step = 1; step < partitions; step++) {
      int i = (partition_id + step) % partitions;
      recv_threads.emplace_back(
          [&](int i) {
            for (int s_i = 0; s_i < sockets; s_i++) {
              MPI_Status recv_status;
              MPI_Probe(i, PassMessage, MPI_COMM_WORLD, &recv_status);
              MPI_Get_count(&recv_status, MPI_CHAR,
                            &recv_buffer[i][s_i]->count);
              MPI_Recv(recv_buffer[i][s_i]->data, recv_buffer[i][s_i]->count,
                       MPI_CHAR, i, PassMessage, MPI_COMM_WORLD,
                       MPI_STATUS_IGNORE);
              recv_buffer[i][s_i]->count /= size_of_msg(feature_size);
              // printf("recv_success\n");
            }
          },
          i);
    }
    for (int step = 1; step < partitions; step++) {
      int i = (partition_id + step) % partitions;
      recv_threads[step - 1].join();
      recv_queue[recv_queue_size] = i;
      recv_queue_mutex.lock();
      recv_queue_size += 1;
      recv_queue_mutex.unlock();
    }
  }
  void recv_master_to_mirror() {
    for (int step = 1; step < partitions; step++) {
      int i = (partition_id + step) % partitions;
      for (int s_i = 0; s_i < sockets; s_i++) {
        MPI_Status recv_status;
        MPI_Probe(i, PassMessage, MPI_COMM_WORLD, &recv_status);
        MPI_Get_count(&recv_status, MPI_CHAR, &recv_buffer[i][s_i]->count);
        MPI_Recv(recv_buffer[i][s_i]->data, recv_buffer[i][s_i]->count,
                 MPI_CHAR, i, PassMessage, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        recv_buffer[i][s_i]->count /= size_of_msg(feature_size);
        // printf("recv_success\n");
      }
      recv_queue[recv_queue_size] = i;
      recv_queue_mutex.lock();
      recv_queue_size += 1;
      recv_queue_mutex.unlock();
    }
  }
  void recv_mirror_to_master() {
    for (int step = 1; step < partitions; step++) {
      int i = (partition_id - step + partitions) % partitions;
      recv_threads.emplace_back(
          [&](int i) {
            for (int s_i = 0; s_i < sockets; s_i++) {
              MPI_Status recv_status;
              MPI_Probe(i, PassMessage, MPI_COMM_WORLD, &recv_status);
              MPI_Get_count(&recv_status, MPI_CHAR,
                            &recv_buffer[i][s_i]->count);
              MPI_Recv(recv_buffer[i][s_i]->data, recv_buffer[i][s_i]->count,
                       MPI_CHAR, i, PassMessage, MPI_COMM_WORLD,
                       MPI_STATUS_IGNORE);
              recv_buffer[i][s_i]->count /= size_of_msg(feature_size);
            }
          },
          i);
    }
    for (int step = 1; step < partitions; step++) {
      int i = (partition_id - step + partitions) % partitions;
      recv_threads[step - 1].join();
      recv_queue[recv_queue_size] = i;
      recv_queue_mutex.lock();
      recv_queue_size += 1;
      recv_queue_mutex.unlock();
    }
    recv_queue[recv_queue_size] = partition_id;
    recv_queue_mutex.lock();
    recv_queue_size += 1;
    recv_queue_mutex.unlock();
  }
  void run_all_master_to_mirror() {
    Send = new std::thread([&]() { send_master_to_mirror(); });
    Recv = new std::thread([&]() { recv_master_to_mirror(); });
  }
  void run_all_master_to_mirror_no_wait() {
    Send = new std::thread([&]() { send_master_to_mirror_no_wait(); });
    Recv = new std::thread([&]() { recv_master_to_mirror_no_wait(); });
  }
  void run_all_mirror_to_master() {
    Send = new std::thread([&]() { send_mirror_to_master(); });
    Recv = new std::thread([&]() { recv_mirror_to_master(); });
  }

  void send_master_to_mirror_lock_free_no_wait() {
    for (int step = 0; step < partitions; step++) {
      if (step == partitions - 1) {
        break;
      }
      while (true) {
        send_queue_mutex.lock();
        bool condition = (send_queue_size <= step);
        send_queue_mutex.unlock();
        if (!condition)
          break;
        __asm volatile("pause" ::: "memory");
      }
      int i = send_queue[step];
      for (int s_i = 0; s_i < sockets;
           s_i++) { // printf("send_success part_id %d\n",partition_id);
        MPI_Send(send_buffer[i][s_i]->data,
                 size_of_msg(feature_size) * send_buffer[i][s_i]->count,
                 MPI_CHAR, i, PassMessage, MPI_COMM_WORLD);
      }
    }
  }
  void run_all_master_to_mirror_lock_free_no_wait() {
    Send =
        new std::thread([&]() { send_master_to_mirror_lock_free_no_wait(); });
    Recv = new std::thread([&]() { recv_master_to_mirror_no_wait(); });
  }

private:
  void flush_local_send_buffer_buffer(int t_i, int f_size) {
    int s_i = get_socket_id(t_i);
    int pos =
        __sync_fetch_and_add(&send_buffer[current_send_part_id][s_i]->count,
                             local_send_buffer[t_i]->count);
    // if(partition_id==1)
    // printf("sizeofM<float>(f_size)%d %d %d
    // %d\n",size_of_msg(f_size),local_send_buffer_limit,local_send_buffer[t_i]->count,send_buffer[current_send_part_id][s_i]->count);

    if (local_send_buffer[t_i]->count != 0)
      memcpy(send_buffer[current_send_part_id][s_i]->data +
                 (size_of_msg(f_size)) * pos,
             local_send_buffer[t_i]->data,
             (size_of_msg(f_size)) * local_send_buffer[t_i]->count);
    local_send_buffer[t_i]->count = 0;
  }
  inline int get_socket_id(int thread_id) {
    return thread_id / threads_per_socket;
  }

  inline int get_socket_offset(int thread_id) {
    return thread_id % threads_per_socket;
  }

  inline size_t size_of_msg(int f_size) {
    return sizeof(VertexId) + sizeof(ValueType) * f_size;
  }
  VertexId current_send_part_id;
  VertexId threads_per_socket;
  VertexId threads;
  VertexId sockets;
  VertexId partitions;
  VertexId feature_size;
  VertexId partition_id;
  VertexId *partition_offset;
  VertexId owned_vertices;
  MessageBuffer **
      *send_buffer; // MessageBuffer* [partitions] [sockets]; numa-aware
  MessageBuffer **
      *recv_buffer; // MessageBuffer* [partitions] [sockets]; numa-aware
  size_t local_send_buffer_limit;
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
  void broadcast(t_v *buffer) {
    MPI_Datatype f_vid_t = get_mpi_data_type<t_v>();
    MPI_Bcast(buffer, weight_row * weight_col, f_vid_t, 0, MPI_COMM_WORLD);
  }
};

#endif /* NETWORK_HPP */

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
#include <functional>
#include <mutex>
#include <thread>
#include <malloc.h>
#include <numa.h>
#include <omp.h>
#include <stdio.h>

#include "comm/network.h"
#include "dep/gemini/mpi.hpp"
#include "dep/gemini/type.hpp"
#if CUDA_ENABLE
#include "cuda/ntsCUDA.hpp"
#endif

const int DEFAULT_MESSAGEBUFFER_CAPACITY = 4096;

MessageBuffer::MessageBuffer(): 
  capacity(0),
  count(0),
  data(nullptr),
  pinned(false) {
}

// initialize message buffer on the specific node
// default size is 4096
void MessageBuffer::init(int socket_id) {
  capacity = DEFAULT_MESSAGEBUFFER_CAPACITY;
  count = 0;
  data = (char *)numa_alloc_onnode(capacity, socket_id);
}

MessageBuffer::~MessageBuffer() {
  if (!data) {
    return;
  }
#if CUDA_ENABLE
  if (!pinned) {
    numa_free(data, capacity);
  } else {
    cudaFreeHost(data);
  }
#else
  numa_free(data, capacity);
#endif
}

// realloc the buffer with new capacity
// we shouldn't call it on pinned memory
// maybe we should add a assertion here in debug mode
void MessageBuffer::resize(size_t new_capacity) {
  if (new_capacity > capacity) {
    char *new_data = NULL;
    new_data = (char *)numa_realloc(data, capacity, new_capacity);
    assert(new_data != NULL);
    data = new_data;
    capacity = new_capacity; 
    pinned = false;
  }
}

// free the previous buffer and allocate pinned memory
// if cuda is not enabled, then we will fallback to normal resize
void MessageBuffer::resize_pinned(size_t new_capacity) {
  if (new_capacity < capacity) {
    return;
  }
#if CUDA_ENABLE
  if (!pinned) {
    numa_free(data, capacity);
  } else {
    cudaFreeHost(data);
  }
  char *new_data = (char *)cudaMallocPinned(new_capacity);
  data = new_data;
  capacity = new_capacity; 
  pinned = true;
#else
  char *new_data = NULL;
  new_data = (char *)numa_realloc(data, capacity, new_capacity);
  assert(new_data != NULL);
  data = new_data;
  capacity = new_capacity; 
  pinned = false;
#endif
}

/**
 * @brief 
 * the offset of ith MsgUnit is i * (msg_unit_size + sizeof(MsgUnit_buffer))
 * 
 * @param i index
 * @param msg_unit_size size of MsgData
 * @return int* pointer to MsgUnit
 */
int* MessageBuffer::getMsgUnit(int i, int msg_unit_size) {
  (int *)this->data + i * (msg_unit_size + sizeof(MsgUnit_buffer));
}

/**
 * @brief 
 * get corresponding msg data, according to the layout mentioned in Messagebuffer, 
 * we should add another unit offset while trying to get data
 * @tparam t_v message data type
 * @param i index
 * @param msg_unit_size size of MsgData
 * @return t_v* pointer to MsgData
 */
template <typename t_v>
t_v* MessageBuffer::getMsgData(int i, int msg_unit_size) {
  (t_v *)this->data + i * (msg_unit_size + sizeof(MsgUnit_buffer)) +
    sizeof(MsgUnit_buffer);
}

/**
 * @brief 
 * copy msg_unit_size bytes from buffer to MessageBuffer
 * @tparam t_v message data type
 * @param i index
 * @param msg_unit_size size of MsgData
 * @param buffer buffer which holds new data
 */
template <typename t_v>
void MessageBuffer::setMsgData(int i, int msg_unit_size, t_v *buffer) {
  memcpy(this->data + i * (msg_unit_size + sizeof(MsgUnit_buffer)) +
            sizeof(MsgUnit_buffer),
          buffer, msg_unit_size);
}

// NtsGraphCommunicator world

/**
 * @brief 
 * initialize the communicator for NTS
 * 
 * @param partition_offset_ partition offset array
 * @param owned_vertices_ number of owned vertices
 * @param partitions_ number of partitions
 * @param sockets_ number of sockets
 * @param threads_ number of threads
 * @param partition_id_ partition ID
 * @param lsbl local send buffer limit. count for local_send_buffer should less than lsbl
 */
void NtsGraphCommunicator::init(VertexId *partition_offset_, VertexId owned_vertices_,
          VertexId partitions_, VertexId sockets_, VertexId threads_,
          VertexId partition_id_, size_t lsbl) {
  partition_offset = partition_offset_;
  owned_vertices = owned_vertices_;
  partitions = partitions_;
  sockets = sockets_;
  threads = threads_;
  local_send_buffer_limit = lsbl;
  partition_id = partition_id_;
  threads_per_socket = threads / sockets;

  // for every threads, allocate message buffer and initialize it.
  local_send_buffer = new MessageBuffer *[threads];
  for (int t_i = 0; t_i < threads; t_i++) {
    local_send_buffer[t_i] = (MessageBuffer *)numa_alloc_onnode(
        sizeof(MessageBuffer), get_socket_id(t_i));
    local_send_buffer[t_i]->init(get_socket_id(t_i));
  }

  // buffer[partition_num][sockets], for sending and receiving
  // for every socket, it has message buffer for all partitions
  // I wonder the reason why we use [partition_num][socket_num] instead of
  // [socket_num][partition_num]. is it for better locality use when we 
  // are receiving the message for other partition?
  // I think answer is no, since they are allocated in numa-aware form.
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

/**
 * @brief 
 * join all the threads and free the queue.
 * corresponding to init_communicator
 */
void NtsGraphCommunicator::release_communicator() {
  Send->join();
  Recv->join();
  delete[] send_queue;
  delete[] recv_queue;
  delete Send;
  delete Recv;
  send_threads.clear();
  recv_threads.clear();
}

/**
 * @brief 
 * init communicator for this layer. i.e. allocating send_queue, send_buffer etc.
 * @param feature_size_ feature size for this layer
 * @param et communication type, could be Master2Mirror or Mirror2Master
 * @param dl device location. e.g. CPU_T
 */
void NtsGraphCommunicator::init_layer_all(VertexId feature_size_, CommType et, DeviceLocation dl) {
  // allocating queue and resizing message buffer
  init_communicator(feature_size_);
  init_local_message_buffer();
  // initialize message buffer according to et and dl
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

/**
 * @brief 
 * initialize send_queue and recv_queue
 * @param feature_size_ feature size for this layer
 */
void NtsGraphCommunicator::init_communicator(VertexId feature_size_) {
  send_queue = new int[partitions];
  recv_queue = new int[partitions];
  send_queue_size = 0;
  recv_queue_size = 0;
  feature_size = feature_size_;
}

/**
 * @brief 
 * initialize local send buffer, resize the message buffer
 * to the feature_size of this layer
 */
void NtsGraphCommunicator::init_local_message_buffer() {

  for (int t_i = 0; t_i < threads; t_i++) {
    local_send_buffer[t_i]->resize(size_of_msg(feature_size) *
                                    local_send_buffer_limit);
    local_send_buffer[t_i]->count = 0;
  }
}

/**
 * @brief 
 * initialize message buffer for master2mirror direction.
 * buffer will be placed on CPU.
 */
void NtsGraphCommunicator::init_message_buffer_master_to_mirror() {
  for (int i = 0; i < partitions; i++) {
    for (int s_i = 0; s_i < sockets; s_i++) {
      // since we will also send and recv message locally, so we don't use find-grained buffer 
      // which might introduce more code and increase complexity
      // for every recv buffer, resize to feature_size * (vertex_num for this parittion) * sockets
      recv_buffer[i][s_i]->resize(
          size_of_msg(feature_size) *
          (partition_offset[i + 1] - partition_offset[i]) * sockets);
      // for every send buffer, resize to feature_size * owned_vertex_num * sockets
      send_buffer[i][s_i]->resize(size_of_msg(feature_size) * owned_vertices *
                                  sockets);
      send_buffer[i][s_i]->count = 0;
      recv_buffer[i][s_i]->count = 0;
    }
  }
}

/**
 * @brief 
 * initialize message buffer for mirror2master direction.
 * buffer will be placed on CPU.
 */
void NtsGraphCommunicator::init_message_buffer_mirror_to_master() {
  // printf("%d %d %d %d %d %d
  // %d\n",owned_vertices,sockets,feature_size,local_send_buffer_limit,partitions,partition_offset[0],partition_offset[1]);
  for (int i = 0; i < partitions; i++) {
    for (int s_i = 0; s_i < sockets; s_i++) {
      // for every recv buffer, we are the master, and thus, using owned_vertex_num
      recv_buffer[i][s_i]->resize(size_of_msg(feature_size) * owned_vertices *
                                  sockets);
      // for sending, buffer size should be the owned_vertex_num for that partition
      send_buffer[i][s_i]->resize(
          size_of_msg(feature_size) *
          (partition_offset[i + 1] - partition_offset[i]) * sockets);
      send_buffer[i][s_i]->count = 0;
      recv_buffer[i][s_i]->count = 0;
    }
  }
}

/**
 * @brief 
 * initialize message buffer for master2mirror direction.
 * buffer will be pinned on GPU.
 */
void NtsGraphCommunicator::init_message_buffer_master_to_mirror_pipe() {
  // logic here will be the same at CPU version.
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

/**
 * @brief 
 * initialize message buffer for mirror2master direction.
 * buffer will be pinned on GPU.
 */
void NtsGraphCommunicator::init_message_buffer_mirror_to_master_pipe() {
  // printf("%d %d %d %d %d %d
  // %d\n",owned_vertices,sockets,feature_size,local_send_buffer_limit,partitions,partition_offset[0],partition_offset[1]);
  // logic here will be the same at CPU version.
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

/**
 * @brief 
 * notify that the partition is ready to send. we should call it 
 * after we place data to send_buffer
 * @param partition_id_ partition id
 * @param flush_local_buffer whether we need to flush local buffer to send buffer
 */
void NtsGraphCommunicator::trigger_one_partition(VertexId partition_id_,
                            bool flush_local_buffer) {
  if (flush_local_buffer == true) {
    achieve_local_message(partition_id_);
  }
  if (partition_id_ == partition_id) {
    //  printf("local triggered %d\n",partition_id_);
    partition_is_ready_for_recv(partition_id_);
  } else {
    // printf("remote triggered %d\n",partition_id_);
    partition_is_ready_for_send(partition_id_);
  }
}

/**
 * @brief 
 * ready to recv specific parititon.
 * Used for recv local message
 * @param partition_id_ partition id
 */
void NtsGraphCommunicator::partition_is_ready_for_recv(VertexId partition_id_) {
  recv_queue[recv_queue_size] = partition_id_;
  recv_queue_mutex.lock();
  recv_queue_size += 1;
  recv_queue_mutex.unlock();
}

/**
 * @brief 
 * flush local send buffer to send buffer for current_send_partition
 * @param current_send_partition_id_ send partition id
 */
void NtsGraphCommunicator::achieve_local_message(VertexId current_send_partition_id_) {
  current_send_part_id = current_send_partition_id_;
#pragma omp parallel for
  for (int t_i = 0; t_i < threads; t_i++) {
    flush_local_send_buffer_buffer(t_i, feature_size);
  }
}

/**
 * @brief 
 * ready to send specific partition.
 * We will push this partition to send_queue
 * @param partition_id_ partition id
 */
void NtsGraphCommunicator::partition_is_ready_for_send(VertexId partition_id_) {
  if (partition_id_ != partition_id) {
    send_queue[send_queue_size] = partition_id_;
    send_queue_mutex.lock();
    send_queue_size += 1;
    send_queue_mutex.unlock();
  }
}

/**
 * @brief 
 * recv the message from one partition in the specific step.
 * @param workerId partitionID where we received message in this particular step
 * @param step the step which specifies the partitionID. think about the steps in ring scheduling 
 * @return MessageBuffer** message buffer for all sockets from one partition
 */
MessageBuffer **NtsGraphCommunicator::recv_one_partition(int &workerId, int step) {
  // for this type looping, please refer to this answer
  // https://stackoverflow.com/questions/50428450/what-does-asm-volatile-pause-memory-do
  // the equivalent of c++ style loop is the combination of atmoic load and _mm_pause
  // wait till this partiiton is avaliable
  // the reason we use spinlock is because the cv is heavy-weight
  while (true) {
    recv_queue_mutex.lock();
    bool condition = (recv_queue_size <= step);
    recv_queue_mutex.unlock();
    if (!condition)
      break;
    __asm volatile("pause" ::: "memory");
  }
  // assign the partition id
  int i = recv_queue[step];
  workerId = i;

  // if partition is local partition. we return the send_buffer directly
  if (i == partition_id) {
    return send_buffer[i];
  } else {
    return recv_buffer[i];
  }
}

/**
 * @brief 
 * place message(vertex and data) to local send buffer, and flush the buffer
 * if we reach the limit
 * @param vtx source vertex
 * @param buffer vertex feature data buffer
 * @param f_size feature size
 */
void NtsGraphCommunicator::emit_buffer(VertexId vtx, ValueType *buffer, int f_size) {
  // get current thread id
  int t_i = omp_get_thread_num();
  char *s_buffer = NULL;
  s_buffer = (char *)local_send_buffer[t_i]->data;

  // copy the vertex to send buffer
  memcpy(s_buffer + local_send_buffer[t_i]->count * size_of_msg(f_size), &vtx,
          sizeof(VertexId));
  // copy the feature data to send buffer
  memcpy(s_buffer + local_send_buffer[t_i]->count * size_of_msg(f_size) +
              sizeof(VertexId),
          buffer, sizeof(float) * f_size);
  // question, is this operation safe? shall we use atomic operation?
  local_send_buffer[t_i]->count += 1;

  if (local_send_buffer[t_i]->count == local_send_buffer_limit) {
    flush_local_send_buffer_buffer(t_i, f_size);
  }
}

/**
 * @brief 
 * place the data directly on send buffer based on vertexID instead of local send buffer
 * @param vtx source vertex
 * @param buffer vertex feature data buffer 
 * @param write_index index where we want to place data to
 * @param f_size feature size
 */
void NtsGraphCommunicator::emit_buffer_lock_free(VertexId vtx, ValueType *buffer,
                            VertexId write_index, int f_size) {
  int t_i = omp_get_thread_num();

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

/**
 * @brief 
 * send data from mirror to master in all steps
 */
void NtsGraphCommunicator::send_mirror_to_master() {
  for (int step = 0; step < partitions; step++) {
    if (step == partitions - 1) {
      break;
    }
    // waiting for the data of current step to be prepared
    while (true) {
      send_queue_mutex.lock();
      bool condition = (send_queue_size <= step);
      send_queue_mutex.unlock();
      if (!condition)
        break;
      __asm volatile("pause" ::: "memory");
    }
    // send to partition i
    int i = send_queue[step];
    for (int s_i = 0; s_i < sockets; s_i++) {
#if BIG_MESSAGE       
/*Message less than 8GB*/
      float* send_msg=(float*)send_buffer[i][s_i]->data;
      MPI_Send(send_msg,
                elements_of_msg(feature_size) * send_buffer[i][s_i]->count,
                MPI_FLOAT, i, PassMessage, MPI_COMM_WORLD);
//      printf("MI-MA SEND 8GB\n");
#else
/*Message less than 2GB*/        
      MPI_Send(send_buffer[i][s_i]->data,
                size_of_msg(feature_size) * send_buffer[i][s_i]->count,
                MPI_CHAR, i, PassMessage, MPI_COMM_WORLD);  
#endif  
    }
  }
}

/**
 * @brief 
 * send data from master to mirror in all steps.
 * we will retrieve partition id from send_queue
 */
void NtsGraphCommunicator::send_master_to_mirror_no_wait() {
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

    // local vertex data will be placed on send_buffer[partition_id]
    // so we just send it to all peers
    for (int s_i = 0; s_i < sockets; s_i++) { 
#if BIG_MESSAGE         
/*Message less than 8GB*/
      float* send_msg=(float*)send_buffer[partition_id][s_i]->data;
      MPI_Send(send_msg, elements_of_msg(feature_size) *
                    send_buffer[partition_id][s_i]->count,
                MPI_FLOAT, i, PassMessage, MPI_COMM_WORLD);
//      printf("MA-MI SEND 8GB\n");
#else
/*Message less than 2GB*/   
      MPI_Send(send_buffer[partition_id][s_i]->data,
                size_of_msg(feature_size) *
                    send_buffer[partition_id][s_i]->count,
                MPI_CHAR, i, PassMessage, MPI_COMM_WORLD);
#endif      
      
/*End*/           
    }
  }
}

/**
 * @brief 
 * send data from master to mirror.
 * we will not retrive target partition ID from send_queue, instead we send it
 * in a ring style.
 */
void NtsGraphCommunicator::send_master_to_mirror() {
// for message less than 2GB    
  for (int step = 1; step < partitions; step++) {
    int i = (partition_id - step + partitions) % partitions;
    for (int s_i = 0; s_i < sockets; s_i++) { // printf("send_success\n");
      MPI_Send(send_buffer[partition_id][s_i]->data,
                size_of_msg(feature_size) *
                    send_buffer[partition_id][s_i]->count,
                MPI_CHAR, i, PassMessage, MPI_COMM_WORLD);
    }
  }
// for message less than 8GB    
//  for (int step = 1; step < partitions; step++) {
//    int i = (partition_id - step + partitions) % partitions;
//    for (int s_i = 0; s_i < sockets; s_i++) { // printf("send_success\n");
//        float* send_msg=(float*)send_buffer[partition_id][s_i]->data;
//      MPI_Send(send_msg, elements_of_msg(feature_size) *
//                    send_buffer[partition_id][s_i]->count,
//                MPI_FLOAT, i, PassMessage, MPI_COMM_WORLD);
//    }
//  }
}

/**
 * @brief 
 * spawn threads waiting to recv message from master.
 * push the partitionID after we've received the message
 */
void NtsGraphCommunicator::recv_master_to_mirror_no_wait() {
  for (int step = 1; step < partitions; step++) {
    int i = (partition_id + step) % partitions;
    recv_threads.emplace_back(
        [&](int i) {
          for (int s_i = 0; s_i < sockets; s_i++) {
            MPI_Status recv_status;
            MPI_Probe(i, PassMessage, MPI_COMM_WORLD, &recv_status);
#if BIG_MESSAGE            
/*Message less than 8GB*/
            float* recv_msg=(float*)recv_buffer[i][s_i]->data;
            MPI_Get_count(&recv_status, MPI_FLOAT,
                          &recv_buffer[i][s_i]->count);
            MPI_Recv(recv_msg, recv_buffer[i][s_i]->count,
                      MPI_FLOAT, i, PassMessage, MPI_COMM_WORLD,
                      MPI_STATUS_IGNORE);
            recv_buffer[i][s_i]->count /= elements_of_msg(feature_size);
//            printf("MA-MI RECV 8GB\n");
#else
/*Message less than 2GB*/            
            MPI_Get_count(&recv_status, MPI_CHAR,
                          &recv_buffer[i][s_i]->count);
            MPI_Recv(recv_buffer[i][s_i]->data, recv_buffer[i][s_i]->count,
                      MPI_CHAR, i, PassMessage, MPI_COMM_WORLD,
                      MPI_STATUS_IGNORE);
            recv_buffer[i][s_i]->count /= size_of_msg(feature_size);            
#endif            
          }
        },
        i);
  }
  // join the recv threads according to the step.
  // push the new partition ID to recv_queue to indicate 
  // that the message from this partition has prepared.
  for (int step = 1; step < partitions; step++) {
    int i = (partition_id + step) % partitions;
    recv_threads[step - 1].join();
    recv_queue[recv_queue_size] = i;
    recv_queue_mutex.lock();
    recv_queue_size += 1;
    recv_queue_mutex.unlock();
  }
}

/**
 * @brief 
 * recv the message from master to mirror. 
 * instead of spawning thread waiting for the message, we do it in
 * a blocked manner.
 */
void NtsGraphCommunicator::recv_master_to_mirror() {
  for (int step = 1; step < partitions; step++) {
    int i = (partition_id + step) % partitions;
    for (int s_i = 0; s_i < sockets; s_i++) {
      MPI_Status recv_status;
      MPI_Probe(i, PassMessage, MPI_COMM_WORLD, &recv_status);
//    for messages less than 2GB      
      MPI_Get_count(&recv_status, MPI_CHAR, &recv_buffer[i][s_i]->count);
      MPI_Recv(recv_buffer[i][s_i]->data, recv_buffer[i][s_i]->count,
                MPI_CHAR, i, PassMessage, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      recv_buffer[i][s_i]->count /= size_of_msg(feature_size);
      
//    for messages less than 8GB 
//      float* recv_msg=(float*)recv_buffer[i][s_i]->data;
//      MPI_Get_count(&recv_status, MPI_FLOAT, &recv_buffer[i][s_i]->count);
//      MPI_Recv(recv_msg, recv_buffer[i][s_i]->count,
//                MPI_FLOAT, i, PassMessage, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//      recv_buffer[i][s_i]->count /= elements_of_msg(feature_size);      
      
    }
    recv_queue[recv_queue_size] = i;
    recv_queue_mutex.lock();
    recv_queue_size += 1;
    recv_queue_mutex.unlock();
  }
}

/**
 * @brief 
 * spawn threads waiting to recv the message from mirror.
 * And push partition ID into recv_queue to indicate the corresponding partition is ready
 */
void NtsGraphCommunicator::recv_mirror_to_master() {
  for (int step = 1; step < partitions; step++) {
    int i = (partition_id - step + partitions) % partitions;
    recv_threads.emplace_back(
        [&](int i) {
          for (int s_i = 0; s_i < sockets; s_i++) {
            MPI_Status recv_status;
            MPI_Probe(i, PassMessage, MPI_COMM_WORLD, &recv_status);
#if BIG_MESSAGE            
/*Message less than 8GB*/
            float* recv_msg=(float*)recv_buffer[i][s_i]->data;
            MPI_Get_count(&recv_status, MPI_FLOAT,
                          &recv_buffer[i][s_i]->count);
            MPI_Recv(recv_msg, recv_buffer[i][s_i]->count,
                      MPI_FLOAT, i, PassMessage, MPI_COMM_WORLD,
                      MPI_STATUS_IGNORE);
            recv_buffer[i][s_i]->count /= elements_of_msg(feature_size);
//            printf("MI-MA RECV 8GB\n");
#else
/*Message less than 2GB*/            
            MPI_Get_count(&recv_status, MPI_CHAR,
                          &recv_buffer[i][s_i]->count);
            MPI_Recv(recv_buffer[i][s_i]->data, recv_buffer[i][s_i]->count,
                      MPI_CHAR, i, PassMessage, MPI_COMM_WORLD,
                      MPI_STATUS_IGNORE);
            recv_buffer[i][s_i]->count /= size_of_msg(feature_size);            
#endif            
/*End*/
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
  // last partition is local partition. we will retrive messages from send buffer directly
  recv_queue[recv_queue_size] = partition_id;
  recv_queue_mutex.lock();
  recv_queue_size += 1;
  recv_queue_mutex.unlock();
}

void NtsGraphCommunicator::run_all_master_to_mirror() {
  Send = new std::thread([&]() { send_master_to_mirror(); });
  Recv = new std::thread([&]() { recv_master_to_mirror(); });
}

void NtsGraphCommunicator::run_all_master_to_mirror_no_wait() {
  Send = new std::thread([&]() { send_master_to_mirror_no_wait(); });
  Recv = new std::thread([&]() { recv_master_to_mirror_no_wait(); });
}

void NtsGraphCommunicator::run_all_mirror_to_master() {
  Send = new std::thread([&]() { send_mirror_to_master(); });
  Recv = new std::thread([&]() { recv_mirror_to_master(); });
}

/**
 * @brief 
 * send message from master to mirror.
 * the difference between this function and send_master_to_mirror_no_wait
 * is we will send different data to different partition, instead of 
 * sending message all from send_buffer[partition_id]
 */
void NtsGraphCommunicator::send_master_to_mirror_lock_free_no_wait() {
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

void NtsGraphCommunicator::run_all_master_to_mirror_lock_free_no_wait() {
  Send =
      new std::thread([&]() { send_master_to_mirror_lock_free_no_wait(); });
  Recv = new std::thread([&]() { recv_master_to_mirror_no_wait(); });
}

/**
 * @brief 
 * flush local send buffer to the send buffer
 * @param t_i thread id
 * @param f_size feature size for this layer
 */
void NtsGraphCommunicator::flush_local_send_buffer_buffer(int t_i, int f_size) {
  int s_i = get_socket_id(t_i);
  // get the previous send_buffer count and add local_send_buffer count to it
  int pos =
      __sync_fetch_and_add(&send_buffer[current_send_part_id][s_i]->count,
                            local_send_buffer[t_i]->count);
  // if(partition_id==1)
  // printf("sizeofM<float>(f_size)%d %d %d
  // %d\n",size_of_msg(f_size),local_send_buffer_limit,local_send_buffer[t_i]->count,send_buffer[current_send_part_id][s_i]->count);

  // copy data from local_send_buffer to send_buffer
  if (local_send_buffer[t_i]->count != 0)
    memcpy(send_buffer[current_send_part_id][s_i]->data +
                (size_of_msg(f_size)) * pos,
            local_send_buffer[t_i]->data,
            (size_of_msg(f_size)) * local_send_buffer[t_i]->count);
  // clear local_send_buffer
  local_send_buffer[t_i]->count = 0;
}
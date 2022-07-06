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
#ifndef NTSBASEOP_HPP
#define NTSBASEOP_HPP
#include "core/graph.hpp"
#include "core/PartitionedGraph.hpp"
#include <immintrin.h>
namespace nts {
namespace op {

class ntsGraphOp {
public:
  Graph<Empty> *graph_;
  VertexSubset *active_;
    PartitionedGraph *partitioned_graph_;
  ntsGraphOp() {}
  ntsGraphOp(PartitionedGraph *partitioned_graph,VertexSubset *active) {
    graph_ = partitioned_graph->graph_;
    partitioned_graph_=partitioned_graph;
    active_ = active;
  }
  ntsGraphOp(Graph<Empty> *graph) {
    graph_ = graph;
  }
  virtual NtsVar forward(NtsVar &f_input)=0;
  virtual NtsVar forward(NtsVar &f_input,NtsVar &f_input1){
      LOG_INFO("forward(x,w) is not implemented");
      assert(0);
  }
  virtual NtsVar backward(NtsVar &output_grad)=0;
  virtual NtsVar get_additional_grad(){
      LOG_INFO("get_additional_grad is not implemented");
      assert(0);
  }
};


class ntsNNBaseOp {
public:
  ntsNNBaseOp(){}
  ntsNNBaseOp(int layer_){
  layer=layer_;}
  NtsVar *f_input;
  NtsVar *f_output; 
  int layer=-1;
  virtual NtsVar forward(NtsVar &f_input)=0;
  virtual NtsVar backward(NtsVar &output_grad)=0;
  
};


inline void nts_comp_non_avx256(ValueType *output, ValueType *input, ValueType weight,
          int feat_size) {
  for (int i = 0; i < feat_size; i++) {
    output[i] += input[i] * weight;
  }
}

inline ValueType dot_product(ValueType *a, ValueType *b,
          int feat_size) {
    ValueType c=0.0;
    for (int i = 0; i < feat_size; i++) {
        c += a[i] * b[i];
    }
    return c;
}

//avx256
inline void nts_comp(ValueType *output, ValueType *input, ValueType weight,
          int feat_size) { 
#ifdef __AVX__  // support AVX   
  // printf("use avx version nts_comp\n");
  const int LEN=8;
  int loop=feat_size/LEN;
  int res=feat_size%LEN;
  __m256 w=_mm256_broadcast_ss(reinterpret_cast<float const *>(&weight));
  for(int i=0;i<loop;i++){
    __m256 source= _mm256_loadu_ps(reinterpret_cast<float const *>(&(input[i*LEN])));
    __m256 destination= _mm256_loadu_ps(reinterpret_cast<float const *>(&(output[i*LEN])));
    _mm256_storeu_ps(&(output[i*LEN]),_mm256_add_ps(_mm256_mul_ps(source,w),destination));
  }
  for (int i = LEN*loop; i < feat_size; i++) {
    output[i] += input[i] * weight;
  }
#else // not support AVX
  // printf("use normal version nts_comp\n");
  for (int i = 0; i < feat_size; i++) {
    output[i] += input[i] * weight;
  }
#endif
}


/**
 * @brief
 * do output += input at feature(array) level
 * @param input input feature
 * @param output output feature
 * @param feat_size feature size
 */
inline void nts_acc(ValueType *output, ValueType *input, int feat_size) {
  for (int i = 0; i < feat_size; i++) {
    // atomic add
    write_add(&output[i], input[i]);
  }
}

inline void nts_acc(ValueType *output, ValueType *input,ValueType weight, int feat_size) {
  for (int i = 0; i < feat_size; i++) {
    // atomic add
    write_add(&output[i], input[i]*weight);
  }
}

/**
 * @brief
 * do output += input at feature(array) level
 * @param input input feature
 * @param output output feature
 * @param feat_size feature size
 */
inline void nts_min(ValueType *output, ValueType *input, VertexId *record, int feat_size, VertexId e_index) {
  for (int i = 0; i < feat_size; i++) {
    // atomic add
    if(write_min(&output[i], input[i])){
        record[i]=e_index;
    }
  }
}

/**
 * @brief
 * do output += input at feature(array) level
 * @param input input feature
 * @param output output feature
 * @param feat_size feature size
 */
inline void nts_max(ValueType *output, ValueType *input, VertexId *record, int feat_size, VertexId e_index) {
  for (int i = 0; i < feat_size; i++) {
    // atomic add
    if(write_max(&output[i], input[i])){
        record[i]=e_index;
    }
  }
}

inline void nts_assign(ValueType *message, ValueType *feature, VertexId* record,
          int feat_size) {
    for(int i=0;i<feat_size;i++){
        message[(record[i]*feat_size)+i]=feature[i];
    }
}
/**
 * @brief
 * copy feature_size elements from b_src[d_offset * feature_size]
 * to d_dst[s_offset * feature_size]
 * @param b_dst dst buffer
 * @param d_offset dst offset, should be a vertex id
 * @param b_src src buffer
 * @param s_offset src offset, should be a vertex id
 * @param feat_size feature size that every vertex have
 */
inline void nts_copy(ValueType *b_dst, long d_offset, ValueType *b_src, VertexId s_offset,
          int feat_size, int counts) {
  // length is the byte level space cost for a vertex feature data
  VertexId length = sizeof(ValueType) * feat_size;
  // LOG_INFO("length %d feat_size %d d_offset %d s_offset
  // %d\n",length,feat_size,d_offset,s_offset);
  memcpy((char *)b_dst + d_offset * length, (char *)b_src + s_offset * length,
         length*counts);
}

/**
 * @brief
 * return 1 / sqrt(out_degree[src] * in_degree[dst]).
 * normally used as edge weight
 * @param src src id
 * @param dst dst id
 * @return ValueType
 */
inline ValueType nts_norm_degree(Graph<Empty> *graph_, VertexId src, VertexId dst) {
  return 1 / ((ValueType)std::sqrt(graph_->out_degree_for_backward[src]) *
              (ValueType)std::sqrt(graph_->in_degree_for_backward[dst]));
}

/**
 * @brief
 * get out degree for v
 * @param v vertex id
 * @return ValueType
 */
inline ValueType nts_out_degree(Graph<Empty> *graph_, VertexId v) {
  return (ValueType)(graph_->out_degree_for_backward[v]);
}

/**
 * @brief
 * get in degree for v
 * @param v vertex id
 * @return ValueType
 */
inline ValueType nts_in_degree(Graph<Empty> *graph_, VertexId v) {
  return (ValueType)(graph_->in_degree_for_backward[v]);
}

} // namespace graphop
} // namespace nts



//class ntsNNOp {
//public:
//
//  ntsNNOp() {}
//  virtual NtsVar &forward(NtsVar &input) = 0;
//  virtual NtsVar backward(NtsVar &output_grad) = 0;
//};

#endif

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

#ifndef TYPE_HPP
#define TYPE_HPP

#include <stdint.h>

// class Network;

enum DeviceLocation { CPU_T, GPU_T };

struct Empty {};

typedef uint32_t VertexId;
typedef uint64_t EdgeId;
typedef float ValueType;

struct VertexIndex {
  VertexId bufferIndex;
  VertexId positionIndex;
};

struct BackVertexIndex {
  VertexId *vertexSocketPosition;
  BackVertexIndex() { vertexSocketPosition = nullptr; }
  void setSocket(int socketNum) {
    vertexSocketPosition = new VertexId[socketNum];
    memset(vertexSocketPosition, -1, sizeof(VertexId) * socketNum);
  }
};

struct CscChunk {
  VertexId *dstList;
  VertexId *srcList;
  VertexId src[2];
  VertexId dst[2];
  int numOfEdge;
  int counter;
  int featureSize;
  CscChunk() {
    numOfEdge = 0;
    featureSize = 0;
    counter = 0;
  }
};
struct COOChunk {
  VertexId *dstList;
  VertexId *srcList;
  VertexId *dst_delta;
  VertexId *src_delta;
  int *partition_offset;
  int partitions;
  // vertex range for this chunk
  VertexId src_range[2];
  // vertex range for local partition
  VertexId dst_range[2];
  // number of edges from this chunk to local partition
  int numofedges;
  int counter;
  int featureSize;
  COOChunk() {
    numofedges = 0;
    featureSize = 0;
    counter = 0;
  }
  VertexId *src() { return srcList; }
  VertexId *dst() { return dstList; }
  VertexId *src_p() { return src_delta; }
  VertexId *dst_p() { return dst_delta; }
  void init_partition_offset(int partitions_) {
    partitions = partitions_;
    partition_offset = new int[partitions_];
    memset(partition_offset, 0, sizeof(int) * partitions_);
  }
  int get_edge_partition_size(int i) {
    return partition_offset[i + 1] - partition_offset[i];
  }
};

template <typename EdgeData> struct EdgeUnit {
  VertexId src;
  VertexId dst;
  EdgeData edge_data;
} __attribute__((packed));

template <> struct EdgeUnit<Empty> {
  VertexId src;
  union {
    VertexId dst;
    Empty edge_data;
  };
} __attribute__((packed));

template <typename EdgeData> struct AdjUnit {
  VertexId neighbour;
  EdgeData edge_data;
} __attribute__((packed));

template <> struct AdjUnit<Empty> {
  union {
    VertexId neighbour;
    Empty edge_data;
  };
} __attribute__((packed));

struct CompressedAdjIndexUnit {
  EdgeId index;
  VertexId vertex;
} __attribute__((packed));

template <typename EdgeData> struct VertexAdjList {
  AdjUnit<EdgeData> *begin;
  AdjUnit<EdgeData> *end;
  VertexAdjList() : begin(nullptr), end(nullptr) {}
  VertexAdjList(AdjUnit<EdgeData> *begin, AdjUnit<EdgeData> *end)
      : begin(begin), end(end) {}
};

#endif

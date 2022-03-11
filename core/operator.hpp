/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   operator.hpp
 * Author: wangqg
 *
 * Created on September 2, 2019, 8:52 PM
 */

#ifndef OPERATOR_HPP
#define OPERATOR_HPP
#include "core/math.hpp"

const double d = (double)0.8;
#define VECTOR_LENGTH 64
enum Ptype { Para1, Para2, Para3 };

void inc(float *dst, float *src, float **weight) {
  // compute(0)
}

void gatherIn() {}
void gatherOut() {}
void gatherEdgeMessage(VertexId src, VertexId dst) {}

void aggregate(nodeVector *output, nodeVector *curr,
               VertexAdjList<Empty> incoming_adj, W *weight) {
  nodeVector *tmpout = new nodeVector;
  for (AdjUnit<Empty> *ptr = incoming_adj.begin; ptr != incoming_adj.end;
       ptr++) {
    VertexId dst = ptr->neighbour;
    //        summation(tmpout, curr+dst);//self defined aggregate function
    //        matrixMulVector(output, weight ,tmpout);
  }
}

#endif /* OPERATOR_HPP */

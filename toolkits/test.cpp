/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include "core/graph.hpp"
#include "core/operator.hpp"
//#include "torch/script.h"
//
#include "torch/torch.h"

struct Net : torch::nn::Module {

  torch::Tensor W;
  torch::Tensor A = torch::rand({4, 4});

  Net(int64_t N, int64_t D) {
    W = register_parameter("W", torch::randn({N, D}));
    for (int i = 0; i < A.size(0); i++) {
      A[i][i] = 1;
    }
  }

  torch::Tensor forward(torch::Tensor x) {
    x = A * x * W;
    x = torch::relu(x);
    return x;
  }
};
void TEST_setnode() {
  printf("TEST_SETNODE\n");
  nodeVector *node = new nodeVector;
  node->set(0.5);
  for (int i = 0; i < VECTOR_LENGTH; i++) {
    printf("%f\t", node->data[i]);
  }
  printf("\n");
  node->setRandom(1.0);
  for (int i = 0; i < VECTOR_LENGTH; i++) {
    printf("%f\t", node->data[i]);
  }
  W *weight = new W;
  weight->set(0.5);
  float sum = 0.0;
  for (int i = 0; i < VECTOR_LENGTH; i++) {
    for (int j = 0; j < VECTOR_LENGTH; j++) {
      //  printf("%f ",weight->data[i][j]);
      sum += weight->get()[i][j];
    }
    //  printf("\n");
  }
  printf("\n");
  printf("%f\n", sum);
  delete node;
  delete weight;
}

void TEST_summation() {
  printf("TEST_SUM\n");
  nodeVector *node1 = new nodeVector;
  nodeVector *node2 = new nodeVector;
  node1->set(0.5);
  node2->set(0.3);
  summation(node1->get(), node2->get());
  for (int i = 0; i < VECTOR_LENGTH; i++) {
    printf("%f\t", node1->get()[i]);
  }
  printf("\n");
  delete node1;
  delete node2;
}
void TEST_multi() {
  nodeVector *node1 = new nodeVector;
  nodeVector *node2 = new nodeVector;
  W *weight = new W;
  node1->set(0.3);
  node2->set(0.5);
  weight->setRandom(1.0);
  vectorMulMatrix(node1->get(), node2->get(), weight->get());
  for (int i = 0; i < VECTOR_LENGTH; i++) {
    printf("%f ", node1->data[i]);
  }
  printf("\n##############################\n");
  weight->trans();
  matrixMulVector(node1->get(), weight->getTrans(), node2->get());
  for (int i = 0; i < VECTOR_LENGTH; i++) {
    printf("%f ", node1->data[i]);
  }
}
void TEST_LERUANDSIGMOD() {
  printf("TEST_function\n");
  nodeVector *node = new nodeVector;
  node->set(1.0);
  leru(node->get());
  for (int i = 0; i < VECTOR_LENGTH / 8; i++) {
    printf("%f\t", node->data[i]);
  }
  sigmoid(node->get());
  printf("\n");
  for (int i = 0; i < VECTOR_LENGTH / 8; i++) {
    printf("%f\t", node->data[i]);
  }
  printf("\n");
  node->set(-1);
  leru(node->get());
  for (int i = 0; i < VECTOR_LENGTH / 8; i++) {
    printf("%f\t", node->data[i]);
  }
  delete node;
  printf("\n");
}

void TEST_gatherEdgeMessage() {}
void TEST_aggregate() {}
int main() {
  TEST_setnode();
  TEST_summation();
  TEST_multi();
  TEST_LERUANDSIGMOD();
  return 0;
}

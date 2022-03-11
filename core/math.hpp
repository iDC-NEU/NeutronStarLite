/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   math.hpp
 * Author: wangqg
 *
 * Created on August 11, 2019, 5:07 PM
 */

#ifndef MATH_HPP
#include <math.h>
#define EXP 2.718281828459
#define VECTOR_LENGTH 64
typedef struct factor {
  float data[VECTOR_LENGTH];
  factor() { memset(data, 0, VECTOR_LENGTH * sizeof(float)); }

  inline float *get() { return data; }
  void set(float S) {
    for (int i = 0; i < VECTOR_LENGTH; i++) {
      data[i] = S;
    }
  }
  void setRandom(float up) {
    for (int i = 0; i < VECTOR_LENGTH; i++) {
      data[i] =
          static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / up));
    }
  }

} nodeVector;

typedef struct ParameterW {
  float **data;
  float **transp;
  ParameterW() {
    transp = new float *[VECTOR_LENGTH];
    data = new float *[VECTOR_LENGTH];
    for (int i = 0; i < VECTOR_LENGTH; i++) {
      data[i] = new float[VECTOR_LENGTH];
      memset(data[i], 0, sizeof(float) * VECTOR_LENGTH);
      transp[i] = new float[VECTOR_LENGTH];
      memset(transp[i], 0, sizeof(float) * VECTOR_LENGTH);
    }
  }
  void trans() {
    for (int i = 0; i < VECTOR_LENGTH; i++) {
      for (int j = 0; j < VECTOR_LENGTH; j++) {
        transp[j][i] = data[i][j];
      }
    }
    // return transp;
  }
  inline float **get() { return data; }
  inline float **getTrans() { return transp; }
  inline float **set(float S) {
    for (int i = 0; i < VECTOR_LENGTH; i++) {
      for (int j = 0; j < VECTOR_LENGTH; j++) {
        data[i][j] = S;
      }
    }
  }
  inline float **setRandom(float up) {
    for (int i = 0; i < VECTOR_LENGTH; i++) {
      for (int j = 0; j < VECTOR_LENGTH; j++) {
        data[i][j] =
            static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / up));
      }
    }
  }
} W;

typedef struct AllParameter {
  W W1;
  W W2;
  W W3;
  bool is_W1;
  bool is_W2;
  bool is_W3;
} parameters;

inline int summation(float *dst, const float *src) { // vector+vector
  for (int i = 0; i < VECTOR_LENGTH; i++) {
    dst[i] += src[i];
  }
  return 1;
}
int vectorMulMatrix(float *dst, float *src,
                    float **weight) { // vector*weight  cachefriendly multi.
  for (int i = 0; i < VECTOR_LENGTH; i++) {
    dst[i] = 0.0;
  }
  for (int j = 0; j < VECTOR_LENGTH; j++) {
    for (int i = 0; i < VECTOR_LENGTH; i++) {
      dst[i] += weight[j][i] * src[j];
    }
  }
  return 0;
}
int matrixMulVector(float *dst, float **weight,
                    float *src) { // vector*weight  cachefriendly multi.
  for (int i = 0; i < VECTOR_LENGTH; i++) {
    dst[i] = 0;
    for (int j = 0; j < VECTOR_LENGTH; j++) {
      dst[i] += weight[i][j] * src[j];
    }
  }
  return 0;
}

inline bool leru(float *input) {
  for (int i = 0; i < VECTOR_LENGTH; i++) {
    if (input[i] > 0) {
      // return true;
      continue;
    } else {
      input[i] = 0;
    }
  }
}

inline void sigmoid(float *input) {
  for (int i = 0; i < VECTOR_LENGTH; i++) {
    input[i] = 1.0 / (1.0 + exp(0 - input[i]));
  }
}

#endif /* MATH_HPP */

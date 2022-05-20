#include <atomic>
#include <vector>

#ifndef BITMAP_HPP
#define BITMAP_HPP

#define WORD_OFFSET(i) ((i) >> 6)
#define BIT_OFFSET(i) ((i)&0x3f)

class Bitmap {
  // using ElemType = std::atomic<unsigned long>;
public:
  // std::vector may affect runtime performance because it has additional
  // checking could be replaced by a simple array std::vector<ElemType> data;
  // graph.hpp is using raw data buffer directly, thus
  // we can't port it into c++ style directly
  // TODO: fix the inappropriate useage on bitmap
  size_t size;
  unsigned long *data;

  Bitmap() : size(0), data(nullptr) {}

  Bitmap(size_t size_) : size(size_) {
    data = new unsigned long[WORD_OFFSET(size) + 1];
    clear();
  }

  ~Bitmap() {
    // regardless whether the pointer is valid or not
    delete[] data;
  }

  void clear() {
  // not sure whether for range loop in c++ will be optimized by omp
  // since compiler essentially replace it by iterator
  size_t bm_size = WORD_OFFSET(size);
#pragma omp parallel for
  for (size_t i = 0; i <= bm_size; i++) {
    data[i] = 0;
  }
}

void fill() {
  size_t bm_size = WORD_OFFSET(size);
#pragma omp parallel for
  for (size_t i = 0; i < bm_size; i++) {
    data[i] = 0xffffffffffffffff;
  }
  data[bm_size] = 0;
  for (size_t i = (bm_size << 6); i < size; i++) {
    data[bm_size] |= 1ul << BIT_OFFSET(i);
  }
}

// TODO: identify the pattern of using bitmap
// and optimize it with weaker consistency model
// to gain performance
void set_bit(size_t i) {
  __sync_fetch_and_or(data + WORD_OFFSET(i), 1ul << BIT_OFFSET(i));
}

unsigned long get_bit(size_t i) {
  return data[WORD_OFFSET(i)] & (1ul << BIT_OFFSET(i));
}

};

typedef Bitmap VertexSubset;

#endif
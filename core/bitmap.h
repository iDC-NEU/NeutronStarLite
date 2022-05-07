#include <atomic>
#include <vector>

#ifndef BITMAP_H
#define BITMAP_H

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

  // clear all bits
  void clear();

  // setting all bits to 1
  void fill();

  // get corresponding bit indexed by i
  unsigned long get_bit(size_t i);

  // set corresponding bit indexed by i
  void set_bit(size_t i);
};

typedef Bitmap VertexSubset;

#endif
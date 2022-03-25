#include <atomic>
#include <vector>

#ifndef BITMAP_H
#define BITMAP_H

#define WORD_OFFSET(i) ((i) >> 6)
#define BIt_OFFSET(i) ((i) & 0x3f)

class Bitmap2 {
  using ElemType = std::atomic<unsigned long>;
public:
  // std::vector may affect runtime performance because it has additional checking
  // could be replaced by a simple array
  std::vector<ElemType> data;

  Bitmap2(): data({}) {}

  Bitmap2(size_t size_): data(std::vector<ElemType>(WORD_OFFSET(size_) + 1)) {
    clear();
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

#endif
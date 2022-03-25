#include "bitmap2.h"

void Bitmap2::clear() {
  // not sure whether for range loop in c++ will be optimized by omp
  // since compiler essentially replace it by iterator
  size_t sz = data.size();
  #pragma omp parallel for
  for (size_t i = 0; i < sz; i++) {
    data[i].store(0, std::memory_order::memory_order_relaxed);
  }
}

void Bitmap2::fill() {
  size_t sz = data.size();
  for (size_t i = 0; i < sz; i++) {
    data[i].store(UINT64_MAX, std::memory_order::memory_order_relaxed);
  }
}

// TODO: identify the pattern of using bitmap
// and optimize it with weaker consistency model
// to gain performance
void Bitmap2::set_bit(size_t i) {
  data[WORD_OFFSET(i)].fetch_or(1ul << BIt_OFFSET(i));
}

unsigned long Bitmap2::get_bit(size_t i) {
  return data[WORD_OFFSET(i)].load() & (1ul << BIt_OFFSET(i));
}
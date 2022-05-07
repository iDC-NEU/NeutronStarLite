#include "bitmap.h"

void Bitmap::clear() {
  // not sure whether for range loop in c++ will be optimized by omp
  // since compiler essentially replace it by iterator
  size_t bm_size = WORD_OFFSET(size);
#pragma omp parallel for
  for (size_t i = 0; i <= bm_size; i++) {
    data[i] = 0;
  }
}

void Bitmap::fill() {
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
void Bitmap::set_bit(size_t i) {
  __sync_fetch_and_or(data + WORD_OFFSET(i), 1ul << BIT_OFFSET(i));
}

unsigned long Bitmap::get_bit(size_t i) {
  return data[WORD_OFFSET(i)] & (1ul << BIT_OFFSET(i));
}
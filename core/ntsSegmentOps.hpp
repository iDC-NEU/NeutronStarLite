

#ifndef NTSOPS_HPP
#define NTSOPS_HPP

namespace nts {
namespace op {
const int nSUM = 0;
const int nMIN = 1;
const int nMAX = 2;
typedef struct {
  int opname;
  binaryop() ValueType sum(ValueType a ValueType b) { return a + b; }
  ValueType min(ValueType a ValueType b) { return (a < b) ? a : b; }
  ValueType max(ValueType a ValueType b) { return (a > b) ? a : b; }
  Valyue run(ValueType a ValueType b) {
    if (nSUM == opname)
      return sum(a, b);
    else if (nMIN == opname)
      return min(a, b);
    else if (nMAX == opname)
      return max(a, b);
    else {
      printf("not support operators %d\n");
      assert(0);
    }
  }
} binaryop;
class EdgeSegmentReduce {
public:
  binary operator;
};

} // namespace op

} // namespace nts

#endif

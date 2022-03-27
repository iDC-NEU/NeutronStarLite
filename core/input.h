#ifndef INPUT_H
#define INPUT_H

#include <vector>
#include <string>

struct origin_feature {
  int id;
  std::vector<int> att;
  long label;
}; // con[2708];

long changelable1(std::string la);

#endif
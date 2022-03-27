#include <map>
#include <string>
#include "input.h"

long changelable1(std::string la) {
  std::map<std::string, long> label;

  label["Case_Based"] = 0;
  label["Genetic_Algorithms"] = 1;
  label["Neural_Networks"] = 2;
  label["Probabilistic_Methods"] = 3;
  label["Reinforcement_Learning"] = 4;
  label["Rule_Learning"] = 5;
  label["Theory"] = 6;
  long l = label[la];
  // test=label.find("Theory");
  return l;
}
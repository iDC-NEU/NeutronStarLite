#ifndef INPUT_HPP
#define INPUT_HPP
#include <vector>
struct origin_feature
{
    int id;
    std::vector<int> att;
    long label;
};//con[2708];

long changelable1(std::string la)
{
    std::map<std::string,long> label;

    label["Case_Based"]=0;
    label["Genetic_Algorithms"]=1;
    label["Neural_Networks"]=2;
    label["Probabilistic_Methods"]=3;
    label["Reinforcement_Learning"]=4;
    label["Rule_Learning"]=5;
    label["Theory"]=6;
    long l=label[la];
    //test=label.find("Theory");
    return l;
}




#endif
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <string.h>
#include <cstring>
#include <vector>
#include <assert.h>
using namespace std;
void generate_edge(int vertices,std::string edge_file, bool self_cycle=true, bool bidirection=true){
    
    std::string edge_file_sorted=edge_file+".bin";
    std::ifstream fp_edge(edge_file.c_str(), std::ios::in);      
    std::ofstream fp_edge_sorted(edge_file_sorted.c_str(), std::ios::binary);
    std:: string lineStr;
    std::vector<std::vector<int>>edge_list;
    std::vector<int>row;
    row.resize(0);
    long init_e_count=0;
    edge_list.resize(vertices,row);
    if (!fp_edge.is_open())
        {
            std::cout<<"open "<< edge_file<<" fail!"<<std::endl;
            return;
        }
    while(getline(fp_edge,lineStr)){
        init_e_count++;
        stringstream ss(lineStr);
        std::string vtx;
        int edge[2]={0,0};
        int i=0;
        while(getline(ss,vtx,',')){
            edge[i]=std::atoi(vtx.c_str());
            i++;
        }
//        if(init_e_count>1166200){
//            std::cout<<init_e_count<<" "<<edge[0]<<" "<<edge[1]<<std::endl;
//        }
        edge_list[edge[0]].push_back(edge[1]);
        if(bidirection==true){
            edge_list[edge[1]].push_back(edge[0]);
        }
    }
    
    
    
    for(int i=0;i<vertices;i++){
        std::vector<int>::iterator vector_iter;
        if(self_cycle==true){
            edge_list[i].push_back(i);
        }
        sort(edge_list[i].begin(),edge_list[i].end());
        vector_iter=unique(edge_list[i].begin(),edge_list[i].end());
        if(vector_iter!=edge_list[i].end()){
            edge_list[i].erase(edge_list[i].begin(),edge_list[i].end());
        }
//        fp_edge_sorted.write((char*)&i,sizeof(int));
//        fp_edge_sorted.write((char*)&i,sizeof(int));
    }
    long e_count=0;
    for(int i=0;i<vertices;i++){
        for(int j=0;j<edge_list[i].size();j++){
            e_count++;
            assert(edge_list[i][j]<vertices);
        fp_edge_sorted.write((char*)&i,sizeof(int));
        fp_edge_sorted.write((char*)&(edge_list[i][j]),sizeof(int)); 
        }

    }
    printf("init edge_count:%ld\nwrited edge_count:%ld\n",init_e_count,e_count);
    
    
    fp_edge_sorted.close();
    fp_edge.close();
    //fclose(fp_label);   
/*    std::ifstream fp_edge_test(edge_file_sorted.c_str(), std::ios::binary);
    for(int i=0;i<23446805;i++){
        int src;
        int dst;
        fp_edge_test.read((char*)&src,sizeof(int));
        fp_edge_test.read((char*)&dst,sizeof(int));
        if(i>23446705)
        std::cout<<"edge[0]: "<<src<<" edge[1]: "<<dst<<std::endl;
        //fp_edge_sorted<<dst<<" "<<src<<std::endl;
    }
    */
    return;
}

    
int main(int argc, char ** argv){
    if(argc<5){
        printf("|V| |self_cycle| |bi_direction| edge_file_name(*.csv)\n");
    }
    else{
//generate_edge(169343,"edge.csv");
   bool self_cycle= atoi(argv[2])>0;
   bool bi_direction= atoi(argv[3])>0;
   std::cout<<"|V|:\t"<<  atoi(argv[1])<<"\nself_cyc:\t"
            <<self_cycle<<"\nbi_direct:\t"<<bi_direction
            <<"\nedge_file:\t"<<argv[4]<<std::endl;  
  generate_edge(atoi(argv[1]),argv[4],self_cycle,bi_direction);
    }
return 0;
}

/*
Copyright (c) 2021-2022 Qiange Wang, Northeastern University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#if CUDA_ENABLE
#include "ntsCUDA.hpp"
#endif

#include <vector>
#include <map>
#include <algorithm>
#ifndef COOCSC_HPP
#define COOCSC_HPP

class sampCSC{
public:    
    sampCSC(){
        v_size=0;
        e_size=0;
        column_offset.clear();    
        row_indices.clear();
        src_index.clear();
        destination.clear();
        source.clear();
    }
    sampCSC(VertexId v_, VertexId e_){
        v_size=v_;
        e_size=e_;
        column_offset.clear();    
        row_indices.clear();
        src_index.clear();
        destination.clear();
        source.clear();
    }
    sampCSC(VertexId v_){
        v_size=v_;
        e_size=0;
        column_offset.clear();    
        row_indices.clear();
        src_index.clear();
        destination.clear();
        source.clear();
    }
    ~sampCSC(){
        column_offset.clear();    
        row_indices.clear();
        src_index.clear();
        destination.clear();
        source.clear();
    }
    void postprocessing(){
        src_size=0;
        row_indices_debug.resize(row_indices.size(),0);
        for(VertexId i_src=0;i_src<row_indices.size();i_src++){
          //  printf("debug %d\n",i_src);
          row_indices_debug[i_src]=row_indices[i_src];
          if(0xFFFFFFFF==row_indices[i_src]){
              continue;
          }
            std::map<VertexId, VertexId>::iterator iter;
            iter = src_index.find(row_indices[i_src]);  
            //printf("%d\n",iter == src_index.end());
            if(iter == src_index.end()){   
            //    printf("debug %d\n",i_src);
                src_index.insert(std::make_pair(row_indices[i_src], src_size));
                src_size++;
           //     printf("debug %d\n",i_src);
                source.push_back(row_indices[i_src]);
                row_indices[i_src]=src_size-1;
                //reset src for computation
            }
            else{
                // redundant continue;
                assert(row_indices[i_src]==iter->first);
                row_indices[i_src]=iter->second; //reset src for computation
            }
        }
    }
    void allocate_vertex(){
        destination.resize(v_size,0);       
        column_offset.resize(v_size+1,0);
    }
    void allocate_co_from_dst(){
        v_size=destination.size();
        column_offset.resize(v_size+1,0);
    }
    void allocate_edge(){
        assert(0);
        row_indices.resize(e_size,0);
    }
    void allocate_edge(VertexId e_size_){
        e_size=e_size_;
        row_indices.resize(e_size,0);
    }
    void allocate_all(){
        allocate_vertex();
        allocate_edge();
    }
    VertexId c_o(VertexId vid){
        return column_offset[vid];
    }
    VertexId r_i(VertexId vid){
        return row_indices[vid];
    }
    std::vector<VertexId>& dst(){
        return destination;
    }
    std::vector<VertexId>& src(){
        return source;
    }
    std::vector<VertexId>& c_o(){
        return column_offset;
    }
    std::vector<VertexId>& r_i(){
        return row_indices;
    }
    VertexId get_distinct_src_size(){
        return src_size;
    }
    void debug(){
        printf("print one layer:\ndst:\t");
        for(int i=0;i<destination.size();i++){
            printf("%d\t",destination[i]);
        }printf("\nc_o:\t");
        for(int i=0;i<column_offset.size();i++){
            printf("%d\t",column_offset[i]);
        }printf("\nr_i:\t");
        for(int i=0;i<row_indices.size();i++){
            printf("%d\t",row_indices[i]);
        }printf("\nrid:\t");
        for(int i=0;i<row_indices_debug.size();i++){
            printf("%d\t",row_indices_debug[i]);
        }printf("\nsrc:\t");
        for(int i=0;i<source.size();i++){
            printf("%d\t",source[i]);
        }printf("\n\n");
    }
    
private:
std::vector<VertexId> column_offset;//local offset    
std::vector<VertexId> row_indices;//local id
std::vector<VertexId> row_indices_debug;//local id

std::vector<VertexId> source;//global id
std::vector<VertexId> destination;//global_id

std::map<VertexId,VertexId> src_index;//set

VertexId v_size; //dst_size
VertexId e_size; // edge size
VertexId src_size;//distinct src size
};



#endif
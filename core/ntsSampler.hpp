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
#ifndef NTSSAMPLER_HPP
#define NTSSAMPLER_HPP
#include <mutex>
#include <cmath>
#include "FullyRepGraph.hpp"
class Sampler{
public:
    std::vector<SampledSubgraph*> work_queue;// excepted to be single write multi read
    std::mutex queue_start_lock;
    int queue_start;
    std::mutex queue_end_lock;
    int queue_end;
    FullyRepGraph* whole_graph;
    VertexId start_vid,end_vid;
    VertexId work_range[2];
    VertexId work_offset;
    VertexId sg_size;
    Sampler(FullyRepGraph* whole_graph_, VertexId work_start,VertexId work_end){
        whole_graph=whole_graph_;
        queue_start=0;
        queue_end=0;
        work_range[0]=work_start;
        work_range[1]=work_end;
        work_offset=work_start;
        sg_size=0;
    }
    ~Sampler(){
        clear_queue();
    }
    bool has_rest(){
        return queue_start<queue_end;
    }
    SampledSubgraph* get_one(){
//        while(true){
//            bool condition=queue_start<queue_end;
//            if(condition){
//                break;
//            }
//         __asm volatile("pause" ::: "memory");  
//        }
        queue_start_lock.lock();
        VertexId id=queue_start++;
        queue_start_lock.unlock();
        return work_queue[id];
    }
    void clear_queue(){
        for(VertexId i=0;i<work_queue.size();i++){
            delete work_queue[i];
        }
    } 
    bool sample_not_finished(){
        return work_offset<work_range[1];
    }
    void restart(){
        work_offset=work_range[0];
        sg_size=0;
    }
    void sampled_a_new_graph(int layers_, int batch_size_,std::vector<int> fanout_){
        assert(work_offset<work_range[1]);
        int actual_batch_size=std::min((VertexId)batch_size_,work_range[1]-work_offset);
        SampledSubgraph* ssg=new SampledSubgraph(layers_,actual_batch_size,fanout_);  
        
        for(int i=0;i<layers_;i++){
            ssg->sample_preprocessing(i);
            //whole_graph->SyncAndLog("preprocessing");
            if(i==0){
              ssg->sample_load_destination([&](std::vector<VertexId>& destination){
                  for(int j=0;j<actual_batch_size;j++){
                      destination[j]=work_offset++;
                  }
              },i);
              //whole_graph->SyncAndLog("sample_load_destination");
            }else{
               ssg->sample_load_destination(i); 
              //whole_graph->SyncAndLog("sample_load_destination2");
            }
            ssg->init_co([&](VertexId dst){
                VertexId nbrs=whole_graph->column_offset[dst+1]
                                 -whole_graph->column_offset[dst];
            return (nbrs>fanout_[i]) ? fanout_[i] : nbrs;
            },i);
            ssg->sample_processing([&](VertexId fanout_i,
                std::vector<VertexId> &destination,
                    std::vector<VertexId> &column_offset,
                        std::vector<VertexId> &row_indices,VertexId id){
                VertexId dst= destination[id];
                for(VertexId src_idx=whole_graph->column_offset[dst];
                        src_idx<whole_graph->column_offset[dst+1];src_idx++){
                    VertexId write_pos=(src_idx-whole_graph->column_offset[dst])%fanout_i;
                    write_pos+=column_offset[id];
                    row_indices[write_pos]=whole_graph->row_indices[src_idx];
                }
            });
            //whole_graph->SyncAndLog("sample_processing");
            ssg->sample_postprocessing();
            //whole_graph->SyncAndLog("sample_postprocessing");
        }
        work_queue.push_back(ssg);
        queue_end_lock.lock();
        queue_end++;
        queue_end_lock.unlock();
        if(work_offset>=work_range[1])
            sg_size=work_queue.size();
    }
};


#endif
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
#include <random>
#include <cmath>
#include <stdlib.h>
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
    std::vector<VertexId> sample_nids;
    Sampler(FullyRepGraph* whole_graph_, VertexId work_start,VertexId work_end){
        whole_graph=whole_graph_;
        queue_start=-1;
        queue_end=0;
        work_range[0]=work_start;
        work_range[1]=work_end;
        work_offset=work_start;
    }
    Sampler(FullyRepGraph* whole_graph_, std::vector<VertexId>& index){
        assert(index.size() > 0);
        sample_nids.assign(index.begin(), index.end());
        assert(sample_nids.size() == index.size());
        whole_graph=whole_graph_;
        queue_start=-1;
        queue_end=0;
        work_range[0]=0;
        work_range[1]=sample_nids.size();
        work_offset=0;
    }
    ~Sampler(){
        clear_queue();
    }
    bool has_rest(){
        bool condition=false;
        int cond_start=0;
        queue_start_lock.lock();
        cond_start=queue_start;
        queue_start_lock.unlock();
        
        int cond_end=0;
        queue_end_lock.lock();
        cond_end=queue_end;
        queue_end_lock.unlock();
       
        condition=cond_start<cond_end&&cond_start>=0;
        return condition;
    }
//    bool has_rest(){
//        bool condition=false;
//        condition=queue_start<queue_end&&queue_start>=0;
//        return condition;
//    }
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
        assert(id<work_queue.size());
        return work_queue[id];
    }
    void clear_queue(){
        for(VertexId i=0;i<work_queue.size();i++){
            delete work_queue[i];
        }
        work_queue.clear();
    } 
    bool sample_not_finished(){
        return work_offset<work_range[1];
    }
    void restart(){
        work_offset=work_range[0];
        queue_start=-1;
        queue_end=0;
    }
    
    int random_uniform_int(const int min = 0, const int max = 1) {
        // thread_local std::default_random_engine generator;
        static thread_local std::mt19937 generator;
        std::uniform_int_distribution<int> distribution(min, max);
        return distribution(generator);
    }

    void reservoir_sample(int layers_, int batch_size_,std::vector<int> fanout_){
        assert(work_offset<work_range[1]);
        int actual_batch_size=std::min((VertexId)batch_size_,work_range[1]-work_offset);
        SampledSubgraph* ssg=new SampledSubgraph(layers_,fanout_);  
        
        for(int i=0;i<layers_;i++){
            ssg->sample_preprocessing(i);
            //whole_graph->SyncAndLog("preprocessing");
            if(i==0){
              ssg->sample_load_destination([&](std::vector<VertexId>& destination){
                  for(int j=0;j<actual_batch_size;j++){
                    //   destination.push_back(work_offset++);
                    destination.push_back(sample_nids[work_offset++]);
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
                    VertexId dst,
                    std::vector<VertexId> &column_offset,
                        std::vector<VertexId> &row_indices,VertexId id){
                for(VertexId src_idx=whole_graph->column_offset[dst];
                        src_idx<whole_graph->column_offset[dst+1];src_idx++){
                    //ReservoirSampling
                    VertexId write_pos=(src_idx-whole_graph->column_offset[dst]);
                    if(write_pos<fanout_i){
                        write_pos+=column_offset[id];
                        row_indices[write_pos]=whole_graph->row_indices[src_idx];
                    }else{
                        // VertexId random=rand()%write_pos;
                        VertexId random=random_uniform_int(0, write_pos-1);
                        if(random<fanout_i){
                          row_indices[random+column_offset[id]]=  
                                  whole_graph->row_indices[src_idx];
                        }
                    }
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
        if(work_queue.size()==1){
            queue_start_lock.lock();
            queue_start=0;
            queue_start_lock.unlock();
        }
    }
};


#endif
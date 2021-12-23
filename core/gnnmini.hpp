/*
Copyright (c) 2014-2015 Xiaowei Zhu, Tsinghua University

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
#ifndef GNNMINI_HPP
#define GNNMINI_HPP
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <assert.h>

#include <map>
#include "graph.hpp"
#include <unistd.h>
#include <math.h>
//#include "comm/Network.hpp"
#include "cuda/test.hpp"
#include "core/input.hpp"

long changelable(std::string la)
{
    std::map<std::string, long> label;

    label["Case_Based"] = 0;
    label["Genetic_Algorithms"] = 1;
    label["Neural_Networks"] = 2;
    label["Probabilistic_Methods"] = 3;
    label["Reinforcement_Learning"] = 4;
    label["Rule_Learning"] = 5;
    label["Theory"] = 6;
    long l = label[la];
    //test=label.find("Theory");
    return l;
}


class GNNDatum
{
public:
    gnncontext *gnnctx;
    float *local_feature;
    long *local_label;
    int *local_mask;
    Graph<Empty>* graph;
    // train:    0 
    // eval:     1
    // test:     2
    GNNDatum(gnncontext *_gnnctx, Graph<Empty>* graph_)
    {
        gnnctx = _gnnctx;
        local_feature = new float[gnnctx->l_v_num * gnnctx->layer_size[0]];
        local_label = new long[gnnctx->l_v_num];
        local_mask= new int[gnnctx->l_v_num];
        memset(local_mask,0,sizeof(int)*gnnctx->l_v_num);
        graph=graph_;
    }
    void random_generate()
    {
        for (int i = 0; i < gnnctx->l_v_num; i++)
        {
            for (int j = 0; j < gnnctx->layer_size[0]; j++)
            {
                local_feature[i * gnnctx->layer_size[0] + j] = 1.0;
            }
            local_label[i] = rand() % gnnctx->label_num;
            local_mask[i]=i%3;
        }
    }
    void registLabel(NtsVar &target)
    {
        target = graph->Nts->NewLeafKLongTensor(local_label,{gnnctx->l_v_num});
                //torch::from_blob(local_label, gnnctx->l_v_num, torch::kLong);
    }
    void registMask(NtsVar &mask){
        mask= graph->Nts->NewLeafKIntTensor(local_mask,{gnnctx->l_v_num,1});
                //torch::from_blob(local_mask, {gnnctx->l_v_num,1}, torch::kInt32);
    }
     void readFtrFrom1(std::string inputF,std::string inputL)
    {
       
        std::string str;
        std::ifstream input_ftr(inputF.c_str(), std::ios::in);
        std::ifstream input_lbl(inputL.c_str(), std::ios::in);
        //std::ofstream outputl("cora.labeltable",std::ios::out);
       // ID    F   F   F   F   F   F   F   L
        if (!input_ftr.is_open())
        {
            std::cout<<"open feature file fail!"<<std::endl;
            return;
        }
        if (!input_lbl.is_open())
        {
            std::cout<<"open label file fail!"<<std::endl;
            return;
        }
        float *con_tmp = new float[gnnctx->layer_size[0]];
        std::string la;
        //std::cout<<"finish1"<<std::endl;
        VertexId id=0;
        while (input_ftr >> id)
        {
            VertexId size_0=gnnctx->layer_size[0];
            VertexId id_trans=id-gnnctx->p_v_s;
            if((gnnctx->p_v_s<=id)&&(gnnctx->p_v_e>id)){
                for (int i = 0; i < size_0; i++){
                    input_ftr >> local_feature[size_0*id_trans+i];
                }
                input_lbl >> la;
                input_lbl >> local_label[id_trans];
                local_mask[id_trans]=id%3;
            }else{
                for (int i = 0; i < size_0; i++){
                    input_ftr >> con_tmp[i];   
                }
                input_lbl >> la;
                input_lbl >> la;
            }
        }
        free(con_tmp);
        input_ftr.close();
        input_lbl.close();
    }
    void readFeature_Label_Mask(std::string inputF,std::string inputL,std::string inputM)
    {
       
        std::string str;
        std::ifstream input_ftr(inputF.c_str(), std::ios::in);
        std::ifstream input_lbl(inputL.c_str(), std::ios::in);
        std::ifstream input_msk(inputM.c_str(), std::ios::in);
        //std::ofstream outputl("cora.labeltable",std::ios::out);
       // ID    F   F   F   F   F   F   F   L
        if (!input_ftr.is_open())
        {
            std::cout<<"open feature file fail!"<<std::endl;
            return;
        }
        if (!input_lbl.is_open())
        {
            std::cout<<"open label file fail!"<<std::endl;
            return;
        }
        if (!input_msk.is_open())
        {
            std::cout<<"open mask file fail!"<<std::endl;
            return;
        }
        float *con_tmp = new float[gnnctx->layer_size[0]];
        std::string la;
        //std::cout<<"finish1"<<std::endl;
        VertexId id=0;
        while (input_ftr >> id)
        {
            VertexId size_0=gnnctx->layer_size[0];
            VertexId id_trans=id-gnnctx->p_v_s;
            if((gnnctx->p_v_s<=id)&&(gnnctx->p_v_e>id)){
                for (int i = 0; i < size_0; i++){
                    input_ftr >> local_feature[size_0*id_trans+i];
                }
                input_lbl >> la;
                input_lbl >> local_label[id_trans];
                
                input_msk >>la;
                std::string msk;
                input_msk >>msk;
                //std::cout<<la<<" "<<msk<<std::endl;
                if(msk.compare("train")==0){
                    local_mask[id_trans]=0;
                }else if (msk.compare("eval")==0||msk.compare("val")==0){
                    local_mask[id_trans]=1;
                }else if (msk.compare("test")==0){
                    local_mask[id_trans]=2;
                }else{
                    local_mask[id_trans]=3;
                }
                
            }else{
                for (int i = 0; i < size_0; i++){
                    input_ftr >> con_tmp[i];   
                }
                
                input_lbl >> la;
                input_lbl >> la;
                
                input_msk >>la;
                input_msk >>la;
            }
        }
        free(con_tmp);
        input_ftr.close();
        input_lbl.close();
    }
};

template <typename t_v, typename t_l>
class GTensor
{

public:
    Graph<Empty> *graph_;
    VertexSubset *active_;
    VertexId start_,end_,range_;

    int *size_at_layer;

    GTensor(Graph<Empty> *graph, VertexSubset *active)
    {
        graph_ = graph;
        active_ = active;
        start_ = graph->gnnctx->p_v_s;
        end_ = graph->gnnctx->p_v_e;
        range_=end_-start_;
    }
    void comp(float* input,float*output,float weight,int feat_size){
        for(int i=0;i<feat_size;i++){
            output[i]+=input[i]*weight;
        }
    }
    void acc(float* input,float*output,int feat_size){
        for(int i=0;i<feat_size;i++){
            write_add(&output[i],input[i]);
        }
    }
    float norm_degree(VertexId src,VertexId dst){
        return 1/((ValueType)std::sqrt(graph_->out_degree_for_backward[src])* 
                    (ValueType)std::sqrt(graph_->in_degree_for_backward[dst]));              
    }
    float out_degree(VertexId v){
        return (ValueType)(graph_->out_degree_for_backward[v]);
    }
    float in_degree(VertexId v){
        return (ValueType)(graph_->in_degree_for_backward[v]);
    }
    
    void SampleStrategy(VertexId dst,std::vector<CSC_segment_pinned *> &subgraphs){
    }
    void SampleStage(NtsVar &X, NtsVar &Y,
                                  std::vector<CSC_segment_pinned *> &subgraphs, Bitmap* VertexToSample){
        int feature_size=1;
//        graph_->process_edges_backward_decoupled<int, VertexId>( // For EACH Vertex Processing
//            [&](VertexId src, VertexAdjList<Empty> outgoing_adj,VertexId thread_id,VertexId recv_id) {           //pull
//                VertexId src_trans=src-graph_->partition_offset[recv_id];
//                VertexId msg;
//                if(!VertexToSample->get_bit(src_trans)){
//                    return;
//                }
//                for(long d_idx=subgraphs[recv_id]->row_offset[src_trans];d_idx<subgraphs[recv_id]->row_offset[src_trans+1];d_idx++)
//                {
//                    VertexId dst=subgraphs[recv_id]->column_indices[d_idx];
//                    VertexId dst_trans=dst-start_;
//                }
//                graph_->NtsComm->emit_buffer(src, &msg,feature_size);
//            },
//            [&](VertexId src, VertexId* msg) {
//                return 1;
//            },
//            feature_size,
//            active_);
    }
    void ProcessForwardCPU(NtsVar &X, NtsVar &Y,std::vector<CSC_segment_pinned *> &subgraphs,
                                   std::function<float(VertexId&, VertexId&)> weight_fun)
    {
        float* X_buffer=graph_->Nts->getWritableBuffer(X,torch::DeviceType::CPU);
        float* Y_buffer=graph_->Nts->getWritableBuffer(Y,torch::DeviceType::CPU);
        memset(Y_buffer,0,sizeof(float)*X.size(0)*X.size(1));
        int feature_size=graph_->gnnctx->layer_size[graph_->rtminfo->curr_layer];
        
        //graph_->process_edges_forward_debug<int,float>( // For EACH Vertex Processing
        graph_->process_edges_forward_decoupled<int,float>( // For EACH Vertex Processing
            [&](VertexId src) {
                    //graph_->emit_buffer(src, X_buffer+(src-graph_->gnnctx->p_v_s)*feature_size, feature_size);
                   graph_->NtsComm->emit_buffer(src, X_buffer+(src-graph_->gnnctx->p_v_s)*feature_size, feature_size);
                },
            [&](VertexId dst, CSC_segment_pinned* subgraph,char* recv_buffer, std::vector<VertexId>& src_index,VertexId recv_id) {
                VertexId dst_trans=dst-graph_->partition_offset[graph_->partition_id];
                for(long idx=subgraph->column_offset[dst_trans];idx<subgraph->column_offset[dst_trans+1];idx++){
                    VertexId src=subgraph->row_indices[idx];
                    VertexId src_trans=src-graph_->partition_offset[recv_id];
                    float* local_input=(float*)(recv_buffer+graph_->sizeofM<float>(feature_size)*src_index[src_trans]+sizeof(VertexId));
                    float* local_output=Y_buffer+dst_trans*feature_size;
//                    if(dst==0&&recv_id==0){
//                        printf("DEBUGGGG%d :%d %f\n",feature_size,subgraph->column_offset[dst_trans+1]-subgraph->column_offset[dst_trans],local_input[7]);
//                    }
                    comp(local_input,local_output,weight_fun(src,dst),feature_size);
                }
            },
            subgraphs,
            feature_size,
            active_);
    }    
    
    
    
    void PropagateForwardCPU_debug(NtsVar &X, NtsVar &Y,
                                  std::vector<CSC_segment_pinned *> &subgraphs){
       float* X_buffer=graph_->Nts->getWritableBuffer(X,torch::DeviceType::CPU);
       float* Y_buffer=graph_->Nts->getWritableBuffer(Y,torch::DeviceType::CPU);
       memset(Y_buffer,0,sizeof(float)*X.size(0)*X.size(1));
       //int feature_size=graph_->gnnctx->layer_size[graph_->rtminfo->curr_layer];
       int feature_size=X.size(1);
       //graph_->process_edges_forward_debug<int,float>( // For EACH Vertex Processing
       graph_->process_edges_forward_decoupled<int,float>( // For EACH Vertex Processing
           [&](VertexId src) {
                   //graph_->emit_buffer(src, X_buffer+(src-graph_->gnnctx->p_v_s)*feature_size, feature_size);
                  graph_->NtsComm->emit_buffer(src, X_buffer+(src-graph_->gnnctx->p_v_s)*feature_size, feature_size);
               },
           [&](VertexId dst, CSC_segment_pinned* subgraph,char* recv_buffer, std::vector<VertexId>& src_index,VertexId recv_id) {
               VertexId dst_trans=dst-graph_->partition_offset[graph_->partition_id];
               for(long idx=subgraph->column_offset[dst_trans];idx<subgraph->column_offset[dst_trans+1];idx++){
                    VertexId src=subgraph->row_indices[idx];
                    VertexId src_trans=src-graph_->partition_offset[recv_id];
                    float* local_input=(float*)(recv_buffer+graph_->sizeofM<float>(feature_size)*src_index[src_trans]+sizeof(VertexId));
                    float* local_output=Y_buffer+dst_trans*feature_size;
//                    if(dst==0&&recv_id==0){
//                        printf("DEBUGGGG%d :%d %f\n",feature_size,subgraph->column_offset[dst_trans+1]-subgraph->column_offset[dst_trans],local_input[7]);
//                    }
                    comp(local_input,local_output,norm_degree(src,dst),feature_size);
                    //comp(local_input,local_output,norm_degree(src,dst),feature_size);
                }
            },
            subgraphs,
            feature_size,
            active_);
    }
    void PropagateBackwardCPU_debug(NtsVar &X_grad, NtsVar &Y_grad,std::vector<CSC_segment_pinned *> &subgraphs)
    {       
        float* X_grad_buffer=graph_->Nts->getWritableBuffer(X_grad,torch::DeviceType::CPU);
        float* Y_grad_buffer=graph_->Nts->getWritableBuffer(Y_grad,torch::DeviceType::CPU);
        memset(Y_grad_buffer,0,sizeof(float)*X_grad.size(0)*X_grad.size(1));
        //int feature_size=graph_->gnnctx->layer_size[graph_->rtminfo->curr_layer];
        int feature_size=X_grad.size(1);
        float* output_buffer=new float[feature_size*graph_->threads];
        //graph_->process_edges_backward<int, float>( // For EACH Vertex Processing
        graph_->process_edges_backward_decoupled<int, float>( // For EACH Vertex Processing
            [&](VertexId src, VertexAdjList<Empty> outgoing_adj,VertexId thread_id,VertexId recv_id) {           //pull
                float* local_output_buffer=output_buffer+feature_size*thread_id;
                memset(local_output_buffer,0,sizeof(float)*feature_size);
                VertexId src_trans=src-graph_->partition_offset[recv_id];
                //for (AdjUnit<Empty> *ptr = outgoing_adj.begin; ptr != outgoing_adj.end; ptr++)
                for(long d_idx=subgraphs[recv_id]->row_offset[src_trans];d_idx<subgraphs[recv_id]->row_offset[src_trans+1];d_idx++)
                {
                    //VertexId dst = ptr->neighbour;
                    VertexId dst=subgraphs[recv_id]->column_indices[d_idx];
                    VertexId dst_trans=dst-start_;
                    float* local_input_buffer=X_grad_buffer+(dst_trans)*feature_size;  
                    comp(local_input_buffer,local_output_buffer,norm_degree(src,dst),feature_size);   
                    //comp(local_input_buffer,local_output_buffer,1,feature_size);     
                }
                //graph_->emit_buffer(src, local_output_buffer,feature_size);
                graph_->NtsComm->emit_buffer(src, local_output_buffer,feature_size);
            },
            [&](VertexId src, float* msg) {
                acc(msg,Y_grad_buffer+(src-start_)*feature_size,feature_size);
                return 1;
            },
            feature_size,
            active_);
    }
    void PropagateForwardCPU_Lockfree(NtsVar &X, NtsVar &Y,
                                  std::vector<CSC_segment_pinned *> &subgraphs){
       float* X_buffer=graph_->Nts->getWritableBuffer(X,torch::DeviceType::CPU);
       float* Y_buffer=graph_->Nts->getWritableBuffer(Y,torch::DeviceType::CPU);
       memset(Y_buffer,0,sizeof(float)*X.size(0)*X.size(1));
       //int feature_size=graph_->gnnctx->layer_size[graph_->rtminfo->curr_layer];
       int feature_size=X.size(1);
       graph_->process_edges_forward_decoupled_lock_free<int,float>( // For EACH Vertex Processing
           [&](VertexId src,int current_send_partition) {
               if(graph_->rtminfo->lock_free){
                    VertexId src_trans=src-graph_->gnnctx->p_v_s;
                    if(subgraphs[current_send_partition]->get_forward_active(src_trans)){
                        VertexId write_index=subgraphs[current_send_partition]->forward_message_index[src_trans];
                        graph_->NtsComm->emit_buffer_lock_free(src, X_buffer+(src-graph_->gnnctx->p_v_s)*feature_size,write_index, feature_size);
                    }
               }else{
                   graph_->NtsComm->emit_buffer(src, X_buffer+(src-graph_->gnnctx->p_v_s)*feature_size, feature_size);
               }
               },
           [&](VertexId dst, CSC_segment_pinned* subgraph,char* recv_buffer, std::vector<VertexId>& src_index,VertexId recv_id) {
               VertexId dst_trans=dst-graph_->partition_offset[graph_->partition_id];
               for(long idx=subgraph->column_offset[dst_trans];idx<subgraph->column_offset[dst_trans+1];idx++){
                    VertexId src=subgraph->row_indices[idx];
                    VertexId src_trans=src-graph_->partition_offset[recv_id];
                    float* local_input=(float*)(recv_buffer+graph_->sizeofM<float>(feature_size)*src_index[src_trans]+sizeof(VertexId));
                    float* local_output=Y_buffer+dst_trans*feature_size;
                    comp(local_input,local_output,norm_degree(src,dst),feature_size);
                }
            },
            subgraphs,
            feature_size,
            active_);
    }

    void PropagateBackwardCPU_Lockfree(NtsVar &X_grad, NtsVar &Y_grad,std::vector<CSC_segment_pinned *> &subgraphs)
    {       
        float* X_grad_buffer=graph_->Nts->getWritableBuffer(X_grad,torch::DeviceType::CPU);
        float* Y_grad_buffer=graph_->Nts->getWritableBuffer(Y_grad,torch::DeviceType::CPU);
        memset(Y_grad_buffer,0,sizeof(float)*X_grad.size(0)*X_grad.size(1));
        //int feature_size=graph_->gnnctx->layer_size[graph_->rtminfo->curr_layer];
        int feature_size=X_grad.size(1);
        float* output_buffer=new float[feature_size*graph_->threads];
        graph_->process_edges_backward_decoupled<int, float>( // For EACH Vertex Processing
            [&](VertexId src, VertexAdjList<Empty> outgoing_adj,VertexId thread_id,VertexId recv_id) {           //pull
                float* local_output_buffer=output_buffer+feature_size*thread_id;
                memset(local_output_buffer,0,sizeof(float)*feature_size);
                VertexId src_trans=src-graph_->partition_offset[recv_id];
                for(long d_idx=subgraphs[recv_id]->row_offset[src_trans];d_idx<subgraphs[recv_id]->row_offset[src_trans+1];d_idx++)
                {
                    VertexId dst=subgraphs[recv_id]->column_indices[d_idx];
                    VertexId dst_trans=dst-start_;
                    float* local_input_buffer=X_grad_buffer+(dst_trans)*feature_size;  
                    comp(local_input_buffer,local_output_buffer,norm_degree(src,dst),feature_size);   
                }
                if(graph_->rtminfo->lock_free){
                    if(subgraphs[recv_id]->source_active->get_bit(src_trans)){
                        VertexId write_index=subgraphs[recv_id]->backward_message_index[src_trans]; 
                        graph_->NtsComm->emit_buffer_lock_free(src, local_output_buffer,write_index, feature_size);
                    }
                }else{
                    graph_->NtsComm->emit_buffer(src, local_output_buffer,feature_size);
                }
            },
            [&](VertexId src, float* msg) {
                acc(msg,Y_grad_buffer+(src-start_)*feature_size,feature_size);
                return 1;
            },
            feature_size,
            active_);
            delete [] output_buffer;
    }    
    
    void PropagateForwardCPU(NtsVar &X, NtsVar &Y)
    {
        float* X_buffer=graph_->Nts->getWritableBuffer(X,torch::DeviceType::CPU);
        float* Y_buffer=graph_->Nts->getWritableBuffer(Y,torch::DeviceType::CPU);
        memset(Y_buffer,0,sizeof(float)*X.size(0)*X.size(1));
        int feature_size=graph_->gnnctx->layer_size[graph_->rtminfo->curr_layer];
        float* output_buffer=new float[feature_size*graph_->threads];
        memset(output_buffer,0,sizeof(float)*feature_size*graph_->threads);//=new float[feature_size*graph_->threads];
        graph_->process_edges_forward<int,float>( // For EACH Vertex Processing
            [&](VertexId dst, VertexAdjList<Empty> incoming_adj,VertexId thread_id) {
                float* local_output_buffer=output_buffer+feature_size*thread_id;
                memset(local_output_buffer,0,sizeof(float)*feature_size);
                for (AdjUnit<Empty> *ptr = incoming_adj.begin; ptr != incoming_adj.end; ptr++)
                { //pull model
                    VertexId src = ptr->neighbour;
                    float* local_input_buffer=X_buffer+(src-start_)*feature_size;  
                    comp(local_input_buffer,local_output_buffer,1,feature_size);
                }
               graph_->emit_buffer(dst, local_output_buffer,feature_size);
            },
            [&](VertexId dst, float* msg) {
                acc(msg,Y_buffer+(dst-start_)*feature_size,feature_size);
                return 0;
            },
            feature_size,
            active_);
            free(output_buffer);
    }
    void PropagateBackwardCPU(NtsVar &X_grad, NtsVar &Y_grad)
    {       
        float* X_grad_buffer=graph_->Nts->getWritableBuffer(X_grad,torch::DeviceType::CPU);
        float* Y_grad_buffer=graph_->Nts->getWritableBuffer(Y_grad,torch::DeviceType::CPU);
        memset(Y_grad_buffer,0,sizeof(float)*X_grad.size(0)*X_grad.size(1));
        int feature_size=graph_->gnnctx->layer_size[graph_->rtminfo->curr_layer];
        float* output_buffer=new float[feature_size*graph_->threads];
        graph_->process_edges_backward<int, float>( // For EACH Vertex Processing
            [&](VertexId src, VertexAdjList<Empty> outgoing_adj,VertexId thread_id,VertexId recv_id) {           //pull
                float* local_output_buffer=output_buffer+feature_size*thread_id;
                memset(local_output_buffer,0,sizeof(float)*feature_size);
                for (AdjUnit<Empty> *ptr = outgoing_adj.begin; ptr != outgoing_adj.end; ptr++)
                {
                    VertexId dst = ptr->neighbour;
                    float* local_input_buffer=X_grad_buffer+(dst-start_)*feature_size;  
                    comp(local_input_buffer,local_output_buffer,norm_degree(src,dst),feature_size);        
                }
                graph_->emit_buffer(src, local_output_buffer,feature_size);
            },
            [&](VertexId src, float* msg) {
                acc(msg,Y_grad_buffer+(src-start_)*feature_size,feature_size);
                return 0;
            },
            feature_size,
            active_);
    } 
    void Process_GPU_overlap_explict(NtsVar &X, NtsVar &Y, std::vector<CSC_segment_pinned *> &graph_partitions)
    {
        int feature_size=X.size(1);
        bool selective = graph_->rtminfo->reduce_comm;
        int layer = graph_->rtminfo->curr_layer;
        //if (!selective)
        {       graph_->compute_sync_decoupled<int, float>(
                X,
                graph_partitions,
                [&](VertexId src, VertexAdjList<Empty> outgoing_adj) { //pull
                    graph_->NtsComm->emit_buffer(src, graph_->output_cpu_buffer + (src)*feature_size, feature_size);
                },
                Y,
                feature_size);
        }
        //else{
        //     graph_->compute_sync_explict<int, float>(
        //        X,
        //        graph_partitions,
        //        [&](VertexId src, VertexAdjList<Empty> outgoing_adj) { //pull
        //            graph_->emit_buffer(src, graph_->output_cpu_buffer + (src)*current_layer_size, current_layer_size);
        //        },
        //        Y.packed_accessor<float, 2>().data());
        //}
    }
    
    void Process_GPU_overlap_sync_compute_explict(NtsVar &X, NtsVar &Y, std::vector<CSC_segment_pinned *> &graph_partitions)
    {
        //int current_layer_size = ;//graph_->gnnctx->layer_size[graph_->rtminfo->curr_layer];
        int feature_size=X.size(1);
        bool selective = graph_->rtminfo->reduce_comm;
        int layer = graph_->rtminfo->curr_layer;
        NtsVar X_cpu=X.cpu();
        float *X_buffered=X_cpu.accessor<float,2>().data();
        
        //if (!selective)
        { // original communication
            graph_->sync_compute_decoupled<int, float>(
                X,
                graph_partitions,
                [&](VertexId src) { 
                    graph_->NtsComm->emit_buffer(src, X_buffered+(src-graph_->gnnctx->p_v_s)*feature_size, feature_size);
                },
                Y,
                feature_size);
        }
       // else
       // { 
//            graph_->sync_compute<int, float>(
//                X,
//                graph_partitions,
//                [&](VertexId src) { //pull
//                    if (!graph_->RepVtx[layer]->get_bit(src))
//                    {
//                        graph_->emit_buffer(src, X_buffered + (src)*current_layer_size, current_layer_size);
//                    }
//                },
//                Y.packed_accessor<float, 2>().data());
        //}
    }
    
     void GraphPropagateForwardEdgeComputation(NtsVar &src_input_origin,
                                               NtsVar &dst_output,
                                               std::vector<CSC_segment_pinned *> &graph_partitions,
                                               std::function<NtsVar(NtsVar&)> PreComputation,
                                               std::function<NtsVar(NtsVar&, NtsVar&, NtsVar&, NtsVar&, NtsScheduler* nts)> EdgeComputation)
    {
        int current_layer_size = graph_->gnnctx->layer_size[graph_->rtminfo->curr_layer];
        bool selective = graph_->rtminfo->reduce_comm;
        int layer = graph_->rtminfo->curr_layer;
        NtsVar X_cpu=src_input_origin.cpu();
        float *X_buffered=X_cpu.accessor<float,2>().data();
            graph_->sync_compute_edge_decoupled<int, float>(
                src_input_origin,
                graph_partitions,
                [&](VertexId src) {
                    graph_->NtsComm->emit_buffer(src, X_buffered+(src-graph_->gnnctx->p_v_s)*current_layer_size, current_layer_size);
                },
                [&](NtsVar &d_i){
                    return PreComputation(d_i);
                },
                [&](NtsVar &s_i,NtsVar &s_i_t, NtsVar &d_i, NtsVar &d_i_t,NtsScheduler* nts){
                    return EdgeComputation(s_i,s_i_t, d_i,d_i_t,nts);
                },
                dst_output);
            //printf("done!\n");
        
    }
    void GraphPropagateBackwardEdgeComputation(NtsVar &src_input_origin,
                                               NtsVar &dst_grad_input,
                                               NtsVar &dst_grad_output,
                                               std::vector<CSC_segment_pinned *> &graph_partitions,
                                               std::function<NtsVar(NtsVar&)> PreComputation,
                                               std::function<NtsVar(NtsVar&,NtsVar&,NtsVar&,NtsVar&,NtsScheduler* nts)> EdgeComputation,
                                               std::function<NtsVar(NtsVar&,NtsVar&,NtsScheduler* nts)> EdgeBackward)
    {
        int current_layer_size = graph_->gnnctx->layer_size[graph_->rtminfo->curr_layer];
        bool selective = graph_->rtminfo->reduce_comm;
        int layer = graph_->rtminfo->curr_layer;
            //printf("done?\n");
            graph_->compute_sync_edge_decoupled<int, float>(
                dst_grad_input,
                src_input_origin,
                graph_partitions,
                [&](VertexId src, VertexAdjList<Empty> outgoing_adj) {
                    graph_->NtsComm->emit_buffer(src, graph_->output_cpu_buffer + (src)*current_layer_size, current_layer_size);
                },
                [&](NtsVar &d_i){
                    return PreComputation(d_i);
                },
                [&](NtsVar &s_i,NtsVar &s_i_t, NtsVar &d_i, NtsVar &d_i_t,NtsScheduler* nts){
                    return EdgeComputation(s_i,s_i_t, d_i,d_i_t,nts);
                },
                [&](NtsVar &b_i, NtsVar &c_i,NtsScheduler* nts){
                    return EdgeBackward(b_i, c_i, nts);
                },
                dst_grad_output);
            //printf("done!\n");
        
    }
    
   inline void GraphPropagateForward(NtsVar &X, NtsVar &Y, std::vector<CSC_segment_pinned *> &graph_partitions)
    {
       Process_GPU_overlap_sync_compute_explict(X,Y,graph_partitions);
    }

   inline void GraphPropagateBackward(NtsVar &X, NtsVar &Y, std::vector<CSC_segment_pinned *> &graph_partitions)
    {
       //Process_GPU_overlap_lite(X,Y,graph_partitions);
       Process_GPU_overlap_explict(X,Y,graph_partitions);
       
    }
   void GenerateGraphSegment(std::vector<CSC_segment_pinned *> &graph_partitions, DeviceLocation dt, std::function<float(VertexId,VertexId)>weight_compute)
    {
        graph_partitions.clear();
        int *tmp_column_offset = new int[graph_->vertices + 1];
        int *tmp_row_offset=new int[graph_->vertices + 1];
        memset(tmp_column_offset, 0, sizeof(int) * (graph_->vertices + 1));//
        memset(tmp_row_offset, 0, sizeof(int) * (graph_->vertices + 1));//
        for (int i = 0; i < graph_->graph_shard_in.size(); i++)
        {
            graph_partitions.push_back(new CSC_segment_pinned);
            graph_partitions[i]->init(graph_->graph_shard_in[i]->src_range[0],
                                      graph_->graph_shard_in[i]->src_range[1], 
                                      graph_->graph_shard_in[i]->dst_range[0],
                                      graph_->graph_shard_in[i]->dst_range[1], 
                                      graph_->graph_shard_in[i]->numofedges,dt);
            graph_partitions[i]->allocVertexAssociateData();
            graph_partitions[i]->allocEdgeAssociateData();
            graph_partitions[i]->getDevicePointerAll();
            memset(tmp_column_offset, 0, sizeof(int) * (graph_->vertices + 1));
            memset(tmp_row_offset, 0, sizeof(int) * (graph_->vertices + 1));
            for (int j = 0; j < graph_partitions[i]->edge_size; j++)
            {
                VertexId v_src_m = graph_->graph_shard_in[i]->src_delta[j];
                VertexId v_dst_m = graph_->graph_shard_in[i]->dst_delta[j];
                VertexId v_dst   = v_dst_m-graph_partitions[i]->dst_range[0];
                VertexId v_src   = v_src_m-graph_partitions[i]->src_range[0];
                
                tmp_column_offset[v_dst + 1] += 1;
                tmp_row_offset[v_src + 1] += 1;///
                //graph_partitions[i]->weight_buffer[j]=(ValueType)std::sqrt(graph->out_degree_for_backward[v_src])*(ValueType)std::sqrt(graph->in_degree_for_backward[v_dst]);
            }
            for (int j = 0; j < graph_partitions[i]->batch_size_forward; j++)
            {
                tmp_column_offset[j + 1] += tmp_column_offset[j];
                graph_partitions[i]->column_offset[j + 1] = tmp_column_offset[j + 1];
            }

            for (int j = 0; j < graph_partitions[i]->batch_size_backward; j++)///
            {
                tmp_row_offset[j + 1] += tmp_row_offset[j];
                graph_partitions[i]->row_offset[j + 1] = tmp_row_offset[j + 1];
            }
             
            for (int j = 0; j < graph_partitions[i]->edge_size; j++)
            {
                //if(graph->partition_id==0)std::cout<<"After j edges: "<<j<<std::endl;
                VertexId v_src_m = graph_->graph_shard_in[i]->src_delta[j];
                VertexId v_dst_m = graph_->graph_shard_in[i]->dst_delta[j];
                VertexId v_dst   = v_dst_m-graph_partitions[i]->dst_range[0];
                VertexId v_src   = v_src_m-graph_partitions[i]->src_range[0];
                
                
                graph_partitions[i]->source[tmp_column_offset[v_dst]] = (long)(v_src_m);
                graph_partitions[i]->destination[tmp_column_offset[v_dst]] = (long)(v_dst_m);
                graph_partitions[i]->source_backward[tmp_row_offset[v_src]]=(long)(v_src_m);
                
                graph_partitions[i]->src_set_active(v_src_m);//source_active->set_bit(v_src);
                graph_partitions[i]->dst_set_active(v_dst_m);//destination_active->set_bit(v_dst);
                
                graph_partitions[i]->row_indices[tmp_column_offset[v_dst]] = v_src_m;
                graph_partitions[i]->edge_weight_forward[tmp_column_offset[v_dst]++] = weight_compute(v_src_m,v_dst_m);
                            
                graph_partitions[i]->column_indices[tmp_row_offset[v_src]] = v_dst_m;///
                graph_partitions[i]->edge_weight_backward[tmp_row_offset[v_src]++] = weight_compute(v_src_m,v_dst_m);
            }
//                int s=0;
//                for(int l=graph_partitions[i]->src_range[0];l<graph_partitions[i]->src_range[1];l++){
//                    if(graph_partitions[i]->src_get_active(l)){
//                        s++;
//                    }
//                }
//                printf("debug_pre %d\n",s);
        }
        if(GPU_T==dt){
            int max_batch_size = 0;
            for (int i = 0; i < graph_partitions.size(); i++)
            {
                max_batch_size = std::max(max_batch_size, graph_partitions[i]->batch_size_forward);
            }
            graph_->output_gpu_buffered =graph_->Nts->NewLeafTensor({max_batch_size, graph_->gnnctx->max_layer},torch::DeviceType::CUDA);
        }
        delete[] tmp_column_offset;
        delete[] tmp_row_offset;
        if (graph_->partition_id == 0)
            printf("GNNmini::Preprocessing[Graph Segments Prepared]\n");
    }
   void GenerateMessageBitmap(std::vector<CSC_segment_pinned *> &graph_partitions){
        int feature_size=1;
        graph_->process_edges_backward<int, VertexId>( // For EACH Vertex Processing
            [&](VertexId src, VertexAdjList<Empty> outgoing_adj,VertexId thread_id,VertexId recv_id) {           //pull
                VertexId src_trans=src-graph_->partition_offset[recv_id];
                if(graph_partitions[recv_id]->source_active->get_bit(src_trans)){
                    VertexId part=(VertexId)graph_->partition_id;
                    graph_->emit_buffer(src, &part,feature_size);
                }
            },
            [&](VertexId master, VertexId* msg) {
                VertexId part=*msg;
                graph_partitions[part]->set_forward_active(master-graph_->gnnctx->p_v_s);//destination_mirror_active->set_bit(master-start_);
                return 0;
            },
            feature_size,
            active_);
            
            size_t basic_chunk=64;
        for(int i=0;i<graph_partitions.size();i++){
            graph_partitions[i]->backward_message_index=new VertexId[graph_partitions[i]->batch_size_backward];
            graph_partitions[i]->forward_message_index=new VertexId[graph_partitions[i]->batch_size_forward];
            memset(graph_partitions[i]->backward_message_index,0,sizeof(VertexId)*graph_partitions[i]->batch_size_backward);
            memset(graph_partitions[i]->forward_message_index,0,sizeof(VertexId)*graph_partitions[i]->batch_size_forward);
            int backward_write_offset=0;
            for (VertexId begin_v_i =graph_partitions[i]->src_range[0];begin_v_i<graph_partitions[i]->src_range[1]; begin_v_i += 1){
                VertexId v_i = begin_v_i;
                VertexId v_trans=v_i-graph_partitions[i]->src_range[0];
                if (graph_partitions[i]->src_get_active(v_i))
                    graph_partitions[i]->backward_message_index[v_trans]=backward_write_offset++;
            }
            
            int forward_write_offset=0;
            for (VertexId begin_v_i =graph_partitions[i]->dst_range[0];begin_v_i<graph_partitions[i]->dst_range[1]; begin_v_i += 1){
                VertexId v_i = begin_v_i;
                VertexId v_trans=v_i-graph_partitions[i]->dst_range[0];
                if (graph_partitions[i]->get_forward_active(v_trans))
                    graph_partitions[i]->forward_message_index[v_trans]=forward_write_offset++;
            }
            //printf("forward_write_offset %d\n",forward_write_offset);
        }
         if (graph_->partition_id == 0)
            printf("GNNmini::Preprocessing[Compressed Message Prepared]\n");    
        
   }
   
   void TestGeneratedBitmap(std::vector<CSC_segment_pinned *> &subgraphs){
       for(int i=0;i<subgraphs.size();i++){
           int count_act_src=0;
           int count_act_dst=0;
           int count_act_master=0;
           for(int j=subgraphs[i]->dst_range[0];j<subgraphs[i]->dst_range[1];j++){
               if(subgraphs[i]->dst_get_active(j)){
                   count_act_dst++;
               }
//               if(subgraphs[i]->to_this_part_get_active(j)){
//                   count_act_master++;
//               }
           }
           for(int j=subgraphs[i]->src_range[0];j<subgraphs[i]->src_range[1];j++){
               if(subgraphs[i]->src_get_active(j)){
                   count_act_src++;
               }
           }
        printf("PARTITION:%d CHUNK %d ACTIVE_SRC %d ACTIVE_DST %d ACTIVE_MIRROR %d\n",graph_->partition_id,i,count_act_src,count_act_dst,count_act_master);   
       } 
   }
          
   
};


#endif

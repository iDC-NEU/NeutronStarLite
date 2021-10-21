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

// #define VECTOR_LENGTH 3703
// #define WEIGHT_ROW 3703
// #define WEIGHT_COL 3703
// #define NODE_NUMBER 3327
// #define LABEL_NUMBER 6
// #define SIZE_LAYER_1    3703
// #define SIZE_LAYER_2    128
// #define OUTPUT_LAYER_3  6

// #define VECTOR_LENGTH 500
// #define WEIGHT_ROW 500
// #define WEIGHT_COL 500
// #define NODE_NUMBER 19717
// #define LABEL_NUMBER 3
// #define SIZE_LAYER_1    500
// #define SIZE_LAYER_2    128
// #define OUTPUT_LAYER_3  3

//#define VECTOR_LENGTH 1433
//#define WEIGHT_ROW 1433
//#define WEIGHT_COL 1433
//#define NODE_NUMBER 2708
//#define LABEL_NUMBER 7
//#define SIZE_LAYER_1 1433
//#define SIZE_LAYER_2    128
//#define OUTPUT_LAYER_3  7

#define VECTOR_LENGTH 100
#define WEIGHT_ROW 100
#define WEIGHT_COL 100
#define NODE_NUMBER 875713
#define LABEL_NUMBER 20
#define SIZE_LAYER_1 100
#define SIZE_LAYER_2 64
#define OUTPUT_LAYER_3 20
#define MAX_LAYER 100

/*
#define VECTOR_LENGTH 400
#define WEIGHT_ROW 400
#define WEIGHT_COL 400
#define NODE_NUMBER 1632803
#define LABEL_NUMBER 16
#define SIZE_LAYER_1 400
#define SIZE_LAYER_2 128
#define OUTPUT_LAYER_3 16
#define MAX_LAYER 400
*/
/*
 #define VECTOR_LENGTH 300
 #define WEIGHT_ROW 300
 #define WEIGHT_COL 300
 #define NODE_NUMBER 1971281
 #define LABEL_NUMBER 16
 #define SIZE_LAYER_1 300
 #define SIZE_LAYER_2   256
 #define OUTPUT_LAYER_3  16
 #define MAX_LAYER 300
*/
/*
 #define VECTOR_LENGTH 352
 #define WEIGHT_ROW 352
 #define WEIGHT_COL 352
 #define NODE_NUMBER 1696415
 #define LABEL_NUMBER 16
 #define SIZE_LAYER_1 64
 #define SIZE_LAYER_2   352
 #define OUTPUT_LAYER_3  16
 #define MAX_LAYER 352
*/

// #define VECTOR_LENGTH 300
// #define WEIGHT_ROW 300
// #define WEIGHT_COL 300
// #define NODE_NUMBER 3072626
// #define LABEL_NUMBER 7
// #define SIZE_LAYER_1 300
// #define SIZE_LAYER_2    16
// #define OUTPUT_LAYER_3  7

template <int size_>
struct nodeVector
{
    ValueType data[size_];
} __attribute__((packed));

//typedef struct factor2 {
//    float weight[WEIGHT_ROW][WEIGHT_COL];
//} weightVector;
template <typename t_v, int SIZE_>
struct EdgeFeature
{
    t_v data[SIZE_];
};
struct compress_feature
{
    int key;
    float value;
};

typedef struct graph_Tensor
{
    NtsVar src;
    NtsVar dst;
    NtsVar weight;
    int edge_size;
    int batch_size;
    int feature_size;
    int src_range[2];
    int dst_range[2];
    float *weight_buffer;
} edge_list;

struct content
{
    int id;
    ValueType att[SIZE_LAYER_1];
    long label;
}; //con[2708];

struct Compressed_Feature
{
    VertexId *column_index;
    ValueType *value_list;
    VertexId *position;
    int count;
};
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
    GNNDatum(gnncontext *_gnnctx)
    {
        gnnctx = _gnnctx;
        local_feature = new float[gnnctx->l_v_num * gnnctx->layer_size[0]];
        local_label = new long[gnnctx->l_v_num];
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
        }
    }
     void readCORA(Graph<Empty> *graph)
    {
       
        std::string str;
        std::ifstream input("cora.content", std::ios::in);
        //std::ofstream outputl("cora.labeltable",std::ios::out);
       // ID    F   F   F   F   F   F   F   L
        if (!input.is_open())
        {
            //cout<<"open file fail!"<<endl;
            return;
        }
        float *con_tmp = new float[gnnctx->layer_size[0]];
        int numOfData = 0;
        std::string la;
        //std::cout<<"finish1"<<std::endl;
        VertexId id=0;
        while (input >> id)
        {
            VertexId size_0=gnnctx->layer_size[0];
            VertexId id_trans=id-gnnctx->p_v_s;
            if((gnnctx->p_v_s<=id)&&(gnnctx->p_v_e>id)){
                for (int i = 0; i < size_0; i++){
                    input >> local_feature[size_0*id_trans+i];
                }
                input >> la;
                local_label[id_trans]= changelable(la);
            }else{
                for (int i = 0; i < size_0; i++){
                    input >> con_tmp[i];   
                }
                input >> la;
            }
        }
        free(con_tmp);
        input.close();
    }
    
    void registLabel(NtsVar &target)
    {
        target = torch::from_blob(local_label, gnnctx->l_v_num, torch::kLong);
    }
    void read_from_txt()
    {
        ;
    }
    void read_from_binary()
    {
        ;
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
    
    void PropagateForwardCPU_debug(NtsVar &X, NtsVar &Y,std::vector<CSC_segment_pinned *> &graph_partitions)
    {
        float* X_buffer=graph_->Nts->getWritableBuffer(X,torch::DeviceType::CPU);
        float* Y_buffer=graph_->Nts->getWritableBuffer(Y,torch::DeviceType::CPU);
        memset(Y_buffer,0,sizeof(float)*X.size(0)*X.size(1));
        int feature_size=graph_->gnnctx->layer_size[graph_->rtminfo->curr_layer];
        
//        std::vector<CSC_segment_pinned *> &graph_partitions,
//                                int feature_size,
//                                Bitmap *active, 
        graph_->process_edges_forward_debug<int,float>( // For EACH Vertex Processing
            [&](VertexId src) {
                    graph_->emit_buffer(src, X_buffer+(src-graph_->gnnctx->p_v_s)*feature_size, feature_size);
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
            graph_partitions,
            feature_size,
            active_);
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
                    comp(local_input_buffer,local_output_buffer,norm_degree(src,dst),feature_size);
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
            [&](VertexId src, VertexAdjList<Empty> outgoing_adj,VertexId thread_id) {           //pull
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
        int current_layer_size = graph_->gnnctx->layer_size[graph_->rtminfo->curr_layer];
        bool selective = graph_->rtminfo->reduce_comm;
        int layer = graph_->rtminfo->curr_layer;

        if (!selective)
        { // original communication
            //printf("done?\n");
            graph_->compute_sync_explict<int, float>(
                X,
                graph_partitions,
                [&](VertexId src, VertexAdjList<Empty> outgoing_adj) { //pull
                    //nodeVector<CURRENT_LAYER_SIZE> sum;
                    //memcpy(sum.data, graph_->output_cpu_buffer + (src)*current_layer_size, sizeof(t_v) * CURRENT_LAYER_SIZE);
                    graph_->emit_buffer(src, graph_->output_cpu_buffer + (src)*current_layer_size, current_layer_size);
                },
                Y.packed_accessor<float, 2>().data());
            //printf("done!\n");
        }
        else
        { //selective comunication
            graph_->compute_sync_explict<int, float>(
                X,
                graph_partitions,
                [&](VertexId src, VertexAdjList<Empty> outgoing_adj) { //pull
                    if (!graph_->RepVtx[layer]->get_bit(src))
                    {
                        // nodeVector<CURRENT_LAYER_SIZE> sum;
                        // memcpy(sum.data, graph_->output_cpu_buffer + (src)*current_layer_size, sizeof(t_v) * CURRENT_LAYER_SIZE);
                        // graph_->emit(src, sum);
                        graph_->emit_buffer(src, graph_->output_cpu_buffer + (src)*current_layer_size, current_layer_size);
                    }
                },
                Y.packed_accessor<float, 2>().data());
        }
    }
    
    void Process_GPU_overlap_sync_compute_explict(NtsVar &X, NtsVar &Y, std::vector<CSC_segment_pinned *> &graph_partitions)
    {
        int current_layer_size = graph_->gnnctx->layer_size[graph_->rtminfo->curr_layer];
        bool selective = graph_->rtminfo->reduce_comm;
        int layer = graph_->rtminfo->curr_layer;
        NtsVar X_cpu=X.cpu();
        float *X_buffered=X_cpu.accessor<float,2>().data();
        //float *X_buffered1=new float[current_layer_size];
        if (!selective)
        { // original communication
            //printf("done?\n");
            graph_->sync_compute<int, float>(
                X,
                graph_partitions,
                [&](VertexId src) { //push
//                    if(src==400000){
//                        printf("data %f\n",*(X_buffered+(src-graph_->gnnctx->p_v_s)*current_layer_size));
//                    }
                    graph_->emit_buffer(src, X_buffered+(src-graph_->gnnctx->p_v_s)*current_layer_size, current_layer_size);
                },
                Y.packed_accessor<float, 2>().data());
            //printf("done!\n");
        }
        else
        { //selective comunication
            graph_->sync_compute<int, float>(
                X,
                graph_partitions,
                [&](VertexId src) { //pull
                    if (!graph_->RepVtx[layer]->get_bit(src))
                    {
                        graph_->emit_buffer(src, X_buffered + (src)*current_layer_size, current_layer_size);
                    }
                },
                Y.packed_accessor<float, 2>().data());
        }
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
            //printf("done?\n");
            graph_->sync_compute_edge_computation<int, float>(
                src_input_origin,
                graph_partitions,
                [&](VertexId src) {
                    graph_->emit_buffer(src, X_buffered+(src-graph_->gnnctx->p_v_s)*current_layer_size, current_layer_size);
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
            graph_->compute_sync_edge_computation<int, float>(
                dst_grad_input,
                src_input_origin,
                graph_partitions,
                [&](VertexId src, VertexAdjList<Empty> outgoing_adj) {
                    graph_->emit_buffer(src, graph_->output_cpu_buffer + (src)*current_layer_size, current_layer_size);
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
   
     void Process_GPU_overlap_lite(NtsVar &X, NtsVar &Y, std::vector<CSC_segment_pinned *> &graph_partitions)
    {
        int current_layer_size = graph_->gnnctx->layer_size[graph_->rtminfo->curr_layer];
        bool selective = graph_->rtminfo->reduce_comm;
        int layer = graph_->rtminfo->curr_layer;

        if (!selective)
        { // original communication
            //printf("done?\n");
            graph_->compute_sync_lite<int, float>(
                X,
                graph_partitions,
                [&](VertexId src, VertexAdjList<Empty> outgoing_adj) { //pull
                    //nodeVector<CURRENT_LAYER_SIZE> sum;
                    //memcpy(sum.data, graph_->output_cpu_buffer + (src)*current_layer_size, sizeof(t_v) * CURRENT_LAYER_SIZE);
                    graph_->emit_buffer(src, graph_->output_cpu_buffer + (src)*current_layer_size, current_layer_size);
                },
                Y.packed_accessor<float, 2>().data());
            //printf("done!\n");
        }
        else
        { //selective comunication
            graph_->compute_sync_lite<int, float>(
                X,
                graph_partitions,
                [&](VertexId src, VertexAdjList<Empty> outgoing_adj) { //pull
                    if (!graph_->RepVtx[layer]->get_bit(src))
                    {
                        // nodeVector<CURRENT_LAYER_SIZE> sum;
                        // memcpy(sum.data, graph_->output_cpu_buffer + (src)*current_layer_size, sizeof(t_v) * CURRENT_LAYER_SIZE);
                        // graph_->emit(src, sum);
                        graph_->emit_buffer(src, graph_->output_cpu_buffer + (src)*current_layer_size, current_layer_size);
                    }
                },
                Y.packed_accessor<float, 2>().data());
        }
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
   void GenerateGraphSegment(std::vector<CSC_segment_pinned *> &graph_partitions, bool overlap = false)
    {
        graph_partitions.clear();
        int *tmp_column_offset = new int[graph_->vertices + 1];
        int *tmp_row_offset=new int[graph_->vertices + 1];
        memset(tmp_column_offset, 0, sizeof(int) * (graph_->vertices + 1));//
        memset(tmp_row_offset, 0, sizeof(int) * (graph_->vertices + 1));//
        for (int i = 0; i < graph_->graph_shard_in.size(); i++)
        {
            //int i=0;
            graph_partitions.push_back(new CSC_segment_pinned);
            memset(tmp_column_offset, 0, sizeof(int) * (graph_->vertices + 1));
            memset(tmp_row_offset, 0, sizeof(int) * (graph_->vertices + 1));
            graph_partitions[i]->edge_size = graph_->graph_shard_in[i]->numofedges;
            graph_partitions[i]->dst_range[0] = graph_->graph_shard_in[i]->dst_range[0]; //all src is the same
            graph_partitions[i]->dst_range[1] = graph_->graph_shard_in[i]->dst_range[1];
            graph_partitions[i]->src_range[0] = graph_->graph_shard_in[i]->src_range[0]; //all dst is the same
            graph_partitions[i]->src_range[1] = graph_->graph_shard_in[i]->src_range[1];
            
            long column_offset_size = graph_partitions[i]->dst_range[1] - graph_partitions[i]->dst_range[0] + 1;
            long row_offset_size = graph_partitions[i]->src_range[1] - graph_partitions[i]->src_range[0] + 1;///
            
            graph_partitions[i]->batch_size_forward = column_offset_size-1;
            graph_partitions[i]->batch_size_backward = row_offset_size-1;
            graph_partitions[i]->feature_size = graph_->gnnctx->layer_size[0];
            graph_partitions[i]->column_offset = (VertexId *)cudaMallocPinned(column_offset_size * sizeof(VertexId));                 //torch::zeros({1,column_offset_size},torch::kInt32);
            graph_partitions[i]->row_indices = (VertexId *)cudaMallocPinned((graph_partitions[i]->edge_size + 1) * sizeof(VertexId));
            graph_partitions[i]->edge_weight_forward = (float *)cudaMallocPinned((graph_partitions[i]->edge_size + 1) * sizeof(VertexId));
            
            graph_partitions[i]->row_offset = (VertexId *)cudaMallocPinned(row_offset_size * sizeof(VertexId));///
            graph_partitions[i]->column_indices = (VertexId *)cudaMallocPinned((graph_partitions[i]->edge_size + 1) * sizeof(VertexId));///
            graph_partitions[i]->edge_weight_backward = (float *)cudaMallocPinned((graph_partitions[i]->edge_size + 1) * sizeof(VertexId));///

            //torch::zeros({1,graph_partitions[i]->edge_size},torch::kInt32);
            graph_partitions[i]->destination = (long *)cudaMallocPinned((graph_partitions[i]->edge_size + 1) * sizeof(long));
            graph_partitions[i]->source      = (long *)cudaMallocPinned((graph_partitions[i]->edge_size + 1) * sizeof(long));
            
            graph_partitions[i]->column_offset_gpu = (VertexId *)getDevicePointer(graph_partitions[i]->column_offset);
            graph_partitions[i]->row_indices_gpu = (VertexId *)getDevicePointer(graph_partitions[i]->row_indices);
            graph_partitions[i]->edge_weight_forward_gpu = (float *)getDevicePointer(graph_partitions[i]->edge_weight_forward);
            
            graph_partitions[i]->row_offset_gpu = (VertexId *)getDevicePointer(graph_partitions[i]->row_offset);///
            graph_partitions[i]->column_indices_gpu = (VertexId *)getDevicePointer(graph_partitions[i]->column_indices);///
            graph_partitions[i]->edge_weight_backward_gpu = (float *)getDevicePointer(graph_partitions[i]->edge_weight_backward);/// 
            
            
            graph_partitions[i]->source_gpu = (long *)getDevicePointer(graph_partitions[i]->source);///
            graph_partitions[i]->destination_gpu = (long *)getDevicePointer(graph_partitions[i]->destination);///
            

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
            for (int j = 0; j < column_offset_size - 1; j++)
            {
                tmp_column_offset[j + 1] += tmp_column_offset[j];
                graph_partitions[i]->column_offset[j + 1] = tmp_column_offset[j + 1];
            }
            
            for (int j = 0; j < row_offset_size - 1; j++)///
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
                
                graph_partitions[i]->row_indices[tmp_column_offset[v_dst]] = v_src_m;
                graph_partitions[i]->edge_weight_forward[tmp_column_offset[v_dst]++] = norm_degree(v_src_m,v_dst_m);
                            
                graph_partitions[i]->column_indices[tmp_row_offset[v_src]] = v_dst_m;///
                graph_partitions[i]->edge_weight_backward[tmp_row_offset[v_src]++] = norm_degree(v_src_m,v_dst_m);
            }
        }
        {
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
            printf("GNNmini::Preprocessing[Prepare Forward CSC Edges for GPU]\n");
    }
   
};


/*
void inference(NtsVar tt_cpu, Graph<Empty> *graph, Embeddings<ValueType, long> *embedding,
               tensorSet *pytool, GnnUnit *Gnn_v1, GnnUnit *Gnn_v2)
{
    int correct = 0;
    for (int k = 0; k < embedding->rownum; k++)
    {
        ValueType max = -100000.0;
        long id = -1;
        for (int i = 0; i < LABEL_NUMBER; i++)
        {
            if (max < tt_cpu.accessor<ValueType, 2>().data()[k * LABEL_NUMBER + i])
            {
                max = tt_cpu.accessor<ValueType, 2>().data()[k * LABEL_NUMBER + i];
                id = i;
            }
        }
        if (id == pytool->target.accessor<long, 1>().data()[k])
            correct += 1;
    }
    std::cout << "\ncorrect number on training:" << correct << "\t" << ((ValueType)correct / (ValueType)graph->vertices) << std::endl;
    std::cout << "loss at" << graph->partition_id << "is :" << pytool->loss << std::endl;
    int correct_test = 0;
    for (int k = graph->vertices; k < NODE_NUMBER; k++)
    {
        ValueType max = -100000.0;
        long id = -1;
        NtsVar test = torch::from_blob(&(embedding->con[k].att[0]), {1, SIZE_LAYER_1});
        NtsVar final_ = torch::relu(test.mm(Gnn_v1->W.cpu())).mm(Gnn_v2->W.cpu()).log_softmax(1);
        for (int i = 0; i < LABEL_NUMBER; i++)
        {
            if (max < final_.accessor<ValueType, 2>().data()[i])
            {
                max = final_.accessor<ValueType, 2>().data()[i];
                id = i;
            }
        }
        if (id == embedding->con[k].label)
            correct_test += 1;
    }
    std::cout << "\ncorrect number on testing:" << correct_test << "\t" << ((ValueType)correct_test / (ValueType)(NODE_NUMBER - graph->vertices)) << std::endl;
}
 * 
 */
#endif

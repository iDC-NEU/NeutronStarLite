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
#ifndef GCN_HPP
#define GCN_HPP
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <assert.h>

#include <map>
#include "core/graph.hpp"
#include <unistd.h>
#include <math.h>
#include "torch/torch.h"
#include "torch/csrc/autograd/generated/variable_factories.h"
#include "torch/nn/module.h"
#include "comm/Network.hpp"
#include "cuda/test.hpp"
#include "core/input.hpp"
const double d = (double) 0.8;
#define VECTOR_LENGTH 1433
#define WEIGHT_ROW 1433
#define WEIGHT_COL 1433
#define EDGE_LENGTH  4



#define NODE_NUMBER 2708
#define LABEL_NUMBER 7
#define SIZE_LAYER_1    1433
#define SIZE_LAYER_2    16 
#define OUTPUT_LAYER_3  7

template<int size_>
struct nodeVector{
    ValueType data[size_];
}__attribute__((packed));

//typedef struct factor2 {
//    float weight[WEIGHT_ROW][WEIGHT_COL];
//} weightVector;
template <typename t_v,int SIZE_>
struct EdgeFeature{
    t_v data[SIZE_];
};

struct content
{
    int id;
    ValueType att[SIZE_LAYER_1];
    long label;
};//con[2708];

long changelable(std::string la)
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

struct Intermediate : torch::nn::Module {
    torch::Tensor W;
    ValueType *W_from;

    Intermediate(size_t w, size_t h) {
        //        at::TensorOptions *opt=new at::TensorOptions;
        //       opt->requires_grad(true);
        //  torch::randn
        //     A=torch::randn(torch::randn({w,h},opt));
        W = register_parameter("W", torch::randn({w, h}));
        W_from=new ValueType[w*h];

    }
    void resetW(size_t w,size_t h,ValueType* buffer){
        memcpy(W_from, buffer,sizeof(ValueType)*w*h);
        torch::Tensor new_weight_tensor = torch::from_blob(W_from,{w, h});
        W.set_data(new_weight_tensor);
    }
    
    torch::Tensor forward(torch::Tensor x) {
        x = x.mm(W);
        return x;
    }
};


struct GnnUnit : torch::nn::Module {
    torch::Tensor W;
    ValueType *W_from;
    ValueType *W_from_gpu;
    //gpu_processor *gp;
    GnnUnit(size_t w, size_t h) {
        //        at::TensorOptions *opt=new at::TensorOptions;
        //       opt->requires_grad(true);
        //  torch::randn
        //     A=torch::randn(torch::randn({w,h},opt));
        W = register_parameter("W", torch::randn({w, h},torch::kFloat));
        W_from=new ValueType[w*h];

    }
    void resetW(size_t w,size_t h,ValueType* buffer){
        memcpy(W_from, buffer,sizeof(ValueType)*w*h);
        torch::Tensor new_weight_tensor = torch::from_blob(W_from,{w, h});
        W.set_data(new_weight_tensor);
    }
    
    void learn(torch::Tensor from, ValueType learning_rate){
        torch::Tensor a=(W-(from*learning_rate));
//        int r=0,c=0;
//        r=a.size(0);
//        c=a.size(1);
//        memcpy(W_from, a.accessor<float,2>().data(),sizeof(float)*r*c);
//        torch::Tensor new_weight_tensor = torch::from_blob(W_from,{r, c});
        W.set_data(a);
    }
    void learn_gpu(torch::Tensor from, ValueType learning_rate){
        torch::Tensor a=(W-(from*learning_rate));
        int r=0,c=0;
        r=a.size(0);
        c=a.size(1);
        W.set_data(a);
        //W=a;
    }
    torch::Tensor forward(torch::Tensor x) {

        torch::Tensor x1 = x.mm(W);
        return x1;
    }

    torch::Tensor forward2(torch::Tensor x) {
        return torch::sigmoid(x);
    }
    torch::Tensor forward3(torch::Tensor x) {

        x = x.mm(W);
        return x.log_softmax(1);
    }
};

template <typename T_v, typename T_l>
class Embeddings {
public:

    Embeddings() {

    }
    T_v* start_v = NULL;
 //   T_v** partial_grad=NULL;
    GnnUnit *Gnn_v1 = NULL;
    GnnUnit *Gnn_v2 =NULL;
    T_l *label = NULL;
    T_v* local_grad=NULL;
    T_v* aggre_grad=NULL;
    
    int rownum;
    int start;
    int start_vertex;
    content *con;
    std::vector<origin_feature> con1;

    void init(Graph<Empty>* graph) {
        start_v = new ValueType [graph->vertices*SIZE_LAYER_1];//graph->alloc_vertex_array<float>(VECTOR_LENGTH);

        label = graph->alloc_vertex_array<T_l>();
        rownum = (graph->partition_offset[graph->partition_id + 1] - graph->partition_offset[graph->partition_id]);
        start = SIZE_LAYER_1 * (graph->partition_offset[graph->partition_id]);
        start_vertex = graph->partition_offset[graph->partition_id];
        printf("graph %d\n",graph->vertices);
        
    }
    void initStartWith( int index, ValueType with,int i) {
 
        *(start_v + index * SIZE_LAYER_1 + i) = with;
            //   curr[index].data[i] = (float)with;
      
    }
    void initStartWith( int index, ValueType with) {
        for (int i = 0; i < SIZE_LAYER_1; i++) {
            *(start_v + index * SIZE_LAYER_1 + i) = with;
            //   curr[index].data[i] = (float)with;
        }
    }
    void readlabel(Graph<Empty>* graph) {
        graph->fill_vertex_array(label, (long) 1);
    }

    void readlabel1(Graph<Empty>* graph) {
        //graph->fill_vertex_array(label, (long) 1);
        std::string str;
        std::ifstream input("cora.content",std::ios::in);
        if(!input.is_open())
        {
            //cout<<"open file fail!"<<endl;
            return ;
        }
        con=new content[NODE_NUMBER];
        int numOfData=0;
        std::string la;
        std::cout<<"finish1"<<std::endl;
        while(input>>con[numOfData].id)
        {
            for(int i=0;i<SIZE_LAYER_1;i++)
            {
                input>>con[numOfData].att[i];
            }
            input>>la;
            //cout<<"==============================="<<endl;
            con[numOfData].label=changelable(la);
         //   std::cout<<"lable: "<<con[numOfData].label<<" "<<numOfData<<std::endl;
            numOfData++;
        }
    //    std::cout<<"finish1"<<std::endl;
        for(int i=0;i<NODE_NUMBER;i++){
            if(i<graph->vertices)
            label[i]=con[i].label;
        }
    //    std::cout<<"finish"<<std::endl;
        input.close();
    }

    void readlebel_pub(Graph<Empty>* graph){
            std::string str;
    //readfeature
    std::ifstream input("./pubmed_data/fea.txt",std::ios::in);
    if(!input.is_open())
    {
        std::cout<<"open file fail!"<<std::endl;
        return 1;
    }
    int n=0;
    std::string lable;
    while(n<NODE_NUMBER)
    {
        con[n].id=n;
        for(int i=0;i<SIZE_LAYER_1;i++)
        {
            input>>con[n].att[i];
        }
        //std::cout<<std::endl;
        n++;
    }
    input.close();


    //readlabel
    std::ifstream inputlabel("./pubmed_data/y.txt",std::ios::in);
    if(!inputlabel.is_open())
    {
        std::cout<<"open y file fail!"<<std::endl;
        return 1;
    }
    int l=0;
    while(l<NODE_NUMBER)
    {
        inputlabel>>con[l].id;
        l++;
    }
    inputlabel.close();
    }
    
void readData_txt(Graph<Empty>*graph, 
    std::string feature_fname,std::string label_fname){
    std::ifstream input_feature(feature_fname,std::ios::in);
    if(!input_feature.is_open()){
        std::cout<<"open "<<feature_fname<<" fail!"<<std::endl;
        return 1;
    }
    std::ifstream input_label(label_fname,std::ios::in);
    if(!input_label.is_open())
    {
        std::cout<<"open y file fail!"<<std::endl;
        return 1;
    }

    int f_size=0;
    int tmp_buffer;
    while (f_size<graph->partition_offset[graph->partition_id+1]){
        if(f_size>=graph->partition_offset[graph->partition_id]){
            origin_feature f_unit;
            f_unit.id=f_size;
            f_unit.att.resize(SIZE_LAYER_1,0);
            input_label>>f_unit.label;
            for(int i=0;i<SIZE_LAYER_1;i++){
                input_feature>>f_unit.att[i];
            }  
            con1.push_back(f_unit);
        }else{
            input_label>>tmp_buffer;
            for(int i=0;i<SIZE_LAYER_1;i++){
                input_feature>>tmp_buffer;
            }   
        }
    }
        input_label.close();
        input_feature.close();
    }

void readData_binary(Graph<Empty>*graph, 
    std::string feature_fname,std::string label_fname){
    std::ifstream input_feature(feature_fname,std::ios::in);
    if(!input_feature.is_open()){
        std::cout<<"open "<<feature_fname<<" fail!"<<std::endl;
        return 1;
    }
    std::ifstream input_label(label_fname,std::ios::in);
    if(!input_label.is_open())
    {
        std::cout<<"open y file fail!"<<std::endl;
        return 1;
    }

    int f_size=0;
    int tmp_buffer;
    while (f_size<graph->partition_offset[graph->partition_id+1]){
        if(f_size>=graph->partition_offset[graph->partition_id]){
            origin_feature f_unit;
            f_unit.id=f_size;
            f_unit.att.resize(SIZE_LAYER_1,0);
            input_label>>f_unit.label;
            for(int i=0;i<SIZE_LAYER_1;i++){
                input_feature>>f_unit.att[i];
            }  
            con1.push_back(f_unit);
        }else{
            input_label>>tmp_buffer;
            for(int i=0;i<SIZE_LAYER_1;i++){
                input_feature>>tmp_buffer;
            }   
        }
    }
        input_label.close();
        input_feature.close();
    }

    
};
class tensorSet{
public:  
    std::vector<torch::optim::SGD> optimizers;
    std::vector<torch::Tensor> x;// after graph engine;
    std::vector<torch::Tensor> y;
    std::vector<torch::Tensor> localGrad;
    std::vector<torch::Tensor> backwardGrad;
    torch::Tensor target; //read label
    torch::Tensor loss;
    torch::Tensor in_degree;
    torch::Tensor out_degree;
    
    int layers=0;
    tensorSet(int layers_){
        for(int i=0;i<layers_;i++){
        x.push_back(torch::tensor(0.0,torch::kFloat));
        y.push_back(torch::tensor(0.0,torch::kFloat));
        localGrad.push_back(torch::tensor(0.0,torch::kFloat));
        backwardGrad.push_back(torch::tensor(0.0,torch::kFloat));
        layers=layers_;
        }
        
    }
    void registOptimizer(torch::optim::SGD opt){
        opt.zero_grad();
        optimizers.push_back(opt);     
    }

    void updateX(int layer_,torch::Tensor src){
            x[layer_]=src;    
    }

template <typename T_l>   
    void registLabel(T_l* label, int start, int rownum){
        target=torch::from_blob(label+start, rownum,torch::kLong);     
    }
};

void init_parameter(Network<ValueType> * comm,Graph<Empty> * graph,GnnUnit* gnn,Embeddings<ValueType,long>* embedding){

     int r,c;
     r=gnn->W.accessor<ValueType,2>().size(0);
     c=gnn->W.accessor<ValueType,2>().size(1);
    if(graph->partition_id==0){//first layer
       comm->wrtBuffertoBuff(gnn->W.accessor<ValueType,2>().data(),r, c);
    }
     comm->broadcastW(gnn->W.accessor<ValueType,2>().data(),r,c);// comm buffer
     
     gnn->resetW(r,c,comm->buffer);
}
template<typename t_v>
torch::Tensor unified_parameter(Network<ValueType> *comm,torch::Tensor res){
        
   //return res;
    int r=res.accessor<ValueType,2>().size(0);
    int c=res.accessor<ValueType,2>().size(1);
        comm->wrtBuffertoBuff(res.accessor<ValueType,2>().data(),r,c); //write para to send buffer
        comm->gatherW(r,c);// gather from others to recvbuffer
        comm->computeW(r,c);// compute new para on recvbuffer
        comm->broadcastW(comm->recv_buffer,r,c);
        return torch::from_blob(comm->buffer,{r,c});
}

template<typename t_v,typename t_l,int MAX_EMBEDDING_SIZE>
class GTensor{
    
public:
    Graph<Empty> *graph_;
    Embeddings<t_v,t_l> *embedding_;
    t_v*  edgeValue;
    
    torch::Tensor pre_value;
    torch::Tensor value;  
    torch::Tensor pre_grad;
    torch::Tensor grad;
    
    
    
    t_v** value_buffer;
    t_v** grad_buffer;
    std::vector<torch::Tensor> value_local;
    std::vector<torch::Tensor> grad_local;
    t_v** partial_grad_buffer;
    
    
    void ** curr;
    void ** next;
    void ** commbuffer;
    t_v* data_coomm;
    nodeVector<MAX_EMBEDDING_SIZE> *curr_local;
    nodeVector<MAX_EMBEDDING_SIZE> *next_local;
    
    
    int start_;
    int rownum_;
    int layer_;
    int current_layer;
    const int s=0;
    VertexSubset *active_;
    
    
    int* size_at_layer;
    
    inline void set_size(std::vector<int>size_){

    }
    inline void zeroCurr(){
        memset(curr_local, 0,sizeof(nodeVector<MAX_EMBEDDING_SIZE>)*rownum_);
    }
//    inline void zeroCurr(int layer){
//        memset(curr[layer], 0,sizeof(nodeVector<VECTOR_LENGTH>)*rownum_);
//    }
    inline void zeroNext(){
        memset(next_local, 0,sizeof(nodeVector<MAX_EMBEDDING_SIZE>)*rownum_);
    }
//    inline void zeroNext(int layer){
//        memset(next_local, 0,sizeof(nodeVector<VECTOR_LENGTH>)*rownum_);
//    }
    inline void cpToCurr(int vtx,t_v* src){
        memcpy(curr_local[vtx-start_].data, src, sizeof(nodeVector<MAX_EMBEDDING_SIZE>));
    }
template<int EBDING_SIZE>
    inline void cpToCurr(int vtx,t_v* src){
        memcpy(((nodeVector<EBDING_SIZE>*)curr_local)[vtx-start_].data, src, sizeof(nodeVector<EBDING_SIZE>));
    }
template<int EBDING_SIZE>
    inline void cpToNext(int vtx,t_v* src){
        memcpy(((nodeVector<EBDING_SIZE>*)next_local)[vtx-start_].data, src, sizeof(nodeVector<EBDING_SIZE>));
    }

    inline void cpToNext(int vtx,t_v* src){
        memcpy(next_local[vtx-start_].data, src, sizeof(nodeVector<MAX_EMBEDDING_SIZE>));
    }

    inline void addToNext(int vtx, t_v* src){
        for(int i=0;i<MAX_EMBEDDING_SIZE;i++){
            next_local[vtx-start_].data[i]+=src[i];
        }
    }
template<int EBDING_SIZE>
    inline void addToNext(int vtx, t_v* src){
        for(int i=0;i<EBDING_SIZE;i++){
            ((nodeVector<EBDING_SIZE>*)next_local)[vtx-start_].data[i]+=src[i];
        }
    }
template<int EBDING_SIZE>
    inline void addToNextValue(int layer, int vtx, t_v* src){
        for(int i=0;i<EBDING_SIZE;i++){
            ((nodeVector<EBDING_SIZE>*)value_buffer[layer])[vtx-start_].data[i]+=src[i];
        }
    }
template<int EBDING_SIZE>
    inline void addToNextGrad(int layer, int vtx, t_v* src){
        for(int i=0;i<EBDING_SIZE;i++){
            ((nodeVector<EBDING_SIZE>*)grad_buffer[layer])[vtx-start_].data[i]+=src[i];
        }
    }

    inline void zeroValue(int layer){
        memset(value_buffer[layer], 0,sizeof(t_v)*size_at_layer[layer]*rownum_);
        // memcpy(valuebuffer[vtx-start_].data, src, sizeof(nodeVector));
    }
    inline void zeroGrad(int layer){
        memset(grad_buffer[layer], 0, sizeof(t_v)*size_at_layer[layer]*rownum_);
        // memcpy(valuebuffer[vtx-start_].data, src, sizeof(nodeVector));
    }
    GTensor(Graph<Empty> * graph,Embeddings<t_v,t_l>*embedding,VertexSubset * active, int layer,std::vector<int>size_){
        graph_=graph;
        embedding_=embedding;
        active_=active;
        start_=graph->partition_offset[graph->partition_id];
        rownum_=graph->partition_offset[graph->partition_id+1]-graph->partition_offset[graph->partition_id];
        layer_=layer;
        current_layer=-1;
        size_at_layer= new int[layer+1];
        data_coomm=new t_v[SIZE_LAYER_1*graph->vertices];
       for(int i=0;i<layer_+1;i++){
            size_at_layer[i]=size_[i];
        }
       // partial_grad_buffer=new t_v[VECTOR_LENGTH*VECTOR_LENGTH];
        value_buffer=new t_v*[layer];
        grad_buffer=new t_v*[layer];
//        curr= new void*[layer];
//        next= new void*[layer];
        partial_grad_buffer=new t_v*[layer];
    
        curr_local=graph->alloc_vertex_array_local<nodeVector<MAX_EMBEDDING_SIZE>>();
        next_local=graph->alloc_vertex_array_local<nodeVector<MAX_EMBEDDING_SIZE>>();
        value_local.clear();
        grad_local.clear();
        for(int i=0;i<layer;i++){
            torch::Tensor x1,x2;
            value_local.push_back(x1);
            grad_local.push_back(x2);
            value_buffer[i]=graph->alloc_vertex_array_local<t_v>(size_at_layer[i]);
            grad_buffer[i]=graph->alloc_vertex_array_local<t_v>(size_at_layer[i]);
//            curr[layer]=(void*)graph->alloc_vertex_array_local<nodeVector<size_at_layer[layer]>>();
//            next[layer]=(void*)graph->alloc_vertex_array_local<nodeVector<size_at_layer[layer]>>();
          
        }
        graph_->process_vertices<t_v>(//init  the vertex state.
                [&](VertexId vtx) {
                    graph->in_degree_for_backward[vtx]=graph->in_degree_for_backward[vtx]+1;//local
                    graph->out_degree_for_backward[vtx]=graph->out_degree_for_backward[vtx]+1;//local
                    return (ValueType) 1;
                }, active_ );   
        
    }
    torch::Tensor applyVertex(int local_layer, torch::Tensor vertex_data){
        torch::Tensor destination;
        bool s=value_local[local_layer].accessor<ValueType,2>().size(0)==vertex_data.accessor<ValueType,2>().size(0);
        bool t=value_local[local_layer].accessor<ValueType,2>().size(1)==vertex_data.accessor<ValueType,2>().size(1);
        //assert(source.accessor<float,2>().size(0)==vertex_data.accessor<float,2>().size(0));
        //assert(source.accessor<float,2>().size(1)==vertex_data.accessor<float,2>().size(1));
        assert((s&&t)==1);
        destination=value_local[local_layer]*vertex_data;
    }
    torch::Tensor applyVertexPre(torch::Tensor source,torch::Tensor vertex_data){
        torch::Tensor destination;
        bool s=source.accessor<ValueType,2>().size(0)==vertex_data.accessor<ValueType,2>().size(0);
        bool t=source.accessor<ValueType,2>().size(1)==vertex_data.accessor<ValueType,2>().size(1);
        //assert(source.accessor<float,2>().size(0)==vertex_data.accessor<float,2>().size(0));
        //assert(source.accessor<float,2>().size(1)==vertex_data.accessor<float,2>().size(1));
        assert((s&&t)==1);
        return source*vertex_data;
    }
template<int CURRENT_LAYER_SIZE> 
    torch::Tensor Propagate(int local_layer){
        zeroValue(local_layer);
        
    graph_->process_vertices<t_v>(//init  the vertex state.
                [&](VertexId vtx) {
                    cpToCurr<CURRENT_LAYER_SIZE>(vtx,pre_value.accessor<t_v,2>().data()+(vtx-start_)*CURRENT_LAYER_SIZE);//local
                    return (ValueType) 1;
                }, active_ );   
    // printf("unfinish propagate at layer [%d]\n",local_layer);
    graph_->process_edges<int, nodeVector<CURRENT_LAYER_SIZE>>(// For EACH Vertex Processing
        [&](VertexId dst, VertexAdjList<Empty> incoming_adj) {//pull
            nodeVector<CURRENT_LAYER_SIZE> sum;
            memset(sum.data,0,sizeof(ValueType)*CURRENT_LAYER_SIZE);
            for (AdjUnit<Empty> * ptr = incoming_adj.begin; ptr != incoming_adj.end; ptr++) {//pull model
                VertexId src = ptr->neighbour;
                for (int i = 0; i < CURRENT_LAYER_SIZE; i++) {
                    sum.data[i] += ((nodeVector<CURRENT_LAYER_SIZE>*)curr_local)[src-start_].data[i]/((ValueType)std::sqrt(graph_->out_degree_for_backward[src])*(ValueType)std::sqrt(graph_->in_degree_for_backward[dst]));
                }
            }
            graph_->emit(dst, sum);
        },
        [&](VertexId dst, nodeVector<CURRENT_LAYER_SIZE> msg) {
            //addToNext<CURRENT_LAYER_SIZE>(dst,msg.data);//local
            addToNextValue<CURRENT_LAYER_SIZE>(local_layer,dst,msg.data);
            return 0;
        }, active_);
        
        value_local[local_layer]=torch::from_blob(value_buffer[local_layer],{rownum_,CURRENT_LAYER_SIZE},torch::kFloat);
        current_layer=local_layer;
        if(graph_->partition_id==0)
        printf("finish propagate at layer [%d]\n",local_layer);
        return value_local[local_layer];
        
    }
template<int CURRENT_LAYER_SIZE>   
    void Propagate_backward(int local_layer){
        zeroGrad(local_layer);//local
      graph_->process_vertices<ValueType>(//init  the vertex state.
                [&](VertexId vtx) {
                    cpToNext<CURRENT_LAYER_SIZE>(vtx,pre_grad.accessor<t_v,2>().data()+(vtx-start_)*CURRENT_LAYER_SIZE );//local
                     return 1;
                }, active_ );  
        //start graph engine.
      graph_->process_edges_backward<int, nodeVector<CURRENT_LAYER_SIZE>>(// For EACH Vertex Processing
        [&](VertexId src, VertexAdjList<Empty> outgoing_adj) {//pull
            nodeVector<CURRENT_LAYER_SIZE> sum;
            memset(sum.data,0,sizeof(ValueType)*CURRENT_LAYER_SIZE);
            for (AdjUnit<Empty> * ptr = outgoing_adj.begin; ptr != outgoing_adj.end; ptr++) {//pull model
                VertexId dst = ptr->neighbour;
                for (int i = 0; i < CURRENT_LAYER_SIZE; i++) {
                     sum.data[i] += ((nodeVector<CURRENT_LAYER_SIZE>*)next_local)[dst-start_].data[i]/(
                        (ValueType)std::sqrt(graph_->out_degree_for_backward[src])*(ValueType)std::sqrt(graph_->in_degree_for_backward[dst]));
                }
            }
            graph_->emit(src, sum);
        },
        [&](VertexId src, nodeVector<CURRENT_LAYER_SIZE> msg) {
            addToNextGrad<CURRENT_LAYER_SIZE>(local_layer,src,msg.data);
            return 0;
        }, active_ );    

      grad_local[local_layer]=torch::from_blob(grad_buffer[local_layer],{rownum_,CURRENT_LAYER_SIZE});//local
      grad=grad_local[local_layer];
      if(graph_->partition_id==0)
        printf("finish backward propagate at layer [%d]\n",local_layer);
      
    }

template<int CURRENT_LAYER_SIZE>   
    void Sync_data(torch::Tensor& data,torch::Tensor &data_to){
        data_to.zero_();
        //std::cout<<"start"<<start_<<" "<<data.accessor<t_v,2>().size(1)<<std::endl;
        //memset(data_coomm,0,sizeof(t_v)*CURRENT_LAYER_SIZE*graph_->vertices);
      graph_->process_edges<int, nodeVector<CURRENT_LAYER_SIZE>>(// For EACH Vertex Processing
        [&](VertexId src, VertexAdjList<Empty> outgoing_adj) {//pull
        // if(src>=2200){
        //     std::cout<<"yes "<<src<<" is called"<<std::endl;
        // }
            nodeVector<CURRENT_LAYER_SIZE> sum;
            memcpy(sum.data,data.accessor<t_v,2>().data()+(src)*CURRENT_LAYER_SIZE,sizeof(t_v)*CURRENT_LAYER_SIZE);
            graph_->emit(src, sum);
        },
        [&](VertexId src, nodeVector<CURRENT_LAYER_SIZE> msg) {
            //memset(data_coomm+(src)*CURRENT_LAYER_SIZE,0,sizeof(t_v)*CURRENT_LAYER_SIZE);
            for(int i=0;i<CURRENT_LAYER_SIZE;i++){
               //data_coomm[(src)*CURRENT_LAYER_SIZE+i]+=msg.data[i];
               data_to.accessor<t_v,2>().data()[(src-start_)*CURRENT_LAYER_SIZE+i]+=msg.data[i]; 
            }
            return 0;
        }, active_ ); 
        //return torch::from_blob(data_coomm+(start_)*CURRENT_LAYER_SIZE,{rownum_,CURRENT_LAYER_SIZE},torch::kFloat);
    } 
template<int CURRENT_LAYER_SIZE>   
    torch::Tensor Sync_data_new(torch::Tensor& data){
        std::cout<<"start"<<start_<<" "<<data.accessor<t_v,2>().size(1)<<std::endl;
        memset(data_coomm,0,sizeof(t_v)*CURRENT_LAYER_SIZE*graph_->vertices);
      graph_->process_edges<int, nodeVector<CURRENT_LAYER_SIZE>>(// For EACH Vertex Processing
        [&](VertexId src, VertexAdjList<Empty> outgoing_adj) {//pull
        // if(src>=2200){
        //     std::cout<<"yes "<<src<<" is called"<<std::endl;
        // }
            nodeVector<CURRENT_LAYER_SIZE> sum;
            memcpy(sum.data,data.accessor<t_v,2>().data()+(src)*CURRENT_LAYER_SIZE,sizeof(t_v)*CURRENT_LAYER_SIZE);
            graph_->emit(src, sum);
        },
        [&](VertexId src, nodeVector<CURRENT_LAYER_SIZE> msg) {
            //memset(data_coomm+(src)*CURRENT_LAYER_SIZE,0,sizeof(t_v)*CURRENT_LAYER_SIZE);
            for(int i=0;i<CURRENT_LAYER_SIZE;i++){
               data_coomm[(src)*CURRENT_LAYER_SIZE+i]+=msg.data[i]; 
            }
            return 0;
        }, active_ ); 
        return torch::from_blob(data_coomm+(start_)*CURRENT_LAYER_SIZE,{rownum_,CURRENT_LAYER_SIZE},torch::kFloat);
    } 


template<int CURRENT_LAYER_SIZE> 
    torch::Tensor Test_Propagate(int local_layer){
        zeroCurr();//local
        zeroNext();//local
        zeroValue(local_layer);
        nodeVector<CURRENT_LAYER_SIZE> test;
        for(int i=0;i<CURRENT_LAYER_SIZE;i++){
            test.data[i]=1.0;
        }
        int tagg=0;
     graph_->process_vertices<t_v>(//init  the vertex state.
                [&](VertexId vtx) {
                    cpToCurr<CURRENT_LAYER_SIZE>(vtx,test.data);//local
                    return (ValueType) 1;
                }, active_ );
    graph_->process_edges<int, nodeVector<CURRENT_LAYER_SIZE>>(// For EACH Vertex Processing
        [&](VertexId dst, VertexAdjList<Empty> incoming_adj) {//pull
            nodeVector<CURRENT_LAYER_SIZE> sum;
            memset(sum.data,0,sizeof(ValueType)*CURRENT_LAYER_SIZE);
            for (AdjUnit<Empty> * ptr = incoming_adj.begin; ptr != incoming_adj.end; ptr++) {//pull model
                VertexId src = ptr->neighbour;
                tagg++;
                for (int i = 0; i < CURRENT_LAYER_SIZE; i++) {
                    sum.data[i] += ((nodeVector<CURRENT_LAYER_SIZE>*)curr_local)[src-start_].data[i];//local
                }
            }
            graph_->emit(dst, sum);
        },
        [&](VertexId dst, nodeVector<CURRENT_LAYER_SIZE> msg) {
            addToNext<CURRENT_LAYER_SIZE>(dst,msg.data);//local
            return 0;
        }, active_);
        
        graph_->process_vertices<ValueType>(//init the vertex state.
                [&](VertexId vtx) {
                memcpy(this->value_buffer[local_layer]+CURRENT_LAYER_SIZE*(vtx-start_),
                        ((nodeVector<CURRENT_LAYER_SIZE>*)next_local)[vtx-start_].data,sizeof(t_v)*CURRENT_LAYER_SIZE);
                return 0;
                }, active_);
        
        value_local[local_layer]=torch::from_blob(value_buffer[local_layer],{rownum_,CURRENT_LAYER_SIZE},torch::kFloat);
        current_layer=local_layer;
        if(graph_->partition_id==0)
        {int tag=0;
            for(int i=0;i<graph_->vertices;i++){
                if(value_buffer[local_layer][i*CURRENT_LAYER_SIZE+3]-(ValueType)graph_->in_degree_for_backward[i]<=0.00001)
           // printf("%f\t%d\n",value_buffer[local_layer][i*CURRENT_LAYER_SIZE+3],graph_->in_degree_for_backward[i]);
                    tag++;
        }
        printf("finish TEST_propagate at layer [%d] validate %d vertices \n",local_layer,tag);
        }
        return value_local[local_layer];
            
    }

    void setValueFromTensor(torch::Tensor new_tensor){
        pre_value=new_tensor;   
        zeroCurr();//local 
        
    }
    void setValueFromNative(int layer, t_v* data,int offset){
        pre_value=torch::from_blob(data+offset,{rownum_,size_at_layer[layer]},torch::kFloat);
                zeroCurr();//local 
    }
   void setValueFromNative(t_v* data,int offset){
        pre_value=torch::from_blob(data+offset,{rownum_,VECTOR_LENGTH},torch::kFloat);
                zeroCurr();//local 
    }
    
    
    void setGradFromNative(t_v* data,int offset){
        pre_grad=torch::from_blob(data+offset,{rownum_,VECTOR_LENGTH},torch::kFloat);
    }
    void setGradFromTensor(torch::Tensor new_tensor){
        pre_grad=new_tensor;    
    }
    torch::Tensor v(int local_layer){
        return value_local[local_layer];
    }
    torch::Tensor require_grad(){
        return grad;
    }

};

void compute(Graph<Empty> * graph, int iterations);
void compute_GPU(Graph<Empty> * graph, int iterations);
void compute_single_GPU(Graph<Empty> * graph, int iterations);
void compute_dist_GPU(Graph<Empty> * graph, int iterations);

#endif
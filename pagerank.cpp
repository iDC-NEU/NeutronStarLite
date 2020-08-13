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
const double d = (double) 0.8;
#define VECTOR_LENGTH 1433
#define WEIGHT_ROW 1433
#define WEIGHT_COL 1433
#define EDGE_LENGTH  4
#define NODE_NUMBER 2708
#define LABEL_NUMBER 7

#define SIZE_LAYER_1    1433
#define SIZE_LAYER_2    128 
#define OUTPUT_LAYER_3  7

template<int size_>
struct nodeVector{
    float data[size_];
}__attribute__((packed));

//typedef struct factor2 {
//    float weight[WEIGHT_ROW][WEIGHT_COL];
//} weightVector;
template <typename t_v>
struct EdgeVector{
    t_v data[EDGE_LENGTH];
};

struct content
{
    int id;
    float att[VECTOR_LENGTH];
    long label;
}con[NODE_NUMBER];

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
    float *W_from;

    Intermediate(size_t w, size_t h) {
        //        at::TensorOptions *opt=new at::TensorOptions;
        //       opt->requires_grad(true);
        //  torch::randn
        //     A=torch::randn(torch::randn({w,h},opt));
        W = register_parameter("W", torch::randn({w, h}));
        W_from=new float[w*h];

    }
    void resetW(size_t w,size_t h,float* buffer){
        memcpy(W_from, buffer,sizeof(float)*w*h);
        torch::Tensor new_weight_tensor = torch::from_blob(W_from,{w, h});
        W.set_data(new_weight_tensor);
    }
    
    torch::Tensor forward(torch::Tensor x) {

       // auto tmp_acc_c = W.accessor<float, 2>();
        x = x.mm(W);
        //x = torch::tanh(x);
        //x=torch::relu(x);
        return x;
    }
};


struct GnnUnit : torch::nn::Module {
    torch::Tensor W;
    torch::Tensor X1;
    float *W_from;
    float *W_from_gpu;
    gpu_processor *gp;
    GnnUnit(size_t w, size_t h) {
        //        at::TensorOptions *opt=new at::TensorOptions;
        //       opt->requires_grad(true);
        //  torch::randn
        //     A=torch::randn(torch::randn({w,h},opt));
	gp=new gpu_processor();
        W = register_parameter("W", torch::randn({w, h}));
     //   X1= register_parameter("X1",torch::randn({2708,1433}));
        W_from=new float[w*h];
        W_from_gpu=gp->allocate_GPU(w*h*sizeof(float));
    }
    void resetW(size_t w,size_t h,float* buffer){
        memcpy(W_from, buffer,sizeof(float)*w*h);
        torch::Tensor new_weight_tensor = torch::from_blob(W_from,{w, h});
        W.set_data(new_weight_tensor);
    }
    
    void learn(torch::Tensor from, float learning_rate){
        torch::Tensor a=(W-(from*0.05));
        int r=0,c=0;
        r=a.size(0);
        c=a.size(1);
        memcpy(W_from, a.accessor<float,2>().data(),sizeof(float)*r*c);
        torch::Tensor new_weight_tensor = torch::from_blob(W_from,{r, c});
        W.set_data(new_weight_tensor);
    }

    void learn2(torch::Tensor from, float learning_rate){
        torch::Tensor a=(W-(from*0.05));
        int r=0,c=0;
        r=a.size(0);
        c=a.size(1);
	//std::cout<<a<<std::endl;
      //  memcpy(W_from_gpu, a.packed_accessor<float,2>().data(),sizeof(float)*r*c);
      //  torch::Tensor new_weight_tensor = torch::from_blob(W_from,{r, c});
        W.set_data(a);
    }
 
    torch::Tensor forward(torch::Tensor x) {

       // auto tmp_acc_c = W.accessor<float, 2>();
        x = x.mm(W);
        //x = torch::tanh(x);
        //x=torch::relu(x);
        return x;
    }

    torch::Tensor forward2(torch::Tensor x) {
    //    auto tmp_acc_c = W.accessor<float, 2>();
     //   x = x.mm(W);
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

    void init(Graph<Empty>* graph) {
        start_v = new float [graph->vertices*VECTOR_LENGTH];//graph->alloc_vertex_array<float>(VECTOR_LENGTH);
 //       next_v = graph->alloc_vertex_array<float>(VECTOR_LENGTH);
 //       partial_grad=new T_v*[2];
        //partial_grad[0]= graph->alloc_vertex_array<float>(VECTOR_LENGTH);
     //   Weight = new weightVector(); // reuse
        label = graph->alloc_vertex_array<T_l>();
        //curr = graph->alloc_vertex_array<nodeVector>();
        //next = graph->alloc_vertex_array<nodeVector>();
        Gnn_v1 = new GnnUnit(WEIGHT_ROW, WEIGHT_COL);
        Gnn_v2 =new GnnUnit(WEIGHT_ROW, LABEL_NUMBER);
       /* new Next_v2 for GNN*/
        local_grad=new float[WEIGHT_ROW*WEIGHT_COL];
        rownum = (graph->partition_offset[graph->partition_id + 1] - graph->partition_offset[graph->partition_id]);
        start = VECTOR_LENGTH * (graph->partition_offset[graph->partition_id]);
        aggre_grad=new float[VECTOR_LENGTH*VECTOR_LENGTH];
        
    }
    void initStartWith( int index, float with,int i) {
 
        *(start_v + index * VECTOR_LENGTH + i) = with;
            //   curr[index].data[i] = (float)with;
      
    }
    void initStartWith( int index, float with) {
        for (int i = 0; i < VECTOR_LENGTH; i++) {
            *(start_v + index * VECTOR_LENGTH + i) = with;
            //   curr[index].data[i] = (float)with;
        }
    }
    void readlabel(Graph<Empty>* graph) {
        graph->fill_vertex_array(label, (long) 1);
    }
    void readEmbedding(Graph<Empty>* graph){
        ;
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
        
        //graph->featureSize=VECTOR_LENGTH;
        
        int numOfData=0;
        std::string la;
        while(input>>con[numOfData].id)
        {
            for(int i=0;i<SIZE_LAYER_1;i++)
            {
                input>>con[numOfData].att[i];
            }
            input>>la;
            //cout<<"==============================="<<endl;
            con[numOfData].label=changelable(la);
            //cout<<"lable: "<<con[numOfData].label<<std::endl;
            numOfData++;
        }
      
        for(int i=0;i<NODE_NUMBER;i++){
            label[i]=con[i].label;
            //std::cout<<label[i]<<std::endl;
        }
        
        input.close();
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
        x.push_back(torch::tensor(0.0));
        y.push_back(torch::tensor(0.0));
        localGrad.push_back(torch::tensor(0.0));
        backwardGrad.push_back(torch::tensor(0.0));
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

void init_parameter(Network<float> * comm,Graph<Empty> * graph,GnnUnit* gnn,Embeddings<float,long>* embedding){
    
     int r,c;
     r=gnn->W.accessor<float,2>().size(0);
     c=gnn->W.accessor<float,2>().size(1);
    if(graph->partition_id==0){//first layer
     //   embedding->wrtPara2W(graph,gnn);//write para to temp Weight 
    //    comm->wrtWtoBuff(embedding->Weight); //write para to send buffer
    //    comm->wrtBuffertoBuff(gnn->)
     //  std::cout<<"init\n"<<gnn->W<<std::endl;
       comm->wrtBuffertoBuff(gnn->W.accessor<float,2>().data(),r, c);
      //  comm->wrtBuffertoBuff(gnn->W.accessor<float,2>().data());
    }
//     if(graph->partition_id==0){
//     printf("test 111%d %d\n",r,c);
//     }
     comm->broadcastW(gnn->W.accessor<float,2>().data(),r,c);// comm buffer
     
     gnn->resetW(r,c,comm->buffer);
}
template<typename t_v>
torch::Tensor unified_parameter(Network<float> *comm,torch::Tensor res){
    
    //torch::Tensor s;
    int r=res.accessor<float,2>().size(0);
    int c=res.accessor<float,2>().size(1);
        comm->wrtBuffertoBuff(res.accessor<float,2>().data(),r,c); //write para to send buffer 
        comm->gatherW(r,c);// gather from others to recvbuffer
        comm->computeW(r,c);// compute new para on recvbuffer
        comm->broadcastW(comm->recv_buffer,r,c);
        return torch::from_blob(comm->buffer,{r,c});
}

//template<typename t_v>
//void unified_parameter(Network<float> *comm, t_v* buffer, int r,int c){
//    
//    //torch::Tensor s;
//        comm->wrtBuffertoBuff(buffer,r,c); //write para to send buffer
//        comm->gatherW(r,c);// gather from others to recvbuffer
//        comm->computeW(r,c);// compute new para on recvbuffer
//        comm->broadcastW(comm->recv_buffer,r,c);
//}




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
//    inline void cpToCurr(int layer,int vtx,t_v* src){
//        memcpy(((nodeVector<10>**)curr)[layer][vtx-start_].data, src, sizeof(nodeVector<10>));
//    }
    inline void cpToNext(int vtx,t_v* src){
        memcpy(next_local[vtx-start_].data, src, sizeof(nodeVector<MAX_EMBEDDING_SIZE>));
    }
//    inline void cpToNext(int layer,int vtx,t_v* src){
//        memcpy(((nodeVector<size_at_layer[layer]>**)next)[layer][vtx-start_].data, src, sizeof(nodeVector<size_at_layer[layer]>));
//    }
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
            printf("size_at_alyer %d  %d\n",i,size_at_layer[i]);
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
                    return (float) 1;
                }, active_ );   
        
    }
    torch::Tensor applyVertex(int local_layer, torch::Tensor vertex_data){
        torch::Tensor destination;
        bool s=value_local[local_layer].accessor<float,2>().size(0)==vertex_data.accessor<float,2>().size(0);
        bool t=value_local[local_layer].accessor<float,2>().size(1)==vertex_data.accessor<float,2>().size(1);
        //assert(source.accessor<float,2>().size(0)==vertex_data.accessor<float,2>().size(0));
        //assert(source.accessor<float,2>().size(1)==vertex_data.accessor<float,2>().size(1));
        assert((s&&t)==1);
        destination=value_local[local_layer]*vertex_data;
    }
    torch::Tensor applyVertexPre(torch::Tensor source,torch::Tensor vertex_data){
        torch::Tensor destination;
        bool s=source.accessor<float,2>().size(0)==vertex_data.accessor<float,2>().size(0);
        bool t=source.accessor<float,2>().size(1)==vertex_data.accessor<float,2>().size(1);
        //assert(source.accessor<float,2>().size(0)==vertex_data.accessor<float,2>().size(0));
        //assert(source.accessor<float,2>().size(1)==vertex_data.accessor<float,2>().size(1));
        assert((s&&t)==1);
        return source*vertex_data;
    }
template<int CURRENT_LAYER_SIZE> 
    torch::Tensor Propagate(int local_layer){
        zeroCurr();//local
        zeroNext();//local
        zeroValue(local_layer);
    graph_->process_vertices<t_v>(//init  the vertex state.
                [&](VertexId vtx) {
                    cpToCurr<CURRENT_LAYER_SIZE>(vtx,pre_value.accessor<t_v,2>().data()+(vtx-start_)*CURRENT_LAYER_SIZE);//local
                    return (float) 1;
                }, active_ );   
     
    graph_->process_edges<int, nodeVector<CURRENT_LAYER_SIZE>>(// For EACH Vertex Processing
        [&](VertexId dst, VertexAdjList<Empty> incoming_adj) {//pull
            nodeVector<CURRENT_LAYER_SIZE> sum;
            memset(sum.data,0,sizeof(float)*CURRENT_LAYER_SIZE);
            for (AdjUnit<Empty> * ptr = incoming_adj.begin; ptr != incoming_adj.end; ptr++) {//pull model
                VertexId src = ptr->neighbour;
                for (int i = 0; i < CURRENT_LAYER_SIZE; i++) {
                    sum.data[i] += ((nodeVector<CURRENT_LAYER_SIZE>*)curr_local)[src-start_].data[i]/(
                        (float)std::sqrt(graph_->out_degree_for_backward[src])*(float)std::sqrt(graph_->in_degree_for_backward[dst])
                     );//local
                }
            }
            if(dst<graph_->partition_offset[graph_->partition_id+1]&&dst>graph_->partition_offset[graph_->partition_id])
            for (int i = 0; i < CURRENT_LAYER_SIZE; i++) {
                    sum.data[i] += ((nodeVector<CURRENT_LAYER_SIZE>*)curr_local)[dst-start_].data[i]/(
                        (float)std::sqrt(graph_->out_degree_for_backward[dst])*(float)std::sqrt(graph_->in_degree_for_backward[dst])
                     );//local
            }
            graph_->emit(dst, sum);
        },
        [&](VertexId dst, nodeVector<CURRENT_LAYER_SIZE> msg) {
            addToNext<CURRENT_LAYER_SIZE>(dst,msg.data);//local
            return 0;
        }, active_);
        
        graph_->process_vertices<float>(//init the vertex state.
                [&](VertexId vtx) {
                memcpy(this->value_buffer[local_layer]+CURRENT_LAYER_SIZE*(vtx-start_),
                        ((nodeVector<CURRENT_LAYER_SIZE>*)next_local)[vtx-start_].data,sizeof(t_v)*CURRENT_LAYER_SIZE);
                return 0;
                }, active_);
        
        value_local[local_layer]=torch::from_blob(value_buffer[local_layer],{rownum_,CURRENT_LAYER_SIZE},torch::kFloat);
        current_layer=local_layer;
        if(graph_->partition_id==0)
        printf("finish propagate at layer [%d]\n",local_layer);
        return value_local[local_layer];
        
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
                    return (float) 1;
                }, active_ );
    graph_->process_edges<int, nodeVector<CURRENT_LAYER_SIZE>>(// For EACH Vertex Processing
        [&](VertexId dst, VertexAdjList<Empty> incoming_adj) {//pull
            nodeVector<CURRENT_LAYER_SIZE> sum;
            memset(sum.data,0,sizeof(float)*CURRENT_LAYER_SIZE);
            for (AdjUnit<Empty> * ptr = incoming_adj.begin; ptr != incoming_adj.end; ptr++) {//pull model
                VertexId src = ptr->neighbour;
                tagg++;
                for (int i = 0; i < CURRENT_LAYER_SIZE; i++) {
                    sum.data[i] += ((nodeVector<CURRENT_LAYER_SIZE>*)curr_local)[src-start_].data[i];//local
                }
            }
            if(dst<graph_->partition_offset[graph_->partition_id+1]&&dst>=graph_->partition_offset[graph_->partition_id])
            for (int i = 0; i < CURRENT_LAYER_SIZE; i++) {
                    sum.data[i] += ((nodeVector<CURRENT_LAYER_SIZE>*)curr_local)[dst-start_].data[i];//local
            }
            if(dst==2){
                printf("first%f\n",sum.data[0]);
            }
            graph_->emit(dst, sum);
        },
        [&](VertexId dst, nodeVector<CURRENT_LAYER_SIZE> msg) {
            addToNext<CURRENT_LAYER_SIZE>(dst,msg.data);//local
            return 0;
        }, active_);
        
        graph_->process_vertices<float>(//init the vertex state.
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
                if(value_buffer[local_layer][i*CURRENT_LAYER_SIZE+3]-(float)graph_->in_degree_for_backward[i]<=1.00001)
           // printf("%f\t%d\n",value_buffer[local_layer][i*CURRENT_LAYER_SIZE+3],graph_->in_degree_for_backward[i]);
                    tag++;
        }
        printf("finish TEST_propagate at layer [%d] validate %d vertices \n",local_layer,tag);
        }
        return value_local[local_layer];
        
        
    }


    
template<int CURRENT_LAYER_SIZE>   
    void Propagate_backward(int local_layer){
        zeroCurr();//local
        zeroNext();//local    
        zeroGrad(local_layer);//local
      graph_->process_vertices<float>(//init  the vertex state.
                [&](VertexId vtx) {
                    cpToCurr<CURRENT_LAYER_SIZE>(vtx,pre_grad.accessor<t_v,2>().data()+(vtx-start_)*CURRENT_LAYER_SIZE );//local
                     return 1;
                }, active_ );  
        //start graph engine.
      graph_->process_edges_backward<int, nodeVector<CURRENT_LAYER_SIZE>>(// For EACH Vertex Processing
        [&](VertexId src, VertexAdjList<Empty> outgoing_adj) {//pull
            nodeVector<CURRENT_LAYER_SIZE> sum;
            memset(sum.data,0,sizeof(float)*CURRENT_LAYER_SIZE);
            for (AdjUnit<Empty> * ptr = outgoing_adj.begin; ptr != outgoing_adj.end; ptr++) {//pull model
                VertexId dst = ptr->neighbour;
                for (int i = 0; i < CURRENT_LAYER_SIZE; i++) {
                     sum.data[i] += ((nodeVector<CURRENT_LAYER_SIZE>*)curr_local)[dst-start_].data[i]/(
                        (float)std::sqrt(graph_->out_degree_for_backward[src])*(float)std::sqrt(graph_->in_degree_for_backward[dst])
                     );//local
                   // sum.data[i] += embedding_->curr[dst].data[i];//
                }
            if(src<graph_->partition_offset[graph_->partition_id+1]&&src>graph_->partition_offset[graph_->partition_id])
            for (int i = 0; i < CURRENT_LAYER_SIZE; i++) {
                    sum.data[i] += ((nodeVector<CURRENT_LAYER_SIZE>*)curr_local)[src-start_].data[i]/(
                        (float)std::sqrt(graph_->out_degree_for_backward[src])*(float)std::sqrt(graph_->in_degree_for_backward[src])
                     );//local
            }

            }
            graph_->emit(src, sum);
        },
        [&](VertexId src, nodeVector<CURRENT_LAYER_SIZE> msg) {
            addToNext<CURRENT_LAYER_SIZE>(src,msg.data);
            return 0;
        }, active_ );    
//3.3.4write partial gradient to local and update the gradient of W1
    
      graph_->process_vertices<float>(//init  the vertex state.
                [&](VertexId vtx) {
                memcpy(this->grad_buffer[local_layer]+CURRENT_LAYER_SIZE*(vtx-start_),
                        ((nodeVector<CURRENT_LAYER_SIZE>*)next_local)[vtx-start_].data,sizeof(t_v)*CURRENT_LAYER_SIZE);//local
                
                return 0;
                }, active_);
      grad_local[local_layer]=torch::from_blob(grad_buffer[local_layer],{rownum_,CURRENT_LAYER_SIZE});//local
      grad=grad_local[local_layer];
      if(graph_->partition_id==0)
        printf("finish backward propagate at layer [%d]\n",local_layer);
      
    }
    
    
    
    void setValueFromTensor(torch::Tensor new_tensor){
        pre_value=new_tensor;   
        zeroCurr();//local 
        
    }
    void setValueFromNative(int layer, t_v* data,int offset){
        pre_value=torch::from_blob(data+offset,{rownum_,size_at_layer[layer]});
                zeroCurr();//local 
    }
   void setValueFromNative(t_v* data,int offset){
        pre_value=torch::from_blob(data+offset,{rownum_,VECTOR_LENGTH});
                zeroCurr();//local 
    }
    
    
    void setGradFromNative(t_v* data,int offset){
        pre_grad=torch::from_blob(data+offset,{rownum_,VECTOR_LENGTH});
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



void compute(Graph<Empty> * graph, int iterations) {
    
   
	gpu_processor *gp=new gpu_processor(); 
    Embeddings<float, long> *embedding = new Embeddings<float, long>();
    embedding->init(graph);
    embedding->readlabel1(graph);
    
    Network<float> *comm=new Network<float>(graph,WEIGHT_ROW,WEIGHT_COL);
    Network<float>* comm1=new Network<float>(graph,WEIGHT_ROW,WEIGHT_COL);
    comm->setWsize(WEIGHT_ROW,WEIGHT_COL);
    comm1->setWsize(WEIGHT_ROW,WEIGHT_COL);
    tensorSet *pytool=new tensorSet(2);
    pytool->in_degree=torch::from_blob(graph->in_degree+graph->partition_offset[graph->partition_id],{embedding->rownum,1});
    pytool->out_degree=torch::from_blob(graph->in_degree+graph->partition_offset[graph->partition_id],{embedding->rownum,1});
/*1 INIT STAGE*/    
   // GTensor<float,Empty> gt=new  GTensor(comm, graph);
    
    GnnUnit* Gnn_v1 = new GnnUnit(SIZE_LAYER_1, SIZE_LAYER_2);
    GnnUnit* Gnn_v2 =new GnnUnit(SIZE_LAYER_2, OUTPUT_LAYER_3);
    pytool->registOptimizer(torch::optim::SGD(Gnn_v1->parameters(), 0.05));//new
    pytool->registOptimizer(torch::optim::SGD(Gnn_v2->parameters(), 0.05));//new
    pytool->registLabel<long>(embedding->label,graph->partition_offset[graph->partition_id],graph->partition_offset[graph->partition_id+1]-graph->partition_offset[graph->partition_id]);//new

/*init W with new */
  
    init_parameter(comm, graph,Gnn_v1, embedding);
    init_parameter(comm1, graph,Gnn_v2, embedding);
    VertexSubset * active = graph->alloc_vertex_subset();
    active->fill();
    std::vector<int> layer_size(0);
    layer_size.push_back(SIZE_LAYER_1);
    layer_size.push_back(SIZE_LAYER_2);
    layer_size.push_back(OUTPUT_LAYER_3);
    GTensor<float,long,VECTOR_LENGTH> *gt=new GTensor<float, long,VECTOR_LENGTH>(graph,embedding,active,2,layer_size); 
    
    
    Intermediate *inter=new Intermediate(embedding->rownum,SIZE_LAYER_2);
    Intermediate *X1=new Intermediate(embedding->rownum,SIZE_LAYER_2);
    
    graph->process_vertices<float>(//init  the vertex state.
        [&](VertexId vtx){
            int start=(graph->partition_offset[graph->partition_id]);
                    for(int i=0;i<SIZE_LAYER_1;i++){
                        //*(start_v + vtx * VECTOR_LENGTH + i) = con[j].att[i];
                        embedding->initStartWith(vtx,con[vtx].att[i],i);
                    }     
            return (float)1;
        },
    active
    );


       gt->Test_Propagate<SIZE_LAYER_1>(0);
        
    for (int i_i = 0; i_i < iterations; i_i++) {
            gt->setValueFromNative(embedding->start_v,embedding->start);  
            
         if(graph->partition_id==0)
        std::cout<<"start  ["<<i_i<<"]  epoch+++++++++++++++++++++++++++++++++++++++++++++"<<std::endl;


        /*2. FORWARD STAGE*/      
//2.1.1 start the forward of the first layer
            
        gt->Propagate<SIZE_LAYER_1>(0);
        pytool->updateX(0,gt->value_local[0]); 
        
        
        inter->W.set_data(Gnn_v1->forward(pytool->x[0]));//torch::from_blob(embedding->Gnn_v1->forward(pytool->x[0]).accessor<float,2>().data(),{embedding->rownum,VECTOR_LENGTH});
        pytool->y[0]=torch::relu(inter->W);//new
        
//        std::cout<<pytool->y[0].size(0)<<"\t"<<pytool->y[0].size(1)<<std::endl;
//2.2.1 init the second layer               
        gt->setValueFromTensor(pytool->y[0]); 
//2.2.2 forward the second layer                     
        gt->Propagate<SIZE_LAYER_2>(1);             
/*3 BACKWARD STAGE*/
//        printf("%d\t%d\n",gt->value_local[1].accessor<float,2>().size(0),gt->value_local[1].accessor<float,2>().size(1));
//3.1 compute the output of the second layer.
        pytool->updateX(1,gt->value_local[1]);//new
        pytool->x[1].set_requires_grad(true);
        
//        printf("%d\t%d\n",Gnn_v2->W.accessor<float,2>().size(0),gt->value_local[1].accessor<float,2>().size(1));

        torch::DeviceType device_type;
        device_type = torch::kCUDA;
        torch::Device device(device_type,0);
        Gnn_v2->to(device);
               
        torch::Tensor x1_data = pytool->x[1].to(device);
        x1_data.set_requires_grad(true);

        pytool->y[1] = Gnn_v2->forward3(x1_data);
        pytool->y[1].set_requires_grad(true);
        torch::Tensor tt=pytool->y[1].log_softmax(1);//CUDA

        torch::Tensor target_data  = pytool->target.to(device);
        pytool->loss =torch::nll_loss(pytool->y[1],target_data);//new
        pytool->loss.backward();
        std::cout<< x1_data.grad;
     
       // torch::Tensor res = Gnn_v2->W.grad();
       // float* datafromgpu  = res.packed_accessor<float,2>().data();
       // int r = res.packed_accessor<float,2>().size(0);
       // int c = res.packed_accessor<float,2>().size(1);
       // float* buffertocpu=new float[r*c];
       // gp->cp_data_from_gpu2cpu(buffertocpu,datafromgpu,r*c);
       //	torch::Tensor buffertensor=torch::from_blob(buffertocpu,{r,c});
//3.2 compute the gradient of the second layer.     
       // torch::Tensor aggregate_grad2 = unified_parameter<float>(comm1,buffertensor);//Gnn_v2->W.grad()
       // torch::Tensor aggregate_grad2_data = aggregate_grad2.to(device);
       // Gnn_v2->learn2(aggregate_grad2_data,0.05);//reset from new          
//3.3.3 backward the partial gradient from 2-layer to 1-layer    torch::Tensor partial_grad_layer2=pytool->x[1].grad();     
      
    
     //   std::cout<< x1_data.grad().size(0)<<"    "<< x1_data.grad().size(1)<<std::endl;
     //   float* x1datafromgpu  = x1data.packed_accessor<float,2>().data();
     //   int x1r = x1data.packed_accessor<float,2>().size(0);
     //   int x1c = x1data.packed_accessor<float,2>().size(1);
     //   float* x1buffertocpu=new float[x1r*x1c];
     //   gp->cp_data_from_gpu2cpu(x1buffertocpu,x1datafromgpu,x1r*x1c);
     //   torch::Tensor x1buffertensor=torch::from_blob(x1buffertocpu,{x1r,x1c});

        
 /*   gt->setGradFromTensor(pytool->x[1].grad());
        gt->Propagate_backward<SIZE_LAYER_2>(0);
//*3.3.1  compute  W1's partial gradient in first layer   
        pytool->y[0].backward();//new
        pytool->localGrad[0]=inter->W.grad();//new          
        
torch::Tensor new_combine_grad=torch::zeros({SIZE_LAYER_1,SIZE_LAYER_2});

for(int i=0;i<graph->owned_vertices;i++){
    new_combine_grad=new_combine_grad+
            pytool->x[0].index_select(0,torch::tensor(i,torch::kLong)).t().mm(
            pytool->localGrad[0].index_select(0,torch::tensor(i,torch::kLong))
            )*gt->grad_local[0].index_select(0,torch::tensor(i,torch::kLong));
}
        
//3.3.4write partial gradient to local and update the gradient of W1
      //  std::cout<<graph->partition_id<<"before\n"<<embedding->Gnn_v1->W<<std::endl;  
     torch::Tensor aggregate_grad = unified_parameter<float>(comm,(new_combine_grad));
   //  std::cout<<"grad of gnn_v1\n"<<aggregate_grad<<std::endl; 
     Gnn_v1->learn(aggregate_grad,0.05); 

     if(graph->partition_id==0) 
         std::cout<<"LOSS:\t"<<pytool->loss<<std::endl;    
     
     if(i_i==(iterations-1)&&graph->partition_id==0){
         std::cout<<"+++++++++++++++++++++++ finish ++++++++++++++++++++++++"<<std::endl;
             if(i_i==iterations-1){
     //      std::cout<<pytool->y[1].softmax(1).log().accessor<float,2>().size(0)<<" "<<pytool->y[1].softmax(1).log().accessor<float,2>().size(1)<<std::endl;
            int correct=0;
            for(int k=0;k<embedding->rownum;k++){
                float max= -100000.0;
                long id=-1;
                for(int i=0;i<LABEL_NUMBER;i++){
                    if (max<tt.accessor<float,2>().data()[k*LABEL_NUMBER+i]){
                        max=tt.accessor<float,2>().data()[k*LABEL_NUMBER+i];
                        id=i;
                    }
                }
                   if(id==pytool->target.accessor<long,1>().data()[k])
                   correct+=1;
//               }
            }
            std::cout<<"\ncorrect number:"<<correct<<std::endl;
        }
        std::cout<<"loss at"<<graph->partition_id<<"is :"<<pytool->loss<<std::endl;
         
     }*/
    }
     //cudaDeviceReset();
     delete active;     
}

typedef struct ED{
    char a[40];
}EdgeDatas;
int main(int argc, char ** argv) {
    MPI_Instance mpi(&argc, &argv);
 printf("hello\n");
    if (argc < 4) {
        printf("pagerank [file] [vertices] [iterations]\n");
        exit(-1);
    }
        
        
    Graph<Empty> * graph;
    graph = new Graph<Empty>();
    graph->load_directed(argv[1], std::atoi(argv[2]));
    graph->generate_backward_structure();  
    int iterations = std::atoi(argv[3]);
    printf("hello world\n");
    double exec_time = 0;
    exec_time -= get_time();
    compute(graph, iterations);
      exec_time += get_time();
  if (graph->partition_id==0) {
    printf("exec_time=%lf(s)\n", exec_time);
  }

//      char a[40];
//      printf("size: %d",sizeof(nodeVector));
     
//      EdgeUnit<EdgeVector<float>> *ss=new EdgeUnit<EdgeVector<float>>;
      float a[6]={1,2,3,4,5,6};
      float a1[4]={-3,4,-5,6};
//      memcpy(ss,a,sizeof(float)*6);
//        if (graph->partition_id==0)
//           printf("size: %d %f %f %f %f",sizeof(EdgeUnit<EdgeVector<float>>),ss->edge_data.data[0],ss->edge_data.data[1],ss->edge_data.data[2],ss->edge_data.data[3]);
    delete graph;
    
    
    std::vector<int> s;
    s.push_back(1);
    s.push_back(2);
    s.push_back(3);
    std::cout<<sizeof(s)<<std::endl;
    
//   if(graph->partition_id==0){
//    float b1[2]={1,2};
//    float b2[2]={-13,16};
//    float c[4]={0};
//    torch::Tensor w=torch::from_blob(a1,{2,2});
//    torch::Tensor w0=torch::from_blob(a1,{2,2});
//    torch::Tensor w1=torch::from_blob(a,{2,2});
//    torch::Tensor ans=torch::from_blob(c,{2,2});
//    torch::Tensor x1=torch::from_blob(b1,{1,2});
//    torch::Tensor y1=torch::from_blob(b2,{1,2});
////    y1.set_requires_grad(true);
//    w.set_requires_grad(true);
//    w0.set_requires_grad(true);
////    torch::Tensor y=x1.mm(w);
////
////    torch::Tensor y2=y.mm(w1);
////            y2.backward();
////    torch::Tensor x2=x1.mm(w);
////    torch::Tensor y2=y1.mm(w1);
////    x2.backward();
////    y2.backward();
//    //ans=w.index_select(0,torch::tensor(0,torch::kLong))*w.t();
//    torch::Tensor tmp=torch::ones({1,2});
//    tmp=torch::from_blob(b2,{1,2});tmp.set_requires_grad(true);
//    //tmp=x1.mm(w);
//    //ans=torch::sigmoid(x1.mm(w));
//    //ans.backward();
//    w*x1.t();
//    std::cout<<"tmp\n"<<tmp<<"\nans\n"<<ans<<"\n"<<w*x1.t()<<"\n"<<w*x1<<std::endl;
//    //ans.index_copy(0,torch::tensor(1,torch::kLong),w.index_select(0,torch::tensor(1,torch::kLong)));
//    //std::cout<<w<<"\n"<<w.grad()<<std::endl;
//
//    }
    
    return 0;
}



//        /*2. FORWARD STAGE*/      
////2.1.1 start the forward of the first layer
//        gt->Propagate(0);
//        pytool->updateX(0,gt->value_local[0]); 
//        
//        pytool->y[0]=torch::relu(embedding->Gnn_v1->forward(pytool->x[0]));//new
//        
//        
////2.2.1 init the second layer               
//        gt->setValueFromTensor(pytool->y[0]); 
//////2.2.2 forward the second layer                     
//        gt->Propagate(1);             
///*3 BACKWARD STAGE*/
////3.1 compute the output of the second layer.
//        pytool->updateX(1,gt->value_local[1]);//new
//        pytool->x[1].set_requires_grad(true);
//        pytool->y[1]=embedding->Gnn_v2->forward(pytool->x[1]);//new
////        pytool->y[1].backward();
//        torch::Tensor tt=pytool->y[1].log_softmax(1);
//        pytool->loss =torch::nll_loss(tt,pytool->target);//new
//        pytool->loss.backward();
////3.2 compute the gradient of the second layer.  
//      //  pytool->loss.backward();//new
//        // pytool->optimizers[1].step();//new        
//        torch::Tensor aggregate_grad2= unified_parameter<float>(comm1,embedding->Gnn_v2->W.grad()); 
//         embedding->Gnn_v2->learn(aggregate_grad2,0.05);//reset from new 
////*3.3.1  compute  W1's partial gradient in first layer   
//        pytool->y[0].backward();//new
//        pytool->localGrad[0]=embedding->Gnn_v1->W.grad();//new                
////3.3.3 backward the partial gradient from 2-layer to 1-layer    torch::Tensor partial_grad_layer2=pytool->x[1].grad(); 
//
//        
//        gt->setGradFromTensor(pytool->x[1].grad());
//        gt->Propagate_backward(0);
//        gt->aggregateGrad(0);
//        
//        
//        
////3.3.4write partial gradient to local and update the gradient of W1
//      //  std::cout<<graph->partition_id<<"before\n"<<embedding->Gnn_v1->W<<std::endl;  
//     pytool->backwardGrad[0]=torch::from_blob(gt->partial_grad_buffer[0],{VECTOR_LENGTH, VECTOR_LENGTH});//new
//     torch::Tensor aggregate_grad = unified_parameter<float>(comm,(pytool->localGrad[0]*pytool->backwardGrad[0]));
//   //  std::cout<<"grad of gnn_v1\n"<<aggregate_grad<<std::endl; 
//     embedding->Gnn_v1->learn(aggregate_grad,0.05); 
         







//        gt->Propagate(0);
//         
//        pytool->updateX(0,gt->value_local[0]);             
//        pytool->y[0]=embedding->Gnn_v2->forward(pytool->x[0]);//new
//       // printf("size:\t%d  %d\n",pytool->y[0].size(0),pytool->y[0].size(1));
//        torch::Tensor tt=pytool->y[0].log_softmax(1);
//        pytool->loss =torch::nll_loss(tt,pytool->target);//new
//        pytool->loss.backward();
////3.2 compute the gradient of the second layer.  
//      //  pytool->loss.backward();//new
//        // pytool->optimizers[1].step();//new        
//        torch::Tensor aggregate_grad2= unified_parameter<float>(comm1,embedding->Gnn_v2->W.grad()); 
// //       std::cout<<"grad of gnn_v2\n"<<aggregate_grad2<<std::endl;
//       
//         //pytool->optimizers[1].step();
//         embedding->Gnn_v2->learn(aggregate_grad2,0.05);//reset from new 


//    torch::Tensor Propagate_Tensor(int local_layer){
//        zeroCurr();//local
//        zeroNext();//local   
//        zeroValue(local_layer);
//        graph_->message_unit_size=size_at_layer[local_layer]*sizeof(t_v);
//        
//        printf("%d\n",graph_->message_unit_size);
//    graph_->process_vertices<t_v>(//init  the vertex state.
//                [&](VertexId vtx) {
//                    cpToCurr(vtx,pre_value.accessor<t_v,2>().data()+(vtx-start_)*size_at_layer[local_layer]);//local
//                    return (float) 1;
//                }, active_ );   
//     
//    graph_->process_edges_buffer<int, float>(// For EACH Vertex Processing
//        [&](VertexId dst, VertexAdjList<Empty> incoming_adj) {//pull
//            nodeVector<VECTOR_LENGTH> sum;
//            //t_v* sum= new t_v*[size_at_layer[local_layer]];
//            memset(sum.data,0,sizeof(float)*size_at_layer[local_layer]);
//            for (AdjUnit<Empty> * ptr = incoming_adj.begin; ptr != incoming_adj.end; ptr++) {//pull model
//                VertexId src = ptr->neighbour;
//                for (int i = 0; i < size_at_layer[local_layer]; i++) {
//                    sum.data[i] += curr_local[src-start_].data[i]/(
//                        (float)std::sqrt(graph_->out_degree_for_backward[src])*(float)std::sqrt(graph_->in_degree_for_backward[dst])
//                     );//local
//                }
//            }
//            if(dst<graph_->partition_offset[graph_->partition_id+1]&&dst>graph_->partition_offset[graph_->partition_id])
//            for (int i = 0; i < size_at_layer[local_layer]; i++) {
//                    sum.data[i] += curr_local[dst-start_].data[i]/(
//                        (float)std::sqrt(graph_->out_degree_for_backward[dst])*(float)std::sqrt(graph_->in_degree_for_backward[dst])
//                     );//local
//            }
//            if(dst==0)
//            graph_->emit_buffer(dst, sum.data);
//        },
//        [&](VertexId dst, t_v *msg) {
//            addToNext(dst,msg,size_at_layer[local_layer]);//local
//            return 0;
//        }, active_);
        
//        graph_->process_vertices<float>(//init the vertex state.
//                [&](VertexId vtx) {
//                memcpy(this->value_buffer[local_layer]+size_at_layer[local_layer]*(vtx-start_),
//                        next_local[vtx-start_].data,sizeof(t_v)*size_at_layer[local_layer]);
//                return 0;
//                }, active_);
        
//                value_local[local_layer]=torch::from_blob(value_buffer[local_layer],{rownum_,size_at_layer[local_layer]});
//        current_layer=local_layer;
//        return value_local[local_layer];
//    }   
//    void Propagate_backward_tensor(int local_layer){
//        zeroCurr();//local
//        zeroNext();//local    
//        zeroGrad(local_layer);//local
//        int local_layer_size=local_layer+1;
//        graph_->message_unit_size=size_at_layer[local_layer]*sizeof(t_v);
//      graph_->process_vertices<float>(//init  the vertex state.
//                [&](VertexId vtx) {
//                    cpToCurr(vtx,pre_grad.accessor<t_v,2>().data()+(vtx-start_)*size_at_layer[local_layer_size]);//local
//                     return 1;
//                }, active_ );  
//        //start graph engine.
//      graph_->process_edges_backward_buffer<int,t_v>(// For EACH Vertex Processing
//        [&](VertexId src, VertexAdjList<Empty> outgoing_adj) {//pull
//            nodeVector<VECTOR_LENGTH> sum;
//            memset(sum.data,0,sizeof(float)*VECTOR_LENGTH);
//            for (AdjUnit<Empty> * ptr = outgoing_adj.begin; ptr != outgoing_adj.end; ptr++) {//pull model
//                VertexId dst = ptr->neighbour;
//                for (int i = 0; i < size_at_layer[local_layer_size]; i++) {
//                     sum.data[i] += curr_local[dst-start_].data[i]/(
//                        (float)std::sqrt(graph_->out_degree_for_backward[src])*(float)std::sqrt(graph_->in_degree_for_backward[dst])
//                     );//local
//                   // sum.data[i] += embedding_->curr[dst].data[i];//
//                }
//            if(src<graph_->partition_offset[graph_->partition_id+1]&&src>graph_->partition_offset[graph_->partition_id])
//            for (int i = 0; i < size_at_layer[local_layer_size]; i++) {
//                    sum.data[i] += curr_local[src-start_].data[i]/(
//                        (float)std::sqrt(graph_->out_degree_for_backward[src])*(float)std::sqrt(graph_->in_degree_for_backward[src])
//                     );//local
//            }
//
//            }
//            graph_->emit_buffer(src, sum.data);
//        },
//        [&](VertexId src, t_v* msg) {
//            addToNext(src,msg,size_at_layer[local_layer_size]);
//            return 0;
//        }, active_ );    
////3.3.4write partial gradient to local and update the gradient of W1
//    
//      graph_->process_vertices<float>(//init  the vertex state.
//                [&](VertexId vtx) {
//                memcpy(this->grad_buffer[local_layer]+size_at_layer[local_layer_size]*(vtx-start_),
//                        next_local[vtx-start_].data,sizeof(t_v)*size_at_layer[local_layer_size]);//local
//                
//                return 0;
//                }, active_);
//      grad_local[local_layer]=torch::from_blob(grad_buffer[local_layer],{rownum_,size_at_layer[local_layer_size]});//local
//      grad=grad_local[local_layer];
//      
//    }

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
    gpu_processor *gp;
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
    content *con=NULL;

    void init(Graph<Empty>* graph) {
        start_v = new ValueType [graph->vertices*SIZE_LAYER_1];//graph->alloc_vertex_array<float>(VECTOR_LENGTH);
 //       next_v = graph->alloc_vertex_array<float>(VECTOR_LENGTH);
 //       partial_grad=new T_v*[2];
        //partial_grad[0]= graph->alloc_vertex_array<float>(VECTOR_LENGTH);
     //   Weight = new weightVector(); // reuse
        label = graph->alloc_vertex_array<T_l>();
        //curr = graph->alloc_vertex_array<nodeVector>();
        //next = graph->alloc_vertex_array<nodeVector>();
//        Gnn_v1 = new GnnUnit(WEIGHT_ROW, WEIGHT_COL);
//        Gnn_v2 =new GnnUnit(WEIGHT_ROW, LABEL_NUMBER);
       /* new Next_v2 for GNN*/
//        local_grad=new float[WEIGHT_ROW*WEIGHT_COL];
        rownum = (graph->partition_offset[graph->partition_id + 1] - graph->partition_offset[graph->partition_id]);
        start = SIZE_LAYER_1 * (graph->partition_offset[graph->partition_id]);
        con=new content[NODE_NUMBER];
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
    printf("hello-1\n");
     int r,c;
     r=gnn->W.accessor<ValueType,2>().size(0);
     c=gnn->W.accessor<ValueType,2>().size(1);
     printf("hello0\n");
    if(graph->partition_id==0){//first layer
     //   embedding->wrtPara2W(graph,gnn);//write para to temp Weight 
    //    comm->wrtWtoBuff(embedding->Weight); //write para to send buffer
    //    comm->wrtBuffertoBuff(gnn->)
     //  std::cout<<"init\n"<<gnn->W<<std::endl;
       comm->wrtBuffertoBuff(gnn->W.accessor<ValueType,2>().data(),r, c);
      //  comm->wrtBuffertoBuff(gnn->W.accessor<float,2>().data());
    }
//     if(graph->partition_id==0){
//     printf("test 111%d %d\n",r,c);
//     }
     comm->broadcastW(gnn->W.accessor<ValueType,2>().data(),r,c);// comm buffer
     
     gnn->resetW(r,c,comm->buffer);
}
template<typename t_v>
torch::Tensor unified_parameter(Network<ValueType> *comm,torch::Tensor res){
    
    return res;
    //torch::Tensor s;
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
                    sum.data[i] += ((nodeVector<CURRENT_LAYER_SIZE>*)curr_local)[src-start_].data[i]/(
                        (ValueType)std::sqrt(graph_->out_degree_for_backward[src])*(ValueType)std::sqrt(graph_->in_degree_for_backward[dst]));
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

void compute(Graph<Empty> * graph, int iterations) {
    
   
	//gpu_processor *gp=new gpu_processor(); 
    Embeddings<ValueType, long> *embedding = new Embeddings<ValueType, long>();
    embedding->init(graph);
    embedding->readlabel1(graph);
    
    Network<ValueType> *comm=new Network<ValueType>(graph,WEIGHT_ROW,WEIGHT_COL);
    Network<ValueType>* comm1=new Network<ValueType>(graph,WEIGHT_ROW,WEIGHT_COL);
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
    GTensor<ValueType,long,VECTOR_LENGTH> *gt=new GTensor<ValueType, long,VECTOR_LENGTH>(graph,embedding,active,2,layer_size); 
    
    
    Intermediate *inter=new Intermediate(embedding->rownum,SIZE_LAYER_2);
    
    graph->process_vertices<ValueType>(//init  the vertex state.
        [&](VertexId vtx){
            int start=(graph->partition_offset[graph->partition_id]);
                    for(int i=0;i<SIZE_LAYER_1;i++){
                        //*(start_v + vtx * VECTOR_LENGTH + i) = con[j].att[i];
                        embedding->initStartWith(vtx,embedding->con[vtx].att[i],i);
                    }     
            return (ValueType)1;
        },
    active
    );


       gt->Test_Propagate<SIZE_LAYER_1>(0);
        
    for (int i_i = 0; i_i < iterations; i_i++) {
            gt->setValueFromNative(embedding->start_v,embedding->start);  
            if(i_i>0){
                //inter->zero_grad();
                //pytool->x[1].grad().zero_();
                Gnn_v1->zero_grad();
                Gnn_v2->zero_grad();
            }
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


        pytool->y[1] = Gnn_v2->forward(pytool->x[1]);
        torch::Tensor tt=pytool->y[1].log_softmax(1);//CUDA

        pytool->loss =torch::nll_loss(tt,pytool->target);//new
        pytool->loss.backward();
//3.2 compute the gradient of the second layer.     
        torch::Tensor aggregate_grad2 = unified_parameter<ValueType>(comm1,Gnn_v2->W.grad());//Gnn_v2->W.grad()
        Gnn_v2->learn(aggregate_grad2,0.01);//reset from new          
//3.3.3 backward the partial gradient from 2-layer to 1-layer    torch::Tensor partial_grad_layer2=pytool->x[1].grad();          
    gt->setGradFromTensor(pytool->x[1].grad());
        gt->Propagate_backward<SIZE_LAYER_2>(0);
//*3.3.1  compute  W1's partial gradient in first layer   
        pytool->y[0].backward();//new
        pytool->localGrad[0]=inter->W.grad();//new          
    
        
torch::Tensor new_combine_grad=torch::zeros({SIZE_LAYER_1,SIZE_LAYER_2});

for(int i=0;i<graph->owned_vertices;i++){
    new_combine_grad=new_combine_grad+
            pytool->x[0][i].reshape({SIZE_LAYER_1,1}).mm(
            pytool->localGrad[0][i].reshape({1,SIZE_LAYER_2})
            ) *gt->grad_local[0][i].reshape({1,SIZE_LAYER_2});
}


     torch::Tensor aggregate_grad = unified_parameter<ValueType>(comm,(new_combine_grad));
     Gnn_v1->learn(aggregate_grad,0.01); 

     
     
     
     
     if(graph->partition_id==0) 
         std::cout<<"LOSS:\t"<<pytool->loss<<std::endl;    
     
     if(i_i==(iterations-1)&&graph->partition_id==0){
         std::cout<<"+++++++++++++++++++++++ finish ++++++++++++++++++++++++"<<std::endl;
             if(i_i==iterations-1){
     //      std::cout<<pytool->y[1].softmax(1).log().accessor<float,2>().size(0)<<" "<<pytool->y[1].softmax(1).log().accessor<float,2>().size(1)<<std::endl;
            int correct=0;
            for(int k=0;k<embedding->rownum;k++){
                ValueType max= -100000.0;
                long id=-1;
                for(int i=0;i<LABEL_NUMBER;i++){
                    if (max<tt.accessor<ValueType,2>().data()[k*LABEL_NUMBER+i]){
                        max=tt.accessor<ValueType,2>().data()[k*LABEL_NUMBER+i];
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
        
         int correct_test=0;
            for(int k=graph->vertices;k<NODE_NUMBER;k++){
                ValueType max= -100000.0;
                long id=-1;
                torch::Tensor test=torch::from_blob(embedding->con[k].att,{1,SIZE_LAYER_1});
                torch::Tensor final_=torch::relu(test.mm(Gnn_v1->W)).mm(Gnn_v2->W).log_softmax(1);
                for(int i=0;i<LABEL_NUMBER;i++){
                    if (max<final_.accessor<ValueType,2>().data()[i]){
                        max=final_.accessor<ValueType,2>().data()[i];
                        id=i;
                    }
                }
                   if(id==pytool->target.accessor<long,1>().data()[k])
                   correct_test+=1;
//               }
            }
            std::cout<<"\ncorrect number on training:"<<correct_test<<std::endl;     
     }
     
    }
     delete active;     
}


void compute_GPU(Graph<Empty> * graph, int iterations) {
    ValueType learn_rate=0.01;
    gpu_processor *gp=new gpu_processor(); //GPU
    Embeddings<ValueType, long> *embedding = new Embeddings<ValueType, long>();
    embedding->init(graph);
    embedding->readlabel1(graph);
    
    Network<ValueType> *comm=new Network<ValueType>(graph,WEIGHT_ROW,WEIGHT_COL);
    Network<ValueType>* comm1=new Network<ValueType>(graph,WEIGHT_ROW,WEIGHT_COL);
    comm->setWsize(WEIGHT_ROW,WEIGHT_COL);
    comm1->setWsize(WEIGHT_ROW,WEIGHT_COL);
    tensorSet *pytool=new tensorSet(2);
    pytool->in_degree=torch::from_blob(graph->in_degree+graph->partition_offset[graph->partition_id],{embedding->rownum,1});
    pytool->out_degree=torch::from_blob(graph->in_degree+graph->partition_offset[graph->partition_id],{embedding->rownum,1});
/*1 INIT STAGE*/    
   // GTensor<float,Empty> gt=new  GTensor(comm, graph);
    
    GnnUnit* Gnn_v1 = new GnnUnit(SIZE_LAYER_1, SIZE_LAYER_2);
    GnnUnit* Gnn_v1_1 = new GnnUnit(SIZE_LAYER_1, SIZE_LAYER_2);//commnet
    
    GnnUnit* Gnn_v2 =new GnnUnit(SIZE_LAYER_2, OUTPUT_LAYER_3);
    GnnUnit* Gnn_v2_1 =new GnnUnit(SIZE_LAYER_2, OUTPUT_LAYER_3);//commnet
    pytool->registOptimizer(torch::optim::SGD(Gnn_v1->parameters(), 0.05));//new
    pytool->registOptimizer(torch::optim::SGD(Gnn_v2->parameters(), 0.05));//new
     pytool->registOptimizer(torch::optim::SGD(Gnn_v1_1->parameters(), 0.05));//commnet
      pytool->registOptimizer(torch::optim::SGD(Gnn_v2_1->parameters(), 0.05));//commnet
    pytool->registLabel<long>(embedding->label,graph->partition_offset[graph->partition_id],graph->partition_offset[graph->partition_id+1]-graph->partition_offset[graph->partition_id]);//new
/*init W with new */
  
    init_parameter(comm, graph,Gnn_v1, embedding);
    init_parameter(comm1, graph,Gnn_v2, embedding);
      printf("hello\n");
    VertexSubset * active = graph->alloc_vertex_subset();
    active->fill();
    std::vector<int> layer_size(0);
    layer_size.push_back(SIZE_LAYER_1);
    layer_size.push_back(SIZE_LAYER_2);
    layer_size.push_back(OUTPUT_LAYER_3);
    GTensor<ValueType,long,VECTOR_LENGTH> *gt=new GTensor<ValueType, long,VECTOR_LENGTH>(graph,embedding,active,2,layer_size); 
    gpu_processor *gp1=new gpu_processor(SIZE_LAYER_1,SIZE_LAYER_2,2208);
    
    torch::Tensor new_combine_grad=torch::zeros({SIZE_LAYER_1,SIZE_LAYER_2},torch::kFloat).cuda();
    std::vector<torch::Tensor> partial_new_combine_grad(0);
    for(int i=0;i<graph->threads;i++){
    partial_new_combine_grad.push_back(new_combine_grad);
}
    
    
    
    graph->process_vertices<ValueType>(//init  the vertex state.
        [&](VertexId vtx){
            int start=(graph->partition_offset[graph->partition_id]);
                    for(int i=0;i<SIZE_LAYER_1;i++){
                        embedding->initStartWith(vtx,embedding->con[vtx].att[i],i);
                    }     
            return (ValueType)1;
        },
    active
    );

    //   gt->Test_Propagate<SIZE_LAYER_1>(0);
     /*GPU  */
       torch::Device GPU(torch::kCUDA,0);
       torch::Device CPU(torch::kCPU,0); 
       torch::Tensor target_gpu  = pytool->target.cuda();
       torch::Tensor inter1_gpu=torch::zeros({embedding->rownum,SIZE_LAYER_2},at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));      
       torch::Tensor inter2_gpu=torch::zeros({embedding->rownum,SIZE_LAYER_2},at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
       Gnn_v2->to(GPU);
       Gnn_v1->to(GPU);
       
       
     double exec_time = 0;
    exec_time -= get_time();  
    for (int i_i = 0; i_i < iterations; i_i++) {
        if(i_i!=0){
           //inter1_gpu.grad().zero_();
           //inter2_gpu.grad().zero_(); 
            Gnn_v1->zero_grad();
            Gnn_v2->zero_grad();
        }

         gt->setValueFromNative(embedding->start_v,embedding->start);  
            
         if(graph->partition_id==0)
        std::cout<<"start  ["<<i_i<<"]  epoch+++++++++++++++++++++++++++++++++++++++++++++"<<std::endl;
//layer 1;
        gt->Propagate<SIZE_LAYER_1>(0);
        pytool->updateX(0,gt->value_local[0]); 
        
        inter1_gpu.set_data(Gnn_v1->forward(pytool->x[0].cuda()));//torch::from_blob(embedding->Gnn_v1->forward(pytool->x[0]).accessor<float,2>().data(),{embedding->rownum,VECTOR_LENGTH});
        pytool->y[0]=torch::relu(inter1_gpu);//new
        
        gt->setValueFromTensor(pytool->y[0].cpu());                  
        gt->Propagate<SIZE_LAYER_2>(1);             
        pytool->updateX(1,gt->value_local[1]);//new

        inter2_gpu.set_data(pytool->x[1].cuda());
        pytool->y[1] = Gnn_v2->forward(inter2_gpu);
        
        torch::Tensor tt=pytool->y[1].log_softmax(1);//CUDA
        pytool->loss =torch::nll_loss(tt,target_gpu);//new
        pytool->loss.backward(); 
        torch::Tensor aggregate_grad2= unified_parameter<ValueType>(comm1,Gnn_v2->W.grad().cpu()); 
      
         Gnn_v2->learn_gpu(aggregate_grad2.cuda(),learn_rate);//reset from new        
        // std::cout<<inter1->W.grad().size(0)<<"    "<<inter1->W.grad().size(1)<<std::endl;
        gt->setGradFromTensor(inter2_gpu.grad().cpu());
        gt->Propagate_backward<SIZE_LAYER_2>(0);
        pytool->y[0].backward();//new
        pytool->localGrad[0]=inter1_gpu.grad();//new          
        
        

        

//compute gradient
    for(int i=0;i<graph->threads;i++){
    partial_new_combine_grad[i].zero_();
} new_combine_grad.zero_();

torch::Tensor x0_gpu=pytool->x[0].cuda();
torch::Tensor grad0_gpu=gt->grad_local[0].cuda();        

//gp1->aggregate_grad(x0_gpu.packed_accessor<ValueType,2>().data(),
//        pytool->localGrad[0].packed_accessor<ValueType,2>().data(),
//        grad0_gpu.packed_accessor<ValueType,2>().data(),embedding->rownum);
//new_combine_grad=torch::from_blob(gp1->get_grad(),{SIZE_LAYER_1,SIZE_LAYER_2}).cuda();

    graph->process_vertices<ValueType>(//init  the vertex state.
        [&](VertexId vtx){
            int thread_id = omp_get_thread_num();
           partial_new_combine_grad[thread_id]=partial_new_combine_grad[thread_id]+
            x0_gpu[vtx].reshape({SIZE_LAYER_1,1}).mm(
            pytool->localGrad[0][vtx].reshape({1,SIZE_LAYER_2})
             *grad0_gpu[vtx].reshape({1,SIZE_LAYER_2}));
                    
            return (ValueType)1;
        },
    active
    );
    for(int i=0;i<graph->threads;i++){
       new_combine_grad=new_combine_grad+partial_new_combine_grad[i];
    }
    //learn  
     torch::Tensor aggregate_grad = unified_parameter<ValueType>(comm,new_combine_grad.cpu());
     Gnn_v1->learn_gpu(aggregate_grad.cuda(),learn_rate); 
     
     
     
     
     
     if(graph->partition_id==0) 
         std::cout<<"LOSS:\t"<<pytool->loss<<std::endl;    
     torch::Tensor tt_cpu=tt.cpu();
     if(i_i==(iterations-1)&&graph->partition_id==0){
         std::cout<<"+++++++++++++++++++++++ finish ++++++++++++++++++++++++"<<std::endl;
         exec_time += get_time();
             if (graph->partition_id==0) {
                 printf("exec_time=%lf(s)\n", exec_time);
            }
             if(i_i==iterations-1){
     //      std::cout<<pytool->y[1].softmax(1).log().accessor<float,2>().size(0)<<" "<<pytool->y[1].softmax(1).log().accessor<float,2>().size(1)<<std::endl;
            int correct=0;
            for(int k=0;k<embedding->rownum;k++){
                ValueType max= -100000.0;
                long id=-1;
                for(int i=0;i<LABEL_NUMBER;i++){
                    if (max<tt_cpu.accessor<ValueType,2>().data()[k*LABEL_NUMBER+i]){
                        max=tt_cpu.accessor<ValueType,2>().data()[k*LABEL_NUMBER+i];
                        id=i;
                    }
                }
                   if(id==pytool->target.accessor<long,1>().data()[k])
                   correct+=1;
//               }
            }
            std::cout<<"\ncorrect number on training:"<<correct<<"\t"<<((ValueType)correct/(ValueType)graph->vertices)<<std::endl;
        }
        std::cout<<"loss at"<<graph->partition_id<<"is :"<<pytool->loss<<std::endl;
        int correct_test=0;
            for(int k=graph->vertices;k<NODE_NUMBER;k++){
                ValueType max= -100000.0;
                long id=-1;
                torch::Tensor test=torch::from_blob(embedding->con[k].att,{1,SIZE_LAYER_1});
                torch::Tensor final_=torch::relu(test.mm(Gnn_v1->W.cpu())).mm(Gnn_v2->W.cpu()).log_softmax(1);
                for(int i=0;i<LABEL_NUMBER;i++){
                    if (max<final_.accessor<ValueType,2>().data()[i]){
                        max=final_.accessor<ValueType,2>().data()[i];
                        id=i;
                    }
                }
                   if(id==embedding->con[k].label)
                   correct_test+=1;
//               }
                //printf("%ld\t%ld\n",id,pytool->target.accessor<long,1>().data()[k]);
            }
            std::cout<<"\ncorrect number on testing:"<<correct_test<<"\t"<<((ValueType)correct_test/(ValueType)(NODE_NUMBER-graph->vertices))<<std::endl;
        
        
        
         
     }
    }
     delete active;     
}


void compute_single_GPU(Graph<Empty> * graph, int iterations) {
    ValueType learn_rate=0.01;
    VertexSubset * active = graph->alloc_vertex_subset();
    active->fill();
    Embeddings<ValueType, long> *embedding = new Embeddings<ValueType, long>();
    embedding->init(graph);
    embedding->readlabel1(graph);
    
    Network<ValueType> *comm=new Network<ValueType>(graph,WEIGHT_ROW,WEIGHT_COL);
    Network<ValueType>* comm1=new Network<ValueType>(graph,WEIGHT_ROW,WEIGHT_COL);
    comm->setWsize(WEIGHT_ROW,WEIGHT_COL);
    comm1->setWsize(WEIGHT_ROW,WEIGHT_COL);
    tensorSet *pytool=new tensorSet(2);
    //pytool->in_degree=torch::from_blob(graph->in_degree+graph->partition_offset[graph->partition_id],{embedding->rownum,1});
    //pytool->out_degree=torch::from_blob(graph->in_degree+graph->partition_offset[graph->partition_id],{embedding->rownum,1});
/*init GPU*/
    aggregate_engine  *ag_e=new aggregate_engine();
    ag_e->reconfig_data(graph->vertices,SIZE_LAYER_2,graph->vertices,SIZE_LAYER_1,TENSOR_TYPE);
    ag_e->init_intermediate_gradient();
    
    
    graph_engine *gr_e=new graph_engine();
    VertexId* incoming_adj_index=new VertexId[graph->vertices+1];
    VertexId* incoming_adj_index_backward=new VertexId[graph->vertices+1];
    printf("edges:%d",graph->edges);
    ValueType* weight=new ValueType[graph->edges+1];
    ValueType*weight_backward=new ValueType[graph->edges+1];
     graph->process_vertices<ValueType>(//init  the vertex state.
        [&](VertexId vtx){
           // graph->in
          incoming_adj_index[vtx]=(VertexId)graph->incoming_adj_index[0][vtx];
          incoming_adj_index_backward[vtx]=(VertexId)graph->incoming_adj_index_backward[0][vtx];
          for(int i=graph->incoming_adj_index[0][vtx];i<graph->incoming_adj_index[0][vtx+1];i++){
              VertexId dst=graph->incoming_adj_list[0][i].neighbour;
              weight[i]=(ValueType)std::sqrt(graph->in_degree[vtx])+
                                    (ValueType)std::sqrt(graph->out_degree[dst]);
          }
          for(int i=graph->incoming_adj_index_backward[0][vtx];i<graph->incoming_adj_index_backward[0][vtx+1];i++){
              VertexId dst=graph->incoming_adj_list_backward[0][i].neighbour;
              weight_backward[i]=(ValueType)std::sqrt(graph->out_degree[vtx])+
                                    (ValueType)std::sqrt(graph->in_degree[dst]);
          }

            return (ValueType)1;
        },
    active
    );incoming_adj_index[graph->vertices]=(VertexId)graph->incoming_adj_index[0][graph->vertices];
     incoming_adj_index_backward[graph->vertices]=(VertexId)graph->incoming_adj_index_backward[0][graph->vertices];   

    gr_e->load_graph(graph->vertices,graph->edges,false,
        graph->vertices,graph->edges,false,
            incoming_adj_index,(VertexId*)graph->incoming_adj_list[0],
                incoming_adj_index_backward,(VertexId*)graph->incoming_adj_list_backward[0],
                    0,graph->vertices,0,graph->vertices,SIZE_LAYER_1);
    //gr_e->redirect_input_output();
//    for(int i=2200;i<2208;i++){
//        printf("nei\t%d\t%d\n",((VertexId*)graph->incoming_adj_list[0])[i],graph->incoming_adj_list[0][i].neighbour);
//    }
    
    
/*1 INIT STAGE*/    
   // GTensor<float,Empty> gt=new  GTensor(comm, graph);
    
    
    
    GnnUnit* Gnn_v1 = new GnnUnit(SIZE_LAYER_1, SIZE_LAYER_2);
    GnnUnit* Gnn_v1_1 = new GnnUnit(SIZE_LAYER_1, SIZE_LAYER_2);//commnet
    
    GnnUnit* Gnn_v2 =new GnnUnit(SIZE_LAYER_2, OUTPUT_LAYER_3);
    GnnUnit* Gnn_v2_1 =new GnnUnit(SIZE_LAYER_2, OUTPUT_LAYER_3);//commnet
    pytool->registOptimizer(torch::optim::SGD(Gnn_v1->parameters(), 0.05));//new
    pytool->registOptimizer(torch::optim::SGD(Gnn_v2->parameters(), 0.05));//new
    pytool->registOptimizer(torch::optim::SGD(Gnn_v1_1->parameters(), 0.05));//commnet
    pytool->registOptimizer(torch::optim::SGD(Gnn_v2_1->parameters(), 0.05));//commnet
    pytool->registLabel<long>(embedding->label,graph->partition_offset[graph->partition_id],graph->partition_offset[graph->partition_id+1]-graph->partition_offset[graph->partition_id]);//new
/*init W with new */
  
    init_parameter(comm, graph,Gnn_v1, embedding);
    init_parameter(comm1, graph,Gnn_v2, embedding);
      printf("hello\n");

    std::vector<int> layer_size(0);
    layer_size.push_back(SIZE_LAYER_1);
    layer_size.push_back(SIZE_LAYER_2);
    layer_size.push_back(OUTPUT_LAYER_3);
    GTensor<ValueType,long,VECTOR_LENGTH> *gt=new GTensor<ValueType, long,VECTOR_LENGTH>(graph,embedding,active,2,layer_size); 
    
    torch::Tensor new_combine_grad=torch::zeros({SIZE_LAYER_1,SIZE_LAYER_2},torch::kFloat).cuda();
    std::vector<torch::Tensor> partial_new_combine_grad(0);
    for(int i=0;i<graph->threads;i++){
    partial_new_combine_grad.push_back(new_combine_grad);
}

    graph->process_vertices<ValueType>(//init  the vertex state.
        [&](VertexId vtx){
            int start=(graph->partition_offset[graph->partition_id]);
                    for(int i=0;i<SIZE_LAYER_1;i++){
                         embedding->initStartWith(vtx,embedding->con[vtx].att[i],i);//embedding->con[vtx].att[i]
                    }     
            return (ValueType)1;
        },
    active
    );

    //   gt->Test_Propagate<SIZE_LAYER_1>(0);
     /*GPU  */
       torch::Device GPU(torch::kCUDA,0);
       torch::Device CPU(torch::kCPU,0); 
       torch::Tensor target_gpu  = pytool->target.cuda();
       torch::Tensor inter1_gpu=torch::zeros({embedding->rownum,SIZE_LAYER_2},at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));      
       torch::Tensor inter2_gpu=torch::zeros({embedding->rownum,SIZE_LAYER_2},at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
       Gnn_v2->to(GPU);
       Gnn_v1->to(GPU);
       
     //  torch::Tensor X1_gpu=torch::zeros({embedding->rownum,SIZE_LAYER_1},at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));      
       
       torch::Tensor X0_gpu=torch::from_blob(embedding->start_v+embedding->start,{embedding->rownum,SIZE_LAYER_1},torch::kFloat).cuda();
       torch::Tensor Y0_gpu=torch::zeros({embedding->rownum,SIZE_LAYER_1},at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
       torch::Tensor X1_gpu=torch::zeros({embedding->rownum,SIZE_LAYER_2},at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
       torch::Tensor Y1_gpu=torch::zeros({embedding->rownum,SIZE_LAYER_2},at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
       torch::Tensor Y1_inv_gpu=torch::zeros({embedding->rownum,SIZE_LAYER_2},at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
       torch::Tensor Y0_inv_gpu=torch::zeros({embedding->rownum,SIZE_LAYER_2},at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
       torch::Tensor W_for_gpu=torch::from_blob(weight,{graph->edges+1,1},torch::kFloat).cuda();
       torch::Tensor W_back_gpu=torch::from_blob(weight_backward,{graph->edges+1,1},torch::kFloat).cuda();
       torch::Tensor Out0_gpu;
       torch::Tensor Out1_gpu;
       
     double exec_time = 0;
    exec_time -= get_time();  
    for (int i_i = 0; i_i < iterations; i_i++) {
        if(i_i!=0){
           //inter1_gpu.grad().zero_();
           //inter2_gpu.grad().zero_(); 
            Gnn_v1->zero_grad();
            Gnn_v2->zero_grad();
        }
           

    
         if(graph->partition_id==0)
        std::cout<<"start  ["<<i_i<<"]  epoch+++++++++++++++++++++++++++++++++++++++++++++"<<std::endl;
//layer 1;         
        gr_e->redirect_input_output(X0_gpu.packed_accessor<float,2>().data(),
            Y0_gpu.packed_accessor<float,2>().data(),
                W_for_gpu.packed_accessor<float,2>().data(), SCALA_TYPE, SIZE_LAYER_1);
         gr_e->forward();   
     
        inter1_gpu.set_data(Gnn_v1->forward(Y0_gpu));//torch::from_blob(embedding->Gnn_v1->forward(pytool->x[0]).accessor<float,2>().data(),{embedding->rownum,VECTOR_LENGTH});
        Out0_gpu=torch::relu(inter1_gpu);//new
        
//layer 2;         
        gr_e->redirect_input_output(Out0_gpu.packed_accessor<float,2>().data(),
            Y1_gpu.packed_accessor<float,2>().data(),
                W_for_gpu.packed_accessor<float,2>().data(), SCALA_TYPE, SIZE_LAYER_2);
         gr_e->forward();   
       
        inter2_gpu.set_data(Y1_gpu.cuda());
        Out1_gpu = Gnn_v2->forward(inter2_gpu);
        
//output;        
        torch::Tensor tt=Out1_gpu.log_softmax(1);//CUDA
        pytool->loss =torch::nll_loss(tt,target_gpu);//new
        pytool->loss.backward(); 

//inv layer 2;        
//        torch::Tensor aggregate_grad2= unified_parameter<ValueType>(comm1,Gnn_v2->W.grad().cpu()); 
//         Gnn_v2->learn_gpu(aggregate_grad2.cuda(),learn_rate);//reset from new   
        Gnn_v2->learn_gpu(Gnn_v2->W.grad(),learn_rate);//signle node
         

//inv layer 1;
    // 2->1
        gr_e->redirect_input_output(inter2_gpu.grad().packed_accessor<float,2>().data(),
            Y1_inv_gpu.packed_accessor<float,2>().data(),
                W_back_gpu.packed_accessor<float,2>().data(), SCALA_TYPE, SIZE_LAYER_2);
         gr_e->backward();    
    // layer 1 local     
        Out0_gpu.backward();//new  
    // layer 1 combine
        new_combine_grad.zero_();
        ag_e->redirect_input_output(Y1_inv_gpu.packed_accessor<float,2>().data(),
            inter1_gpu.grad().packed_accessor<float,2>().data(),
                Y0_gpu.packed_accessor<float,2>().data(),
                    new_combine_grad.packed_accessor<float,2>().data());
        ag_e->aggregate_grad();        
        
//learn  
//     torch::Tensor aggregate_grad = unified_parameter<ValueType>(comm,new_combine_grad.cpu());
//     Gnn_v1->learn_gpu(aggregate_grad.cuda(),learn_rate);        
        Gnn_v1->learn_gpu(new_combine_grad,learn_rate);  
 
     
  
     
     if(graph->partition_id==0) 
         std::cout<<"LOSS:\t"<<pytool->loss<<std::endl;    
     torch::Tensor tt_cpu=tt.cpu();
     if(i_i==(iterations-1)&&graph->partition_id==0){
         std::cout<<"+++++++++++++++++++++++ finish ++++++++++++++++++++++++"<<std::endl;
         exec_time += get_time();
             if (graph->partition_id==0) {
                 printf("exec_time=%lf(s)\n", exec_time);
            }
             if(i_i==iterations-1){
     //      std::cout<<pytool->y[1].softmax(1).log().accessor<float,2>().size(0)<<" "<<pytool->y[1].softmax(1).log().accessor<float,2>().size(1)<<std::endl;
            int correct=0;
            for(int k=0;k<embedding->rownum;k++){
                ValueType max= -100000.0;
                long id=-1;
                for(int i=0;i<LABEL_NUMBER;i++){
                    if (max<tt_cpu.accessor<ValueType,2>().data()[k*LABEL_NUMBER+i]){
                        max=tt_cpu.accessor<ValueType,2>().data()[k*LABEL_NUMBER+i];
                        id=i;
                    }
                }
                   if(id==pytool->target.accessor<long,1>().data()[k])
                   correct+=1;
//               }
            }
            std::cout<<"\ncorrect number on training:"<<correct<<"\t"<<((ValueType)correct/(ValueType)graph->vertices)<<std::endl;
        }
        std::cout<<"loss at"<<graph->partition_id<<"is :"<<pytool->loss<<std::endl;
        int correct_test=0;
            for(int k=graph->vertices;k<NODE_NUMBER;k++){
                ValueType max= -100000.0;
                long id=-1;
                torch::Tensor test=torch::from_blob(embedding->con[k].att,{1,SIZE_LAYER_1});
                torch::Tensor final_=torch::relu(test.mm(Gnn_v1->W.cpu())).mm(Gnn_v2->W.cpu()).log_softmax(1);
                for(int i=0;i<LABEL_NUMBER;i++){
                    if (max<final_.accessor<ValueType,2>().data()[i]){
                        max=final_.accessor<ValueType,2>().data()[i];
                        id=i;
                    }
                }
                   if(id==embedding->con[k].label)
                   correct_test+=1;
//               }
                //printf("%ld\t%ld\n",id,pytool->target.accessor<long,1>().data()[k]);
            }
            std::cout<<"\ncorrect number on testing:"<<correct_test<<"\t"<<((ValueType)correct_test/(ValueType)(NODE_NUMBER-graph->vertices))<<std::endl;
        
        
        
         
     }
    }
     delete active;     
}


int main(int argc, char ** argv) {
    MPI_Instance mpi(&argc, &argv);
 printf("hello\n");
    if (argc < 5) {
        printf("pagerank [file] [vertices] [iterations] [CPU/GPU]\n");
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
    if(std::string(argv[4])==std::string("GPU")){
        printf("%s g engine start",argv[4]);
    compute_GPU(graph, iterations);
    }else if(std::string(argv[4])==std::string("CPU")){
        printf("%s c engine start",argv[4]);
    compute(graph, iterations);
    }else if(std::string(argv[4])==std::string("TEST")){
        printf("%s c engine start",argv[4]);
    compute_single_GPU(graph, iterations);
    }else{
        ;
    }
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
    
    
//    std::vector<int> s;
//    s.push_back(1);
//    s.push_back(2);
//    s.push_back(3);
//    std::cout<<sizeof(s)<<std::endl;
//    
   if(graph->partition_id==0){
    float b1[2]={1,2};
    float b2[2]={2,4};
    float c[4]={1,1,3,4};
    torch::Tensor w=torch::from_blob(a1,{2,2});
    torch::Tensor w0=torch::from_blob(a1,{2,2});
    torch::Tensor w1=torch::from_blob(a,{2,2});
    torch::Tensor ans=torch::from_blob(c,{2,1});
    torch::Tensor x1=torch::from_blob(b1,{1,2});
    torch::Tensor y1=torch::from_blob(b2,{1,2});
//    y1.set_requires_grad(true);
    w.set_requires_grad(true);
    w0.set_requires_grad(true);
//    torch::Tensor y=x1.mm(w);
//
//    torch::Tensor y2=y.mm(w1);
//            y2.backward();
//    torch::Tensor x2=x1.mm(w);
//    torch::Tensor y2=y1.mm(w1);
//    x2.backward();
//    y2.backward();
    //ans=w.index_select(0,torch::tensor(0,torch::kLong))*w.t();
    torch::Tensor tmp=torch::ones({1,2});
    tmp=torch::from_blob(b2,{1,2});tmp.set_requires_grad(true);
    //tmp=x1.mm(w);
    //ans=torch::sigmoid(x1.mm(w));
    //ans.backward();
   // std::cout<<"::\n"<<w[torch::Scalar(0)].reshape({1,2})<<std::endl;
   // std::cout<<"::\n"<<w<<std::endl;
    //ans.index_copy(0,torch::tensor(1,torch::kLong),w.index_select(0,torch::tensor(1,torch::kLong)));
    //std::cout<<w<<"\n"<<w.grad()<<std::endl;
//    torch::Tensor inter1=torch::ones({2,2},at::TensorOptions().requires_grad(true));
//    torch::Tensor y=x1.mm(inter1);
//    std::cout<<y<<std::endl;
//    y.backward();
//    std::cout<<inter1.grad()<<std::endl;
//    inter1.set_data(w0);
//    torch::Tensor y2=x1.mm(inter1);
//    y2.backward();
//    std::cout<<inter1.grad()<<std::endl;
    std::cout<<ans.mm(x1*y1)<<std::endl;    
//    gpu_processor *s=new gpu_processor(2,2,1);
//    s->aggregate_grad(ans.packed_accessor<float,2>().data(),
//            x1.cuda().accessor<float,2>().data(),
//            y1.cuda().accessor<float,2>().data(),1);
//    std::cout<<torch::from_blob(s->get_grad(),{2,2}).cpu()<<std::endl;
    }
    
    return 0;
}
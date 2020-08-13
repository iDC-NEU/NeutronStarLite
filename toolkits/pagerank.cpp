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
#include "core/input.hpp"

// #define VECTOR_LENGTH 3703
// #define WEIGHT_ROW 3703
// #define WEIGHT_COL 3703
// #define NODE_NUMBER 3327
// #define LABEL_NUMBER 6
// #define SIZE_LAYER_1    3703
// #define SIZE_LAYER_2    16
// #define OUTPUT_LAYER_3  6

// #define VECTOR_LENGTH 500
// #define WEIGHT_ROW 500
// #define WEIGHT_COL 500
// #define NODE_NUMBER 19717
// #define LABEL_NUMBER 3
// #define SIZE_LAYER_1    500
// #define SIZE_LAYER_2    16
// #define OUTPUT_LAYER_3  3

#define VECTOR_LENGTH 1433
#define WEIGHT_ROW 1433
#define WEIGHT_COL 1433
#define NODE_NUMBER 2708
#define LABEL_NUMBER 7
#define SIZE_LAYER_1 1433
#define SIZE_LAYER_2    16 
#define OUTPUT_LAYER_3  7

//#define VECTOR_LENGTH 300
//#define WEIGHT_ROW 300
//#define WEIGHT_COL 300
//#define NODE_NUMBER 875713
//#define LABEL_NUMBER 7
//#define SIZE_LAYER_1 300
//#define SIZE_LAYER_2    16 
//#define OUTPUT_LAYER_3  7


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
struct compress_feature{
    int key;
    float value;
};

typedef struct graph_Tensor{
    torch::Tensor src;
    torch::Tensor dst;
    torch::Tensor weight;
    int edge_size;
    int batch_size;
    int feature_size;
    int src_range[2];
    int dst_range[2];
    float* weight_buffer;
}edge_list;

struct content
{
    int id;
    ValueType att[SIZE_LAYER_1];
    long label;
};//con[2708];

struct Compressed_Feature{
    VertexId* column_index;
    ValueType* value_list;
    VertexId* position;
    int count;
};
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
    Compressed_Feature* cf;
    std::vector<origin_feature> con1;
    
    
    void generate_compressed_feature(Graph<Empty>*graph){
        cf=new Compressed_Feature;
        cf->column_index=new VertexId[graph->owned_vertices+1];
        memset(cf->column_index,0,(graph->owned_vertices+1)*sizeof(VertexId));
        cf->count=0;
        int bias=graph->partition_offset[graph->partition_id];
        for(int i=0;i<graph->owned_vertices;i++){
            for(int j=0;j<SIZE_LAYER_1;j++){
                if(con[i+bias].att[j]!=0.0){
                    cf->count++;
                    cf->column_index[i+1]+=1;
                }
            }
        }
        for(int i=0;i<graph->owned_vertices;i++){
            cf->column_index[i+1]=cf->column_index[i]+cf->column_index[i+1];
        }
        std::cout<<cf->count<<std::endl;
        cf->position   = new VertexId[cf->count+1];
        cf->value_list = new ValueType[cf->count+1];
        memset(cf->position,0,(cf->count+1)*sizeof(VertexId));
        memset(cf->value_list,0,(cf->count+1)*sizeof(ValueType));
        for(int i=0;i<graph->owned_vertices;i++){
            int local_index=0;
            for(int j=0;j<SIZE_LAYER_1;j++){
                if(con[i+bias].att[j]!=0.0){
                    cf->position[cf->column_index[i]+(local_index)]=j;
                    cf->value_list[cf->column_index[i]+(local_index)]=con[i+bias].att[j];
                    local_index++; 
                }
            }
            if(local_index!=(cf->column_index[i+1]-cf->column_index[i])){
                std::cout<<i<<std::endl;
            }
        }
        std::cout<<"generate compressed_structure finished\n";
        
    }
    void init(Graph<Empty>* graph) {
        start_v = new ValueType [graph->vertices*SIZE_LAYER_1];//graph->alloc_vertex_array<float>(VECTOR_LENGTH);

        label = graph->alloc_vertex_array<T_l>();
        rownum = (graph->partition_offset[graph->partition_id + 1] - graph->partition_offset[graph->partition_id]);
        start = SIZE_LAYER_1 * (graph->partition_offset[graph->partition_id]);
        start_vertex = graph->partition_offset[graph->partition_id];
        printf("graph %ld\n",graph->vertices);
        
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
    inline void readlabel_(Graph<Empty>* graph) {
        readlabel1(graph);
        generate_compressed_feature(graph);
        
        //writedatatonodelabel();
    }
    void writedatatonodelabel(){
        std::ofstream outputf("./node_table",std::ios::out);
        std::ofstream outputtest("./test_table",std::ios::out);
        std::ofstream outputtrain("./train_table",std::ios::out);
        outputtest<<"id:int64	weight:float"<<std::endl;
        outputtrain<<"id:int64	weight:float"<<std::endl;
        outputf<<"id:int64	label:int64	feature:string"<<std::endl;
        for(int i=0;i<NODE_NUMBER;i++){
            outputf<<i<<"\t"<< con[i].label<<"\t";
            for(int j=0;j<SIZE_LAYER_1;j++){
                if(j!=(SIZE_LAYER_1-1))
                  outputf<<con[i].att[j]<<":";
                else
                  outputf<<con[i].att[j]<<std::endl;  
            }
            if(i<(0.8*NODE_NUMBER)){
                outputtrain<<i<<"\t"<<"1.0"<<std::endl;
            }else{
               outputtest<<i<<"\t"<<"1.0"<<std::endl;
            }
        }
        outputf.close();
        outputtest.close();
        outputtrain.close();
    }
    void readlabel1(Graph<Empty>* graph) {
        //graph->fill_vertex_array(label, (long) 1);
        std::string str;
        std::ifstream input("cora.content",std::ios::in);
        //std::ofstream outputl("cora.labeltable",std::ios::out);

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
        std::cout<<"finish??"<<std::endl;
    }

    void readlabel_pub(Graph<Empty>* graph){
            std::string str;
    //readfeature
     con=new content[NODE_NUMBER];
    std::ifstream input("./pubmed_data/fea.txt",std::ios::in);
    if(!input.is_open())
    {
        std::cout<<"open file fail!"<<std::endl;
    }
    std::cout<<"read data started!"<<std::endl;
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
     std::cout<<"read data finished!"<<std::endl;
    input.close();


    //readlabel
    std::ifstream inputlabel("./pubmed_data/labels.txt",std::ios::in);
    if(!inputlabel.is_open())
    {
        std::cout<<"open y file fail!"<<std::endl;
    }
    int l=0;
    while(l<NODE_NUMBER)
    {
        inputlabel>>con[l].label;
        l++;
    }
    inputlabel.close();
         for(int i=0;i<NODE_NUMBER;i++){
            if(i<graph->vertices)
            label[i]=con[i].label;
        }
    }
void readlabel_citeseer(Graph<Empty>* graph){
            std::string str;
     con=new content[NODE_NUMBER];
    std::ifstream input("./citeseer_data/fea.txt",std::ios::in);
    if(!input.is_open())
    {
        std::cout<<"open file fail!"<<std::endl;
        return;
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
    std::ifstream inputlabel("./citeseer_data/labels.txt",std::ios::in);
    if(!inputlabel.is_open())
    {
        std::cout<<"open y file fail!"<<std::endl;
        return;
    }
    int l=0;
    while(l<NODE_NUMBER)
    {
        inputlabel>>con[l].label;
        //label[l]=con[l].label;
        l++;
    }
     for(int i=0;i<NODE_NUMBER;i++){
            if(i<graph->vertices)
            label[i]=con[i].label;
        }
    inputlabel.close();
    }
void readlabel_google(Graph<Empty>* graph){
            std::string str;
     con=new content[NODE_NUMBER];
    std::ifstream input("/home/wangqg/NetBeansProjects/DataPreProcessing/google/google_contents",std::ios::in);
    if(!input.is_open())
    {
        std::cout<<"open file fail!"<<std::endl;
        return;
    }
    int n=0;
    std::string lable;
    while(n<NODE_NUMBER)
    {
        int tmpid;
        input>>tmpid;
        input>>con[n].label;
        con[n].id=n;
        for(int i=0;i<SIZE_LAYER_1;i++)
        {
            input>>con[n].att[i];
        }
        //std::cout<<std::endl;
        n++;
    }
    input.close();
     for(int i=0;i<NODE_NUMBER;i++){
            if(i<graph->vertices)
            label[i]=con[i].label;
        }
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
     
    if(graph->partition_id==1){//first layer
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
                    //if(0.0!=((nodeVector<CURRENT_LAYER_SIZE>*)curr_local)[src-start_].data[i])
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
    torch::Tensor Propagate_inplace(int local_layer,ValueType *input){
        zeroValue(local_layer);
 
    graph_->process_edges<int, nodeVector<CURRENT_LAYER_SIZE>>(// For EACH Vertex Processing
        [&](VertexId dst, VertexAdjList<Empty> incoming_adj) {//pull
            nodeVector<CURRENT_LAYER_SIZE> sum;
            memset(sum.data,0,sizeof(ValueType)*CURRENT_LAYER_SIZE);
            for (AdjUnit<Empty> * ptr = incoming_adj.begin; ptr != incoming_adj.end; ptr++) {//pull model
                VertexId src = ptr->neighbour;
                for (int i = 0; i < CURRENT_LAYER_SIZE; i++) {
                    if(0.0!=((nodeVector<CURRENT_LAYER_SIZE>*)input)[src-start_].data[i])
                    sum.data[i] += ((nodeVector<CURRENT_LAYER_SIZE>*)input)[src-start_].data[i]/((ValueType)std::sqrt(graph_->out_degree_for_backward[src])*(ValueType)std::sqrt(graph_->in_degree_for_backward[dst]));
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
                    //if(0!=((nodeVector<CURRENT_LAYER_SIZE>*)next_local)[dst-start_].data[i])
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
    void setValueFromNative(t_v* data,int offset,int length_){
        pre_value=torch::from_blob(data+offset,{rownum_,length_},torch::kFloat);
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
void generate_edge_list_Tensor(Graph<Empty> * graph, std::vector<edge_list*> &graph_partitions,int batch_size){
    graph_partitions.clear();
    for(int i=0;i<graph->graph_shard.size();i++){
        graph_partitions.push_back(new edge_list);
        graph_partitions[i]->edge_size=graph->graph_shard[i]->numofedges;
        graph_partitions[i]->dst_range[0]=graph->graph_shard[i]->dst_range[0];
        graph_partitions[i]->dst_range[1]=graph->graph_shard[i]->dst_range[1];
        graph_partitions[i]->src_range[0]=graph->graph_shard[i]->src_range[0];
        graph_partitions[i]->src_range[1]=graph->graph_shard[i]->src_range[1];
        graph_partitions[i]->batch_size=batch_size;
        graph_partitions[i]->feature_size=SIZE_LAYER_1;
        graph_partitions[i]->dst=torch::from_blob(graph->graph_shard[i]->dst_delta,{1,graph_partitions[i]->edge_size},torch::kInt32);
        graph_partitions[i]->src=torch::from_blob(graph->graph_shard[i]->src_delta,{1,graph_partitions[i]->edge_size},torch::kInt32);
        graph_partitions[i]->weight_buffer=new float[graph_partitions[i]->edge_size];
//   std::cout<<"generate_edge_list"<<graph_partitions[i]->edge_size<<" "<<std::endl;
        for(int j=0;j<graph_partitions[i]->edge_size;j++){
            VertexId v_src=graph->graph_shard[i]->src_delta[j];
            VertexId v_dst=graph->graph_shard[i]->dst_delta[j];
             graph_partitions[i]->weight_buffer[j]=(ValueType)std::sqrt(graph->out_degree_for_backward[v_src])*(ValueType)std::sqrt(graph->in_degree_for_backward[v_dst]);
        }
        graph_partitions[i]->weight=torch::from_blob(graph_partitions[i]->weight_buffer,{1,graph_partitions[i]->edge_size},torch::kFloat);
    }
    std::cout<<"generate_edge_list_Tensor"<<std::endl;
//    std::cout<<graph_partitions[12]->src_range[0]<<" "<<graph_partitions[12]->src_range[1]<<std::endl;
//    std::cout<<graph_partitions[12]->dst_range[0]<<" "<<graph_partitions[12]->dst_range[1]<<std::endl;
    //std::cout<<graph_partitions[15]->src.t()<<std::endl;
}
void propagate_forward_gpu_shard(Graph<Empty> *graph, torch::Tensor input_cpu, torch::Tensor &output_cpu,
                        std::vector<edge_list*> graph_partitions,int feature_size){
    torch::Tensor batch_graph;
    torch::Tensor output_gpu;
    torch::Tensor input_gpu;
    torch::Tensor weight_gpu;
    int src_blocks=graph->owned_vertices/graph_partitions[0]->batch_size+(int)((graph->owned_vertices%graph_partitions[0]->batch_size)>0);
    int dst_blocks=graph_partitions.size()/src_blocks;
    
    output_cpu.zero_();
    output_gpu=torch::zeros({graph_partitions[0]->batch_size,feature_size},
                                            at::TensorOptions().device_index(0).dtype(torch::kFloat));
    input_gpu=torch::zeros({graph_partitions[0]->batch_size,feature_size},
                                            at::TensorOptions().device_index(0).dtype(torch::kFloat));
    //std::cout<<"HELLO"<<std::endl;
    for(int i=0;i<dst_blocks;i++){
        output_gpu.zero_();
        for(int j=0;j<src_blocks;j++){
            input_gpu.set_data(input_cpu.slice(0,graph_partitions[i+j*dst_blocks]->src_range[0],graph_partitions[i+j*dst_blocks]->src_range[1],1).cuda()); 
                        weight_gpu=graph_partitions[i+j*dst_blocks]->weight.cuda();
            torch::Tensor src_l=graph_partitions[i+j*dst_blocks]->src.cuda();
            torch::Tensor dst_l=graph_partitions[i+j*dst_blocks]->dst.cuda();
            //kernal function call;
            /*
             * output_gpu
             * input_gpu
             * graph_partitions[i]->src
             * graph_partitions[i]->dst
             * graph_partitions[i]->weight;
             * graph_partitions[i]->src_range[0],[1];
             * graph_partitions[j]->dst_range[0],[1];
             * graph_partitions[i]->batch_size
             * graph_partitions[i]->edge_size
             * graph_partitions[i]->feature_size
             */
           // std::cout<<"run one batch"<<std::endl;
       forward_on_GPU(input_gpu.packed_accessor<float,2>().data(),output_gpu.packed_accessor<float,2>().data(),weight_gpu.packed_accessor<float,2>().data(),//data 
        (uint32_t*)src_l.packed_accessor<int,2>().data(),(uint32_t*)dst_l.packed_accessor<int,2>().data(),//graph
        graph_partitions[i+j*dst_blocks]->src_range[0], graph_partitions[i+j*dst_blocks]->src_range[1],
        graph_partitions[i+j*src_blocks]->dst_range[0], graph_partitions[i+j*dst_blocks]->dst_range[1],
	graph_partitions[i+j*dst_blocks]->edge_size,graph_partitions[i+j*dst_blocks]->batch_size,feature_size);
            
        }
        torch::Tensor output_slice=output_cpu.slice(0,graph_partitions[i]->dst_range[0],graph_partitions[i]->dst_range[1],1);
        //std::cout<<"size"<<output_slice.size(0)<<" "<<output_gpu.size(0);
        if(i!=(dst_blocks-1)){
            output_slice+=output_gpu.cpu();
        }
        else{
            output_slice+=output_gpu.slice(0,0,output_slice.size(0),1).cpu();
        }
        
    }
    
}

void propagate_backward_gpu_shard(Graph<Empty> *graph, torch::Tensor input_cpu, torch::Tensor &output_cpu,
                        std::vector<edge_list*> graph_partitions,int partitions){
    torch::Tensor output_gpu;
    torch::Tensor input_gpu;
    output_cpu.zero_();
    for(int i=0;i<partitions;i++){
        output_gpu=torch::zeros({graph_partitions[i]->batch_size,graph_partitions[i]->feature_size},
                                            at::TensorOptions().device_index(0).dtype(torch::kFloat));
        for(int j=0;j<partitions;j++){
            input_gpu=input_cpu.slice(0,graph_partitions[i]->src_range[0],graph_partitions[i]->src_range[1],1).cuda();
            //kernal function call;
            /*
             * output_gpu
             * input_gpu
             * graph_partitions[i]->src
             * graph_partitions[i]->dst
             * graph_partitions[i]->weight;
             * graph_partitions[i]->src_range[0],[1];
             * graph_partitions[j]->dst_range[0],[1];
             * graph_partitions[i]->batch_size
             * graph_partitions[i]->edge_size
             * graph_partitions[i]->feature_size
             */
            backward_on_GPU(input_gpu.packed_accessor<float,2>().data(),output_gpu.packed_accessor<float,2>().data(),graph_partitions[i]->weight.packed_accessor<float,2>().data(),//data 
            (uint32_t*)graph_partitions[i]->src.packed_accessor<int,2>().data(),(uint32_t*)graph_partitions[i]->dst.packed_accessor<int,2>().data(),//graph
            graph_partitions[i]->src_range[0], graph_partitions[i]->src_range[1],
            graph_partitions[i]->dst_range[0], graph_partitions[i]->dst_range[1],
            graph_partitions[i]->edge_size,graph_partitions[i]->batch_size,graph_partitions[i]->feature_size);   
        }
        torch::Tensor output_slice=output_cpu.slice(0,graph_partitions[i]->dst_range[0],graph_partitions[i]->dst_range[1],1);
        output_slice+=output_gpu.cpu();
        
    }
    
}
void generate_weight_and_csr(Graph<Empty> *graph,  VertexSubset * active,
        VertexId* incoming_adj_index, VertexId* incoming_adj_index_backward,
        float* weight, float *weight_backward, torch::Tensor& weight_forward_T,torch::Tensor& weight_backward_T){
         graph->process_vertices<ValueType>(//init  the vertex state.
        [&](VertexId vtx){
           // graph->in
          incoming_adj_index[vtx]=(VertexId)graph->incoming_adj_index[0][vtx];
          incoming_adj_index_backward[vtx]=(VertexId)graph->incoming_adj_index_backward[0][vtx];
          for(int i=graph->incoming_adj_index[0][vtx];i<graph->incoming_adj_index[0][vtx+1];i++){
              VertexId dst=graph->incoming_adj_list[0][i].neighbour;
              weight[i]=(ValueType)std::sqrt(graph->in_degree_for_backward[vtx])*(ValueType)std::sqrt(graph->out_degree_for_backward[dst]);
              if(weight[i]==0){
                  std::cout<<"alert"<<i<<std::endl;
              }
          }
          for(int i=graph->incoming_adj_index_backward[0][vtx];i<graph->incoming_adj_index_backward[0][vtx+1];i++){
              VertexId dst=graph->incoming_adj_list_backward[0][i].neighbour;
              weight_backward[i]=(ValueType)std::sqrt(graph->out_degree_for_backward[vtx])*(ValueType)std::sqrt(graph->in_degree_for_backward[dst]);
               if(weight_backward[i]==0){
                  std::cout<<"alert"<<i<<std::endl;
              }
          }

            return (ValueType)1;
        },
    active
    );
    incoming_adj_index[graph->vertices]=(VertexId)graph->incoming_adj_index[0][graph->vertices];
    incoming_adj_index_backward[graph->vertices]=(VertexId)graph->incoming_adj_index_backward[0][graph->vertices]; 
    weight_forward_T=torch::from_blob(weight,{graph->edges+1,1},torch::kFloat).cuda();
    weight_backward_T=torch::from_blob(weight_backward,{graph->edges+1,1},torch::kFloat).cuda();  
}
 void inference(torch::Tensor tt_cpu,Graph<Empty> * graph, Embeddings<ValueType, long> *embedding,
                tensorSet *pytool,GnnUnit* Gnn_v1,GnnUnit* Gnn_v2){
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
            }
            std::cout<<"\ncorrect number on training:"<<correct<<"\t"<<((ValueType)correct/(ValueType)graph->vertices)<<std::endl;
        std::cout<<"loss at"<<graph->partition_id<<"is :"<<pytool->loss<<std::endl;
        int correct_test=0;
            for(int k=graph->vertices;k<NODE_NUMBER;k++){
                ValueType max= -100000.0;
                long id=-1;
                torch::Tensor test=torch::from_blob(&(embedding->con[k].att[0]),{1,SIZE_LAYER_1});
                torch::Tensor final_=torch::relu(test.mm(Gnn_v1->W.cpu())).mm(Gnn_v2->W.cpu()).log_softmax(1);
                for(int i=0;i<LABEL_NUMBER;i++){
                    if (max<final_.accessor<ValueType,2>().data()[i]){
                        max=final_.accessor<ValueType,2>().data()[i];
                        id=i;
                    }
                }
                   if(id==embedding->con[k].label)
                   correct_test+=1;
            }
            std::cout<<"\ncorrect number on testing:"<<correct_test<<"\t"<<((ValueType)correct_test/(ValueType)(NODE_NUMBER-graph->vertices))<<std::endl;
        }
/* CPU single experiment*/
 void compute(Graph<Empty> * graph, int iterations) {
    
   
	//gpu_processor *gp=new gpu_processor(); 
    Embeddings<ValueType, long> *embedding = new Embeddings<ValueType, long>();
    embedding->init(graph);
    embedding->readlabel_(graph);
    
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


torch::Tensor new_combine_grad=torch::zeros({SIZE_LAYER_1,SIZE_LAYER_2});
    double exec_time = 0;
     double graph_time = 0;
    exec_time -= get_time(); 
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
        
        std::cout<<pytool->x[0].size(0)<<"TETSTETSTETSTE"<<pytool->x[0].size(1)<<std::endl;
        inter->W.set_data(Gnn_v1->forward(pytool->x[0]));//torch::from_blob(embedding->Gnn_v1->forward(pytool->x[0]).accessor<float,2>().data(),{embedding->rownum,VECTOR_LENGTH});
        pytool->y[0]=torch::relu(inter->W);//new
        
        
//        std::cout<<pytool->y[0].size(0)<<"\t"<<pytool->y[0].size(1)<<std::endl;
//2.2.1 init the second layer               
        gt->setValueFromTensor(pytool->y[0]); 
//2.2.2 forward the second layer                     
        gt->Propagate<SIZE_LAYER_2>(1);             
/*3 BACKWARD STAGE*/
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

new_combine_grad.zero_();
/*graph->process_vertices<ValueType>(//init  the vertex state.
        [&](VertexId vtx){
         new_combine_grad_local[omp_get_thread_num()]=new_combine_grad[omp_get_thread_num()]+
            pytool->x[0].slice(0,vtx,vtx+1,1).t().mm(
            (pytool->localGrad[0].slice(0,vtx,vtx+1,1)*gt->grad_local[0].slice(0,vtx,vtx+1,1)));
            return (ValueType)1;
        },
    active
    );
for(int i=0;i<new_combine_grad_local.size();i++){
      new_combine_grad+=new_combine_grad_local[i];
    }*/
new_combine_grad=pytool->x[0].slice(0,0,+graph->owned_vertices,1).t().mm(gt->grad_local[0]);

     torch::Tensor aggregate_grad = unified_parameter<ValueType>(comm,(new_combine_grad));
     Gnn_v1->learn(aggregate_grad,0.01); 

     
     
     
     
     if(graph->partition_id==0){ 
         std::cout<<"LOSS:\t"<<pytool->loss<<std::endl;    
     }
     
//     if(i_i==(iterations-1)&&graph->partition_id==0){
//        inference(tt,graph, embedding, pytool,Gnn_v1,Gnn_v2);
//     }
     
    }
      exec_time += get_time();
             if (graph->partition_id==0) {
                 printf("exec_time=%lf(s)\n", exec_time);
            }
     delete active;     
}
/*CPU single order*/
 void compute_gf(Graph<Empty> * graph, int iterations) {
    
   
	//gpu_processor *gp=new gpu_processor(); 
    Embeddings<ValueType, long> *embedding = new Embeddings<ValueType, long>();
    embedding->init(graph);
    embedding->readlabel_(graph);
    
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

 std::vector<torch::Tensor>new_combine_grad_local(graph->threads,torch::zeros({SIZE_LAYER_1,SIZE_LAYER_2})); 
 torch::Tensor x0=torch::from_blob(embedding->start_v,{graph->vertices,SIZE_LAYER_1});
torch::Tensor new_combine_grad=torch::zeros({SIZE_LAYER_1,SIZE_LAYER_2});
    double exec_time = 0;
     double graph_time = 0;
    exec_time -= get_time(); 
    for (int i_i = 0; i_i < iterations; i_i++) {
             
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
        torch::Tensor x0_trans=Gnn_v1->forward(x0);
        //std::cout<<"stage1"<<std::endl;
        gt->setValueFromNative(x0_trans.accessor<float,2>().data(),graph->partition_offset[graph->partition_id]*SIZE_LAYER_2); 
        //std::cout<<"stage2"<<std::endl;
        gt->Propagate<SIZE_LAYER_2>(0);
        pytool->updateX(0,gt->value_local[0]); 
        
        //std::cout<<pytool->x[0].size(0)<<"TETSTETSTETSTE"<<pytool->x[0].size(1)<<std::endl;
        inter->W.set_data(pytool->x[0]);//torch::from_blob(embedding->Gnn_v1->forward(pytool->x[0]).accessor<float,2>().data(),{embedding->rownum,VECTOR_LENGTH});
        pytool->y[0]=torch::relu(inter->W);//new
        
        
//        std::cout<<pytool->y[0].size(0)<<"\t"<<pytool->y[0].size(1)<<std::endl;
//2.2.1 init the second layer               
        gt->setValueFromTensor(pytool->y[0]); 
//2.2.2 forward the second layer                     
        gt->Propagate<SIZE_LAYER_2>(1);             
/*3 BACKWARD STAGE*/
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
        torch::Tensor tmpgrad=pytool->localGrad[0]*gt->grad_local[0];
        gt->setGradFromTensor(tmpgrad);
        gt->Propagate_backward<SIZE_LAYER_2>(1);
        
        


//for(int i=0;i<graph->owned_vertices;i++){
//    new_combine_grad=new_combine_grad+
//            pytool->x[0].index_select(0,torch::tensor(i,torch::kLong)).t().mm(
//            pytool->localGrad[0].index_select(0,torch::tensor(i,torch::kLong))
//            )*gt->grad_local[0].index_select(0,torch::tensor(i,torch::kLong));
//}

//std::cout<<"SIZE_LAYER 1"<<SIZE_LAYER_1<<"SIZE_LAYER_2"<<SIZE_LAYER_2<<std::endl;
//std::cout<<"threadnum"<<graph->threads<<std::endl;
//std::cout<<pytool->x[0].size(0)<<"     "<<pytool->x[0].size(1)<<std::endl;
//std::cout<<pytool->localGrad[0].size(0)<<"     "<<pytool->localGrad[0].size(1)<<std::endl;
//std::cout<<gt->grad_local[0].size(0)<<"     "<<gt->grad_local[0].size(1)<<std::endl;

new_combine_grad.zero_();
printf("hellow\n");
std::cout<<x0.size(0)<<"pp"<< x0.size(1)<<std::endl;
std::cout<<gt->grad_local[0].size(0)<<"qq"<< gt->grad_local[0].size(1)<<std::endl;

/*
graph->process_vertices<ValueType>(//init  the vertex state.
        [&](VertexId vtx){
         new_combine_grad_local[omp_get_thread_num()]=new_combine_grad[omp_get_thread_num()]+
            x0.slice(0,vtx-graph->partition_offset[graph->partition_id],vtx+1-graph->partition_offset[graph->partition_id],1).t().mm(gt->grad_local[1].slice(0,vtx-graph->partition_offset[graph->partition_id],vtx+1-graph->partition_offset[graph->partition_id],1));
            return (ValueType)1;
        },
    active
    );
for(int i=0;i<new_combine_grad_local.size();i++){
      new_combine_grad+=new_combine_grad_local[i];
    }*/
int offset_=graph->partition_offset[graph->partition_id];
new_combine_grad=x0.slice(0,offset_,offset_+graph->owned_vertices,1).t().mm(gt->grad_local[1]);

     torch::Tensor aggregate_grad = unified_parameter<ValueType>(comm,(new_combine_grad));
     Gnn_v1->learn(aggregate_grad,0.01); 

     
     
     
     
     if(graph->partition_id==0){ 
         std::cout<<"LOSS:\t"<<pytool->loss<<std::endl;    
     }
     
//     if(i_i==(iterations-1)&&graph->partition_id==0){
//        inference(tt,graph, embedding, pytool,Gnn_v1,Gnn_v2);
//     }
     
    }
      exec_time += get_time();
             if (graph->partition_id==0) {
                 printf("exec_time=%lf(s)\n", exec_time);
            }
     delete active;     
}
void compute_GPU(Graph<Empty> * graph, int iterations) {
    ValueType learn_rate=0.01;
    //gpu_processor *gp=new gpu_processor(); //GPU
    Embeddings<ValueType, long> *embedding = new Embeddings<ValueType, long>();
    embedding->init(graph);
    embedding->readlabel_(graph);
    
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
    VertexSubset * active = graph->alloc_vertex_subset();
    active->fill();
    std::vector<int> layer_size(0);
    layer_size.push_back(SIZE_LAYER_1);
    layer_size.push_back(SIZE_LAYER_2);
    layer_size.push_back(OUTPUT_LAYER_3);
    GTensor<ValueType,long,VECTOR_LENGTH> *gt=new GTensor<ValueType, long,VECTOR_LENGTH>(graph,embedding,active,2,layer_size); 
    //gpu_processor *gp1=new gpu_processor(SIZE_LAYER_1,SIZE_LAYER_2,2208);
    
    torch::Tensor new_combine_grad=torch::zeros({SIZE_LAYER_1,SIZE_LAYER_2},torch::kFloat).cuda();
    std::vector<torch::Tensor> partial_new_combine_grad(0);
    for(int i=0;i<graph->threads;i++){
    partial_new_combine_grad.push_back(new_combine_grad);
}
     aggregate_engine  *ag_e=new aggregate_engine();
    ag_e->reconfig_data(graph->vertices,SIZE_LAYER_2,graph->vertices,SIZE_LAYER_1,TENSOR_TYPE);
    ag_e->init_intermediate_gradient();
    
    
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
//           inter2_gpu.grad().zero_(); 
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
        new_combine_grad.zero_();
        
      new_combine_grad=x0_gpu.t().mm(pytool->localGrad[0]*grad0_gpu);
        
        /*ag_e->redirect_input_output(grad0_gpu.packed_accessor<float,2>().data(),
            pytool->localGrad[0].packed_accessor<float,2>().data(),
                x0_gpu.packed_accessor<float,2>().data(),
                    new_combine_grad.packed_accessor<float,2>().data());
        ag_e->aggregate_grad(); */       

    //learn  
     torch::Tensor aggregate_grad = unified_parameter<ValueType>(comm,new_combine_grad.cpu());
     Gnn_v1->learn_gpu(aggregate_grad.cuda(),learn_rate); 
     
     

     if(graph->partition_id==0) 
         std::cout<<"LOSS:\t"<<pytool->loss<<std::endl; 

     torch::Tensor tt_cpu=tt.cpu();
     if(i_i==(iterations-1)&&graph->partition_id==0){
         exec_time += get_time();
             if (graph->partition_id==0) {
                 printf("exec_time=%lf(s)\n", exec_time);
                 std::cout<<graph->vertices<<" "<<graph->edges<<std::endl;
            }
         inference(tt_cpu,graph, embedding, pytool,Gnn_v1,Gnn_v2);
          
     }
    }
     delete active;     
}
/*GPU single*/
void compute_single_GPU(Graph<Empty> * graph, int iterations) {
    ValueType learn_rate=0.01;
    VertexSubset * active = graph->alloc_vertex_subset();
    active->fill();
    Embeddings<ValueType, long> *embedding = new Embeddings<ValueType, long>();
    embedding->init(graph);
    embedding->readlabel_(graph);

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
    graph->generate_COO(active);
    graph->reorder_COO(4096);
    
    VertexId* incoming_adj_index=new VertexId[graph->vertices+1];
    VertexId* incoming_adj_index_backward=new VertexId[graph->vertices+1];
    ValueType* weight=new ValueType[graph->edges+1];
    ValueType*weight_backward=new ValueType[graph->edges+1];
     graph->process_vertices<ValueType>(//init  the vertex state.
        [&](VertexId vtx){
           // graph->in
          incoming_adj_index[vtx]=(VertexId)graph->incoming_adj_index[0][vtx];
          incoming_adj_index_backward[vtx]=(VertexId)graph->incoming_adj_index_backward[0][vtx];
          for(int i=graph->incoming_adj_index[0][vtx];i<graph->incoming_adj_index[0][vtx+1];i++){
              VertexId dst=graph->incoming_adj_list[0][i].neighbour;
              weight[i]=(ValueType)std::sqrt(graph->in_degree_for_backward[vtx])*(ValueType)std::sqrt(graph->out_degree_for_backward[dst]);
              if(weight[i]==0){
                  std::cout<<"alert"<<i<<std::endl;
              }
          }
          for(int i=graph->incoming_adj_index_backward[0][vtx];i<graph->incoming_adj_index_backward[0][vtx+1];i++){
              VertexId dst=graph->incoming_adj_list_backward[0][i].neighbour;
              weight_backward[i]=(ValueType)std::sqrt(graph->out_degree_for_backward[vtx])*(ValueType)std::sqrt(graph->in_degree_for_backward[dst]);
               if(weight_backward[i]==0){
                  std::cout<<"alert"<<i<<std::endl;
              }
          }

            return (ValueType)1;
        },
    active
    );
    incoming_adj_index[graph->vertices]=(VertexId)graph->incoming_adj_index[0][graph->vertices];
    incoming_adj_index_backward[graph->vertices]=(VertexId)graph->incoming_adj_index_backward[0][graph->vertices];   

    gr_e->load_graph(graph->vertices,graph->edges,false,
        graph->vertices,graph->edges,false,
            incoming_adj_index,(VertexId*)graph->incoming_adj_list[0],
                incoming_adj_index_backward,(VertexId*)graph->incoming_adj_list_backward[0],
                    0,graph->vertices,0,graph->vertices,SIZE_LAYER_1);

    //  gr_e->load_graph_for_COO(graph->vertices,graph->edges,false,
    //         graph->_graph_cpu->srcList,graph->_graph_cpu->dstList,
    //                 0,graph->vertices,0,graph->vertices,SIZE_LAYER_1);
    // gr_e->_graph_cuda->init_partitions(graph->partitions,graph->_graph_cpu->partition_offset,graph->partition_id);

/*1 INIT STAGE*/    
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
//     torch::Tensor inter2_gpu=torch::zeros({embedding->rownum,SIZE_LAYER_2},at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
       Gnn_v2->to(GPU);
       Gnn_v1->to(GPU);
       
     //  torch::Tensor X1_gpu=torch::zeros({embedding->rownum,SIZE_LAYER_1},at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));      
       
       torch::Tensor X0_gpu=torch::from_blob(embedding->start_v+embedding->start,{embedding->rownum,SIZE_LAYER_1},torch::kFloat).cuda();
       torch::Tensor Y0_gpu=torch::zeros({embedding->rownum,SIZE_LAYER_1},at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
       torch::Tensor Y1_gpu=torch::zeros({embedding->rownum,SIZE_LAYER_2},at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
       torch::Tensor Y1_inv_gpu=torch::zeros({embedding->rownum,SIZE_LAYER_2},at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
       torch::Tensor W_for_gpu=torch::from_blob(weight,{graph->edges+1,1},torch::kFloat).cuda();
       torch::Tensor W_back_gpu=torch::from_blob(weight_backward,{graph->edges+1,1},torch::kFloat).cuda();
       torch::Tensor Out0_gpu;
       torch::Tensor Out1_gpu;
       
       
     double exec_time = 0;
     double graph_time = 0;
    exec_time -= get_time(); 
    double tmp_time=0;
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
        tmp_time-= get_time();       
        gr_e->forward_one_step(X0_gpu.packed_accessor<float,2>().data(),
            Y0_gpu.packed_accessor<float,2>().data(),
                W_for_gpu.packed_accessor<float,2>().data(), SCALA_TYPE, SIZE_LAYER_1);
        inter1_gpu.set_data(Gnn_v1->forward(Y0_gpu));
        tmp_time+=get_time();
        graph_time+=tmp_time;
        tmp_time=0;
        Out0_gpu=torch::relu(inter1_gpu);//new

//layer 2;    
        tmp_time-= get_time();      
        gr_e->forward_one_step(Out0_gpu.packed_accessor<float,2>().data(),
            Y1_gpu.packed_accessor<float,2>().data(),
                W_for_gpu.packed_accessor<float,2>().data(), SCALA_TYPE, SIZE_LAYER_2);  
        Out1_gpu = Gnn_v2->forward(Y1_gpu);
        tmp_time+=get_time();
        graph_time+=tmp_time;
        tmp_time=0;
//output;        
        torch::Tensor tt=Out1_gpu.log_softmax(1);//CUDA
        pytool->loss =torch::nll_loss(tt,target_gpu);//new
        pytool->loss.backward();
//inv layer 2;        
        Gnn_v2->learn_gpu(Gnn_v2->W.grad(),learn_rate);//signle node     
//inv layer 1;
    // 2->1
    tmp_time-= get_time(); 
        gr_e->backward_one_step(Y1_gpu.grad().packed_accessor<float,2>().data(),
            Y1_inv_gpu.packed_accessor<float,2>().data(),
                W_back_gpu.packed_accessor<float,2>().data(), SCALA_TYPE, SIZE_LAYER_2); 
       tmp_time+=get_time();
        graph_time+=tmp_time;
        tmp_time=0;
                

    // layer 1 local     
        Out0_gpu.backward();//new  
    // layer 1 combine
        new_combine_grad.zero_();
        
        /*ag_e->redirect_input_output(Y1_inv_gpu.packed_accessor<float,2>().data(),
            inter1_gpu.grad().packed_accessor<float,2>().data(),
                Y0_gpu.packed_accessor<float,2>().data(),
                    new_combine_grad.packed_accessor<float,2>().data());
        ag_e->aggregate_grad();        */
        new_combine_grad=Y0_gpu.t().mm(Y1_inv_gpu*inter1_gpu.grad());
//learn  
        Gnn_v1->learn_gpu(new_combine_grad,learn_rate); 
         

     if(graph->partition_id==0) 
         std::cout<<"LOSS:\t"<<pytool->loss<<std::endl;    
     if(i_i==(iterations-1)&&graph->partition_id==0){
         torch::Tensor tt_cpu=tt.cpu();
         std::cout<<"+++++++++++++++++++++++ finish ++++++++++++++++++++++++"<<std::endl;
         exec_time += get_time();
             if (graph->partition_id==0) {
                 printf("exec_time=%lf(s)  graph_time=%lf(s)\n", exec_time,graph_time);
                 std::cout<<graph->vertices<<" "<<graph->edges<<std::endl;
            }
             if(i_i==iterations-1){
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
            }
            std::cout<<"\ncorrect number on training:"<<correct<<"\t"<<((ValueType)correct/(ValueType)graph->vertices)<<std::endl;
        }
        std::cout<<"loss at"<<graph->partition_id<<"is :"<<pytool->loss<<std::endl;
        int correct_test=0;
            for(int k=graph->vertices;k<NODE_NUMBER;k++){
                ValueType max= -100000.0;
                long id=-1;
                torch::Tensor test=torch::from_blob(&(embedding->con[k].att[0]),{1,SIZE_LAYER_1});
                torch::Tensor final_=torch::relu(test.mm(Gnn_v1->W.cpu())).mm(Gnn_v2->W.cpu()).log_softmax(1);
                for(int i=0;i<LABEL_NUMBER;i++){
                    if (max<final_.accessor<ValueType,2>().data()[i]){
                        max=final_.accessor<ValueType,2>().data()[i];
                        id=i;
                    }
                }
                   if(id==embedding->con[k].label)
                   correct_test+=1;
            }
            std::cout<<"\ncorrect number on testing:"<<correct_test<<"\t"<<((ValueType)correct_test/(ValueType)(NODE_NUMBER-graph->vertices))<<std::endl;
        
        
        
         
     }
    }
     delete active;     
}
/*GPU single order*/void compute_single_gf_GPU(Graph<Empty> * graph, int iterations) {
    ValueType learn_rate=0.01;
    VertexSubset * active = graph->alloc_vertex_subset();
    active->fill();
    Embeddings<ValueType, long> *embedding = new Embeddings<ValueType, long>();
    embedding->init(graph);
    embedding->readlabel_(graph);
    
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
    graph->generate_COO(active);
    graph->reorder_COO(4096);
    
    VertexId* incoming_adj_index=new VertexId[graph->vertices+1];
    VertexId* incoming_adj_index_backward=new VertexId[graph->vertices+1];
    ValueType* weight=new ValueType[graph->edges+1];
    ValueType*weight_backward=new ValueType[graph->edges+1];
     graph->process_vertices<ValueType>(//init  the vertex state.
        [&](VertexId vtx){
           // graph->in
          incoming_adj_index[vtx]=(VertexId)graph->incoming_adj_index[0][vtx];
          incoming_adj_index_backward[vtx]=(VertexId)graph->incoming_adj_index_backward[0][vtx];
          for(int i=graph->incoming_adj_index[0][vtx];i<graph->incoming_adj_index[0][vtx+1];i++){
              VertexId dst=graph->incoming_adj_list[0][i].neighbour;
              weight[i]=(ValueType)std::sqrt(graph->in_degree[vtx])*(ValueType)std::sqrt(graph->out_degree[dst]);
          }
          for(int i=graph->incoming_adj_index_backward[0][vtx];i<graph->incoming_adj_index_backward[0][vtx+1];i++){
              VertexId dst=graph->incoming_adj_list_backward[0][i].neighbour;
              weight_backward[i]=(ValueType)std::sqrt(graph->out_degree[vtx])*(ValueType)std::sqrt(graph->in_degree[dst]);
          }

            return (ValueType)1;
        },
    active
    );
    incoming_adj_index[graph->vertices]=(VertexId)graph->incoming_adj_index[0][graph->vertices];
    incoming_adj_index_backward[graph->vertices]=(VertexId)graph->incoming_adj_index_backward[0][graph->vertices];   

    gr_e->load_graph(graph->vertices,graph->edges,false,
        graph->vertices,graph->edges,false,
            incoming_adj_index,(VertexId*)graph->incoming_adj_list[0],
                incoming_adj_index_backward,(VertexId*)graph->incoming_adj_list_backward[0],
                    0,graph->vertices,0,graph->vertices,SIZE_LAYER_1);

    //  gr_e->load_graph_for_COO(graph->vertices,graph->edges,false,
    //         graph->_graph_cpu->srcList,graph->_graph_cpu->dstList,
    //                 0,graph->vertices,0,graph->vertices,SIZE_LAYER_1);
    // gr_e->_graph_cuda->init_partitions(graph->partitions,graph->_graph_cpu->partition_offset,graph->partition_id);

/*1 INIT STAGE*/    
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

    std::vector<int> layer_size(0);
    layer_size.push_back(SIZE_LAYER_1);
    layer_size.push_back(SIZE_LAYER_2);
    layer_size.push_back(OUTPUT_LAYER_3);
    GTensor<ValueType,long,VECTOR_LENGTH> *gt=new GTensor<ValueType, long,VECTOR_LENGTH>(graph,embedding,active,2,layer_size); 
    
    torch::Tensor new_combine_grad=torch::zeros({SIZE_LAYER_1,SIZE_LAYER_2},torch::kFloat).cuda();

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
//     torch::Tensor inter2_gpu=torch::zeros({embedding->rownum,SIZE_LAYER_2},at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
       Gnn_v2->to(GPU);
       Gnn_v1->to(GPU);
       
     //  torch::Tensor X1_gpu=torch::zeros({embedding->rownum,SIZE_LAYER_1},at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));      
       
       torch::Tensor X0_gpu=torch::from_blob(embedding->start_v+embedding->start,{embedding->rownum,SIZE_LAYER_1},torch::kFloat).cuda();
       torch::Tensor X0_gpu_trans=torch::zeros({embedding->rownum,SIZE_LAYER_2},torch::kFloat).cuda();
       torch::Tensor Y0_gpu=torch::zeros({embedding->rownum,SIZE_LAYER_2},at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
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

        X0_gpu_trans=Gnn_v1->forward(X0_gpu);
        gr_e->forward_one_step(X0_gpu_trans.packed_accessor<float,2>().data(),
            Y0_gpu.packed_accessor<float,2>().data(),
                W_for_gpu.packed_accessor<float,2>().data(), SCALA_TYPE, SIZE_LAYER_2);

        Out0_gpu=torch::relu(Y0_gpu);
//layer 2;         
        gr_e->forward_one_step(Out0_gpu.packed_accessor<float,2>().data(),
            Y1_gpu.packed_accessor<float,2>().data(),
                W_for_gpu.packed_accessor<float,2>().data(), SCALA_TYPE, SIZE_LAYER_2);  
        Out1_gpu = Gnn_v2->forward(Y1_gpu);
//output;        
        torch::Tensor tt=Out1_gpu.log_softmax(1);//CUDA
        pytool->loss =torch::nll_loss(tt,target_gpu);//new
        pytool->loss.backward();
//inv layer 2;        
        Gnn_v2->learn_gpu(Gnn_v2->W.grad(),learn_rate);//signle node     
//inv layer 1;

    // 2->1
        gr_e->backward_one_step(Y1_gpu.grad().packed_accessor<float,2>().data(),
            Y1_inv_gpu.packed_accessor<float,2>().data(),
                W_back_gpu.packed_accessor<float,2>().data(), SCALA_TYPE, SIZE_LAYER_2); 

    // layer 1 local     
        Out0_gpu.backward();//new  
        torch::Tensor tmp=Y1_inv_gpu*Y0_gpu.grad();
        gr_e->backward_one_step(tmp.packed_accessor<float,2>().data(),
            Y0_inv_gpu.packed_accessor<float,2>().data(),
                W_back_gpu.packed_accessor<float,2>().data(), SCALA_TYPE, SIZE_LAYER_2); 

    // layer 1 combine
        new_combine_grad.zero_();
        new_combine_grad=X0_gpu.t().mm(Y0_inv_gpu);       
//learn  

        Gnn_v1->learn_gpu(new_combine_grad,learn_rate);  

     if(graph->partition_id==0) 
         std::cout<<"LOSS:\t"<<pytool->loss<<std::endl;    
     if(i_i==(iterations-1)&&graph->partition_id==0){
         torch::Tensor tt_cpu=tt.cpu();
         std::cout<<"+++++++++++++++++++++++ finish ++++++++++++++++++++++++"<<std::endl;
         exec_time += get_time();
             if (graph->partition_id==0) {
                 printf("exec_time=%lf(s)\n", exec_time);
            }
             if(i_i==iterations-1){
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
            }
            std::cout<<"\ncorrect number on training:"<<correct<<"\t"<<((ValueType)correct/(ValueType)graph->vertices)<<std::endl;
        }
        std::cout<<"loss at"<<graph->partition_id<<"is :"<<pytool->loss<<std::endl;
        int correct_test=0;
            for(int k=graph->vertices;k<NODE_NUMBER;k++){
                ValueType max= -100000.0;
                long id=-1;
                torch::Tensor test=torch::from_blob(&(embedding->con[k].att[0]),{1,SIZE_LAYER_1});
                torch::Tensor final_=torch::relu(test.mm(Gnn_v1->W.cpu())).mm(Gnn_v2->W.cpu()).log_softmax(1);
                for(int i=0;i<LABEL_NUMBER;i++){
                    if (max<final_.accessor<ValueType,2>().data()[i]){
                        max=final_.accessor<ValueType,2>().data()[i];
                        id=i;
                    }
                }
                   if(id==embedding->con[k].label)
                   correct_test+=1;
            }
            std::cout<<"\ncorrect number on testing:"<<correct_test<<"\t"<<((ValueType)correct_test/(ValueType)(NODE_NUMBER-graph->vertices))<<std::endl;     
     }

    }
     delete active;     
}
void compute_dist_GPU(Graph<Empty> * graph, int iterations) {
    ValueType learn_rate=0.01;
    VertexSubset * active = graph->alloc_vertex_subset();
    active->fill();
    Embeddings<ValueType, long> *embedding = new Embeddings<ValueType, long>();
    embedding->init(graph);
    embedding->readlabel_(graph);
    
    Network<ValueType> *comm=new Network<ValueType>(graph,WEIGHT_ROW,WEIGHT_COL);
    Network<ValueType>* comm1=new Network<ValueType>(graph,WEIGHT_ROW,WEIGHT_COL);
    comm->setWsize(WEIGHT_ROW,WEIGHT_COL);
    comm1->setWsize(WEIGHT_ROW,WEIGHT_COL);
    tensorSet *pytool=new tensorSet(2);
    //pytool->in_degree=torch::from_blob(graph->in_degree+graph->partition_offset[graph->partition_id],{embedding->rownum,1});
    //pytool->out_degree=torch::from_blob(graph->in_degree+graph->partition_offset[graph->partition_id],{embedding->rownum,1});
/*init GPU*/
    aggregate_engine  *ag_e=new aggregate_engine();
    ag_e->reconfig_data(graph->owned_vertices,SIZE_LAYER_2,graph->owned_vertices,SIZE_LAYER_1,TENSOR_TYPE);
    ag_e->init_intermediate_gradient();
    
    graph_engine *gr_e=new graph_engine();
    // graph->generate_COO(active);
    // std::cout<<graph->partition_offset[0]<<" "<<graph->partition_offset[1]<<" "<<graph->partition_offset[2]<<std::endl;
    // graph->reorder_COO();

    std::cout<<graph->owned_vertices<<" "<<graph->edges<<std::endl;

    VertexId* incoming_adj_index=new VertexId[graph->vertices+1];
    VertexId* incoming_adj_index_backward=new VertexId[graph->vertices+1];
    for(VertexId vtx=0;vtx<graph->vertices;vtx++){
        if(graph->incoming_adj_index[0][vtx+1]==0)
        graph->incoming_adj_index[0][vtx+1]=graph->incoming_adj_index[0][vtx];
        if(graph->incoming_adj_index_backward[0][vtx+1]==0)
        graph->incoming_adj_index_backward[0][vtx+1]=graph->incoming_adj_index_backward[0][vtx];
         }

    EdgeId edges_for=graph->incoming_adj_index[0][graph->vertices];
    EdgeId edges_back=graph->incoming_adj_index_backward[0][graph->vertices];
     ValueType* weight=new ValueType[edges_for+1];
     ValueType* weight_backward=new ValueType[edges_back+1];

    for(VertexId vtx=0;vtx<graph->vertices;vtx++){
        incoming_adj_index[vtx]=(VertexId)graph->incoming_adj_index[0][vtx];
        incoming_adj_index_backward[vtx]=(VertexId)graph->incoming_adj_index_backward[0][vtx];
           for(int i=graph->incoming_adj_index[0][vtx];i<graph->incoming_adj_index[0][vtx+1];i++){
              VertexId dst=graph->incoming_adj_list[0][i].neighbour;
              weight[i]=(ValueType)std::sqrt(graph->in_degree_for_backward[vtx])*(ValueType)std::sqrt(graph->out_degree_for_backward[dst]);
          }
          for(int i=graph->incoming_adj_index_backward[0][vtx];i<graph->incoming_adj_index_backward[0][vtx+1];i++){
              VertexId dst=graph->incoming_adj_list_backward[0][i].neighbour;
              weight_backward[i]=(ValueType)std::sqrt(graph->out_degree_for_backward[vtx])*(ValueType)std::sqrt(graph->in_degree_for_backward[dst]);
          }        
    }

    incoming_adj_index[graph->vertices]=(VertexId)graph->incoming_adj_index[0][graph->vertices];
    incoming_adj_index_backward[graph->vertices]=(VertexId)graph->incoming_adj_index_backward[0][graph->vertices];   
    std::cout<<"something error"<<incoming_adj_index[graph->vertices-1]<<" "<<(VertexId)graph->incoming_adj_index[0][graph->vertices-1]<<std::endl;

     std::cout<<"edge_for: "<<edges_for<<"edge_back: "<<edges_back<<" "<<graph->partition_id<<std::endl;
    gr_e->load_graph(graph->vertices,edges_for,false,
        graph->vertices,edges_back,false,
            incoming_adj_index,(VertexId*)graph->incoming_adj_list[0],
                incoming_adj_index_backward,(VertexId*)graph->incoming_adj_list_backward[0],
                    graph->partition_offset[graph->partition_id],graph->partition_offset[graph->partition_id+1],0,graph->vertices,SIZE_LAYER_1);
/*1 INIT STAGE*/    
    GnnUnit* Gnn_v1 = new GnnUnit(SIZE_LAYER_1, SIZE_LAYER_2);    
    GnnUnit* Gnn_v2 =new GnnUnit(SIZE_LAYER_2, OUTPUT_LAYER_3);
          std::cout<<Gnn_v1->W.size(0)<<"   "<<Gnn_v1->W.size(1)<<" "<<graph->partition_id<<std::endl;
    pytool->registOptimizer(torch::optim::SGD(Gnn_v1->parameters(), 0.05));//new
    pytool->registOptimizer(torch::optim::SGD(Gnn_v2->parameters(), 0.05));//new
    pytool->registLabel<long>(embedding->label,graph->partition_offset[graph->partition_id],
                graph->partition_offset[graph->partition_id+1]-graph->partition_offset[graph->partition_id]);//new
/*init W with new */

    init_parameter(comm, graph,Gnn_v1, embedding);
    init_parameter(comm1, graph,Gnn_v2, embedding);
  
    std::vector<int> layer_size(0);
    layer_size.push_back(SIZE_LAYER_1);
    layer_size.push_back(SIZE_LAYER_2);
    layer_size.push_back(OUTPUT_LAYER_3);
    GTensor<ValueType,long,VECTOR_LENGTH> *gt=new GTensor<ValueType, long,VECTOR_LENGTH>(graph,embedding,active,2,layer_size); 
    
    torch::Tensor new_combine_grad=torch::zeros({SIZE_LAYER_1,SIZE_LAYER_2},torch::kFloat).cuda();
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


     /*GPU  */
       torch::Device GPU(torch::kCUDA,0);
       torch::Device CPU(torch::kCPU,0); 
       torch::Tensor target_gpu  = pytool->target.cuda();
       torch::Tensor inter1_gpu=torch::zeros({embedding->rownum,SIZE_LAYER_2},at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));      
       Gnn_v2->to(GPU);
       Gnn_v1->to(GPU);
 
       torch::Tensor X0_gpu=torch::from_blob(embedding->start_v+embedding->start,{embedding->rownum,SIZE_LAYER_1},torch::kFloat).cuda();
       torch::Tensor Y0_gpu=torch::zeros({embedding->rownum,SIZE_LAYER_1},at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
       torch::Tensor Y1_gpu=torch::zeros({embedding->rownum,SIZE_LAYER_2},at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
       torch::Tensor Y1_inv_gpu=torch::zeros({embedding->rownum,SIZE_LAYER_2},at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
       torch::Tensor W_for_gpu=torch::from_blob(weight,{edges_for,1},torch::kFloat).cuda();
       torch::Tensor W_back_gpu=torch::from_blob(weight_backward,{edges_back+1,1},torch::kFloat).cuda();
       torch::Tensor Out0_gpu;
       torch::Tensor Out1_gpu;
       torch::Tensor Y0_gpu_buffered=torch::zeros({graph->vertices,SIZE_LAYER_1},
                            at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
       torch::Tensor Y1_gpu_buffered=torch::zeros({graph->vertices,SIZE_LAYER_2},
                            at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));                    
       torch::Tensor Y1_inv_gpu_buffered=torch::zeros({graph->vertices,SIZE_LAYER_1},
                            at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
       
     double exec_time = 0;
    exec_time -= get_time();  

    for (int i_i = 0; i_i < iterations; i_i++) {
        if(i_i!=0){
            Gnn_v1->zero_grad();
            Gnn_v2->zero_grad();
        }
              
         if(graph->partition_id==0)
        std::cout<<"start  ["<<i_i<<"]  epoch+++++++++++++++++++++++++++++++++++++++++++++"<<std::endl;
//layer 1;        
        gr_e->forward_one_step(X0_gpu.packed_accessor<float,2>().data(),
            Y0_gpu_buffered.packed_accessor<float,2>().data(),
                W_for_gpu.packed_accessor<float,2>().data(), SCALA_TYPE, SIZE_LAYER_1);
                
        torch::Tensor comm_data=Y0_gpu_buffered.cpu();
        torch::Tensor Y0_cpu=Y0_gpu.cpu();
        gt->Sync_data<SIZE_LAYER_1>(comm_data,Y0_cpu);
        Y0_gpu.set_data(Y0_cpu.cuda());
    
        inter1_gpu.set_data(Gnn_v1->forward(Y0_gpu));//torch::from_blob(embedding->Gnn_v1->forward(pytool->x[0]).accessor<float,2>().data(),{embedding->rownum,VECTOR_LENGTH});
        Out0_gpu=torch::relu(inter1_gpu);//new

//layer 2;        

        gr_e->forward_one_step(Out0_gpu.packed_accessor<float,2>().data(),
            Y1_gpu_buffered.packed_accessor<float,2>().data(),
                W_for_gpu.packed_accessor<float,2>().data(), SCALA_TYPE, SIZE_LAYER_2);  
       

        torch::Tensor comm_data1=Y1_gpu_buffered.cpu();
        torch::Tensor Y1_cpu=Y1_gpu.cpu();
        gt->Sync_data<SIZE_LAYER_2>(comm_data1,Y1_cpu);
        Y1_gpu.set_data(Y1_cpu.cuda());
        Out1_gpu = Gnn_v2->forward(Y1_gpu);
//output;        
        torch::Tensor tt=Out1_gpu.log_softmax(1);//CUDA
        pytool->loss =torch::nll_loss(tt,target_gpu);//new
        pytool->loss.backward(); 
      //inv layer 2;        
       torch::Tensor aggregate_grad2= unified_parameter<ValueType>(comm1,Gnn_v2->W.grad().cpu()); 
        Gnn_v2->learn_gpu(aggregate_grad2.cuda(),learn_rate);//reset from new   
        //std::cout<<Gnn_v2->W.grad().cpu()[0]<<std::endl;
//        Gnn_v2->learn_gpu(Gnn_v2->W.grad(),learn_rate);//signle node
    // s1=Gnn_v1->W.cpu();
    // s2=Gnn_v2->W.cpu();
    
    // std::cout<<"valdate middle"<<s1.accessor<ValueType,2>().data()[0]<<" "<<s2.accessor<ValueType,2>().data()[0]
    // <<" "<<aggregate_grad2.accessor<ValueType,2>().data()[0]<<" "<<s3.accessor<ValueType,2>().data()[0]<<std::endl;

         
//inv layer 1;
    // 2->1
    std::cout<<"here on"<<Y1_gpu.grad().size(0)<<Y1_gpu.grad().size(1)<<std::endl;
        gr_e->backward_one_step(Y1_gpu.grad().packed_accessor<float,2>().data(),
            Y1_inv_gpu_buffered.packed_accessor<float,2>().data(),
                W_back_gpu.packed_accessor<float,2>().data(), SCALA_TYPE, SIZE_LAYER_2); 


        torch::Tensor comm_data2=Y1_inv_gpu_buffered.cpu();
        torch::Tensor Y1_inv_cpu=Y1_inv_gpu.cpu();
        gt->Sync_data<SIZE_LAYER_2>(comm_data2,Y1_inv_cpu);
        Y1_inv_gpu.set_data(Y1_inv_cpu.cuda());

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
     torch::Tensor aggregate_grad = unified_parameter<ValueType>(comm,new_combine_grad.cpu());
     Gnn_v1->learn_gpu(aggregate_grad.cuda(),learn_rate); 

     if(graph->partition_id==0) 
         std::cout<<"LOSS:\t"<<pytool->loss<<std::endl;    
     if(i_i==(iterations-1)){//&&graph->partition_id==0
         torch::Tensor tt_cpu=tt.cpu();
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
            std::cout<<"\ncorrect number on training:"<<correct<<"\t"<<((ValueType)correct/(ValueType)graph->owned_vertices)<<"on id"<<graph->partition_id<<std::endl;
        }
        std::cout<<"loss at"<<graph->partition_id<<"is :"<<pytool->loss<<std::endl;
        int correct_test=0;
            for(int k=graph->vertices;k<NODE_NUMBER;k++){
                ValueType max= -100000.0;
                long id=-1;
                torch::Tensor test=torch::from_blob(&(embedding->con[k].att[0]),{1,SIZE_LAYER_1});
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
            torch::Tensor ses=Gnn_v1->W.cpu();
            torch::Tensor ses1=Gnn_v2->W.cpu();
            //std::cout<<ses.accessor<ValueType,2>().data()[0]<<std::endl;
            std::cout<<"valdate"<<ses.accessor<ValueType,2>().data()[0]<<" "<<ses1.accessor<ValueType,2>().data()[0]<<"correct number on testing:"<<correct_test<<"\t"<<((ValueType)correct_test/(ValueType)(NODE_NUMBER-graph->vertices))<<"on id"<<graph->partition_id<<std::endl;       
     }
    }
    delete active;     
}
void compute_dist_gf_GPU(Graph<Empty> * graph, int iterations) {
    ValueType learn_rate=0.01;
    VertexSubset * active = graph->alloc_vertex_subset();
    active->fill();
    Embeddings<ValueType, long> *embedding = new Embeddings<ValueType, long>();
    embedding->init(graph);
    embedding->readlabel_(graph);
    
    Network<ValueType> *comm=new Network<ValueType>(graph,WEIGHT_ROW,WEIGHT_COL);
    Network<ValueType>* comm1=new Network<ValueType>(graph,WEIGHT_ROW,WEIGHT_COL);
    comm->setWsize(WEIGHT_ROW,WEIGHT_COL);
    comm1->setWsize(WEIGHT_ROW,WEIGHT_COL);
    tensorSet *pytool=new tensorSet(2);
    //pytool->in_degree=torch::from_blob(graph->in_degree+graph->partition_offset[graph->partition_id],{embedding->rownum,1});
    //pytool->out_degree=torch::from_blob(graph->in_degree+graph->partition_offset[graph->partition_id],{embedding->rownum,1});
/*init GPU*/
    aggregate_engine  *ag_e=new aggregate_engine();
    ag_e->reconfig_data(graph->owned_vertices,SIZE_LAYER_2,graph->owned_vertices,SIZE_LAYER_1,TENSOR_TYPE);
    ag_e->init_intermediate_gradient();
    
    graph_engine *gr_e=new graph_engine();
    // graph->generate_COO(active);
    // std::cout<<graph->partition_offset[0]<<" "<<graph->partition_offset[1]<<" "<<graph->partition_offset[2]<<std::endl;
    // graph->reorder_COO();

    // for(int i=graph->edges-100;i<graph->edges;i++){
    //     std::cout<<graph->_graph_cpu->srcList[i]<<" "<<graph->_graph_cpu->dstList[i]<<std::endl;
    // }
    std::cout<<graph->owned_vertices<<" "<<graph->edges<<std::endl;

    VertexId* incoming_adj_index=new VertexId[graph->vertices+1];
    VertexId* incoming_adj_index_backward=new VertexId[graph->vertices+1];
      for(VertexId vtx=0;vtx<graph->vertices;vtx++){
        if(graph->incoming_adj_index[0][vtx+1]==0)
        graph->incoming_adj_index[0][vtx+1]=graph->incoming_adj_index[0][vtx];
        if(graph->incoming_adj_index_backward[0][vtx+1]==0)
        graph->incoming_adj_index_backward[0][vtx+1]=graph->incoming_adj_index_backward[0][vtx];
         }
    EdgeId edges_for=graph->incoming_adj_index[0][graph->vertices];
    EdgeId edges_back=graph->incoming_adj_index_backward[0][graph->vertices];
     ValueType* weight=new ValueType[edges_for+1];
     ValueType* weight_backward=new ValueType[edges_back+1];

    for(VertexId vtx=0;vtx<graph->vertices+1;vtx++){
        incoming_adj_index[vtx]=(VertexId)graph->incoming_adj_index[0][vtx];
        incoming_adj_index_backward[vtx]=(VertexId)graph->incoming_adj_index_backward[0][vtx];
           for(int i=graph->incoming_adj_index[0][vtx];i<graph->incoming_adj_index[0][vtx+1];i++){
              VertexId dst=graph->incoming_adj_list[0][i].neighbour;
              weight[i]=(ValueType)std::sqrt(graph->in_degree_for_backward[vtx])*(ValueType)std::sqrt(graph->out_degree_for_backward[dst]);
          }
          for(int i=graph->incoming_adj_index_backward[0][vtx];i<graph->incoming_adj_index_backward[0][vtx+1];i++){
              VertexId dst=graph->incoming_adj_list_backward[0][i].neighbour;
              weight_backward[i]=(ValueType)std::sqrt(graph->out_degree_for_backward[vtx])*(ValueType)std::sqrt(graph->in_degree_for_backward[dst]);
          }        

    }
    incoming_adj_index[graph->vertices]=(VertexId)graph->incoming_adj_index[0][graph->vertices];
    incoming_adj_index_backward[graph->vertices]=(VertexId)graph->incoming_adj_index_backward[0][graph->vertices];   
    std::cout<<"something error"<<incoming_adj_index[graph->vertices-1]<<" "<<(VertexId)graph->incoming_adj_index[0][graph->vertices-1]<<std::endl;

    gr_e->load_graph(graph->vertices,edges_for,false,
        graph->vertices,edges_back,false,
            incoming_adj_index,(VertexId*)graph->incoming_adj_list[0],
                incoming_adj_index_backward,(VertexId*)graph->incoming_adj_list_backward[0],
                    graph->partition_offset[graph->partition_id],graph->partition_offset[graph->partition_id+1],0,graph->vertices,SIZE_LAYER_1);
    
/*1 INIT STAGE*/    
  
    GnnUnit* Gnn_v1 = new GnnUnit(SIZE_LAYER_1, SIZE_LAYER_2);
    
    GnnUnit* Gnn_v2 =new GnnUnit(SIZE_LAYER_2, OUTPUT_LAYER_3);
    pytool->registOptimizer(torch::optim::SGD(Gnn_v1->parameters(), 0.05));//new
    pytool->registOptimizer(torch::optim::SGD(Gnn_v2->parameters(), 0.05));//new

    pytool->registLabel<long>(embedding->label,graph->partition_offset[graph->partition_id],
                graph->partition_offset[graph->partition_id+1]-graph->partition_offset[graph->partition_id]);//new
/*init W with new */
  
    init_parameter(comm, graph,Gnn_v1, embedding);
    init_parameter(comm1, graph,Gnn_v2, embedding);

    std::vector<int> layer_size(0);
    layer_size.push_back(SIZE_LAYER_1);
    layer_size.push_back(SIZE_LAYER_2);
    layer_size.push_back(OUTPUT_LAYER_3);
    GTensor<ValueType,long,VECTOR_LENGTH> *gt=new GTensor<ValueType, long,VECTOR_LENGTH>(graph,embedding,active,2,layer_size); 
    
    torch::Tensor new_combine_grad=torch::zeros({SIZE_LAYER_1,SIZE_LAYER_2},torch::kFloat).cuda();

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
     /*GPU  */
       torch::Device GPU(torch::kCUDA,0);
       torch::Device CPU(torch::kCPU,0); 
       torch::Tensor target_gpu  = pytool->target.cuda();    
       Gnn_v2->to(GPU);
       Gnn_v1->to(GPU);
       
     //  torch::Tensor X1_gpu=torch::zeros({embedding->rownum,SIZE_LAYER_1},at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));      
       
       torch::Tensor X0_gpu=torch::from_blob(embedding->start_v+embedding->start,{embedding->rownum,SIZE_LAYER_1},torch::kFloat).cuda();
       torch::Tensor X0_gpu_trans=torch::zeros({embedding->rownum,SIZE_LAYER_2},torch::kFloat).cuda();
       torch::Tensor Y0_gpu=torch::zeros({embedding->rownum,SIZE_LAYER_2},at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
       torch::Tensor Y1_gpu=torch::zeros({embedding->rownum,SIZE_LAYER_2},at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
       torch::Tensor Y1_inv_gpu=torch::zeros({embedding->rownum,SIZE_LAYER_2},at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
       torch::Tensor Y0_inv_gpu=torch::zeros({embedding->rownum,SIZE_LAYER_2},at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
       torch::Tensor W_for_gpu=torch::from_blob(weight,{edges_for,1},torch::kFloat).cuda();
       torch::Tensor W_back_gpu=torch::from_blob(weight_backward,{edges_back+1,1},torch::kFloat).cuda();
       torch::Tensor Out0_gpu;
       torch::Tensor Out1_gpu;
       torch::Tensor Y0_gpu_buffered=torch::zeros({graph->vertices,SIZE_LAYER_2},
                            at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
       torch::Tensor Y1_gpu_buffered=torch::zeros({graph->vertices,SIZE_LAYER_2},
                            at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));                    
       torch::Tensor Y1_inv_gpu_buffered=torch::zeros({graph->vertices,SIZE_LAYER_2},
                            at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
       torch::Tensor Y0_inv_gpu_buffered=torch::zeros({graph->vertices,SIZE_LAYER_2},
                            at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
       
     double exec_time = 0;
    exec_time -= get_time();  


    for (int i_i = 0; i_i < iterations; i_i++) {
        if(i_i!=0){
           //inter1_gpu.grad().zero_();
           //inter2_gpu.grad().zero_(); 
            Gnn_v1->zero_grad();
            Gnn_v2->zero_grad();
            //Y1_gpu.grad().zero_();
        }
              
         if(graph->partition_id==0)
        std::cout<<"start  ["<<i_i<<"]  epoch+++++++++++++++++++++++++++++++++++++++++++++"<<std::endl;
//layer 1;         



        X0_gpu_trans=Gnn_v1->forward(X0_gpu);
        gr_e->forward_one_step(X0_gpu_trans.packed_accessor<float,2>().data(),
            Y0_gpu_buffered.packed_accessor<float,2>().data(),
                W_for_gpu.packed_accessor<float,2>().data(), SCALA_TYPE, SIZE_LAYER_2);
                
        torch::Tensor comm_data=Y0_gpu_buffered.cpu();
        torch::Tensor Y0_cpu=Y0_gpu.cpu();
        gt->Sync_data<SIZE_LAYER_2>(comm_data,Y0_cpu);
        Y0_gpu.set_data(Y0_cpu.cuda());

        Out0_gpu=torch::relu(Y0_gpu);//new
//layer 2;         
        gr_e->forward_one_step(Out0_gpu.packed_accessor<float,2>().data(),
            Y1_gpu_buffered.packed_accessor<float,2>().data(),
                W_for_gpu.packed_accessor<float,2>().data(), SCALA_TYPE, SIZE_LAYER_2);  
       
        torch::Tensor comm_data1=Y1_gpu_buffered.cpu();
        torch::Tensor Y1_cpu=Y1_gpu.cpu();
        gt->Sync_data<SIZE_LAYER_2>(comm_data1,Y1_cpu);
        Y1_gpu.set_data(Y1_cpu.cuda());
        Out1_gpu = Gnn_v2->forward(Y1_gpu);
//output;        
        torch::Tensor tt=Out1_gpu.log_softmax(1);//CUDA
        pytool->loss =torch::nll_loss(tt,target_gpu);//new
        pytool->loss.backward(); 
      //inv layer 2;        
       torch::Tensor aggregate_grad2= unified_parameter<ValueType>(comm1,Gnn_v2->W.grad().cpu()); 
        Gnn_v2->learn_gpu(aggregate_grad2.cuda(),learn_rate);//reset from new   
        std::cout<<"error"<<std::endl;
//inv layer 1;
    // 2->1
        gr_e->backward_one_step(Y1_gpu.grad().packed_accessor<float,2>().data(),
            Y1_inv_gpu_buffered.packed_accessor<float,2>().data(),
                W_back_gpu.packed_accessor<float,2>().data(), SCALA_TYPE, SIZE_LAYER_2); 
        torch::Tensor comm_data2=Y1_inv_gpu_buffered.cpu();
        torch::Tensor Y1_inv_cpu=Y1_inv_gpu.cpu();
        gt->Sync_data<SIZE_LAYER_2>(comm_data2,Y1_inv_cpu);
        Y1_inv_gpu.set_data(Y1_inv_cpu.cuda());
   
    // layer 1 local     
        Out0_gpu.backward();//new  
        torch::Tensor tmp=Y1_inv_gpu*Y0_gpu.grad();
        gr_e->backward_one_step(tmp.packed_accessor<float,2>().data(),
            Y0_inv_gpu_buffered.packed_accessor<float,2>().data(),
                W_back_gpu.packed_accessor<float,2>().data(), SCALA_TYPE, SIZE_LAYER_2); 
        torch::Tensor comm_data3=Y0_inv_gpu_buffered.cpu();
        torch::Tensor Y0_inv_cpu=Y0_inv_gpu.cpu();
        gt->Sync_data<SIZE_LAYER_2>(comm_data3,Y0_inv_cpu);
        Y0_inv_gpu.set_data(Y0_inv_cpu.cuda());

    // layer 1 combine
        new_combine_grad.zero_();
        new_combine_grad=X0_gpu.t().mm(Y0_inv_gpu); 


//learn  
     torch::Tensor aggregate_grad = unified_parameter<ValueType>(comm,new_combine_grad.cpu());
     Gnn_v1->learn_gpu(aggregate_grad.cuda(),learn_rate); 
   
       
 
     
  
     
     if(graph->partition_id==0) 
         std::cout<<"LOSS:\t"<<pytool->loss<<std::endl;    
     if(i_i==(iterations-1)){//&&graph->partition_id==0
         torch::Tensor tt_cpu=tt.cpu();
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
            std::cout<<"\ncorrect number on training:"<<correct<<"\t"<<((ValueType)correct/(ValueType)graph->owned_vertices)<<"on id"<<graph->partition_id<<std::endl;
        }
        std::cout<<"loss at"<<graph->partition_id<<"is :"<<pytool->loss<<std::endl;
        int correct_test=0;
            for(int k=graph->vertices;k<NODE_NUMBER;k++){
                ValueType max= -100000.0;
                long id=-1;
                torch::Tensor test=torch::from_blob(&(embedding->con[k].att[0]),{1,SIZE_LAYER_1});
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
            torch::Tensor ses=Gnn_v1->W.cpu();
            torch::Tensor ses1=Gnn_v2->W.cpu();
            //std::cout<<ses.accessor<ValueType,2>().data()[0]<<std::endl;
            std::cout<<"valdate"<<ses.accessor<ValueType,2>().data()[0]<<" "<<ses1.accessor<ValueType,2>().data()[0]<<"correct number on testing:"<<correct_test<<"\t"<<((ValueType)correct_test/(ValueType)(NODE_NUMBER-graph->vertices))<<"on id"<<graph->partition_id<<std::endl;       
     }
    }
//     delete active;     
}
void statistic(Graph<Empty> *graph,int workers){
int worker_num=workers;
int *offset=new int[worker_num+1];
long *first_layer=new long[worker_num+1];
int *first_layer_unique=new int[worker_num+1];
long *second_layer=new long[worker_num+1];
int *second_layer_unique=new int[worker_num+1];
char *bitmap_second=new char[graph->vertices+1];
char *bitmap_first=new char[graph->vertices+1];

long first_layer_vertices=0,first_layer_vertice_dump=0;
long long first_layer_edges=0,first_layer_edges_dump=0;
long second_layer_vertices=0,second_layer_vertice_dump=0;
long long second_layer_edges=0,second_layer_edges_dump=0;


memset(offset,0,sizeof(int)*(worker_num+1));
memset(first_layer,0,sizeof(long)*(worker_num+1));
memset(first_layer_unique,0,sizeof(int)*(worker_num+1));
memset(second_layer,0,sizeof(long)*(worker_num+1));
memset(second_layer_unique,0,sizeof(int)*(worker_num+1));
int worker_vertices=graph->vertices/worker_num;
for(int i=0;i<worker_num;i++){
offset[i+1]=offset[i]+worker_vertices;
}

double cnt=0;
int id=0;
for(int i=0;i<graph->vertices;i++){
        cnt=graph->in_degree[i]+cnt;
    if(cnt>(graph->edges/worker_num)*(id+1)){
        offset[id+1]=i;
        id++;
    }
}


std::cout<<std::endl<<"++++++++++status++++++++++"<<std::endl;
std::cout<<std::endl<<"worker_num: "<<worker_num<<"++++"<<std::endl;
std::cout<<"graph->vertices: "<<graph->vertices<<std::endl;
std::cout<<"graph->edges   : "<<graph->edges<<std::endl;
std::cout<<"partition: ";
offset[worker_num]=graph->vertices;
for(int i=0;i<worker_num;i++){
    std::cout<<offset[i+1]-offset[i]<<" ";
}std::cout<<std::endl;

for(int i=0;i<worker_num;i++){
    memset(bitmap_first,0,sizeof(char)*(graph->vertices+1));
    memset(bitmap_second,0,sizeof(char)*(graph->vertices+1));
    for(int j=offset[i];j<offset[i+1];j++){
        first_layer[i]+=graph->in_degree[j];
        for (int k=graph->incoming_adj_index[0][j];k<graph->incoming_adj_index[0][j+1];k++){
            int neighbour=graph->incoming_adj_list[0][k].neighbour;
            bitmap_first[neighbour]=1;
            for(int l=graph->incoming_adj_index[0][neighbour];l<graph->incoming_adj_index[0][neighbour+1];l++){
                    bitmap_second[graph->incoming_adj_list[0][l].neighbour]=1;
            }
            //second_layer[i]+=graph->in_degree[neighbour];
        }
    }
    for(int l=0;l<graph->vertices;l++){
        if(bitmap_second[l]){
            second_layer_unique[i]+=1;
        }
        if(bitmap_first[l]){
            first_layer_unique[i]+=bitmap_first[l];
            second_layer[i]+=graph->in_degree[l];
        }
    }
}
std::cout<<"first_layer:\t";
for(int i=0;i<worker_num;i++){
    std::cout<<first_layer[i]<<" ";
    first_layer_edges+=first_layer[i];
}std::cout<<std::endl;
std::cout<<"first_layer_unique:\t";
for(int i=0;i<worker_num;i++){
    std::cout<<first_layer_unique[i]<<" ";
    first_layer_vertices+=first_layer_unique[i];
}std::cout<<std::endl;
std::cout<<"second_layer:\t";
for(int i=0;i<worker_num;i++){
    std::cout<<second_layer[i]<<" ";
    second_layer_edges+=second_layer[i];
}std::cout<<std::endl;
std::cout<<"second_layer_unique:\t";
for(int i=0;i<worker_num;i++){
    std::cout<<second_layer_unique[i]<<" ";
    second_layer_vertices+=second_layer_unique[i];
}std::cout<<std::endl;
std::cout<<" first layer edges dump:\t"<<first_layer_edges<<" ratio: "<<(long double)first_layer_edges/graph->edges<<std::endl;
std::cout<<"second layer edges dump:\t"<<second_layer_edges<<" ratio: "<<(long double)second_layer_edges/graph->edges<<std::endl;
std::cout<<" first layer feature dump:\t"<<first_layer_vertices<<" ratio: "<<(long double)first_layer_vertices/graph->vertices<<std::endl;
std::cout<<"second layer feature dump:\t"<<second_layer_vertices<<" ratio: "<<(long double)second_layer_vertices/graph->vertices<<std::endl;




std::cout<<std::endl<<"++++++++++status++++++++++"<<std::endl;

}
void compute_GPU_shard(Graph<Empty> * graph, int iterations) {
    int batch_size=2048;
    int batch_num=graph->owned_vertices/batch_size+(graph->owned_vertices>1);
    ValueType learn_rate=0.01;
    //gpu_processor *gp=new gpu_processor(); //GPU
    Embeddings<ValueType, long> *embedding = new Embeddings<ValueType, long>();
    embedding->init(graph);
    embedding->readlabel_(graph);
    
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
    VertexSubset * active = graph->alloc_vertex_subset();
    active->fill();
    std::vector<int> layer_size(0);
    layer_size.push_back(SIZE_LAYER_1);
    layer_size.push_back(SIZE_LAYER_2);
    layer_size.push_back(OUTPUT_LAYER_3);
    GTensor<ValueType,long,VECTOR_LENGTH> *gt=new GTensor<ValueType, long,VECTOR_LENGTH>(graph,embedding,active,2,layer_size); 
    //gpu_processor *gp1=new gpu_processor(SIZE_LAYER_1,SIZE_LAYER_2,2208);
    
    torch::Tensor new_combine_grad=torch::zeros({SIZE_LAYER_1,SIZE_LAYER_2},torch::kFloat).cuda();
    std::vector<torch::Tensor> partial_new_combine_grad(0);
    for(int i=0;i<graph->threads;i++){
    partial_new_combine_grad.push_back(new_combine_grad);
}
     aggregate_engine  *ag_e=new aggregate_engine();
    ag_e->reconfig_data(graph->vertices,SIZE_LAYER_2,graph->vertices,SIZE_LAYER_1,TENSOR_TYPE);
    ag_e->init_intermediate_gradient();
    
    
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
       //torch::Tensor inter2_gpu=torch::zeros({embedding->rownum,SIZE_LAYER_2},at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
       Gnn_v2->to(GPU);
       Gnn_v1->to(GPU);
       
       torch::Tensor current_batch_1=torch::zeros({batch_size,SIZE_LAYER_2},at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
       torch::Tensor last_batch_1;
       torch::Tensor current_batch=torch::zeros({batch_size,SIZE_LAYER_2},at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
       torch::Tensor last_batch;
       torch::Tensor tmp_batch;
       torch::Tensor y0=torch::zeros({embedding->rownum,SIZE_LAYER_2},at::TensorOptions().dtype(torch::kFloat));
       torch::Tensor batch_grad=torch::zeros({embedding->rownum,SIZE_LAYER_2},at::TensorOptions().dtype(torch::kFloat));
       torch::Tensor local_grad0=torch::zeros({embedding->rownum,SIZE_LAYER_2},at::TensorOptions().dtype(torch::kFloat));
       for(int i=0;i<batch_num;i++){
            int start=i*batch_size;
            int end=std::min((i+1)*batch_size,(int)graph->owned_vertices);
            if (i==(batch_num-1)){
                last_batch_1=torch::zeros({end-start,SIZE_LAYER_2},
                     at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
                last_batch=torch::zeros({end-start,SIZE_LAYER_2},
                     at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
            }
       }
       
       
       
     double exec_time = 0;
    exec_time -= get_time();  
    for (int i_i = 0; i_i < iterations; i_i++) {
        if(i_i!=0){
           //inter1_gpu.grad().zero_();
           //inter2_gpu.grad().zero_(); 
            //current_batch.grad().zero_();
            Gnn_v1->zero_grad();
            Gnn_v2->zero_grad();
        }

         gt->setValueFromNative(embedding->start_v,embedding->start);  
            
         if(graph->partition_id==0)
        std::cout<<"start  ["<<i_i<<"]  epoch+++++++++++++++++++++++++++++++++++++++++++++"<<std::endl;
//layer 1;
        gt->Propagate<SIZE_LAYER_1>(0);
        pytool->updateX(0,gt->value_local[0]); 
        
        //first layer
        y0.zero_();
        torch::Tensor y0_tmp;
        for(int i=0;i<batch_num;i++){//second batch
            int start=i*batch_size;
            int end=std::min((i+1)*batch_size,(int)graph->owned_vertices);          
            if(i!=batch_num-1){
               current_batch_1.set_data(Gnn_v1->forward(pytool->x[0].slice(0,start,end,1).cuda()));
               y0_tmp=torch::relu(current_batch_1);//new
               y0_tmp.backward();//new
               torch::Tensor tmp=y0.slice(0,start,end,1);
               tmp+=y0_tmp.cpu();
               torch::Tensor tmp1=local_grad0.slice(0,start,end,1);
               tmp1+=current_batch_1.grad().cpu();
               current_batch_1.grad().zero_();
            }
            else{
               last_batch_1.set_data(Gnn_v1->forward(pytool->x[0].slice(0,start,end,1).cuda()));
               y0_tmp=torch::relu(last_batch_1);//new
               y0_tmp.backward();//new
               torch::Tensor tmp=y0.slice(0,start,end,1);
               tmp+=y0_tmp.cpu();
               torch::Tensor tmp1=local_grad0.slice(0,start,end,1);
               tmp1+=last_batch_1.grad().cpu();
               last_batch_1.grad().zero_();
            }
   
        }
        
        inter1_gpu.set_data(Gnn_v1->forward(pytool->x[0].cuda()));//torch::from_blob(embedding->Gnn_v1->forward(pytool->x[0]).accessor<float,2>().data(),{embedding->rownum,VECTOR_LENGTH});
        pytool->y[0]=torch::relu(inter1_gpu);//new
        pytool->y[0].backward();//new
//        pytool->localGrad[0]=inter1_gpu.grad();//new
            pytool->localGrad[0]=local_grad0.cuda();
        
        
            gt->setValueFromTensor(y0.cpu());
        //gt->setValueFromTensor(pytool->y[0].cpu());                  
        gt->Propagate<SIZE_LAYER_2>(1);             
        pytool->updateX(1,gt->value_local[1]);//new
        
        torch::Tensor tt;
        torch::Tensor y1;
        std::cout<<batch_num<<std::endl;
        for(int i=0;i<batch_num;i++){//second batch
//            current_batch.grad().zero_();
            int start=i*batch_size;
            int end=std::min((i+1)*batch_size,(int)graph->owned_vertices);
//                current_batch.zero_();             
            if(i!=batch_num-1){
//                current_batch=torch::zeros({end-start,SIZE_LAYER_2},
//                    at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
                current_batch.set_data(pytool->x[1].slice(0,start,end,1).cuda());
                            //current_batchs_gpu[i]=current_batchs[i].cuda();
                y1 = Gnn_v2->forward(current_batch);
                tt=y1.log_softmax(1);//CUDA
                pytool->loss =torch::nll_loss(tt,target_gpu.slice(0,start,end,1));//new
                pytool->loss.backward();

                torch::Tensor tmp= batch_grad.slice(0,start,end,1);
                tmp+=current_batch.grad().cpu();
                current_batch.grad().zero_();
                //std::cout<<start<<" "<<end<<std::endl;
            }
            else{
                last_batch.set_data(pytool->x[1].slice(0,start,end,1).cuda());
                            //current_batchs_gpu[i]=current_batchs[i].cuda();
                y1 = Gnn_v2->forward(last_batch);
                tt=y1.log_softmax(1);//CUDA
                pytool->loss =torch::nll_loss(tt,target_gpu.slice(0,start,end,1));//new
                pytool->loss.backward();

                torch::Tensor tmp= batch_grad.slice(0,start,end,1);
                tmp+=last_batch.grad().cpu();
                last_batch.grad().zero_();
                //std::cout<<start<<" "<<end<<std::endl;
            }
   
        }
        
        
        torch::Tensor aggregate_grad2= unified_parameter<ValueType>(comm1,Gnn_v2->W.grad().cpu()); 
         Gnn_v2->learn_gpu(aggregate_grad2.cuda(),learn_rate);//reset from new        
         
        // std::cout<<inter1->W.grad().size(0)<<"    "<<inter1->W.grad().size(1)<<std::endl;
        gt->setGradFromTensor(batch_grad);
        gt->Propagate_backward<SIZE_LAYER_2>(0);

//compute gradient
    for(int i=0;i<graph->threads;i++){
    partial_new_combine_grad[i].zero_();
} new_combine_grad.zero_();

torch::Tensor x0_gpu=pytool->x[0].cuda();
torch::Tensor grad0_gpu=gt->grad_local[0].cuda();        
        new_combine_grad.zero_();
        ag_e->redirect_input_output(grad0_gpu.packed_accessor<float,2>().data(),
            pytool->localGrad[0].packed_accessor<float,2>().data(),
                x0_gpu.packed_accessor<float,2>().data(),
                    new_combine_grad.packed_accessor<float,2>().data());
        ag_e->aggregate_grad();        

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
            for(int k=0;k<tt.size(0);k++){
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
                torch::Tensor test=torch::from_blob(&(embedding->con[k].att[0]),{1,SIZE_LAYER_1});
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
void compute_single_GPU_shard_graph(Graph<Empty> * graph, int iterations) {
    ValueType learn_rate=0.01;
    const int BATCH_SIZE=15717;
    VertexSubset * active = graph->alloc_vertex_subset();
    active->fill();
    Embeddings<ValueType, long> *embedding = new Embeddings<ValueType, long>();
    embedding->init(graph);
    embedding->readlabel_(graph);

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
    graph->generate_COO(active);
    graph->reorder_COO(BATCH_SIZE);
     std::vector<edge_list*> edge_list;
    generate_edge_list_Tensor(graph,edge_list,BATCH_SIZE);  
    
    VertexId* incoming_adj_index=new VertexId[graph->vertices+1];
    VertexId* incoming_adj_index_backward=new VertexId[graph->vertices+1];
    ValueType* weight=new ValueType[graph->edges+1];
    ValueType*weight_backward=new ValueType[graph->edges+1];  
    torch::Tensor W_for_gpu;
    torch::Tensor W_back_gpu;
    generate_weight_and_csr(graph, active, incoming_adj_index, incoming_adj_index_backward,
        weight, weight_backward,W_for_gpu, W_back_gpu);

/*1 INIT STAGE*/    
    GnnUnit* Gnn_v1 = new GnnUnit(SIZE_LAYER_1, SIZE_LAYER_2);
    GnnUnit* Gnn_v1_1 = new GnnUnit(SIZE_LAYER_1, SIZE_LAYER_2);//commnet
    GnnUnit* Gnn_v2 =new GnnUnit(SIZE_LAYER_2, OUTPUT_LAYER_3);
    GnnUnit* Gnn_v2_1 =new GnnUnit(SIZE_LAYER_2, OUTPUT_LAYER_3);//commnet
    pytool->registOptimizer(torch::optim::SGD(Gnn_v1->parameters(), 0.05));//new
    pytool->registOptimizer(torch::optim::SGD(Gnn_v2->parameters(), 0.05));//new
    pytool->registLabel<long>(embedding->label,graph->partition_offset[graph->partition_id],graph->partition_offset[graph->partition_id+1]-graph->partition_offset[graph->partition_id]);//new
/*init W with new */
    init_parameter(comm, graph,Gnn_v1, embedding);
    init_parameter(comm1, graph,Gnn_v2, embedding);

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
       torch::Tensor target_gpu  = pytool->target.cuda();
       torch::Tensor inter1_gpu=torch::zeros({embedding->rownum,SIZE_LAYER_2},at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));      
       Gnn_v2->to(GPU);
       Gnn_v1->to(GPU);
       
     //  torch::Tensor X1_gpu=torch::zeros({embedding->rownum,SIZE_LAYER_1},at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));      
       torch::Tensor X0_cpu=torch::from_blob(embedding->start_v+embedding->start,{embedding->rownum,SIZE_LAYER_1},torch::kFloat);
       
       torch::Tensor Y0_gpu=torch::zeros({embedding->rownum,SIZE_LAYER_1},at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
       torch::Tensor Y0_cpu=torch::zeros({embedding->rownum,SIZE_LAYER_1},at::TensorOptions().dtype(torch::kFloat));
       torch::Tensor X1_cpu=torch::zeros({embedding->rownum,SIZE_LAYER_2},at::TensorOptions().dtype(torch::kFloat));
       torch::Tensor Y1_gpu=torch::zeros({embedding->rownum,SIZE_LAYER_2},at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
       torch::Tensor Y1_cpu=torch::zeros({embedding->rownum,SIZE_LAYER_2},at::TensorOptions().dtype(torch::kFloat));
       torch::Tensor Y1_inv_gpu=torch::zeros({embedding->rownum,SIZE_LAYER_2},at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
       torch::Tensor Y1_inv_cpu=torch::zeros({embedding->rownum,SIZE_LAYER_2},at::TensorOptions().dtype(torch::kFloat));
       torch::Tensor Out0_gpu;
       torch::Tensor Out1_gpu;
       
       
     double exec_time = 0;
     double graph_time = 0;
    exec_time -= get_time(); 
    double tmp_time=0;
    for (int i_i = 0; i_i < iterations; i_i++) {
        if(i_i!=0){

            Gnn_v1->zero_grad();
            Gnn_v2->zero_grad();
        }
              
         if(graph->partition_id==0)
        std::cout<<"start  ["<<i_i<<"]  epoch+++++++++++++++++++++++++++++++++++++++++++++"<<std::endl;
//layer 1;     
   

        Y0_cpu.zero_();
        propagate_forward_gpu_shard(graph, X0_cpu, Y0_cpu, edge_list,SIZE_LAYER_1);   
        Y0_gpu=Y0_cpu.cuda();
        
        std::cout<<">>>>"<<std::endl;
        inter1_gpu.set_data(Gnn_v1->forward(Y0_gpu));
        Out0_gpu=torch::relu(inter1_gpu);//new

//layer 2; 
        X1_cpu.set_data(Out0_gpu.cpu());
        Y1_cpu.zero_();
        propagate_forward_gpu_shard(graph, X1_cpu, Y1_cpu, edge_list,SIZE_LAYER_2); 
        Y1_gpu.set_data(Y1_cpu.cuda());
          
        
        Out1_gpu = Gnn_v2->forward(Y1_gpu);
//output;        
        torch::Tensor tt=Out1_gpu.log_softmax(1);//CUDA
        pytool->loss =torch::nll_loss(tt,target_gpu);//new
        pytool->loss.backward();
//inv layer 2;        
        Gnn_v2->learn_gpu(Gnn_v2->W.grad(),learn_rate);//signle node     
//inv layer 1;
    // 2->1
        
        Y1_inv_cpu.zero_();
        propagate_forward_gpu_shard(graph, Y1_gpu.grad().cpu(), Y1_inv_cpu, edge_list,SIZE_LAYER_2); 
        Y1_inv_gpu=Y1_inv_cpu.cuda();        
                
  
        Out0_gpu.backward();//new  
        new_combine_grad.zero_();
        /*ag_e->redirect_input_output(Y1_inv_gpu.packed_accessor<float,2>().data(),
            inter1_gpu.grad().packed_accessor<float,2>().data(),
                Y0_gpu.packed_accessor<float,2>().data(),
                    new_combine_grad.packed_accessor<float,2>().data());
        ag_e->aggregate_grad(); */
        new_combine_grad=Y0_gpu.t().mm(Y1_inv_gpu*inter1_gpu.grad());
//learn  
        Gnn_v1->learn_gpu(new_combine_grad,learn_rate); 
         

     if(graph->partition_id==0) 
         std::cout<<"LOSS:\t"<<pytool->loss<<std::endl; 
     if(i_i==(iterations-1)&&graph->partition_id==0){     
         std::cout<<"+++++++++++++++++++++++ finish ++++++++++++++++++++++++"<<std::endl;
         exec_time += get_time();
             if (graph->partition_id==0) {
                 printf("exec_time=%lf(s)  graph_time=%lf(s)\n", exec_time,graph_time);
                 std::cout<<graph->vertices<<" "<<graph->edges<<std::endl;
            }
         inference(tt.cpu(),graph, embedding, pytool,Gnn_v1,Gnn_v2);
     
     }
    }
     delete active;     
}

int main(int argc, char ** argv) {
    MPI_Instance mpi(&argc, &argv);
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
    }else
    if(std::string(argv[4])==std::string("GPU_shard")){
        printf("%s g engine start",argv[4]);
       // compute_GPU_shard(graph, iterations);
        compute_single_GPU_shard_graph(graph, iterations);
    }else if(std::string(argv[4])==std::string("CPU")){
        printf("%s c engine start",argv[4]);
    compute(graph, iterations);
    }else if(std::string(argv[4])==std::string("CPUGF")){
        printf("%s c engine start",argv[4]);
    compute_gf(graph, iterations);
    }else if(std::string(argv[4])==std::string("TEST")){
        printf("%s c engine start",argv[4]);
    compute_single_GPU(graph, iterations);
    }else if(std::string(argv[4])==std::string("TESTGF")){
        printf("%s c engine start",argv[4]);
    compute_single_gf_GPU(graph, iterations);
    }else if(std::string(argv[4])==std::string("DIST")){
        compute_dist_GPU(graph, iterations);
        std::cout<<"id"<<graph->partition_id<<" vertices "<<graph->owned_vertices;
    }else if(std::string(argv[4])==std::string("TESTGFD")){
        printf("%s c engine start",argv[4]);
    compute_dist_gf_GPU(graph, iterations);
    }else if(std::string(argv[4])==std::string("status")){
        printf("%s statistic",argv[4]);
    statistic(graph,4);
    statistic(graph,8);
    statistic(graph,16);
    statistic(graph,32);
        float *a=new float[8];
    for(int i=0;i<8;i++){
        a[i]=(float)i;
    }
    int *b=new int[8];
    for(int i=0;i<4;i++){
        b[i]=(float)i;
    }
//     int *c=new int[8];
//     c[0]=2;c[1]=1;c[2]=1;c[3]=1;
//     c[4]=1;c[5]=1;c[6]=1;c[7]=0;
//    torch::Tensor test=torch::from_blob(a,{1,8});
//    torch::Tensor w=torch::from_blob(b,{2,2});
//    //torch::Tensor dst=torch::zeros({1,3});
//    torch::Tensor index=torch::from_blob(c,{1,8},torch::kLong);
//    torch::Tensor dst;
//    dst.scatter(1,index,test);
    //std::cout<<test.slice(0,0,2,1)<<std::endl;
    }else if(std::string(argv[4])==std::string("COO")){
    VertexSubset * active = graph->alloc_vertex_subset();
    active->fill();
    graph->generate_COO(active);
    graph->reorder_COO(4096);

    }
      exec_time += get_time();
  if (graph->partition_id==0) {
    printf("exec_time=%lf(s)\n", exec_time);

  }
   delete graph;


    return 0;
}
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
#include "comm/Network.hpp"
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
    torch::Tensor src;
    torch::Tensor dst;
    torch::Tensor weight;
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

struct Intermediate : torch::nn::Module
{
    torch::Tensor W;
    ValueType *W_from;

    Intermediate(size_t w, size_t h)
    {
        //        at::TensorOptions *opt=new at::TensorOptions;
        //       opt->requires_grad(true);
        //  torch::randn
        //     A=torch::randn(torch::randn({w,h},opt));
        W = register_parameter("W", torch::randn({w, h}));
        W_from = new ValueType[w * h];
    }
    void resetW(size_t w, size_t h, ValueType *buffer)
    {
        memcpy(W_from, buffer, sizeof(ValueType) * w * h);
        torch::Tensor new_weight_tensor = torch::from_blob(W_from, {w, h});
        W.set_data(new_weight_tensor);
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = x.mm(W);
        return x;
    }
};

struct GnnUnit : torch::nn::Module
{
    torch::Tensor W;
    ValueType *W_from;
    ValueType *w_gradient_buffer;
    Network_simple<float> *network_simple;
    int row, col;
    torch::Tensor W_gradient;
    //gpu_processor *gp;
    GnnUnit(size_t w, size_t h)
    {
        //        at::TensorOptions *opt=new at::TensorOptions;
        //       opt->requires_grad(true);
        //  torch::randn
        //     A=torch::randn(torch::randn({w,h},opt));
        row = w;
        col = h;
        W = register_parameter("W", torch::randn({w, h}, torch::kFloat));
        W_from = new ValueType[w * h];
        w_gradient_buffer = new ValueType[w * h];
        memset(w_gradient_buffer, 0, sizeof(float) * w * h);
        W_gradient = torch::from_blob(w_gradient_buffer, {w, h}, torch::kFloat);
        network_simple = new Network_simple<float>(row, col);
    }
    void init_parameter()
    {
        network_simple->broadcast(W.accessor<ValueType, 2>().data());
    }
    void all_reduce_to_gradient(torch::Tensor from)
    {
        W_gradient.set_data(from);
        //memcpy(w_gradient_buffer, from.accessor<ValueType, 2>().data(), sizeof(float) * row * col);
        //std::cout << from[0][3] << "ss" << w_gradient_buffer[3] << "ds" << W_gradient[0][3] << std::endl;
        network_simple->all_reduce_sum(W_gradient.accessor<ValueType, 2>().data());
        //W_gradient = torch::from_blob(w_gradient_buffer, {row, col}, torch::kFloat);
    }
    void resetW(size_t w, size_t h, ValueType *buffer)
    {
        memcpy(W_from, buffer, sizeof(ValueType) * w * h);
        torch::Tensor new_weight_tensor = torch::from_blob(W_from, {w, h});
        W.set_data(new_weight_tensor);
    }

    void learnC2G(ValueType learning_rate)
    {
        torch::Tensor tmp = W_gradient.cuda();
        torch::Tensor a = (W - (tmp * learning_rate));
        W.set_data(a);
    }
    
    void learnC2G_with_decay(ValueType learning_rate,ValueType weight_decay)
    {
        torch::Tensor tmp = W_gradient.cuda();
        torch::Tensor a = (W - (tmp * learning_rate))*(1-weight_decay);
        W.set_data(a);
    }


    void learn(torch::Tensor from, ValueType learning_rate)
    {
        torch::Tensor a = (W - (from * learning_rate));

        W.set_data(a);
    }
    void learn_gpu(torch::Tensor from, ValueType learning_rate)
    {
        torch::Tensor a = (W - (from * learning_rate));
        W.set_data(a);
        //W=a;
    }
    torch::Tensor forward(torch::Tensor x)
    {

        torch::Tensor x1 = x.mm(W);
        return x1;
    }

    torch::Tensor forward2(torch::Tensor x)
    {
        return torch::sigmoid(x);
    }
    torch::Tensor forward3(torch::Tensor x)
    {

        x = x.mm(W);
        return x.log_softmax(1);
    }
};

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
    void registLabel(torch::Tensor &target)
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

template <typename T_v, typename T_l>
class Embeddings
{
public:
    Embeddings()
    {
    }
    T_v *start_v = NULL;
    //   T_v** partial_grad=NULL;
    GnnUnit *Gnn_v1 = NULL;
    GnnUnit *Gnn_v2 = NULL;
    T_l *label = NULL;
    T_v *local_grad = NULL;
    T_v *aggre_grad = NULL;

    int rownum;
    int start;
    int start_vertex;
    content *con;
    Compressed_Feature *cf;
    std::vector<origin_feature> con1;

    void generate_compressed_feature(Graph<Empty> *graph)
    {
        cf = new Compressed_Feature;
        cf->column_index = new VertexId[graph->owned_vertices + 1];
        memset(cf->column_index, 0, (graph->owned_vertices + 1) * sizeof(VertexId));
        cf->count = 0;
        int bias = graph->partition_offset[graph->partition_id];
        for (int i = 0; i < graph->owned_vertices; i++)
        {
            for (int j = 0; j < SIZE_LAYER_1; j++)
            {
                if (con[i + bias].att[j] != 0.0)
                {
                    cf->count++;
                    cf->column_index[i + 1] += 1;
                }
            }
        }
        for (int i = 0; i < graph->owned_vertices; i++)
        {
            cf->column_index[i + 1] = cf->column_index[i] + cf->column_index[i + 1];
        }
        std::cout << cf->count << std::endl;
        cf->position = new VertexId[cf->count + 1];
        cf->value_list = new ValueType[cf->count + 1];
        memset(cf->position, 0, (cf->count + 1) * sizeof(VertexId));
        memset(cf->value_list, 0, (cf->count + 1) * sizeof(ValueType));
        for (int i = 0; i < graph->owned_vertices; i++)
        {
            int local_index = 0;
            for (int j = 0; j < SIZE_LAYER_1; j++)
            {
                if (con[i + bias].att[j] != 0.0)
                {
                    cf->position[cf->column_index[i] + (local_index)] = j;
                    cf->value_list[cf->column_index[i] + (local_index)] = con[i + bias].att[j];
                    local_index++;
                }
            }
            if (local_index != (cf->column_index[i + 1] - cf->column_index[i]))
            {
                std::cout << i << std::endl;
            }
        }
        std::cout << "generate compressed_structure finished\n";
    }
    void init(Graph<Empty> *graph)
    {
        start_v = new ValueType[graph->vertices * SIZE_LAYER_1]; //graph->alloc_vertex_array<float>(VECTOR_LENGTH);

        label = graph->alloc_vertex_array<T_l>();
        rownum = (graph->partition_offset[graph->partition_id + 1] - graph->partition_offset[graph->partition_id]);
        start = SIZE_LAYER_1 * (graph->partition_offset[graph->partition_id]);
        start_vertex = graph->partition_offset[graph->partition_id];
        //printf("graph %ld\n", graph->vertices);
    }
    void initStartWith(int index, ValueType with, int i)
    {

        *(start_v + index * SIZE_LAYER_1 + i) = with;
        //   curr[index].data[i] = (float)with;
    }
    void initStartWith(int index, ValueType with)
    {
        for (int i = 0; i < SIZE_LAYER_1; i++)
        {
            *(start_v + index * SIZE_LAYER_1 + i) = with;
            //   curr[index].data[i] = (float)with;
        }
    }
    void readlabel(Graph<Empty> *graph)
    {
        graph->fill_vertex_array(label, (long)1);
    }
    inline void readlabel_(Graph<Empty> *graph)
    {
        //readlabel1(graph);
        //readlabel_pub(graph);
        //readlabel_google(graph);
        readlabel_orkut(graph);
        //generate_compressed_feature(graph);
        //readlabel_citeseer(graph);
        //writedatatonodelabel();
    }
    void writedatatonodelabel()
    {
        std::ofstream outputf("./node_table", std::ios::out);
        std::ofstream outputtest("./test_table", std::ios::out);
        std::ofstream outputtrain("./train_table", std::ios::out);
        outputtest << "id:int64	weight:float" << std::endl;
        outputtrain << "id:int64	weight:float" << std::endl;
        outputf << "id:int64	label:int64	feature:string" << std::endl;
        for (int i = 0; i < NODE_NUMBER; i++)
        {
            outputf << i << "\t" << con[i].label << "\t";
            for (int j = 0; j < SIZE_LAYER_1; j++)
            {
                if (j != (SIZE_LAYER_1 - 1))
                    outputf << con[i].att[j] << ":";
                else
                    outputf << con[i].att[j] << std::endl;
            }
            if (i < (0.8 * NODE_NUMBER))
            {
                outputtrain << i << "\t"
                            << "1.0" << std::endl;
            }
            else
            {
                outputtest << i << "\t"
                           << "1.0" << std::endl;
            }
        }
        outputf.close();
        outputtest.close();
        outputtrain.close();
    }
    void readlabel1(Graph<Empty> *graph)
    {
        //graph->fill_vertex_array(label, (long) 1);
        std::string str;
        std::ifstream input("cora.content", std::ios::in);
        //std::ofstream outputl("cora.labeltable",std::ios::out);

        if (!input.is_open())
        {
            //cout<<"open file fail!"<<endl;
            return;
        }
        con = new content[NODE_NUMBER];
        int numOfData = 0;
        std::string la;
        //std::cout<<"finish1"<<std::endl;
        while (input >> con[numOfData].id)
        {

            for (int i = 0; i < SIZE_LAYER_1; i++)
            {
                input >> con[numOfData].att[i];
            }
            input >> la;
            //cout<<"==============================="<<endl;
            con[numOfData].label = changelable(la);

            //   std::cout<<"lable: "<<con[numOfData].label<<" "<<numOfData<<std::endl;
            numOfData++;
        }
        //    std::cout<<"finish1"<<std::endl;
        for (int i = 0; i < NODE_NUMBER; i++)
        {
            if (i < graph->vertices)
                label[i] = con[i].label;
        }
        //    std::cout<<"finish"<<std::endl;
        input.close();
        // std::cout<<"finish??"<<std::endl;
    }

    void readlabel_pub(Graph<Empty> *graph)
    {
        std::string str;
        //readfeature
        con = new content[NODE_NUMBER];
        std::ifstream input("./pubmed_data/fea.txt", std::ios::in);
        if (!input.is_open())
        {
            std::cout << "open file fail!" << std::endl;
        }
        std::cout << "read data started!" << std::endl;
        int n = 0;
        std::string lable;
        while (n < NODE_NUMBER)
        {
            con[n].id = n;
            for (int i = 0; i < SIZE_LAYER_1; i++)
            {
                input >> con[n].att[i];
            }
            //std::cout<<std::endl;
            n++;
        }
        std::cout << "read data finished!" << std::endl;
        input.close();

        //readlabel
        std::ifstream inputlabel("./pubmed_data/labels.txt", std::ios::in);
        if (!inputlabel.is_open())
        {
            std::cout << "open y file fail!" << std::endl;
        }
        int l = 0;
        while (l < NODE_NUMBER)
        {
            inputlabel >> con[l].label;
            l++;
        }
        inputlabel.close();
        for (int i = 0; i < NODE_NUMBER; i++)
        {
            if (i < graph->vertices)
                label[i] = con[i].label;
        }
    }
    void readlabel_citeseer(Graph<Empty> *graph)
    {
        std::string str;
        con = new content[NODE_NUMBER];
        std::ifstream input("./citeseer_data/fea.txt", std::ios::in);
        if (!input.is_open())
        {
            std::cout << "open file fail!" << std::endl;
            return;
        }
        int n = 0;
        std::string lable;
        while (n < NODE_NUMBER)
        {
            con[n].id = n;
            for (int i = 0; i < SIZE_LAYER_1; i++)
            {
                input >> con[n].att[i];
            }
            //std::cout<<std::endl;
            n++;
        }
        input.close();

        //readlabel
        std::ifstream inputlabel("./citeseer_data/labels.txt", std::ios::in);
        if (!inputlabel.is_open())
        {
            std::cout << "open y file fail!" << std::endl;
            return;
        }
        int l = 0;
        while (l < NODE_NUMBER)
        {
            inputlabel >> con[l].label;
            //label[l]=con[l].label;
            l++;
        }
        for (int i = 0; i < NODE_NUMBER; i++)
        {
            if (i < graph->vertices)
                label[i] = con[i].label;
        }
        inputlabel.close();
    }
    void readlabel_google(Graph<Empty> *graph)
    {
        std::string str;
        con = new content[NODE_NUMBER];
        std::ifstream input("/home/wangqg/NetBeansProjects/DataPreProcessing/google/google_contents", std::ios::in);
        if (!input.is_open())
        {
            std::cout << "open file fail!" << std::endl;
            return;
        }
        int n = 0;
        std::string lable;
        while (n < NODE_NUMBER)
        {
            int tmpid;
            input >> tmpid;
            input >> con[n].label;
            con[n].id = n;
            for (int i = 0; i < SIZE_LAYER_1; i++)
            {
                input >> con[n].att[i];
            }
            //std::cout<<std::endl;
            n++;
        }
        input.close();
        for (int i = 0; i < NODE_NUMBER; i++)
        {
            if (i < graph->vertices)
                label[i] = con[i].label;
        }
    }

    void readlabel_orkut(Graph<Empty> *graph)
    {
        std::string str;
        con = new content[NODE_NUMBER];
        int n = 0;
        std::string lable;
        while (n < NODE_NUMBER)
        {
            int tmpid;
            tmpid = n;
            con[n].label = rand() % LABEL_NUMBER;
            con[n].id = n;
            //std::cout<<std::endl;
            n++;
        }
        for (int i = 0; i < NODE_NUMBER; i++)
        {
            if (i < graph->vertices)
                label[i] = con[i].label;
        }
    }
};
/*
class tensorSet
{
public:
    std::vector<torch::optim::SGD> optimizers;
    std::vector<torch::Tensor> x; // after graph engine;
    std::vector<torch::Tensor> y;
    std::vector<torch::Tensor> localGrad;
    std::vector<torch::Tensor> backwardGrad;
    torch::Tensor target; //read label
    torch::Tensor loss;
    torch::Tensor in_degree;
    torch::Tensor out_degree;

    int layers = 0;
    tensorSet(int layers_)
    {
        for (int i = 0; i < layers_; i++)
        {
            x.push_back(torch::tensor(0.0, torch::kFloat));
            y.push_back(torch::tensor(0.0, torch::kFloat));
            localGrad.push_back(torch::tensor(0.0, torch::kFloat));
            backwardGrad.push_back(torch::tensor(0.0, torch::kFloat));
            layers = layers_;
        }
    }
    void registOptimizer(torch::optim::SGD opt)
    {
        opt.zero_grad();
        optimizers.push_back(opt);
    }

    void updateX(int layer_, torch::Tensor src)
    {
        x[layer_] = src;
    }

    template <typename T_l>
    void registLabel(T_l *label, int start, int rownum)
    {
        target = torch::from_blob(label + start, rownum, torch::kLong);
    }
};
*/
void init_parameter(Network<ValueType> *comm, Graph<Empty> *graph, GnnUnit *gnn, Embeddings<ValueType, long> *embedding)
{

    int r, c;
    r = gnn->W.accessor<ValueType, 2>().size(0);
    c = gnn->W.accessor<ValueType, 2>().size(1);

    if (graph->partition_id == 1)
    { //first layer
        comm->wrtBuffertoBuff(gnn->W.accessor<ValueType, 2>().data(), r, c);
    }
    comm->broadcastW(gnn->W.accessor<ValueType, 2>().data(), r, c); // comm buffer

    gnn->resetW(r, c, comm->buffer);
}
template <typename t_v>
torch::Tensor unified_parameter(Network<ValueType> *comm, torch::Tensor res)
{
    //return res;
    int r = res.accessor<ValueType, 2>().size(0);
    int c = res.accessor<ValueType, 2>().size(1);
    comm->wrtBuffertoBuff(res.accessor<ValueType, 2>().data(), r, c); //write para to send buffer
    comm->gatherW(r, c);                                              // gather from others to recvbuffer
    comm->computeW(r, c);                                             // compute new para on recvbuffer
    comm->broadcastW(comm->recv_buffer, r, c);
    return torch::from_blob(comm->buffer, {r, c});
}

template <typename t_v, typename t_l, int MAX_EMBEDDING_SIZE>
class GTensor
{

public:
    Graph<Empty> *graph_;
    Embeddings<t_v, t_l> *embedding_;
    t_v *edgeValue;

    torch::Tensor pre_value;
    torch::Tensor value;
    torch::Tensor pre_grad;
    torch::Tensor grad;

    t_v **value_buffer;
    t_v **grad_buffer;
    std::vector<torch::Tensor> value_local;
    std::vector<torch::Tensor> grad_local;
    t_v **partial_grad_buffer;

    void **curr;
    void **next;
    void **commbuffer;
    t_v *data_coomm;
    nodeVector<MAX_EMBEDDING_SIZE> *curr_local;
    nodeVector<MAX_EMBEDDING_SIZE> *next_local;

    int start_;
    int rownum_;
    int layer_;
    int current_layer;
    const int s = 0;
    VertexSubset *active_;

    int *size_at_layer;

    inline void set_size(std::vector<int> size_)
    {
    }
    inline void zeroCurr()
    {
        memset(curr_local, 0, sizeof(nodeVector<MAX_EMBEDDING_SIZE>) * rownum_);
    }
    //    inline void zeroCurr(int layer){
    //        memset(curr[layer], 0,sizeof(nodeVector<MAX_LAYER>)*rownum_);
    //    }
    inline void zeroNext()
    {
        memset(next_local, 0, sizeof(nodeVector<MAX_EMBEDDING_SIZE>) * rownum_);
    }
    //    inline void zeroNext(int layer){
    //        memset(next_local, 0,sizeof(nodeVector<MAX_LAYER>)*rownum_);
    //    }
    inline void cpToCurr(int vtx, t_v *src)
    {
        memcpy(curr_local[vtx - start_].data, src, sizeof(nodeVector<MAX_EMBEDDING_SIZE>));
    }
    template <int EBDING_SIZE>
    inline void cpToCurr(int vtx, t_v *src)
    {
        memcpy(((nodeVector<EBDING_SIZE> *)curr_local)[vtx - start_].data, src, sizeof(nodeVector<EBDING_SIZE>));
    }
    template <int EBDING_SIZE>
    inline void cpToNext(int vtx, t_v *src)
    {
        memcpy(((nodeVector<EBDING_SIZE> *)next_local)[vtx - start_].data, src, sizeof(nodeVector<EBDING_SIZE>));
    }

    inline void cpToNext(int vtx, t_v *src)
    {
        memcpy(next_local[vtx - start_].data, src, sizeof(nodeVector<MAX_EMBEDDING_SIZE>));
    }

    inline void addToNext(int vtx, t_v *src)
    {
        for (int i = 0; i < MAX_EMBEDDING_SIZE; i++)
        {
            next_local[vtx - start_].data[i] += src[i];
        }
    }
    template <int EBDING_SIZE>
    inline void addToNext(int vtx, t_v *src)
    {
        for (int i = 0; i < EBDING_SIZE; i++)
        {
            ((nodeVector<EBDING_SIZE> *)next_local)[vtx - start_].data[i] += src[i];
        }
    }
    template <int EBDING_SIZE>
    inline void addToNextValue(int layer, int vtx, t_v *src)
    {
        for (int i = 0; i < EBDING_SIZE; i++)
        {
            ((nodeVector<EBDING_SIZE> *)value_buffer[layer])[vtx - start_].data[i] += src[i];
        }
    }
    template <int EBDING_SIZE>
    inline void addToNextGrad(int layer, int vtx, t_v *src)
    {
        for (int i = 0; i < EBDING_SIZE; i++)
        {
            ((nodeVector<EBDING_SIZE> *)grad_buffer[layer])[vtx - start_].data[i] += src[i];
        }
    }

    inline void zeroValue(int layer)
    {
        memset(value_buffer[layer], 0, sizeof(t_v) * size_at_layer[layer] * rownum_);
        // memcpy(valuebuffer[vtx-start_].data, src, sizeof(nodeVector));
    }
    inline void zeroGrad(int layer)
    {
        memset(grad_buffer[layer], 0, sizeof(t_v) * size_at_layer[layer] * rownum_);
        // memcpy(valuebuffer[vtx-start_].data, src, sizeof(nodeVector));
    }
    GTensor(Graph<Empty> *graph, Embeddings<t_v, t_l> *embedding, VertexSubset *active, int layer, std::vector<int> size_)
    {
        graph_ = graph;
        embedding_ = embedding;
        active_ = active;
        start_ = graph->partition_offset[graph->partition_id];
        rownum_ = graph->partition_offset[graph->partition_id + 1] - graph->partition_offset[graph->partition_id];
        layer_ = layer;
        current_layer = -1;
        size_at_layer = new int[layer + 1];
        data_coomm = new t_v[SIZE_LAYER_1 * graph->vertices];
        for (int i = 0; i < layer_ + 1; i++)
        {
            size_at_layer[i] = size_[i];
        }
        // partial_grad_buffer=new t_v[MAX_LAYER*MAX_LAYER];
        value_buffer = new t_v *[layer];
        grad_buffer = new t_v *[layer];
        //        curr= new void*[layer];
        //        next= new void*[layer];
        partial_grad_buffer = new t_v *[layer];

        curr_local = graph->alloc_vertex_array_local<nodeVector<MAX_EMBEDDING_SIZE>>();
        next_local = graph->alloc_vertex_array_local<nodeVector<MAX_EMBEDDING_SIZE>>();
        value_local.clear();
        grad_local.clear();
        for (int i = 0; i < layer; i++)
        {
            torch::Tensor x1, x2;
            value_local.push_back(x1);
            grad_local.push_back(x2);
            value_buffer[i] = graph->alloc_vertex_array_local<t_v>(size_at_layer[i]);
            grad_buffer[i] = graph->alloc_vertex_array_local<t_v>(size_at_layer[i]);
            //            curr[layer]=(void*)graph->alloc_vertex_array_local<nodeVector<size_at_layer[layer]>>();
            //            next[layer]=(void*)graph->alloc_vertex_array_local<nodeVector<size_at_layer[layer]>>();
        }
        graph_->process_vertices<t_v>( //init  the vertex state.
            [&](VertexId vtx) {
                if (graph_->in_degree_for_backward[vtx] < 1)
                    graph->in_degree_for_backward[vtx] = 1; //local
                if (graph_->out_degree_for_backward[vtx] < 1)
                    graph->out_degree_for_backward[vtx] = 1; //local
                return (ValueType)1;
            },
            active_);
    }

    GTensor(Graph<Empty> *graph, VertexSubset *active)
    {
        graph_ = graph;
        active_ = active;
        start_ = graph->partition_offset[graph->partition_id];
        rownum_ = graph->partition_offset[graph->partition_id + 1] - graph->partition_offset[graph->partition_id];
    }

    torch::Tensor applyVertex(int local_layer, torch::Tensor vertex_data)
    {
        torch::Tensor destination;
        bool s = value_local[local_layer].accessor<ValueType, 2>().size(0) == vertex_data.accessor<ValueType, 2>().size(0);
        bool t = value_local[local_layer].accessor<ValueType, 2>().size(1) == vertex_data.accessor<ValueType, 2>().size(1);
        //assert(source.accessor<float,2>().size(0)==vertex_data.accessor<float,2>().size(0));
        //assert(source.accessor<float,2>().size(1)==vertex_data.accessor<float,2>().size(1));
        assert((s && t) == 1);
        destination = value_local[local_layer] * vertex_data;
    }
    torch::Tensor applyVertexPre(torch::Tensor source, torch::Tensor vertex_data)
    {
        torch::Tensor destination;
        bool s = source.accessor<ValueType, 2>().size(0) == vertex_data.accessor<ValueType, 2>().size(0);
        bool t = source.accessor<ValueType, 2>().size(1) == vertex_data.accessor<ValueType, 2>().size(1);
        //assert(source.accessor<float,2>().size(0)==vertex_data.accessor<float,2>().size(0));
        //assert(source.accessor<float,2>().size(1)==vertex_data.accessor<float,2>().size(1));
        assert((s && t) == 1);
        return source * vertex_data;
    }
    template <int CURRENT_LAYER_SIZE>
    torch::Tensor Propagate(int local_layer)
    {
        zeroValue(local_layer);

        graph_->process_vertices<t_v>( //init  the vertex state.
            [&](VertexId vtx) {
                cpToCurr<CURRENT_LAYER_SIZE>(vtx, pre_value.accessor<t_v, 2>().data() + (vtx - start_) * CURRENT_LAYER_SIZE); //local
                return (ValueType)1;
            },
            active_);
        // printf("unfinish propagate at layer [%d]\n",local_layer);
        graph_->process_edges<int, nodeVector<CURRENT_LAYER_SIZE>>( // For EACH Vertex Processing
            [&](VertexId dst, VertexAdjList<Empty> incoming_adj) {  //pull
                nodeVector<CURRENT_LAYER_SIZE> sum;
                memset(sum.data, 0, sizeof(ValueType) * CURRENT_LAYER_SIZE);
                for (AdjUnit<Empty> *ptr = incoming_adj.begin; ptr != incoming_adj.end; ptr++)
                { //pull model
                    VertexId src = ptr->neighbour;
                    for (int i = 0; i < CURRENT_LAYER_SIZE; i++)
                    {
                        //if(0.0!=((nodeVector<CURRENT_LAYER_SIZE>*)curr_local)[src-start_].data[i])
                        sum.data[i] += ((nodeVector<CURRENT_LAYER_SIZE> *)curr_local)[src - start_].data[i] / ((ValueType)std::sqrt(graph_->out_degree_for_backward[src]) * (ValueType)std::sqrt(graph_->in_degree_for_backward[dst]));
                    }
                }
                graph_->emit(dst, sum);
            },
            [&](VertexId dst, nodeVector<CURRENT_LAYER_SIZE> msg) {
                //addToNext<CURRENT_LAYER_SIZE>(dst,msg.data);//local
                addToNextValue<CURRENT_LAYER_SIZE>(local_layer, dst, msg.data);
                return 0;
            },
            active_);

        value_local[local_layer] = torch::from_blob(value_buffer[local_layer], {rownum_, CURRENT_LAYER_SIZE}, torch::kFloat);
        current_layer = local_layer;
        if (graph_->partition_id == 0)
            printf("finish propagate at layer [%d]\n", local_layer);
        return value_local[local_layer];
    }
    template <int CURRENT_LAYER_SIZE>
    torch::Tensor Propagate_inplace(int local_layer, ValueType *input)
    {
        zeroValue(local_layer);

        graph_->process_edges<int, nodeVector<CURRENT_LAYER_SIZE>>( // For EACH Vertex Processing
            [&](VertexId dst, VertexAdjList<Empty> incoming_adj) {  //pull
                nodeVector<CURRENT_LAYER_SIZE> sum;
                memset(sum.data, 0, sizeof(ValueType) * CURRENT_LAYER_SIZE);
                for (AdjUnit<Empty> *ptr = incoming_adj.begin; ptr != incoming_adj.end; ptr++)
                { //pull model
                    VertexId src = ptr->neighbour;
                    for (int i = 0; i < CURRENT_LAYER_SIZE; i++)
                    {
                        if (0.0 != ((nodeVector<CURRENT_LAYER_SIZE> *)input)[src - start_].data[i])
                            sum.data[i] += ((nodeVector<CURRENT_LAYER_SIZE> *)input)[src - start_].data[i] / ((ValueType)std::sqrt(graph_->out_degree_for_backward[src]) * (ValueType)std::sqrt(graph_->in_degree_for_backward[dst]));
                    }
                }
                graph_->emit(dst, sum);
            },
            [&](VertexId dst, nodeVector<CURRENT_LAYER_SIZE> msg) {
                //addToNext<CURRENT_LAYER_SIZE>(dst,msg.data);//local
                addToNextValue<CURRENT_LAYER_SIZE>(local_layer, dst, msg.data);
                return 0;
            },
            active_);

        value_local[local_layer] = torch::from_blob(value_buffer[local_layer], {rownum_, CURRENT_LAYER_SIZE}, torch::kFloat);
        current_layer = local_layer;
        if (graph_->partition_id == 0)
            printf("finish propagate at layer [%d]\n", local_layer);
        return value_local[local_layer];
    }
    template <int CURRENT_LAYER_SIZE>
    void Propagate_backward(int local_layer)
    {
        zeroGrad(local_layer);               //local
        graph_->process_vertices<ValueType>( //init  the vertex state.
            [&](VertexId vtx) {
                cpToNext<CURRENT_LAYER_SIZE>(vtx, pre_grad.accessor<t_v, 2>().data() + (vtx - start_) * CURRENT_LAYER_SIZE); //local
                return 1;
            },
            active_);
        //start graph engine.
        graph_->process_edges_backward<int, nodeVector<CURRENT_LAYER_SIZE>>( // For EACH Vertex Processing
            [&](VertexId src, VertexAdjList<Empty> outgoing_adj) {           //pull
                nodeVector<CURRENT_LAYER_SIZE> sum;
                memset(sum.data, 0, sizeof(ValueType) * CURRENT_LAYER_SIZE);
                for (AdjUnit<Empty> *ptr = outgoing_adj.begin; ptr != outgoing_adj.end; ptr++)
                { //pull model
                    VertexId dst = ptr->neighbour;
                    for (int i = 0; i < CURRENT_LAYER_SIZE; i++)
                    {
                        //if(0!=((nodeVector<CURRENT_LAYER_SIZE>*)next_local)[dst-start_].data[i])
                        sum.data[i] += ((nodeVector<CURRENT_LAYER_SIZE> *)next_local)[dst - start_].data[i] / ((ValueType)std::sqrt(graph_->out_degree_for_backward[src]) * (ValueType)std::sqrt(graph_->in_degree_for_backward[dst]));
                    }
                }
                graph_->emit(src, sum);
            },
            [&](VertexId src, nodeVector<CURRENT_LAYER_SIZE> msg) {
                addToNextGrad<CURRENT_LAYER_SIZE>(local_layer, src, msg.data);
                return 0;
            },
            active_);

        grad_local[local_layer] = torch::from_blob(grad_buffer[local_layer], {rownum_, CURRENT_LAYER_SIZE}); //local
        grad = grad_local[local_layer];
        if (graph_->partition_id == 0)
            printf("finish backward propagate at layer [%d]\n", local_layer);
    }

    template <int CURRENT_LAYER_SIZE>
    void Sync_data_optional(torch::Tensor &data, torch::Tensor &data_to)
    {
        data_to.zero_();

        printf("new sync\n");
        //std::cout<<"start"<<start_<<" "<<data.accessor<t_v,2>().size(1)<<std::endl;
        //memset(data_coomm,0,sizeof(t_v)*CURRENT_LAYER_SIZE*graph_->vertices);
        graph_->process_edges<int, nodeVector<CURRENT_LAYER_SIZE>>( // For EACH Vertex Processing
            [&](VertexId src, VertexAdjList<Empty> outgoing_adj) {  //pull
                                                                    // if(src>=2200){
                                                                    //     std::cout<<"yes "<<src<<" is called"<<std::endl;
                                                                    // }
                nodeVector<CURRENT_LAYER_SIZE> sum;
                if (graph_->out_degree_for_backward[src] <= 3)
                {
                    memcpy(sum.data, data.accessor<t_v, 2>().data() + (src)*CURRENT_LAYER_SIZE, sizeof(t_v) * CURRENT_LAYER_SIZE);
                    graph_->emit(src, sum);
                }
            },
            [&](VertexId src, nodeVector<CURRENT_LAYER_SIZE> msg) {
                //memset(data_coomm+(src)*CURRENT_LAYER_SIZE,0,sizeof(t_v)*CURRENT_LAYER_SIZE);
                for (int i = 0; i < CURRENT_LAYER_SIZE; i++)
                {
                    //data_coomm[(src)*CURRENT_LAYER_SIZE+i]+=msg.data[i];
                    data_to.accessor<t_v, 2>().data()[(src - start_) * CURRENT_LAYER_SIZE + i] += msg.data[i];
                }
                return 0;
            },
            active_);
    }
    template <int CURRENT_LAYER_SIZE>
    void Sync_data(torch::Tensor &data, torch::Tensor &data_to)
    {
        data_to.zero_();
        graph_->process_edges_simple<int, nodeVector<CURRENT_LAYER_SIZE>>( // For EACH Vertex Processing
            [&](VertexId src, VertexAdjList<Empty> outgoing_adj) {         //pull
                nodeVector<CURRENT_LAYER_SIZE> sum;
                //if(graph_->out_degree_for_backward[src]<=3){
                memcpy(sum.data, data.accessor<t_v, 2>().data() + (src)*CURRENT_LAYER_SIZE, sizeof(t_v) * CURRENT_LAYER_SIZE);
                graph_->emit(src, sum);
                //}
            },
            [&](VertexId src, nodeVector<CURRENT_LAYER_SIZE> msg) {
                //            if(src==1023&&graph_->partition_id==0){
                //                  printf("CPUget %f\n",msg.data[0]);
                //              }
                //memset(data_coomm+(src)*CURRENT_LAYER_SIZE,0,sizeof(t_v)*CURRENT_LAYER_SIZE);
                for (int i = 0; i < CURRENT_LAYER_SIZE; i++)
                {
                    //data_coomm[(src)*CURRENT_LAYER_SIZE+i]+=msg.data[i];
                    data_to.accessor<t_v, 2>().data()[(src - start_) * CURRENT_LAYER_SIZE + i] += msg.data[i];
                }
                return 0;
            },
            active_);
        //return torch::from_blob(data_coomm+(start_)*CURRENT_LAYER_SIZE,{rownum_,CURRENT_LAYER_SIZE},torch::kFloat);
    }

    template <int CURRENT_LAYER_SIZE>
    void Sync_data_gpu(torch::Tensor &data, torch::Tensor &data_to, bool selective = false)
    {
        if (!selective)
        {                                                                                   // original communication
            graph_->process_edges_with_GPU_aggregator<int, nodeVector<CURRENT_LAYER_SIZE>>( // For EACH Vertex Processing
                [&](VertexId src, VertexAdjList<Empty> outgoing_adj) {                      //pull
                    nodeVector<CURRENT_LAYER_SIZE> sum;
                    memcpy(sum.data, data.accessor<t_v, 2>().data() + (src)*CURRENT_LAYER_SIZE, sizeof(t_v) * CURRENT_LAYER_SIZE);
                    graph_->emit(src, sum);
                },
                data_to.packed_accessor<float, 2>().data(), CURRENT_LAYER_SIZE, active_);
        }
        else
        {                                                                                   //selective comunication
            graph_->process_edges_with_GPU_aggregator<int, nodeVector<CURRENT_LAYER_SIZE>>( // For EACH Vertex Processing
                [&](VertexId src, VertexAdjList<Empty> outgoing_adj) {                      //pull
                    nodeVector<CURRENT_LAYER_SIZE> sum;
                    memcpy(sum.data, data.accessor<t_v, 2>().data() + (src)*CURRENT_LAYER_SIZE, sizeof(t_v) * CURRENT_LAYER_SIZE);
                    graph_->emit(src, sum);
                },
                data_to.packed_accessor<float, 2>().data(), CURRENT_LAYER_SIZE, active_);
        }

        //return torch::from_blob(data_coomm+(start_)*CURRENT_LAYER_SIZE,{rownum_,CURRENT_LAYER_SIZE},torch::kFloat);
    }


    
    void Process_GPU_overlap_explict(torch::Tensor &X, float *Y_buffered, torch::Tensor &Y, std::vector<CSC_segment_pinned *> &graph_partitions)
    {
        int current_layer_size = graph_->gnnctx->layer_size[graph_->rtminfo->curr_layer];
        bool selective = graph_->rtminfo->reduce_comm;
        int layer = graph_->rtminfo->curr_layer;

        if (!selective)
        { // original communication
            //printf("done?\n");
            graph_->compute_sync_explict<int, float>(
                X,
                Y_buffered,
                graph_partitions,
                [&](VertexId src, VertexAdjList<Empty> outgoing_adj) { //pull
                    //nodeVector<CURRENT_LAYER_SIZE> sum;
                    //memcpy(sum.data, Y_buffered + (src)*current_layer_size, sizeof(t_v) * CURRENT_LAYER_SIZE);
                    graph_->emit_buffer(src, Y_buffered + (src)*current_layer_size, current_layer_size);
                },
                Y.packed_accessor<float, 2>().data());
            //printf("done!\n");
        }
        else
        { //selective comunication
            graph_->compute_sync_explict<int, float>(
                X,
                Y_buffered,
                graph_partitions,
                [&](VertexId src, VertexAdjList<Empty> outgoing_adj) { //pull
                    if (!graph_->RepVtx[layer]->get_bit(src))
                    {
                        // nodeVector<CURRENT_LAYER_SIZE> sum;
                        // memcpy(sum.data, Y_buffered + (src)*current_layer_size, sizeof(t_v) * CURRENT_LAYER_SIZE);
                        // graph_->emit(src, sum);
                        graph_->emit_buffer(src, Y_buffered + (src)*current_layer_size, current_layer_size);
                    }
                },
                Y.packed_accessor<float, 2>().data());
        }
    }
    
    void Process_GPU_overlap_sync_compute_explict(torch::Tensor &X, float *Y_buffered, torch::Tensor &Y, std::vector<CSC_segment_pinned *> &graph_partitions)
    {
        int current_layer_size = graph_->gnnctx->layer_size[graph_->rtminfo->curr_layer];
        bool selective = graph_->rtminfo->reduce_comm;
        int layer = graph_->rtminfo->curr_layer;
        torch::Tensor X_cpu=X.cpu();
        float *X_buffered=X_cpu.accessor<float,2>().data();
        //float *X_buffered1=new float[current_layer_size];
        if (!selective)
        { // original communication
            //printf("done?\n");
            graph_->sync_compute<int, float>(
                X,
                Y_buffered,
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
                Y_buffered,
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
    
     void GraphPropagateForwardEdgeComputation(torch::Tensor &src_input_origin,
                                               torch::Tensor &src_input_transferred,
                                               torch::Tensor &dst_output,
                                               std::vector<CSC_segment_pinned *> &graph_partitions,
                                               std::function<torch::Tensor(torch::Tensor&)> PreComputation,
                                               std::function<torch::Tensor(torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&, EdgeNNModule* edgeop)> EdgeComputation)
    {
        int current_layer_size = graph_->gnnctx->layer_size[graph_->rtminfo->curr_layer];
        bool selective = graph_->rtminfo->reduce_comm;
        int layer = graph_->rtminfo->curr_layer;
        torch::Tensor X_cpu=src_input_origin.cpu();
        float *X_buffered=X_cpu.accessor<float,2>().data();
            //printf("done?\n");
            graph_->sync_compute_edge_computation<int, float>(
                src_input_origin,
                src_input_transferred,
                graph_partitions,
                [&](VertexId src) {
                    graph_->emit_buffer(src, X_buffered+(src-graph_->gnnctx->p_v_s)*current_layer_size, current_layer_size);
                },
                [&](torch::Tensor &d_i){
                    return PreComputation(d_i);
                },
                [&](torch::Tensor &s_i,torch::Tensor &s_i_t, torch::Tensor &d_i, torch::Tensor &d_i_t,EdgeNNModule* edgeop){
                    return EdgeComputation(s_i,s_i_t, d_i,d_i_t,edgeop);
                },
                dst_output);
            //printf("done!\n");
        
    }
     
     
    void GraphPropagateBackwardEdgeComputation(torch::Tensor &src_input_origin,
                                               torch::Tensor &dst_grad_input,
                                               torch::Tensor &dst_grad_output,
                                               float* Y_buffered,
                                               std::vector<CSC_segment_pinned *> &graph_partitions,
                                               std::function<torch::Tensor(torch::Tensor&)> PreComputation,
                                               std::function<torch::Tensor(torch::Tensor&,torch::Tensor&,torch::Tensor&,torch::Tensor&,EdgeNNModule* edgeop)> EdgeComputation,
                                               std::function<torch::Tensor(torch::Tensor&,torch::Tensor&,EdgeNNModule* edgeop)> EdgeBackward)
    {
        int current_layer_size = graph_->gnnctx->layer_size[graph_->rtminfo->curr_layer];
        bool selective = graph_->rtminfo->reduce_comm;
        int layer = graph_->rtminfo->curr_layer;
            //printf("done?\n");
            graph_->compute_sync_edge_computation<int, float>(
                dst_grad_input,
                src_input_origin,
                Y_buffered,
                graph_partitions,
                [&](VertexId src, VertexAdjList<Empty> outgoing_adj) {
                    graph_->emit_buffer(src, Y_buffered + (src)*current_layer_size, current_layer_size);
                },
                [&](torch::Tensor &d_i){
                    return PreComputation(d_i);
                },
                [&](torch::Tensor &s_i,torch::Tensor &s_i_t, torch::Tensor &d_i, torch::Tensor &d_i_t,EdgeNNModule* edgeop){
                    return EdgeComputation(s_i,s_i_t, d_i,d_i_t,edgeop);
                },
                [&](torch::Tensor &b_i, torch::Tensor &c_i,EdgeNNModule* edgeop){
                    return EdgeBackward(b_i, c_i, edgeop);
                },
                dst_grad_output);
            //printf("done!\n");
        
    }
    
    
     void Process_GPU_overlap_lite(torch::Tensor &X, float *Y_buffered, torch::Tensor &Y, std::vector<CSC_segment_pinned *> &graph_partitions)
    {
        int current_layer_size = graph_->gnnctx->layer_size[graph_->rtminfo->curr_layer];
        bool selective = graph_->rtminfo->reduce_comm;
        int layer = graph_->rtminfo->curr_layer;

        if (!selective)
        { // original communication
            //printf("done?\n");
            graph_->compute_sync_lite<int, float>(
                X,
                Y_buffered,
                graph_partitions,
                [&](VertexId src, VertexAdjList<Empty> outgoing_adj) { //pull
                    //nodeVector<CURRENT_LAYER_SIZE> sum;
                    //memcpy(sum.data, Y_buffered + (src)*current_layer_size, sizeof(t_v) * CURRENT_LAYER_SIZE);
                    graph_->emit_buffer(src, Y_buffered + (src)*current_layer_size, current_layer_size);
                },
                Y.packed_accessor<float, 2>().data());
            //printf("done!\n");
        }
        else
        { //selective comunication
            graph_->compute_sync_lite<int, float>(
                X,
                Y_buffered,
                graph_partitions,
                [&](VertexId src, VertexAdjList<Empty> outgoing_adj) { //pull
                    if (!graph_->RepVtx[layer]->get_bit(src))
                    {
                        // nodeVector<CURRENT_LAYER_SIZE> sum;
                        // memcpy(sum.data, Y_buffered + (src)*current_layer_size, sizeof(t_v) * CURRENT_LAYER_SIZE);
                        // graph_->emit(src, sum);
                        graph_->emit_buffer(src, Y_buffered + (src)*current_layer_size, current_layer_size);
                    }
                },
                Y.packed_accessor<float, 2>().data());
        }
    }
    
   inline void GraphPropagateForward(torch::Tensor &X, float *Y_buffered, torch::Tensor &Y, std::vector<CSC_segment_pinned *> &graph_partitions)
    {
       Process_GPU_overlap_sync_compute_explict(X,Y_buffered,Y,graph_partitions);
    }

   inline void GraphPropagateBackward(torch::Tensor &X, float *Y_buffered, torch::Tensor &Y, std::vector<CSC_segment_pinned *> &graph_partitions)
    {
       Process_GPU_overlap_lite(X,Y_buffered,Y,graph_partitions);
       //Process_GPU_overlap_sync_compute_explict(X,Y_buffered,Y,graph_partitions);
       
    }

    template <int CURRENT_LAYER_SIZE>
    torch::Tensor Test_Propagate(int local_layer)
    {
        zeroCurr(); //local
        zeroNext(); //local
        zeroValue(local_layer);
        nodeVector<CURRENT_LAYER_SIZE> test;
        for (int i = 0; i < CURRENT_LAYER_SIZE; i++)
        {
            test.data[i] = 1.0;
        }
        int tagg = 0;
        graph_->process_vertices<t_v>( //init  the vertex state.
            [&](VertexId vtx) {
                cpToCurr<CURRENT_LAYER_SIZE>(vtx, test.data); //local
                return (ValueType)1;
            },
            active_);
        graph_->process_edges<int, nodeVector<CURRENT_LAYER_SIZE>>( // For EACH Vertex Processing
            [&](VertexId dst, VertexAdjList<Empty> incoming_adj) {  //pull
                nodeVector<CURRENT_LAYER_SIZE> sum;
                memset(sum.data, 0, sizeof(ValueType) * CURRENT_LAYER_SIZE);
                for (AdjUnit<Empty> *ptr = incoming_adj.begin; ptr != incoming_adj.end; ptr++)
                { //pull model
                    VertexId src = ptr->neighbour;
                    tagg++;
                    for (int i = 0; i < CURRENT_LAYER_SIZE; i++)
                    {
                        sum.data[i] += ((nodeVector<CURRENT_LAYER_SIZE> *)curr_local)[src - start_].data[i]; //local
                    }
                }
                graph_->emit(dst, sum);
            },
            [&](VertexId dst, nodeVector<CURRENT_LAYER_SIZE> msg) {
                addToNext<CURRENT_LAYER_SIZE>(dst, msg.data); //local
                return 0;
            },
            active_);

        graph_->process_vertices<ValueType>( //init the vertex state.
            [&](VertexId vtx) {
                memcpy(this->value_buffer[local_layer] + CURRENT_LAYER_SIZE * (vtx - start_),
                       ((nodeVector<CURRENT_LAYER_SIZE> *)next_local)[vtx - start_].data, sizeof(t_v) * CURRENT_LAYER_SIZE);
                return 0;
            },
            active_);

        value_local[local_layer] = torch::from_blob(value_buffer[local_layer], {rownum_, CURRENT_LAYER_SIZE}, torch::kFloat);
        current_layer = local_layer;
        if (graph_->partition_id == 0)
        {
            int tag = 0;
            for (int i = 0; i < graph_->vertices; i++)
            {
                if (value_buffer[local_layer][i * CURRENT_LAYER_SIZE + 3] - (ValueType)graph_->in_degree_for_backward[i] <= 0.00001)
                    // printf("%f\t%d\n",value_buffer[local_layer][i*CURRENT_LAYER_SIZE+3],graph_->in_degree_for_backward[i]);
                    tag++;
            }
            printf("finish TEST_propagate at layer [%d] validate %d vertices \n", local_layer, tag);
        }
        return value_local[local_layer];
    }
    void setValueFromTensor(torch::Tensor new_tensor)
    {
        pre_value = new_tensor;
        zeroCurr(); //local
    }
    void setValueFromNative(int layer, t_v *data, int offset)
    {
        pre_value = torch::from_blob(data + offset, {rownum_, size_at_layer[layer]}, torch::kFloat);
        zeroCurr(); //local
    }
    void setValueFromNative(t_v *data, int offset)
    {
        pre_value = torch::from_blob(data + offset, {rownum_, MAX_LAYER}, torch::kFloat);
        zeroCurr(); //local
    }
    void setValueFromNative(t_v *data, int offset, int length_)
    {
        pre_value = torch::from_blob(data + offset, {rownum_, length_}, torch::kFloat);
        zeroCurr(); //local
    }

    void setGradFromNative(t_v *data, int offset)
    {
        pre_grad = torch::from_blob(data + offset, {rownum_, MAX_LAYER}, torch::kFloat);
    }
    void setGradFromTensor(torch::Tensor new_tensor)
    {
        pre_grad = new_tensor;
    }
    torch::Tensor v(int local_layer)
    {
        return value_local[local_layer];
    }
    torch::Tensor require_grad()
    {
        return grad;
    }
};

void generate_edge_list_Tensor(Graph<Empty> *graph, std::vector<edge_list *> &graph_partitions, int batch_size)
{
    graph_partitions.clear();
    for (int i = 0; i < graph->graph_shard.size(); i++)
    {
        graph_partitions.push_back(new edge_list);
        graph_partitions[i]->edge_size = graph->graph_shard[i]->numofedges;
        graph_partitions[i]->dst_range[0] = graph->graph_shard[i]->dst_range[0];
        graph_partitions[i]->dst_range[1] = graph->graph_shard[i]->dst_range[1];
        graph_partitions[i]->src_range[0] = graph->graph_shard[i]->src_range[0];
        graph_partitions[i]->src_range[1] = graph->graph_shard[i]->src_range[1];
        //if(graph->partition_id==1)std::cout<<"test_range in generate"<<graph->graph_shard[i]->src_range[0]<<" "<<graph->graph_shard[i]->src_range[1]<<std::endl;
        //if(graph->partition_id==1)std::cout<<"test_range in generate"<<graph->graph_shard[i]->dst_range[0]<<" "<<graph->graph_shard[i]->dst_range[1]<<std::endl;
        graph_partitions[i]->batch_size = batch_size;
        graph_partitions[i]->feature_size = SIZE_LAYER_1;
        graph_partitions[i]->dst = torch::from_blob(graph->graph_shard[i]->dst_delta, {1, graph_partitions[i]->edge_size}, torch::kInt32);
        graph_partitions[i]->src = torch::from_blob(graph->graph_shard[i]->src_delta, {1, graph_partitions[i]->edge_size}, torch::kInt32);
        graph_partitions[i]->weight_buffer = new float[graph_partitions[i]->edge_size];
        //   std::cout<<"generate_edge_list"<<graph_partitions[i]->edge_size<<" "<<std::endl;
        for (int j = 0; j < graph_partitions[i]->edge_size; j++)
        {
            VertexId v_src = graph->graph_shard[i]->src_delta[j];
            VertexId v_dst = graph->graph_shard[i]->dst_delta[j];
            graph_partitions[i]->weight_buffer[j] = (ValueType)std::sqrt(graph->out_degree_for_backward[v_src]) * (ValueType)std::sqrt(graph->in_degree_for_backward[v_dst]);
        }
        graph_partitions[i]->weight = torch::from_blob(graph_partitions[i]->weight_buffer, {1, graph_partitions[i]->edge_size}, torch::kFloat);
    }
    //std::cout<<"generate_edge_list_Tensor"<<graph->graph_shard.size()<<std::endl;
    //std::cout<<"graph_edges "<<graph_partitions[0]->edge_size<<" "<<graph->edges<<std::endl;
}
void generate_CSC_Segment_Tensor(Graph<Empty> *graph, std::vector<CSC_segment *> &graph_partitions, int batch_size, bool overlap = false)
{
    graph_partitions.clear();
    int *tmp_column_offset = new int[graph->vertices + 1];
    memset(tmp_column_offset, 0, sizeof(int) * (graph->vertices + 1));
    std::cout << "shard_size" << graph->graph_shard.size() << " " << std::endl;
    for (int i = 0; i < graph->graph_shard.size(); i++)
    {
        //int i=0;
        graph_partitions.push_back(new CSC_segment);
        memset(tmp_column_offset, 0, sizeof(int) * (graph->vertices + 1));
        graph_partitions[i]->edge_size = graph->graph_shard[i]->numofedges;
        graph_partitions[i]->dst_range[0] = graph->graph_shard[i]->dst_range[0];
        graph_partitions[i]->dst_range[1] = graph->graph_shard[i]->dst_range[1];
        graph_partitions[i]->src_range[0] = graph->graph_shard[i]->src_range[0];
        graph_partitions[i]->src_range[1] = graph->graph_shard[i]->src_range[1];
        int column_offset_size = graph->graph_shard[i]->dst_range[1] - graph->graph_shard[i]->dst_range[0] + 1;
        graph_partitions[i]->batch_size = graph->graph_shard[i]->dst_range[1] - graph->graph_shard[i]->dst_range[0];
        ;
        graph_partitions[i]->feature_size = SIZE_LAYER_1;

        graph_partitions[i]->column_offset = torch::zeros({1, column_offset_size}, torch::kInt32);
        graph_partitions[i]->row_indices = torch::zeros({1, graph_partitions[i]->edge_size}, torch::kInt32);
        graph_partitions[i]->weight_buffer = new float[graph_partitions[i]->edge_size];
        //   std::cout<<"generate_edge_list"<<column_offset_size<<" "<<std::endl;
        for (int j = 0; j < graph_partitions[i]->edge_size; j++)
        {
            VertexId v_src = graph->graph_shard[i]->src_delta[j];
            VertexId v_dst = graph->graph_shard[i]->dst_delta[j] - graph_partitions[i]->dst_range[0];
            tmp_column_offset[v_dst + 1] += 1;
            //graph_partitions[i]->weight_buffer[j]=(ValueType)std::sqrt(graph->out_degree_for_backward[v_src])*(ValueType)std::sqrt(graph->in_degree_for_backward[v_dst]);
        }
        for (int j = 0; j < column_offset_size - 1; j++)
        {
            tmp_column_offset[j + 1] += tmp_column_offset[j];
            graph_partitions[i]->column_offset.accessor<int, 2>().data()[j + 1] = tmp_column_offset[j + 1];
        }

        for (int j = 0; j < graph_partitions[i]->edge_size; j++)
        {
            //if(graph->partition_id==0)std::cout<<"After j edges: "<<j<<std::endl;
            VertexId v_src = graph->graph_shard[i]->src_delta[j];
            VertexId v_dst = graph->graph_shard[i]->dst_delta[j] - graph_partitions[i]->dst_range[0];
            graph_partitions[i]->row_indices.accessor<int, 2>().data()[tmp_column_offset[v_dst]] = v_src;
            graph_partitions[i]->weight_buffer[tmp_column_offset[v_dst]++] = 1; //(ValueType)std::sqrt(graph->out_degree_for_backward[v_src]) * (ValueType)std::sqrt(graph->in_degree_for_backward[v_dst]);
        }
        graph_partitions[i]->edge_weight = torch::from_blob(graph_partitions[i]->weight_buffer, {1, graph_partitions[i]->edge_size}, torch::kFloat);
    }
    //   printf("compute has started %d\n",400);
    delete[] tmp_column_offset;
}

void generate_CSC_Segment_Tensor_pinned(Graph<Empty> *graph, std::vector<CSC_segment_pinned *> &graph_partitions, bool overlap = false)
{
    graph_partitions.clear();
    int *tmp_column_offset = new int[graph->vertices + 1];
    memset(tmp_column_offset, 0, sizeof(int) * (graph->vertices + 1));
    //std::cout << graph->graph_shard.size() << " " << std::endl;
    for (int i = 0; i < graph->graph_shard.size(); i++)
    {
        //int i=0;
        graph_partitions.push_back(new CSC_segment_pinned);
        memset(tmp_column_offset, 0, sizeof(int) * (graph->vertices + 1));
        graph_partitions[i]->edge_size = graph->graph_shard[i]->numofedges;
        graph_partitions[i]->dst_range[0] = graph->graph_shard[i]->dst_range[0];
        graph_partitions[i]->dst_range[1] = graph->graph_shard[i]->dst_range[1];
        graph_partitions[i]->src_range[0] = graph->graph_shard[i]->src_range[0];
        graph_partitions[i]->src_range[1] = graph->graph_shard[i]->src_range[1];
        long column_offset_size = graph->graph_shard[i]->dst_range[1] - graph->graph_shard[i]->dst_range[0] + 1;
        
        graph_partitions[i]->batch_size = graph->graph_shard[i]->dst_range[1] - graph->graph_shard[i]->dst_range[0];
        graph_partitions[i]->feature_size = graph->gnnctx->layer_size[0];
        graph_partitions[i]->column_offset = (VertexId *)cudaMallocPinned(column_offset_size * sizeof(VertexId));                 //torch::zeros({1,column_offset_size},torch::kInt32);
        graph_partitions[i]->row_indices = (VertexId *)cudaMallocPinned((graph_partitions[i]->edge_size + 1) * sizeof(VertexId)); //torch::zeros({1,graph_partitions[i]->edge_size},torch::kInt32);
        graph_partitions[i]->destination = (long *)cudaMallocPinned((graph_partitions[i]->edge_size + 1) * sizeof(long));
        graph_partitions[i]->source      = (long *)cudaMallocPinned((graph_partitions[i]->edge_size + 1) * sizeof(long));
        graph_partitions[i]->edge_weight = (float *)cudaMallocPinned((graph_partitions[i]->edge_size + 1) * sizeof(VertexId));

        graph_partitions[i]->column_offset_gpu = (VertexId *)getDevicePointer(graph_partitions[i]->column_offset);
        graph_partitions[i]->row_indices_gpu = (VertexId *)getDevicePointer(graph_partitions[i]->row_indices);
        graph_partitions[i]->edge_weight_gpu = (float *)getDevicePointer(graph_partitions[i]->edge_weight);

        //   std::cout<<"generate_edge_list"<<column_offset_size<<" "<<std::endl;
        for (int j = 0; j < graph_partitions[i]->edge_size; j++)
        {
            VertexId v_src = graph->graph_shard[i]->src_delta[j];
            VertexId v_dst = graph->graph_shard[i]->dst_delta[j] - graph_partitions[i]->dst_range[0];
            tmp_column_offset[v_dst + 1] += 1;
            //graph_partitions[i]->weight_buffer[j]=(ValueType)std::sqrt(graph->out_degree_for_backward[v_src])*(ValueType)std::sqrt(graph->in_degree_for_backward[v_dst]);
        }
        for (int j = 0; j < column_offset_size - 1; j++)
        {
            tmp_column_offset[j + 1] += tmp_column_offset[j];
            graph_partitions[i]->column_offset[j + 1] = tmp_column_offset[j + 1];
        }

        for (int j = 0; j < graph_partitions[i]->edge_size; j++)
        {
            //if(graph->partition_id==0)std::cout<<"After j edges: "<<j<<std::endl;
            VertexId v_src = graph->graph_shard[i]->src_delta[j];
            VertexId v_dst = graph->graph_shard[i]->dst_delta[j] - graph_partitions[i]->dst_range[0];
            graph_partitions[i]->row_indices[tmp_column_offset[v_dst]] = v_src;
            graph_partitions[i]->source[tmp_column_offset[v_dst]] = (long)v_src;
            graph_partitions[i]->destination[tmp_column_offset[v_dst]] = (long)(v_dst+ graph_partitions[i]->dst_range[0]);
            graph_partitions[i]->edge_weight[tmp_column_offset[v_dst]++] = 1; // (ValueType)std::sqrt(graph->out_degree_for_backward[v_src]) * (ValueType)std::sqrt(graph->in_degree_for_backward[v_dst]);
        }
    }
    if (overlap)
    {
        int max_batch_size = 0;
        for (int i = 0; i < graph_partitions.size(); i++)
        {
            max_batch_size = std::max(max_batch_size, graph_partitions[i]->batch_size);
        }
        graph->output_gpu_buffered = torch::zeros({max_batch_size, graph->gnnctx->max_layer},
                                                  at::TensorOptions().device_index(0).dtype(torch::kFloat));
    }
    delete[] tmp_column_offset;
    if (graph->partition_id == 0)
        printf("GNNmini::Preprocessing[Prepare Backward CSC Edges for GPU]\n");
}

void generate_Forward_Segment_Tensor_pinned(Graph<Empty> *graph, std::vector<CSC_segment_pinned *> &graph_partitions, bool overlap = false)
{
    graph_partitions.clear();
    int *tmp_column_offset = new int[graph->vertices + 1];
    int *tmp_row_offset=new int[graph->vertices + 1];
    memset(tmp_column_offset, 0, sizeof(int) * (graph->vertices + 1));///
    memset(tmp_row_offset, 0, sizeof(int) * (graph->vertices + 1));///
    //std::cout << graph->graph_shard.size() << " " << std::endl;
    for (int i = 0; i < graph->graph_shard.size(); i++)
    {
        //int i=0;
        graph_partitions.push_back(new CSC_segment_pinned);
        memset(tmp_column_offset, 0, sizeof(int) * (graph->vertices + 1));
        memset(tmp_row_offset, 0, sizeof(int) * (graph->vertices + 1));
        graph_partitions[i]->edge_size = graph->graph_shard[i]->numofedges;
        graph_partitions[i]->dst_range[0] = graph->graph_shard[i]->src_range[0];
        graph_partitions[i]->dst_range[1] = graph->graph_shard[i]->src_range[1];
        graph_partitions[i]->src_range[0] = graph->graph_shard[i]->dst_range[0];
        graph_partitions[i]->src_range[1] = graph->graph_shard[i]->dst_range[1];
        long column_offset_size = graph_partitions[i]->dst_range[1] - graph_partitions[i]->dst_range[0] + 1;
        
        long row_offset_size = graph_partitions[i]->src_range[1] - graph_partitions[i]->src_range[0] + 1;///
         
        graph_partitions[i]->batch_size = column_offset_size-1;
        graph_partitions[i]->batch_size_forward = column_offset_size-1;
        graph_partitions[i]->batch_size_backward = row_offset_size-1;
        
        graph_partitions[i]->feature_size = graph->gnnctx->layer_size[0];
        graph_partitions[i]->column_offset = (VertexId *)cudaMallocPinned(column_offset_size * sizeof(VertexId));                 //torch::zeros({1,column_offset_size},torch::kInt32);
        graph_partitions[i]->row_indices = (VertexId *)cudaMallocPinned((graph_partitions[i]->edge_size + 1) * sizeof(VertexId));
        graph_partitions[i]->edge_weight = (float *)cudaMallocPinned((graph_partitions[i]->edge_size + 1) * sizeof(VertexId));
        
        graph_partitions[i]->row_offset = (VertexId *)cudaMallocPinned(row_offset_size * sizeof(VertexId));///
        graph_partitions[i]->column_indices = (VertexId *)cudaMallocPinned((graph_partitions[i]->edge_size + 1) * sizeof(VertexId));///
        graph_partitions[i]->edge_weight_backward = (float *)cudaMallocPinned((graph_partitions[i]->edge_size + 1) * sizeof(VertexId));///
        
        
        //torch::zeros({1,graph_partitions[i]->edge_size},torch::kInt32);
        graph_partitions[i]->destination = (long *)cudaMallocPinned((graph_partitions[i]->edge_size + 1) * sizeof(long));
        graph_partitions[i]->source      = (long *)cudaMallocPinned((graph_partitions[i]->edge_size + 1) * sizeof(long));

        graph_partitions[i]->column_offset_gpu = (VertexId *)getDevicePointer(graph_partitions[i]->column_offset);
        graph_partitions[i]->row_indices_gpu = (VertexId *)getDevicePointer(graph_partitions[i]->row_indices);
        graph_partitions[i]->edge_weight_gpu = (float *)getDevicePointer(graph_partitions[i]->edge_weight);
        
        graph_partitions[i]->row_offset_gpu = (VertexId *)getDevicePointer(graph_partitions[i]->row_offset);///
        graph_partitions[i]->column_indices_gpu = (VertexId *)getDevicePointer(graph_partitions[i]->column_indices);///
        graph_partitions[i]->edge_weight_backward_gpu = (float *)getDevicePointer(graph_partitions[i]->edge_weight_backward);/// 
        
        
        graph_partitions[i]->source_gpu = (long *)getDevicePointer(graph_partitions[i]->source);///
        graph_partitions[i]->destination_gpu = (long *)getDevicePointer(graph_partitions[i]->destination);///

        //   std::cout<<"generate_edge_list"<<column_offset_size<<" "<<std::endl;
        for (int j = 0; j < graph_partitions[i]->edge_size; j++)
        {
            VertexId v_dst_m = graph->graph_shard[i]->src_delta[j];
            VertexId v_src_m = graph->graph_shard[i]->dst_delta[j];
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
            VertexId v_dst_m = graph->graph_shard[i]->src_delta[j];
            VertexId v_src_m = graph->graph_shard[i]->dst_delta[j];
            VertexId v_dst   = v_dst_m-graph_partitions[i]->dst_range[0];
            VertexId v_src   = v_src_m-graph_partitions[i]->src_range[0];
            
            
            graph_partitions[i]->source[tmp_column_offset[v_dst]] = (long)(v_src_m);
            graph_partitions[i]->destination[tmp_column_offset[v_dst]] = (long)(v_dst_m);
            
            graph_partitions[i]->row_indices[tmp_column_offset[v_dst]] = v_src_m;
            graph_partitions[i]->edge_weight[tmp_column_offset[v_dst]++] = 1;
            
            graph_partitions[i]->column_indices[tmp_row_offset[v_src]] = v_dst_m;///
           graph_partitions[i]->edge_weight_backward[tmp_row_offset[v_src]++] = 1; ///

        
        }
    }
    
    
    //if (overlap)
    {
        int max_batch_size = 0;
        for (int i = 0; i < graph_partitions.size(); i++)
        {
            max_batch_size = std::max(max_batch_size, graph_partitions[i]->batch_size);
        }
        graph->output_gpu_buffered = torch::zeros({max_batch_size, graph->gnnctx->max_layer},
                                                  at::TensorOptions().device_index(0).dtype(torch::kFloat));
    }
    delete[] tmp_column_offset;
    delete[] tmp_row_offset;
    if (graph->partition_id == 0)
        printf("GNNmini::Preprocessing[Prepare Forward CSC Edges for GPU]\n");
}


void propagate_forward_gpu_shard(Graph<Empty> *graph, torch::Tensor input_cpu, torch::Tensor &output_cpu,
                                 std::vector<edge_list *> &graph_partitions, int feature_size)
{
    torch::Tensor output_gpu;
    torch::Tensor input_gpu;
    torch::Tensor weight_gpu;
    int src_blocks = graph->owned_vertices / graph_partitions[0]->batch_size + (int)((graph->owned_vertices % graph_partitions[0]->batch_size) > 0);
    int dst_blocks = graph_partitions.size() / src_blocks;
    std::cout << src_blocks << " " << dst_blocks << std::endl;
    output_cpu.zero_();
    output_gpu = torch::zeros({graph_partitions[0]->batch_size, feature_size},
                              at::TensorOptions().device_index(0).dtype(torch::kFloat));
    //input_gpu=torch::zeros({graph_partitions[0]->batch_size,feature_size},
    //                                        at::TensorOptions().device_index(0).dtype(torch::kFloat));

    for (int i = 0; i < dst_blocks; i++)
    {
        output_gpu.zero_();
        for (int j = 0; j < src_blocks; j++)
        {
            double movein_time = 0;
            movein_time -= MPI_Wtime();
            torch::Tensor input_gpu = input_cpu.slice(0, graph_partitions[i + j * dst_blocks]->src_range[0] - graph->partition_offset[graph->partition_id],
                                                      graph_partitions[i + j * dst_blocks]->src_range[1] - graph->partition_offset[graph->partition_id], 1)
                                          .cuda();

            weight_gpu = graph_partitions[i + j * dst_blocks]->weight.cuda();
            torch::Tensor src_l = graph_partitions[i + j * dst_blocks]->src.cuda();
            torch::Tensor dst_l = graph_partitions[i + j * dst_blocks]->dst.cuda();
            VertexId src_start = graph_partitions[i + j * dst_blocks]->src_range[0];
            VertexId src_end = graph_partitions[i + j * dst_blocks]->src_range[1];
            VertexId dst_start = graph_partitions[i + j * dst_blocks]->dst_range[0];
            VertexId dst_end = graph_partitions[i + j * dst_blocks]->dst_range[1];
            movein_time += MPI_Wtime();
            graph->all_movein_time += movein_time;
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
            double kernel_time = 0;
            kernel_time -= MPI_Wtime();
            forward_on_GPU(input_gpu.packed_accessor<float, 2>().data(), output_gpu.packed_accessor<float, 2>().data(), weight_gpu.packed_accessor<float, 2>().data(), //data
                           (uint32_t *)src_l.packed_accessor<int, 2>().data(), (uint32_t *)dst_l.packed_accessor<int, 2>().data(),                                     //graph
                           src_start, src_end, dst_start, dst_end,
                           graph_partitions[i + j * dst_blocks]->edge_size, graph_partitions[i + j * dst_blocks]->batch_size, feature_size);
            kernel_time += MPI_Wtime();
            graph->all_kernel_time += kernel_time;
            //because the src_range has quite different meaning in GPU and CPU
        }
        double moveout_time = 0;
        moveout_time -= MPI_Wtime();
        torch::Tensor output_slice = output_cpu.slice(0, graph_partitions[i]->dst_range[0], graph_partitions[i]->dst_range[1], 1);
        if (i != (dst_blocks - 1))
        {
            move_result_out(output_cpu.accessor<float, 2>().data() + (graph_partitions[i]->dst_range[0] * feature_size), output_gpu.packed_accessor<float, 2>().data(), graph_partitions[i]->dst_range[0], graph_partitions[i]->dst_range[1], feature_size);
        }
        else
        {
            move_result_out(output_cpu.accessor<float, 2>().data() + (graph_partitions[i]->dst_range[0] * feature_size), output_gpu.packed_accessor<float, 2>().data(), graph_partitions[i]->dst_range[0], graph_partitions[i]->dst_range[1], feature_size);
        }
        moveout_time += MPI_Wtime();
        graph->all_moveout_time += moveout_time;
    }
}

void move_embedding_out(torch::Tensor &t_host, torch::Tensor &t_device, int src, int dst, int feature_size)
{
    move_result_out(t_host.accessor<float, 2>().data() + src * feature_size, t_device.packed_accessor<float, 2>().data(), src, dst, feature_size);
}

void propagate_forward_gpu_shard_CSC(Graph<Empty> *graph, torch::Tensor input_cpu, torch::Tensor &output_cpu,
                                     std::vector<CSC_segment *> &graph_partitions, int feature_size)
{
    torch::Tensor output_gpu;
    torch::Tensor input_gpu;
    torch::Tensor weight_gpu;
    int src_blocks = graph->owned_vertices / graph_partitions[0]->batch_size + (int)((graph->owned_vertices % graph_partitions[0]->batch_size) > 0);
    int dst_blocks = graph_partitions.size() / src_blocks;
    //std::cout<<src_blocks<<" "<<dst_blocks<<std::endl;
    //output_cpu.zero_();
    output_gpu = torch::zeros({graph_partitions[0]->batch_size, feature_size},
                              at::TensorOptions().device_index(0).dtype(torch::kFloat));

    for (int i = 0; i < dst_blocks; i++)
    {
        output_gpu.zero_();
        for (int j = 0; j < src_blocks; j++)
        {
            double movein_time = 0;
            movein_time -= get_time();
            torch::Tensor input_gpu = input_cpu.slice(0, graph_partitions[i + j * dst_blocks]->src_range[0] - graph->partition_offset[graph->partition_id],
                                                      graph_partitions[i + j * dst_blocks]->src_range[1] - graph->partition_offset[graph->partition_id], 1)
                                          .cuda();
            //if(graph->partition_id==0) std::cout<<"we fail on worker0"<<" "<<graph_partitions[j]->src_range[0]<<" "<<graph_partitions[j]->src_range[1]<<" "<<graph->partition_offset[graph->partition_id]<<std::endl;

            weight_gpu = graph_partitions[i + j * dst_blocks]->edge_weight.cuda();
            torch::Tensor row_indices = graph_partitions[i + j * dst_blocks]->row_indices.cuda();
            torch::Tensor column_offset = graph_partitions[i + j * dst_blocks]->column_offset.cuda();
            VertexId src_start = graph_partitions[i + j * dst_blocks]->src_range[0];
            VertexId src_end = graph_partitions[i + j * dst_blocks]->src_range[1];
            VertexId dst_start = graph_partitions[i + j * dst_blocks]->dst_range[0];
            VertexId dst_end = graph_partitions[i + j * dst_blocks]->dst_range[1];
            movein_time += get_time();
            graph->all_movein_time += movein_time;
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
            double kernel_time = 0;
            kernel_time -= get_time();
            Gather_By_Dst_From_Src(input_gpu.packed_accessor<float, 2>().data(), output_gpu.packed_accessor<float, 2>().data(), weight_gpu.packed_accessor<float, 2>().data(), //data
                               (uint32_t *)row_indices.packed_accessor<int, 2>().data(), (uint32_t *)column_offset.packed_accessor<int, 2>().data(),                       //graph
                               src_start, src_end, dst_start, dst_end,
                               graph_partitions[i + j * dst_blocks]->edge_size, graph_partitions[i + j * dst_blocks]->batch_size, feature_size, true);
            kernel_time += get_time();
            graph->all_kernel_time += kernel_time;
            //because the src_range has quite different meaning in GPU and CPU
        }
        double moveout_time = 0;
        moveout_time -= get_time();
        torch::Tensor output_slice = output_cpu.slice(0, graph_partitions[i]->dst_range[0], graph_partitions[i]->dst_range[1], 1);
        if (i != (dst_blocks - 1))
        {
            //          output_slice+=output_gpu.cpu();
            move_result_out(output_cpu.accessor<float, 2>().data() + (graph_partitions[i]->dst_range[0] * feature_size), output_gpu.packed_accessor<float, 2>().data(), graph_partitions[i]->dst_range[0], graph_partitions[i]->dst_range[1], feature_size, true);
        }
        else
        {
            //            output_slice+=output_gpu.slice(0,0,output_slice.size(0),1).cpu();
            move_result_out(output_cpu.accessor<float, 2>().data() + (graph_partitions[i]->dst_range[0] * feature_size), output_gpu.packed_accessor<float, 2>().data(), graph_partitions[i]->dst_range[0], graph_partitions[i]->dst_range[1], feature_size, true);
            //std::cout<<" \n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ "<<std::endl;
        }
        moveout_time += get_time();
        graph->all_moveout_time += moveout_time;
    }
}

void propagate_forward_gpu_shard_CSC_test(Graph<Empty> *graph, torch::Tensor input_cpu, torch::Tensor &output_cpu,
                                          std::vector<CSC_segment *> &graph_partitions, int feature_size, int batch_size)
{

    torch::Tensor output_gpu;
    torch::Tensor input_gpu;
    torch::Tensor weight_gpu;
    int src_blocks = 1;
    int dst_blocks = graph_partitions.size();
    //std::cout<<src_blocks<<" "<<dst_blocks<<std::endl;
    //output_cpu.zero_();
    output_gpu = torch::zeros({batch_size, feature_size},
                              at::TensorOptions().device_index(0).dtype(torch::kFloat));
    double movein_time = 0;
    movein_time -= get_time();
    input_gpu = input_cpu.cuda();
    movein_time += get_time();
    graph->all_movein_time += movein_time;
    for (int i = 0; i < dst_blocks; i++)
    {
        output_gpu.zero_();
        double movein_time = 0;
        movein_time -= get_time();
        //if(graph->partition_id==0) std::cout<<"we fail on worker0"<<" "<<graph_partitions[j]->src_range[0]<<" "<<graph_partitions[j]->src_range[1]<<" "<<graph->partition_offset[graph->partition_id]<<std::endl;

        weight_gpu = graph_partitions[i]->edge_weight.cuda();
        torch::Tensor row_indices = graph_partitions[i]->row_indices.cuda();
        torch::Tensor column_offset = graph_partitions[i]->column_offset.cuda();
        VertexId src_start = graph_partitions[i]->src_range[0];
        VertexId src_end = graph_partitions[i]->src_range[1];
        VertexId dst_start = graph_partitions[i]->dst_range[0];
        VertexId dst_end = graph_partitions[i]->dst_range[1];
        movein_time += get_time();
        graph->all_movein_time += movein_time;
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
        double kernel_time = 0;
        kernel_time -= get_time();
        Gather_By_Dst_From_Src(input_gpu.packed_accessor<float, 2>().data(), output_gpu.packed_accessor<float, 2>().data(), weight_gpu.packed_accessor<float, 2>().data(), //data
                           (uint32_t *)row_indices.packed_accessor<int, 2>().data(), (uint32_t *)column_offset.packed_accessor<int, 2>().data(),                       //graph
                           src_start, src_end, dst_start, dst_end,
                           graph_partitions[i]->edge_size, graph_partitions[i]->batch_size, feature_size, true);
        kernel_time += get_time();
        graph->all_kernel_time += kernel_time;
        //because the src_range has quite different meaning in GPU and CPU

        double moveout_time = 0;
        moveout_time -= get_time();
        torch::Tensor output_slice = output_cpu.slice(0, graph_partitions[i]->dst_range[0], graph_partitions[i]->dst_range[1], 1);
        if (i != (dst_blocks - 1))
        {
            //          output_slice+=output_gpu.cpu();
            move_result_out(output_cpu.accessor<float, 2>().data() + (graph_partitions[i]->dst_range[0] * feature_size), output_gpu.packed_accessor<float, 2>().data(), graph_partitions[i]->dst_range[0], graph_partitions[i]->dst_range[1], feature_size, true);
        }
        else
        {
            //            output_slice+=output_gpu.slice(0,0,output_slice.size(0),1).cpu();
            move_result_out(output_cpu.accessor<float, 2>().data() + (graph_partitions[i]->dst_range[0] * feature_size), output_gpu.packed_accessor<float, 2>().data(), graph_partitions[i]->dst_range[0], graph_partitions[i]->dst_range[1], feature_size, true);
            //std::cout<<" \n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ "<<std::endl;
        }
        moveout_time += get_time();
        graph->all_moveout_time += moveout_time;
    }
}

void generate_weight_and_csr(Graph<Empty> *graph, VertexSubset *active,
                             VertexId *incoming_adj_index, VertexId *incoming_adj_index_backward,
                             float *weight, float *weight_backward, torch::Tensor &weight_forward_T, torch::Tensor &weight_backward_T)
{
    graph->process_vertices<ValueType>( //init  the vertex state.
        [&](VertexId vtx) {
            // graph->in
            incoming_adj_index[vtx] = (VertexId)graph->incoming_adj_index[0][vtx];
            incoming_adj_index_backward[vtx] = (VertexId)graph->incoming_adj_index_backward[0][vtx];
            for (int i = graph->incoming_adj_index[0][vtx]; i < graph->incoming_adj_index[0][vtx + 1]; i++)
            {
                VertexId dst = graph->incoming_adj_list[0][i].neighbour;
                weight[i] = (ValueType)std::sqrt(graph->in_degree_for_backward[vtx]) * (ValueType)std::sqrt(graph->out_degree_for_backward[dst]);
            }
            for (int i = graph->incoming_adj_index_backward[0][vtx]; i < graph->incoming_adj_index_backward[0][vtx + 1]; i++)
            {
                VertexId dst = graph->incoming_adj_list_backward[0][i].neighbour;
                weight_backward[i] = (ValueType)std::sqrt(graph->out_degree_for_backward[vtx]) * (ValueType)std::sqrt(graph->in_degree_for_backward[dst]);
            }

            return (ValueType)1;
        },
        active);
    incoming_adj_index[graph->vertices] = (VertexId)graph->incoming_adj_index[0][graph->vertices];
    incoming_adj_index_backward[graph->vertices] = (VertexId)graph->incoming_adj_index_backward[0][graph->vertices];
    weight_forward_T = torch::from_blob(weight, {graph->edges + 1, 1}, torch::kFloat).cuda();
    weight_backward_T = torch::from_blob(weight_backward, {graph->edges + 1, 1}, torch::kFloat).cuda();
}
void generate_weight_and_csr_numa(Graph<Empty> *graph, VertexSubset *active,
                                  VertexId *incoming_adj_index, VertexId *incoming_adj_index_backward,
                                  float *weight, float *weight_backward, torch::Tensor &weight_forward_T, torch::Tensor &weight_backward_T)
{

    int write_position = 0;
    int write_position_edge = 0;
    int write_position_backward = 0;
    int write_position_backward_edge = 0;
    for (int v_i = 0; v_i < graph->vertices; v_i++)
    {
        incoming_adj_index[v_i] = write_position;
        incoming_adj_index_backward[v_i] = write_position_backward;
        for (int s_i = 0; s_i < graph->sockets; s_i++)
        {
            write_position += graph->incoming_adj_index[s_i][v_i + 1] - graph->incoming_adj_index[s_i][v_i];
            write_position_backward += graph->incoming_adj_index_backward[s_i][v_i + 1] - graph->incoming_adj_index_backward[s_i][v_i];
            for (int i = graph->incoming_adj_index[s_i][v_i]; i < graph->incoming_adj_index[s_i][v_i + 1]; i++)
            {
                VertexId dst = graph->incoming_adj_list[s_i][i].neighbour;
                weight[write_position_edge++] = (ValueType)std::sqrt(graph->in_degree_for_backward[v_i]) * (ValueType)std::sqrt(graph->out_degree_for_backward[dst]);
            }
            for (int i = graph->incoming_adj_index_backward[s_i][v_i]; i < graph->incoming_adj_index_backward[s_i][v_i + 1]; i++)
            {
                VertexId dst = graph->incoming_adj_list_backward[s_i][i].neighbour;
                weight_backward[write_position_backward_edge++] = (ValueType)std::sqrt(graph->out_degree_for_backward[v_i]) * (ValueType)std::sqrt(graph->in_degree_for_backward[dst]);
            }
        }
    }
    incoming_adj_index[graph->vertices] = write_position;
    incoming_adj_index_backward[graph->vertices] = write_position_backward;
    weight_forward_T = torch::from_blob(weight, {graph->edges + 1, 1}, torch::kFloat).cuda();
    weight_backward_T = torch::from_blob(weight_backward, {graph->edges + 1, 1}, torch::kFloat).cuda();
}

/*
void inference(torch::Tensor tt_cpu, Graph<Empty> *graph, Embeddings<ValueType, long> *embedding,
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
        torch::Tensor test = torch::from_blob(&(embedding->con[k].att[0]), {1, SIZE_LAYER_1});
        torch::Tensor final_ = torch::relu(test.mm(Gnn_v1->W.cpu())).mm(Gnn_v2->W.cpu()).log_softmax(1);
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

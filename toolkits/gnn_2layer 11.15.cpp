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
#include <map>
#include "core/graph.hpp"
#include <unistd.h>
#include <math.h>
#include "torch/torch.h"
#include "torch/csrc/autograd/generated/variable_factories.h"
#include "torch/nn/module.h"
const double d = (double) 0.8;
#define VECTOR_LENGTH 4
#define WEIGHT_ROW 4
#define WEIGHT_COL 4
class Network;
class GnnUnit;
typedef struct factor {
    float data[VECTOR_LENGTH];
} nodeVector;

typedef struct factor2 {
    float weight[WEIGHT_ROW][WEIGHT_COL];
} weightVector;

class Network {
public:
    float* recv_buffer;
    float* buffer;
    int worknum;
    int workid = -1;
    int weight_row=0;
    int weight_col=0;

    Network(Graph<Empty> * graph,int weight_row_,int weight_col_) {
        worknum = graph->partitions;
        workid = graph->partition_id;
        weight_row=weight_row_;
        weight_col=weight_col_;
        if (graph->partition_id == 0) {
            recv_buffer = new float[worknum * weight_row * weight_col];
        }
        buffer = new float[weight_row * weight_col];
    }
    void setWsize(int weight_row_,int weight_col_){
        weight_row=weight_row_;
        weight_col=weight_col_;
        realloc(buffer,weight_row*weight_col*sizeof(float));
    }
    void wrtWtoBuff(weightVector *w) {
        memcpy(buffer, w->weight, sizeof (float)*weight_row * weight_col);
       // memcpy(buffer, weight, sizeof (float)*WEIGHT_ROW * WEIGHT_COL);
    }
    void wrtBuffertoBuff(float *buffer_) {
        memcpy(buffer, buffer_, sizeof (float)*weight_row * weight_col);
       // memcpy(buffer, weight, sizeof (float)*WEIGHT_ROW * WEIGHT_COL);
    }

    void gatherW() {
        if (workid == 0) {
            //接收数组
            memset(recv_buffer, 0, sizeof (float)*(worknum) * weight_row * weight_col);
            memcpy(recv_buffer, buffer, sizeof (float)*weight_row * weight_col);

            for (int i = 1; i < (worknum); i++) {
                MPI_Recv(recv_buffer + i * weight_row*weight_col, weight_row*weight_col, MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        } else {
            //发送数组
            MPI_Send(buffer, weight_row*weight_col, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        }
    }

    void computeW() {
        if (workid == 0) {
            for (int i = 1; i < worknum; i++) {
                for (int j = 0; j < weight_row * weight_col; j++) {
                    recv_buffer[j] = recv_buffer[j] + recv_buffer[j + i * weight_row * weight_col];
                }
            }
          //  printf("\n");
            for (int i = 0; i < weight_row; i++) {
                for (int j = 0; j < weight_col; j++) {
                    recv_buffer[weight_col * i + j] /= worknum;
         //           printf("%f\t", recv_buffer[4 * i + j]);
                }
           //     printf("\n");
            }

        }
    }

    void broadcastW(float* buffer_) {
        memcpy(buffer, buffer_, sizeof (float)*weight_row * weight_col);
        if (workid == 0) {
            for (int i = 1; i < (worknum); i++) {
                MPI_Send(buffer, weight_row*weight_col, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
            }

        }
        else {
            MPI_Recv(buffer, weight_row*weight_col, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    void broadcastW() {
      //  memcpy(buffer, recv_buffer, sizeof (float)*WEIGHT_ROW * WEIGHT_COL);
        if (workid == 0) {
            for (int i = 1; i < (worknum); i++) {
                MPI_Send(buffer, weight_row*weight_col, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
            }

        }
        else {
            MPI_Recv(buffer, weight_row*weight_col, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

    }

};

struct GnnUnit : torch::nn::Module {
    torch::Tensor W;
    float *W_from;

    GnnUnit(size_t w, size_t h) {
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
        x = torch::log_softmax(x,1);
        //x=torch::relu(x);
        return x;
    }

    torch::Tensor forward2(torch::Tensor x) {
    //    auto tmp_acc_c = W.accessor<float, 2>();
     //   x = x.mm(W);
        return torch::sigmoid(x);
    }
};

template <typename T_v, typename T_l>
class Embeddings {
public:

    Embeddings() {

    }
    T_v* curr_v = NULL;
//    T_v* next_v = NULL;
    T_v** local_next= NULL;
    T_v* partial_grad=NULL;
    GnnUnit *Gnn_v1 = NULL;
    GnnUnit *Gnn_v2 =NULL;
    T_l *label = NULL;
    weightVector * Weight = NULL;
    T_v* local_grad=NULL;
    T_v* aggre_grad=NULL;
    

    nodeVector * curr = NULL;
    nodeVector * next = NULL;
    
    int rownum;
    int start;

    void init(Graph<Empty>* graph) {
        curr_v = new float [graph->vertices*VECTOR_LENGTH];//graph->alloc_vertex_array<float>(VECTOR_LENGTH);
 //       next_v = graph->alloc_vertex_array<float>(VECTOR_LENGTH);
        partial_grad= graph->alloc_vertex_array<float>(VECTOR_LENGTH);
        Weight = new weightVector(); // reuse
        label = graph->alloc_vertex_array<T_l>();
        curr = graph->alloc_vertex_array<nodeVector>();
        next = graph->alloc_vertex_array<nodeVector>();
        Gnn_v1 = new GnnUnit(WEIGHT_ROW, WEIGHT_COL);
        Gnn_v2 =new GnnUnit(WEIGHT_ROW, WEIGHT_COL);
       /* new Next_v2 for GNN*/
        local_next=new float*[2];
        local_next[0]=graph->alloc_vertex_array<float>(VECTOR_LENGTH);
        local_next[1]=graph->alloc_vertex_array<float>(VECTOR_LENGTH);
        local_grad=new float[WEIGHT_ROW*WEIGHT_COL];
        rownum = (graph->partition_offset[graph->partition_id + 1] - graph->partition_offset[graph->partition_id]);
        start = VECTOR_LENGTH * (graph->partition_offset[graph->partition_id]);
        aggre_grad=new float[VECTOR_LENGTH*VECTOR_LENGTH];
        
    }
    void cpToPartialGradVFrom(int index, float* from) {
        memcpy(partial_grad + index*VECTOR_LENGTH, from, VECTOR_LENGTH * sizeof (float));
        //  memcpy(curr[index].data,from,VECTOR_LENGTH*sizeof(float));

    }
    void initPartialGradWith( int index, float with) {
        for (int i = 0; i < VECTOR_LENGTH; i++) {
            *(partial_grad + index * VECTOR_LENGTH + i) = with;
            //   curr[index].data[i] = (float)with;
        }
    }
    
    void initCurrWith(int index, float with) {
        for (int i = 0; i < VECTOR_LENGTH; i++) {
            //   *(curr_v+index*VECTOR_LENGTH+i)=with;
            curr[index].data[i] = (float) with;
        }
    }

    void initNextWith(int index, float with) {
        for (int i = 0; i < VECTOR_LENGTH; i++) {
            //   *(curr_v+index*VECTOR_LENGTH+i)=with;
            next[index].data[i] = (float) with;
        }
    }

    void initLocalNextWith(int layer, int index, float with) {
        for (int i = 0; i < VECTOR_LENGTH; i++) {
            *(local_next[layer] + index * VECTOR_LENGTH + i) = with;
            //   curr[index].data[i] = (float)with;
        }
    }

    void cpToCurrFrom(int index, float* from) {
        // memcpy(curr_v+index*VECTOR_LENGTH,from,VECTOR_LENGTH*sizeof(float));
        memcpy(curr[index].data, from, VECTOR_LENGTH * sizeof (float));

    }

    void cpToNextFrom(int index, float* from) {
        // memcpy(curr_v+index*VECTOR_LENGTH,from,VECTOR_LENGTH*sizeof(float));
        memcpy(next[index].data, from, VECTOR_LENGTH * sizeof (float));

    }

    void cpToLocalNextFrom(int layer,int index, float* from) {
        memcpy(local_next[layer] + index*VECTOR_LENGTH, from, VECTOR_LENGTH * sizeof (float));
        //  memcpy(curr[index].data,from,VECTOR_LENGTH*sizeof(float));

    }

    void addToLocalNextFrom(int layer,int index, float* from) {
        for (int i = 0; i < VECTOR_LENGTH; i++) {
            local_next[layer][index * VECTOR_LENGTH + i] += from[i];
        }
    }

    void addToNextFrom(int index, float* from) {
        for (int i = 0; i < VECTOR_LENGTH; i++) {
            next[index].data[i] += from[i];
        }
    }

    void addToCurrFrom(int index, float* from) {
        for (int i = 0; i < VECTOR_LENGTH; i++) {
            curr[index].data[i] += from[i];
        }
    }

    nodeVector getCurr(int idx) {
        return curr[idx];
    }

    void readlabel(Graph<Empty>* graph) {
        graph->fill_vertex_array(label, (long) 1);
    }
    void readEmbedding(Graph<Empty>* graph){
        ;
    }
     
    void wrtPara2W(Graph<Empty>* graph, GnnUnit* gnn_v){
           for (const auto& pair : gnn_v->named_parameters()) {
           //     printf("Worker %d:\n", graph->partition_id);
            //      std::cout << pair.key() << ":\n" << pair.value() << std::endl;
            if (pair.key() == "W") {               
                auto tmp_acc = pair.value().accessor<float, 2>();
                for (int i = 0; i < tmp_acc.size(0); i++) {
                    for (int j = 0; j < tmp_acc.size(1); j++) {
                        Weight->weight[i][j] = tmp_acc[i][j];
                  //      printf("%f\t", tmp_acc[i][j]);
                    }
                  //  printf("\n");
                }
            }
        }
    }
    void wrtPartialGrad2localgrad( float * buffer_){
        memcpy(local_grad,buffer_,sizeof(float)*WEIGHT_ROW*WEIGHT_COL);
    }
    void aggregateGrad(Graph<Empty> * graph){
        memset(aggre_grad, 0,sizeof(float)*VECTOR_LENGTH*VECTOR_LENGTH);

            for(int i=start/VECTOR_LENGTH;i<start/VECTOR_LENGTH+rownum;i++){
                for(int j=0;j<VECTOR_LENGTH;j++){
                    aggre_grad[j]+=partial_grad[i*VECTOR_LENGTH+j];
                }
            }
                for(int i=0;i<VECTOR_LENGTH;i++){
                    aggre_grad[i]/=graph->owned_vertices;
 
                }

            for(int i=VECTOR_LENGTH;i<VECTOR_LENGTH*VECTOR_LENGTH;i++){
                aggre_grad[i]=aggre_grad[i%VECTOR_LENGTH];
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
template <typename T_v>  
    void updateX(int layer_,T_v* x_,int h,int w){
        x[layer_]=torch::from_blob(x_,{h,w});
    }
template <typename T_l>   
    void registLabel(T_l* label, int start, int rownum){
        //opt.zero_grad();
        target=torch::from_blob(label+start, rownum,torch::kLong);     
    }
};

void compute(Graph<Empty> * graph, int iterations) {
    
    int rownum = (graph->partition_offset[graph->partition_id + 1] - graph->partition_offset[graph->partition_id]);
    int start = VECTOR_LENGTH * (graph->partition_offset[graph->partition_id]);
    Embeddings<float, long> *embedding = new Embeddings<float, long>();
    embedding->init(graph);
    embedding->readlabel(graph);
    Network *comm=new Network(graph,WEIGHT_ROW,WEIGHT_COL);
    Network* comm1=new Network(graph,WEIGHT_ROW,WEIGHT_COL);
    comm->setWsize(WEIGHT_ROW,WEIGHT_COL);
    comm1->setWsize(WEIGHT_ROW,WEIGHT_COL);
    tensorSet *pytool=new tensorSet(2);
/*1 INIT STAGE*/    
    
    pytool->registOptimizer(torch::optim::SGD(embedding->Gnn_v1->parameters(), 0.05));//new
    pytool->registOptimizer(torch::optim::SGD(embedding->Gnn_v2->parameters(), 0.05));//new
    pytool->registLabel<long>(embedding->label,embedding->start/VECTOR_LENGTH,embedding->rownum);//new


    //debug variable
//    float all1=0,all2=0;
//    double allByGemini=0;

    /*init W with new */
    if(graph->partition_id==0){//first layer
        embedding->wrtPara2W(graph,embedding->Gnn_v1);//write para to temp Weight 
        comm->wrtWtoBuff(embedding->Weight); //write para to send buffer
      
    }
     comm->broadcastW(); // comm buffer
     embedding->Gnn_v1->resetW(WEIGHT_ROW,WEIGHT_COL, comm->buffer);//reset from new 
 
    if(graph->partition_id==0){ //second layer
        embedding->wrtPara2W(graph,embedding->Gnn_v2);//write para to temp Weight 
        comm1->wrtWtoBuff(embedding->Weight); 
     }
     comm1->broadcastW();
     embedding->Gnn_v2->resetW(WEIGHT_ROW,WEIGHT_COL, comm1->buffer);
     
    VertexSubset * active = graph->alloc_vertex_subset();
    active->fill();

    graph->process_vertices<float>(//init  the vertex state.
            [&](VertexId vtx) {
                embedding->initCurrWith(vtx, (float) 1);
                return (float) 1;
            },
    active
            );


            
    for (int i_i = 0; i_i < iterations; i_i++) {
        std::cout<<"start one epoch"<<std::endl;
/*2. FORWARD STAGE*/
        graph->process_vertices<float>(//init  the vertex state.
                [&](VertexId vtx) {
                    embedding->initLocalNextWith(1, vtx, (float) 0);
                    embedding->initNextWith(vtx, (float) 0);
                    return (float) 1;
                },
        active
                );
//2.1.1 start the forward of the first layer
        graph->process_edges<int, nodeVector>(// For EACH Vertex Processing
                [&](VertexId src) {;},
        [&](VertexId src, nodeVector msg, VertexAdjList<Empty> outgoing_adj) {//push model
            return 0;},
        [&](VertexId dst, VertexAdjList<Empty> incoming_adj) {//pull
            nodeVector sum;
            for (int i = 0; i < VECTOR_LENGTH; i++) {
                sum.data[i] = 0;
            }
            for (AdjUnit<Empty> * ptr = incoming_adj.begin; ptr != incoming_adj.end; ptr++) {//pull model
                VertexId src = ptr->neighbour;
                for (int i = 0; i < VECTOR_LENGTH; i++) {
                    sum.data[i] += embedding->curr[src].data[i]/(float)graph->out_degree[src];//
                }

            }
            graph->emit(dst, sum);
        },
        [&](VertexId dst, nodeVector msg) {
            embedding->addToNextFrom(dst, msg.data);
            return 0;
        }, active );
        graph->process_vertices<float>(//init  the vertex state.
                [&](VertexId vtx) {
                embedding->addToLocalNextFrom(0,vtx, embedding->next[vtx].data);
                return 0;
                }, active);
                
//2.1.2 compute tensor from the first layer                
       
        pytool->updateX(0,embedding->local_next[0] + embedding->start,embedding->rownum,VECTOR_LENGTH);//new
        pytool->y[0]=embedding->Gnn_v1->forward(pytool->x[0]);//new
        auto accs=pytool->y[0].accessor<float,2>();//new
       
//2.2.1 init the second layer       
        graph->process_vertices<float>(//init  the vertex state.
                [&](VertexId vtx) {
                    embedding->initCurrWith(vtx, (float) 0);//初始化current
                    embedding->initLocalNextWith(1, vtx, (float) 0);//init LOCALNEXT 
                    if(vtx<start/VECTOR_LENGTH+rownum&&vtx>=start/VECTOR_LENGTH){
                        embedding->cpToCurrFrom(vtx, accs.data()+(vtx-start/VECTOR_LENGTH)*VECTOR_LENGTH);//init curr with the previous layer
                    }
                    embedding->initNextWith(vtx, (float) 0);
                    return (float) 1;
                },
        active
                );
                
//2.2.2 forward the second layer                     
        graph->process_edges<int, nodeVector>(// For EACH Vertex Processing
                [&](VertexId src) {;},
        [&](VertexId src, nodeVector msg, VertexAdjList<Empty> outgoing_adj) {//push model
            return 0;},
        [&](VertexId dst, VertexAdjList<Empty> incoming_adj) {//pull
            nodeVector sum;
            for (int i = 0; i < VECTOR_LENGTH; i++) {
                sum.data[i] = 0;
            }
            for (AdjUnit<Empty> * ptr = incoming_adj.begin; ptr != incoming_adj.end; ptr++) {//pull model
                VertexId src = ptr->neighbour;
                for (int i = 0; i < VECTOR_LENGTH; i++) {
                    sum.data[i] += embedding->curr[src].data[i]/(float)graph->out_degree[src];//
                }

            }
            graph->emit(dst, sum);
        },
        [&](VertexId dst, nodeVector msg) {
            embedding->addToNextFrom(dst, msg.data);
            return 0;
        }, active );
        
        graph->process_vertices<float>(//init  the vertex state.
                [&](VertexId vtx) {
                embedding->addToLocalNextFrom(1,vtx, embedding->next[vtx].data);
                return 0;
                }, active);
//  finish all foward process of these two layer  

/*3 BACKWARD STAGE*/

         
        pytool->updateX(1,embedding->local_next[1] + embedding->start,embedding->rownum,VECTOR_LENGTH);//new
        pytool->x[1].set_requires_grad(true);
        pytool->y[1]=embedding->Gnn_v2->forward(pytool->x[1]);//new
        
         
//3.2 compute the gradient of the second layer.  

         pytool->loss =torch::nll_loss(pytool->y[1],pytool->target);//new
         pytool->loss.backward();//new
         pytool->optimizers[1].step();//new
         
         
         
        embedding->wrtPara2W(graph,embedding->Gnn_v2);//write para to temp Weight 
        comm1->wrtWtoBuff(embedding->Weight); //write para to send buffer
        comm1->gatherW();// gather from others to recvbuffer
        comm1->computeW();// compute new para on recvbuffer
        comm1->broadcastW(comm1->recv_buffer);// broadcast W from given buffer to all other workers
        embedding->Gnn_v2->resetW(WEIGHT_ROW,WEIGHT_COL, comm1->buffer);//reset from new 
        std::cout<<embedding->Gnn_v2->W<<std::endl;

//*3.3.1  compute  W1's partial gradient in first layer   
        pytool->y[0].backward();//new
        pytool->localGrad[0]=embedding->Gnn_v1->W.grad();//new
        auto accsser=pytool->localGrad[0].accessor<float,2>();
//*3.3.2  balance W1's partial gradient in first layer        
     
        comm->wrtBuffertoBuff(accsser.data());
        comm->gatherW();// gather from others to recvbuffer
        comm->computeW();// compute new para on recvbuffer
        comm->broadcastW(comm->recv_buffer);// broadcast W from given buffer to all other workers
        embedding->wrtPartialGrad2localgrad(comm->buffer);
        
        
        
        pytool->localGrad[0].set_data(torch::from_blob(embedding->local_grad,{WEIGHT_ROW,WEIGHT_COL}));//new

        
//3.3.3 backward the partial gradient from 2-layer to 1-layer 
        torch::Tensor partial_grad_layer2=pytool->x[1].grad();
        auto gradient_acc=partial_grad_layer2.accessor<float,2>();
        
  
         graph->process_vertices<double>(//init  the vertex state.
                [&](VertexId vtx) {
                 //   return (double) embedding->local_next[1][vtx*VECTOR_LENGTH];
                    embedding->cpToCurrFrom(vtx,gradient_acc.data()+vtx*VECTOR_LENGTH-start);
                     embedding->initNextWith(vtx,(float)0);
                     embedding->initPartialGradWith(vtx,(float)0);
                     return 1;
                }, active );
                
        //start graph engine.
        graph->process_edges_backward<int, nodeVector>(// For EACH Vertex Processing
        [&](VertexId src, VertexAdjList<Empty> outgoing_adj) {//pull
            nodeVector sum;
            for (int i = 0; i < VECTOR_LENGTH; i++) {
                sum.data[i] = 0;
            }
            for (AdjUnit<Empty> * ptr = outgoing_adj.begin; ptr != outgoing_adj.end; ptr++) {//pull model
                VertexId dst = ptr->neighbour;
                for (int i = 0; i < VECTOR_LENGTH; i++) {
                    sum.data[i] += embedding->curr[dst].data[i];//
                }

            }
            graph->emit(src, sum);
        },
        [&](VertexId src, nodeVector msg) {
            embedding->addToNextFrom(src, msg.data);
            return 0;
        }, active ); 
       
//3.3.4write partial gradient to local and update the gradient of W1
         graph->process_vertices<float>(//init  the vertex state.
                [&](VertexId vtx) {
                embedding->cpToPartialGradVFrom(vtx, embedding->next[vtx].data);
                return 0;
                }, active);
                            

       embedding->aggregateGrad(graph);//new
       pytool->backwardGrad[0]=torch::from_blob(embedding->aggre_grad,{VECTOR_LENGTH, VECTOR_LENGTH});//new
       embedding->Gnn_v1->W.set_data(embedding->Gnn_v1->W-(pytool->localGrad[0]*pytool->backwardGrad[0])*0.05);//new
       
       std::cout<<graph->partition_id<<"before\n"<<embedding->Gnn_v1->W<<std::endl;  
       std::cout<<"combine grade\n"<<pytool->localGrad[0]*pytool->backwardGrad[0]*0.05<<std::endl;//new
        
       
            
        embedding->wrtPara2W(graph,embedding->Gnn_v1);//write para to temp Weight 
        comm->wrtWtoBuff(embedding->Weight); //write para to send buffer
        comm->gatherW();// gather from others to recvbuffer
        comm->computeW();// compute new para on recvbuffer
        comm->broadcastW(comm->recv_buffer);// broadcast W from given buffer to all other workers
        embedding->Gnn_v1->resetW(WEIGHT_ROW,WEIGHT_COL, comm->buffer);
                
//sum and aggregate partial gradient and compute the final result
        
    
        std::cout<<graph->partition_id<<"after\n"<<embedding->Gnn_v1->W<<std::endl;  
            
 /*4 init the next time*/       
        graph->process_vertices<float>(//init  the vertex state.
            [&](VertexId vtx) {
                embedding->cpToCurrFrom(vtx,embedding->local_next[1]+vtx*VECTOR_LENGTH);
                return (float) 1;
            },
    active
            );

        //std::swap(embedding->curr, embedding->next);

    }


    graph->dealloc_vertex_array(embedding->curr);
    graph->dealloc_vertex_array(embedding->next);
     delete active;
}

int main(int argc, char ** argv) {
    MPI_Instance mpi(&argc, &argv);

    if (argc < 4) {
        printf("pagerank [file] [vertices] [iterations]\n");
        exit(-1);
    }

    Graph<Empty> * graph;
    graph = new Graph<Empty>();
    graph->load_directed(argv[1], std::atoi(argv[2]));
    int iterations = std::atoi(argv[3]);
    printf("hello world\n");
    double exec_time = 0;
    exec_time -= get_time();
    compute(graph, iterations);
      exec_time += get_time();
  if (graph->partition_id==0) {
    printf("exec_time=%lf(s)\n", exec_time);
  
      
      float a1[6]={13.0,3.0,1.0,2.0,2.0,2.0};
      float w1[4]={1.0,4.0,3.0,4.0};
      float w2[4]={2.0,3.0,5.0,7.0};
      float w3[2]={3.0,6.0};
      float y[4]={7,10,15,22};
      float w1w2[4]={18,36,38,76};
      
//      torch::Tensor x1=torch::from_blob(a1,{2,2});
//      torch::Tensor x2=torch::from_blob(a1,{2,2});
//      torch::Tensor x3=torch::from_blob(a1,{1,2});
//      torch::Tensor w_1=torch::from_blob(w1,{2, 2});
//      torch::Tensor w_3=torch::from_blob(w3,{2,1});
//      w_1.set_requires_grad(true);
//      w_3.set_requires_grad(true);
//      x1.set_requires_grad(true);
//      x2.set_requires_grad(true);
//      torch::Tensor w_2=torch::from_blob(w2,{2, 2});
//      torch::Tensor y1=torch::from_blob(y,{2, 2});
//      w_2.set_requires_grad(true);
//      y1.set_requires_grad(true);
//      torch::Tensor s=w_1.mm(w_2);
    
      
      
//      y1=torch::sigmoid(x1.mm(w_1));//.mm(w_3); //.mm(w_1)
//      y1.backward();
//      torch::Tensor grad1=w_1.grad();
//      std::cout<<"grad1:\n"<<grad1<<std::endl;
//      
//      torch::Tensor y2=x2.mm(w_2).mm(w_3);//.mm(x2);
//      y2.backward();
//      std::cout<<y2<<std::endl;
//      torch::Tensor grad2=x2.grad();
//      std::cout<<"grad2:\n"<<grad2<<std::endl;
      
//      y1=torch::sigmoid(x1.mm(w_1)).mm(w_2).mm(w_3); //.mm(w_1)
//      y1.backward();
//      torch::Tensor grad2=w_1.grad();
//      std::cout<<"gradall:\n"<<grad2<<std::endl;
      
      
      
   //   torch::Tensor y2=x2.mm(w_2);
  //    y2.backward();
    //  torch::Tensor grad_com=w_1.grad().mm(w_1.t());
//      x2.set_data(y1);
//      torch::Tensor y2=x2.mm(w_2);
//      y2.backward();
//      torch::Tensor grad2=x2.grad();
   //   std::cout<<"grad_2:\n"<<grad_com<<std::endl;
//      torch::Tensor all=grad2*grad1;
//    std::cout<<"all:\n"<<all<<std::endl;
      
      
//       torch::Tensor gad2=x1.grad();
//       std::cout<<"grad_all:\n"<<gad2<<std::endl;
//       torch::Tensor ga=gad.mm(gad1);
//       std::cout<<"grad_compute\n"<<ga<<std::endl;
       
       
       
    }
    delete graph;
    return 0;
}





/*
    //     float * recv_buffer = NULL;
//        if (graph->partition_id == 0) {
//            //接收数组
//            recv_buffer = new float[(graph->partitions) * WEIGHT_ROW * WEIGHT_COL];
//            memset(recv_buffer, 0, sizeof (float)*(graph->partitions) * WEIGHT_ROW * WEIGHT_COL);
//            memcpy(recv_buffer, embedding->Weight->weight, sizeof (float)*WEIGHT_ROW * WEIGHT_COL);
//
//            for (int i = 1; i < (graph->partitions); i++) {
//                MPI_Recv(recv_buffer + i * WEIGHT_ROW*WEIGHT_COL, WEIGHT_ROW*WEIGHT_COL, MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//            }
//        } else {
//            //发送数组
//            MPI_Send(buffer, WEIGHT_ROW*WEIGHT_COL, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
//        }

//        if (graph->partition_id == 0) {
//            for (int i = 1; i < graph->partitions; i++) {
//                for (int j = 0; j < WEIGHT_ROW * WEIGHT_COL; j++) {
//                    recv_buffer[j] = recv_buffer[j] + recv_buffer[j + i * WEIGHT_ROW * WEIGHT_COL];
//                }
//            }
//            printf("\n");
//            printf("Worker 0 after receive:\n", graph->partition_id);
//            for (int i = 0; i < 4; i++) {
//                for (int j = 0; j < 4; j++) {
//                    recv_buffer[4 * i + j] /= graph->partitions;
//                    printf("%f\t", recv_buffer[4 * i + j]);
//                }
//                printf("\n");
//            }
//
//        }
        //  float * recv_weight_buffer=new float[WEIGHT_ROW*WEIGHT_COL];
//        if (graph->partition_id == 0) {
//            for (int i = 1; i < (graph->partitions); i++) {
//                MPI_Send(recv_buffer, WEIGHT_ROW*WEIGHT_COL, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
//            }
//            memcpy(buffer, recv_buffer, sizeof (float)*WEIGHT_ROW * WEIGHT_COL);
//        }
//        else {
//            MPI_Recv(buffer, WEIGHT_ROW*WEIGHT_COL, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//        }


//        if (graph->partition_id == 1) {
//            printf("Worker 1 after scatter:\n", graph->partition_id);
//            for (int i = 0; i < 4; i++) {
//                for (int j = 0; j < 4; j++) {
//                    printf("%f\t", comm->buffer[4 * i + j]);
//                }
//                printf("\n");
//            }
//        }
//        printf("\n");
 */
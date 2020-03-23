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

    GnnUnit(size_t w, size_t h) {
        //        at::TensorOptions *opt=new at::TensorOptions;
        //       opt->requires_grad(true);
        //  torch::randn
        //     A=torch::randn(torch::randn({w,h},opt));
        W = register_parameter("W", torch::randn({w, h}));

    }
    void resetW(size_t w,size_t h,float* buffer){
        torch::Tensor new_weight_tensor = torch::from_blob(buffer,{w, h});
        W.set_data(new_weight_tensor);
    }
    torch::Tensor forward(torch::Tensor x) {

        auto tmp_acc_c = W.accessor<float, 2>();
        x = x.mm(W);
        return torch::log_softmax(x, 1);
    }

    torch::Tensor forward2(torch::Tensor x) {
        auto tmp_acc_c = W.accessor<float, 2>();
        x = x.mm(W);
        return torch::relu(x);
    }
};

template <typename T_v, typename T_l>
class Embeddings {
public:

    Embeddings() {

    }
    T_v* curr_v = NULL;
    T_v* next_v = NULL;
    T_v** local_next= NULL;
    GnnUnit *Gnn_v1 = NULL;
    GnnUnit *Gnn_v2 =NULL;
    T_l *label = NULL;
    weightVector * Weight = NULL;

    nodeVector * curr = NULL;
    nodeVector * next = NULL;

    void init(Graph<Empty>* graph) {
        curr_v = new float [graph->vertices*VECTOR_LENGTH];//graph->alloc_vertex_array<float>(VECTOR_LENGTH);
        next_v = graph->alloc_vertex_array<float>(VECTOR_LENGTH);
        Weight = new weightVector(); // reuse
        label = graph->alloc_vertex_array<T_l>();
        curr = graph->alloc_vertex_array<nodeVector>();
        next = graph->alloc_vertex_array<nodeVector>();
        Gnn_v1 = new GnnUnit(WEIGHT_ROW, WEIGHT_COL);
        Gnn_v2 =new GnnUnit(WEIGHT_ROW, WEIGHT_COL);
       /* new Next_v2 for GNN*/
        local_next=new float*[2];
        local_next[1]=graph->alloc_vertex_array<float>(VECTOR_LENGTH);
        local_next[2]=graph->alloc_vertex_array<float>(VECTOR_LENGTH);
        
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
    void initNextVWith( int index, float with) {
        for (int i = 0; i < VECTOR_LENGTH; i++) {
            *(next_v + index * VECTOR_LENGTH + i) = with;
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

    void cpToNextVFrom(int index, float* from) {
        memcpy(next_v + index*VECTOR_LENGTH, from, VECTOR_LENGTH * sizeof (float));
        //  memcpy(curr[index].data,from,VECTOR_LENGTH*sizeof(float));

    }

    void addToLocalNextFrom(int layer,int index, float* from) {
        for (int i = 0; i < VECTOR_LENGTH; i++) {
            local_next[layer][index * VECTOR_LENGTH + i] += from[i];
        }
    }

    void addToNextVFrom(int index, float* from) {
        for (int i = 0; i < VECTOR_LENGTH; i++) {
            next_v[index * VECTOR_LENGTH + i] += from[i];
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
                    //    printf("%f\t", tmp_acc[i][j]);
                    }
                 //   printf("\n");
                }
            }
        }
    }
};

void compute(Graph<Empty> * graph, int iterations) {
    
    int rownum = (graph->partition_offset[graph->partition_id + 1] - graph->partition_offset[graph->partition_id]);
    int start = VECTOR_LENGTH * (graph->partition_offset[graph->partition_id]);
    Embeddings<float, long> *embedding = new Embeddings<float, long>();
    embedding->init(graph);
    embedding->readlabel(graph);
    Network *comm=new Network(graph,WEIGHT_ROW,WEIGHT_COL);
    comm->setWsize(WEIGHT_ROW,WEIGHT_COL);
    
    torch::optim::SGD optimizer(embedding->Gnn_v1->parameters(), 0.05);
    optimizer.zero_grad();
    torch::Tensor target = torch::from_blob((embedding->label),{rownum}, torch::kLong);
    //torch::optim::SGD optimizer()
    
    
    
    
 
    
    /*init W with new */
    if(graph->partition_id==0){
        embedding->wrtPara2W(graph,embedding->Gnn_v1);//write para to temp Weight 
        comm->wrtWtoBuff(embedding->Weight); //write para to send buffer
    }
     comm->broadcastW(); // comm buffer
     embedding->Gnn_v1->resetW(WEIGHT_ROW,WEIGHT_COL, comm->buffer);//reset from new 
     
    
   
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


        int test_index = 0;
        printf("============start curr========================\n");
        printf("curr %d\t%f\t%f\n", graph->in_degree[test_index], embedding->curr[test_index].data[0], embedding->next_v[test_index * VECTOR_LENGTH + 0]);
        //printf("%d\t%f\t%f\n",graph->in_degree[300000],embedding->next[300000].data[0],embedding->curr_v[300000*VECTOR_LENGTH+0]);


        graph->process_vertices<float>(//init  the vertex state.
                [&](VertexId vtx) {
                    embedding->initNextVWith(vtx, (float) 0);
                    embedding->initNextWith(vtx, (float) 0);
                    return (float) 1;
                },
        active
                );

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
                embedding->addToNextVFrom(vtx, embedding->next[vtx].data);
                return 0;
                }, active);
                
        
        float all1=0,all2=0;
        double allByGemini=0;
       allByGemini=graph->process_vertices<double>(//init  the vertex state.
                [&](VertexId vtx) {
                    return (double) embedding->next_v[vtx*VECTOR_LENGTH];
                }, active );

        
                
                
        printf("row_num:%d\nstart:%d (%d)\n", rownum, start/VECTOR_LENGTH , graph->partition_offset[graph->partition_id + 1]);
        
        for(int i=0;i<graph->vertices;i++){
           // if(i<start/4||i>(start/VECTOR_LENGTH+rownum)){
            all1+=embedding->curr[i].data[1];//curr_v[i*VECTOR_LENGTH];
            all2+=embedding->next[i].data[1];
            
          //  }
        }
        //for (int i=0;i<get)
         //   printf("thread %d\t ",graph->local_partition_offset[1]);
        printf("next %d: all curr:%f\t all next: %f all_bg%f\n", graph->partition_id,all1,all2,allByGemini);
        
        printf("second layer\n+++++++++++++++++++++++++++++++++++++\n");
        
//        graph->process_vertices<float>(//init  the vertex state.
//                [&](VertexId vtx) {
//                    embedding->initNextVWith(vtx, (float) 0);
//                    embedding->initNextWith(vtx, (float) 0);
//                    embedding->initCurrWith(vtx, (float) 0.1);
//                    return (float) 1;
//                },
//        active
//                );
//
//                
//                
//        graph->process_edges<int, nodeVector>(// For EACH Vertex Processing
//                [&](VertexId src) {;},
//        [&](VertexId src, nodeVector msg, VertexAdjList<Empty> outgoing_adj) {//push model
//            return 0;},
//        [&](VertexId dst, VertexAdjList<Empty> incoming_adj) {//pull
//            nodeVector sum;
//            for (int i = 0; i < VECTOR_LENGTH; i++) {
//                sum.data[i] = 0;
//            }
//            for (AdjUnit<Empty> * ptr = incoming_adj.begin; ptr != incoming_adj.end; ptr++) {//pull model
//                VertexId src = ptr->neighbour;
//                for (int i = 0; i < VECTOR_LENGTH; i++) {
//                    sum.data[i] += embedding->curr[src].data[i]/(float)graph->out_degree[src];//
//                }
//
//            }
//            graph->emit(dst, sum);
//        },
//        [&](VertexId dst, nodeVector msg) {
//            embedding->addToNextFrom(dst, msg.data);
//            return 0;
//        }, active );
//        graph->process_vertices<float>(//init  the vertex state.
//                [&](VertexId vtx) {
//                embedding->addToNextVFrom(vtx, embedding->next[vtx].data);
//                return 0;
//                }, active);
//                all1=0;all2=0;allByGemini=0;
//            for(int i=start/4;i<start/4+rownum;i++){
//           // if(i<start/4||i>(start/VECTOR_LENGTH+rownum)){
//            all1+=embedding->curr[i].data[1];//curr_v[i*VECTOR_LENGTH];
//            all2+=embedding->next[i].data[1];
//            
//          //  }
//        }
//          allByGemini=graph->process_vertices<double>(//init  the vertex state.
//                [&](VertexId vtx) {
//                    return (double) embedding->next_v[vtx*VECTOR_LENGTH];
//                }, active );
//        //for (int i=0;i<get)
//         //   printf("thread %d\t ",graph->local_partition_offset[1]);
//        printf("next %d: all curr:%f\t all next: %f all_bg%f\n", graph->partition_id,all1,all2,allByGemini);
//        
//        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
     //   torch::optim::SGD optimizer(embedding->Gnn_v->parameters(), 0.05);
     //   optimizer.zero_grad();
        torch::Tensor x = torch::from_blob(embedding->next_v + start, {rownum, VECTOR_LENGTH}); //regist new embedding as input!!
        torch::Tensor prediction = embedding->Gnn_v1->forward(x);
     //   torch::Tensor target = torch::from_blob((embedding->label),{rownum}, torch::kLong);
        torch::Tensor loss = torch::nll_loss(prediction, target);
       // auto tmp_acc = loss.accessor<float,0>();
       // printf("++++++++++++%d,%f",graph->partition_id,tmp_acc[0]);
        std::cout<<"loss at"<<graph->partition_id<<"is :"<<loss<<std::endl;
        loss.backward();
        //optimizer.zero_grad();
        optimizer.step(); 
        
        
        embedding->wrtPara2W(graph,embedding->Gnn_v1);//write para to temp Weight 
        comm->wrtWtoBuff(embedding->Weight); //write para to send buffer
        comm->gatherW();// gather from others to recvbuffer
        comm->computeW();// compute new para on recvbuffer
        comm->broadcastW(comm->recv_buffer);// broadcast W from given buffer to all other workers
        embedding->Gnn_v1->resetW(WEIGHT_ROW,WEIGHT_COL, comm->buffer);//reset from new 
        
        
        for (const auto& pair : embedding->Gnn_v1->named_parameters()) {
            if(graph->partition_id==0);
         //     std::cout << pair.key() << ":\n" << pair.value() << std::endl;
        }

        std::swap(embedding->curr, embedding->next);

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
  }
    //  for (int run=0;run<5;run++) {
    //    compute(graph, iterations);
    //  }

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
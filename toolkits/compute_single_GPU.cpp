#include "core/gcn.hpp"
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
    graph->generate_COO(active);
    graph->reorder_COO();

    // for(int i=graph->edges-100;i<graph->edges;i++){
    //     std::cout<<graph->_graph_cpu->srcList[i]<<" "<<graph->_graph_cpu->dstList[i]<<std::endl;
    // }
    std::cout<<graph->vertices<<" "<<graph->edges<<std::endl;


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
              weight[i]=(ValueType)std::sqrt(graph->in_degree[vtx])+(ValueType)std::sqrt(graph->out_degree[dst]);
          }
          for(int i=graph->incoming_adj_index_backward[0][vtx];i<graph->incoming_adj_index_backward[0][vtx+1];i++){
              VertexId dst=graph->incoming_adj_list_backward[0][i].neighbour;
              weight_backward[i]=(ValueType)std::sqrt(graph->out_degree[vtx])+(ValueType)std::sqrt(graph->in_degree[dst]);
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
        gr_e->forward_one_step(X0_gpu.packed_accessor<float,2>().data(),
            Y0_gpu.packed_accessor<float,2>().data(),
                W_for_gpu.packed_accessor<float,2>().data(), SCALA_TYPE, SIZE_LAYER_1);
//        std::cout<<Y0_gpu.cpu().t()[0]<<std::endl;
        inter1_gpu.set_data(Gnn_v1->forward(Y0_gpu));//torch::from_blob(embedding->Gnn_v1->forward(pytool->x[0]).accessor<float,2>().data(),{embedding->rownum,VECTOR_LENGTH});
        //std::cout<<inter1_gpu.t().cpu()[0]<<std::endl;
        Out0_gpu=torch::relu(inter1_gpu);//new
        
        std::cout<<"TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT"<<std::endl;
//layer 2;         
        gr_e->forward_one_step(Out0_gpu.packed_accessor<float,2>().data(),
            Y1_gpu.packed_accessor<float,2>().data(),
                W_for_gpu.packed_accessor<float,2>().data(), SCALA_TYPE, SIZE_LAYER_2);  
        Out1_gpu = Gnn_v2->forward(Y1_gpu);
//output;        
        torch::Tensor tt=Out1_gpu.log_softmax(1);//CUDA
        pytool->loss =torch::nll_loss(tt,target_gpu);//new
        pytool->loss.backward(); 

           // std::cout<<target_gpu.cpu()<<std::endl;
 //       std::cout<<Gnn_v2->W.grad().cpu().t()[1]<<std::endl;
//inv layer 2;        
        Gnn_v2->learn_gpu(Gnn_v2->W.grad(),learn_rate);//signle node

         
//inv layer 1;
    // 2->1
        gr_e->backward_one_step(Y1_gpu.grad().packed_accessor<float,2>().data(),
            Y1_inv_gpu.packed_accessor<float,2>().data(),
                W_back_gpu.packed_accessor<float,2>().data(), SCALA_TYPE, SIZE_LAYER_2); 

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
     if(i_i==(iterations-1)&&graph->partition_id==0){
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
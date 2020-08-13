#include "core/gcn.hpp"

void compute_GPU(Graph<Empty> * graph, int iterations) {
    ValueType learn_rate=0.01;
    //gpu_processor *gp=new gpu_processor(); //GPU
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


//    graph->process_vertices<ValueType>(//init  the vertex state.
//        [&](VertexId vtx){
//            int thread_id = omp_get_thread_num();
//           partial_new_combine_grad[thread_id]=partial_new_combine_grad[thread_id]+
//            x0_gpu[vtx].reshape({SIZE_LAYER_1,1}).mm(
//            pytool->localGrad[0][vtx].reshape({1,SIZE_LAYER_2})
//             *grad0_gpu[vtx].reshape({1,SIZE_LAYER_2}));
//                    
//            return (ValueType)1;
//        },
//    active
//    );
//    for(int i=0;i<graph->threads;i++){
//       new_combine_grad=new_combine_grad+partial_new_combine_grad[i];
//    }

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
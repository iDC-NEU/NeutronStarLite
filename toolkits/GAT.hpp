#include "core/gnnmini.hpp"
#include <c10/cuda/CUDAStream.h>


class GAT_impl{
public:
        
    int iterations;
    ValueType learn_rate;
    ValueType weight_decay;
    
    //graph
    VertexSubset *active;
    Graph<Empty>* graph;
    std::vector<CSC_segment_pinned *>subgraphs;
    //NN
    GNNDatum *gnndatum;
    NtsVar L_GT_C;
    NtsVar L_GT_G;
    std::map<std::string,NtsVar>I_data;
    GTensor<ValueType, long> *gt;
    //Variables
    std::vector<GnnUnit*>P;
    std::vector<NtsVar>X;
    std::vector<NtsVar>Y;
    std::vector<NtsVar>X_grad;
    NtsVar F;
    NtsVar loss;
    NtsVar tt;
    
    double exec_time = 0;
    double all_sync_time = 0;
    double sync_time = 0;
    double all_graph_sync_time = 0;
    double graph_sync_time = 0;
    double all_compute_time = 0;
    double compute_time = 0;
    double all_copy_time = 0;
    double copy_time = 0;
    double graph_time = 0;
    double all_graph_time = 0;

    
    GAT_impl(Graph<Empty> *graph_, int iterations_, bool process_local = false, bool process_overlap = false){
        graph=graph_;
        iterations=iterations_;
        learn_rate = 0.01;
        weight_decay=0.05;
        active = graph->alloc_vertex_subset();
        active->fill();
        
        graph->init_gnnctx(graph->config->layer_string);
        graph->init_rtminfo();
        graph->rtminfo->process_local = graph->config->process_local;
        graph->rtminfo->reduce_comm = graph->config->process_local;
        graph->rtminfo->copy_data = false;
        graph->rtminfo->process_overlap = graph->config->overlap;
        graph->rtminfo->with_weight=true;
       
        
    } 
    void init_graph(){
        //std::vector<CSC_segment_pinned *> csc_segment;
        graph->generate_COO(active);
        graph->reorder_COO_W2W();
        gt = new GTensor<ValueType, long>(graph, active);
        gt->GenerateGraphSegment(subgraphs, true);
        gt->GenerateMessageBitmap(subgraphs);
        graph->init_message_buffer();
    }
    void init_nn(){
        GNNDatum *gnndatum = new GNNDatum(graph->gnnctx);
        gnndatum->random_generate();
        gnndatum->registLabel(L_GT_C);
        L_GT_G = L_GT_C.cuda();
        
        for(int i=0;i<graph->gnnctx->layer_size.size()-1;i++){
            P.push_back(new GnnUnit(graph->gnnctx->layer_size[i], graph->gnnctx->layer_size[i+1]));
            P.push_back(new GnnUnit(2*graph->gnnctx->layer_size[i+1],1));
        }
        
        torch::Device GPU(torch::kCUDA, 0);
        for(int i=0;i<P.size();i++){
            P[i]->init_parameter();
            P[i]->to(GPU);
        }
        
        F=graph->Nts->NewOnesTensor({graph->gnnctx->l_v_num, graph->gnnctx->layer_size[0]},torch::DeviceType::CPU);
        for(int i=0;i<graph->gnnctx->layer_size.size()-1;i++){
            Y.push_back(graph->Nts->NewKeyTensor(
                                {graph->gnnctx->l_v_num, 
                                   graph->gnnctx->layer_size[i+1]},
                                       torch::DeviceType::CUDA));
            X_grad.push_back(graph->Nts->NewKeyTensor(
                                {graph->gnnctx->l_v_num, 
                                   graph->gnnctx->layer_size[i]},
                                       torch::DeviceType::CUDA));
        }
        for(int i=0;i<graph->gnnctx->layer_size.size();i++){
        NtsVar d;X.push_back(d);
        }
        
        X[0]=F.cuda().set_requires_grad(true);
    }

    
inline NtsVar preComputation(NtsVar& master_input){
     size_t layer=graph->rtminfo->curr_layer;
        return P[layer*2]->forward(master_input);
}   

NtsVar vertexForward(NtsVar &a, NtsVar &x){
    
    size_t layer=graph->rtminfo->curr_layer;
    if(layer==0){
        return torch::relu(a).set_requires_grad(true);
    }
    else if(layer==1){
        NtsVar y = torch::relu(a);
        y = y.log_softmax(1); //CUDA
        y = torch::nll_loss(y, L_GT_G);
        return y;
    }
}
                             //grad to message            //grad to src   
NtsVar edge_Backward(NtsVar &message_grad, NtsVar &src_grad, NtsScheduler* nts){
      I_data["msg_grad"]=at::_sparse_softmax_backward_data(
               nts->PrepareMessage(message_grad),
                I_data["atten"],
                 nts->BYDST(),
                  I_data["w"]);
      I_data["msg"].backward(I_data["msg_grad"].coalesce().values(),true);
      I_data["src_trans"].backward(src_grad,true);
      return I_data["src"].grad();
    
 }
NtsVar edge_Forward(NtsVar &src_input, NtsVar &src_input_transfered, 
                             NtsVar &dst_input, NtsVar &dst_input_transfered, 
                                    NtsScheduler* nts){
     size_t layer =graph->rtminfo->curr_layer;
     if(graph->rtminfo->forward==true){
         I_data["src"]=src_input;
        nts->SerializeToCPU("src_cached",src_input);
     }else{
         I_data["src"]=nts->DeSerializeFromCPU("src_cached");
     }
     
     src_input_transfered=I_data["src_trans"]=P[layer*2]->forward(I_data["src"]);
     I_data["dst_trans"]=dst_input_transfered;//finish apply W
     I_data["msg"]=torch::cat({nts->ScatterSrc(I_data["src_trans"]), nts->ScatterDst(I_data["dst_trans"])},1);                    
     I_data["msg"]=torch::leaky_relu(torch::exp(P[layer*2+1]->forward(I_data["msg"])),1.0);  
     I_data["w"]=nts->PrepareMessage(I_data["msg"]);
     I_data["atten"]=at::_sparse_softmax(I_data["w"],graph->Nts->BYDST());
     
     //src_input_transfered=I_data["src_trans"];
     return I_data["atten"].coalesce().values();    
}
 
void vertexBackward(){
    
    int layer=graph->rtminfo->curr_layer;
    if(layer==0){
        X[1].backward(X_grad[1]); //new
    }
    else if(layer==1){
        loss.backward();
    }
}



void Backward(){
    graph->rtminfo->forward = false;
    for(int i=graph->gnnctx->layer_size.size()-2;i>=0;i--){
        graph->rtminfo->curr_layer = i;
        vertexBackward();
        NtsVar grad_to_Y=Y[i].grad();
        gt->GraphPropagateBackwardEdgeComputation(X[i], 
                                              grad_to_Y,                                              
                                              X_grad[i],
                                              subgraphs,
                                              [&](NtsVar &master_input){//pre computation
                                                   return preComputation(master_input);
                                              },
                                              [&](NtsVar &mirror_input,NtsVar &mirror_input_transfered,
                                                       NtsVar &X, NtsVar &X_trans, NtsScheduler* nts){//edge computation
                                                   return edge_Forward(mirror_input,mirror_input_transfered,X, X_trans, nts);
                                              },
                                              [&](NtsVar &b,NtsVar &c, NtsScheduler* nts){//edge computation
                                                   return edge_Backward(b, c,nts);
                                              });
    }

    for(int i=0;i<P.size();i++){
        P[i]->all_reduce_to_gradient(P[i]->W.grad().cpu()); //W2->W.grad().cpu()
        P[i]->learnC2G_with_decay(learn_rate,weight_decay);
    }
}

void Forward(){
    graph->rtminfo->forward = true;
    for(int i=0;i<graph->gnnctx->layer_size.size()-1;i++){
        graph->rtminfo->curr_layer = i;
        gt->GraphPropagateForwardEdgeComputation(X[i],Y[i],subgraphs,
                [&](NtsVar &master_input){//pre computation
                   return preComputation(master_input);
                },
                [&](NtsVar &mirror_input,NtsVar &mirror_input_transfered,
                           NtsVar &X, NtsVar &X_trans, NtsScheduler* nts){//edge computation
                               return edge_Forward(mirror_input,mirror_input_transfered,X, X_trans, nts);
                 });
        X[i+1]=vertexForward(Y[i], X[i]);
    }
    loss=X[graph->gnnctx->layer_size.size()-1];
    
}

/*GPU dist*/ void run()
{
    if (graph->partition_id == 0)
        printf("GNNmini::Engine[Dist.GPU.GATimpl] running [%d] Epochs\n",iterations);
//        graph->print_info();

    exec_time -= get_time();
    for (int i_i = 0; i_i < iterations; i_i++){
        graph->rtminfo->epoch = i_i;
        if (i_i != 0){
            for(int i=0;i<P.size();i++){
                P[i]->zero_grad();
            }
        }

        Forward();
        Backward();
        
        if (graph->partition_id == 0)
            std::cout << "GNNmini::Running.Epoch["<<i_i<<"]:loss\t" << loss << std::endl;
         
    }
    exec_time += get_time();

    delete active;
}







void DEBUGINFO(){

    if (graph->partition_id == 0)
    {
        printf("\n#Timer Info Start:\n");
        printf("#all_time=%lf(s)\n", exec_time);
        printf("#sync_time=%lf(s)\n", all_sync_time);
        printf("#all_graph_sync_time=%lf(s)\n", all_graph_sync_time);
        printf("#copy_time=%lf(s)\n", all_copy_time);
        printf("#nn_time=%lf(s)\n", all_compute_time);
        printf("#graph_time=%lf(s)\n", all_graph_time);
        printf("#communicate_extract+send=%lf(s)\n", graph->all_compute_time);
        printf("#communicate_processing_received=%lf(s)\n", graph->all_overlap_time);
        printf("#communicate_processing_received.copy=%lf(s)\n", graph->all_recv_copy_time);
        printf("#communicate_processing_received.kernel=%lf(s)\n", graph->all_recv_kernel_time);
        printf("#communicate_processing_received.wait=%lf(s)\n", graph->all_recv_wait_time);
        printf("#communicate_wait=%lf(s)\n", graph->all_wait_time);
        printf("#streamed kernel_time=%lf(s)\n", graph->all_kernel_time);
        printf("#streamed movein_time=%lf(s)\n", graph->all_movein_time);
        printf("#streamed moveout_time=%lf(s)\n", graph->all_moveout_time);
        printf("#cuda wait time=%lf(s)\n", graph->all_cuda_sync_time);
        printf("#graph repliation time=%lf(s)\n", graph->all_replication_time);
        printf("#Timer Info End\n");
    }
    //      NtsVar tt_cpu=tt.cpu();
    //  if(i_i==(iterations-1)&&graph->partition_id==0){
    //     inference(tt_cpu,graph, embedding, pytool,W1,W2);
    //  }
    double max_time = 0;
    double mean_time = 0;
    double another_time = 0;
    MPI_Datatype l_vid_t = get_mpi_data_type<double>();
    MPI_Allreduce(&all_graph_time, &max_time, 1, l_vid_t, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&exec_time, &another_time, 1, l_vid_t, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&graph->all_replication_time, &mean_time, 1, l_vid_t, MPI_SUM, MPI_COMM_WORLD);
    if (graph->partition_id == 0)
        printf("ALL TIME = %lf(s) GRAPH TIME = %lf(s) MEAN TIME = %lf(s)\n", 
                another_time, max_time / graph->partitions, mean_time / graph->partitions);
}

};
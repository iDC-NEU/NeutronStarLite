#include "core/gnnmini.hpp"
#include <c10/cuda/CUDAStream.h>


class GCN_impl{
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

    
    GCN_impl(Graph<Empty> *graph_, int iterations_, bool process_local = false, bool process_overlap = false){
        graph=graph_;
        iterations=iterations_;
        learn_rate = 0.01;
        weight_decay=0.05;
        active = graph->alloc_vertex_subset();
        active->fill();
        
        graph->init_gnnctx(graph->config->layer_string);
            //rtminfo initialize
        graph->init_rtminfo();
        graph->rtminfo->process_local = graph->config->process_local;
        graph->rtminfo->reduce_comm = graph->config->process_local;
        graph->rtminfo->copy_data = false;
        graph->rtminfo->process_overlap = graph->config->overlap;
        graph->rtminfo->with_weight=true;
        graph->rtminfo->with_cuda=true;
       
    } 
    void init_graph(){
        //std::vector<CSC_segment_pinned *> csc_segment;
        graph->generate_COO(active);
        graph->reorder_COO_W2W();
        //generate_CSC_Segment_Tensor_pinned(graph, csc_segment, true);
        gt = new GTensor<ValueType, long>(graph, active);
        gt->GenerateGraphSegment(subgraphs, true);
        double load_rep_time = 0;
        load_rep_time -= get_time();
        graph->load_replicate3(graph->gnnctx->layer_size);
        load_rep_time += get_time();
        if (graph->partition_id == 0)
        printf("#load_rep_time=%lf(s)\n", load_rep_time);
        graph->init_message_buffer();

    }
    void init_nn(){
        GNNDatum *gnndatum = new GNNDatum(graph->gnnctx);
        gnndatum->random_generate();
        gnndatum->registLabel(L_GT_C);
        L_GT_G = L_GT_C.cuda();
        
        for(int i=0;i<graph->gnnctx->layer_size.size()-1;i++){
            P.push_back(new GnnUnit(graph->gnnctx->layer_size[i], graph->gnnctx->layer_size[i+1]));
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
                                   graph->gnnctx->layer_size[i]},
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

NtsVar vertexForward(NtsVar &a, NtsVar &x){
    NtsVar y;
    int layer=graph->rtminfo->curr_layer;
    if(layer==0){
        y=torch::relu(P[layer]->forward(a)).set_requires_grad(true);

    }
    else if(layer==1){
        y = P[layer]->forward(a);
        y = y.log_softmax(1); //CUDA
        y = torch::nll_loss(y, L_GT_G);   
    }
    return y;
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
    gt->GraphPropagateBackward(grad_to_Y, X_grad[i], subgraphs);
    }
    
    for(int i=0;i<P.size()-1;i++){
        P[i]->all_reduce_to_gradient(P[i]->W.grad().cpu());
        P[i]->learnC2G_with_decay(learn_rate,weight_decay);
    }
    
        
}

void Forward(){
    graph->rtminfo->forward = true;
    
    for(int i=0;i<graph->gnnctx->layer_size.size()-1;i++){
    graph->rtminfo->curr_layer = i;
    gt->GraphPropagateForward(X[i], Y[i], subgraphs);
    X[i+1]=vertexForward(Y[i],X[i]);
    }
    loss=X[graph->gnnctx->layer_size.size()-1];
}




/*GPU dist*/ void run()
{
    if (graph->partition_id == 0)
        printf("GNNmini::Engine[Dist.GPU.GCNimpl] running [%d] Epochs\n",iterations);
        graph->print_info();
        
    exec_time -= get_time();
    for (int i_i = 0; i_i < iterations; i_i++){
        graph->rtminfo->epoch = i_i;
        if (i_i != 0){
            for(int i=0;i<P.size();i++)
            P[i]->zero_grad();
            for(int i=0;i<graph->gnnctx->layer_size.size()-1;i++){
            //X_grad[i].zero_();
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
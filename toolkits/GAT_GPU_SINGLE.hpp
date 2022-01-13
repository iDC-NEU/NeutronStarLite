#include "core/gnnmini.hpp"
#include <c10/cuda/CUDAStream.h>




class GAT_GPU_SINGLE_impl{
public:
        
    int iterations;
    ValueType learn_rate;
    ValueType weight_decay;
    ValueType drop_rate;
    ValueType alpha;
    ValueType beta1;
    ValueType beta2;
    ValueType epsilon;
    ValueType decay_rate;
    ValueType decay_epoch;
    
    //graph
    VertexSubset *active;
    Graph<Empty>* graph;
    std::vector<CSC_segment_pinned *>subgraphs;
    //NN
    GNNDatum *gnndatum;
    NtsVar L_GT_C;
    NtsVar L_GT_G;
    NtsVar MASK;
    NtsVar MASK_gpu;
    std::map<std::string,NtsVar>I_data;
    GTensor<ValueType, long> *gt;
    //Variables
    std::vector<Parameter*>P;
    std::vector<NtsVar>mirror_in;
    std::vector<NtsVar>E_in;
    std::vector<NtsVar>E_out;
    std::vector<NtsVar>E_out_grad;
    std::vector<NtsVar>X;
    std::vector<NtsVar>M;
    std::vector<NtsVar>M_grad;
    NtsVar F;
    NtsVar loss;
    NtsVar tt;
    torch::nn::Dropout drpmodel;
    
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

    
    GAT_GPU_SINGLE_impl(Graph<Empty> *graph_, int iterations_, bool process_local = false, bool process_overlap = false){
        graph=graph_;
        iterations=iterations_;
        
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
        graph->rtminfo->lock_free=graph->config->lock_free;
       
    }
    inline int pos(int p,int l){
        return l*graph->partitions+p;
    }
    void init_graph(){
        //std::vector<CSC_segment_pinned *> csc_segment;
        graph->generate_COO();
        graph->reorder_COO_W2W();
        //generate_CSC_Segment_Tensor_pinned(graph, csc_segment, true);
        gt = new GTensor<ValueType, long>(graph, active);
        gt->GenerateGraphSegment(subgraphs, GPU_T,[&](VertexId src,VertexId dst){
            return gt->norm_degree(src,dst);
            });
        double load_rep_time = 0;
        load_rep_time -= get_time();
       // graph->load_replicate3(graph->gnnctx->layer_size);
        load_rep_time += get_time();
        if (graph->partition_id == 0)
        printf("#load_rep_time=%lf(s)\n", load_rep_time);

    }
    void init_nn(){
        
        learn_rate = graph->config->learn_rate;
        weight_decay=graph->config->weight_decay;
        drop_rate=graph->config->drop_rate;
        alpha=graph->config->learn_rate;
        decay_rate=graph->config->decay_rate;
        decay_epoch=graph->config->decay_epoch;
        beta1=0.9;
        beta2=0.999;
        epsilon=1e-9;
        
        GNNDatum *gnndatum = new GNNDatum(graph->gnnctx,graph);
        graph->cachedData=new CachedData(graph->partitions,graph->gnnctx->layer_size.size());
        if(0==graph->config->feature_file.compare("random")){
            gnndatum->random_generate();
        }else{
            gnndatum->readFeature_Label_Mask(graph->config->feature_file,graph->config->label_file,graph->config->mask_file);
        }
        gnndatum->registLabel(L_GT_C);
        gnndatum->registMask(MASK);
        L_GT_G = L_GT_C.cuda();
        MASK_gpu=MASK.cuda();
        
        for(int i=0;i<graph->gnnctx->layer_size.size()-1;i++){
            P.push_back(new Parameter(graph->gnnctx->layer_size[i], 
                        graph->gnnctx->layer_size[i+1],alpha,beta1,beta2,epsilon,weight_decay));
            P.push_back(new Parameter(2*graph->gnnctx->layer_size[i+1], 
                        1,alpha,beta1,beta2,epsilon,weight_decay));
        }
        
        torch::Device GPU(torch::kCUDA, 0);
        for(int i=0;i<P.size();i++){
            P[i]->init_parameter();
            P[i]->set_decay(decay_rate,decay_epoch);
            P[i]->to(GPU);
            P[i]->Adam_to_GPU();
            
        }
        drpmodel=torch::nn::Dropout(torch::nn::DropoutOptions().p(drop_rate).inplace(true));
        
//        F=graph->Nts->NewOnesTensor({graph->gnnctx->l_v_num, graph->gnnctx->layer_size[0]},torch::DeviceType::CPU);
        
        F=graph->Nts->NewLeafTensor(gnndatum->local_feature,
                {graph->gnnctx->l_v_num,graph->gnnctx->layer_size[0]},
                                     torch::DeviceType::CPU);
        
        for(int i=0;i<graph->gnnctx->layer_size.size()+1;i++){
            X.push_back(graph->Nts->NewKeyTensor(
                                {graph->gnnctx->l_v_num, 
                                   graph->gnnctx->layer_size[i]},
                                       torch::DeviceType::CUDA));
        if(i<graph->gnnctx->layer_size.size()){
            NtsVar d;M_grad.push_back(d);
            }
        }      
                
        for(int i=0;i<graph->gnnctx->layer_size.size();i++){
            NtsVar d;M.push_back(d);
            for(int j=0;j<graph->partitions;j++){
                mirror_in.push_back(d);
                E_in.push_back(d);
                E_out.push_back(d);
                E_out_grad.push_back(d);
            }
        }
        X[0]=F.cuda();
    }

//inline NtsVar preComputation(NtsVar& master_input){
//     size_t layer=graph->rtminfo->curr_layer;
//        return P[layer*2]->forward(master_input).set_requires_grad(true);
//}  
NtsVar edge_Forward(NtsVar &master_input){
     //init processor
     size_t layer=graph->rtminfo->curr_layer;
     torch::Tensor src_input_transfered=P[layer*2]->forward(master_input);
     int feature_size=src_input_transfered.size(1);
     
     graph->Nts->InitBlock(subgraphs[0], graph->rtminfo, feature_size,
                            feature_size, 0, layer);
     mirror_in[0]=graph->Nts->NewKeyTensor(src_input_transfered);
     mirror_in[0].set_data(src_input_transfered);
     NtsVar m_i_t=graph->Nts->ScatterSrc(mirror_in[0]);
     E_in[0]=torch::cat({m_i_t, graph->Nts->ScatterDst(src_input_transfered)},1);  
     E_out[0]=torch::leaky_relu(torch::exp(P[layer*2+1]->forward(E_in[0])),1.0); 
     //E_out[0].backward(torch::ones_like(E_out[0]));
     //std::cout<<"hello "<<mirror_in[0].size(0)<<" "<<mirror_in[0].size(1)<<" "<<mirror_in[0].grad().dim()<<std::endl;
//     E_in[pos(graph->cpp,layer)]=nts->PrepareMessage(E_in[pos(graph->cpp,layer)]);
//     E_in[pos(graph->cpp,layer)]=E_in[pos(graph->cpp,layer)].coalesce();
//     std::cout<<"before"<<E_in[pos(graph->cpp,layer)].values().size(0)<<" "<<E_in[pos(graph->cpp,layer)].values().size(1)<<std::endl;
//     E_out[pos(graph->cpp,layer)]=at::_sparse_softmax(E_in[pos(graph->cpp,layer)],graph->Nts->BYDST());
     return m_i_t*E_out[0];    
}

NtsVar edge_Backward(NtsVar &message_grad, NtsVar &src_input_trans, 
                                    NtsScheduler* nts){
     size_t layer =graph->rtminfo->curr_layer;
     src_input_trans=mirror_in[pos(graph->cpp,layer)];
     std::cout<<"a.grad "<<src_input_trans.grad().dim()<<std::endl;
//     mirror_in[pos(graph->cpp,layer)]=src_input_transfered;
//     E_in[pos(graph->cpp,layer)]=torch::cat({nts->ScatterSrc(src_input_transfered), nts->ScatterDst(dst_input_transfered)},1);                    
     E_out[pos(graph->cpp,layer)]=torch::leaky_relu(torch::exp(P[layer*2+1]->forward(E_in[pos(graph->cpp,layer)])),1.0);  
     
    //std::cout<<"before"<<message_grad.size(0)<<" "<<message_grad.size(1)<<std::endl;
//    NtsVar msg_grad=at::_sparse_softmax_backward_data(
//               nts->PrepareMessage(E_in[pos(graph->cpp,layer)].indices(),message_grad),
//                E_out[pos(graph->cpp,layer)],
//                 nts->BYDST(),
//                  E_in[pos(graph->cpp,layer)]);
     std::cout<<"E_in"<<E_in[pos(graph->cpp,layer)].size(0)<<std::endl;
      E_out[pos(graph->cpp,layer)].backward(torch::ones_like(E_out[pos(graph->cpp,layer)]),true);
//      I_data["src_trans"].backward(src_grad,true);
      //std::cout<<"hello "<<src_input_trans.size(0)<<" "<<src_input_trans.size(1)<<" "<<src_input_trans.grad().dim()<<std::endl;
      return src_input_trans.grad();    
}


void Loss(){
    torch::Tensor a=X[graph->gnnctx->layer_size.size()-1].log_softmax(1);
    torch::Tensor mask_train=MASK_gpu.eq(0);
     loss=torch::nll_loss(
                a.masked_select(mask_train
                        .expand({mask_train.size(0),a.size(1)}))
                            .view({-1,a.size(1)}),
                L_GT_G.masked_select(mask_train.view({mask_train.size(0)})));
}

void Backward(){
    graph->rtminfo->forward = false;
    loss.backward();
    
    for(int i=graph->gnnctx->layer_size.size()-2;i>=0;i--){
    graph->rtminfo->curr_layer = i;

    NtsVar grad_to_X=X[i+1].grad();
    X[i+1].grad().zero_();
   // grad_to_X=2*torch::ones_like(X[i+1]);
   // if(i==0)std::cout<<M_grad[i]<<std::endl;
    gt->BackwardScatterMessage(grad_to_X,  M_grad[i], subgraphs);
  //  if(i==0)std::cout<<M_grad[i]<<std::endl;
    M[i].backward(M_grad[i]);    
    }
    
}

void Forward(){
    graph->rtminfo->forward = true;    
    for(int i=0;i<graph->gnnctx->layer_size.size()-1;i++){
        graph->rtminfo->curr_layer = i;
        M[i]=edge_Forward(X[i]);
//        torch::Tensor k=torch::ones_like(M[i]);
        gt->ForwardAggMessage(M[i],X[i+1],subgraphs);
//        int test=2700;
//        if(i==0)std::cout<<"DEBUG "<<subgraphs[0]->column_offset[test+1]-subgraphs[0]->column_offset[test]<<" "<<X[i+1][test][0]<<std::endl;
        this->M_grad[i]=graph->Nts->NewLeafTensor(M[i]);
    }
//    printf("finish forward\n");
}

void Update(){
    for(int i=0;i<P.size()-1;i++){
        P[i]->all_reduce_to_gradient(P[i]->W.grad().cpu());
        P[i]->learnC2G_with_decay_Adam();
        P[i]->next();
    }
}



/*GPU dist*/ void run()
{
    if (graph->partition_id == 0)
        printf("GNNmini::Engine[Dist.GPU.GCNimpl] running [%d] Epochs\n",iterations);
       // graph->print_info();
        
    exec_time -= get_time();
    for (int i_i = 0; i_i < iterations; i_i++){
        graph->rtminfo->epoch = i_i;
        if (i_i != 0){
            for(int i=0;i<P.size();i++){
                P[i]->zero_grad();
            }
        }
        
       Forward();
//       Test(0);
//       Test(1);
//       Test(2);
       Loss();
       Backward();
       Update();
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
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
    std::vector<CSC_segment_pinned *>csc_segment;//discard
    std::vector<CSC_segment_pinned *>subgraphs;
    //NN
    GNNDatum *gnndatum;
    GnnUnit *W1;
    GnnUnit *W2;
    GnnUnit *a1;
    GnnUnit *a2;
    torch::Tensor target;
    torch::Tensor target_gpu;
    std::map<std::string,torch::Tensor>I_data;
    typedef std::map<std::string,torch::Tensor> VARS;
    GTensor<ValueType, long, MAX_LAYER> *gt;
    //Tensor
    torch::Tensor new_combine_grad;// = torch::zeros({graph->gnnctx->layer_size[0], graph->gnnctx->layer_size[1]}, torch::kFloat).cuda();
    torch::Tensor inter1_gpu;// = torch::zeros({graph->gnnctx->l_v_num, graph->gnnctx->layer_size[1]}, at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
    torch::Tensor X0_cpu;// = torch::ones({graph->gnnctx->l_v_num, graph->gnnctx->layer_size[0]}, at::TensorOptions().dtype(torch::kFloat));
    torch::Tensor X0_gpu;
    torch::Tensor X1_gpu;
    torch::Tensor Y0_gpu;
    torch::Tensor Y1_gpu;
    torch::Tensor X1_gpu_grad;
    torch::Tensor X0_gpu_grad;
    torch::Tensor Out0_gpu;
    torch::Tensor Out1_gpu;
    torch::Tensor loss;
    torch::Tensor tt;
    float *Y0_cpu_buffered;
    
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
            //rtminfo initialize
        graph->init_rtminfo();
        graph->rtminfo->process_local = graph->config->process_local;
        graph->rtminfo->reduce_comm = graph->config->process_local;
        graph->rtminfo->copy_data = false;
        graph->rtminfo->process_overlap = graph->config->overlap;
        graph->rtminfo->with_weight=false;
       
        
    } 
    void init_graph(){
        //std::vector<CSC_segment_pinned *> csc_segment;
        graph->generate_COO(active);
        graph->reorder_COO_W2W();
        generate_CSC_Segment_Tensor_pinned(graph, csc_segment, true);
        generate_Forward_Segment_Tensor_pinned(graph, subgraphs, true);
        //if (graph->config->process_local)
        double load_rep_time = 0;
        load_rep_time -= get_time();
        graph->load_replicate3(graph->gnnctx->layer_size);
        load_rep_time += get_time();
        if (graph->partition_id == 0)
        printf("#load_rep_time=%lf(s)\n", load_rep_time);
        graph->init_blockinfo();
        graph->init_message_map_amount();
        graph->init_message_buffer();
        gt = new GTensor<ValueType, long, MAX_LAYER>(graph, active);
    }
    void init_nn(){
    GNNDatum *gnndatum = new GNNDatum(graph->gnnctx);
    gnndatum->random_generate();
    gnndatum->registLabel(target);
    target_gpu = target.cuda();
    W1 = new GnnUnit(graph->gnnctx->layer_size[0], graph->gnnctx->layer_size[1]);
    W2 = new GnnUnit(graph->gnnctx->layer_size[1], graph->gnnctx->layer_size[2]);
    a1 = new GnnUnit(2*graph->gnnctx->layer_size[1],1);
    a2 = new GnnUnit(2*graph->gnnctx->layer_size[2],1);
    W1->init_parameter();
    W2->init_parameter();
    a1->init_parameter();
    a2->init_parameter();
    torch::Device GPU(torch::kCUDA, 0);
    W2->to(GPU);
    W1->to(GPU);
    a2->to(GPU);
    a1->to(GPU);
    
    //X0_cpu = torch::ones({graph->gnnctx->l_v_num, graph->gnnctx->layer_size[0]}, at::TensorOptions().dtype(torch::kFloat));
    X0_cpu=graph->EdgeOp->NewLeafTensor({graph->gnnctx->l_v_num, graph->gnnctx->layer_size[0]},torch::DeviceType::CPU);
    //X0_gpu = torch::zeros({graph->gnnctx->l_v_num, graph->gnnctx->layer_size[0]}, at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
    X0_gpu=graph->EdgeOp->NewKeyTensor({graph->gnnctx->l_v_num, graph->gnnctx->layer_size[0]},torch::DeviceType::CUDA);
    //X1_gpu.set_requires_grad(true);
    Y0_gpu = graph->EdgeOp->NewKeyTensor({graph->gnnctx->l_v_num, graph->gnnctx->layer_size[1]},torch::DeviceType::CUDA);
    Y1_gpu = graph->EdgeOp->NewKeyTensor({graph->gnnctx->l_v_num, graph->gnnctx->layer_size[2]},torch::DeviceType::CUDA);
    X1_gpu_grad = graph->EdgeOp->NewKeyTensor({graph->gnnctx->l_v_num, graph->gnnctx->layer_size[1]},torch::DeviceType::CUDA);
    X0_gpu_grad = graph->EdgeOp->NewKeyTensor({graph->gnnctx->l_v_num, graph->gnnctx->layer_size[0]},torch::DeviceType::CUDA);
//    Y0_cpu_buffered = (float *)cudaMallocPinned(((long)graph->vertices) * graph->gnnctx->max_layer * sizeof(float));
//    if (Y0_cpu_buffered == NULL)
//        printf("allocate fail\n");
    }

    
inline torch::Tensor preComputation(torch::Tensor& master_input){
     size_t layer=graph->rtminfo->curr_layer;
     if(layer==0){
        return W1->forward(master_input);
     }else{
        return W2->forward(master_input);
     }
}   

torch::Tensor vertexForward(torch::Tensor &a, torch::Tensor &x){
    
    size_t layer=graph->rtminfo->curr_layer;
    //printf("called %d\n",layer);
    if(layer==0){
        return torch::relu(a).set_requires_grad(true);

    }
    else if(layer==1){
        //printf("called\n");
        torch::Tensor y = torch::relu(a);
        y = y.log_softmax(1); //CUDA
        y = torch::nll_loss(y, target_gpu);
        return y;
    }
}
                             //grad to message            //grad to src from aggregate   
  torch::Tensor edge_Backward(torch::Tensor &message_grad, torch::Tensor &src_grad, EdgeNNModule* edgeop){
      //VARS I_data=edgeop->InterVar;
     edgeop->with_weight=true;
             edgeop->InterVar;
      I_data["msg_grad_1"]=at::_sparse_softmax_backward_data(
               edgeop->PrepareMessage(message_grad),
                I_data["atten"],
                 edgeop->BYDST(),
                  I_data["w"]);
      
      I_data["msg"].backward(I_data["msg_grad_1"].coalesce().values(),true);
      I_data["src_data"].backward(src_grad,true);
      return I_data["fetched_src_data"].grad();
       
 }
  torch::Tensor edge_Forward(torch::Tensor &src_input, torch::Tensor &src_input_transfered, 
                             torch::Tensor &dst_input, torch::Tensor &dst_input_transfered, 
                                                                        EdgeNNModule* edgeop){
     if(graph->rtminfo->forward==true){
        edgeop->SerializeToCPU("cached_src_data",src_input);
        if(graph->rtminfo->curr_layer==0){
            I_data["src_data"]=W1->forward(src_input);
        }else{
            I_data["src_data"]=W2->forward(src_input);
        }
     }
     else{
        //I_data["fetched_src_data"]=edgeop->DeSerializeTensorToGPU(I_data[graph->VarEncode("cached_src_data")]);
         I_data["fetched_src_data"]=edgeop->DeSerializeFromCPU("cached_src_data");
        if(graph->rtminfo->curr_layer==0){
            I_data["src_data"]=W1->forward(I_data["fetched_src_data"]);
        }else{
            I_data["src_data"]=W2->forward(I_data["fetched_src_data"]);
        }
     }
        src_input_transfered=I_data["src_data"];
        I_data["dst_data"]=dst_input_transfered;//finish apply W
        I_data["msg"]=torch::cat({edgeop->ScatterSrc(I_data["src_data"]),
                                   edgeop->ScatterDst(I_data["dst_data"])}
                                   ,1);
     if(graph->rtminfo->curr_layer==0){                         
        I_data["msg"]=torch::leaky_relu(torch::exp(a1->forward(I_data["msg"])),1.0);  
     }else{
        I_data["msg"]=torch::leaky_relu(torch::exp(a2->forward(I_data["msg"])),1.0);  
     }
          
        I_data["w"]=edgeop->PrepareMessage(I_data["msg"]);
        I_data["atten"]=at::_sparse_softmax(I_data["w"],graph->EdgeOp->BYDST());
        edgeop->with_weight=true;
        return I_data["atten"].coalesce().values();    
     }
 
/* 
 * libtorch 1.7 and its higher versions have conflict 
 * with the our openmp based parallel processing that inherit from Gemini [OSDI 2016].
 * So in this example we use Libtorch 1.5 as auto differentiation tool.
 * As 'autograd' function is not explict supported in C++ release of libtorch 1.5, we illustrate 
 * the example in a implicit manner.
 */
void vertexBackward(){
    
    int layer=graph->rtminfo->curr_layer;
    if(layer==0){
        X1_gpu.backward(X1_gpu_grad); //new
    }
    else if(layer==1){
        loss.backward();
    }
}



void Allbackward(){
    graph->rtminfo->forward = false;
    graph->rtminfo->curr_layer = 1;
    vertexBackward();
    torch::Tensor grad_to_Y1=Y1_gpu.grad();
    gt->GraphPropagateBackwardEdgeComputation(X1_gpu, 
                                              grad_to_Y1,                                              
                                              X1_gpu_grad,
                                              subgraphs,
                                              [&](torch::Tensor &master_input){//pre computation
                                                   return preComputation(master_input);
                                              },
                                              [&](torch::Tensor &mirror_input,torch::Tensor &mirror_input_transfered,
                                                       torch::Tensor &X, torch::Tensor &X_trans, EdgeNNModule* edgeop){//edge computation
                                                   return edge_Forward(mirror_input,mirror_input_transfered,X, X_trans, edgeop);
                                              },
                                              [&](torch::Tensor &b,torch::Tensor &c, EdgeNNModule* edgeop){//edge computation
                                                   return edge_Backward(b, c,edgeop);
                                              });
    graph->rtminfo->curr_layer = 0;
    vertexBackward();
    torch::Tensor grad_to_Y0=Y0_gpu.grad();
    gt->GraphPropagateBackwardEdgeComputation(X0_gpu, 
                                              grad_to_Y0,                                              
                                              X0_gpu_grad,
                                              subgraphs,
                                              [&](torch::Tensor &master_input){//pre computation
                                                   return preComputation(master_input);
                                              },
                                              [&](torch::Tensor &mirror_input,torch::Tensor &mirror_input_transfered,
                                                       torch::Tensor &X, torch::Tensor &X_trans, EdgeNNModule* edgeop){//edge computation
                                                   return edge_Forward(mirror_input,mirror_input_transfered,X, X_trans, edgeop);
                                              },
                                              [&](torch::Tensor &b,torch::Tensor &c, EdgeNNModule* edgeop){//edge computation
                                                   return edge_Backward(b, c,edgeop);
                                              }); 

   
    W2->all_reduce_to_gradient(W2->W.cpu()); //W2->W.grad().cpu()
    W2->learnC2G_with_decay(learn_rate,weight_decay);
    
    a2->all_reduce_to_gradient(a2->W.cpu()); //W2->W.grad().cpu()
    a2->learnC2G_with_decay(learn_rate,weight_decay);  
    
    W1->all_reduce_to_gradient(W1->W.cpu());
    W1->learnC2G_with_decay(learn_rate,weight_decay);
    
    a1->all_reduce_to_gradient(a1->W.cpu());
    a1->learnC2G_with_decay(learn_rate,weight_decay); 
}

/*GPU dist*/ void forward()
{
    if (graph->partition_id == 0)
        printf("GNNmini::Engine[Dist.GPU.GCNimpl] running [%d] Epochs\n",iterations);
        graph->print_info();

    exec_time -= get_time();
    for (int i_i = 0; i_i < iterations; i_i++)
    {
        if (i_i != 0)
        {
            W1->zero_grad();
            W2->zero_grad();
            a1->zero_grad();
            a2->zero_grad();
        }
        
        graph->rtminfo->epoch = i_i;
        graph->rtminfo->curr_layer = 0;
        graph->rtminfo->forward = true;
        torch::Tensor X0_gpu_trans;
        gt->GraphPropagateForwardEdgeComputation(X0_gpu,X0_gpu_trans,Y0_gpu,subgraphs,
                [&](torch::Tensor &master_input){//pre computation
                   return preComputation(master_input);
                },
                [&](torch::Tensor &mirror_input,torch::Tensor &mirror_input_transfered,
                           torch::Tensor &X, torch::Tensor &X_trans, EdgeNNModule* edgeop){//edge computation
                               return edge_Forward(mirror_input,mirror_input_transfered,X, X_trans, edgeop);
                 });
        X1_gpu=vertexForward(Y0_gpu, X0_gpu_trans);
       
        
        graph->rtminfo->curr_layer = 1;
        torch::Tensor X1_gpu_trans;
        gt->GraphPropagateForwardEdgeComputation(X1_gpu,X1_gpu_trans,Y1_gpu,subgraphs,
                [&](torch::Tensor &master_input){//pre computation
                   return preComputation(master_input);
                },
                [&](torch::Tensor &mirror_input,torch::Tensor &mirror_input_transfered,
                           torch::Tensor &X, torch::Tensor &X_trans, EdgeNNModule* edgeop){//edge computation
                               return edge_Forward(mirror_input,mirror_input_transfered,X, X_trans, edgeop);
                 });
        loss=vertexForward(Y1_gpu, X1_gpu_trans);
        
        Allbackward();
        
        if (graph->partition_id == 0)
            std::cout << "GNNmini::Running.Epoch["<<i_i<<"]:loss\t" << loss << std::endl;
        
        if (i_i == (iterations - 1))
        { 
            exec_time += get_time();
            //DEBUGINFO();
        }
         
    }

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
    //      torch::Tensor tt_cpu=tt.cpu();
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
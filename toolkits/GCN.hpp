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
    std::vector<CSC_segment_pinned *>csc_segment;//discard
    std::vector<CSC_segment_pinned *>subgraphs;
    //NN
    GNNDatum *gnndatum;
    GnnUnit *W1;
    GnnUnit *W2;
    torch::Tensor target;
    torch::Tensor target_gpu;
    std::map<std::string,torch::Tensor>I_data;
    GTensor<ValueType, long, MAX_LAYER> *gt;
    //Tensor
    torch::Tensor new_combine_grad;// = torch::zeros({graph->gnnctx->layer_size[0], graph->gnnctx->layer_size[1]}, torch::kFloat).cuda();
    torch::Tensor inter1_gpu;// = torch::zeros({graph->gnnctx->l_v_num, graph->gnnctx->layer_size[1]}, at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
    torch::Tensor X0_cpu;// = torch::ones({graph->gnnctx->l_v_num, graph->gnnctx->layer_size[0]}, at::TensorOptions().dtype(torch::kFloat));
    torch::Tensor X0_gpu;
    torch::Tensor Y0_gpu;
    torch::Tensor Y01_gpu; 
    torch::Tensor Y1_gpu; 
    torch::Tensor Y1_inv_gpu;
    torch::Tensor Out0_gpu;
    torch::Tensor Out1_gpu;
    torch::Tensor loss;
    torch::Tensor tt;
    
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
        graph->rtminfo->with_weight=false;
        graph->rtminfo->with_cuda=true;
       
    } 
    void init_graph(){
        //std::vector<CSC_segment_pinned *> csc_segment;
        graph->generate_COO(active);
        graph->reorder_COO_W2W();
        //generate_CSC_Segment_Tensor_pinned(graph, csc_segment, true);
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
    W1->init_parameter();
    W2->init_parameter();
    torch::Device GPU(torch::kCUDA, 0);
    W2->to(GPU);
    W1->to(GPU);
    
    X0_cpu = torch::ones({graph->gnnctx->l_v_num, graph->gnnctx->layer_size[0]}, at::TensorOptions().dtype(torch::kFloat));
    X0_gpu = X0_cpu.cuda();
    Y0_gpu = torch::zeros({graph->gnnctx->l_v_num, graph->gnnctx->layer_size[0]}, at::TensorOptions().device_index(0).dtype(torch::kFloat));
    Y01_gpu = torch::ones({graph->gnnctx->l_v_num, graph->gnnctx->layer_size[1]}, at::TensorOptions().device_index(0).dtype(torch::kFloat));
    Y1_gpu = torch::zeros({graph->gnnctx->l_v_num, graph->gnnctx->layer_size[1]}, at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
    Y1_inv_gpu = torch::zeros({graph->gnnctx->l_v_num, graph->gnnctx->layer_size[1]}, at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
    
    }

void vertexForward(torch::Tensor &a, torch::Tensor &x, torch::Tensor &y){
    
    int layer=graph->rtminfo->curr_layer;
    if(layer==0){
        y=torch::relu(W1->forward(a));

    }
    else if(layer==1){
        Out1_gpu = W2->forward(Y1_gpu);
        tt = Out1_gpu.log_softmax(1); //CUDA
        y = torch::nll_loss(tt, target_gpu);   
    }
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
        Out0_gpu.backward(Y1_inv_gpu); //new
    }
    else if(layer==1){
        loss.backward();
    }
}

void Allbackward(){
    graph->rtminfo->curr_layer = 1;
    vertexBackward();
    W2->all_reduce_to_gradient(W2->W.cpu()); //W2->W.grad().cpu()
    W2->learnC2G_with_decay(learn_rate,weight_decay);
    
    torch::Tensor y1grad=Y1_gpu.grad();
    gt->GraphPropagateBackward(y1grad, Y1_inv_gpu, subgraphs);
    
    graph->rtminfo->curr_layer = 0;
    vertexBackward();
    W1->all_reduce_to_gradient(W1->W.cpu());
    W1->learnC2G_with_decay(learn_rate,weight_decay);
    
        
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


        }
        
        graph->rtminfo->epoch = i_i;
        
        graph->rtminfo->forward = true;
        graph->rtminfo->curr_layer = 0;
        gt->GraphPropagateForward(X0_gpu, Y0_gpu, subgraphs);
        vertexForward(Y0_gpu, X0_gpu, Out0_gpu);
        
        graph->rtminfo->curr_layer = 1;
        gt->GraphPropagateForward(Out0_gpu, Y1_gpu, subgraphs);
        vertexForward(Y1_gpu, Out0_gpu, loss);

        graph->rtminfo->forward = false;
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
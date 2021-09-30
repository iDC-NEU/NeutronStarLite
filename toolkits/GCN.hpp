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
    std::vector<CSC_segment_pinned *>csc_segment;
    std::vector<CSC_segment_pinned *>forward_csc_segment;
    //NN
    GNNDatum *gnndatum;
    GnnUnit *Gnn_v1;
    GnnUnit *Gnn_v2;
    torch::Tensor target;
    torch::Tensor target_gpu;
    GTensor<ValueType, long, MAX_LAYER> *gt;
    //Tensor
    torch::Tensor new_combine_grad;// = torch::zeros({graph->gnnctx->layer_size[0], graph->gnnctx->layer_size[1]}, torch::kFloat).cuda();
    torch::Tensor inter1_gpu;// = torch::zeros({graph->gnnctx->l_v_num, graph->gnnctx->layer_size[1]}, at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
    torch::Tensor X0_cpu;// = torch::ones({graph->gnnctx->l_v_num, graph->gnnctx->layer_size[0]}, at::TensorOptions().dtype(torch::kFloat));
    torch::Tensor X0_gpu;
    torch::Tensor Y0_gpu;
    torch::Tensor Y1_gpu; 
    torch::Tensor Y1_inv_gpu;
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
       
        
    } 
    void init_graph(){
        //std::vector<CSC_segment_pinned *> csc_segment;
        graph->generate_COO(active);
        graph->reorder_COO_W2W();
        generate_CSC_Segment_Tensor_pinned(graph, csc_segment, true);
        generate_Forward_Segment_Tensor_pinned(graph, forward_csc_segment, true);
        //if (graph->config->process_local)
        double load_rep_time = 0;
        load_rep_time -= get_time();
        graph->load_replicate3(graph->gnnctx->layer_size);
        load_rep_time += get_time();
        if (graph->partition_id == 0)
        printf("#load_rep_time=%lf(s)\n", load_rep_time);
        graph->init_blockinfo();
        graph->init_message_map_amount();
        gt = new GTensor<ValueType, long, MAX_LAYER>(graph, active);
    }
    void init_nn(){
    GNNDatum *gnndatum = new GNNDatum(graph->gnnctx);
    gnndatum->random_generate();
    gnndatum->registLabel(target);
    target_gpu = target.cuda();
    Gnn_v1 = new GnnUnit(graph->gnnctx->layer_size[0], graph->gnnctx->layer_size[1]);
    Gnn_v2 = new GnnUnit(graph->gnnctx->layer_size[1], graph->gnnctx->layer_size[2]);
    Gnn_v1->init_parameter();
    Gnn_v2->init_parameter();
    torch::Device GPU(torch::kCUDA, 0);
    Gnn_v2->to(GPU);
    Gnn_v1->to(GPU);
    
    X0_cpu = torch::ones({graph->gnnctx->l_v_num, graph->gnnctx->layer_size[0]}, at::TensorOptions().dtype(torch::kFloat));
    X0_gpu = X0_cpu.cuda();
    Y0_gpu = torch::zeros({graph->gnnctx->l_v_num, graph->gnnctx->layer_size[0]}, at::TensorOptions().device_index(0).dtype(torch::kFloat));
    Y1_gpu = torch::zeros({graph->gnnctx->l_v_num, graph->gnnctx->layer_size[1]}, at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
    Y1_inv_gpu = torch::zeros({graph->gnnctx->l_v_num, graph->gnnctx->layer_size[1]}, at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
    Y0_cpu_buffered = (float *)cudaMallocPinned(((long)graph->vertices) * graph->gnnctx->max_layer * sizeof(float));
    if (Y0_cpu_buffered == NULL)
        printf("allocate fail\n");
    
    }

void vertexForward(torch::Tensor &a, torch::Tensor &x, torch::Tensor &y){
    
    int layer=graph->rtminfo->curr_layer;
    if(layer==0){
        y=torch::relu(Gnn_v1->forward(a));

    }
    else if(layer==1){
        Out1_gpu = Gnn_v2->forward(Y1_gpu);
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
        Gnn_v1->all_reduce_to_gradient(Gnn_v2->W.cpu());
        //Gnn_v1->learnC2G(learn_rate);
        Gnn_v1->learnC2G_with_decay(learn_rate,weight_decay);

    }
    else if(layer==1){
        loss.backward();
        Gnn_v2->all_reduce_to_gradient(Gnn_v2->W.cpu()); //Gnn_v2->W.grad().cpu()
        //Gnn_v2->learnC2G(learn_rate);
        Gnn_v2->learnC2G_with_decay(learn_rate,weight_decay);
    }
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
    //     inference(tt_cpu,graph, embedding, pytool,Gnn_v1,Gnn_v2);
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



void Allbackward(){
    graph->rtminfo->curr_layer = 1;
    vertexBackward();
    //gt->Process_GPU_overlap_explict(Y1_gpu.grad(), Y0_cpu_buffered, Y1_inv_gpu, csc_segment);
    gt->GraphPropagateBackward(Y1_gpu.grad(), Y0_cpu_buffered, Y1_inv_gpu, csc_segment);
    graph->rtminfo->curr_layer = 0;
    vertexBackward();
        
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
            Gnn_v1->zero_grad();
            Gnn_v2->zero_grad();
            //Gnn_v1->W*=(1-weight_decay);
            //Gnn_v2->W*=(1-weight_decay);
        }
        
        int test_id=34;
        graph->rtminfo->epoch = i_i;
        
        graph->rtminfo->curr_layer = 0;
        gt->GraphPropagateForward(X0_gpu, Y0_cpu_buffered, Y0_gpu, forward_csc_segment);
//        if(graph->partition_id==1)
//        std::cout <<test_id + graph->partition_offset[graph->partition_id]<<" "<< graph->out_degree_for_backward[test_id + graph->partition_offset[graph->partition_id]] << " " << Y0_gpu[test_id][0] << std::endl;
      
        vertexForward(Y0_gpu, X0_gpu, Out0_gpu);
        
        graph->rtminfo->curr_layer = 1;
        gt->GraphPropagateForward(Out0_gpu, Y0_cpu_buffered, Y1_gpu, forward_csc_segment);
        //gt->Process_GPU_overlap_lite(Out0_gpu, Y0_cpu_buffered, Y1_gpu, csc_segment);
        vertexForward(Y1_gpu, Out0_gpu, loss);

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
};

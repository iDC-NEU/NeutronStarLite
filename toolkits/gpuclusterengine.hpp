#include "core/gnnmini.hpp"
#include <c10/cuda/CUDAStream.h>

void compute_dist_GPU(Graph<Empty> *graph, int iterations)
{
    ValueType learn_rate = 0.01;
    VertexSubset *active = graph->alloc_vertex_subset();
    active->fill();
    std::cout << "nani" << std::endl;
    Embeddings<ValueType, long> *embedding = new Embeddings<ValueType, long>();
    embedding->init(graph);
    embedding->readlabel_(graph);

    Network<ValueType> *comm = new Network<ValueType>(graph, MAX_LAYER, MAX_LAYER);
    Network<ValueType> *comm1 = new Network<ValueType>(graph, MAX_LAYER, MAX_LAYER);
    comm->setWsize(MAX_LAYER, MAX_LAYER);
    comm1->setWsize(MAX_LAYER, MAX_LAYER);
    tensorSet *pytool = new tensorSet(2);
    //pytool->in_degree=torch::from_blob(graph->in_degree+graph->partition_offset[graph->partition_id],{embedding->rownum,1});
    //pytool->out_degree=torch::from_blob(graph->in_degree+graph->partition_offset[graph->partition_id],{embedding->rownum,1});
    /*init GPU*/
    aggregate_engine *ag_e = new aggregate_engine();
    ag_e->reconfig_data(graph->owned_vertices, SIZE_LAYER_2, graph->owned_vertices, SIZE_LAYER_1, TENSOR_TYPE);
    ag_e->init_intermediate_gradient();

    graph_engine *gr_e = new graph_engine();
    // graph->generate_COO(active);
    // std::cout<<graph->partition_offset[0]<<" "<<graph->partition_offset[1]<<" "<<graph->partition_offset[2]<<std::endl;
    // graph->reorder_COO();

    std::cout << graph->owned_vertices << " " << graph->edges << std::endl;

    VertexId *incoming_adj_index = new VertexId[graph->vertices + 1];
    VertexId *incoming_adj_index_backward = new VertexId[graph->vertices + 1];
    for (VertexId vtx = 0; vtx < graph->vertices; vtx++)
    {
        if (graph->incoming_adj_index[0][vtx + 1] == 0)
            graph->incoming_adj_index[0][vtx + 1] = graph->incoming_adj_index[0][vtx];
        if (graph->incoming_adj_index_backward[0][vtx + 1] == 0)
            graph->incoming_adj_index_backward[0][vtx + 1] = graph->incoming_adj_index_backward[0][vtx];
    }

    EdgeId edges_for = graph->incoming_adj_index[0][graph->vertices];
    EdgeId edges_back = graph->incoming_adj_index_backward[0][graph->vertices];
    ValueType *weight = new ValueType[edges_for + 1];
    ValueType *weight_backward = new ValueType[edges_back + 1];

    for (VertexId vtx = 0; vtx < graph->vertices; vtx++)
    {
        incoming_adj_index[vtx] = (VertexId)graph->incoming_adj_index[0][vtx];
        incoming_adj_index_backward[vtx] = (VertexId)graph->incoming_adj_index_backward[0][vtx];
        for (int i = graph->incoming_adj_index[0][vtx]; i < graph->incoming_adj_index[0][vtx + 1]; i++)
        {
            VertexId dst = graph->incoming_adj_list[0][i].neighbour;
            weight[i] = (ValueType)std::sqrt(graph->in_degree_for_backward[vtx]) * (ValueType)std::sqrt(graph->out_degree_for_backward[dst]);
        }
        for (int i = graph->incoming_adj_index_backward[0][vtx]; i < graph->incoming_adj_index_backward[0][vtx + 1]; i++)
        {
            VertexId dst = graph->incoming_adj_list_backward[0][i].neighbour;
            weight_backward[i] = (ValueType)std::sqrt(graph->out_degree_for_backward[vtx]) * (ValueType)std::sqrt(graph->in_degree_for_backward[dst]);
        }
    }

    incoming_adj_index[graph->vertices] = (VertexId)graph->incoming_adj_index[0][graph->vertices];
    incoming_adj_index_backward[graph->vertices] = (VertexId)graph->incoming_adj_index_backward[0][graph->vertices];
    std::cout << "something error" << incoming_adj_index[graph->vertices - 1] << " " << (VertexId)graph->incoming_adj_index[0][graph->vertices - 1] << std::endl;

    gr_e->load_graph(graph->vertices, edges_for, false,
                     graph->vertices, edges_back, false,
                     incoming_adj_index, (VertexId *)graph->incoming_adj_list[0],
                     incoming_adj_index_backward, (VertexId *)graph->incoming_adj_list_backward[0],
                     graph->partition_offset[graph->partition_id], graph->partition_offset[graph->partition_id + 1], 0, graph->vertices, SIZE_LAYER_1);
    /*1 INIT STAGE*/
    GnnUnit *Gnn_v1 = new GnnUnit(SIZE_LAYER_1, SIZE_LAYER_2);
    GnnUnit *Gnn_v2 = new GnnUnit(SIZE_LAYER_2, OUTPUT_LAYER_3);
    std::cout << Gnn_v1->W.size(0) << "   " << Gnn_v1->W.size(1) << " " << graph->partition_id << std::endl;
    pytool->registOptimizer(torch::optim::SGD(Gnn_v1->parameters(), 0.05)); //new
    pytool->registOptimizer(torch::optim::SGD(Gnn_v2->parameters(), 0.05)); //new
    pytool->registLabel<long>(embedding->label, graph->partition_offset[graph->partition_id],
                              graph->partition_offset[graph->partition_id + 1] - graph->partition_offset[graph->partition_id]); //new
                                                                                                                                /*init W with new */

    init_parameter(comm, graph, Gnn_v1, embedding);
    init_parameter(comm1, graph, Gnn_v2, embedding);

    std::vector<int> layer_size(0);
    layer_size.push_back(SIZE_LAYER_1);
    layer_size.push_back(SIZE_LAYER_2);
    layer_size.push_back(OUTPUT_LAYER_3);
    GTensor<ValueType, long, MAX_LAYER> *gt = new GTensor<ValueType, long, MAX_LAYER>(graph, embedding, active, 2, layer_size);

    torch::Tensor new_combine_grad = torch::zeros({SIZE_LAYER_1, SIZE_LAYER_2}, torch::kFloat).cuda();
    graph->process_vertices<ValueType>( //init  the vertex state.
        [&](VertexId vtx) {
            int start = (graph->partition_offset[graph->partition_id]);
            for (int i = 0; i < SIZE_LAYER_1; i++)
            {
                embedding->initStartWith(vtx, embedding->con[vtx].att[i], i); //embedding->con[vtx].att[i]
            }
            return (ValueType)1;
        },
        active);

    /*GPU  */
    torch::Device GPU(torch::kCUDA, 0);
    torch::Device CPU(torch::kCPU, 0);
    torch::Tensor target_gpu = pytool->target.cuda();
    torch::Tensor inter1_gpu = torch::zeros({embedding->rownum, SIZE_LAYER_2}, at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
    Gnn_v2->to(GPU);
    Gnn_v1->to(GPU);

    torch::Tensor X0_gpu = torch::from_blob(embedding->start_v + embedding->start, {embedding->rownum, SIZE_LAYER_1}, torch::kFloat).cuda();
    torch::Tensor Y0_gpu = torch::zeros({embedding->rownum, SIZE_LAYER_1}, at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
    torch::Tensor Y1_gpu = torch::zeros({embedding->rownum, SIZE_LAYER_2}, at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
    torch::Tensor Y1_inv_gpu = torch::zeros({embedding->rownum, SIZE_LAYER_2}, at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
    torch::Tensor W_for_gpu = torch::from_blob(weight, {edges_for, 1}, torch::kFloat).cuda();
    torch::Tensor W_back_gpu = torch::from_blob(weight_backward, {edges_back + 1, 1}, torch::kFloat).cuda();
    torch::Tensor Out0_gpu;
    torch::Tensor Out1_gpu;
    torch::Tensor Y0_gpu_buffered = torch::zeros({graph->vertices, SIZE_LAYER_1},
                                                 at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
    torch::Tensor Y1_gpu_buffered = torch::zeros({graph->vertices, SIZE_LAYER_2},
                                                 at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
    torch::Tensor Y1_inv_gpu_buffered = torch::zeros({graph->vertices, SIZE_LAYER_1},
                                                     at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));

    double exec_time = 0;
    exec_time -= get_time();

    for (int i_i = 0; i_i < iterations; i_i++)
    {
        if (i_i != 0)
        {
            Gnn_v1->zero_grad();
            Gnn_v2->zero_grad();
        }

        if (graph->partition_id == 0)
            std::cout << "start  [" << i_i << "]  epoch+++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
        //layer 1;
        gr_e->forward_one_step(X0_gpu.packed_accessor<float, 2>().data(),
                               Y0_gpu_buffered.packed_accessor<float, 2>().data(),
                               W_for_gpu.packed_accessor<float, 2>().data(), SCALA_TYPE, SIZE_LAYER_1);

        torch::Tensor comm_data = Y0_gpu_buffered.cpu();
        torch::Tensor Y0_cpu = Y0_gpu.cpu();
        gt->Sync_data<SIZE_LAYER_1>(comm_data, Y0_cpu);
        Y0_gpu.set_data(Y0_cpu.cuda());

        inter1_gpu.set_data(Gnn_v1->forward(Y0_gpu)); //torch::from_blob(embedding->Gnn_v1->forward(pytool->x[0]).accessor<float,2>().data(),{embedding->rownum,VECTOR_LENGTH});
        Out0_gpu = torch::relu(inter1_gpu);           //new

        //layer 2;

        gr_e->forward_one_step(Out0_gpu.packed_accessor<float, 2>().data(),
                               Y1_gpu_buffered.packed_accessor<float, 2>().data(),
                               W_for_gpu.packed_accessor<float, 2>().data(), SCALA_TYPE, SIZE_LAYER_2);

        torch::Tensor comm_data1 = Y1_gpu_buffered.cpu();
        torch::Tensor Y1_cpu = Y1_gpu.cpu();
        gt->Sync_data<SIZE_LAYER_2>(comm_data1, Y1_cpu);
        Y1_gpu.set_data(Y1_cpu.cuda());
        Out1_gpu = Gnn_v2->forward(Y1_gpu);
        //output;
        torch::Tensor tt = Out1_gpu.log_softmax(1);     //CUDA
        pytool->loss = torch::nll_loss(tt, target_gpu); //new
        pytool->loss.backward();
        //inv layer 2;
        torch::Tensor aggregate_grad2 = unified_parameter<ValueType>(comm1, Gnn_v2->W.grad().cpu());
        Gnn_v2->learn_gpu(aggregate_grad2.cuda(), learn_rate); //reset from new

        //inv layer 1;
        // 2->1
        gr_e->backward_one_step(Y1_gpu.grad().packed_accessor<float, 2>().data(),
                                Y1_inv_gpu_buffered.packed_accessor<float, 2>().data(),
                                W_back_gpu.packed_accessor<float, 2>().data(), SCALA_TYPE, SIZE_LAYER_2);

        torch::Tensor comm_data2 = Y1_inv_gpu_buffered.cpu();
        torch::Tensor Y1_inv_cpu = Y1_inv_gpu.cpu();
        gt->Sync_data<SIZE_LAYER_2>(comm_data2, Y1_inv_cpu);
        Y1_inv_gpu.set_data(Y1_inv_cpu.cuda());

        // layer 1 local
        Out0_gpu.backward(); //new
                             // layer 1 combine
        new_combine_grad.zero_();
        new_combine_grad = Y0_gpu.t().mm(Y1_inv_gpu * inter1_gpu.grad());

        //learn
        torch::Tensor aggregate_grad = unified_parameter<ValueType>(comm, new_combine_grad.cpu());
        Gnn_v1->learn_gpu(aggregate_grad.cuda(), learn_rate);

        if (graph->partition_id == 0)
            std::cout << "LOSS:\t" << pytool->loss << std::endl;

        if (i_i == (iterations - 1))
        { //&&graph->partition_id==0
            torch::Tensor tt_cpu = tt.cpu();
            if (i_i == (iterations - 1) && graph->partition_id == 0)
            {
                inference(tt_cpu, graph, embedding, pytool, Gnn_v1, Gnn_v2);
            }
        }
    }
    delete active;
}
void compute_dist_gf_GPU(Graph<Empty> *graph, int iterations)
{
    ValueType learn_rate = 0.01;
    VertexSubset *active = graph->alloc_vertex_subset();
    active->fill();
    Embeddings<ValueType, long> *embedding = new Embeddings<ValueType, long>();
    embedding->init(graph);
    embedding->readlabel_(graph);

    Network<ValueType> *comm = new Network<ValueType>(graph, MAX_LAYER, MAX_LAYER);
    Network<ValueType> *comm1 = new Network<ValueType>(graph, MAX_LAYER, MAX_LAYER);
    comm->setWsize(MAX_LAYER, MAX_LAYER);
    comm1->setWsize(MAX_LAYER, MAX_LAYER);
    tensorSet *pytool = new tensorSet(2);
    //pytool->in_degree=torch::from_blob(graph->in_degree+graph->partition_offset[graph->partition_id],{embedding->rownum,1});
    //pytool->out_degree=torch::from_blob(graph->in_degree+graph->partition_offset[graph->partition_id],{embedding->rownum,1});
    /*init GPU*/
    aggregate_engine *ag_e = new aggregate_engine();
    ag_e->reconfig_data(graph->owned_vertices, SIZE_LAYER_2, graph->owned_vertices, SIZE_LAYER_1, TENSOR_TYPE);
    ag_e->init_intermediate_gradient();

    graph_engine *gr_e = new graph_engine();
    // graph->generate_COO(active);
    // std::cout<<graph->partition_offset[0]<<" "<<graph->partition_offset[1]<<" "<<graph->partition_offset[2]<<std::endl;
    // graph->reorder_COO();

    // for(int i=graph->edges-100;i<graph->edges;i++){
    //     std::cout<<graph->_graph_cpu->srcList[i]<<" "<<graph->_graph_cpu->dstList[i]<<std::endl;
    // }
    std::cout << graph->owned_vertices << " " << graph->edges << std::endl;

    VertexId *incoming_adj_index = new VertexId[graph->vertices + 1];
    VertexId *incoming_adj_index_backward = new VertexId[graph->vertices + 1];
    for (VertexId vtx = 0; vtx < graph->vertices; vtx++)
    {
        if (graph->incoming_adj_index[0][vtx + 1] == 0)
            graph->incoming_adj_index[0][vtx + 1] = graph->incoming_adj_index[0][vtx];
        if (graph->incoming_adj_index_backward[0][vtx + 1] == 0)
            graph->incoming_adj_index_backward[0][vtx + 1] = graph->incoming_adj_index_backward[0][vtx];
    }
    EdgeId edges_for = graph->incoming_adj_index[0][graph->vertices];
    EdgeId edges_back = graph->incoming_adj_index_backward[0][graph->vertices];
    ValueType *weight = new ValueType[edges_for + 1];
    ValueType *weight_backward = new ValueType[edges_back + 1];

    for (VertexId vtx = 0; vtx < graph->vertices + 1; vtx++)
    {
        incoming_adj_index[vtx] = (VertexId)graph->incoming_adj_index[0][vtx];
        incoming_adj_index_backward[vtx] = (VertexId)graph->incoming_adj_index_backward[0][vtx];
        for (int i = graph->incoming_adj_index[0][vtx]; i < graph->incoming_adj_index[0][vtx + 1]; i++)
        {
            VertexId dst = graph->incoming_adj_list[0][i].neighbour;
            weight[i] = (ValueType)std::sqrt(graph->in_degree_for_backward[vtx]) * (ValueType)std::sqrt(graph->out_degree_for_backward[dst]);
        }
        for (int i = graph->incoming_adj_index_backward[0][vtx]; i < graph->incoming_adj_index_backward[0][vtx + 1]; i++)
        {
            VertexId dst = graph->incoming_adj_list_backward[0][i].neighbour;
            weight_backward[i] = (ValueType)std::sqrt(graph->out_degree_for_backward[vtx]) * (ValueType)std::sqrt(graph->in_degree_for_backward[dst]);
        }
    }
    incoming_adj_index[graph->vertices] = (VertexId)graph->incoming_adj_index[0][graph->vertices];
    incoming_adj_index_backward[graph->vertices] = (VertexId)graph->incoming_adj_index_backward[0][graph->vertices];
    std::cout << "something error" << incoming_adj_index[graph->vertices - 1] << " " << (VertexId)graph->incoming_adj_index[0][graph->vertices - 1] << std::endl;

    gr_e->load_graph(graph->vertices, edges_for, false,
                     graph->vertices, edges_back, false,
                     incoming_adj_index, (VertexId *)graph->incoming_adj_list[0],
                     incoming_adj_index_backward, (VertexId *)graph->incoming_adj_list_backward[0],
                     graph->partition_offset[graph->partition_id], graph->partition_offset[graph->partition_id + 1], 0, graph->vertices, SIZE_LAYER_1);

    /*1 INIT STAGE*/

    GnnUnit *Gnn_v1 = new GnnUnit(SIZE_LAYER_1, SIZE_LAYER_2);

    GnnUnit *Gnn_v2 = new GnnUnit(SIZE_LAYER_2, OUTPUT_LAYER_3);
    pytool->registOptimizer(torch::optim::SGD(Gnn_v1->parameters(), 0.05)); //new
    pytool->registOptimizer(torch::optim::SGD(Gnn_v2->parameters(), 0.05)); //new

    pytool->registLabel<long>(embedding->label, graph->partition_offset[graph->partition_id],
                              graph->partition_offset[graph->partition_id + 1] - graph->partition_offset[graph->partition_id]); //new
                                                                                                                                /*init W with new */

    init_parameter(comm, graph, Gnn_v1, embedding);
    init_parameter(comm1, graph, Gnn_v2, embedding);

    std::vector<int> layer_size(0);
    layer_size.push_back(SIZE_LAYER_1);
    layer_size.push_back(SIZE_LAYER_2);
    layer_size.push_back(OUTPUT_LAYER_3);
    GTensor<ValueType, long, MAX_LAYER> *gt = new GTensor<ValueType, long, MAX_LAYER>(graph, embedding, active, 2, layer_size);

    torch::Tensor new_combine_grad = torch::zeros({SIZE_LAYER_1, SIZE_LAYER_2}, torch::kFloat).cuda();

    graph->process_vertices<ValueType>( //init  the vertex state.
        [&](VertexId vtx) {
            int start = (graph->partition_offset[graph->partition_id]);
            for (int i = 0; i < SIZE_LAYER_1; i++)
            {
                embedding->initStartWith(vtx, embedding->con[vtx].att[i], i); //embedding->con[vtx].att[i]
            }
            return (ValueType)1;
        },
        active);
    /*GPU  */
    torch::Device GPU(torch::kCUDA, 0);
    torch::Device CPU(torch::kCPU, 0);
    torch::Tensor target_gpu = pytool->target.cuda();
    Gnn_v2->to(GPU);
    Gnn_v1->to(GPU);

    //  torch::Tensor X1_gpu=torch::zeros({embedding->rownum,SIZE_LAYER_1},at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));

    torch::Tensor X0_gpu = torch::from_blob(embedding->start_v + embedding->start, {embedding->rownum, SIZE_LAYER_1}, torch::kFloat).cuda();
    torch::Tensor X0_gpu_trans = torch::zeros({embedding->rownum, SIZE_LAYER_2}, torch::kFloat).cuda();
    torch::Tensor Y0_gpu = torch::zeros({embedding->rownum, SIZE_LAYER_2}, at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
    torch::Tensor Y1_gpu = torch::zeros({embedding->rownum, SIZE_LAYER_2}, at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
    torch::Tensor Y1_inv_gpu = torch::zeros({embedding->rownum, SIZE_LAYER_2}, at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
    torch::Tensor Y0_inv_gpu = torch::zeros({embedding->rownum, SIZE_LAYER_2}, at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
    torch::Tensor W_for_gpu = torch::from_blob(weight, {edges_for, 1}, torch::kFloat).cuda();
    torch::Tensor W_back_gpu = torch::from_blob(weight_backward, {edges_back + 1, 1}, torch::kFloat).cuda();
    torch::Tensor Out0_gpu;
    torch::Tensor Out1_gpu;
    torch::Tensor Y0_gpu_buffered = torch::zeros({graph->vertices, SIZE_LAYER_2},
                                                 at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
    torch::Tensor Y1_gpu_buffered = torch::zeros({graph->vertices, SIZE_LAYER_2},
                                                 at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
    torch::Tensor Y1_inv_gpu_buffered = torch::zeros({graph->vertices, SIZE_LAYER_2},
                                                     at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
    torch::Tensor Y0_inv_gpu_buffered = torch::zeros({graph->vertices, SIZE_LAYER_2},
                                                     at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));

    double exec_time = 0;
    exec_time -= get_time();

    for (int i_i = 0; i_i < iterations; i_i++)
    {
        if (i_i != 0)
        {
            //inter1_gpu.grad().zero_();
            //inter2_gpu.grad().zero_();
            Gnn_v1->zero_grad();
            Gnn_v2->zero_grad();
            //Y1_gpu.grad().zero_();
        }

        if (graph->partition_id == 0)
            std::cout << "start  [" << i_i << "]  epoch+++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
        //layer 1;

        X0_gpu_trans = Gnn_v1->forward(X0_gpu);
        gr_e->forward_one_step(X0_gpu_trans.packed_accessor<float, 2>().data(),
                               Y0_gpu_buffered.packed_accessor<float, 2>().data(),
                               W_for_gpu.packed_accessor<float, 2>().data(), SCALA_TYPE, SIZE_LAYER_2);

        torch::Tensor comm_data = Y0_gpu_buffered.cpu();
        torch::Tensor Y0_cpu = Y0_gpu.cpu();
        gt->Sync_data<SIZE_LAYER_2>(comm_data, Y0_cpu);
        Y0_gpu.set_data(Y0_cpu.cuda());

        Out0_gpu = torch::relu(Y0_gpu); //new
                                        //layer 2;
        gr_e->forward_one_step(Out0_gpu.packed_accessor<float, 2>().data(),
                               Y1_gpu_buffered.packed_accessor<float, 2>().data(),
                               W_for_gpu.packed_accessor<float, 2>().data(), SCALA_TYPE, SIZE_LAYER_2);

        torch::Tensor comm_data1 = Y1_gpu_buffered.cpu();
        torch::Tensor Y1_cpu = Y1_gpu.cpu();
        gt->Sync_data<SIZE_LAYER_2>(comm_data1, Y1_cpu);
        Y1_gpu.set_data(Y1_cpu.cuda());
        Out1_gpu = Gnn_v2->forward(Y1_gpu);
        //output;
        torch::Tensor tt = Out1_gpu.log_softmax(1);     //CUDA
        pytool->loss = torch::nll_loss(tt, target_gpu); //new
        pytool->loss.backward();
        //inv layer 2;
        torch::Tensor aggregate_grad2 = unified_parameter<ValueType>(comm1, Gnn_v2->W.grad().cpu());
        Gnn_v2->learn_gpu(aggregate_grad2.cuda(), learn_rate); //reset from new
        std::cout << "error" << std::endl;
        //inv layer 1;
        // 2->1
        gr_e->backward_one_step(Y1_gpu.grad().packed_accessor<float, 2>().data(),
                                Y1_inv_gpu_buffered.packed_accessor<float, 2>().data(),
                                W_back_gpu.packed_accessor<float, 2>().data(), SCALA_TYPE, SIZE_LAYER_2);
        torch::Tensor comm_data2 = Y1_inv_gpu_buffered.cpu();
        torch::Tensor Y1_inv_cpu = Y1_inv_gpu.cpu();
        gt->Sync_data<SIZE_LAYER_2>(comm_data2, Y1_inv_cpu);
        Y1_inv_gpu.set_data(Y1_inv_cpu.cuda());

        // layer 1 local
        Out0_gpu.backward(); //new
        torch::Tensor tmp = Y1_inv_gpu * Y0_gpu.grad();
        gr_e->backward_one_step(tmp.packed_accessor<float, 2>().data(),
                                Y0_inv_gpu_buffered.packed_accessor<float, 2>().data(),
                                W_back_gpu.packed_accessor<float, 2>().data(), SCALA_TYPE, SIZE_LAYER_2);
        torch::Tensor comm_data3 = Y0_inv_gpu_buffered.cpu();
        torch::Tensor Y0_inv_cpu = Y0_inv_gpu.cpu();
        gt->Sync_data<SIZE_LAYER_2>(comm_data3, Y0_inv_cpu);
        Y0_inv_gpu.set_data(Y0_inv_cpu.cuda());

        // layer 1 combine
        new_combine_grad.zero_();
        new_combine_grad = X0_gpu.t().mm(Y0_inv_gpu);

        //learn
        torch::Tensor aggregate_grad = unified_parameter<ValueType>(comm, new_combine_grad.cpu());
        Gnn_v1->learn_gpu(aggregate_grad.cuda(), learn_rate);

        if (graph->partition_id == 0)
            std::cout << "LOSS:\t" << pytool->loss << std::endl;

        if (i_i == (iterations - 1))
        { //&&graph->partition_id==0
            torch::Tensor tt_cpu = tt.cpu();
            if (i_i == (iterations - 1) && graph->partition_id == 0)
            {
                inference(tt_cpu, graph, embedding, pytool, Gnn_v1, Gnn_v2);
            }
        }
    }
    //     delete active;
}

/*GPU dist*/ void compute_dist_GPU_with_new_system(Graph<Empty> *graph, int iterations)
{
    if (graph->partition_id == 0)
        printf("dist GCN");
    ValueType learn_rate = 0.01;
    VertexSubset *active = graph->alloc_vertex_subset();
    const int BATCH_SIZE = graph->owned_vertices;
    active->fill();
    Embeddings<ValueType, long> *embedding = new Embeddings<ValueType, long>();
    embedding->init(graph);
    embedding->readlabel_(graph);

    Network<ValueType> *comm = new Network<ValueType>(graph, MAX_LAYER, MAX_LAYER);
    Network<ValueType> *comm1 = new Network<ValueType>(graph, MAX_LAYER, MAX_LAYER);
    comm->setWsize(MAX_LAYER, MAX_LAYER);
    comm1->setWsize(MAX_LAYER, MAX_LAYER);
    tensorSet *pytool = new tensorSet(2);
    /*init GPU*/
    for (int i_s; i_s < graph->sockets; i_s++)
    {
        for (VertexId vtx = 0; vtx < graph->vertices; vtx++)
        {
            if (graph->incoming_adj_index[i_s][vtx + 1] == 0)
                graph->incoming_adj_index[i_s][vtx + 1] = graph->incoming_adj_index[i_s][vtx];
            if (graph->incoming_adj_index_backward[i_s][vtx + 1] == 0)
                graph->incoming_adj_index_backward[i_s][vtx + 1] = graph->incoming_adj_index_backward[i_s][vtx];
        }
    }

    graph->generate_COO(active);
    graph->reorder_COO(BATCH_SIZE);
    std::vector<edge_list *> edge_list;
    generate_edge_list_Tensor(graph, edge_list, BATCH_SIZE);

    //      std::vector<CSC_segment*> csc_segment;
    //     generate_CSC_Segment_Tensor(graph,csc_segment,BATCH_SIZE);
    //     generate_weight_and_csr(graph, active, incoming_adj_index, incoming_adj_index_backward,
    //        weight, weight_backward,W_for_gpu, W_back_gpu);
    //     printf("\n+++++++++++++%d++++++++++comehere\n",graph->partition_id);
    /*1 INIT STAGE*/
    GnnUnit *Gnn_v1 = new GnnUnit(SIZE_LAYER_1, SIZE_LAYER_2);
    GnnUnit *Gnn_v2 = new GnnUnit(SIZE_LAYER_2, OUTPUT_LAYER_3);

    pytool->registOptimizer(torch::optim::SGD(Gnn_v1->parameters(), 0.05)); //new
    pytool->registOptimizer(torch::optim::SGD(Gnn_v2->parameters(), 0.05)); //new
    pytool->registLabel<long>(embedding->label, graph->partition_offset[graph->partition_id],
                              graph->partition_offset[graph->partition_id + 1] - graph->partition_offset[graph->partition_id]); //new
                                                                                                                                /*init W with new */

    init_parameter(comm, graph, Gnn_v1, embedding);
    init_parameter(comm1, graph, Gnn_v2, embedding);

    std::vector<int> layer_size(0);
    layer_size.push_back(SIZE_LAYER_1);
    layer_size.push_back(SIZE_LAYER_2);
    layer_size.push_back(OUTPUT_LAYER_3);
    GTensor<ValueType, long, MAX_LAYER> *gt = new GTensor<ValueType, long, MAX_LAYER>(graph, embedding, active, 2, layer_size);

    torch::Tensor new_combine_grad = torch::zeros({SIZE_LAYER_1, SIZE_LAYER_2}, torch::kFloat).cuda();
    graph->process_vertices<ValueType>( //init  the vertex state.
        [&](VertexId vtx) {
            for (int i = 0; i < SIZE_LAYER_1; i++)
            {
                embedding->initStartWith(vtx, embedding->con[vtx].att[i], i); //embedding->con[vtx].att[i],i);//embedding->con[vtx].att[i]
            }
            return (ValueType)1;
        },
        active);

    /*GPU  */
    torch::Device GPU(torch::kCUDA, 0);
    torch::Tensor target_gpu = pytool->target.cuda();
    torch::Tensor inter1_gpu = torch::zeros({embedding->rownum, SIZE_LAYER_2}, at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
    Gnn_v2->to(GPU);
    Gnn_v1->to(GPU);

    torch::Tensor X0_cpu = torch::from_blob(embedding->start_v + embedding->start, {embedding->rownum, SIZE_LAYER_1}, torch::kFloat);
    torch::Tensor Y0_gpu = torch::zeros({embedding->rownum, SIZE_LAYER_1}, at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
    torch::Tensor Y0_cpu = torch::zeros({embedding->rownum, SIZE_LAYER_1}, torch::kFloat);
    torch::Tensor Y1_gpu = torch::zeros({embedding->rownum, SIZE_LAYER_2}, at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
    torch::Tensor Y1_cpu = torch::zeros({embedding->rownum, SIZE_LAYER_2}, torch::kFloat);
    torch::Tensor Y1_inv_gpu = torch::zeros({embedding->rownum, SIZE_LAYER_2}, at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
    torch::Tensor Y1_inv_cpu = torch::zeros({embedding->rownum, SIZE_LAYER_2}, torch::kFloat);
    torch::Tensor Out0_gpu;
    torch::Tensor Out1_gpu;
    torch::Tensor Y0_cpu_buffered = torch::zeros({graph->vertices, SIZE_LAYER_1}, torch::kFloat);

    torch::Tensor Y1_cpu_buffered = torch::zeros({graph->vertices, SIZE_LAYER_2}, torch::kFloat);
    torch::Tensor Y1_inv_cpu_buffered = torch::zeros({graph->vertices, SIZE_LAYER_2}, torch::kFloat);

    double exec_time = 0;
    exec_time -= get_time();
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
    for (int i_i = 0; i_i < iterations; i_i++)
    {
        if (i_i != 0)
        {
            Gnn_v1->zero_grad();
            Gnn_v2->zero_grad();
        }

        if (graph->partition_id == 0)
            std::cout << "start  [" << i_i << "]  epoch+++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
        //layer 1;

        copy_time = 0;
        copy_time -= get_time();
        Y0_cpu_buffered.zero_();
        copy_time += get_time();
        all_copy_time += copy_time;

        int testid = 0;
        //std::cout<<"wrong with "<<graph->in_degree[testid]<< graph->partition_id<<std::endl;
        graph_time = 0;
        graph_time -= get_time();
        propagate_forward_gpu_shard(graph, X0_cpu, Y0_cpu_buffered, edge_list, SIZE_LAYER_1);
        //std::cout<<"validation with"<<Y0_cpu_buffered[testid][0]<<std::endl;
        graph_time += get_time();
        all_graph_time += graph_time;

        Y0_cpu.zero_();
        graph_sync_time = 0;
        graph_sync_time -= get_time();
        gt->Sync_data<SIZE_LAYER_1>(Y0_cpu_buffered, Y0_cpu);
        graph_sync_time += get_time();
        all_graph_sync_time += graph_sync_time;

        copy_time = 0;
        copy_time -= get_time();
        Y0_gpu.set_data(Y0_cpu.cuda());
        copy_time += get_time();
        all_copy_time += copy_time;

        copy_time = 0;
        copy_time -= get_time();
        inter1_gpu.set_data(Gnn_v1->forward(Y0_gpu)); //torch::from_blob(embedding->Gnn_v1->forward(pytool->x[0]).accessor<float,2>().data(),{embedding->rownum,VECTOR_LENGTH});
        copy_time += get_time();
        all_copy_time += copy_time;

        compute_time = 0;
        compute_time -= get_time();
        Out0_gpu = torch::relu(inter1_gpu); //new
                                            //c10::cuda::CUDAStream::synchronize();
                                            //	CUDA_SYNCHRONIZE();
        compute_time += get_time();
        all_compute_time += compute_time;
        //layer 2;
        copy_time = 0;
        copy_time -= get_time();
        Y1_cpu_buffered.zero_();
        copy_time += get_time();
        all_copy_time += copy_time;

        graph_time = 0;
        graph_time -= get_time();
        propagate_forward_gpu_shard(graph, Out0_gpu.cpu(), Y1_cpu_buffered, edge_list, SIZE_LAYER_2);
        graph_time += get_time();
        all_graph_time += graph_time;

        Y1_cpu.zero_();
        graph_sync_time = 0;
        graph_sync_time -= get_time();
        gt->Sync_data<SIZE_LAYER_2>(Y1_cpu_buffered, Y1_cpu);
        graph_sync_time += get_time();
        all_graph_sync_time += graph_sync_time;
        copy_time = 0;
        copy_time -= get_time();
        Y1_gpu.set_data(Y1_cpu.cuda());
        copy_time += get_time();
        all_copy_time += copy_time;
        compute_time = 0;
        compute_time -= get_time();
        Out1_gpu = Gnn_v2->forward(Y1_gpu);
        //output;
        torch::Tensor tt = Out1_gpu.log_softmax(1);     //CUDA
        pytool->loss = torch::nll_loss(tt, target_gpu); //new
        pytool->loss.backward();
        //	CUDA_SYNCHRONIZE();
        compute_time += get_time();
        all_compute_time += compute_time;
        //inv layer 2;

        sync_time = 0;
        sync_time -= get_time();
        torch::Tensor aggregate_grad2 = unified_parameter<ValueType>(comm1, Gnn_v2->W.grad().cpu());
        sync_time += get_time();
        all_sync_time += sync_time;

        compute_time = 0;
        compute_time -= get_time();
        Gnn_v2->learn_gpu(aggregate_grad2.cuda(), learn_rate); //reset from new
                                                               //        CUDA_SYNCHRONIZE();
        compute_time += get_time();
        all_compute_time += compute_time;
        //inv layer 1;
        // 2->1
        copy_time = 0;
        copy_time -= get_time();
        Y1_inv_cpu_buffered.zero_();
        copy_time += get_time();
        all_copy_time += copy_time;

        graph_time = 0;
        graph_time -= get_time();
        propagate_forward_gpu_shard(graph, Y1_gpu.grad().cpu(), Y1_inv_cpu_buffered, edge_list, SIZE_LAYER_2);
        graph_time += get_time();
        all_graph_time += graph_time;
        copy_time = 0;
        copy_time -= get_time();
        Y1_inv_cpu.zero_();
        copy_time += get_time();
        all_copy_time += copy_time;

        graph_sync_time = 0;
        graph_sync_time -= get_time();
        gt->Sync_data<SIZE_LAYER_2>(Y1_inv_cpu_buffered, Y1_inv_cpu);
        graph_sync_time += get_time();
        all_graph_sync_time += graph_sync_time;

        copy_time = 0;
        copy_time -= get_time();
        Y1_inv_gpu.set_data(Y1_inv_cpu.cuda());
        copy_time += get_time();
        all_copy_time += copy_time;

        compute_time = 0;
        compute_time -= get_time();
        Out0_gpu.backward(); //new
        new_combine_grad.zero_();
        new_combine_grad = Y0_gpu.t().mm(Y1_inv_gpu * inter1_gpu.grad());
        //        CUDA_SYNCHRONIZE();
        compute_time += get_time();
        all_compute_time += compute_time;
        //learn
        sync_time = 0;
        sync_time -= get_time();
        torch::Tensor aggregate_grad = unified_parameter<ValueType>(comm, new_combine_grad.cpu());
        sync_time += get_time();
        all_sync_time += sync_time;

        compute_time = 0;
        compute_time -= get_time();
        Gnn_v1->learn_gpu(aggregate_grad.cuda(), learn_rate);
        //     CUDA_SYNCHRONIZE();
        compute_time += get_time();
        all_compute_time += compute_time;

        if (graph->partition_id == 0)
            std::cout << "LOSS:\t" << pytool->loss << std::endl;

        if (i_i == (iterations - 1))
        { //&&graph->partition_id==0
            exec_time += get_time();
            if (graph->partition_id == 0)
            {
                printf("DEBUG INFO START:::::::::::::::::::::::::::::\n");
                printf("all_time=%lf(s)\n", exec_time);
                printf("sync_time=%lf(s)\n", all_sync_time);
                printf("all_graph_sync_time=%lf(s)\n", all_graph_sync_time);
                printf("copy_time=%lf(s)\n", all_copy_time);
                printf("nn_time=%lf(s)\n", all_compute_time);
                printf("graph_time=%lf(s)\n", all_graph_time);
                printf("communicate_extract+send=%lf(s)\n", graph->all_compute_time);
                printf("communicate_processing_received=%lf(s)\n", graph->all_overlap_time);
                printf("communicate_wait=%lf(s)\n", graph->all_wait_time);
                printf("streamed kernel_time=%lf(s)\n", graph->all_kernel_time);
                printf("streamed movein_time=%lf(s)\n", graph->all_movein_time);
                printf("streamed moveout_time=%lf(s)\n", graph->all_moveout_time);
                printf("DEBUG INFO END:::::::::::::::::::::::::::::::\n");
            }
            torch::Tensor tt_cpu = tt.cpu();
            if (i_i == (iterations - 1) && graph->partition_id == 0)
            {
                inference(tt_cpu, graph, embedding, pytool, Gnn_v1, Gnn_v2);
            }
        }
    }
    delete active;
}

/*GPU dist*/ void compute_dist_GPU_with_new_system_CSC(Graph<Empty> *graph, int iterations, bool GPUaggregator)
{
    if (graph->partition_id == 0)
        printf("dist GCN CSC\n");
    ValueType learn_rate = 0.01;
    VertexSubset *active = graph->alloc_vertex_subset();
    const int BATCH_SIZE = 200000; //graph->owned_vertices;
    active->fill();
    Embeddings<ValueType, long> *embedding = new Embeddings<ValueType, long>();
    embedding->init(graph);
    embedding->readlabel_(graph);
    Network<ValueType> *comm = new Network<ValueType>(graph, MAX_LAYER, MAX_LAYER);
    Network<ValueType> *comm1 = new Network<ValueType>(graph, MAX_LAYER, MAX_LAYER);
    comm->setWsize(MAX_LAYER, MAX_LAYER);
    comm1->setWsize(MAX_LAYER, MAX_LAYER);
    tensorSet *pytool = new tensorSet(2);
    /*init GPU*/
    for (int i_s; i_s < graph->sockets; i_s++)
    {
        for (VertexId vtx = 0; vtx < graph->vertices; vtx++)
        {
            if (graph->incoming_adj_index[i_s][vtx + 1] == 0)
                graph->incoming_adj_index[i_s][vtx + 1] = graph->incoming_adj_index[i_s][vtx];
            if (graph->incoming_adj_index_backward[i_s][vtx + 1] == 0)
                graph->incoming_adj_index_backward[i_s][vtx + 1] = graph->incoming_adj_index_backward[i_s][vtx];
        }
    }

    graph->generate_COO(active);
    graph->reorder_COO(BATCH_SIZE);
    std::vector<CSC_segment *> csc_segment;
    generate_CSC_Segment_Tensor(graph, csc_segment, BATCH_SIZE, false);
    //     printf("\n+++++++++++++%d++++++++++comehere\n",graph->partition_id);
    /*1 INIT STAGE*/
    //graph->generarte_selective_bitmap();
    GnnUnit *Gnn_v1 = new GnnUnit(SIZE_LAYER_1, SIZE_LAYER_2);
    GnnUnit *Gnn_v2 = new GnnUnit(SIZE_LAYER_2, OUTPUT_LAYER_3);

    pytool->registOptimizer(torch::optim::SGD(Gnn_v1->parameters(), 0.05)); //new
    pytool->registOptimizer(torch::optim::SGD(Gnn_v2->parameters(), 0.05)); //new
    pytool->registLabel<long>(embedding->label, graph->partition_offset[graph->partition_id],
                              graph->partition_offset[graph->partition_id + 1] - graph->partition_offset[graph->partition_id]); //new

    init_parameter(comm, graph, Gnn_v1, embedding);
    init_parameter(comm1, graph, Gnn_v2, embedding);

    std::vector<int> layer_size(0);
    layer_size.push_back(SIZE_LAYER_1);
    layer_size.push_back(SIZE_LAYER_2);
    layer_size.push_back(OUTPUT_LAYER_3);
    GTensor<ValueType, long, MAX_LAYER> *gt = new GTensor<ValueType, long, MAX_LAYER>(graph, embedding, active, 2, layer_size);

    torch::Tensor new_combine_grad = torch::zeros({SIZE_LAYER_1, SIZE_LAYER_2}, torch::kFloat).cuda();
    graph->process_vertices<ValueType>( //init  the vertex state.
        [&](VertexId vtx) {
            for (int i = 0; i < SIZE_LAYER_1; i++)
            {
                embedding->initStartWith(vtx, embedding->con[vtx].att[i], i); //embedding->con[vtx].att[i],i);//embedding->con[vtx].att[i]
            }
            return (ValueType)1;
        },
        active);

    torch::Device GPU(torch::kCUDA, 0);
    torch::Tensor target_gpu = pytool->target.cuda();
    torch::Tensor inter1_gpu = torch::zeros({embedding->rownum, SIZE_LAYER_2}, at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
    Gnn_v2->to(GPU);
    Gnn_v1->to(GPU);

    torch::Tensor X0_cpu = torch::from_blob(embedding->start_v + embedding->start, {embedding->rownum, SIZE_LAYER_1}, torch::kFloat);
    torch::Tensor Y0_gpu = torch::zeros({embedding->rownum, SIZE_LAYER_1}, at::TensorOptions().device_index(0).dtype(torch::kFloat));
    torch::Tensor Y0_cpu = torch::zeros({embedding->rownum, SIZE_LAYER_1}, torch::kFloat);
    torch::Tensor Y1_gpu = torch::zeros({embedding->rownum, SIZE_LAYER_2}, at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
    torch::Tensor Y1_cpu = torch::zeros({embedding->rownum, SIZE_LAYER_2}, torch::kFloat);
    torch::Tensor Y1_inv_gpu = torch::zeros({embedding->rownum, SIZE_LAYER_2}, at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
    torch::Tensor Y1_inv_cpu = torch::zeros({embedding->rownum, SIZE_LAYER_2}, torch::kFloat);
    torch::Tensor Out0_gpu;
    torch::Tensor Out1_gpu;
    torch::Tensor Y0_cpu_buffered = torch::zeros({graph->vertices, SIZE_LAYER_1}, torch::kFloat);

    torch::Tensor Y1_cpu_buffered = torch::zeros({graph->vertices, SIZE_LAYER_2}, torch::kFloat);
    torch::Tensor Y1_inv_cpu_buffered = torch::zeros({graph->vertices, SIZE_LAYER_2}, torch::kFloat);

    double exec_time = 0;
    exec_time -= get_time();
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
    for (int i_i = 0; i_i < iterations; i_i++)
    {
        if (i_i != 0)
        {
            Gnn_v1->zero_grad();
            Gnn_v2->zero_grad();
        }

        if (graph->partition_id == 0)
            std::cout << "start  [" << i_i << "]  epoch+++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
        //layer 1;

        if (i_i == 0 || true)
        {
            copy_time = 0;
            copy_time -= get_time();
            Y0_cpu_buffered.zero_();
            copy_time += get_time();
            all_copy_time += copy_time;

            int testid = 0;
            //std::cout<<"wrong with "<<graph->in_degree[testid]<< graph->partition_id<<std::endl;
            graph_time = 0;
            graph_time -= get_time();
            propagate_forward_gpu_shard_CSC(graph, X0_cpu, Y0_cpu_buffered, csc_segment, SIZE_LAYER_1);
            //std::cout<<"validation with"<<Y0_cpu_buffered[testid][0]<<std::endl;
            graph_time += get_time();
            all_graph_time += graph_time;
        }

        if (i_i == 0 || true)
        {
            Y0_cpu.zero_();
            graph_sync_time = 0;
            graph_sync_time -= get_time();
            if (!GPUaggregator)
            {
                gt->Sync_data<SIZE_LAYER_1>(Y0_cpu_buffered, Y0_cpu);
                graph_sync_time += get_time();
                all_graph_sync_time += graph_sync_time;

                copy_time = 0;
                copy_time -= get_time();
                Y0_gpu.set_data(Y0_cpu.cuda());
                copy_time += get_time();
                all_copy_time += copy_time;
            }
            else
            {
                // printf("hello disator\n");
                gt->Sync_data_gpu<SIZE_LAYER_1>(Y0_cpu_buffered, Y0_gpu, false);
                graph_sync_time += get_time();
                all_graph_sync_time += graph_sync_time;
            }
        }

        copy_time = 0;
        copy_time -= get_time();
        inter1_gpu.set_data(Gnn_v1->forward(Y0_gpu)); //torch::from_blob(embedding->Gnn_v1->forward(pytool->x[0]).accessor<float,2>().data(),{embedding->rownum,VECTOR_LENGTH});
        copy_time += get_time();
        all_copy_time += copy_time;

        compute_time = 0;
        compute_time -= get_time();
        Out0_gpu = torch::relu(inter1_gpu); //new

        compute_time += get_time();
        all_compute_time += compute_time;
        //layer 2;
        copy_time = 0;
        copy_time -= get_time();
        Y1_cpu_buffered.zero_();
        copy_time += get_time();
        all_copy_time += copy_time;

        graph_time = 0;
        graph_time -= get_time();
        propagate_forward_gpu_shard_CSC(graph, Out0_gpu.cpu(), Y1_cpu_buffered, csc_segment, SIZE_LAYER_2);
        graph_time += get_time();
        all_graph_time += graph_time;

        Y1_cpu.zero_();
        graph_sync_time = 0;
        graph_sync_time -= get_time();

        if (!GPUaggregator)
        {
            gt->Sync_data<SIZE_LAYER_2>(Y1_cpu_buffered, Y1_cpu);
            graph_sync_time += get_time();
            all_graph_sync_time += graph_sync_time;
            copy_time = 0;
            copy_time -= get_time();
            Y1_gpu.set_data(Y1_cpu.cuda());
            copy_time += get_time();
            all_copy_time += copy_time;
        }
        else
        {
            // printf("hello disator\n");
            gt->Sync_data_gpu<SIZE_LAYER_2>(Y1_cpu_buffered, Y1_gpu);
            graph_sync_time += get_time();
            all_graph_sync_time += graph_sync_time;
            //printf("finish disator\n");
        }
        //

        compute_time = 0;
        compute_time -= get_time();
        Out1_gpu = Gnn_v2->forward(Y1_gpu);
        //output;
        torch::Tensor tt = Out1_gpu.log_softmax(1);     //CUDA
        pytool->loss = torch::nll_loss(tt, target_gpu); //new
        pytool->loss.backward();
        //	CUDA_SYNCHRONIZE();
        compute_time += get_time();
        all_compute_time += compute_time;
        //inv layer 2;

        sync_time = 0;
        sync_time -= get_time();
        torch::Tensor aggregate_grad2 = unified_parameter<ValueType>(comm1, Gnn_v2->W.grad().cpu());
        sync_time += get_time();
        all_sync_time += sync_time;

        compute_time = 0;
        compute_time -= get_time();
        Gnn_v2->learn_gpu(aggregate_grad2.cuda(), learn_rate); //reset from new
                                                               //        CUDA_SYNCHRONIZE();
        compute_time += get_time();
        all_compute_time += compute_time;
        //inv layer 1;
        // 2->1
        copy_time = 0;
        copy_time -= get_time();
        Y1_inv_cpu_buffered.zero_();
        copy_time += get_time();
        all_copy_time += copy_time;

        graph_time = 0;
        graph_time -= get_time();
        propagate_forward_gpu_shard_CSC(graph, Y1_gpu.grad().cpu(), Y1_inv_cpu_buffered, csc_segment, SIZE_LAYER_2);
        graph_time += get_time();
        all_graph_time += graph_time;

        copy_time = 0;
        copy_time -= get_time();
        Y1_inv_cpu.zero_();
        copy_time += get_time();
        all_copy_time += copy_time;

        graph_sync_time = 0;
        graph_sync_time -= get_time();
        if (!GPUaggregator)
        {
            gt->Sync_data<SIZE_LAYER_2>(Y1_inv_cpu_buffered, Y1_inv_cpu);
            graph_sync_time += get_time();
            all_graph_sync_time += graph_sync_time;

            copy_time = 0;
            copy_time -= get_time();
            Y1_inv_gpu.set_data(Y1_inv_cpu.cuda());
            copy_time += get_time();
            all_copy_time += copy_time;
        }
        else
        {
            // printf("hello disator\n");
            gt->Sync_data_gpu<SIZE_LAYER_2>(Y1_inv_cpu_buffered, Y1_inv_gpu);
            graph_sync_time += get_time();
            all_graph_sync_time += graph_sync_time;
        }

        compute_time = 0;
        compute_time -= get_time();
        Out0_gpu.backward(); //new
        new_combine_grad.zero_();
        new_combine_grad = Y0_gpu.t().mm(Y1_inv_gpu * inter1_gpu.grad());
        //        CUDA_SYNCHRONIZE();
        compute_time += get_time();
        all_compute_time += compute_time;
        //learn
        sync_time = 0;
        sync_time -= get_time();
        torch::Tensor aggregate_grad = unified_parameter<ValueType>(comm, new_combine_grad.cpu());
        sync_time += get_time();
        all_sync_time += sync_time;

        compute_time = 0;
        compute_time -= get_time();
        Gnn_v1->learn_gpu(aggregate_grad.cuda(), learn_rate);
        //     CUDA_SYNCHRONIZE();
        compute_time += get_time();
        all_compute_time += compute_time;

        if (graph->partition_id == 0)
            std::cout << "LOSS:\t" << pytool->loss << std::endl;

        if (i_i == (iterations - 1))
        { //&&graph->partition_id==0
            exec_time += get_time();
            if (graph->partition_id == 0)
            {
                printf("DEBUG INFO START:::::::::::::::::::::::::::::\n");
                printf("all_time=%lf(s)\n", exec_time);
                printf("sync_time=%lf(s)\n", all_sync_time);
                printf("all_graph_sync_time=%lf(s)\n", all_graph_sync_time);
                printf("copy_time=%lf(s)\n", all_copy_time);
                printf("nn_time=%lf(s)\n", all_compute_time);
                printf("graph_time=%lf(s)\n", all_graph_time);
                printf("communicate_extract+send=%lf(s)\n", graph->all_compute_time);
                printf("communicate_processing_received=%lf(s)\n", graph->all_overlap_time);
                printf("communicate_processing_received.copy=%lf(s)\n", graph->all_recv_copy_time);
                printf("communicate_processing_received.kernel=%lf(s)\n", graph->all_recv_kernel_time);
                printf("communicate_processing_received.wait=%lf(s)\n", graph->all_recv_wait_time);
                printf("communicate_wait=%lf(s)\n", graph->all_wait_time);
                printf("streamed kernel_time=%lf(s)\n", graph->all_kernel_time);
                printf("streamed movein_time=%lf(s)\n", graph->all_movein_time);
                printf("streamed moveout_time=%lf(s)\n", graph->all_moveout_time);
                printf("DEBUG INFO END:::::::::::::::::::::::::::::::\n");
            }
            //         torch::Tensor tt_cpu=tt.cpu();
            //     if(i_i==(iterations-1)&&graph->partition_id==0){
            //        inference(tt_cpu,graph, embedding, pytool,Gnn_v1,Gnn_v2);
            //     }
        }
    }

    delete active;
}

/*GPU dist*/ void compute_dist_GPU_with_CSC_overlap_1(Graph<Empty> *graph, int iterations, bool process_local = false, bool process_overlap = false)
{

    if (graph->partition_id == 0)
        printf("GNNmini::Engine[Dist.GPU.GCN]\n");

    ValueType learn_rate = 0.01;
    VertexSubset *active = graph->alloc_vertex_subset();
    active->fill();
    // init graph
    graph->init_gnnctx(graph->config->layer_string);
    graph->generate_COO(active);
    graph->reorder_COO_W2W();
    std::vector<CSC_segment_pinned *> csc_segment;
    generate_CSC_Segment_Tensor_pinned(graph, csc_segment, true);
    //if (graph->config->process_local)
    double load_rep_time = 0;
    load_rep_time -= get_time();
    graph->load_replicate3(graph->gnnctx->layer_size);

    load_rep_time += get_time();
    if (graph->partition_id == 0)
        printf("#load_rep_time=%lf(s)\n", load_rep_time);

    graph->init_blockinfo();
    //input

    graph->init_message_map_amount();

    torch::Tensor target;
    GNNDatum *gnndatum = new GNNDatum(graph->gnnctx);
    gnndatum->random_generate();
    gnndatum->registLabel(target);
    torch::Tensor target_gpu = target.cuda();

    GnnUnit *Gnn_v1 = new GnnUnit(graph->gnnctx->layer_size[0], graph->gnnctx->layer_size[1]);
    GnnUnit *Gnn_v3 = new GnnUnit(graph->gnnctx->layer_size[0], graph->gnnctx->layer_size[1]);
    GnnUnit *Gnn_v2 = new GnnUnit(graph->gnnctx->layer_size[1], graph->gnnctx->layer_size[2]);
    Gnn_v1->init_parameter();
    Gnn_v2->init_parameter();
    torch::Device GPU(torch::kCUDA, 0);
    torch::Device CPU(torch::kCPU, 0);
    Gnn_v2->to(GPU);
    Gnn_v1->to(GPU);
    //std::cout << graph->gnnctx->layer_size[1] << std::endl;

    GTensor<ValueType, long, MAX_LAYER> *gt = new GTensor<ValueType, long, MAX_LAYER>(graph, active);

    //intermediate tensors
    torch::Tensor new_combine_grad = torch::zeros({graph->gnnctx->layer_size[0], graph->gnnctx->layer_size[1]}, torch::kFloat).cuda();

    torch::Tensor inter1_gpu = torch::zeros({graph->gnnctx->l_v_num, graph->gnnctx->layer_size[1]}, at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
    torch::Tensor X0_cpu = torch::ones({graph->gnnctx->l_v_num, graph->gnnctx->layer_size[0]}, at::TensorOptions().dtype(torch::kFloat));
    torch::Tensor X0_gpu = X0_cpu.cuda();
    torch::Tensor Y0_gpu = torch::ones({graph->gnnctx->l_v_num, graph->gnnctx->layer_size[0]}, at::TensorOptions().device_index(0).dtype(torch::kFloat));
    torch::Tensor Y1_gpu = torch::zeros({graph->gnnctx->l_v_num, graph->gnnctx->layer_size[1]}, at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
    torch::Tensor Y1_inv_gpu = torch::zeros({graph->gnnctx->l_v_num, graph->gnnctx->layer_size[1]}, at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
    torch::Tensor Out0_gpu;
    torch::Tensor Out1_gpu;
    torch::Tensor loss;
    float *Y0_cpu_buffered = (float *)cudaMallocPinned(((long)graph->vertices) * graph->gnnctx->max_layer * sizeof(float));
    if (Y0_cpu_buffered == NULL)
        printf("allocate fail\n");

    //timer
    double exec_time = 0;
    exec_time -= get_time();
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

    //rtminfo initialize
    graph->init_rtminfo();
    graph->rtminfo->process_local = graph->config->process_local;
    graph->rtminfo->reduce_comm = graph->config->process_local;
    graph->rtminfo->copy_data = false;
    graph->rtminfo->process_overlap = graph->config->overlap;

    graph->print_info();

    if (graph->partition_id == 0)
        printf("\nGNNmini::Start [%d] Epochs\n", iterations);
    for (int i_i = 0; i_i < iterations; i_i++)
    {
        if (i_i != 0)
        {
            Gnn_v1->zero_grad();
            Gnn_v2->zero_grad();
            CUDA_DEVICE_SYNCHRONIZE();
        }

        if (graph->partition_id == 0)
        {
            printf("\nGNNmini::Running.Epoch[%d]\n", i_i);
        }

        //CUDA_DEVICE_SYNCHRONIZE();
        graph->rtminfo->epoch = i_i;

        torch::Tensor s = X0_cpu.to(GPU);
        getchar();
        torch::Tensor d = Gnn_v1->forward(s);
        torch::Tensor d1 = d.to(GPU);
        torch::Tensor s1 = s.to(GPU);
        Gnn_v3->to(GPU);
        getchar();
        s1.backward();
        std::cout << Gnn_v3->W.grad();
        getchar();
    }

    delete active;
}

/*GPU dist*/ void compute_dist_GPU_with_CSC_overlap(Graph<Empty> *graph, int iterations, bool process_local = false, bool process_overlap = false)
{

    if (graph->partition_id == 0)
        printf("GNNmini::Engine[Dist.GPU.GCN]\n");

    ValueType learn_rate = 0.01;
    VertexSubset *active = graph->alloc_vertex_subset();
    active->fill();
    // init graph
    graph->init_gnnctx(graph->config->layer_string);
    graph->generate_COO(active);
    graph->reorder_COO_W2W();
    std::vector<CSC_segment_pinned *> csc_segment;
    generate_CSC_Segment_Tensor_pinned(graph, csc_segment, true);
    //if (graph->config->process_local)
    double load_rep_time = 0;
    load_rep_time -= get_time();
    graph->load_replicate3(graph->gnnctx->layer_size);

    load_rep_time += get_time();
    if (graph->partition_id == 0)
        printf("#load_rep_time=%lf(s)\n", load_rep_time);

    graph->init_blockinfo();
    //input

    graph->init_message_map_amount();

    torch::Tensor target;
    GNNDatum *gnndatum = new GNNDatum(graph->gnnctx);
    gnndatum->random_generate();
    gnndatum->registLabel(target);
    torch::Tensor target_gpu = target.cuda();

    GnnUnit *Gnn_v1 = new GnnUnit(graph->gnnctx->layer_size[0], graph->gnnctx->layer_size[1]);
    GnnUnit *Gnn_v2 = new GnnUnit(graph->gnnctx->layer_size[1], graph->gnnctx->layer_size[2]);
    Gnn_v1->init_parameter();
    Gnn_v2->init_parameter();
    torch::Device GPU(torch::kCUDA, 0);
    Gnn_v2->to(GPU);
    Gnn_v1->to(GPU);
    //std::cout << graph->gnnctx->layer_size[1] << std::endl;

    GTensor<ValueType, long, MAX_LAYER> *gt = new GTensor<ValueType, long, MAX_LAYER>(graph, active);

    //intermediate tensors
    torch::Tensor new_combine_grad = torch::zeros({graph->gnnctx->layer_size[0], graph->gnnctx->layer_size[1]}, torch::kFloat).cuda();

    torch::Tensor inter1_gpu = torch::zeros({graph->gnnctx->l_v_num, graph->gnnctx->layer_size[1]}, at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
    torch::Tensor X0_cpu = torch::ones({graph->gnnctx->l_v_num, graph->gnnctx->layer_size[0]}, at::TensorOptions().dtype(torch::kFloat));
    torch::Tensor X0_gpu = X0_cpu.cuda();
    torch::Tensor Y0_gpu = torch::zeros({graph->gnnctx->l_v_num, graph->gnnctx->layer_size[0]}, at::TensorOptions().device_index(0).dtype(torch::kFloat));
    torch::Tensor Y1_gpu = torch::zeros({graph->gnnctx->l_v_num, graph->gnnctx->layer_size[1]}, at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
    torch::Tensor Y1_inv_gpu = torch::zeros({graph->gnnctx->l_v_num, graph->gnnctx->layer_size[1]}, at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
    torch::Tensor Out0_gpu;
    torch::Tensor Out1_gpu;
    torch::Tensor loss;
    float *Y0_cpu_buffered = (float *)cudaMallocPinned(((long)graph->vertices) * graph->gnnctx->max_layer * sizeof(float));
    if (Y0_cpu_buffered == NULL)
        printf("allocate fail\n");

    //timer
    double exec_time = 0;
    exec_time -= get_time();
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

    //rtminfo initialize
    graph->init_rtminfo();
    graph->rtminfo->process_local = graph->config->process_local;
    graph->rtminfo->reduce_comm = graph->config->process_local;
    graph->rtminfo->copy_data = false;
    graph->rtminfo->process_overlap = graph->config->overlap;

    graph->print_info();

    if (graph->partition_id == 0)
        printf("\nGNNmini::Start [%d] Epochs\n", iterations);
    for (int i_i = 0; i_i < iterations; i_i++)
    {
        if (i_i != 0)
        {
            Gnn_v1->zero_grad();
            Gnn_v2->zero_grad();
            CUDA_DEVICE_SYNCHRONIZE();
        }

        if (graph->partition_id == 0)
        {
            printf("\nGNNmini::Running.Epoch[%d]\n", i_i);
        }

        //CUDA_DEVICE_SYNCHRONIZE();
        graph->rtminfo->epoch = i_i;

        graph_time = 0;
        graph_time -= get_time();

        graph->rtminfo->curr_layer = 0;
        //std::cout << X0_gpu[0][0] << std::endl;
        gt->Process_GPU_overlap_explict(X0_gpu, Y0_cpu_buffered, Y0_gpu, csc_segment);

        graph_time += get_time();
        all_graph_time += graph_time;

        int test_id = 0;
        std::cout << graph->in_degree_for_backward[test_id + graph->partition_offset[graph->partition_id]] << " " << Y0_gpu[test_id][0] << std::endl;

        copy_time = 0;
        copy_time -= get_time();

        inter1_gpu.set_data(Gnn_v1->forward(Y0_gpu));

        copy_time += get_time();
        all_copy_time += copy_time;

        compute_time = 0;
        compute_time -= get_time();

        Out0_gpu = torch::relu(inter1_gpu); //new
        CUDA_DEVICE_SYNCHRONIZE();
        compute_time += get_time();
        all_compute_time += compute_time;
        // //layer 2;

        graph_time = 0;
        graph_time -= get_time();

        graph->rtminfo->curr_layer = 1;
        gt->Process_GPU_overlap_explict(Out0_gpu, Y0_cpu_buffered, Y1_gpu, csc_segment);

        graph_time += get_time();
        all_graph_time += graph_time;

        compute_time = 0;
        compute_time -= get_time();

        Out1_gpu = Gnn_v2->forward(Y1_gpu);
        torch::Tensor tt = Out1_gpu.log_softmax(1); //CUDA
        loss = torch::nll_loss(tt, target_gpu);     //new
        loss.backward();
        CUDA_DEVICE_SYNCHRONIZE();
        compute_time += get_time();
        all_compute_time += compute_time;

        sync_time = 0;
        sync_time -= get_time();

        Gnn_v2->all_reduce_to_gradient(Gnn_v2->W.cpu()); //Gnn_v2->W.grad().cpu()
        //std::cout << Gnn_v2->W.cpu()[0][3] << std::endl;
        //std::cout << Gnn_v2->W_gradient[0][3] << std::endl;
        sync_time += get_time();
        all_sync_time += sync_time;

        compute_time = 0;
        compute_time -= get_time();

        Gnn_v2->learnC2G(learn_rate); //reset from new
        CUDA_DEVICE_SYNCHRONIZE();
        compute_time += get_time();
        all_compute_time += compute_time;

        graph_time = 0;
        graph_time -= get_time();

        graph->rtminfo->curr_layer = 1;
        gt->Process_GPU_overlap_explict(Y1_gpu.grad(), Y0_cpu_buffered, Y1_inv_gpu, csc_segment);

        graph_time += get_time();
        all_graph_time += graph_time;

        compute_time = 0;
        compute_time -= get_time();

        Out0_gpu.backward(); //new
        new_combine_grad.zero_();
        new_combine_grad = Y0_gpu.t().mm(Y1_inv_gpu * inter1_gpu.grad());
        //printf("hello_world\n");
        CUDA_DEVICE_SYNCHRONIZE();
        compute_time += get_time();
        all_compute_time += compute_time;
        //learn

        sync_time = 0;
        sync_time -= get_time();

        Gnn_v1->all_reduce_to_gradient(new_combine_grad.cpu());

        sync_time += get_time();
        all_sync_time += sync_time;

        compute_time = 0;
        compute_time -= get_time();

        Gnn_v1->learnC2G(learn_rate);
        CUDA_DEVICE_SYNCHRONIZE();
        compute_time += get_time();
        all_compute_time += compute_time;

        if (graph->partition_id == 0)

            std::cout << "GNNmini::Running.LOSS:\t" << loss << std::endl;

        if (i_i == (iterations - 1))
        { //&&graph->partition_id==0
            exec_time += get_time();
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
                printf("ALL TIME = %lf(s) GRAPH TIME = %lf(s) MEAN TIME = %lf(s)\n", another_time, max_time / graph->partitions, mean_time / graph->partitions);
        }
    }

    delete active;
}

/*GPU dist GF*/ void compute_dist_GPU_with_CSC_overlap_exchange(Graph<Empty> *graph, int iterations)
{

    /*
    if (graph->partition_id == 0)
        printf("dist GCN CSC\n");
    ValueType learn_rate = 0.01;
    VertexSubset *active = graph->alloc_vertex_subset();
    const int BATCH_SIZE = graph->owned_vertices; //graph->owned_vertices
    active->fill();
    Embeddings<ValueType, long> *embedding = new Embeddings<ValueType, long>();
    embedding->init(graph);
    embedding->readlabel_(graph);
    Network<ValueType> *comm = new Network<ValueType>(graph, MAX_LAYER, MAX_LAYER);
    Network<ValueType> *comm1 = new Network<ValueType>(graph, MAX_LAYER, MAX_LAYER);
    comm->setWsize(MAX_LAYER, MAX_LAYER);
    comm1->setWsize(MAX_LAYER, MAX_LAYER);
    tensorSet *pytool = new tensorSet(2);

    for (int i_s; i_s < graph->sockets; i_s++)
    {
        for (VertexId vtx = 0; vtx < graph->vertices; vtx++)
        {
            if (graph->incoming_adj_index[i_s][vtx + 1] == 0)
                graph->incoming_adj_index[i_s][vtx + 1] = graph->incoming_adj_index[i_s][vtx];
            if (graph->incoming_adj_index_backward[i_s][vtx + 1] == 0)
                graph->incoming_adj_index_backward[i_s][vtx + 1] = graph->incoming_adj_index_backward[i_s][vtx];
        }
    }

    graph->generate_COO(active);
    graph->reorder_COO_W2W(BATCH_SIZE);
    std::vector<CSC_segment_pinned *> csc_segment;
    generate_CSC_Segment_Tensor_pinned(graph, csc_segment, BATCH_SIZE, true);
    //     printf("\n+++++++++++++%d++++++++++comehere\n",graph->partition_id);
   
    graph->generarte_selective_bitmap();
    GnnUnit *Gnn_v1 = new GnnUnit(SIZE_LAYER_1, SIZE_LAYER_2);
    GnnUnit *Gnn_v2 = new GnnUnit(SIZE_LAYER_2, OUTPUT_LAYER_3);

    pytool->registOptimizer(torch::optim::SGD(Gnn_v1->parameters(), 0.05)); //new
    pytool->registOptimizer(torch::optim::SGD(Gnn_v2->parameters(), 0.05)); //new
    pytool->registLabel<long>(embedding->label, graph->partition_offset[graph->partition_id],
                              graph->partition_offset[graph->partition_id + 1] - graph->partition_offset[graph->partition_id]); //new

    init_parameter(comm, graph, Gnn_v1, embedding);
    init_parameter(comm1, graph, Gnn_v2, embedding);

    std::vector<int> layer_size(0);
    layer_size.push_back(SIZE_LAYER_1);
    layer_size.push_back(SIZE_LAYER_2);
    layer_size.push_back(OUTPUT_LAYER_3);
    GTensor<ValueType, long, MAX_LAYER> *gt = new GTensor<ValueType, long, MAX_LAYER>(graph, embedding, active, 2, layer_size);

    torch::Tensor new_combine_grad = torch::zeros({SIZE_LAYER_1, SIZE_LAYER_2}, torch::kFloat).cuda();
    graph->process_vertices<ValueType>( //init  the vertex state.
        [&](VertexId vtx) {
            for (int i = 0; i < SIZE_LAYER_1; i++)
            {
                embedding->initStartWith(vtx, embedding->con[vtx].att[i], i); //embedding->con[vtx].att[i],i);//embedding->con[vtx].att[i]
            }
            return (ValueType)1;
        },
        active);

    torch::Device GPU(torch::kCUDA, 0);
    torch::Tensor target_gpu = pytool->target.cuda();
    torch::Tensor inter1_gpu = torch::zeros({embedding->rownum, SIZE_LAYER_2}, at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
    Gnn_v2->to(GPU);
    Gnn_v1->to(GPU);
    torch::Tensor X0_gpu_trans = torch::zeros({embedding->rownum, SIZE_LAYER_2}, torch::kFloat).cuda();
    torch::Tensor X0_cpu = torch::from_blob(embedding->start_v + embedding->start, {embedding->rownum, SIZE_LAYER_1}, torch::kFloat);
    torch::Tensor X0_gpu = X0_cpu.cuda();
    torch::Tensor Y0_gpu = torch::zeros({embedding->rownum, SIZE_LAYER_2}, at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
    //torch::Tensor Y0_cpu=torch::zeros({embedding->rownum,SIZE_LAYER_1},torch::kFloat);
    torch::Tensor Y1_gpu = torch::zeros({embedding->rownum, SIZE_LAYER_2}, at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
    //torch::Tensor Y1_cpu=torch::zeros({embedding->rownum,SIZE_LAYER_2},torch::kFloat);
    torch::Tensor Y1_inv_gpu = torch::zeros({embedding->rownum, SIZE_LAYER_2}, at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
    //torch::Tensor Y1_inv_cpu=torch::zeros({embedding->rownum,SIZE_LAYER_2}, torch::kFloat);
    torch::Tensor Y0_inv_gpu = torch::zeros({embedding->rownum, SIZE_LAYER_2}, at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
    torch::Tensor Out0_gpu;

    torch::Tensor Out1_gpu;
    float *Y0_cpu_buffered = (float *)cudaMallocPinned(((long)graph->vertices) * SIZE_LAYER_1 * sizeof(float));

    double exec_time = 0;
    exec_time -= get_time();
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
    for (int i_i = 0; i_i < iterations; i_i++)
    {
        if (i_i != 0)
        {
            Gnn_v1->zero_grad();
            Gnn_v2->zero_grad();
        }

        if (graph->partition_id == 0)
            std::cout << "start  [" << i_i << "]  epoch+++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
        //layer 1;

        X0_gpu_trans = Gnn_v1->forward(X0_gpu);

        graph_time = 0;
        graph_time -= get_time();

        gt->Process_GPU_overlap<SIZE_LAYER_2>(X0_gpu_trans, Y0_cpu_buffered, Y0_gpu, csc_segment, true);
        //gt->Sync_data_gpu<SIZE_LAYER_2>(Y0_cpu_buffered,Y0_gpu);
        graph_time += get_time();
        all_graph_time += graph_time;

        compute_time = 0;
        compute_time -= get_time();

        Out0_gpu = torch::relu(Y0_gpu);

        compute_time += get_time();
        all_compute_time += compute_time;
        //layer 2;

        graph_time = 0;
        graph_time -= get_time();

        gt->Process_GPU_overlap<SIZE_LAYER_2>(Out0_gpu, Y0_cpu_buffered, Y1_gpu, csc_segment, false);
        //  gt->Sync_data_gpu<SIZE_LAYER_2>(Y1_cpu_buffered,Y1_gpu);
        graph_time += get_time();
        all_graph_time += graph_time;

        compute_time = 0;
        compute_time -= get_time();

        Out1_gpu = Gnn_v2->forward(Y1_gpu);
        torch::Tensor tt = Out1_gpu.log_softmax(1);     //CUDA
        pytool->loss = torch::nll_loss(tt, target_gpu); //new
        pytool->loss.backward();

        compute_time += get_time();
        all_compute_time += compute_time;

        sync_time = 0;
        sync_time -= get_time();

        torch::Tensor aggregate_grad2 = unified_parameter<ValueType>(comm1, Gnn_v2->W.grad().cpu());

        sync_time += get_time();
        all_sync_time += sync_time;

        compute_time = 0;
        compute_time -= get_time();

        Gnn_v2->learn_gpu(aggregate_grad2.cuda(), learn_rate); //reset from new

        compute_time += get_time();
        all_compute_time += compute_time;

        graph_time = 0;
        graph_time -= get_time();

        gt->Process_GPU_overlap<SIZE_LAYER_2>(Y1_gpu.grad(), Y0_cpu_buffered, Y1_inv_gpu, csc_segment, false);
        //gt->Sync_data_gpu<SIZE_LAYER_2>(Y1_inv_cpu_buffered,Y1_inv_gpu);
        graph_time += get_time();
        all_graph_time += graph_time;

        compute_time = 0;
        compute_time -= get_time();

        Out0_gpu.backward(); //new
        torch::Tensor tmp = Y1_inv_gpu * Y0_gpu.grad();

        compute_time += get_time();
        all_compute_time += compute_time;
        //learn

        graph_time = 0;
        graph_time -= get_time();

        gt->Process_GPU_overlap<SIZE_LAYER_2>(tmp, Y0_cpu_buffered, Y0_inv_gpu, csc_segment, false);

        graph_time += get_time();
        all_graph_time += graph_time;

        compute_time = 0;
        compute_time -= get_time();

        new_combine_grad.zero_();
        new_combine_grad = X0_gpu.t().mm(Y0_inv_gpu);

        compute_time += get_time();
        all_compute_time += compute_time;

        sync_time = 0;
        sync_time -= get_time();

        torch::Tensor aggregate_grad = unified_parameter<ValueType>(comm, new_combine_grad.cpu());

        sync_time += get_time();
        all_sync_time += sync_time;

        compute_time = 0;
        compute_time -= get_time();

        Gnn_v1->learn_gpu(aggregate_grad.cuda(), learn_rate);

        compute_time += get_time();
        all_compute_time += compute_time;

        if (graph->partition_id == 0)
            std::cout << "LOSS:\t" << pytool->loss << std::endl;

        if (i_i == (iterations - 1))
        { //&&graph->partition_id==0
            exec_time += get_time();
            if (graph->partition_id == 0)
            {
                printf("DEBUG INFO START:::::::::::::::::::::::::::::\n");
                printf("all_time=%lf(s)\n", exec_time);
                printf("sync_time=%lf(s)\n", all_sync_time);
                printf("all_graph_sync_time=%lf(s)\n", all_graph_sync_time);
                printf("copy_time=%lf(s)\n", all_copy_time);
                printf("nn_time=%lf(s)\n", all_compute_time);
                printf("graph_time=%lf(s)\n", all_graph_time);
                printf("communicate_extract+send=%lf(s)\n", graph->all_compute_time);
                printf("communicate_processing_received=%lf(s)\n", graph->all_overlap_time);
                printf("communicate_processing_received.copy=%lf(s)\n", graph->all_recv_copy_time);
                printf("communicate_processing_received.kernel=%lf(s)\n", graph->all_recv_kernel_time);
                printf("communicate_processing_received.wait=%lf(s)\n", graph->all_recv_wait_time);
                printf("communicate_wait=%lf(s)\n", graph->all_wait_time);
                printf("streamed kernel_time=%lf(s)\n", graph->all_kernel_time);
                printf("streamed movein_time=%lf(s)\n", graph->all_movein_time);
                printf("streamed moveout_time=%lf(s)\n", graph->all_moveout_time);
                printf("graph repliation time=%lf(s)\n", graph->all_replication_time);
                printf("DEBUG INFO END:::::::::::::::::::::::::::::::\n");
            }

            //     if(i_i==(iterations-1)&&graph->partition_id==0){
            //         torch::Tensor tt_cpu=tt.cpu();
            //        inference(tt_cpu,graph, embedding, pytool,Gnn_v1,Gnn_v2);
            //     }
        }
    }

    delete active;
    */
}

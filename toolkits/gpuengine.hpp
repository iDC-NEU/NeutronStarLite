#include "core/gnnmini.h"

/*GPU single*/ void compute_single_GPU(Graph<Empty> *graph, int iterations) {
  ValueType learn_rate = 0.01;
  const int BATCH_SIZE = 2208;
  VertexSubset *active = graph->alloc_vertex_subset();
  active->fill();
  Embeddings<ValueType, long> *embedding = new Embeddings<ValueType, long>();
  embedding->init(graph);
  embedding->readlabel_(graph);

  Network<ValueType> *comm =
      new Network<ValueType>(graph, MAX_LAYER, MAX_LAYER);
  Network<ValueType> *comm1 =
      new Network<ValueType>(graph, MAX_LAYER, MAX_LAYER);
  comm->setWsize(MAX_LAYER, MAX_LAYER);
  comm1->setWsize(MAX_LAYER, MAX_LAYER);
  tensorSet *pytool = new tensorSet(2);
  graph->generate_COO(active);
  graph->reorder_COO(BATCH_SIZE);
  std::vector<edge_list *> edge_list;
  generate_edge_list_Tensor(graph, edge_list, BATCH_SIZE);
  /*init GPU*/

  VertexId *incoming_adj_index = new VertexId[graph->vertices + 1];
  VertexId *incoming_adj_index_backward = new VertexId[graph->vertices + 1];
  ValueType *weight = new ValueType[graph->edges + 1];
  ValueType *weight_backward = new ValueType[graph->edges + 1];
  torch::Tensor W_for_gpu;
  torch::Tensor W_back_gpu;
  generate_weight_and_csr(graph, active, incoming_adj_index,
                          incoming_adj_index_backward, weight, weight_backward,
                          W_for_gpu, W_back_gpu);

  /*1 INIT STAGE*/
  GnnUnit *Gnn_v1 = new GnnUnit(SIZE_LAYER_1, SIZE_LAYER_2);
  GnnUnit *Gnn_v1_1 = new GnnUnit(SIZE_LAYER_1, SIZE_LAYER_2); // commnet
  GnnUnit *Gnn_v2 = new GnnUnit(SIZE_LAYER_2, OUTPUT_LAYER_3);
  GnnUnit *Gnn_v2_1 = new GnnUnit(SIZE_LAYER_2, OUTPUT_LAYER_3); // commnet
  pytool->registOptimizer(torch::optim::SGD(Gnn_v1->parameters(), 0.05)); // new
  pytool->registOptimizer(torch::optim::SGD(Gnn_v2->parameters(), 0.05)); // new
  pytool->registOptimizer(
      torch::optim::SGD(Gnn_v1_1->parameters(), 0.05)); // commnet
  pytool->registOptimizer(
      torch::optim::SGD(Gnn_v2_1->parameters(), 0.05)); // commnet
  pytool->registLabel<long>(
      embedding->label, graph->partition_offset[graph->partition_id],
      graph->partition_offset[graph->partition_id + 1] -
          graph->partition_offset[graph->partition_id]); // new
  /*init W with new */

  init_parameter(comm, graph, Gnn_v1, embedding);
  init_parameter(comm1, graph, Gnn_v2, embedding);

  std::vector<int> layer_size(0);
  layer_size.push_back(SIZE_LAYER_1);
  layer_size.push_back(SIZE_LAYER_2);
  layer_size.push_back(OUTPUT_LAYER_3);
  GTensor<ValueType, long, MAX_LAYER> *gt =
      new GTensor<ValueType, long, MAX_LAYER>(graph, embedding, active, 2,
                                              layer_size);

  torch::Tensor new_combine_grad =
      torch::zeros({SIZE_LAYER_1, SIZE_LAYER_2}, torch::kFloat).cuda();
  std::vector<torch::Tensor> partial_new_combine_grad(0);
  for (int i = 0; i < graph->threads; i++) {
    partial_new_combine_grad.push_back(new_combine_grad);
  }

  graph->process_vertices<ValueType>( // init  the vertex state.
      [&](VertexId vtx) {
        int start = (graph->partition_offset[graph->partition_id]);
        for (int i = 0; i < SIZE_LAYER_1; i++) {
          embedding->initStartWith(vtx, embedding->con[vtx].att[i],
                                   i); // embedding->con[vtx].att[i]
        }
        return (ValueType)1;
      },
      active);

  //   gt->Test_Propagate<SIZE_LAYER_1>(0);
  /*GPU  */
  torch::Device GPU(torch::kCUDA, 0);
  torch::Device CPU(torch::kCPU, 0);
  torch::Tensor target_gpu = pytool->target.cuda();
  torch::Tensor inter1_gpu = torch::zeros(
      {embedding->rownum, SIZE_LAYER_2},
      at::TensorOptions().device_index(0).requires_grad(true).dtype(
          torch::kFloat));
  //     torch::Tensor
  //     inter2_gpu=torch::zeros({embedding->rownum,SIZE_LAYER_2},at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
  Gnn_v2->to(GPU);
  Gnn_v1->to(GPU);

  //  torch::Tensor
  //  X1_gpu=torch::zeros({embedding->rownum,SIZE_LAYER_1},at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
  torch::Tensor X0_cpu =
      torch::from_blob(embedding->start_v + embedding->start,
                       {embedding->rownum, SIZE_LAYER_1}, torch::kFloat);
  torch::Tensor Y0_cpu = torch::zeros({embedding->rownum, SIZE_LAYER_1},
                                      at::TensorOptions().dtype(torch::kFloat));
  torch::Tensor X1_cpu = torch::zeros({embedding->rownum, SIZE_LAYER_2},
                                      at::TensorOptions().dtype(torch::kFloat));
  torch::Tensor Y1_cpu = torch::zeros({embedding->rownum, SIZE_LAYER_2},
                                      at::TensorOptions().dtype(torch::kFloat));
  torch::Tensor Y1_inv_cpu =
      torch::zeros({embedding->rownum, SIZE_LAYER_2},
                   at::TensorOptions().dtype(torch::kFloat));
  torch::Tensor Y0_gpu = torch::zeros(
      {embedding->rownum, SIZE_LAYER_1},
      at::TensorOptions().device_index(0).requires_grad(true).dtype(
          torch::kFloat));
  torch::Tensor Y1_gpu = torch::zeros(
      {embedding->rownum, SIZE_LAYER_2},
      at::TensorOptions().device_index(0).requires_grad(true).dtype(
          torch::kFloat));
  torch::Tensor Y1_inv_gpu = torch::zeros(
      {embedding->rownum, SIZE_LAYER_2},
      at::TensorOptions().device_index(0).requires_grad(true).dtype(
          torch::kFloat));
  torch::Tensor Out0_gpu;
  torch::Tensor Out1_gpu;

  double exec_time = 0;
  double graph_time = 0;
  exec_time -= get_time();
  double tmp_time = 0;

  for (int i_i = 0; i_i < iterations; i_i++) {
    if (i_i != 0) {
      // inter1_gpu.grad().zero_();
      // inter2_gpu.grad().zero_();
      Gnn_v1->zero_grad();
      Gnn_v2->zero_grad();
    }

    if (graph->partition_id == 0)
      std::cout << "start  [" << i_i
                << "]  epoch+++++++++++++++++++++++++++++++++++++++++++++"
                << std::endl;
    // layer 1;
    Y0_cpu.zero_();
    propagate_forward_gpu_shard(graph, X0_cpu, Y0_cpu, edge_list, SIZE_LAYER_1);
    //    for(int i=0;i<1433;i++){
    //        std::cout<<Y0_cpu.accessor<float,2>().data()[i]<<"\t";
    //    }
    //    std::cout<<std::endl;
    Y0_gpu = Y0_cpu.cuda();

    inter1_gpu.set_data(Gnn_v1->forward(Y0_gpu));
    Out0_gpu = torch::relu(inter1_gpu); // new

    // layer 2;
    X1_cpu.set_data(Out0_gpu.cpu());
    Y1_cpu.zero_();
    propagate_forward_gpu_shard(graph, X1_cpu, Y1_cpu, edge_list, SIZE_LAYER_2);
    Y1_gpu.set_data(Y1_cpu.cuda());
    Out1_gpu = Gnn_v2->forward(Y1_gpu);

    // output;
    torch::Tensor tt = Out1_gpu.log_softmax(1);     // CUDA
    pytool->loss = torch::nll_loss(tt, target_gpu); // new
    pytool->loss.backward();
    // inv layer 2;
    Gnn_v2->learn_gpu(Gnn_v2->W.grad(), learn_rate); // signle node
                                                     // inv layer 1;
                                                     // 2->1
    Y1_inv_cpu.zero_();
    propagate_forward_gpu_shard(graph, Y1_gpu.grad().cpu(), Y1_inv_cpu,
                                edge_list, SIZE_LAYER_2);
    Y1_inv_gpu = Y1_inv_cpu.cuda();

    // layer 1 local
    Out0_gpu.backward(); // new
                         // layer 1 combine
    new_combine_grad.zero_();

    new_combine_grad = Y0_gpu.t().mm(Y1_inv_gpu * inter1_gpu.grad());
    // learn
    Gnn_v1->learn_gpu(new_combine_grad, learn_rate);
    if (graph->partition_id == 0)
      std::cout << "LOSS:\t" << pytool->loss << std::endl;

    if (i_i == (iterations - 1)) { //&&graph->partition_id==0
      exec_time += get_time();
      if (graph->partition_id == 0) {
        printf("GRAPH_time=%lf(s)\n", exec_time);
      }
      torch::Tensor tt_cpu = tt.cpu();
      if (i_i == (iterations - 1) && graph->partition_id == 0) {
        inference(tt_cpu, graph, embedding, pytool, Gnn_v1, Gnn_v2);
      }
    }
  }
  delete active;
}

/*GPU single order*/ void compute_single_GPU_old(Graph<Empty> *graph,
                                                 int iterations) {
  ValueType learn_rate = 0.01;
  VertexSubset *active = graph->alloc_vertex_subset();
  active->fill();
  Embeddings<ValueType, long> *embedding = new Embeddings<ValueType, long>();
  embedding->init(graph);
  embedding->readlabel_(graph);

  Network<ValueType> *comm =
      new Network<ValueType>(graph, MAX_LAYER, MAX_LAYER);
  Network<ValueType> *comm1 =
      new Network<ValueType>(graph, MAX_LAYER, MAX_LAYER);
  comm->setWsize(MAX_LAYER, MAX_LAYER);
  comm1->setWsize(MAX_LAYER, MAX_LAYER);
  tensorSet *pytool = new tensorSet(2);
  // pytool->in_degree=torch::from_blob(graph->in_degree+graph->partition_offset[graph->partition_id],{embedding->rownum,1});
  // pytool->out_degree=torch::from_blob(graph->in_degree+graph->partition_offset[graph->partition_id],{embedding->rownum,1});
  /*init GPU*/
  aggregate_engine *ag_e = new aggregate_engine();
  ag_e->reconfig_data(graph->vertices, SIZE_LAYER_2, graph->vertices,
                      SIZE_LAYER_1, TENSOR_TYPE);
  ag_e->init_intermediate_gradient();

  graph_engine *gr_e = new graph_engine();
  graph->generate_COO(active);
  graph->reorder_COO(4096);

  VertexId *incoming_adj_index = new VertexId[graph->vertices + 1];
  VertexId *incoming_adj_index_backward = new VertexId[graph->vertices + 1];
  ValueType *weight = new ValueType[graph->edges + 1];
  ValueType *weight_backward = new ValueType[graph->edges + 1];
  graph->process_vertices<ValueType>( // init  the vertex state.
      [&](VertexId vtx) {
        // graph->in
        incoming_adj_index[vtx] = (VertexId)graph->incoming_adj_index[0][vtx];
        incoming_adj_index_backward[vtx] =
            (VertexId)graph->incoming_adj_index_backward[0][vtx];
        for (int i = graph->incoming_adj_index[0][vtx];
             i < graph->incoming_adj_index[0][vtx + 1]; i++) {
          VertexId dst = graph->incoming_adj_list[0][i].neighbour;
          weight[i] = (ValueType)std::sqrt(graph->in_degree[vtx]) *
                      (ValueType)std::sqrt(graph->out_degree[dst]);
        }
        for (int i = graph->incoming_adj_index_backward[0][vtx];
             i < graph->incoming_adj_index_backward[0][vtx + 1]; i++) {
          VertexId dst = graph->incoming_adj_list_backward[0][i].neighbour;
          weight_backward[i] = (ValueType)std::sqrt(graph->out_degree[vtx]) *
                               (ValueType)std::sqrt(graph->in_degree[dst]);
        }

        return (ValueType)1;
      },
      active);
  incoming_adj_index[graph->vertices] =
      (VertexId)graph->incoming_adj_index[0][graph->vertices];
  incoming_adj_index_backward[graph->vertices] =
      (VertexId)graph->incoming_adj_index_backward[0][graph->vertices];

  gr_e->load_graph(graph->vertices, graph->edges, false, graph->vertices,
                   graph->edges, false, incoming_adj_index,
                   (VertexId *)graph->incoming_adj_list[0],
                   incoming_adj_index_backward,
                   (VertexId *)graph->incoming_adj_list_backward[0], 0,
                   graph->vertices, 0, graph->vertices, SIZE_LAYER_1);

  //  gr_e->load_graph_for_COO(graph->vertices,graph->edges,false,
  //         graph->_graph_cpu->srcList,graph->_graph_cpu->dstList,
  //                 0,graph->vertices,0,graph->vertices,SIZE_LAYER_1);
  // gr_e->_graph_cuda->init_partitions(graph->partitions,graph->_graph_cpu->partition_offset,graph->partition_id);

  /*1 INIT STAGE*/
  GnnUnit *Gnn_v1 = new GnnUnit(SIZE_LAYER_1, SIZE_LAYER_2);
  GnnUnit *Gnn_v1_1 = new GnnUnit(SIZE_LAYER_1, SIZE_LAYER_2); // commnet
  GnnUnit *Gnn_v2 = new GnnUnit(SIZE_LAYER_2, OUTPUT_LAYER_3);
  GnnUnit *Gnn_v2_1 = new GnnUnit(SIZE_LAYER_2, OUTPUT_LAYER_3); // commnet
  pytool->registOptimizer(torch::optim::SGD(Gnn_v1->parameters(), 0.05)); // new
  pytool->registOptimizer(torch::optim::SGD(Gnn_v2->parameters(), 0.05)); // new
  pytool->registOptimizer(
      torch::optim::SGD(Gnn_v1_1->parameters(), 0.05)); // commnet
  pytool->registOptimizer(
      torch::optim::SGD(Gnn_v2_1->parameters(), 0.05)); // commnet
  pytool->registLabel<long>(
      embedding->label, graph->partition_offset[graph->partition_id],
      graph->partition_offset[graph->partition_id + 1] -
          graph->partition_offset[graph->partition_id]); // new
  /*init W with new */

  init_parameter(comm, graph, Gnn_v1, embedding);
  init_parameter(comm1, graph, Gnn_v2, embedding);

  std::vector<int> layer_size(0);
  layer_size.push_back(SIZE_LAYER_1);
  layer_size.push_back(SIZE_LAYER_2);
  layer_size.push_back(OUTPUT_LAYER_3);
  GTensor<ValueType, long, MAX_LAYER> *gt =
      new GTensor<ValueType, long, MAX_LAYER>(graph, embedding, active, 2,
                                              layer_size);

  torch::Tensor new_combine_grad =
      torch::zeros({SIZE_LAYER_1, SIZE_LAYER_2}, torch::kFloat).cuda();

  graph->process_vertices<ValueType>( // init  the vertex state.
      [&](VertexId vtx) {
        int start = (graph->partition_offset[graph->partition_id]);
        for (int i = 0; i < SIZE_LAYER_1; i++) {
          embedding->initStartWith(vtx, embedding->con[vtx].att[i],
                                   i); // embedding->con[vtx].att[i]
        }
        return (ValueType)1;
      },
      active);

  //   gt->Test_Propagate<SIZE_LAYER_1>(0);
  /*GPU  */
  torch::Device GPU(torch::kCUDA, 0);
  torch::Device CPU(torch::kCPU, 0);
  torch::Tensor target_gpu = pytool->target.cuda();
  torch::Tensor inter1_gpu = torch::zeros(
      {embedding->rownum, SIZE_LAYER_2},
      at::TensorOptions().device_index(0).requires_grad(true).dtype(
          torch::kFloat));
  //     torch::Tensor
  //     inter2_gpu=torch::zeros({embedding->rownum,SIZE_LAYER_2},at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
  Gnn_v2->to(GPU);
  Gnn_v1->to(GPU);

  //  torch::Tensor
  //  X1_gpu=torch::zeros({embedding->rownum,SIZE_LAYER_1},at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));

  torch::Tensor X0_gpu =
      torch::from_blob(embedding->start_v + embedding->start,
                       {embedding->rownum, SIZE_LAYER_1}, torch::kFloat)
          .cuda();
  // torch::Tensor
  // X0_gpu_trans=torch::zeros({embedding->rownum,SIZE_LAYER_2},torch::kFloat).cuda();
  torch::Tensor Y0_gpu = torch::zeros(
      {embedding->rownum, SIZE_LAYER_1},
      at::TensorOptions().device_index(0).requires_grad(true).dtype(
          torch::kFloat));
  torch::Tensor Y1_gpu = torch::zeros(
      {embedding->rownum, SIZE_LAYER_2},
      at::TensorOptions().device_index(0).requires_grad(true).dtype(
          torch::kFloat));
  torch::Tensor Y1_inv_gpu = torch::zeros(
      {embedding->rownum, SIZE_LAYER_2},
      at::TensorOptions().device_index(0).requires_grad(true).dtype(
          torch::kFloat));
  torch::Tensor Y0_inv_gpu = torch::zeros(
      {embedding->rownum, SIZE_LAYER_2},
      at::TensorOptions().device_index(0).requires_grad(true).dtype(
          torch::kFloat));
  torch::Tensor W_for_gpu =
      torch::from_blob(weight, {graph->edges + 1, 1}, torch::kFloat).cuda();
  torch::Tensor W_back_gpu =
      torch::from_blob(weight_backward, {graph->edges + 1, 1}, torch::kFloat)
          .cuda();
  torch::Tensor Out0_gpu;
  torch::Tensor Out1_gpu;

  double exec_time = 0;
  double graph_time = 0;
  double all_graph_time = 0;
  double all_nn_time = 0;
  double nn_time = 0;
  exec_time -= get_time();
  for (int i_i = 0; i_i < iterations; i_i++) {
    if (i_i != 0) {
      nn_time = 0;
      nn_time -= get_time();

      // inter1_gpu.grad().zero_();
      // inter2_gpu.grad().zero_();
      Gnn_v1->zero_grad();
      Gnn_v2->zero_grad();
      nn_time += get_time();
      all_nn_time += nn_time;
    }
    std::cout << "why nit compile" << std::endl;
    if (graph->partition_id == 0)
      std::cout << "start  [" << i_i
                << "]  epoch+++++++++++++++++++++++++++++++++++++++++++++"
                << std::endl;
    // layer 1;
    graph_time = 0;
    graph_time -= get_time();
    // X0_gpu_trans=Gnn_v1->forward(X0_gpu);
    gr_e->forward_one_step(X0_gpu.packed_accessor<float, 2>().data(),
                           Y0_gpu.packed_accessor<float, 2>().data(),
                           W_for_gpu.packed_accessor<float, 2>().data(),
                           SCALA_TYPE, SIZE_LAYER_1);

    graph_time += get_time();
    all_graph_time += graph_time;
    torch::Tensor tmp1 = Y0_gpu.cpu();
    nn_time = 0;
    nn_time -= get_time();
    printf("wrong\n");
    std::cout << inter1_gpu.size(0) << " " << inter1_gpu.size(1) << "\t\t\t\t"
              << Y0_gpu.size(0) << " " << Y0_gpu.size(1) << std::endl;
    std::cout << Gnn_v1->W.size(0) << " " << Gnn_v1->W.size(1) << std::endl;
    inter1_gpu.set_data(Gnn_v1->forward(Y0_gpu));
    printf("wrong0\n");
    Out0_gpu = torch::relu(inter1_gpu);
    nn_time += get_time();
    all_nn_time += nn_time;
    printf("wrong1\n");
    // layer 2;
    graph_time = 0;
    graph_time -= get_time();
    gr_e->forward_one_step(Out0_gpu.packed_accessor<float, 2>().data(),
                           Y1_gpu.packed_accessor<float, 2>().data(),
                           W_for_gpu.packed_accessor<float, 2>().data(),
                           SCALA_TYPE, SIZE_LAYER_2);
    graph_time += get_time();
    all_graph_time += graph_time;
    torch::Tensor tmp2 = Y1_gpu.cpu();
    nn_time = 0;
    nn_time -= get_time();
    Out1_gpu = Gnn_v2->forward(Y1_gpu);
    // output;
    torch::Tensor tt = Out1_gpu.log_softmax(1);     // CUDA
    pytool->loss = torch::nll_loss(tt, target_gpu); // new
    pytool->loss.backward();
    // inv layer 2;
    Gnn_v2->learn_gpu(Gnn_v2->W.grad(), learn_rate); // signle node
    // inv layer 1;
    nn_time += get_time();
    all_nn_time += nn_time;
    // 2->1
    graph_time = 0;
    graph_time -= get_time();
    gr_e->backward_one_step(Y1_gpu.grad().packed_accessor<float, 2>().data(),
                            Y1_inv_gpu.packed_accessor<float, 2>().data(),
                            W_back_gpu.packed_accessor<float, 2>().data(),
                            SCALA_TYPE, SIZE_LAYER_2);
    graph_time += get_time();
    all_graph_time += graph_time;
    torch::Tensor tmp3 = Y1_inv_gpu.cpu();
    nn_time = 0;
    nn_time -= get_time();
    Out0_gpu.backward(); // new
                         // layer 1 combine
    new_combine_grad.zero_();
    new_combine_grad = Y0_gpu.t().mm(Y1_inv_gpu * inter1_gpu.grad());
    // learn
    Gnn_v1->learn_gpu(new_combine_grad, learn_rate);
    nn_time += get_time();
    all_nn_time += nn_time;
    if (graph->partition_id == 0)
      std::cout << "LOSS:\t" << pytool->loss << std::endl;

    if (i_i == (iterations - 1)) { //&&graph->partition_id==0
      exec_time += get_time();
      if (graph->partition_id == 0) {
        printf("exec_time=%lf(s)\n", exec_time);
        printf("graph_time=%lf(s)\n", all_graph_time);
        printf("kernel_time=%lf(s)\n", gr_e->overall_time);
        printf("nn_time=%lf(s)\n", all_nn_time);
      }
      torch::Tensor tt_cpu = tt.cpu();
      if (i_i == (iterations - 1) && graph->partition_id == 0) {
        inference(tt_cpu, graph, embedding, pytool, Gnn_v1, Gnn_v2);
      }
    }
  }
  delete active;
}

/*GPU single order*/ void compute_single_gf_GPU(Graph<Empty> *graph,
                                                int iterations) {
  ValueType learn_rate = 0.01;
  VertexSubset *active = graph->alloc_vertex_subset();
  active->fill();
  Embeddings<ValueType, long> *embedding = new Embeddings<ValueType, long>();
  embedding->init(graph);
  embedding->readlabel_(graph);

  Network<ValueType> *comm =
      new Network<ValueType>(graph, MAX_LAYER, MAX_LAYER);
  Network<ValueType> *comm1 =
      new Network<ValueType>(graph, MAX_LAYER, MAX_LAYER);
  comm->setWsize(MAX_LAYER, MAX_LAYER);
  comm1->setWsize(MAX_LAYER, MAX_LAYER);
  tensorSet *pytool = new tensorSet(2);
  // pytool->in_degree=torch::from_blob(graph->in_degree+graph->partition_offset[graph->partition_id],{embedding->rownum,1});
  // pytool->out_degree=torch::from_blob(graph->in_degree+graph->partition_offset[graph->partition_id],{embedding->rownum,1});
  /*init GPU*/
  aggregate_engine *ag_e = new aggregate_engine();
  ag_e->reconfig_data(graph->vertices, SIZE_LAYER_2, graph->vertices,
                      SIZE_LAYER_1, TENSOR_TYPE);
  ag_e->init_intermediate_gradient();

  graph_engine *gr_e = new graph_engine();
  graph->generate_COO(active);
  graph->reorder_COO(4096);

  VertexId *incoming_adj_index = new VertexId[graph->vertices + 1];
  VertexId *incoming_adj_index_backward = new VertexId[graph->vertices + 1];
  ValueType *weight = new ValueType[graph->edges + 1];
  ValueType *weight_backward = new ValueType[graph->edges + 1];
  graph->process_vertices<ValueType>( // init  the vertex state.
      [&](VertexId vtx) {
        // graph->in
        incoming_adj_index[vtx] = (VertexId)graph->incoming_adj_index[0][vtx];
        incoming_adj_index_backward[vtx] =
            (VertexId)graph->incoming_adj_index_backward[0][vtx];
        for (int i = graph->incoming_adj_index[0][vtx];
             i < graph->incoming_adj_index[0][vtx + 1]; i++) {
          VertexId dst = graph->incoming_adj_list[0][i].neighbour;
          weight[i] = (ValueType)std::sqrt(graph->in_degree[vtx]) *
                      (ValueType)std::sqrt(graph->out_degree[dst]);
        }
        for (int i = graph->incoming_adj_index_backward[0][vtx];
             i < graph->incoming_adj_index_backward[0][vtx + 1]; i++) {
          VertexId dst = graph->incoming_adj_list_backward[0][i].neighbour;
          weight_backward[i] = (ValueType)std::sqrt(graph->out_degree[vtx]) *
                               (ValueType)std::sqrt(graph->in_degree[dst]);
        }

        return (ValueType)1;
      },
      active);
  incoming_adj_index[graph->vertices] =
      (VertexId)graph->incoming_adj_index[0][graph->vertices];
  incoming_adj_index_backward[graph->vertices] =
      (VertexId)graph->incoming_adj_index_backward[0][graph->vertices];

  gr_e->load_graph(graph->vertices, graph->edges, false, graph->vertices,
                   graph->edges, false, incoming_adj_index,
                   (VertexId *)graph->incoming_adj_list[0],
                   incoming_adj_index_backward,
                   (VertexId *)graph->incoming_adj_list_backward[0], 0,
                   graph->vertices, 0, graph->vertices, SIZE_LAYER_1);

  //  gr_e->load_graph_for_COO(graph->vertices,graph->edges,false,
  //         graph->_graph_cpu->srcList,graph->_graph_cpu->dstList,
  //                 0,graph->vertices,0,graph->vertices,SIZE_LAYER_1);
  // gr_e->_graph_cuda->init_partitions(graph->partitions,graph->_graph_cpu->partition_offset,graph->partition_id);

  /*1 INIT STAGE*/
  GnnUnit *Gnn_v1 = new GnnUnit(SIZE_LAYER_1, SIZE_LAYER_2);
  GnnUnit *Gnn_v1_1 = new GnnUnit(SIZE_LAYER_1, SIZE_LAYER_2); // commnet
  GnnUnit *Gnn_v2 = new GnnUnit(SIZE_LAYER_2, OUTPUT_LAYER_3);
  GnnUnit *Gnn_v2_1 = new GnnUnit(SIZE_LAYER_2, OUTPUT_LAYER_3); // commnet
  pytool->registOptimizer(torch::optim::SGD(Gnn_v1->parameters(), 0.05)); // new
  pytool->registOptimizer(torch::optim::SGD(Gnn_v2->parameters(), 0.05)); // new
  pytool->registOptimizer(
      torch::optim::SGD(Gnn_v1_1->parameters(), 0.05)); // commnet
  pytool->registOptimizer(
      torch::optim::SGD(Gnn_v2_1->parameters(), 0.05)); // commnet
  pytool->registLabel<long>(
      embedding->label, graph->partition_offset[graph->partition_id],
      graph->partition_offset[graph->partition_id + 1] -
          graph->partition_offset[graph->partition_id]); // new
  /*init W with new */

  init_parameter(comm, graph, Gnn_v1, embedding);
  init_parameter(comm1, graph, Gnn_v2, embedding);

  std::vector<int> layer_size(0);
  layer_size.push_back(SIZE_LAYER_1);
  layer_size.push_back(SIZE_LAYER_2);
  layer_size.push_back(OUTPUT_LAYER_3);
  GTensor<ValueType, long, MAX_LAYER> *gt =
      new GTensor<ValueType, long, MAX_LAYER>(graph, embedding, active, 2,
                                              layer_size);

  torch::Tensor new_combine_grad =
      torch::zeros({SIZE_LAYER_1, SIZE_LAYER_2}, torch::kFloat).cuda();

  graph->process_vertices<ValueType>( // init  the vertex state.
      [&](VertexId vtx) {
        int start = (graph->partition_offset[graph->partition_id]);
        for (int i = 0; i < SIZE_LAYER_1; i++) {
          embedding->initStartWith(vtx, embedding->con[vtx].att[i],
                                   i); // embedding->con[vtx].att[i]
        }
        return (ValueType)1;
      },
      active);

  //   gt->Test_Propagate<SIZE_LAYER_1>(0);
  /*GPU  */
  torch::Device GPU(torch::kCUDA, 0);
  torch::Device CPU(torch::kCPU, 0);
  torch::Tensor target_gpu = pytool->target.cuda();
  torch::Tensor inter1_gpu = torch::zeros(
      {embedding->rownum, SIZE_LAYER_2},
      at::TensorOptions().device_index(0).requires_grad(true).dtype(
          torch::kFloat));
  //     torch::Tensor
  //     inter2_gpu=torch::zeros({embedding->rownum,SIZE_LAYER_2},at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));
  Gnn_v2->to(GPU);
  Gnn_v1->to(GPU);

  //  torch::Tensor
  //  X1_gpu=torch::zeros({embedding->rownum,SIZE_LAYER_1},at::TensorOptions().device_index(0).requires_grad(true).dtype(torch::kFloat));

  torch::Tensor X0_gpu =
      torch::from_blob(embedding->start_v + embedding->start,
                       {embedding->rownum, SIZE_LAYER_1}, torch::kFloat)
          .cuda();
  torch::Tensor X0_gpu_trans =
      torch::zeros({embedding->rownum, SIZE_LAYER_2}, torch::kFloat).cuda();
  torch::Tensor Y0_gpu = torch::zeros(
      {embedding->rownum, SIZE_LAYER_2},
      at::TensorOptions().device_index(0).requires_grad(true).dtype(
          torch::kFloat));
  torch::Tensor Y1_gpu = torch::zeros(
      {embedding->rownum, SIZE_LAYER_2},
      at::TensorOptions().device_index(0).requires_grad(true).dtype(
          torch::kFloat));
  torch::Tensor Y1_inv_gpu = torch::zeros(
      {embedding->rownum, SIZE_LAYER_2},
      at::TensorOptions().device_index(0).requires_grad(true).dtype(
          torch::kFloat));
  torch::Tensor Y0_inv_gpu = torch::zeros(
      {embedding->rownum, SIZE_LAYER_2},
      at::TensorOptions().device_index(0).requires_grad(true).dtype(
          torch::kFloat));
  torch::Tensor W_for_gpu =
      torch::from_blob(weight, {graph->edges + 1, 1}, torch::kFloat).cuda();
  torch::Tensor W_back_gpu =
      torch::from_blob(weight_backward, {graph->edges + 1, 1}, torch::kFloat)
          .cuda();
  torch::Tensor Out0_gpu;
  torch::Tensor Out1_gpu;

  double exec_time = 0;
  exec_time -= get_time();
  for (int i_i = 0; i_i < iterations; i_i++) {
    if (i_i != 0) {
      // inter1_gpu.grad().zero_();
      // inter2_gpu.grad().zero_();
      Gnn_v1->zero_grad();
      Gnn_v2->zero_grad();
    }

    if (graph->partition_id == 0)
      std::cout << "start  [" << i_i
                << "]  epoch+++++++++++++++++++++++++++++++++++++++++++++"
                << std::endl;
    // layer 1;

    X0_gpu_trans = Gnn_v1->forward(X0_gpu);
    gr_e->forward_one_step(X0_gpu_trans.packed_accessor<float, 2>().data(),
                           Y0_gpu.packed_accessor<float, 2>().data(),
                           W_for_gpu.packed_accessor<float, 2>().data(),
                           SCALA_TYPE, SIZE_LAYER_2);

    Out0_gpu = torch::relu(Y0_gpu);
    // layer 2;
    gr_e->forward_one_step(Out0_gpu.packed_accessor<float, 2>().data(),
                           Y1_gpu.packed_accessor<float, 2>().data(),
                           W_for_gpu.packed_accessor<float, 2>().data(),
                           SCALA_TYPE, SIZE_LAYER_2);
    Out1_gpu = Gnn_v2->forward(Y1_gpu);
    // output;
    torch::Tensor tt = Out1_gpu.log_softmax(1);     // CUDA
    pytool->loss = torch::nll_loss(tt, target_gpu); // new
    pytool->loss.backward();
    // inv layer 2;
    Gnn_v2->learn_gpu(Gnn_v2->W.grad(), learn_rate); // signle node
                                                     // inv layer 1;

    // 2->1
    gr_e->backward_one_step(Y1_gpu.grad().packed_accessor<float, 2>().data(),
                            Y1_inv_gpu.packed_accessor<float, 2>().data(),
                            W_back_gpu.packed_accessor<float, 2>().data(),
                            SCALA_TYPE, SIZE_LAYER_2);

    // layer 1 local
    Out0_gpu.backward(); // new
    torch::Tensor tmp = Y1_inv_gpu * Y0_gpu.grad();
    gr_e->backward_one_step(tmp.packed_accessor<float, 2>().data(),
                            Y0_inv_gpu.packed_accessor<float, 2>().data(),
                            W_back_gpu.packed_accessor<float, 2>().data(),
                            SCALA_TYPE, SIZE_LAYER_2);

    // layer 1 combine
    new_combine_grad.zero_();
    new_combine_grad = X0_gpu.t().mm(Y0_inv_gpu);
    // learn

    Gnn_v1->learn_gpu(new_combine_grad, learn_rate);

    if (graph->partition_id == 0)
      std::cout << "LOSS:\t" << pytool->loss << std::endl;

    if (i_i == (iterations - 1)) { //&&graph->partition_id==0
      exec_time += get_time();
      if (graph->partition_id == 0) {
        printf("GRAPH_time=%lf(s)\n", exec_time);
      }
      torch::Tensor tt_cpu = tt.cpu();
      if (i_i == (iterations - 1) && graph->partition_id == 0) {
        inference(tt_cpu, graph, embedding, pytool, Gnn_v1, Gnn_v2);
      }
    }
  }
  delete active;
}


#include "comm/logger.h"
#include "core/AutoDiff.h"
#include "core/gnnmini.h"

class GCN_CPU_impl {
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
  // graph
  VertexSubset *active;
  // graph with no edge data
  Graph<Empty> *graph;
  std::vector<CSC_segment_pinned *> subgraphs;
  // NN
  GNNDatum *gnndatum;
  NtsVar L_GT_C;
  NtsVar L_GT_G;
  NtsVar MASK;
  std::map<std::string, NtsVar> I_data;
  GraphOperation *gt;
  // Variables
  std::vector<Parameter *> P;
  std::vector<NtsVar> X;
  std::vector<NtsVar> Y;
  std::vector<NtsVar> X_grad;
  nts::autodiff::ComputionPath *cp;
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

  GCN_CPU_impl(Graph<Empty> *graph_, int iterations_,
               bool process_local = false, bool process_overlap = false) {
    graph = graph_;
    iterations = iterations_;

    active = graph->alloc_vertex_subset();
    active->fill();

    graph->init_gnnctx(graph->config->layer_string);
    // rtminfo initialize
    graph->init_rtminfo();
    graph->rtminfo->process_local = graph->config->process_local;
    graph->rtminfo->reduce_comm = graph->config->process_local;
    graph->rtminfo->copy_data = false;
    graph->rtminfo->process_overlap = graph->config->overlap;
    graph->rtminfo->with_weight = true;
    graph->rtminfo->with_cuda = false;
    graph->rtminfo->lock_free = graph->config->lock_free;
  }
  void init_graph() {
    // std::vector<CSC_segment_pinned *> csc_segment;
    graph->generate_COO();
    graph->reorder_COO_W2W();
    // generate_CSC_Segment_Tensor_pinned(graph, csc_segment, true);
    gt = new GraphOperation(graph, active);
    // generate the representation for subgraph corresponding to the way we partitioned
    // e.g. generate CSC/CSR format representation for every subgraph
    gt->GenerateGraphSegment(subgraphs, CPU_T, [&](VertexId src, VertexId dst) {
      return gt->norm_degree(src, dst);
    });
    // gt->GenerateMessageBitmap(subgraphs);
    // pre-process the data that will be used while doing forward and backward propagation
    // which has better support on multisockets.
    gt->GenerateMessageBitmap_multisokects(subgraphs);
    graph->init_communicatior();
    cp = new nts::autodiff::ComputionPath(gt, subgraphs);
  }
  void init_nn() {

    learn_rate = graph->config->learn_rate;
    weight_decay = graph->config->weight_decay;
    drop_rate = graph->config->drop_rate;
    alpha = graph->config->learn_rate;
    decay_rate = graph->config->decay_rate;
    decay_epoch = graph->config->decay_epoch;
    beta1 = 0.9;
    beta2 = 0.999;
    epsilon = 1e-9;

    GNNDatum *gnndatum = new GNNDatum(graph->gnnctx, graph);
    // gnndatum->random_generate();
    if (0 == graph->config->feature_file.compare("random")) {
      gnndatum->random_generate();
    } else {
      gnndatum->readFeature_Label_Mask(graph->config->feature_file,
                                       graph->config->label_file,
                                       graph->config->mask_file);
    }

    // creating tensor to save Label and Mask
    gnndatum->registLabel(L_GT_C);
    gnndatum->registMask(MASK);

    // initializeing parameter. Creating tensor with shape [layer_size[i], layer_size[i + 1]]
    for (int i = 0; i < graph->gnnctx->layer_size.size() - 1; i++) {
      P.push_back(new Parameter(graph->gnnctx->layer_size[i],
                                graph->gnnctx->layer_size[i + 1], alpha, beta1,
                                beta2, epsilon, weight_decay));
    }

    // synchronize parameter with other processes
    // because we need to guarantee all of workers are using the same model
    for (int i = 0; i < P.size(); i++) {
      P[i]->init_parameter();
      P[i]->set_decay(decay_rate, decay_epoch);
    }
    drpmodel = torch::nn::Dropout(
        torch::nn::DropoutOptions().p(drop_rate).inplace(true));

    F = graph->Nts->NewLeafTensor(
        gnndatum->local_feature,
        {graph->gnnctx->l_v_num, graph->gnnctx->layer_size[0]},
        torch::DeviceType::CPU);

    // Y[i] is the aggregated value at layer[i]
    for (int i = 0; i < graph->gnnctx->layer_size.size() - 1; i++) {
      Y.push_back(graph->Nts->NewKeyTensor(
          {graph->gnnctx->l_v_num, graph->gnnctx->layer_size[i]},
          torch::DeviceType::CPU));
    }

    // X[i] is vertex representation at layer i
    for (int i = 0; i < graph->gnnctx->layer_size.size(); i++) {
      NtsVar d;
      X.push_back(d);
    }
    // X[0] is the initial vertex representation. We created it from local_feature
    X[0] = F.set_requires_grad(true);
  }

  void Test(long s) { // 0 train, //1 eval //2 test
    NtsVar mask_train = MASK.eq(s);
    NtsVar all_train =
        X[graph->gnnctx->layer_size.size() - 1]
            .argmax(1)
            .to(torch::kLong)
            .eq(L_GT_C)
            .to(torch::kLong)
            .masked_select(mask_train.view({mask_train.size(0)}));
    NtsVar all = all_train.sum(0);
    long *p_correct = all.data_ptr<long>();
    long g_correct = 0;
    long p_train = all_train.size(0);
    long g_train = 0;
    MPI_Datatype dt = get_mpi_data_type<long>();
    MPI_Allreduce(p_correct, &g_correct, 1, dt, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&p_train, &g_train, 1, dt, MPI_SUM, MPI_COMM_WORLD);
    float acc_train = 0.0;
    if (g_train > 0)
      acc_train = float(g_correct) / g_train;
    if (graph->partition_id == 0) {
      if (s == 0) {
        LOG_INFO("Train Acc: %f %d %d", acc_train, g_train, g_correct);
      } else if (s == 1) {
        LOG_INFO("Eval Acc: %f %d %d", acc_train, g_train, g_correct);
      } else if (s == 2) {
        LOG_INFO("Test Acc: %f %d %d", acc_train, g_train, g_correct);
      }
    }
  }
  NtsVar vertexForward(NtsVar &a, NtsVar &x) {
    NtsVar y;
    int layer = graph->rtminfo->curr_layer;
    // nn operation. Here is just a simple matmul. i.e. y = activate(a * w)
    if (layer == 0) {
      y = torch::relu(P[layer]->forward(a)).set_requires_grad(true);
    } else if (layer == 1) {
      y = P[layer]->forward(a);
      y = y.log_softmax(1);
    }
    // save the intermediate result for backward propagation
    cp->op_push(a, y, nts::autodiff::NNOP);
    return y;
  }
  void Loss() {
    //  return torch::nll_loss(a,L_GT_C);
    torch::Tensor a = X[graph->gnnctx->layer_size.size() - 1];
    torch::Tensor mask_train = MASK.eq(0);
    loss = torch::nll_loss(
        a.masked_select(mask_train.expand({mask_train.size(0), a.size(1)}))
            .view({-1, a.size(1)}),
        L_GT_C.masked_select(mask_train.view({mask_train.size(0)})));
    cp->op_push(a, loss, nts::autodiff::NNOP);
  }

  void Update() {
    for (int i = 0; i < P.size() - 1; i++) {
      // accumulate the gradient using all_reduce
      P[i]->all_reduce_to_gradient(P[i]->W.grad().cpu());
      // update parameters with Adam optimizer
      P[i]->learnC2C_with_decay_Adam();
      P[i]->next();
    }
  }
  void Forward() {
    graph->rtminfo->forward = true;
    for (int i = 0; i < graph->gnnctx->layer_size.size() - 1; i++) {
      graph->rtminfo->curr_layer = i;
      if (i != 0) {
        X[i] = drpmodel(X[i]);
      }

      //gt->PropagateForwardCPU_Lockfree(X[i], Y[i], subgraphs);
      // gather neithbour's vertex feature
      // the intermediate value is stored in Y
      gt->PropagateForwardCPU_Lockfree_multisockets(X[i], Y[i], subgraphs);

      // push the operation and intermediate result into ComputationPath, for backward propagation
      cp->op_push(X[i], Y[i], nts::autodiff::DIST_CPU);

      // fed aggregation value and vertex feature into nn model
      // and compute the new vertex feature
      X[i + 1] = vertexForward(Y[i], X[i]);
    }
  }

  void run() {
    if (graph->partition_id == 0) {
      LOG_INFO("GNNmini::[Dist.GPU.GCNimpl] running [%d] Epoches\n", iterations);
    }

    exec_time -= get_time();
    for (int i_i = 0; i_i < iterations; i_i++) {
      graph->rtminfo->epoch = i_i;
      if (i_i != 0) {
        // clear the gradient in parameters and values
        for (int i = 0; i < P.size(); i++) {
          P[i]->zero_grad();
          Y[i].grad().zero_();
        }
      }
      Forward();
      Test(0);
      Test(1);
      Test(2);
      Loss();
      // Backward();
      cp->self_backward();
      Update();
      // cp->debug();
      if (graph->partition_id == 0)
        std::cout << "Nts::Running.Epoch[" << i_i << "]:loss\t" << loss
                  << std::endl;
    }
    exec_time += get_time();

    delete active;
  }

  void DEBUGINFO() {

    if (graph->partition_id == 0) {
      printf("\n#Timer Info Start:\n");
      printf("#all_time=%lf(s)\n", exec_time);
      printf("#sync_time=%lf(s)\n", all_sync_time);
      printf("#all_graph_sync_time=%lf(s)\n", all_graph_sync_time);
      printf("#copy_time=%lf(s)\n", all_copy_time);
      printf("#nn_time=%lf(s)\n", all_compute_time);
      printf("#graph_time=%lf(s)\n", all_graph_time);
      printf("#communicate_extract+send=%lf(s)\n", graph->all_compute_time);
      printf("#communicate_processing_received=%lf(s)\n",
             graph->all_overlap_time);
      printf("#communicate_processing_received.copy=%lf(s)\n",
             graph->all_recv_copy_time);
      printf("#communicate_processing_received.kernel=%lf(s)\n",
             graph->all_recv_kernel_time);
      printf("#communicate_processing_received.wait=%lf(s)\n",
             graph->all_recv_wait_time);
      printf("#communicate_wait=%lf(s)\n", graph->all_wait_time);
      printf("#streamed kernel_time=%lf(s)\n", graph->all_kernel_time);
      printf("#streamed movein_time=%lf(s)\n", graph->all_movein_time);
      printf("#streamed moveout_time=%lf(s)\n", graph->all_moveout_time);
      printf("#cuda wait time=%lf(s)\n", graph->all_cuda_sync_time);
      printf("#graph repliation time=%lf(s)\n", graph->all_replication_time);
      printf("#Timer Info End\n");
    }

    double max_time = 0;
    double mean_time = 0;
    double another_time = 0;
    MPI_Datatype l_vid_t = get_mpi_data_type<double>();
    MPI_Allreduce(&all_graph_time, &max_time, 1, l_vid_t, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&exec_time, &another_time, 1, l_vid_t, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&graph->all_replication_time, &mean_time, 1, l_vid_t, MPI_SUM,
                  MPI_COMM_WORLD);
    if (graph->partition_id == 0)
      printf("ALL TIME = %lf(s) GRAPH TIME = %lf(s) MEAN TIME = %lf(s)\n",
             another_time, max_time / graph->partitions,
             mean_time / graph->partitions);
  }
  void test_debug() {

    //    graph->rtminfo->forward = true;
    //    graph->rtminfo->curr_layer=0;
    //   gt->PropagateForwardCPU_debug(X[0], Y[0], subgraphs);
    //    for(VertexId i=0;i<graph->partitions;i++)
    //    if(graph->partition_id==i){
    //        int test=graph->gnnctx->p_v_s+1;
    //        std::cout<<"DEBUG"<<graph->in_degree_for_backward[test]<<" X:
    //        "<<X[0][test-graph->gnnctx->p_v_s][15]<<" Y:
    //        "<<Y[0][test-graph->gnnctx->p_v_s][15]<<std::endl;
    //        test=graph->gnnctx->p_v_e-1;
    //        std::cout<<"DEBUG"<<graph->in_degree_for_backward[test]<<" X:
    //        "<<X[0][test-graph->gnnctx->p_v_s][15]<<" Y:
    //        "<<Y[0][test-graph->gnnctx->p_v_s][15]<<std::endl;
    //        test=(graph->gnnctx->p_v_e+graph->gnnctx->p_v_s)/2+1;
    //        std::cout<<"DEBUG"<<graph->in_degree_for_backward[test]<<" X:
    //        "<<X[0][test-graph->gnnctx->p_v_s][15]<<" Y:
    //        "<<Y[0][test-graph->gnnctx->p_v_s][15]<<std::endl;
    //    }

    //    graph->rtminfo->forward = false;
    //    graph->rtminfo->curr_layer=0;
    //    gt->GraphPropagateBackward(X[0], Y[0], subgraphs);
    //    for(VertexId i=0;i<graph->partitions;i++)
    //    if(graph->partition_id==i){
    //        int test=graph->gnnctx->p_v_s;
    //        std::cout<<"DEBUG"<<graph->out_degree_for_backward[test]<<" X:
    //        "<<X[0][test-graph->gnnctx->p_v_s][15]<<" Y:
    //        "<<Y[0][test-graph->gnnctx->p_v_s][15]<<std::endl;
    //        test=graph->gnnctx->p_v_e-1;
    //        std::cout<<"DEBUG"<<graph->out_degree_for_backward[test]<<" X:
    //        "<<X[0][test-graph->gnnctx->p_v_s][15]<<" Y:
    //        "<<Y[0][test-graph->gnnctx->p_v_s][15]<<std::endl;
    //        test=(graph->gnnctx->p_v_e+graph->gnnctx->p_v_s)/2;
    //        std::cout<<"DEBUG"<<graph->out_degree_for_backward[test]<<" X:
    //        "<<X[0][test-graph->gnnctx->p_v_s][15]<<" Y:
    //        "<<Y[0][test-graph->gnnctx->p_v_s][15]<<std::endl;
    //    }
  }
};

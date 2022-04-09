
#include "comm/logger.h"
#include "core/AutoDiff.h"
#include "core/gnnmini.hpp"

class GGNN_CPU_impl {
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
  std::vector<NtsVar> Ei;
  std::vector<NtsVar> Eo;
  std::vector<NtsVar> Ei_grad;
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

  GGNN_CPU_impl(Graph<Empty> *graph_, int iterations_,
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
    gt->GenerateGraphSegment(subgraphs, CPU_T, [&](VertexId src, VertexId dst) {
      return gt->norm_degree(src, dst);
    });
    // gt->GenerateMessageBitmap(subgraphs);
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
    gnndatum->registLabel(L_GT_C);
    gnndatum->registMask(MASK);

    for (int i = 0; i < graph->gnnctx->layer_size.size() - 1; i++) {
      P.push_back(new Parameter(graph->gnnctx->layer_size[i],
                                graph->gnnctx->layer_size[i], alpha, beta1,
                                beta2, epsilon, weight_decay));
      P.push_back(new Parameter(graph->gnnctx->layer_size[i],
                                graph->gnnctx->layer_size[i + 1], alpha, beta1,
                                beta2, epsilon, weight_decay));
    }
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
//      F = graph->Nts->NewOnesTensor(
//        {graph->gnnctx->l_v_num, graph->gnnctx->layer_size[0]},
//        torch::DeviceType::CPU);
//        for(int i=0;i<F.size(0);i++){
//            F[i]=F[i]*i+1;
//        }

    for (int i = 0; i < graph->gnnctx->layer_size.size() - 1; i++) {
        
      Ei.push_back(graph->Nts->NewKeyTensor(
          {graph->gnnctx->l_e_num, graph->gnnctx->layer_size[i]},
          torch::DeviceType::CPU));
      Y.push_back(graph->Nts->NewKeyTensor(
          {graph->gnnctx->l_v_num, graph->gnnctx->layer_size[i]},
          torch::DeviceType::CPU));
    }
    for (int i = 0; i < graph->gnnctx->layer_size.size(); i++) {
      NtsVar d;
      X.push_back(d);
      Eo.push_back(d);
    }
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
    if (layer == 0) {
      y = torch::relu(P[2*layer+1]->forward(a)).set_requires_grad(true);
    } else if (layer == 1) {
      y = P[2*layer+1]->forward(a);
      y = y.log_softmax(1);
    }
    cp->op_push(a, y, nts::autodiff::NNOP);
    return y;
  }
  NtsVar EdgeForward(NtsVar &ei) {
    NtsVar y;
    int layer = graph->rtminfo->curr_layer;
    y = torch::relu(P[2*layer]->forward(ei).set_requires_grad(true));
    //y=ei*2;
    cp->op_push(ei, y, nts::autodiff::NNOP);
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
      P[i]->all_reduce_to_gradient(P[i]->W.grad().cpu());
      P[i]->learnC2C_with_decay_Adam();
      P[i]->next();
    }
  }
  void Forward() {
    graph->rtminfo->forward = true;
    for (int i = 0; i < graph->gnnctx->layer_size.size() - 1; i++) {
      graph->rtminfo->curr_layer = i;     
        gt->LocalScatter(X[i],Ei[i],subgraphs);
        cp->op_push(X[i], Ei[i], nts::autodiff::SINGLE_CPU_EDGE_SCATTER);
        Eo[i]=EdgeForward(Ei[i]);   
        //Eo[i]=torch::ones_like(Ei[i]);
        gt->LocalAggregate(Eo[i],Y[i],subgraphs);
        cp->op_push(Eo[i], Y[i], nts::autodiff::SINGLE_CPU_EDGE_GATHER);
        X[i + 1] = vertexForward(Y[i], X[i]);
    }
  }

  void run() {
    if (graph->partition_id == 0)
      printf("GNNmini::[Dist.GPU.GCNimpl] running [%d] Epochs\n", iterations);
    exec_time -= get_time();
    for (int i_i = 0; i_i < iterations; i_i++) {
      graph->rtminfo->epoch = i_i;
      if (i_i != 0) {
        for (int i = 0; i < P.size(); i++) {
          P[i]->zero_grad();
        }
        for (int i = 0; i < graph->gnnctx->layer_size.size() - 1; i++){
          Y[i].grad().zero_();
          Ei[i].grad().zero_();
        }
      }
      Forward();
      Test(0);
      Test(1);
      Test(2);
      Loss();
      cp->self_backward();
      Update();
//     cp->debug();
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
//            Eo[i]=torch::ones_like(Ei[i]);
//         gt->LocalAggregate(Eo[i],Y[i],subgraphs);
//      std::cout<<Y[i].t()[0]<<std::endl;  
//      for(int i=2680;i<2708;i++){
//          printf("dgr: %d \n",subgraphs[0]->column_offset[i+1]-subgraphs[0]->column_offset[i]);
//      }
         
//      gt->LocalScatter(X[i],Ei[i],subgraphs);
//      std::cout<<Ei[i].t()[0].slice(0,13540,13566,1).t()<<std::endl;
//      for(int i=2700;i<2708;i++){
//          printf("dst: %d \t",i);
//          for(int j=subgraphs[0]->column_offset[i];j<subgraphs[0]->column_offset[i+1];j++){
//              printf("src:%d ",subgraphs[0]->row_indices[j]);
//            }
//          printf("\n");
//      }
      
  }
};

#include "core/neutronstar.hpp"
class GAT_CPU_DIST_OPTM_impl {
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
  //std::vector<CSC_segment_pinned *> subgraphs;
  // NN
  GNNDatum *gnndatum;
  NtsVar L_GT_C;
  NtsVar L_GT_G;
  NtsVar MASK;
  //GraphOperation *gt;
  PartitionedGraph *partitioned_graph;
  // Variables
  std::vector<Parameter *> P;
  std::vector<Parameter *> al;
  std::vector<Parameter *> ar;
  std::vector<NtsVar> X;
  nts::ctx::NtsContext* ctx;
  
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

  GAT_CPU_DIST_OPTM_impl(Graph<Empty> *graph_, int iterations_,
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
    
    partitioned_graph=new PartitionedGraph(graph, active);
    partitioned_graph->GenerateAll([&](VertexId src, VertexId dst) {
      return nts::op::nts_norm_degree(graph,src, dst);
    },CPU_T,true);
    graph->init_communicatior();
    //cp = new nts::autodiff::ComputionPath(gt, subgraphs);
    ctx=new nts::ctx::NtsContext();
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
    torch::manual_seed(0);
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

    // initializeing parameter. Creating tensor with shape [layer_size[i],
    // layer_size[i + 1]]
    for (int i = 0; i < graph->gnnctx->layer_size.size() - 1; i++) {
      P.push_back(new Parameter(graph->gnnctx->layer_size[i],
                                graph->gnnctx->layer_size[i + 1], alpha, beta1,
                                beta2, epsilon, weight_decay));
      al.push_back(new Parameter(graph->gnnctx->layer_size[i + 1], 1, alpha,
                                beta1, beta2, epsilon, weight_decay));
      ar.push_back(new Parameter(graph->gnnctx->layer_size[i + 1], 1, alpha,
                                beta1, beta2, epsilon, weight_decay));
    }

    // synchronize parameter with other processes
    // because we need to guarantee all of workers are using the same model
    for (int i = 0; i < P.size(); i++) {
      P[i]->init_parameter();
      P[i]->set_decay(decay_rate, decay_epoch);
      al[i]->init_parameter();
      al[i]->set_decay(decay_rate, decay_epoch);
      ar[i]->init_parameter();
      ar[i]->set_decay(decay_rate, decay_epoch);
    }

    F = graph->Nts->NewLeafTensor(
        gnndatum->local_feature,
        {graph->gnnctx->l_v_num, graph->gnnctx->layer_size[0]},
        torch::DeviceType::CPU);

      NtsVar d;
      X.resize(graph->gnnctx->layer_size.size(),d);
    // X[0] is the initial vertex representation. We created it from
    // local_feature
    X[0] = F;
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
  void Loss() {
    //  return torch::nll_loss(a,L_GT_C);
    torch::Tensor a = X[graph->gnnctx->layer_size.size() - 1].log_softmax(1);
    torch::Tensor mask_train = MASK.eq(0);
    loss = torch::nll_loss(
        a.masked_select(mask_train.expand({mask_train.size(0), a.size(1)}))
            .view({-1, a.size(1)}),
        L_GT_C.masked_select(mask_train.view({mask_train.size(0)})));
    ctx->appendNNOp(X[graph->gnnctx->layer_size.size() - 1], loss);
  }

  void Update() {
    for (int i = 0; i < P.size(); i++) {
      // accumulate the gradient using all_reduce
      P[i]->all_reduce_to_gradient(P[i]->W.grad().cpu());
      // update parameters with Adam optimizer
      P[i]->learnC2C_with_decay_Adam();
      P[i]->next();
      
      al[i]->all_reduce_to_gradient(al[i]->W.grad().cpu());
      // update parameters with Adam optimizer
      al[i]->learnC2C_with_decay_Adam();
      al[i]->next();
      
      ar[i]->all_reduce_to_gradient(ar[i]->W.grad().cpu());
      // update parameters with Adam optimizer
      ar[i]->learnC2C_with_decay_Adam();
      ar[i]->next();
    }
  }
  void Forward() {
    graph->rtminfo->forward = true;
    for (int i = 0; i < graph->gnnctx->layer_size.size() - 1; i++) {
      graph->rtminfo->curr_layer = i;
      NtsVar X_trans=ctx->runVertexForward([&](NtsVar x_i_){
        int layer = graph->rtminfo->curr_layer;
        return P[layer]->forward(x_i_);
      },
        X[i]);
      
      NtsVar mirror= ctx->runGraphOp<nts::op::DistGetDepNbrOp>(partitioned_graph,active,X_trans);
      //NtsVar edge_src= ctx->runGraphOp<nts::op::DistScatterSrc>(partitioned_graph,active,mirror);
      NtsVar src_att=ctx->runVertexForward([&](NtsVar x_i_){
        int layer = graph->rtminfo->curr_layer;
        return al[layer]->forward(x_i_);
      },
        mirror);
      NtsVar dst_att=ctx->runVertexForward([&](NtsVar x_i_){
        int layer = graph->rtminfo->curr_layer;
        return ar[layer]->forward(x_i_);
      },
        X_trans);
//      LOG_INFO("dst_att DATA_PTR %ld",(long)dst_att.data_ptr());
      NtsVar e_src_att= ctx->runGraphOp<nts::op::DistScatterSrc>(partitioned_graph,active,src_att);
      NtsVar e_dst_att= ctx->runGraphOp<nts::op::DistScatterDst>(partitioned_graph,active,dst_att);
      NtsVar e_msg=torch::cat({e_src_att,e_dst_att},1);
      
      NtsVar m=ctx->runEdgeForward([&](NtsVar e_msg_){
            int layer = graph->rtminfo->curr_layer;
            NtsVar s=e_msg_.sum(1).unsqueeze(-1);
            return torch::leaky_relu(s,0.2);
        },
      e_msg);//edge NN
            
      NtsVar a=ctx->runGraphOp<nts::op::DistEdgeSoftMax>(partitioned_graph,
              active,m);// edge NN   
     NtsVar nbr= ctx->runGraphOp<nts::op::DistAggregateDstFuseWeight>(partitioned_graph,active,mirror,a); 
     X[i+1]=ctx->runVertexForward([&](NtsVar nbr_){
            return torch::relu(nbr_);
        },nbr);
  
    }
  }

  void run() {
    if (graph->partition_id == 0) {
      LOG_INFO("GNNmini::[Dist.GPU.GCNimpl] running [%d] Epoches\n",
               iterations);
    }

    exec_time -= get_time();
    for (int i_i = 0; i_i < iterations; i_i++) {
      graph->rtminfo->epoch = i_i;
      if (i_i != 0) {
        for (int i = 0; i < P.size(); i++) {
          P[i]->zero_grad();
          al[i]->zero_grad();
          ar[i]->zero_grad();
          
        }
      }
      Forward();
      
      
  //      printf("sizeof %d",sizeof(__m256i));
//      printf("sizeof %d",sizeof(int));
      Test(0);
      Test(1);
      Test(2);
      Loss();
      
      ctx->self_backward(true);
//      LOG_INFO("FINISH BACKWARD");
      Update();
      // ctx->debug();
      if (graph->partition_id == 0)
        std::cout << "Nts::Running.Epoch[" << i_i << "]:loss\t" << loss
                  << std::endl;
    }


    delete active;
  }

};

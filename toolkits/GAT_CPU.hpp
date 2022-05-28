#include "core/neutronstar.hpp"

class GAT_CPU_impl {
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
  //std::vector<CSC_segment_pinned *> subgraphs;
  // NN
  GNNDatum *gnndatum;
  NtsVar L_GT_C;
  NtsVar L_GT_G;
  NtsVar MASK;
  std::map<std::string, NtsVar> I_data;
  //GraphOperation *gt;
  PartitionedGraph * partitioned_graph;
  // Variables
  std::vector<Parameter *> P;
  std::vector<NtsVar> X;
  //nts::autodiff::ComputionPath *cp;
  nts::ctx::NtsContext *ctx;
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

  GAT_CPU_impl(Graph<Empty> *graph_, int iterations_,
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
//    graph->generate_COO();
//    graph->reorder_COO_W2W();
    // generate_CSC_Segment_Tensor_pinned(graph, csc_segment, true);
    
//    gt = new GraphOperation(graph, active);
//    gt->GenerateGraphSegment(subgraphs, CPU_T, [&](VertexId src, VertexId dst) {
//      return gt->norm_degree(src, dst);
//    });
//    // gt->GenerateMessageBitmap(subgraphs);
//    gt->GenerateMessageBitmap_multisokects(subgraphs);
    partitioned_graph=new PartitionedGraph(graph, active);
    partitioned_graph->GenerateAll([&](VertexId src, VertexId dst) {
      return 1;
    },CPU_T);
    
    graph->init_communicatior();
    //cp = new nts::autodiff::ComputionPath(gt, subgraphs, true);
    ctx= new nts::ctx::NtsContext();
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
    gnndatum->registLabel(L_GT_C);
    gnndatum->registMask(MASK);

    for (int i = 0; i < graph->gnnctx->layer_size.size() - 1; i++) {
      P.push_back(new Parameter(graph->gnnctx->layer_size[i],
                                graph->gnnctx->layer_size[i + 1], alpha, beta1,
                                beta2, epsilon, weight_decay));
      P.push_back(new Parameter(graph->gnnctx->layer_size[i + 1] * 2, 1, alpha,
                                beta1, beta2, epsilon, weight_decay));
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

    NtsVar d;
    X.resize(graph->gnnctx->layer_size.size(),d);
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
      P[i]->all_reduce_to_gradient(P[i]->W.grad().cpu());
      P[i]->learnC2C_with_decay_Adam();
      P[i]->next();
    }
  }
  NtsVar preForward(NtsVar &x) {
    NtsVar y;
    int layer = graph->rtminfo->curr_layer;
    y = P[2 * layer]->forward(x).set_requires_grad(true);
    return y;
  }
  NtsVar vertexForward(NtsVar &a, NtsVar &x) {
    return torch::relu(a);
  }

  void Forward() {
    graph->rtminfo->forward = true;
    for (int i = 0; i < graph->gnnctx->layer_size.size() - 1; i++) {
      graph->rtminfo->curr_layer = i;
      NtsVar X_trans=ctx->runVertexForward([&](NtsVar x_i){
            return preForward(x_i);},
        X[i]);//pre apply    
      NtsVar E_msg=ctx->runGraphOp<nts::op::SingleCPUSrcDstScatterOp>(
              partitioned_graph,active,X_trans);// scatterto edge
      
      NtsVar m=ctx->runEdgeForward([&](NtsVar e_msg){
            int layer = graph->rtminfo->curr_layer;
            return torch::leaky_relu(P[2 * layer + 1]->forward(e_msg),0.2);
        },
      E_msg);//edge NN
        
      NtsVar a=ctx->runGraphOp<nts::op::SingleEdgeSoftMax>(partitioned_graph,
              active,m);// edge NN   
      
      NtsVar E_msg_out=ctx->runEdgeForward([&](NtsVar a){
            return E_msg.slice(1, 0, E_msg.size(1) / 2, 1)*a;
        },
      a);//Edge NN 
        
      NtsVar nbr=ctx->runGraphOp<nts::op::SingleCPUDstAggregateOp>(
              partitioned_graph,active,E_msg_out);//agg  
      
      X[i+1]=ctx->runVertexForward([&](NtsVar nbr){
            return torch::relu(nbr);
        },nbr);
    //        printf("hellow\n");
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
      }
      Forward();
      Test(0);
      Test(1);
      Test(2);
      Loss();
//      ctx->debug();
//       ctx->reset();
      ctx->self_backward(true);
      Update();
      //     cp->debug();
      if (graph->partition_id == 0)
        std::cout << "Nts::Running.Epoch[" << i_i << "]:loss\t" << loss
                  << std::endl;
    }
//        NtsVar s=torch::rand({3,1});
//        s[0][0]=1;
//        s[1][0]=2;
//        s[2][0]=3;
//        NtsVar s1=torch::rand({3,1});
//        s1[0][0]=2;
//        s1[1][0]=3;
//        s1[2][0]=4;
//        s.set_requires_grad(true);
//        s1.set_requires_grad(true);
//        NtsVar s2=s.softmax(0);
//        NtsVar s3=s2*s1;
//        std::cout<<"s :\n"<<s<<std::endl;
//        std::cout<<"s1:\n"<<s1<<std::endl;
//        std::cout<<"s2:\n"<<s2<<std::endl;
//        std::cout<<"s3:\n"<<s3<<std::endl;
//        s3.backward(torch::ones_like(s3));
//        std::cout<<"s.grad():\n"<<s.grad()<<std::endl;
//        NtsVar s4=torch::zeros({3,1});
//        s4=(s2*s1)-(s2)*(s2.t().mm(s1));        
//    NtsVar s4=torch::rand({3,3});
//        std::cout<<"s4.grad():\n"<<s4<<"\n"<<s4.sum(-1)<<std::endl;
    exec_time += get_time();

    //    nts::OP::ntsOps  *nop=new nts::OP::ntsOps(graph,active);
    //    nop->segmentReduce(X[0],subgraphs,0);
    // nop.

    delete active;
  }
  
};

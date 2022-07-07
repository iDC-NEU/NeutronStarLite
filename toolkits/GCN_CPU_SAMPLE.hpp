#include "core/neutronstar.hpp"

class GCN_CPU_SAMPLE_impl {
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
  std::vector<NtsVar> X;
  nts::ctx::NtsContext* ctx;
  // Sampler* train_sampler;
  // Sampler* val_sampler;
  // Sampler* test_sampler;
  FullyRepGraph* fully_rep_graph;
  
  NtsVar F;
  NtsVar loss;
  NtsVar tt;
  float acc;
  int batch;
  long correct;
  torch::nn::Dropout drpmodel;

  GCN_CPU_SAMPLE_impl(Graph<Empty> *graph_, int iterations_,
               bool process_local = false, bool process_overlap = false) {
    graph = graph_;
    iterations = iterations_;

    active = graph->alloc_vertex_subset();
    active->fill();

    graph->init_gnnctx(graph->config->layer_string);
    graph->init_gnnctx_fanout(graph->config->fanout_string);
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
    
      fully_rep_graph=new FullyRepGraph(graph);
      fully_rep_graph->GenerateAll();
      fully_rep_graph->SyncAndLog("read_finish");
      // sampler=new Sampler(fully_rep_graph,0,graph->vertices);
      
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
    gnndatum = new GNNDatum(graph->gnnctx, graph);
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

    // X[i] is vertex representation at layer i
    for (int i = 0; i < graph->gnnctx->layer_size.size(); i++) {
      NtsVar d;
      X.push_back(d);
    }
    
    X[0] = F.set_requires_grad(true);
  }

  long getCorrect(NtsVar &input, NtsVar &target) {
    // NtsVar predict = input.log_softmax(1).argmax(1);
    NtsVar predict = input.argmax(1);
    NtsVar output = predict.to(torch::kLong).eq(target).to(torch::kLong);
    return output.sum(0).item<long>();
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
 
  void Loss(NtsVar &left,NtsVar &right) {
    //  return torch::nll_loss(a,L_GT_C);
    torch::Tensor a = left.log_softmax(1);
    NtsVar loss_; 
    loss_= torch::nll_loss(a,right);
    if (ctx->training == true) {
      ctx->appendNNOp(left, loss_);
    }
  }

  void Update() {
    for (int i = 0; i < P.size(); i++) {
      // accumulate the gradient using all_reduce
      P[i]->all_reduce_to_gradient(P[i]->W.grad().cpu());
      // update parameters with Adam optimizer
      P[i]->learnC2C_with_decay_Adam();
      P[i]->next();
    }
  }
  
  void Forward(Sampler* sampler, int type=0) {
    graph->rtminfo->forward = true;
      
    while(sampler->sample_not_finished()){
          sampler->reservoir_sample(graph->gnnctx->layer_size.size()-1,
                                    graph->config->batch_size,
                                    graph->gnnctx->fanout);
    }
      SampledSubgraph *sg;
      // acc=0.0;
      correct = 0;
      batch=0;
      while(sampler->has_rest()){
           sg=sampler->get_one();
           std::vector<NtsVar> X;
           NtsVar d;
           X.resize(graph->gnnctx->layer_size.size(),d);
         
           X[0]=nts::op::get_feature(sg->sampled_sgs[graph->gnnctx->layer_size.size()-2]->src(),F,graph);
           NtsVar target_lab=nts::op::get_label(sg->sampled_sgs[0]->dst(),L_GT_C,graph);
           for(int l=0;l<(graph->gnnctx->layer_size.size()-1);l++){//forward
               
               int hop=(graph->gnnctx->layer_size.size()-2)-l;
               if(l!=0){
                    X[l] = drpmodel(X[l]);
               }
               NtsVar Y_i=ctx->runGraphOp<nts::op::MiniBatchFuseOp>(sg,graph,hop,X[l]);
               X[l + 1]=ctx->runVertexForward([&](NtsVar n_i){
                   if (l==(graph->gnnctx->layer_size.size()-2)) {
                        return P[l]->forward(n_i);
                    }else{
                       return torch::relu(P[l]->forward(n_i));
                    }
                },
                Y_i);
           } 
           Loss(X[graph->gnnctx->layer_size.size()-1],target_lab);
          correct += getCorrect(X[graph->gnnctx->layer_size.size()-1], target_lab);
          if (ctx->training) {
            ctx->self_backward(false);
          }
           Update();
           batch++;
      }

       sampler->clear_queue();
       sampler->restart();
      acc = 1.0 * correct / sampler->work_range[1];
      if (type == 0) {
        LOG_INFO("Train Acc: %f %d %d", acc, correct, sampler->work_range[1]);
      } else if (type == 1) {
        LOG_INFO("Eval Acc: %f %d %d", acc, correct, sampler->work_range[1]);
      } else if (type == 2) {
        LOG_INFO("Test Acc: %f %d %d", acc, correct, sampler->work_range[1]);
      }
  }

  void run() {
    if (graph->partition_id == 0) {
      LOG_INFO("GNNmini::[Dist.GPU.GCNimpl] running [%d] Epoches\n",
               iterations);
    }
    // get train/val/test node index. (may be move this to GNNDatum)
    std::vector<VertexId> train_nids, val_nids, test_nids;
    for (int i = 0; i < graph->gnnctx->l_v_num; ++i) {
      int type = gnndatum->local_mask[i];
      if (type == 0) {
        train_nids.push_back(i);
      } else if (type == 1) {
        val_nids.push_back(i);
      } else if (type == 2) {
        test_nids.push_back(i);
      }
    }
      
    Sampler* train_sampler = new Sampler(fully_rep_graph, train_nids);
    Sampler* eval_sampler = new Sampler(fully_rep_graph, val_nids);
    Sampler* test_sampler = new Sampler(fully_rep_graph, test_nids);

    for (int i_i = 0; i_i < iterations; i_i++) {
      graph->rtminfo->epoch = i_i;
      LOG_INFO("epoch %d", i_i);
      if (i_i != 0) {
        for (int i = 0; i < P.size(); i++) {
          P[i]->zero_grad();
        }
      }
      
      ctx->train();
      Forward(train_sampler, 0);
      
      ctx->eval();
      Forward(eval_sampler, 1);
      Forward(test_sampler, 2);
  //      if (graph->partition_id == 0)
  //        std::cout << "Nts::Running.Epoch[" << i_i << "]:loss\t" << loss
  //                  << std::endl;
    }
    delete active;
  }

};

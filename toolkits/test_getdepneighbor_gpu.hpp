#include "core/neutronstar.hpp"
class test_get_neighbor_gpu {
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

  test_get_neighbor_gpu(Graph<Empty> *graph_, int iterations_,
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
    torch::Tensor a = X[graph->gnnctx->layer_size.size() - 1];
    torch::Tensor mask_train = MASK.eq(0);
    loss = torch::nll_loss(
        a.masked_select(mask_train.expand({mask_train.size(0), a.size(1)}))
            .view({-1, a.size(1)}),
        L_GT_C.masked_select(mask_train.view({mask_train.size(0)})));
    ctx->appendNNOp(a, loss);
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
  void test_mirror() {
    graph->rtminfo->forward = true;
    for (int i = 0; i < graph->gnnctx->layer_size.size() - 1; i++) {
      graph->rtminfo->curr_layer = i;
      if(i==0){
//          int testid=2707;
        for(VertexId val=graph->partition_offset[graph->partition_id];val<graph->partition_offset[graph->partition_id+1];val++)
            X[i][val-graph->partition_offset[graph->partition_id]][10]=(float)(val);
//         LOG_INFO("X_i[0][10](%f, %f)",X[i][testid][0]);
//test mirror        
        NtsVar mirror= ctx->runGraphOp<nts::op::DistGetDepNbrOp>(partitioned_graph,active,X[i]);
        NtsVar x_trans=X[i].cuda();
        NtsVar mirror_gpu= ctx->runGraphOp<nts::op::DistGPUGetDepNbrOp>(partitioned_graph,active,x_trans);
        //NtsVar test_mirror_forward=torch::cat({mirror.slice(1,10,11,1),mirror_gpu.cpu().slice(1,10,11,1)},1);
       // NtsVar mirror.slice(1,10,11,1).eq(mirror_gpu.cpu().slice(1,10,11,1)).sum(0);
        partitioned_graph->SyncAndLog("sync test DistGPUGetDepNbrOp forward: 1 is passed");
        std::cout<<mirror.slice(1,10,11,1).eq(mirror_gpu.cpu().slice(1,10,11,1)).sum(0).eq(mirror.size(0))<<std::endl;     
        NtsVar x_back_cuda=ctx->ntsOp.top().op->backward(mirror_gpu);
                ctx->pop_one_op();
        NtsVar x_back=ctx->ntsOp.top().op->backward(mirror);
        partitioned_graph->SyncAndLog("sync test DistGPUGetDepNbrOp backward: 1 is passed");
        std::cout<<x_back.slice(1,10,11,1).eq(x_back_cuda.cpu().slice(1,10,11,1)).sum(0).eq(x_back.size(0))<<std::endl; 
      }        
    }
  }
  void test_scatter_src() {
    graph->rtminfo->forward = true;
    for (int i = 0; i < graph->gnnctx->layer_size.size() - 1; i++) {
      graph->rtminfo->curr_layer = i;
      if(i==0){
//          int testid=2707;
        for(VertexId val=graph->partition_offset[graph->partition_id];val<graph->partition_offset[graph->partition_id+1];val++)
            X[i][val-graph->partition_offset[graph->partition_id]][10]=(float)(val);       
        NtsVar mirror= ctx->runGraphOp<nts::op::DistGetDepNbrOp>(partitioned_graph,active,X[i]);       
//test DistScatterSrc        
        NtsVar mirror_gpu=mirror.cuda();
        NtsVar edge_src=ctx->runGraphOp<nts::op::DistScatterSrc>(partitioned_graph,active,mirror);
        NtsVar edge_src_gpu= ctx->runGraphOp<nts::op::DistGPUScatterSrc>(partitioned_graph,active,mirror_gpu);
        partitioned_graph->SyncAndLog("sync test DistGPUScatterSrc forward: 1 is passed");
        std::cout<<edge_src.slice(1,10,11,1).eq(edge_src_gpu.cpu().slice(1,10,11,1)).sum(0).eq(edge_src.size(0))<<std::endl;
        NtsVar x_back_cuda=ctx->ntsOp.top().op->backward(edge_src_gpu);
                ctx->pop_one_op();
        NtsVar x_back=ctx->ntsOp.top().op->backward(edge_src);
        partitioned_graph->SyncAndLog("sync test DistGPUScatterSrc backward: 1 is passed");
        std::cout<<x_back.slice(1,10,11,1).eq(x_back_cuda.cpu().slice(1,10,11,1)).sum(0).eq(x_back.size(0))<<std::endl; 
      }        
    }
  }
  void test_scatter_dst() {
    graph->rtminfo->forward = true;
    for (int i = 0; i < graph->gnnctx->layer_size.size() - 1; i++) {
      graph->rtminfo->curr_layer = i;
      if(i==0){
//          int testid=2707;
        for(VertexId val=graph->partition_offset[graph->partition_id];val<graph->partition_offset[graph->partition_id+1];val++)
            for(VertexId j=0;j<X[i].size(1);j++)
            X[i][val-graph->partition_offset[graph->partition_id]][j]=(float)(val);             
        
        partitioned_graph->SyncAndLog("sync");
//test DistScatterDst      
       NtsVar X_i_gpu=X[i].cuda();
        
        NtsVar edge_dst= ctx->runGraphOp<nts::op::DistScatterDst>(partitioned_graph,active,X[i]);
        NtsVar edge_dst_gpu= ctx->runGraphOp<nts::op::DistGPUScatterDst>(partitioned_graph,active,X_i_gpu);
        partitioned_graph->SyncAndLog("sync test DistGPUScatterDst forward: 1 is passed");
        std::cout<<edge_dst.eq(edge_dst_gpu.cpu()).sum(0).sum(0).eq(edge_dst.size(0)*edge_dst.size(1))<<std::endl;
        
        NtsVar x_back_cuda=ctx->ntsOp.top().op->backward(edge_dst_gpu);
                ctx->pop_one_op();
        NtsVar x_back=ctx->ntsOp.top().op->backward(edge_dst);
         partitioned_graph->SyncAndLog("sync test DistGPUScatterDst backward: 1 is passed");
        std::cout<<x_back.eq(x_back_cuda.cpu()).sum(0).sum(0).eq(x_back.size(0)*x_back.size(1))<<std::endl; 

      }        
    }
  }
  void test_gather_dst() {
    graph->rtminfo->forward = true;
    for (int i = 0; i < graph->gnnctx->layer_size.size() - 1; i++) {
      graph->rtminfo->curr_layer = i;
      if(i==0){
//          int testid=2707;
        for(VertexId val=graph->partition_offset[graph->partition_id];val<graph->partition_offset[graph->partition_id+1];val++)
            for(VertexId j=0;j<X[i].size(1);j++)
            X[i][val-graph->partition_offset[graph->partition_id]][j]=(float)(val);             
        
//test DistScatterDst      
       NtsVar X_i_gpu=X[i].cuda();
        
        NtsVar edge_dst= ctx->runGraphOp<nts::op::DistScatterDst>(partitioned_graph,active,X[i]);
        NtsVar edge_dst_gpu= ctx->runGraphOp<nts::op::DistGPUScatterDst>(partitioned_graph,active,X_i_gpu);
        
        NtsVar dst_y=  ctx->runGraphOp<nts::op::DistAggregateDst>(partitioned_graph,active,edge_dst);
        NtsVar dst_y_gpu=  ctx->runGraphOp<nts::op::DistGPUAggregateDst>(partitioned_graph,active,edge_dst_gpu);
        
        partitioned_graph->SyncAndLog("sync test DistGPUAggregateDst forward: 1 is passed");
        std::cout<<dst_y.eq(dst_y_gpu.cpu()).sum(0).sum(0).eq(dst_y.size(0)*dst_y.size(1))<<std::endl;
        
        NtsVar x_back_cuda=ctx->ntsOp.top().op->backward(dst_y_gpu);
                ctx->pop_one_op();
        NtsVar x_back=ctx->ntsOp.top().op->backward(dst_y);
        partitioned_graph->SyncAndLog("sync test DistGPUAggregateDst backward: 1 is passed");
        std::cout<<x_back.eq(x_back_cuda.cpu()).sum(0).sum(0).eq(x_back.size(0)*x_back.size(1))<<std::endl; 

      }        
    }
  }
  
  void test_softmax() {
    graph->rtminfo->forward = true;
    for (int i = 0; i < graph->gnnctx->layer_size.size() - 1; i++) {
      graph->rtminfo->curr_layer = i;
      if(i==0){
//          int testid=2707;
        for(VertexId val=graph->partition_offset[graph->partition_id];val<graph->partition_offset[graph->partition_id+1];val++)
            X[i][val-graph->partition_offset[graph->partition_id]][10]=(float)(val);             
        
//test DistScatterDst      
       NtsVar X_i_gpu=X[i].cuda();
        
        NtsVar edge_dst= ctx->runGraphOp<nts::op::DistScatterDst>(partitioned_graph,active,X[i]);
   
        
        NtsVar m=torch::ones({edge_dst.size(0),1});
        for(VertexId val=0;val< m.size(0);val++)
            m[val][0]=(float)(1);
        NtsVar m_gpu=m.cuda();
        NtsVar a=ctx->runGraphOp<nts::op::DistEdgeSoftMax>(partitioned_graph,active,m);
        NtsVar a_gpu=ctx->runGraphOp<nts::op::DistGPUEdgeSoftMax>(partitioned_graph,active,m_gpu);
        NtsVar test_mirror_forward=torch::cat({a,a_gpu.cpu()},1);
     //   std::cout<<test_mirror_forward.slice(0,1119,2200,1)<<std::endl;
        partitioned_graph->SyncAndLog("sync test DistGPUEdgeSoftMax forward: 1 is passed");
        std::cout<<torch::abs(a-a_gpu.cpu()).le(0.0000001).sum(0).eq(a.size(0))<<std::endl;
        //std::cout<<a.size(0)<<" "<<a.eq(a_gpu.cpu()).sum(0)<<std::endl;
        
        NtsVar x_back_cuda=ctx->ntsOp.top().op->backward(m_gpu);
                ctx->pop_one_op();
        NtsVar x_back=ctx->ntsOp.top().op->backward(m);
         partitioned_graph->SyncAndLog("sync test DistGPUEdgeSoftMax backward: 1 is passed");
       // std::cout<<(x_back-x_back_cuda.cpu()).slice(0,0,6,1)<<std::endl;
        std::cout<<torch::abs(x_back-x_back_cuda.cpu()).le(0.0000001).sum(0).eq(x_back.size(0))<<std::endl;     
      }        

    }
  }
  void run() {
    if (graph->partition_id == 0) {
      LOG_INFO("GNNmini::[Dist.GPU.GCNimpl] running [%d] Epoches\n",
               iterations);
    }

    exec_time -= get_time();
    for (int i_i = 0; i_i < 1; i_i++) {
      graph->rtminfo->epoch = i_i;
      //test_mirror();
      //test_scatter_src();
      //test_scatter_dst();
      test_gather_dst();
      //test_softmax();
     
      if (graph->partition_id == 0)
        std::cout << "Nts::Running.Epoch[" << i_i << "]:loss\t" << loss
                  << std::endl;
    }
 
//    NtsVar s=torch::ones({3,3});
//    NtsVar d=torch::zeros({3,3});
//    NtsVar l= test(d);
//    printf("%ld %ld\n", &l, &d);

    delete active;
  }

};

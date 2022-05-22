#include "core/neutronstar.hpp"
class test_get_neighbor {
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

  test_get_neighbor(Graph<Empty> *graph_, int iterations_,
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
//    // generate the representation for subgraph corresponding to the way we
//    // partitioned e.g. generate CSC/CSR format representation for every
//    // subgraph
//    gt->GenerateGraphSegment(subgraphs, CPU_T, [&](VertexId src, VertexId dst) {
//      return gt->norm_degree(src, dst);
//    });
//    // gt->GenerateMessageBitmap(subgraphs);
//    // pre-process the data that will be used while doing forward and backward
//    // propagation which has better support on multisockets.
//    gt->GenerateMessageBitmap_multisokects(subgraphs);
    
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
 //   ctx->op_push(a, y, nts::ctx::NNOP);
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
  void Forward() {
    graph->rtminfo->forward = true;
    for (int i = 0; i < graph->gnnctx->layer_size.size() - 1; i++) {
      graph->rtminfo->curr_layer = i;
      if(i==0){
//          int testid=2707;
        for(VertexId val=graph->partition_offset[graph->partition_id];val<graph->partition_offset[graph->partition_id+1];val++)
            X[i][val-graph->partition_offset[graph->partition_id]][0]=(float)1;
//         LOG_INFO("X_i[0][10](%f, %f)",X[i][testid][0]);

//test mirror        
        NtsVar mirror= ctx->runGraphOp<nts::op::DistGetDepNbrOp>(partitioned_graph,active,X[i]);
//        NtsVar mirror_s=torch::ones_like(mirror);
//        NtsVar X_i= ctx->ntsOp.top().op->backward(mirror_s);
//        ctx->pop_one_op();
//        NtsVar X_i= ctx->ntsOp.top().op->backward(X_i);
                
//        for(int i=0;i<partitioned_graph->owned_vertices;i++){
//            std::cout<<X_i[i][0]<<std::endl;
//        }
        
//test DistScatterSrc        
        NtsVar edge_src= ctx->runGraphOp<nts::op::DistScatterSrc>(partitioned_graph,active,mirror);
//        NtsVar edge_src_i=torch::ones_like(edge_src);
//        NtsVar mirror_s= ctx->ntsOp.top().op->backward(edge_src_i);
//         for(int i=0;i<partitioned_graph->global_vertices;i++){
//            if((partitioned_graph->MirrorIndex[i+1]-partitioned_graph->MirrorIndex[i])>0){
//                VertexId tmp_s=partitioned_graph->MirrorIndex[i];
//                std::cout<<i<<" "<<partitioned_graph->compressed_row_offset[tmp_s+1]-partitioned_graph->compressed_row_offset[tmp_s]<<" "<<mirror_s[tmp_s][0]<<std::endl;
//            }
//        }       
//        std::cout<<edge.slice(1,0,1,1);
//        for(int i=0;i<partitioned_graph->owned_edges;i++){
//           std::cout<<partitioned_graph->row_indices[i]<<" "<<edge_src[i][0]<<std::endl;
//        }

//test  DistScatterSrc and mirror        
//        NtsVar edge_src_i=torch::ones_like(edge_src);
//        NtsVar mirror_s= ctx->ntsOp.top().op->backward(edge_src_i);
//        ctx->pop_one_op();
//        NtsVar X_s=ctx->ntsOp.top().op->backward(mirror_s);
//        //LOG_INFO("X_s %d %d",X_s.size(0),X_s.size(1));
//        for(int i=0;i<partitioned_graph->owned_vertices;i++){
//              std::cout<<graph->out_degree_for_backward[i+graph->gnnctx->p_v_s]<<" "<<X_s[i][0]<<std::endl;
//        }         
        
        
//test DistScatterDst             
//        NtsVar edge_dst= ctx->runGraphOp<nts::op::DistScatterDst>(partitioned_graph,active,X[i]);
//        for(int i=0;i<partitioned_graph->owned_vertices;i++){
//           for(int j=partitioned_graph->column_offset[i];
//                   j<partitioned_graph->column_offset[i+1];j++){
//              std::cout<<i+graph->gnnctx->p_v_s<<" "<<edge_dst[j][0]<<std::endl;
//           }
//        }        
//        LOG_INFO("y_i size(%d %d)",Y_i.size(0),Y_i.size(1));
//        if(graph->partition_id==1)
//        for(VertexId i=0;i<graph->vertices;i++){
//            if((partitioned_graph->MirrorIndex[i+1]-partitioned_graph->MirrorIndex[i])>0)
//            std::cout<<i<<" "<<Y_i[partitioned_graph->MirrorIndex[i]][0]<<std::endl;
//        }
//        NtsVar X_i=ctx->ntsOp.top().op->backward(Y_i);
 //       if(graph->partition_id==0){
 //       for(VertexId i=0;i<X_i.size(0);i++){
           // if((partitioned_graph->MirrorIndex[i+1]-partitioned_graph->MirrorIndex[i])>0)
//        if(graph->partition_id==0){
//            int i=1020;
//            std::cout<<i+graph->partition_offset[graph->partition_id]<<"t "<<X_i[i][0]<<std::endl;
 //       }
 //       }
        
//test Aggregate
    NtsVar Y= ctx->runGraphOp<nts::op::DistAggregateDst>(partitioned_graph,active,edge_src); 
//        for(VertexId i=0;i<graph->owned_vertices;i++){
//            std::cout<<graph->in_degree_for_backward[i+graph->gnnctx->p_v_s]<<" "<<Y[i][0]<<std::endl;
//        }
//    NtsVar edge_src_backward= ctx->ntsOp.top().op->backward(Y);
//    if(graph->partition_id==0){
//        for(int i=0;i<partitioned_graph->owned_vertices;i++){
//           for(int j=partitioned_graph->column_offset[i];
//                   j<partitioned_graph->column_offset[i+1];j++){
//              std::cout<<graph->in_degree_for_backward[i+graph->gnnctx->p_v_s]<<" "<<edge_src_backward[j][0]<<std::endl;
//           }
//        }
//    }  
      }        
//       NtsVar Y_i= ctx->runGraphOp<nts::op::ForwardCPUfuseOp>(partitioned_graph,active,X[i]);      
//        X[i + 1]=ctx->runVertexForward([&](NtsVar n_i,NtsVar v_i){
//            return vertexForward(n_i, v_i);
//        },
//        Y_i,
//        X[i]);
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
        }
      }
      
      Forward();
      
  //      printf("sizeof %d",sizeof(__m256i));
//      printf("sizeof %d",sizeof(int));
//      Test(0);
//      Test(1);
//      Test(2);
//      Loss();
      //ctx->self_backward();
      //Update();
//       ctx->debug();
      if (graph->partition_id == 0)
        std::cout << "Nts::Running.Epoch[" << i_i << "]:loss\t" << loss
                  << std::endl;
    }
//      int length=68;
//      float* s=new float[length];
//      float* d=new float[length];
//      for (int i=0;i<length;i++){
//          s[i]=(float)i;
//          d[i]=i;
//      } 
//      float weight=2.0;
//      nts::op::nts_comp(d,s,weight,length);
//       for (int i=0;i<length;i++){
//           printf("%f\t",d[i]);
//      }printf("\n");
//    exec_time += get_time();

    delete active;
  }

};

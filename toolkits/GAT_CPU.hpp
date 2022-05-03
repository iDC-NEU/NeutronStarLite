
//#include "comm/logger.h"
//#include "core/AutoDiff.hpp"
//#include "core/gnnmini.hpp"
#include "core/gnnmini.h"
#include "core/NtsEdgeTensor.hpp"

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
  std::vector<NtsVar> E_tmp;
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
    cp = new nts::autodiff::ComputionPath(gt, subgraphs,true);
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
                                graph->gnnctx->layer_size[i+1], alpha, beta1,
                                beta2, epsilon, weight_decay));
      P.push_back(new Parameter(graph->gnnctx->layer_size[i+1]*2,
                                1, alpha, beta1,
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
          {graph->gnnctx->l_e_num, 2*graph->gnnctx->layer_size[i+1]},
          torch::DeviceType::CPU));
      Y.push_back(graph->Nts->NewKeyTensor(
          {graph->gnnctx->l_v_num, graph->gnnctx->layer_size[i]},
          torch::DeviceType::CPU));
    }
    NtsVar d;
    for (int i = 0; i < graph->gnnctx->layer_size.size(); i++) {
      X.push_back(d);
      Eo.push_back(d);
    }
    E_tmp.resize(graph->gnnctx->l_v_num,d);
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
  NtsVar preForward(NtsVar &x) {
    NtsVar y;
    int layer = graph->rtminfo->curr_layer;
      y =P[2*layer]->forward(x).set_requires_grad(true);
    cp->op_push(x, y, nts::autodiff::NNOP);
    return y;
  }
  NtsVar vertexForward(NtsVar &a, NtsVar &x) {
    NtsVar y;
    y=a;
    return y;
  }
  NtsVar EdgeForward(NtsVar &ei) {
    NtsVar y=graph->Nts->NewLeafTensor({graph->gnnctx->l_v_num,ei.size(1)/2},torch::DeviceType::CPU);
    nts::ntsVertexTensor y_vtx(ei.size(1)/2,subgraphs[0],graph->Nts);
    int layer = graph->rtminfo->curr_layer;
    NtsVar m = torch::exp(P[2*layer+1]->forward(ei));
    NtsVar e_src=ei.slice(1,0,ei.size(1)/2,1);
//    NtsVar dst_edge=torch::from_blob(subgraphs[0]->destination,{graph->gnnctx->l_e_num,1},torch::kLong);
//    LOG_INFO("%d",m.dim());
    graph->local_vertex_operation<int, ValueType>( // For EACH Vertex
      [&](VertexId vtx, CSC_segment_pinned *subgraph, VertexId recv_id) {     
        long eid_start=subgraph->column_offset[vtx];
        long eid_end=subgraph->column_offset[vtx+1];
        assert(eid_end<=graph->edges);
        assert(eid_start>=0); 
        NtsVar d=m.slice(0,eid_start,eid_end,1).softmax(0);
        y_vtx.getVtxTensor(vtx)=(e_src.slice(0,eid_start,eid_end,1)*d).sum(0);
      },
      subgraphs, m.size(1), active);
      for(VertexId vtx=0;vtx<graph->gnnctx->l_v_num;vtx++){
        y[vtx]=y_vtx.getVtxTensor(vtx);
      }
    //y.backward(torch::ones_like(y));
    cp->op_push(ei, y, nts::autodiff::NNOP);
//      printf("finish\n");
    return y;
  }
  
  void Forward() {
    graph->rtminfo->forward = true;
    for (int i = 0; i < graph->gnnctx->layer_size.size() - 1; i++) {
      graph->rtminfo->curr_layer = i;
        torch::Tensor X_trans= preForward(X[i]);
        gt->LocalScatter(X_trans,Ei[i],subgraphs,true);
        cp->op_push(X_trans, Ei[i], nts::autodiff::SINGLE_CPU_EDGE_SCATTER);
        X[i+1]=EdgeForward(Ei[i]);

    }
//        printf("hellow\n");
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
//          Y[i].grad().zero_();
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
//    NtsVar s=4*graph->Nts->NewOnesTensor({3,1},torch::DeviceType::CPU);
//    NtsVar s1=graph->Nts->NewOnesTensor({10,1},torch::DeviceType::CPU).set_requires_grad(true);
////    NtsVar indice=torch::range(1,3,1,torch::kLong);
////    std::cout<<indice<<std::endl;
//    s[0]=s1[2]*3;
//    s[2]=s1[5]*4;
//    s.backward(torch::ones_like(s));
//        std::cout<<s1.grad()<<std::endl;
    exec_time += get_time();

    delete active;
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
  
//    NtsVar EdgeForward(NtsVar &ei) {
//    NtsVar y;
//    
//    int layer = graph->rtminfo->curr_layer;
//    NtsVar m = torch::exp(P[2*layer+1]->forward(ei));
//    NtsVar a = torch::zeros_like(m);
//    NtsVar e_dst=ei.slice(1,0,ei.size(1)/2,1);
//    NtsVar v_sum=graph->Nts->NewLeafTensor({graph->gnnctx->l_v_num,1},torch::DeviceType::CPU);
//    
//    NtsVar dst_edge=torch::from_blob(subgraphs[0]->destination,{graph->gnnctx->l_e_num,1},torch::kLong);
////    LOG_INFO("%d %d %d %d",m.size(0),m.size(1),e_dst.size(0),e_dst.size(1));
////    NtsVar sum =graph->Nts->NewLeafTensor({graph->gnnctx->l_v_num,1},
////                    torch::DeviceType::CPU);
//    LOG_INFO("%d",m.sum(0).dim());
//    for(VertexId vtx=0;vtx<graph->gnnctx->l_v_num;vtx++){
//            long eid_start=subgraphs[0]->column_offset[vtx];
//            long eid_end=subgraphs[0]->column_offset[vtx+1];
//            NtsVar d;
//            assert(eid_end<=graph->edges);
//            assert(eid_start>=0); 
//            if(vtx==0){
//                LOG_INFO("%d %d",eid_end-eid_start,m.slice(0,eid_start,eid_end,1).size(0));
//            }
//            if(eid_end>eid_start){
//               d=m.slice(0,eid_start,eid_end,1).softmax(0);
//               for(int eid_i=eid_start;eid_i=eid_end;eid_i++){
//                  a[eid_i]=d[eid_i-eid_start];
//               }
//            }        
//            
//    }
//       //  LOG_INFO("%d",a.dim());
//         LOG_INFO("%d %d",e_dst.size(0),e_dst.size(1));
//       //  LOG_INFO("%d",a.dim());
//        y=e_dst*a;
//    cp->op_push(ei, y, nts::autodiff::NNOP);
//    return y;
//  }
};

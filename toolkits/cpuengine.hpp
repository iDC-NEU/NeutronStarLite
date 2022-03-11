#include "core/gnnmini.hpp"

void read_whole_graph(VertexId *column_offset, VertexId *row_indices,
                      int vertices, int edges, std::string path) {
  memset(column_offset, 0, sizeof(VertexId) * (vertices + 1));
  memset(row_indices, 0, sizeof(VertexId) * edges);
  VertexId *tmp_offset = new VertexId[vertices + 1];
  memset(tmp_offset, 0, sizeof(VertexId) * (vertices + 1));
  long total_bytes = file_size(path.c_str());
#ifdef PRINT_DEBUG_MESSAGES
  if (partition_id == 0) {
    printf("|V| = %u, |E| = %lu\n", vertices, edges);
  }
#endif
  int edge_unit_size = 8;
  EdgeId read_edges = edges;
  long bytes_to_read = edge_unit_size * read_edges;
  long read_offset = 0;
  long read_bytes;
  int fin = open(path.c_str(), O_RDONLY);
  EdgeUnit<Empty> *read_edge_buffer = new EdgeUnit<Empty>[CHUNKSIZE];

  assert(lseek(fin, read_offset, SEEK_SET) == read_offset);
  read_bytes = 0;
  while (read_bytes < bytes_to_read) {
    long curr_read_bytes;
    if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE) {
      curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
    } else {
      curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
    }
    assert(curr_read_bytes >= 0);
    read_bytes += curr_read_bytes;
    EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
    // #pragma omp parallel for
    for (EdgeId e_i = 0; e_i < curr_read_edges; e_i++) {
      VertexId src = read_edge_buffer[e_i].src;
      VertexId dst = read_edge_buffer[e_i].dst;
      tmp_offset[dst + 1]++;
    }
  }
  for (int i = 0; i < vertices; i++) {
    tmp_offset[i + 1] += tmp_offset[i];
  }
  memcpy(column_offset, tmp_offset, sizeof(VertexId) * vertices + 1);
  assert(lseek(fin, read_offset, SEEK_SET) == read_offset);
  read_bytes = 0;
  while (read_bytes < bytes_to_read) {
    long curr_read_bytes;
    if (bytes_to_read - read_bytes > edge_unit_size * CHUNKSIZE) {
      curr_read_bytes = read(fin, read_edge_buffer, edge_unit_size * CHUNKSIZE);
    } else {
      curr_read_bytes = read(fin, read_edge_buffer, bytes_to_read - read_bytes);
    }
    assert(curr_read_bytes >= 0);
    read_bytes += curr_read_bytes;
    EdgeId curr_read_edges = curr_read_bytes / edge_unit_size;
    // #pragma omp parallel for
    for (EdgeId e_i = 0; e_i < curr_read_edges; e_i++) {
      VertexId src = read_edge_buffer[e_i].src;
      VertexId dst = read_edge_buffer[e_i].dst;
      //        if(dst==875710)
      //            printf("%d",read_edge_buffer[e_i].src);
      row_indices[tmp_offset[dst]++] = src;
    }
  }
}

inline bool check(int v_id, Graph<Empty> *graph, VertexId *column_offset,
                  VertexId *row_indices, Bitmap *cached, int degree_full) {
  int count = 0;
  int error = 0;
  // std::cout<<"test";
  for (int i = column_offset[v_id]; i < column_offset[v_id + 1]; i++) {
    if (cached->get_bit(row_indices[i]) ||
        graph->get_partition_id(row_indices[i]) == graph->partition_id)
      count++;
    else
      error++;
    if (error > REPLICATE_THRESHOLD)
      return false;
  }
  if ((degree_full - count) <= REPLICATE_THRESHOLD)
    return true;
  else {
    return false;
  }
}
inline void valid(int v_id, Graph<Empty> *graph, VertexId *column_offset,
                  VertexId *row_indices, Bitmap *cached,
                  int degree_full) { // check
  int count = 0;
  int error = 0;
  // std::cout<<"test";
  for (int i = column_offset[v_id]; i < column_offset[v_id + 1]; i++) {
    if (graph->get_partition_id(row_indices[i]) != graph->partition_id) {
      cached->set_bit(row_indices[i]);
    }
  }
}

void compute_active(Graph<Empty> *graph, int iterations,
                    std::string path) { //分区算法
  int sockets = graph->sockets;
  VertexId *column_offset = new VertexId[graph->vertices + 1];
  VertexId *row_indices = new VertexId[graph->edges + 1];
  Bitmap *cached_0_vertices = new Bitmap(graph->vertices);

  cached_0_vertices->clear();

  VertexSubset *active = graph->alloc_vertex_subset();
  active->fill();

  int *indegree = new int[graph->vertices];
  memset(indegree, 0, sizeof(int));
  int *cahce_vertices_on_worker = new int[graph->partitions];
  int *cahce_vertices_at_layer0_on_worker = new int[graph->partitions];
  int *cahce_edges_at_layer0_on_worker = new int[graph->partitions];
  int *reduce_comm_at_layer0_on_worker = new int[graph->partitions];
  int allreduce_comm_at_layer0_on_worker = 0;

  memset(cahce_vertices_on_worker, 0, sizeof(int) * graph->partitions);
  memset(reduce_comm_at_layer0_on_worker, 0, sizeof(int) * graph->partitions);
  memset(cahce_vertices_at_layer0_on_worker, 0,
         sizeof(int) * graph->partitions);
  memset(cahce_edges_at_layer0_on_worker, 0, sizeof(int) * graph->partitions);

  // var for layer 2
  Bitmap *cached_inter_vertices_when_layer2 = new Bitmap(graph->vertices);
  Bitmap *cached_final_vertices_when_layer2 = new Bitmap(graph->vertices);
  Bitmap *cached_0_feature = new Bitmap(graph->vertices);
  cached_0_feature->clear();
  cached_final_vertices_when_layer2->clear();
  cached_inter_vertices_when_layer2->clear();

  int *cahce_inter_vertices_on_worker_layer2 = new int[graph->partitions];
  int *cahce_edges_at_layer2_on_worker = new int[graph->partitions];
  memset(cahce_inter_vertices_on_worker_layer2, 0,
         sizeof(int) * graph->partitions);
  memset(cahce_edges_at_layer2_on_worker, 0, sizeof(int) * graph->partitions);

  int *redundant_feature_cached = new int[graph->partitions];
  memset(redundant_feature_cached, 0, sizeof(int) * graph->partitions);

  int *redundant_feature_cached_layer2 = new int[graph->partitions];
  memset(redundant_feature_cached_layer2, 0, sizeof(int) * graph->partitions);
  int *redundant_feature_cached_layerall = new int[graph->partitions];
  memset(redundant_feature_cached_layerall, 0, sizeof(int) * graph->partitions);

  read_whole_graph(column_offset, row_indices, graph->vertices, graph->edges,
                   path);
  // std::cout<<row_indices[column_offset[875710]]<<std::endl;

  // printf("graph in_degree %d\n",graph->in_degree[iterations]);
  // printf("graph in_degree_backard
  // %d\n",graph->in_degree_for_backward[iterations]);

  // compute for first layer
  graph->process_edges<int, datum>( // For EACH Vertex Processing
      [&](VertexId dst, VertexAdjList<Empty> incoming_adj) { // pull
        // indegree[dst]=incoming_adj.end-incoming_adj.begin;
        int red = 0;
        for (AdjUnit<Empty> *ptr = incoming_adj.begin; ptr != incoming_adj.end;
             ptr++) { // pull model
          red++;
        }
        indegree[dst] = red;
        if (indegree[dst] <= REPLICATE_THRESHOLD && indegree[dst] > 0) {
          cached_0_vertices->set_bit(dst);
          datum tmp;
          for (int i = 0; i < REPLICATE_THRESHOLD; i++) {
            tmp.data[i] = -1;
          }
          int inc = 0;
          for (AdjUnit<Empty> *ptr = incoming_adj.begin;
               ptr != incoming_adj.end; ptr++) { // pull model
            VertexId src = ptr->neighbour;
            tmp.data[inc++] = src;
          }
          graph->emit(dst, tmp);
        }
      },
      [&](VertexId dst, datum msg) {
        for (int i = 0; i < REPLICATE_THRESHOLD; i++) {
          if (msg.data[i] != -1)
            cached_0_feature->set_bit(msg.data[i]);
        }
        return 0;
      },
      active);
  // printf("finish_firstlayer\n");

  // printf("cache_inter %d  on worker %d\n",cont,graph->partition_id);

  int cont = 0;
  for (int i = 0; i < graph->vertices; i++) {
    if (cached_0_feature->get_bit(i)) {
      cont++;
    }
  }
  redundant_feature_cached[graph->partition_id] = cont;

  Bitmap *cached_0_feature_2 = new Bitmap(graph->vertices);
  cached_0_feature_2->clear();
  // compute for the second layer
  graph->process_edges<int, VertexId>( // For EACH Vertex Processing
      [&](VertexId dst, VertexAdjList<Empty> incoming_adj) { // pull
        // indegree[dst]=incoming_adj.end-incoming_adj.begin;
        if (cached_0_feature->get_bit(dst)) {
          // if((graph->in_degree_for_backward[dst]-indegree[dst])<=REPLICATE_THRESHOLD&&(graph->in_degree_for_backward[dst]-indegree[dst])>0){
          if (check(dst, graph, column_offset, row_indices, cached_0_feature,
                    graph->in_degree_for_backward[dst])) {
            cached_inter_vertices_when_layer2->set_bit(dst);
            valid(dst, graph, column_offset, row_indices, cached_0_feature_2,
                  graph->in_degree_for_backward[dst]);
          }
        }
      },
      [&](VertexId dst, VertexId msg) {
        // addToNext<CURRENT_LAYER_SIZE>(dst,msg.data);
        return 0;
      },
      active);

  cont = 0;
  for (int i = 0; i < graph->vertices; i++) {
    if (cached_0_feature_2->get_bit(i)) {
      cont++;
    }
  }
  redundant_feature_cached_layer2[graph->partition_id] = cont;

  cont = 0;
  for (int i = 0; i < graph->vertices; i++) {
    if (cached_0_feature_2->get_bit(i) || cached_0_feature->get_bit(i)) {
      cont++;
    }
  }
  redundant_feature_cached_layerall[graph->partition_id] = cont;

  for (int i = 0; i < graph->partitions; i++) {
    if (i != graph->partition_id) {
      for (int j = graph->partition_offset[i];
           j < graph->partition_offset[i + 1]; j++) {
        if (cached_0_vertices->get_bit(j)) {
          cahce_vertices_on_worker[i] += 1;
          cahce_edges_at_layer0_on_worker[i] += indegree[j];
        }
        if (cached_inter_vertices_when_layer2->get_bit(j)) {
          cahce_inter_vertices_on_worker_layer2[graph->partition_id] += 1;
          cahce_edges_at_layer2_on_worker[graph->partition_id] +=
              (graph->in_degree_for_backward[j] - indegree[j]);
        }
      }
    }
  }
  for (int i = 0; i < graph->partitions; i++) {
    reduce_comm_at_layer0_on_worker[graph->partition_id] +=
        cahce_vertices_on_worker[i];
  }

  float all_redundant_storage = 0;
  float all_redundant_storage_layer2 = 0;
  float all_redundant_storage_layerall = 0;
  MPI_Datatype vid_t = get_mpi_data_type<int>();
  // printf("CACHED VERTICES_pre %d\n", cahce_vertices_on_worker[0]);
  MPI_Allreduce(MPI_IN_PLACE, cahce_vertices_on_worker, graph->partitions,
                vid_t, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, cahce_edges_at_layer0_on_worker,
                graph->partitions, vid_t, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, reduce_comm_at_layer0_on_worker,
                graph->partitions, vid_t, MPI_SUM, MPI_COMM_WORLD);
  if (graph->partition_id == 0) {

    printf("Vertices: %d Edges: %d\n", graph->vertices, graph->edges);
    printf("DEBUG INFO AT THE FIRST "
           "LAYER######################################:\n");
    for (int i = 0; i < graph->partitions; i++) {
      printf(
          "CACHED VERTICES %d, CACHED edges: %d,RECUDE comm:%d on worker %d\n",
          cahce_vertices_on_worker[i], cahce_edges_at_layer0_on_worker[i],
          reduce_comm_at_layer0_on_worker[i], i);
    }
  }
  for (int i = 0; i < graph->partitions; i++) {
    allreduce_comm_at_layer0_on_worker += reduce_comm_at_layer0_on_worker[i];
  }
  if (graph->partition_id == 0) {
    printf("%d,%d overall message reduction: %f\n",
           allreduce_comm_at_layer0_on_worker,
           (graph->partitions * graph->vertices),
           ((float)allreduce_comm_at_layer0_on_worker /
            (graph->partitions * graph->vertices)));
    printf("###################################################################"
           "\n\n");
  }

  MPI_Allreduce(MPI_IN_PLACE, cahce_inter_vertices_on_worker_layer2,
                graph->partitions, vid_t, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, cahce_edges_at_layer2_on_worker,
                graph->partitions, vid_t, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, redundant_feature_cached, graph->partitions,
                vid_t, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, redundant_feature_cached_layer2,
                graph->partitions, vid_t, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, redundant_feature_cached_layerall,
                graph->partitions, vid_t, MPI_SUM, MPI_COMM_WORLD);
  for (int i = 0; i < graph->partitions; i++) {
    all_redundant_storage += redundant_feature_cached[i];
  }
  for (int i = 0; i < graph->partitions; i++) {
    all_redundant_storage_layer2 += redundant_feature_cached_layer2[i];
  }
  for (int i = 0; i < graph->partitions; i++) {
    all_redundant_storage_layerall += redundant_feature_cached_layerall[i];
  }
  all_redundant_storage /= (graph->partitions * graph->vertices);
  all_redundant_storage_layer2 /= (graph->partitions * graph->vertices);
  all_redundant_storage_layerall /= (graph->partitions * graph->vertices);
  int allreduce_comm_inter_for_layer2 = 0;
  if (graph->partition_id == 0) {
    printf("\nDEBUG INFO AT THE SECOND "
           "LAYER######################################:\n");
    for (int i = 0; i < graph->partitions; i++) {
      printf("feature need to CACHED %d, CACHED edges layer2: %d,inter_layer "
             "RECUDE comm:%d on worker %d\n",
             redundant_feature_cached[i],
             cahce_inter_vertices_on_worker_layer2[i],
             cahce_edges_at_layer2_on_worker[i], i);
      allreduce_comm_inter_for_layer2 +=
          cahce_inter_vertices_on_worker_layer2[i];
    }
    printf("###################################################################"
           "\n\n");
    if (graph->partition_id == 0) {
      printf("%d,%d overall message reduction: %f\n",
             allreduce_comm_inter_for_layer2,
             (graph->partitions * graph->vertices),
             ((float)allreduce_comm_inter_for_layer2 /
              (graph->partitions * graph->vertices)));
      printf("overall redundant storage: %f\n", all_redundant_storage);
      printf("overall redundant storage for layer2: %f\n",
             all_redundant_storage_layer2);
      printf("overall redundant storage for all: %f\n",
             all_redundant_storage_layerall);
      printf("#################################################################"
             "##\n\n");
    }
  }
}
// inline int count_edge(int v_id,Graph<Empty> * graph, VertexId
// *column_offset,VertexId *row_indices,int degree_full,int error=0){//check
//    int count=0;
//    int error=0;
//    //std::cout<<"test";
//    for(int i=column_offset[v_id];i<column_offset[v_id+1];i++){
//        if(graph->get_partition_id(row_indices[i])!=graph->partition_id){
//        cached->set_bit(row_indices[i]);
//        }
//    }
//}
void load_replicate(Graph<Empty> *graph, int iterations,
                    std::string path) { //新的分区算法

  int *indegree = new int[graph->vertices];
  memset(indegree, 0, sizeof(int));
  VertexSubset *active = graph->alloc_vertex_subset();
  active->fill();

  graph->HasRepVtx.clear();
  graph->RepVtx.clear();
  graph->EdgeRemote2Local.clear();
  graph->EdgeRemote2Remote.clear();
  std::vector<std::vector<VertexId>> EdgeTmpRemote2Local;
  std::vector<std::vector<VertexId>> EdgeTmpRemote2Remote;
  int beta = 3;

  for (int i = 0; i < 2; i++) {
    graph->HasRepVtx.push_back(new Bitmap(graph->vertices));
    graph->RepVtx.push_back(new Bitmap(graph->vertices));
    graph->HasRepVtx[i]->clear();
    graph->RepVtx[i]->clear();
    std::vector<VertexId> tmp;
    tmp.clear();
    graph->EdgeRemote2Local.push_back(tmp);
    graph->EdgeRemote2Remote.push_back(tmp);
    EdgeTmpRemote2Local.push_back(tmp);
    EdgeTmpRemote2Remote.push_back(tmp);
  }

  printf("Why mot in\n");
  int sockets = graph->sockets;
  VertexId *column_offset = new VertexId[graph->vertices + 1];
  VertexId *row_indices = new VertexId[graph->edges + 1];
  read_whole_graph(column_offset, row_indices, graph->vertices, graph->edges,
                   path);

  // compute for first layers
  graph->process_edges<int, datum>( // For EACH Vertex Processing
      [&](VertexId dst, VertexAdjList<Empty> incoming_adj) { // pull
        // indegree[dst]=incoming_adj.end-incoming_adj.begin;
        int red = 0;
        for (AdjUnit<Empty> *ptr = incoming_adj.begin; ptr != incoming_adj.end;
             ptr++) { // pull model
          red++;
        }
        indegree[dst] = red;
        if (indegree[dst] <= REPLICATE_THRESHOLD && indegree[dst] > 0) {
          // cached_0_vertices->set_bit(dst);
          graph->RepVtx[0]->set_bit(dst);
          int second_layer_count = 0;
          datum tmp;
          for (int i = 0; i < REPLICATE_THRESHOLD; i++) {
            tmp.data[i] = -1;
          }
          int inc = 0;
          for (AdjUnit<Empty> *ptr = incoming_adj.begin;
               ptr != incoming_adj.end; ptr++) { // pull model
            VertexId src = ptr->neighbour;
            tmp.data[inc++] = src;
            second_layer_count += column_offset[src + 1] - column_offset[src];
          }
          if (second_layer_count > 10) {
            graph->RepVtx[1]->set_bit(dst);
          }
          graph->emit(dst, tmp);
        }
      },
      [&](VertexId dst, datum msg) {
        for (int i = 0; i < REPLICATE_THRESHOLD; i++) {
          if (msg.data[i] != -1)
            // cached_0_feature->set_bit(msg.data[i]);
            graph->HasRepVtx[0]->set_bit(msg.data[i]);
        }
        return 0;
      },
      active);

  //        Bitmap* tmpRepSrc=new Bitmap(graph->vertices);
  //        Bitmap* tmpRepDst=new Bitmap(graph->vertices);
  //        tmpRepSrc->clear();
  //        tmpRepDst->clear();
  // compute for second layers
  //        for(int i=0;i<graph->vertices;i++){
  //            if(graph->HasRepVtx[0]->get_bit(i)){
  //                int edge_count=column_offset[i+1]-column_offset[i];
  //                if(edge_count<10){
  //                    tmpRepDst->set_bit(i);
  //                    for(int j=column_offset[i];j<column_offset[i+1];j++){
  //                    tmpRepSrc->set_bit(j);
  //                    }
  //                }
  //            }
  //        }
}

void compute(Graph<Empty> *graph, int iterations) {

  // gpu_processor *gp=new gpu_processor();
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
  pytool->in_degree = torch::from_blob(
      graph->in_degree + graph->partition_offset[graph->partition_id],
      {embedding->rownum, 1});
  pytool->out_degree = torch::from_blob(
      graph->in_degree + graph->partition_offset[graph->partition_id],
      {embedding->rownum, 1});
  /*1 INIT STAGE*/
  // GTensor<float,Empty> gt=new  GTensor(comm, graph);

  GnnUnit *Gnn_v1 = new GnnUnit(SIZE_LAYER_1, SIZE_LAYER_2);
  GnnUnit *Gnn_v2 = new GnnUnit(SIZE_LAYER_2, OUTPUT_LAYER_3);
  pytool->registOptimizer(torch::optim::SGD(Gnn_v1->parameters(), 0.05)); // new
  pytool->registOptimizer(torch::optim::SGD(Gnn_v2->parameters(), 0.05)); // new
  pytool->registLabel<long>(
      embedding->label, graph->partition_offset[graph->partition_id],
      graph->partition_offset[graph->partition_id + 1] -
          graph->partition_offset[graph->partition_id]); // new

  /*init W with new */

  init_parameter(comm, graph, Gnn_v1, embedding);
  init_parameter(comm1, graph, Gnn_v2, embedding);
  VertexSubset *active = graph->alloc_vertex_subset();
  active->fill();
  std::vector<int> layer_size(0);
  layer_size.push_back(SIZE_LAYER_1);
  layer_size.push_back(SIZE_LAYER_2);
  layer_size.push_back(OUTPUT_LAYER_3);
  GTensor<ValueType, long, MAX_LAYER> *gt =
      new GTensor<ValueType, long, MAX_LAYER>(graph, embedding, active, 2,
                                              layer_size);

  Intermediate *inter = new Intermediate(embedding->rownum, SIZE_LAYER_2);

  graph->process_vertices<ValueType>( // init  the vertex state.
      [&](VertexId vtx) {
        int start = (graph->partition_offset[graph->partition_id]);
        for (int i = 0; i < SIZE_LAYER_1; i++) {
          //*(start_v + vtx * MAX_LAYER + i) = con[j].att[i];
          embedding->initStartWith(vtx, embedding->con[vtx].att[i], i);
        }
        return (ValueType)1;
      },
      active);

  torch::Tensor new_combine_grad = torch::zeros({SIZE_LAYER_1, SIZE_LAYER_2});
  double exec_time = 0;
  double graph_time = 0;
  exec_time -= get_time();
  double nn_time = 0;
  for (int i_i = 0; i_i < iterations; i_i++) {
    gt->setValueFromNative(embedding->start_v, embedding->start);
    nn_time -= get_time();
    if (i_i > 0) {
      // inter->zero_grad();
      // pytool->x[1].grad().zero_();
      Gnn_v1->zero_grad();
      Gnn_v2->zero_grad();
    }
    nn_time += get_time();
    if (graph->partition_id == 0)
      std::cout << "start  [" << i_i
                << "]  epoch+++++++++++++++++++++++++++++++++++++++++++++"
                << std::endl;

    /*2. FORWARD STAGE*/
    // 2.1.1 start the forward of the first layer
    graph_time -= get_time();
    gt->Propagate<SIZE_LAYER_1>(0);
    graph_time += get_time();
    pytool->updateX(0, gt->value_local[0]);

    nn_time -= get_time();
    inter->W.set_data(Gnn_v1->forward(
        pytool->x
            [0])); // torch::from_blob(embedding->Gnn_v1->forward(pytool->x[0]).accessor<float,2>().data(),{embedding->rownum,MAX_LAYER});
    pytool->y[0] = torch::relu(inter->W); // new
    nn_time += get_time();

    //        std::cout<<pytool->y[0].size(0)<<"\t"<<pytool->y[0].size(1)<<std::endl;
    // 2.2.1 init the second layer
    gt->setValueFromTensor(pytool->y[0]);
    // 2.2.2 forward the second layer
    graph_time -= get_time();
    gt->Propagate<SIZE_LAYER_2>(1);
    graph_time += get_time();
    /*3 BACKWARD STAGE*/
    // 3.1 compute the output of the second layer.
    pytool->updateX(1, gt->value_local[1]); // new
    pytool->x[1].set_requires_grad(true);

    //        printf("%d\t%d\n",Gnn_v2->W.accessor<float,2>().size(0),gt->value_local[1].accessor<float,2>().size(1));

    nn_time -= get_time();
    pytool->y[1] = Gnn_v2->forward(pytool->x[1]);
    torch::Tensor tt = pytool->y[1].log_softmax(1); // CUDA

    pytool->loss = torch::nll_loss(tt, pytool->target); // new
    pytool->loss.backward();
    // 3.2 compute the gradient of the second layer.
    torch::Tensor aggregate_grad2 = unified_parameter<ValueType>(
        comm1, Gnn_v2->W.grad());         // Gnn_v2->W.grad()
    Gnn_v2->learn(aggregate_grad2, 0.01); // reset from new
    nn_time += get_time();

    // 3.3.3 backward the partial gradient from 2-layer to 1-layer torch::Tensor
    // partial_grad_layer2=pytool->x[1].grad();
    graph_time -= get_time();
    gt->setGradFromTensor(pytool->x[1].grad());
    gt->Propagate_backward<SIZE_LAYER_2>(0);
    graph_time += get_time();
    //*3.3.1  compute  W1's partial gradient in first layer

    nn_time -= get_time();
    pytool->y[0].backward();                // new
    pytool->localGrad[0] = inter->W.grad(); // new

    new_combine_grad.zero_();
    /*graph->process_vertices<ValueType>(//init  the vertex state.
    [&](VertexId vtx){
     new_combine_grad_local[omp_get_thread_num()]=new_combine_grad[omp_get_thread_num()]+
        pytool->x[0].slice(0,vtx,vtx+1,1).t().mm(
        (pytool->localGrad[0].slice(0,vtx,vtx+1,1)*gt->grad_local[0].slice(0,vtx,vtx+1,1)));
        return (ValueType)1;
    },
active
);
for(int i=0;i<new_combine_grad_local.size();i++){
  new_combine_grad+=new_combine_grad_local[i];
}*/
    new_combine_grad = pytool->x[0]
                           .slice(0, 0, +graph->owned_vertices, 1)
                           .t()
                           .mm(pytool->localGrad[0] * gt->grad_local[0]);

    torch::Tensor aggregate_grad =
        unified_parameter<ValueType>(comm, (new_combine_grad));
    Gnn_v1->learn(aggregate_grad, 0.01);
    nn_time += get_time();

    if (graph->partition_id == 0) {
      std::cout << "LOSS:\t" << pytool->loss << std::endl;
    }

    //     if(i_i==(iterations-1)&&graph->partition_id==0){
    //        inference(tt,graph, embedding, pytool,Gnn_v1,Gnn_v2);
    //     }
  }
  exec_time += get_time();
  if (graph->partition_id == 0) {
    printf("exec_time=%lf(s)\n", exec_time);
    printf("graph_time=%lf(s)\n", graph_time);
    printf("nn_time=%lf(s)\n", nn_time);
  }
  delete active;
}
/*CPU single order*/ void compute_gf(Graph<Empty> *graph, int iterations) {

  // gpu_processor *gp=new gpu_processor();
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
  pytool->in_degree = torch::from_blob(
      graph->in_degree + graph->partition_offset[graph->partition_id],
      {embedding->rownum, 1});
  pytool->out_degree = torch::from_blob(
      graph->in_degree + graph->partition_offset[graph->partition_id],
      {embedding->rownum, 1});
  /*1 INIT STAGE*/
  // GTensor<float,Empty> gt=new  GTensor(comm, graph);

  GnnUnit *Gnn_v1 = new GnnUnit(SIZE_LAYER_1, SIZE_LAYER_2);
  GnnUnit *Gnn_v2 = new GnnUnit(SIZE_LAYER_2, OUTPUT_LAYER_3);
  pytool->registOptimizer(torch::optim::SGD(Gnn_v1->parameters(), 0.05)); // new
  pytool->registOptimizer(torch::optim::SGD(Gnn_v2->parameters(), 0.05)); // new
  pytool->registLabel<long>(
      embedding->label, graph->partition_offset[graph->partition_id],
      graph->partition_offset[graph->partition_id + 1] -
          graph->partition_offset[graph->partition_id]); // new

  /*init W with new */

  init_parameter(comm, graph, Gnn_v1, embedding);
  init_parameter(comm1, graph, Gnn_v2, embedding);
  VertexSubset *active = graph->alloc_vertex_subset();
  active->fill();
  std::vector<int> layer_size(0);
  layer_size.push_back(SIZE_LAYER_1);
  layer_size.push_back(SIZE_LAYER_2);
  layer_size.push_back(OUTPUT_LAYER_3);
  GTensor<ValueType, long, MAX_LAYER> *gt =
      new GTensor<ValueType, long, MAX_LAYER>(graph, embedding, active, 2,
                                              layer_size);

  Intermediate *inter = new Intermediate(embedding->rownum, SIZE_LAYER_2);

  graph->process_vertices<ValueType>( // init  the vertex state.
      [&](VertexId vtx) {
        int start = (graph->partition_offset[graph->partition_id]);
        for (int i = 0; i < SIZE_LAYER_1; i++) {
          //*(start_v + vtx * MAX_LAYER + i) = con[j].att[i];
          embedding->initStartWith(vtx, embedding->con[vtx].att[i], i);
        }
        return (ValueType)1;
      },
      active);

  torch::Tensor x0 =
      torch::from_blob(embedding->start_v, {graph->vertices, SIZE_LAYER_1});
  torch::Tensor new_combine_grad = torch::zeros({SIZE_LAYER_1, SIZE_LAYER_2});
  double exec_time = 0;
  double graph_time = 0;
  exec_time -= get_time();
  for (int i_i = 0; i_i < iterations; i_i++) {

    if (i_i > 0) {
      // inter->zero_grad();
      // pytool->x[1].grad().zero_();
      Gnn_v1->zero_grad();
      Gnn_v2->zero_grad();
    }
    if (graph->partition_id == 0)
      std::cout << "start  [" << i_i
                << "]  epoch+++++++++++++++++++++++++++++++++++++++++++++"
                << std::endl;

    /*2. FORWARD STAGE*/
    // 2.1.1 start the forward of the first layer
    torch::Tensor x0_trans = Gnn_v1->forward(x0);
    // std::cout<<"stage1"<<std::endl;
    gt->setValueFromNative(x0_trans.accessor<float, 2>().data(),
                           graph->partition_offset[graph->partition_id] *
                               SIZE_LAYER_2);
    // std::cout<<"stage2"<<std::endl;
    gt->Propagate<SIZE_LAYER_2>(0);
    pytool->updateX(0, gt->value_local[0]);

    // std::cout<<pytool->x[0].size(0)<<"TETSTETSTETSTE"<<pytool->x[0].size(1)<<std::endl;
    inter->W.set_data(
        pytool->x
            [0]); // torch::from_blob(embedding->Gnn_v1->forward(pytool->x[0]).accessor<float,2>().data(),{embedding->rownum,MAX_LAYER});
    pytool->y[0] = torch::relu(inter->W); // new

    //        std::cout<<pytool->y[0].size(0)<<"\t"<<pytool->y[0].size(1)<<std::endl;
    // 2.2.1 init the second layer
    gt->setValueFromTensor(pytool->y[0]);
    // 2.2.2 forward the second layer
    gt->Propagate<SIZE_LAYER_2>(1);
    /*3 BACKWARD STAGE*/
    // 3.1 compute the output of the second layer.
    pytool->updateX(1, gt->value_local[1]); // new
    pytool->x[1].set_requires_grad(true);

    //        printf("%d\t%d\n",Gnn_v2->W.accessor<float,2>().size(0),gt->value_local[1].accessor<float,2>().size(1));

    pytool->y[1] = Gnn_v2->forward(pytool->x[1]);
    torch::Tensor tt = pytool->y[1].log_softmax(1);     // CUDA
    pytool->loss = torch::nll_loss(tt, pytool->target); // new
    pytool->loss.backward();
    // 3.2 compute the gradient of the second layer.
    torch::Tensor aggregate_grad2 = unified_parameter<ValueType>(
        comm1, Gnn_v2->W.grad()); // Gnn_v2->W.grad()
    Gnn_v2->learn(
        aggregate_grad2,
        0.01); // reset from new
               // 3.3.3 backward the partial gradient from 2-layer to 1-layer
               // torch::Tensor partial_grad_layer2=pytool->x[1].grad();
    gt->setGradFromTensor(pytool->x[1].grad());
    gt->Propagate_backward<SIZE_LAYER_2>(0);
    //*3.3.1  compute  W1's partial gradient in first layer
    pytool->y[0].backward();                // new
    pytool->localGrad[0] = inter->W.grad(); // new
    torch::Tensor tmpgrad = pytool->localGrad[0] * gt->grad_local[0];
    gt->setGradFromTensor(tmpgrad);
    gt->Propagate_backward<SIZE_LAYER_2>(1);

    new_combine_grad.zero_();
    printf("hellow\n");
    std::cout << x0.size(0) << "pp" << x0.size(1) << std::endl;
    std::cout << gt->grad_local[0].size(0) << "qq" << gt->grad_local[0].size(1)
              << std::endl;

    int offset_ = graph->partition_offset[graph->partition_id];
    new_combine_grad = x0.slice(0, offset_, offset_ + graph->owned_vertices, 1)
                           .t()
                           .mm(gt->grad_local[1]);

    torch::Tensor aggregate_grad =
        unified_parameter<ValueType>(comm, (new_combine_grad));
    Gnn_v1->learn(aggregate_grad, 0.01);

    if (graph->partition_id == 0) {
      std::cout << "LOSS:\t" << pytool->loss << std::endl;
    }

    //     if(i_i==(iterations-1)&&graph->partition_id==0){
    //        inference(tt,graph, embedding, pytool,Gnn_v1,Gnn_v2);
    //     }
  }
  exec_time += get_time();
  if (graph->partition_id == 0) {
    printf("exec_time=%lf(s)\n", exec_time);
  }
  delete active;
}
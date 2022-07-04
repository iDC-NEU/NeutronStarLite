/*
Copyright (c) 2021-2022 Qiange Wang, Northeastern University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#ifndef NTSDATALODOR_HPP
#define NTSDATALODOR_HPP
#include <assert.h>
#include <map>
#include <math.h>
#include <stack>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <vector>
#include <sstream>
#include "core/graph.hpp"

class GNNDatum {
public:
  GNNContext *gnnctx;
  Graph<Empty> *graph;
  ValueType *local_feature; // features of local partition
  long *local_label;        // labels of local partition
  int *local_mask; // mask(indicate whether data is for train, eval or test) of
                   // local partition

  // GNN datum world

// train:    0
// val:     1
// test:     2
/**
 * @brief Construct a new GNNDatum::GNNDatum object.
 * initialize GNN Data using GNNContext and Graph.
 * Allocating space to save data. e.g. local feature, local label.
 * @param _gnnctx pointer to GNN Context
 * @param graph_ pointer to Graph
 */
GNNDatum(GNNContext *_gnnctx, Graph<Empty> *graph_) {
  gnnctx = _gnnctx;
  local_feature = new ValueType[gnnctx->l_v_num * gnnctx->layer_size[0]];
  local_label = new long[gnnctx->l_v_num];
  local_mask = new int[gnnctx->l_v_num];
  memset(local_mask, 1, sizeof(int) * gnnctx->l_v_num);
  graph = graph_;
}

/**
 * @brief
 * generate random data for feature, label and mask
 */
void random_generate() {
  for (int i = 0; i < gnnctx->l_v_num; i++) {
    for (int j = 0; j < gnnctx->layer_size[0]; j++) {
      local_feature[i * gnnctx->layer_size[0] + j] = 1.0;
    }
    local_label[i] = rand() % gnnctx->label_num;
    local_mask[i] = i % 3;
  }
}

/**
 * @brief
 * Create tensor corresponding to local label
 * @param target target tensor where we should place local label
 */
void registLabel(NtsVar &target) {
  target = graph->Nts->NewLeafKLongTensor(local_label, {gnnctx->l_v_num});
  // torch::from_blob(local_label, gnnctx->l_v_num, torch::kLong);
}

/**
 * @brief
 * Create tensor corresponding to local mask
 * @param mask target tensor where we should place local mask
 */
void registMask(NtsVar &mask) {
  mask = graph->Nts->NewLeafKIntTensor(local_mask, {gnnctx->l_v_num, 1});
  // torch::from_blob(local_mask, {gnnctx->l_v_num,1}, torch::kInt32);
}

/**
 * @brief
 * read feature and label from file.
 * file format should be  ID Feature * (feature size) Label
 * @param inputF path to input feature
 * @param inputL path to input label
 */
void readFtrFrom1(std::string inputF, std::string inputL) {

  std::string str;
  std::ifstream input_ftr(inputF.c_str(), std::ios::in);
  std::ifstream input_lbl(inputL.c_str(), std::ios::in);
  // std::ofstream outputl("cora.labeltable",std::ios::out);
  // ID    F   F   F   F   F   F   F   L
  if (!input_ftr.is_open()) {
    std::cout << "open feature file fail!" << std::endl;
    return;
  }
  if (!input_lbl.is_open()) {
    std::cout << "open label file fail!" << std::endl;
    return;
  }
  ValueType *con_tmp = new ValueType[gnnctx->layer_size[0]];
  // TODO: figure out what is la
  std::string la;
  // std::cout<<"finish1"<<std::endl;
  VertexId id = 0;
  while (input_ftr >> id) {
    // feature size
    VertexId size_0 = gnnctx->layer_size[0];
    // translate vertex id to local vertex id
    VertexId id_trans = id - gnnctx->p_v_s;
    if ((gnnctx->p_v_s <= id) && (gnnctx->p_v_e > id)) {
      // read feature
      for (int i = 0; i < size_0; i++) {
        input_ftr >> local_feature[size_0 * id_trans + i];
      }
      input_lbl >> la;
      // read label
      input_lbl >> local_label[id_trans];
      // partition data set based on id
      local_mask[id_trans] = id % 3;
    } else {
      // dump the data which doesn't belong to local partition
      for (int i = 0; i < size_0; i++) {
        input_ftr >> con_tmp[i];
      }
      input_lbl >> la;
      input_lbl >> la;
    }
  }
  delete[] con_tmp;
  input_ftr.close();
  input_lbl.close();
}

/**
 * @brief
 * read feature, label and mask from file.
 * @param inputF path to feature file
 * @param inputL path to label file
 * @param inputM path to mask file
 */
void readFeature_Label_Mask(std::string inputF, std::string inputL,
                                      std::string inputM) {

  // logic here is exactly the same as read feature and label from file
  std::string str;
  std::ifstream input_ftr(inputF.c_str(), std::ios::in);
  std::ifstream input_lbl(inputL.c_str(), std::ios::in);
  std::ifstream input_msk(inputM.c_str(), std::ios::in);
  // std::ofstream outputl("cora.labeltable",std::ios::out);
  // ID    F   F   F   F   F   F   F   L
  if (!input_ftr.is_open()) {
    std::cout << "open feature file fail!" << std::endl;
    return;
  }
  if (!input_lbl.is_open()) {
    std::cout << "open label file fail!" << std::endl;
    return;
  }
  if (!input_msk.is_open()) {
    std::cout << "open mask file fail!" << std::endl;
    return;
  }
  ValueType *con_tmp = new ValueType[gnnctx->layer_size[0]];
  std::string la;
  // std::cout<<"finish1"<<std::endl;
  VertexId id = 0;
  while (input_ftr >> id) {
    VertexId size_0 = gnnctx->layer_size[0];
    VertexId id_trans = id - gnnctx->p_v_s;
    if ((gnnctx->p_v_s <= id) && (gnnctx->p_v_e > id)) {
      for (int i = 0; i < size_0; i++) {
        input_ftr >> local_feature[size_0 * id_trans + i];
      }
      input_lbl >> la;
      input_lbl >> local_label[id_trans];

      input_msk >> la;
      std::string msk;
      input_msk >> msk;
      // std::cout<<la<<" "<<msk<<std::endl;
      if (msk.compare("train") == 0) {
        local_mask[id_trans] = 0;
      } else if (msk.compare("eval") == 0 || msk.compare("val") == 0) {
        local_mask[id_trans] = 1;
      } else if (msk.compare("test") == 0) {
        local_mask[id_trans] = 2;
      } else {
        local_mask[id_trans] = 3;
      }

    } else {
      for (int i = 0; i < size_0; i++) {
        input_ftr >> con_tmp[i];
      }

      input_lbl >> la;
      input_lbl >> la;

      input_msk >> la;
      input_msk >> la;
    }
  }
  delete[] con_tmp;
  input_ftr.close();
  input_lbl.close();
}

void readFeature_Label_Mask_OGB(std::string inputF, std::string inputL,
                                      std::string inputM) {

  // logic here is exactly the same as read feature and label from file
  std::string str;
  std::ifstream input_ftr(inputF.c_str(), std::ios::in);
  std::ifstream input_lbl(inputL.c_str(), std::ios::in);
  // ID    F   F   F   F   F   F   F   L
  std::cout<<inputF<<std::endl;
  if (!input_ftr.is_open()) {
    std::cout << "open feature file fail!" << std::endl;
    return;
  }
  if (!input_lbl.is_open()) {
    std::cout << "open label file fail!" << std::endl;
    return;
  }
  ValueType *con_tmp = new ValueType[gnnctx->layer_size[0]];
  std::string la;
  std::string featStr;
  for (VertexId id = 0;id<graph->vertices;id++) {
    VertexId size_0 = gnnctx->layer_size[0];
    VertexId id_trans = id - gnnctx->p_v_s;
    if ((gnnctx->p_v_s <= id) && (gnnctx->p_v_e > id)) {
        getline(input_ftr,featStr);
        std::stringstream ss(featStr);
        std::string feat_u;
        int i=0;
        while(getline(ss,feat_u,',')){
            local_feature[size_0 * id_trans + i]=std::atof(feat_u.c_str());
//            if(id==0){
//                std::cout<<std::atof(feat_u.c_str())<<std::endl;
//            }
            i++;
        }assert(i==size_0);       
      //input_lbl >> la;
      input_lbl >> local_label[id_trans];

    } else {
      getline(input_ftr,featStr);
      input_lbl >> la;
    }
  }
  
  std::string inputM_train=inputM;
  inputM_train.append("/train.csv");
  std::string inputM_val=inputM;
  inputM_val.append("/valid.csv");
  std::string inputM_test=inputM;
  inputM_test.append("/test.csv");
  std::ifstream input_msk_train(inputM_train.c_str(), std::ios::in);
  if (!input_msk_train.is_open()) {
    std::cout << "open input_msk_train file fail!" << std::endl;
    return;
  }
  std::ifstream input_msk_val(inputM_val.c_str(), std::ios::in);
  if (!input_msk_val.is_open()) {
    std::cout <<inputM_val<< "open input_msk_val file fail!" << std::endl;
    return;
  }
  std::ifstream input_msk_test(inputM_test.c_str(), std::ios::in);
  if (!input_msk_test.is_open()) {
    std::cout << "open input_msk_test file fail!" << std::endl;
    return;
  }
  VertexId vtx=0;
  while(input_msk_train>>vtx){//train
      local_mask[vtx] = 0;
  }
  while(input_msk_val>>vtx){//val
      local_mask[vtx] = 1;
  }
  while(input_msk_test>>vtx){//test
      local_mask[vtx] = 2;
  }
  
  
  delete[] con_tmp;
  input_ftr.close();
  input_lbl.close();
}

};

#endif

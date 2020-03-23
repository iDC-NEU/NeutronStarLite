/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include "core/graph.hpp"
#include "core/operator.hpp"
//#include "torch/script.h"
//
#include "torch/torch.h"
#include "torch/csrc/autograd/generated/variable_factories.h"
#define gnnarray GnnUnit**
/**/
void testread(){
     torch::Tensor tmp;
     float data[]={0.2,0.4,0.5,0.6,0.7,0.8};
     float *b=new float[6];
     tmp=torch::from_blob(data,{6});
     auto tmp_acc=tmp.accessor<float,1>();
     for(int i=0;i<tmp_acc.size(0);i++){
         printf("%f\t",tmp_acc[i]);
     }
     //memcpy(b,tmp_acc.data(),6*sizeof(float));
     b=tmp_acc.data();
     printf("\n");
         for(int i=0;i<tmp_acc.size(0);i++){
         printf("%f\t",tmp_acc[i]);
     }
    
printf("hello world\n");
}
struct Net : torch::nn::Module {
    
  torch::Tensor W;
  torch::Tensor A = torch::rand({4,4}); 
  
  Net(int64_t N, int64_t D) {
    W = register_parameter("W", torch::randn({N, D}));
    for(int i = 0; i < A.size(0); i++) {
        A[i][i]=1;
    }
  }

  torch::Tensor forward(torch::Tensor x) {
    x = A*x*W;
    x = torch::relu(x);
    return x;
  }
};
struct GnnUnit: torch::nn::Module{
    torch::Tensor W;
    torch::Tensor A;
    
    GnnUnit(size_t w, size_t h){
//        at::TensorOptions *opt=new at::TensorOptions;
 //       opt->requires_grad(true);
      //  torch::randn
   //     A=torch::randn(torch::randn({w,h},opt));
        W=register_parameter("W",torch::randn({w,h}));
        //A=torch::rand({w,1});
         float *data=new float[4];
         for(int i = 0; i < 4; i++) {
        data[i]=1.5;
        }
         A=torch::from_blob(data,{4});
    }
    torch::Tensor forward(torch::Tensor x) {
        
        auto tmp_acc_c=W.accessor<float,2>();
        printf("======W======\n");
     for(int i=0;i<tmp_acc_c.size(0);i++){
         for(int j=0;j<tmp_acc_c.size(1);j++)
         printf("%f\t",tmp_acc_c[i][j]);
         printf("\n");
     } 
        printf("=======before forward :x.size and w.size=====\n");
        printf("%d,%d,,,%d,%d\n",x.size(0),x.size(1),W.size(0),W.size(1));
        x=x.mm(W);
        return torch::log_softmax(x,1);
    }
};
int main(){
     
    
    float *data=new float[4];
    for(int i = 0; i < 4; i++) {
        data[i]=2.0;
        }
    torch::Tensor x=torch::from_blob(data,{1,4});
    auto tmp_acc=x.accessor<float,2>();
    printf("====x===\n");
     for(int i=0;i<tmp_acc.size(0);i++){
         for(int j=0;j<tmp_acc.size(1);j++){
         printf("%f\t",tmp_acc[i][j]);}
         printf("\n");
     }
    
    long la = 3;//4位，最多4类?
    long *l=&la;
    torch::Tensor label = torch::from_blob(l,{1},torch::kLong);
    
    //auto label1 = torch::empty(1,torch::kLong).random_(3);
    //auto tmp=label1.accessor<long,1>();
    //printf("====label1===%d\n",tmp.size(0));
    //for(int i=0;i<tmp.size(0);i++){
    //    printf("%f\t",tmp[i]);
    //    printf("\n");
    //}
 
     float *data1=new float[4];
    for(int i = 0; i < 4; i++) {
        data1[i]=1.5;
        }
    torch::Tensor a=torch::from_blob(data1,{1,4});
    tmp_acc=a.accessor<float,2>();
    printf("====a===\n");
    for(int i=0;i<tmp_acc.size(0);i++){
         for(int j=0;j<tmp_acc.size(1);j++){
         printf("%f\t",tmp_acc[i][j]);}
         printf("\n");
     }
    printf("===== x=x*a =====\n");
    x=x.mul(a);
    printf("%d,%d\n",x.size(0),x.size(1));
    printf("=====forward()=====\n");
    GnnUnit *gnn=new GnnUnit(4,4);
    /**
     
     */
    //GnnUnit** curr=new (GnnUnit*)[10];
    //for(int i=0;i<10;i++){
    //   curr[i]=new GnnUnit(4,4);
    //}
    /**/
    torch::optim::SGD optimizer(gnn->parameters(), /*lr=*/0.01);
    optimizer.zero_grad();
    torch::Tensor prediction = gnn->forward(x);
    
    printf("=======after forward size=====\n");
    printf("%d,%d,\n",prediction.size(0),prediction.size(1));
    printf("=======after forward x===\n");
    auto tmp_acc_c=prediction.accessor<float,2>();
    for(int i=0;i<tmp_acc_c.size(0);i++){
        for(int j=0;j<tmp_acc_c.size(1);j++)
        printf("%f\t",tmp_acc_c[i][j]);
        printf("\n");
    }
    printf("=========backgrad()=============\n");
    torch::Tensor loss = torch::nll_loss(prediction,label);
    loss.backward();
    optimizer.step();
    std::cout<<loss.item<float>()<<std::endl;
    for (const auto& pair : gnn->named_parameters()) {
        std::cout << pair.key() << ": " << pair.value() << std::endl;
        if(pair.key()=="W"){
            tmp_acc=pair.value().accessor<float,2>();
            for(int i=0;i<tmp_acc.size(0);i++){
                for(int j=0;j<tmp_acc.size(1);j++){
                    printf("%f\t",tmp_acc[i][j]);
                }
                printf("\n");
            }
        }
    }
    
    //可更新W
    torch::Tensor new_data=torch::from_blob(data,{1,4});
    gnn->W.set_data(new_data);
    for (const auto& pair : gnn->named_parameters()) {
        std::cout << pair.key() << ": " << pair.value() << std::endl;
        if(pair.key()=="W"){
            tmp_acc=pair.value().accessor<float,2>();
            for(int i=0;i<tmp_acc.size(0);i++){
                for(int j=0;j<tmp_acc.size(1);j++){
                    printf("%f\t",tmp_acc[i][j]);
                }
                printf("\n");
            }
        }
    }
    
    return 0;
}

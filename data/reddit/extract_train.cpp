#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <string.h>
#include <cstring>
using namespace std;
int generate_Mask(int vertices,std::string mask_file){
    FILE *fp_mask;
    fp_mask=fopen(mask_file.c_str(),"w");
    for(int i=0;i<vertices;i++){
        int a=rand()%10;
        if(a<6){
            fprintf(fp_mask,"%d train\n",i);  
        }else  
        if(a<8){
            fprintf(fp_mask,"%d test\n",i);  
        }else
        if(a<10){
            fprintf(fp_mask,"%d val\n",i);  
        }
    }
    fclose(fp_mask);
            return 0;
    
}
void writeLabels(int vertices,std::string label_file)
    {
        std::string str;
        std::string label_file_output=label_file+"_nts";
        std::ifstream input_lbl(label_file.c_str(), std::ios::in);
        std::ofstream output_lbl(label_file_output.c_str(), std::ios::out);
       // ID    F   F   F   F   F   F   F   L
        std::cout<<"called"<<std::endl;
        if (!input_lbl.is_open())
        {
            std::cout<<"open label file fail!"<<std::endl;
            return;
        }
        std::string la;
        //std::cout<<"finish1"<<std::endl;
        int id=0;
        int label;
        for(int j=0;j<vertices;j++)
        {
                input_lbl >> label;
                output_lbl<<j<<" "<<label<<std::endl;
                
        }
        input_lbl.close();
        output_lbl.close();
    }


int sort_Mask(int vertices,std::string mask_file){
    
    std::cout<<vertices<<std::endl;
    std::string mask_file_sorted=mask_file+"sorted";
    std::ifstream fp_mask(mask_file.c_str(), std::ios::in);      
    std::ofstream fp_mask_sorted(mask_file_sorted.c_str(), std::ios::out);
   if (!fp_mask.is_open())
        {
            std::cout<<"open"<<mask_file<<" fail!"<<std::endl;
            return 0;
        }
    int* value=new int[vertices];
    memset(value,0,sizeof(int)*vertices);
    while(!fp_mask.eof()){
        int id;
        std::string s;
        fp_mask>>id>>s;
        std::cout<<id<<" "<<s<<std::endl;
        if(s.compare("train")==0){
            value[id]=1;
        }else if(s.compare("val")==0){
            value[id]=2;
        }else if(s.compare("test")==0){
            value[id]=3;
        }else{
            ;
        }
    }
    for(int i=0;i<vertices;i++){
        std::string s;
        if(value[i]==1){
        fp_mask_sorted<<i<<" train"<<std::endl;
        }else if(value[i]==2){
        fp_mask_sorted<<i<<" val"<<std::endl;
        }
        if(value[i]==3){
        fp_mask_sorted<<i<<" test"<<std::endl;
        }
    }
    fp_mask_sorted.close();
    fp_mask.close();
    //fclose(fp_mask);
            return 0;
}
int sort_Label(int vertices,std::string label_file){
    
    std::string label_file_sorted=label_file+"sorted";
    std::ifstream fp_label(label_file.c_str(), std::ios::in);      
    std::ofstream fp_label_sorted(label_file_sorted.c_str(), std::ios::out);
  if (!fp_label.is_open())
        {
            std::cout<<"open "<< label_file<<" fail!"<<std::endl;
            return 0;
        }
    int* value=new int[vertices];
    memset(value,0,sizeof(int)*vertices);
    while(!fp_label.eof()){
        int id;
        int v;
        fp_label>>id>>v;
        value[id]=v;
    }
    for(int i=0;i<vertices;i++){
        fp_label_sorted<<i<<" "<<value[i]<<std::endl;
    }
    fp_label_sorted.close();
    fp_label.close();
    //fclose(fp_label);
    return 0;
}
void generate_edge(int vertices,std::string edge_file){
    
    std::string edge_file_sorted=edge_file+".bin";
    std::ifstream fp_edge(edge_file.c_str(), std::ios::in);      
    std::ofstream fp_edge_sorted(edge_file_sorted.c_str(), std::ios::binary);
    if (!fp_edge.is_open())
        {
            std::cout<<"open "<< edge_file<<" fail!"<<std::endl;
            return;
        }
    while(!fp_edge.eof()){
        int src;
        int dst;
        fp_edge>>src>>dst;
        fp_edge_sorted.write((char*)&src,sizeof(int));
        fp_edge_sorted.write((char*)&dst,sizeof(int));
        fp_edge_sorted.write((char*)&dst,sizeof(int));
        fp_edge_sorted.write((char*)&src,sizeof(int));
        //fp_edge_sorted<<dst<<" "<<src<<std::endl;
    }
    for(int i=0;i<vertices;i++){
        fp_edge_sorted.write((char*)&i,sizeof(int));
        fp_edge_sorted.write((char*)&i,sizeof(int));
    }
    fp_edge_sorted.close();
    fp_edge.close();
    //fclose(fp_label);   
    std::ifstream fp_edge_test(edge_file_sorted.c_str(), std::ios::binary);
    for(int i=0;i<23446805;i++){
        int src;
        int dst;
        fp_edge_test.read((char*)&src,sizeof(int));
        fp_edge_test.read((char*)&dst,sizeof(int));
        if(i>23446705)
        std::cout<<"edge[0]: "<<src<<" edge[1]: "<<dst<<std::endl;
        //fp_edge_sorted<<dst<<" "<<src<<std::endl;
    }
    return;
}
void writeFeature(int vertices, int features,std::string input_feature)
    {
        std::string str;
        std::string output_feature=input_feature+"_nts";
        std::ifstream input_ftr(input_feature.c_str(), std::ios::in);
        std::ofstream output_ftr(output_feature.c_str(), std::ios::out);
        std::cout<<"called"<<std::endl;
        if (!input_ftr.is_open())
        {
            std::cout<<"open feature file fail!"<<std::endl;
            return;
        }
//        if (!input_lbl.is_open())
//        {
//            std::cout<<"open label file fail!"<<std::endl;
//            return;
//        }
        float *con_tmp = new float[features];
        std::string la;
        //std::cout<<"finish1"<<std::endl;
        int id=0;
        int label;
        for(int j=0;j<vertices;j++)
        {
            int size_0=features;
            output_ftr<<j<<" ";
                for (int i = 0; i < size_0; i++){
                    input_ftr >> con_tmp[i];
                    if(i!=(size_0-1)){
                        output_ftr<<con_tmp[i]<<" ";
                    }else{
                         output_ftr<<con_tmp[i]<<std::endl;
                    }
                }
//                input_lbl >> label;
//                output_lbl<<j<<" "<<label<<std::endl;
                
        }
        free(con_tmp);
        input_ftr.close();
//        input_lbl.close();
        output_ftr.close();
//        output_lbl.close();
    }
    
int main(int argc, char ** argv){
    
//for reddit
//sort_Mask(232965, "./redditdata/reddit_small/reddit.mask");
//sort_Label(232965, "./redditdata/reddit_small/reddit.labeltable");
generate_edge(232965,"./redditdata/reddit_small/reddit_full.edge.txt");
//writeFeature(232965,602,"./redditdata/reddit_small/reddit.featuretablenorm");

	return 0;
}

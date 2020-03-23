#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <string.h>
#include <cstring>
using namespace std;
int main(int argc, char ** argv){

int *buffer=new int[2];
FILE *fp,*fpwrite;
if(argc<3){
	printf("[filename] and [|trainset|]");
	exit(0);

}
int trainset=std::atoi(argv[2]);
std::string fw=std::string(argv[1]);
fw.append("_write").append(std::string(argv[2]));
fp=fopen(argv[1],"rb");
fpwrite=fopen(fw.c_str(),"wb");

int read_count=0;
fseek(fp,0L,SEEK_END);  
    int size=ftell(fp);  
 printf("filesize:\t%d\n",size);
fseek(fp, 0, SEEK_SET);
while(read_count<size){
	read_count+=sizeof(int)*2;
fread(buffer, sizeof(int)*2, 1, fp);
if(buffer[0]<trainset&&buffer[1]<trainset){
fwrite(buffer, sizeof(int)*2, 1,fpwrite);
printf("%d\t%d\n",buffer[0],buffer[1]);
}
}
fclose(fpwrite);
fclose(fp);



	return 0;
}

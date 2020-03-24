

//funcition of class gpu_processor:
gpu_processor::gpu_processor(int batch_size,int edge_size, 
													int feature_size,graph_type g_t){

		cudaMalloc(&src,sizeof(int)*(edge_size+1));
		cudaMalloc(&dst,sizeof(int)*(batch_size+1));
		cudaMalloc(&edge_data,sizeof(int)*edge_size*feature_size);

		cudaMalloc(&metaInfo,sizeof(int)*8);

		cudaMalloc(&old_feature,sizeof(float)*feature_size*batch_size);
		cudaMalloc(&new_feature,sizeof(float)*feature_size*batch_size);
		feature_buffer=new float[feature_size*batch_size];
		meta=new int [8];
		this->grapht=g_t;
	if(g_t==CSC){
		cudaMalloc(&dst,sizeof(int)*(batch_size+1));
		printf("CUDA_DEBUG::INFO\tGPU processor CSC constructed\n");

	}else if(g_t==PAIR){
		cudaMalloc(&dst,sizeof(int)*(edge_size+1));
		printf("CUDA_DEBUG::INFO\tGPU processor PAIR constructed\n");
	}
}
gpu_processor::gpu_processor(int r, int c,int count){
	cudaMalloc(&combined_grad,sizeof(float)*(r*c));
	cudaMalloc(&buffer,sizeof(float)*(c*count));
	this->r_=r;
	this->c_=c;
}
void gpu_processor::aggregate_grad(float *a,float *b, float *c, int count){
	const int BLOCK_SIZE=128;
	const int THREAD_SIZE=512;
		cudaMemset(&this->combined_grad, 0 ,sizeof(float)*this->r_*this->c_);
		cudaMemset(&this->buffer, 0 ,sizeof(float)*(this->c_*count));
		dot_mult_to_buffer<<<BLOCK_SIZE,THREAD_SIZE>>>(this->buffer,b,c,count,this->r_,this->c_);
		cudaDeviceSynchronize();
		time_mult_to_combine_grad1<<<BLOCK_SIZE,THREAD_SIZE>>>(this->combined_grad,a,this->buffer,count,this->r_,this->c_);
		cudaDeviceSynchronize();
}




float* gpu_processor::allocate_GPU(int size){
	float *data;
 cudaMalloc(&data,sizeof(char)*size);
return data;
}
void gpu_processor::resize_graph(int new_size){
	if(new_size>graph_size()){
		cudaFree(edge_data);
		cudaFree(dst);
		cudaFree(src);
		cudaMalloc(&src,sizeof(int)*(new_size+1));
		cudaMalloc(&dst,sizeof(int)*(new_size+1));
		cudaMalloc(&edge_data,sizeof(float)*new_size*feature_size());

	}
}
void gpu_processor::cp_data_from_gpu2cpu(float* dst, float*src, int size){
	cudaMemcpy(dst,src,size, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
}

//template <typename t_v,t_id>
void gpu_processor::setMetaInfo(
			int src_s,int src_e,
			int dst_s,int dst_e,
			int graph_size,int partition_s,
			int partition_e,int feature_size){
		meta[0]=src_s;meta[1]=src_e;
		meta[2]=dst_s;meta[3]=dst_e;
		meta[4]=graph_size;meta[5]=partition_s;
		meta[6]=partition_e;meta[7]=feature_size;
		cudaMemcpy(metaInfo,meta,sizeof(int)*8, cudaMemcpyHostToDevice);

}


void gpu_processor::load_data2GPU_csc(int * index, int* neighbour,
						float*edge_data,float* old_feature){
cudaMemcpy(this->dst,index,sizeof(int)*(this->batch_size()+1),cudaMemcpyHostToDevice);
cudaMemcpy(this->src,neighbour,sizeof(int)*this->graph_size(),cudaMemcpyHostToDevice);
cudaMemcpy(this->edge_data,edge_data,sizeof(float)*this->graph_size()*this->feature_size(),
												  cudaMemcpyHostToDevice);
cudaMemcpy(this->old_feature,old_feature,sizeof(float)*this->batch_size()*this->feature_size(),
												  cudaMemcpyHostToDevice);

}
void gpu_processor::load_data2GPU_pair(int * index, int* neighbour,
						float*edge_data,float* old_feature){
cudaMemcpy(this->dst,index,sizeof(int)*(this->graph_size()),cudaMemcpyHostToDevice);
cudaMemcpy(this->src,neighbour,sizeof(int)*this->graph_size(),cudaMemcpyHostToDevice);
cudaMemcpy(this->edge_data,edge_data,sizeof(float)*this->graph_size()*this->feature_size(),
												  cudaMemcpyHostToDevice);
cudaMemcpy(this->old_feature,old_feature,sizeof(float)*this->batch_size()*this->feature_size(),
												  cudaMemcpyHostToDevice);

}
void gpu_processor::init_for_newiteration(){
	 cudaMemset(&new_feature, 0 ,sizeof(float)*feature_size()*batch_size());
}
void gpu_processor::fetch_result_fromGPU( float* to_where){
	//here needbias operation;
	cudaMemcpy(to_where,this->new_feature,sizeof(float)*batch_size()*feature_size(),
																		cudaMemcpyDeviceToHost);	
}

void gpu_processor::fetch_result_fromGPU_async( float* to_where){
	//here needbias operation;
	cudaDeviceSynchronize();
	cudaMemcpy(to_where,this->new_feature,sizeof(float)*batch_size()*feature_size(),
																		cudaMemcpyDeviceToHost);	
}

void gpu_processor::run_sync(){
	const int BLOCK_SIZE=512;
	const int THREAD_SIZE=getThreadNum(this->feature_size());
	printf("CUDA_DEBUGE_INFO: RUN_SYNC with \t BLOCK_SIZE:%d\tTHREAD_SIZE:%d\n"); 
	if(this->grapht==CSC){
		process_edges_chunk<float,int><<<BLOCK_SIZE,THREAD_SIZE>>>(this->src, this->dst,this->metaInfo,
										           this->old_feature, this->new_feature,this->edge_data);
	}else if(this->grapht==PAIR){
		process_edges_chunk_one_by_one<float,int><<<BLOCK_SIZE,THREAD_SIZE>>>(this->src, this->dst,this->metaInfo,
										           this->old_feature, this->new_feature,this->edge_data);
	}
	cudaDeviceSynchronize();
}
void gpu_processor::run_async(){
	const int BLOCK_SIZE=512;
	const int THREAD_SIZE=getThreadNum(this->feature_size());
	printf("CUDA_DEBUGE_INFO:\t BLOCK_SIZE:%d\tTHREAD_SIZE:%d\n");
	if(grapht==CSC){
		process_edges_chunk<float,int><<<BLOCK_SIZE,THREAD_SIZE>>>(this->src, this->dst,this->metaInfo,
										           this->old_feature, this->new_feature,this->edge_data);
	}else if(grapht==PAIR){
		process_edges_chunk_one_by_one<float,int><<<BLOCK_SIZE,THREAD_SIZE>>>(this->src, this->dst,this->metaInfo,
										           this->old_feature, this->new_feature,this->edge_data);
	}
	
}

void gpu_processor::debug_all_info(){
show2<int><<<1,1>>>(this->src, this->dst,this->metaInfo, this->old_feature, this->new_feature,this->edge_data);
}
void gpu_processor::debug_new_feature(){
	show<int><<<1,1>>>(this->src, this->dst,this->metaInfo, this->old_feature, this->new_feature,this->edge_data);
}



void load_data_to_gpu(graphOnGPUBlock<float,int> *gp, int * index, int* neighbour,
						float*edge_data,float* old_feature){
	//here  may need bias operation
cudaMemcpy(gp->dst,index,sizeof(int)*(gp->batch_size()+1),cudaMemcpyHostToDevice);
cudaMemcpy(gp->src,neighbour,sizeof(int)*gp->graph_size(),cudaMemcpyHostToDevice);
cudaMemcpy(gp->edge_data,edge_data,sizeof(float)*gp->graph_size()*gp->feature_size(),cudaMemcpyHostToDevice);
cudaMemcpy(gp->old_feature,old_feature,sizeof(float)*gp->batch_size()*gp->feature_size(),cudaMemcpyHostToDevice);

}

void fetch_result_from_gpu(graphOnGPUBlock<float,int> *gp, float* to_where){
	//here needbias operation;
	cudaMemcpy(to_where,gp->new_feature,sizeof(float)*gp->batch_size()*gp->feature_size(),cudaMemcpyDeviceToHost);
}

void run(graphOnGPUBlock<float,int> *gp){
	int block_size=0;
	int Thread_Size=getThreadNum(gp->feature_size());
	process_edges_chunk<float,int><<<1,32>>>(gp->src, gp->dst,gp->metaInfo, gp->old_feature, gp->new_feature,gp->edge_data);
	cudaDeviceSynchronize();
}

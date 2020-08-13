#include "propagate.cuh"
#include "test.hpp"

#define CHECK_CUDA_RESULT(N) {											\
	CUresult result = N;												\
	if (result != 0) {													\
		printf("CUDA call on line %d returned error %d\n", __LINE__,	\
			result);													\
		exit(1);														\
	} }


void forward_on_GPU(float* input,float* output,float* weight_forward,//data 
        VertexId_CUDA* src,VertexId_CUDA *dst,//graph
        VertexId_CUDA src_start, VertexId_CUDA src_end,
        VertexId_CUDA dst_start, VertexId_CUDA dst_end,
	VertexId_CUDA edge_size,VertexId_CUDA batch_size,VertexId_CUDA feature_size){
	
	const int THREAD_SIZE=128;//getThreadNum(_meta->get_feature_size());
	const int BLOCK_SIZE=128;
	//printf("CUDA_DEBUGE_INFO:FORWARD RUN_SYNC with \t BLOCK_SIZE:%d\tTHREAD_SIZE:%d\n",BLOCK_SIZE,THREAD_SIZE); 
		processing_scalar_weight_shard<float,VertexId_CUDA><<<BLOCK_SIZE,THREAD_SIZE>>>(
			src, dst, input, output, weight_forward, 
				src_start, dst_start, edge_size, feature_size);
		cudaDeviceSynchronize();
}

void backward_on_GPU(float* input,float* output,float* weight_backward,//data 
        VertexId_CUDA* src,VertexId_CUDA *dst,//graph
        VertexId_CUDA src_start, VertexId_CUDA src_end,
        VertexId_CUDA dst_start, VertexId_CUDA dst_end,
	VertexId_CUDA edge_size,VertexId_CUDA batch_size,VertexId_CUDA feature_size){
	
	const int THREAD_SIZE=128;//getThreadNum(_meta->get_feature_size());
	const int BLOCK_SIZE=128;
	printf("CUDA_DEBUGE_INFO: BACKWARD RUN_SYNC with \t BLOCK_SIZE:%d\tTHREAD_SIZE:%d\n",BLOCK_SIZE,THREAD_SIZE); 
		processing_scalar_weight_shard<float,VertexId_CUDA><<<BLOCK_SIZE,THREAD_SIZE>>>(
			dst, src, input, output, weight_backward, 
				dst_start, src_start, edge_size, feature_size);
		cudaDeviceSynchronize();
}






COO_graph::COO_graph(VertexId_CUDA vertices,VertexId_CUDA edges){
	_vertices=vertices;
	_edges=edges;
}

void COO_graph::remalloc_src_on_gpu(size_t new_capacity){
	cudaMalloc(&_src,sizeof(VertexId_CUDA)*(new_capacity+1));
}

void COO_graph::remalloc_dst_on_gpu(size_t new_capacity){
	cudaMalloc(&_dst,sizeof(VertexId_CUDA)*(new_capacity+1));
}

void COO_graph::remalloc_dst_on_cpu(size_t new_capacity){
	_dst=new VertexId_CUDA[new_capacity+1];
}

void COO_graph::remalloc_src_on_cpu(size_t new_capacity){
	_src=new VertexId_CUDA[new_capacity+1];
}

void COO_graph::move_src_to_gpu(VertexId_CUDA* src_cpu, VertexId_CUDA edges){
	cudaMemcpy(_src,src_cpu,(edges+1)*(sizeof(VertexId_CUDA)), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
}

void COO_graph::move_dst_to_gpu(VertexId_CUDA* dst_cpu, VertexId_CUDA edges){
	cudaMemcpy(_dst,dst_cpu,(edges+1)*(sizeof(VertexId_CUDA)), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
}


CSC_graph::CSC_graph(VertexId_CUDA vertices,VertexId_CUDA edges,bool has_degree){
	_vertices=vertices;
	_edges=edges;
	_has_degree=has_degree;
	if(edges==39297)
	std::cout<<"new CSC_graph"<<std::endl;
}
void CSC_graph::remalloc_neighbour_on_gpu(size_t new_capacity, bool has_degree){
	
	cudaMalloc(&_neighbour,sizeof(VertexId_CUDA)*(new_capacity+1));
	_e_capacity=new_capacity;
}
void CSC_graph::remalloc_vertex_on_gpu(size_t new_capacity){
	
	cudaMalloc(&_index,sizeof(VertexId_CUDA)*(new_capacity+1));
	_v_capacity=new_capacity;
}

void CSC_graph::remalloc_neighbour_on_cpu(size_t new_capacity, bool has_degree){
	
	_neighbour=new VertexId_CUDA[new_capacity+1];
	_e_capacity=new_capacity;
}
void CSC_graph::remalloc_vertex_on_cpu(size_t new_capacity){
	
	_index=new VertexId_CUDA[new_capacity+1];
	_v_capacity=new_capacity;
}

VertexId_CUDA CSC_graph::start(VertexId_CUDA vertex){
	return _index[vertex]; 
}
VertexId_CUDA CSC_graph::end(VertexId_CUDA vertex){
	return _index[vertex+1];
}
VertexId_CUDA CSC_graph::neighbour(VertexId_CUDA index){
	return _neighbour[index];
}

void CSC_graph::move_vertices_to_gpu(VertexId_CUDA* v_cpu, VertexId_CUDA vertices){
	cudaMemcpy(_index,v_cpu,(vertices+1)*(sizeof(VertexId_CUDA)), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
}

void CSC_graph::move_neighbour_to_gpu(VertexId_CUDA* e_cpu, VertexId_CUDA edges){
	cudaMemcpy(_neighbour,e_cpu,(edges+1)*(sizeof(VertexId_CUDA)), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
}

void graph_engine::test_load_graph(){
	/*
	vertices:  6, edges: 10

0	1   f:|0   1  2  4      b:|0   3  4
0	2     |1   2              |1   0  3
0	4     |2   3              |2   0  1
1   2     |3   0  1           |3   2  5   
2   3     |4   0  5           |4   0
3   0     |5   3              |5   4
3   1
4   0
4   5
5   3
	*/
VertexId_CUDA* f_v=new VertexId_CUDA[7];
f_v[0]=0;f_v[1]=3;f_v[2]=4;f_v[3]=5;f_v[4]=7;f_v[5]=9;f_v[6]=10;
VertexId_CUDA* b_v=new VertexId_CUDA[7];
b_v[0]=0;b_v[1]=2;b_v[2]=4;b_v[3]=6;b_v[4]=8;b_v[5]=9;b_v[6]=10;
VertexId_CUDA* f_n=new VertexId_CUDA[11];
f_n[0]=1;f_n[1]=2;f_n[2]=4;f_n[3]=2;f_n[4]=3;
f_n[5]=0;f_n[6]=1;f_n[7]=0;f_n[8]=5;f_n[9]=3;
VertexId_CUDA* b_n=new VertexId_CUDA[11];
b_n[0]=3;b_n[1]=4;b_n[2]=0;b_n[3]=3;b_n[4]=0;
b_n[5]=1;b_n[6]=2;b_n[7]=5;b_n[8]=0;b_n[9]=4;

	_forward_graph= new CSC_graph(6,10,false);
	_backward_graph= new CSC_graph(6,10,false);

	_forward_graph->remalloc_vertex_on_gpu(6);
	_backward_graph->remalloc_vertex_on_gpu(6);

	_forward_graph->remalloc_neighbour_on_gpu(10,false);
	_backward_graph->remalloc_neighbour_on_gpu(10,false);

	_forward_graph->move_vertices_to_gpu(f_v,6);
	_backward_graph->move_vertices_to_gpu(b_v,6);

	_forward_graph->move_neighbour_to_gpu(f_n,10);
	_backward_graph->move_neighbour_to_gpu(b_n,10);

	_meta=new MetaInfo(0,6,0,6,1);


}

int graph_engine::getThreadNum(int num){
	int s=32;
while(s<num){
	s+=32;
}
return s;
}



/*Useless*/
void graph_engine::load_graph(CSC_graph* forward_graph,CSC_graph *backward_graph,MetaInfo *meta){
	_forward_graph=new CSC_graph(forward_graph->_vertices,
		forward_graph->_edges,forward_graph->_has_degree);
	_backward_graph=new CSC_graph(backward_graph->_vertices,
		backward_graph->_edges,backward_graph->_has_degree);

	_forward_graph->remalloc_vertex_on_gpu(_forward_graph->_vertices);
	_backward_graph->remalloc_vertex_on_gpu(_backward_graph->_vertices);

	_forward_graph->remalloc_neighbour_on_gpu(_forward_graph->_vertices,
		_forward_graph->_has_degree);
	_backward_graph->remalloc_neighbour_on_gpu(_backward_graph->_vertices,
		_backward_graph->_has_degree);	
	
	_forward_graph->move_vertices_to_gpu(forward_graph->_index,
		forward_graph->_vertices);
	_forward_graph->move_neighbour_to_gpu(forward_graph->_neighbour,
		forward_graph->_edges);
	_backward_graph->move_vertices_to_gpu(backward_graph->_index,
		forward_graph->_vertices);
	_backward_graph->move_neighbour_to_gpu(backward_graph->_neighbour,
		backward_graph->_edges);
	_meta=new MetaInfo(meta->_src_s,meta->_src_e,meta->_dst_s,meta->_dst_e,meta->_feature_size);

}




	/*load graph for CSC graph
	qiange wang*/
void graph_engine::load_graph(VertexId_CUDA f_vertices,VertexId_CUDA f_edges,VertexId_CUDA f_has_degree,
	VertexId_CUDA b_vertices,VertexId_CUDA b_edges,VertexId_CUDA b_has_degree,
	VertexId_CUDA* f_index,VertexId_CUDA* f_neighbour,VertexId_CUDA* b_index,VertexId_CUDA* b_neighbour,
	VertexId_CUDA _src_s,VertexId_CUDA _src_e,VertexId_CUDA _dst_s,VertexId_CUDA _dst_e,
	VertexId_CUDA _feature_size){

	//std::cout<<"&&&&&&&&&worker well"<<std::endl;	
	_forward_graph=new CSC_graph(f_vertices, f_edges, f_has_degree);
	//std::cout<<"&&&&&&&&&worker done"<<b_vertices<<" "<<b_edges<<" "<<b_has_degree<<std::endl;	
	_backward_graph=new CSC_graph(b_vertices, b_edges, b_has_degree);
	//if(_backward_graph->_edges==b_edges)
	//std::cout<<"---worker not   done"<<b_vertices<<" "<<b_edges<<" "<<b_has_degree<<std::endl;	
	_forward_graph->remalloc_vertex_on_gpu(f_vertices);
	_backward_graph->remalloc_vertex_on_gpu(b_vertices);

	_forward_graph->remalloc_neighbour_on_gpu(f_edges, f_has_degree);
	_backward_graph->remalloc_neighbour_on_gpu(b_edges, b_has_degree);	
	
	_forward_graph->move_vertices_to_gpu(f_index, f_vertices);
	_forward_graph->move_neighbour_to_gpu(f_neighbour, f_edges);

	_backward_graph->move_vertices_to_gpu(b_index, b_vertices);
	_backward_graph->move_neighbour_to_gpu(b_neighbour,	b_edges);

	_meta=new MetaInfo(_src_s, _src_e, _dst_s, _dst_e, _feature_size);

}
	/*load graph for COO graph
	qiange wang*/
void graph_engine::load_graph_for_COO(VertexId_CUDA f_vertices,VertexId_CUDA f_edges,VertexId_CUDA f_has_degree,
		VertexId_CUDA* f_src,VertexId_CUDA* f_dst,
		VertexId_CUDA _src_s,VertexId_CUDA _src_e,VertexId_CUDA _dst_s,VertexId_CUDA _dst_e,
		VertexId_CUDA _feature_size){
		 _graph_cuda=new COO_graph(f_vertices,f_edges);
		 
		 _graph_cuda->remalloc_src_on_gpu(f_edges);
		 _graph_cuda->remalloc_dst_on_gpu(f_edges);

		 _graph_cuda->move_dst_to_gpu(f_dst,f_edges);
		 _graph_cuda->move_src_to_gpu(f_src,f_edges);

		 _meta=new MetaInfo(_src_s, _src_e, _dst_s, _dst_e, _feature_size);

		}


	

void graph_engine::forward_one_step(float* input,float* output,float* weight,
		weight_type wt,VertexId_CUDA feature_size){
		this->redirect_input_output(input, output, weight, wt, feature_size);
		this->forward();   
}


void graph_engine::forward_one_step_COO(float* input,float* output,float* weight,
	weight_type wt,VertexId_CUDA feature_size){
	this->redirect_input_output(input, output, weight, wt, feature_size);
    this->forward_COO();   
}
void graph_engine::forward_one_step_COO_partition(float* input_partition,
		float* output_partition,float* weight_partition,
	weight_type wt,VertexId_CUDA feature_size, int partition_id){
		int dst_start=_graph_cuda->partition_offset[partition_id];
		int current_batch=_graph_cuda->partition_offset[partition_id+1]-
								_graph_cuda->partition_offset[partition_id];

		for(int i=0;i<_graph_cuda->partitions;i++){
			int current_batch=_graph_cuda->partition_offset[i+1]-
				_graph_cuda->partition_offset[i];
			int src_start=_graph_cuda->partition_offset[i];
			this->redirect_input_output(input_partition, output_partition, weight_partition, wt, feature_size);
			this->forward_COO();   
		}
}



void graph_engine::backward_one_step(float* input,float* output,float* weight,
	weight_type wt,VertexId_CUDA feature_size){
	this->redirect_input_output(input, output, weight, wt, feature_size);
    this->backward();   
}
void graph_engine::backward_one_step_COO(float* input,float* output,float* weight,
	weight_type wt,VertexId_CUDA feature_size){
	this->redirect_input_output(input, output, weight, wt, feature_size);
    this->backward_COO();   
}


void graph_engine::forward(){
	
	const int THREAD_SIZE=128;//getThreadNum(_meta->get_feature_size());
	const int BLOCK_SIZE=128;
	//_wt=NULL_TYPE;
	cudaMemset(_output,0,sizeof(float)*_meta->get_batch()*_meta->get_feature_size());
	printf("CUDA_DEBUGE_INFO: RUN_SYNC with \t BLOCK_SIZE:%d\tTHREAD_SIZE:%d\n",BLOCK_SIZE,THREAD_SIZE); 
	if(_wt==NULL_TYPE){
	//	printf("NULL_TYPE\t %d %d %d \n",_meta->get_batch(),_meta->batch_start_vertex(),_meta->get_feature_size());
		processing_no_weight<float,VertexId_CUDA><<<BLOCK_SIZE,THREAD_SIZE>>>(
			_forward_graph->_neighbour, _forward_graph->_index, _input, _output,_with_weight 
			    ,_meta->get_batch(),_meta->batch_start_vertex(),_meta->get_feature_size());

	}else if(_wt==SCALA_TYPE){
		//printf("SCALA_TYPE\n");
		//std::cout<<_meta->get_batch()<<" "<<_meta->batch_start_vertex()<<" "<<_meta->get_feature_size()<<std::endl;
		processing_scalar_weight<float,VertexId_CUDA><<<BLOCK_SIZE,THREAD_SIZE>>>(
			_forward_graph->_neighbour, _forward_graph->_index, _input, _output,_with_weight 
				,_meta->get_batch(),_meta->batch_start_vertex(),_meta->get_feature_size());
	}else if(_wt==TENSOR_TYPE){
		;
	}
		cudaDeviceSynchronize();
	//	printf("finish\n");
}
void graph_engine::forward_COO(){
	
	const int THREAD_SIZE=128;//getThreadNum(_meta->get_feature_size());
	const int BLOCK_SIZE=128;
	//_wt=NULL_TYPE;
	cudaMemset(_output,0,sizeof(float)*_meta->get_batch()*_meta->get_feature_size());
	printf("CUDA_DEBUGE_INFO: RUN_SYNC with \t BLOCK_SIZE:%d\tTHREAD_SIZE:%d\n",BLOCK_SIZE,THREAD_SIZE); 
	if(_wt==NULL_TYPE){
	//	printf("NULL_TYPE\t %d %d %d \n",_meta->get_batch(),_meta->batch_start_vertex(),_meta->get_feature_size());
		processing_no_weight_one_by_one<float,VertexId_CUDA><<<BLOCK_SIZE,THREAD_SIZE>>>(
			_graph_cuda->_dst, _graph_cuda->_src, _input, _output,_with_weight 
				,_graph_cuda->_edges,0,0,
					_meta->get_feature_size());

	}else if(_wt==SCALA_TYPE){
		printf("SCALA_TYPE COO %d\n",_graph_cuda->_edges);
		processing_scalar_weight_one_by_one<float,VertexId_CUDA><<<BLOCK_SIZE,THREAD_SIZE>>>(
			_graph_cuda->_dst, _graph_cuda->_src, _input, _output,_with_weight 
				,(int)(_graph_cuda->_edges),_meta->batch_start_vertex(),_meta->batch_start_vertex(),
					_meta->get_feature_size());
	}else if(_wt==TENSOR_TYPE){
		;
	}
		cudaDeviceSynchronize();
	//	printf("finish\n");
}


void graph_engine::backward(){
	const int BLOCK_SIZE=128;
	const int THREAD_SIZE=128;
	cudaMemset(_output,0,sizeof(float)*_meta->get_batch()*_meta->get_feature_size());
	if(_wt==NULL_TYPE){
		printf("backward null_type\n");
		processing_no_weight<float,VertexId_CUDA><<<BLOCK_SIZE,THREAD_SIZE>>>(
			_backward_graph->_neighbour, _backward_graph->_index, _input, _output,_with_weight 
			    ,_meta->get_batch(),_meta->batch_start_vertex(),_meta->get_feature_size());

	}else if(_wt==SCALA_TYPE){
		printf("backward SCALA_TYPE\n");
		processing_scalar_weight<float,VertexId_CUDA><<<BLOCK_SIZE,THREAD_SIZE>>>(
			_backward_graph->_neighbour, _backward_graph->_index, _input, _output,_with_weight 
			    ,_meta->get_batch(),_meta->batch_start_vertex(),_meta->get_feature_size());
	}else if(_wt==TENSOR_TYPE){
		;
	}
		cudaDeviceSynchronize();
}
void graph_engine::backward_COO(){
	const int BLOCK_SIZE=128;
	const int THREAD_SIZE=128;
	cudaMemset(_output,0,sizeof(float)*_meta->get_batch()*_meta->get_feature_size());
	if(_wt==NULL_TYPE){
		printf("backward null_type\n");
		processing_no_weight_one_by_one<float,VertexId_CUDA><<<BLOCK_SIZE,THREAD_SIZE>>>(
			_graph_cuda->_src, _graph_cuda->_dst, _input, _output,_with_weight 
				,_meta->get_batch(),_meta->batch_start_vertex(),_meta->batch_start_vertex(),
					_meta->get_feature_size());

	}else if(_wt==SCALA_TYPE){
		printf("backward SCALA_TYPE COO %d %d\n",_graph_cuda->_edges,_meta->batch_start_vertex());
		processing_scalar_weight_one_by_one<float,VertexId_CUDA><<<BLOCK_SIZE,THREAD_SIZE>>>(
			_graph_cuda->_src, _graph_cuda->_dst, _input, _output,_with_weight,
			(int)(_graph_cuda->_edges),_meta->batch_start_vertex(),_meta->batch_start_vertex(),
					_meta->get_feature_size());
	}else if(_wt==TENSOR_TYPE){
		;
	}
		cudaDeviceSynchronize();
}


void graph_engine::set_input(float* input){
	_input=input;
}
void graph_engine::set_output(float* output){
	_output=output;
}
float* graph_engine::get_output(){
	return _output;
}
float* graph_engine::get_input(){
	return _input;
}
void graph_engine::set_weight(float* weight,weight_type wt){
	if(NULL_TYPE!=wt){
		_with_weight=weight;
	}
	_wt=wt;
}
void graph_engine::redirect_input_output(float *input,float *output,
	float* weight,weight_type wt,VertexId_CUDA feature_size){
	_input=input;
	_output=output;
	if(NULL_TYPE!=wt){
		_with_weight=weight;
	}
	_wt=wt;
	_meta->_feature_size=feature_size;

}
void graph_engine::init_cuda_stream(){
	cudaStreamCreate((cudaStream_t*)_cuda_stream);
}
void graph_engine::show(){
	
	show_dot_mult_to_buffer<<<1,1>>>(_input, 1, 6);
	cudaDeviceSynchronize();
	cudaMemset(_output,0,sizeof(float)*6);
}


void aggregate_engine::aggregate_grad(){
	const int BLOCK_SIZE=256;
	const int THREAD_SIZE=256;
//	cudaMemset(final_gradient,0,sizeof(float)*size_remote(1)*size_local(1));
	if(wt_==TENSOR_TYPE){
	//printf("%d %d %d  %d\n",size_remote(0),size_remote(1),size_local(0),size_local(1));
		dot_mult_tensor_weight<<<BLOCK_SIZE,THREAD_SIZE>>>(intermediate_gradient,
			remote_grad,weight,size_remote(0),size_remote(1));
	//	printf("finish dot\n");
		cudaDeviceSynchronize();
		//show_dot_mult_to_buffer<<<1,1>>>(intermediate_gradient, 3, 2);
		
		time_mult1<<<BLOCK_SIZE,THREAD_SIZE>>>(final_gradient,
			local_grad,intermediate_gradient,size_remote(0),size_remote(1),size_local(1));

	}else if(wt_==SCALA_TYPE){
			
		dot_mult_scalar_weight<<<BLOCK_SIZE,THREAD_SIZE>>>(intermediate_gradient,
			remote_grad,weight,size_remote(0),size_remote(1));
		cudaDeviceSynchronize();
		time_mult1<<<BLOCK_SIZE,THREAD_SIZE>>>(final_gradient,
			local_grad,intermediate_gradient,size_remote(0),size_remote(1),size_local(1));
		cudaDeviceSynchronize();
	}
	else{;
		time_mult1<<<BLOCK_SIZE,THREAD_SIZE>>>(final_gradient,
			local_grad,remote_grad,size_remote(0),size_remote(1),size_local(1));
		cudaDeviceSynchronize();		
	}
}


void aggregate_engine::init_intermediate_gradient(){
	//cudaMalloc(&_index,sizeof(size_t)*(new_capacity+1));
	printf("Memory allocate_intermediate_GPU:%d  %d\n",size_remote(0),size_remote(1));
	cudaMalloc(&intermediate_gradient,sizeof(float)*size_remote(0)*size_remote(1));
}

void aggregate_engine::init_final_gradient(){
	//cudaMalloc(&_index,sizeof(size_t)*(new_capacity+1));
	printf("Memory allocate_final_GPU:%d  %d\n",size_local(1),size_remote(1));
	cudaMalloc(&final_gradient,sizeof(float)*size_remote(1)*size_local(1));
}

void aggregate_engine::close_intermediate_gradient(){
	cudaFree(intermediate_gradient);
}








/*

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





int test(){
	printf("hello world\n");
	int size=4;
	float* s= new float[6];
	float* d= new float[6];
	float* g= new float[6];
	float *f= new float[16];
	float *e= new float[24];
	float *result= new float[16];
	g[0]=1;g[1]=1;g[2]=1;g[3]=1;g[4]=1;g[5]=1;
	s[0]=1;s[1]=2;s[2]=3;s[3]=4;s[4]=5;s[5]=6;
	d[0]=1;d[1]=2;d[2]=3;d[3]=4;d[4]=5;d[5]=6;
	float *agg,*s_,*d_,*g_;
	float *buffer;
	cudaMalloc(&agg,sizeof(float)*6*6);
	cudaMalloc(&s_,sizeof(float)*6);
	cudaMalloc(&d_,sizeof(float)*6);
	cudaMalloc(&g_,sizeof(float)*6);
	cudaMalloc(&buffer,sizeof(float)*6);
	cudaMemcpy(s_,s,sizeof(float)*(6),cudaMemcpyHostToDevice);
	cudaMemcpy(d_,d,sizeof(float)*(6),cudaMemcpyHostToDevice);
	cudaMemcpy(g_,g,sizeof(float)*(6),cudaMemcpyHostToDevice);
	printf("hello world1\n");
	dot_mult_to_buffer<<<1,1>>>(buffer, s_, d_, 3, 2, 2);
	cudaDeviceSynchronize();
	show_dot_mult_to_buffer<<<1,1>>>(buffer, 3, 2);
	time_mult_to_combine_grad1<<<5,32>>>(agg,g_,buffer,3,2,2);
	printf("hello world2\n");
	cudaDeviceSynchronize();
	printf("\n");
    show_dot_mult_to_buffer<<<1,1>>>(agg, 2, 2);
    printf("hello world3\n");
	cudaDeviceSynchronize();


    aggregate_engine* ag=new aggregate_engine();
    ag->reconfig_data(3,2,3,2,NULL_TYPE);
    ag->redirect_input_output(s_, d_, g_, agg);
    ag->init_intermediate_gradient();
    ag->aggregate_grad();
	 show_dot_mult_to_buffer<<<1,1>>>(agg, 2, 2);
    printf("hello world4\n");

    graph_engine* gre=new graph_engine();
    gre->test_load_graph();
    gre->redirect_input_output(s_, agg, g_, NULL_TYPE,2);
    gre->show();
	gre->backward();
	show_dot_mult_to_buffer<<<1,1>>>(agg, 1, 6);



cudaDeviceSynchronize();
cudaDeviceReset();
	return 0;
}
*/
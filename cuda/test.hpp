/*
 * test.h
 *
 *  Created on: Dec 3, 2019
 *      Author: wangqg
 */
#include "cuda_runtime.h"
#ifndef TEST_HPP
#define TEST_HPP
#include <iostream>
#include <stdio.h>
#include <vector>
#include <string.h>
//#include"../core/graph.hpp"
typedef uint32_t VertexId_CUDA;
enum graph_type
{
	CSR,
	CSC,
	PAIR
};
enum weight_type
{
	NULL_TYPE,
	SCALA_TYPE,
	TENSOR_TYPE
};
struct MetaInfo
{
public:
	MetaInfo(VertexId_CUDA src_s, VertexId_CUDA src_e, VertexId_CUDA dst_s, VertexId_CUDA dst_e, VertexId_CUDA feature_size)
	{
		_src_s = src_s;
		_src_e = src_e;
		_dst_s = dst_s;
		_dst_e = dst_e;
		_feature_size = feature_size;
	}
	MetaInfo() {}
	VertexId_CUDA _src_s;
	VertexId_CUDA _src_e;
	VertexId_CUDA _dst_s;
	VertexId_CUDA _dst_e;
	VertexId_CUDA _feature_size;
	VertexId_CUDA batch_start_vertex()
	{
		return _src_s;
	}
	VertexId_CUDA get_feature_size()
	{
		return _feature_size;
	}
	VertexId_CUDA get_batch()
	{
		return _dst_e - _dst_s;
	}
};

class CSC_graph
{
public:
	CSC_graph(VertexId_CUDA vertices, VertexId_CUDA edges, bool has_degree);
	CSC_graph() { ; }
	void remalloc_neighbour_on_gpu(size_t new_capacity, bool has_degree);
	void remalloc_vertex_on_gpu(size_t new_capacity);
	void remalloc_neighbour_on_cpu(size_t new_capacity, bool has_degree);
	void remalloc_vertex_on_cpu(size_t new_capacity);
	VertexId_CUDA start(VertexId_CUDA vertex);
	VertexId_CUDA end(VertexId_CUDA vertex);
	VertexId_CUDA neighbour(VertexId_CUDA index);
	void move_vertices_to_gpu(VertexId_CUDA *v_cpu, VertexId_CUDA vertices);
	void move_neighbour_to_gpu(VertexId_CUDA *e_cpu, VertexId_CUDA edges);
	void set_index_neighbour(VertexId_CUDA *index, VertexId_CUDA *neighbour)
	{
		_index = index;
		_neighbour = neighbour;
	}
	size_t _vertices;
	size_t _edges;
	VertexId_CUDA *_index;
	VertexId_CUDA *_neighbour;
	size_t _v_capacity;
	size_t _e_capacity;
	bool _has_degree;
	bool _on_gpu;
};

class COO_graph
{
public:
	COO_graph(VertexId_CUDA vertices, VertexId_CUDA edges);
	COO_graph() { ; }
	void remalloc_src_on_gpu(size_t new_capacity);
	void remalloc_dst_on_gpu(size_t new_capacity);
	void remalloc_src_on_cpu(size_t new_capacity);
	void remalloc_dst_on_cpu(size_t new_capacity);
	void from_CSC_to_COO(VertexId_CUDA *index, VertexId_CUDA *neighbour, VertexId_CUDA vertices);
	void move_src_to_gpu(VertexId_CUDA *v_cpu, VertexId_CUDA edges);
	void move_dst_to_gpu(VertexId_CUDA *e_cpu, VertexId_CUDA edges);
	void set_src_dst(VertexId_CUDA *src, VertexId_CUDA *dst)
	{
		_src = src;
		_dst = dst;
	}
	void init_partitions(int partitions_, int *partition_offset_, int partition_id_)
	{
		partitions = partitions_;
		std::cout << "init_partitions" << std::endl;
		partition_offset = partition_offset_;
		std::cout << "init_partitions1" << std::endl;
		partition_id = partition_id_;
	}
	size_t _vertices;
	size_t _edges;
	VertexId_CUDA *_src;
	VertexId_CUDA *_src_cpu;
	VertexId_CUDA *_dst;
	VertexId_CUDA *_dst_cpu;
	size_t _v_capacity;
	size_t _e_capacity;
	bool _on_gpu;
	int partitions;
	int *partition_offset;
	int partition_id;
};
void ResetDevice();

class Cuda_Stream
{
public:
	Cuda_Stream();
	void destory_Stream();
	cudaStream_t getStream();
	void CUDA_DEVICE_SYNCHRONIZE();
	cudaStream_t stream;
	void move_result_out(float *output, float *input, VertexId_CUDA src, VertexId_CUDA dst, int feature_size, bool sync = true);
	void move_data_in(float *d_pointer, float *h_pointer, VertexId_CUDA start, VertexId_CUDA end, int feature_size, bool sync = true);
	void move_edge_in(VertexId_CUDA *d_pointer, VertexId_CUDA *h_pointer, VertexId_CUDA start, VertexId_CUDA end, int feature_size, bool sync = true);
	void aggregate_comm_result(float *aggregate_buffer, float *input_buffer, VertexId_CUDA data_size, int feature_size, int partition_offset, bool sync = true);
        void deSerializeToGPU(float *input_gpu_buffer, float *input_buffer, VertexId_CUDA data_size, VertexId_CUDA feature_size, VertexId_CUDA partition_start, VertexId_CUDA partition_end, bool sync);
	void aggregate_comm_result_debug(float *aggregate_buffer, float *input_buffer, VertexId_CUDA data_size, VertexId_CUDA feature_size, VertexId_CUDA partition_start, VertexId_CUDA partition_end, bool sync);

	void Gather_By_Dst_From_Src(float *input, float *output, float *weight_forward, //data
							VertexId_CUDA* row_indices,VertexId_CUDA *column_offset,				//graph
							VertexId_CUDA src_start, VertexId_CUDA src_end,
							VertexId_CUDA dst_start, VertexId_CUDA dst_end,
							VertexId_CUDA edges, VertexId_CUDA batch_size,
							VertexId_CUDA feature_size, bool with_weight = false,bool tensor_weight=false);
        void Gather_By_Src_From_Dst(float* input,float* output,float* weight_forward,//data 
                                                        VertexId_CUDA* row_offset,VertexId_CUDA *column_indices,//graph
                                                        VertexId_CUDA src_start, VertexId_CUDA src_end,
                                                        VertexId_CUDA dst_start, VertexId_CUDA dst_end,
                                                        VertexId_CUDA edges,VertexId_CUDA batch_size,
                                                        VertexId_CUDA feature_size,bool with_weight=false,bool tensor_weight=false);
        void Gather_By_Dst_From_Message(float *input, float *output, float *weight_forward, //data
							VertexId_CUDA *src, VertexId_CUDA *dst,				//graph
							VertexId_CUDA src_start, VertexId_CUDA src_end,
							VertexId_CUDA dst_start, VertexId_CUDA dst_end,
							VertexId_CUDA edges, VertexId_CUDA batch_size,
							VertexId_CUDA feature_size, bool with_weight = false,bool tensor_weight=false);
	void Gather_By_Dst_From_Src_Para(float *input, float *output, float *para_forward, //data
							VertexId_CUDA *src, VertexId_CUDA *dst,				//graph
							VertexId_CUDA src_start, VertexId_CUDA src_end,
							VertexId_CUDA dst_start, VertexId_CUDA dst_end,
							VertexId_CUDA edges, VertexId_CUDA batch_size,
							VertexId_CUDA feature_size, bool sync = true);
        void Scatter_Grad_Back_To_Weight(float* input,float* output_grad,float* weight_grad,//data 
                                                        long* src,long *dst,//graph
                                                        VertexId_CUDA src_start, VertexId_CUDA src_end,
                                                        VertexId_CUDA dst_start, VertexId_CUDA dst_end,
                                                        VertexId_CUDA edges,VertexId_CUDA batch_size,
                                                        VertexId_CUDA feature_size,bool tensor_weight=true);
        void Scatter_Grad_Back_To_Message(float* input,float* message_grad,//data 
                                                        VertexId_CUDA* row_indices,VertexId_CUDA *column_offset,
                                                        VertexId_CUDA src_start, VertexId_CUDA src_end,
                                                        VertexId_CUDA dst_start, VertexId_CUDA dst_end,
                                                        VertexId_CUDA edges,VertexId_CUDA batch_size,
                                                        VertexId_CUDA feature_size,bool with_weight=true);
//	void Gather_By_Dst_From_Src_shrink(float *input, float *output, float *weight_forward, //data
//								   VertexId_CUDA *src, VertexId_CUDA *dst,			   //graph
//								   VertexId_CUDA *index_gpu_buffer, VertexId_CUDA *vertex_gpu_buffer,
//								   VertexId_CUDA src_start, VertexId_CUDA src_end,
//								   VertexId_CUDA dst_start, VertexId_CUDA dst_end,
//								   VertexId_CUDA actual_dst_start, VertexId_CUDA batch_size,
//								   VertexId_CUDA feature_size, bool sync = true);
	void process_local(float *local_buffer, float *input_tensor,
					   VertexId_CUDA *src, VertexId_CUDA *dst,
					   VertexId_CUDA *src_index, float *weight_buffer,
					   int dst_offset, int dst_offset_end, int feature_size, int edge_size, bool sync = true);
	void process_local_inter(float *local_buffer, float *input_tensor,
							 VertexId_CUDA *src, VertexId_CUDA *dst,
							 VertexId_CUDA *src_index, VertexId_CUDA *dst_index, float *weight_buffer,
							 int dst_offset, int dst_offset_end, int feature_size, int edge_size, int out_put_buffer_size, bool sync = true);
	void process_local_inter_para(float *local_buffer, float *input_tensor,
							 VertexId_CUDA *src, VertexId_CUDA *dst,
							 VertexId_CUDA *src_index, VertexId_CUDA *dst_index, float *para,
							 int dst_offset, int dst_offset_end, int feature_size, int edge_size, int out_put_buffer_size, bool sync = true);
};

void *cudaMallocPinned(long size_of_bytes);
void *getDevicePointer(void *host_data_to_device);
void *cudaMallocGPU(long size_of_bytes);

void forward_on_GPU(float *input, float *output, float *weight_forward, //data
					VertexId_CUDA *src, VertexId_CUDA *dst,				//graph
					VertexId_CUDA src_start, VertexId_CUDA src_end,
					VertexId_CUDA dst_start, VertexId_CUDA dst_end,
					VertexId_CUDA edges, VertexId_CUDA batch_size, VertexId_CUDA feature_size);
void Gather_By_Dst_From_Src(float *input, float *output, float *weight_forward, //data
						VertexId_CUDA *src, VertexId_CUDA *dst,				//graph
						VertexId_CUDA src_start, VertexId_CUDA src_end,
						VertexId_CUDA dst_start, VertexId_CUDA dst_end,
						VertexId_CUDA edges, VertexId_CUDA batch_size,
						VertexId_CUDA feature_size, bool sync = true);

void backward_on_GPU(float *input, float *output, float *weight_forward, //data
					 VertexId_CUDA *src, VertexId_CUDA *dst,			 //graph
					 VertexId_CUDA src_start, VertexId_CUDA src_end,
					 VertexId_CUDA dst_start, VertexId_CUDA dst_end,
					 VertexId_CUDA edges, VertexId_CUDA batch_size, VertexId_CUDA feature_size);

void move_result_out(float *output, float *input, int src, int dst, int feature_size, bool sync = true);

void move_data_in(float *d_pointer, float *h_pointer, int start, int end, int feature_size, bool sync = true);

void move_edge_in(VertexId_CUDA *d_pointer, VertexId_CUDA *h_pointer, VertexId_CUDA start, VertexId_CUDA end, int feature_size, bool sync = true);

void allocate_gpu_buffer(float **input, int size);
void allocate_gpu_edge(VertexId_CUDA **input, int size);

void aggregate_comm_result(float *aggregate_buffer, float *input_buffer, int data_size, int feature_size, int partition_offset, bool sync = true);

void FreeBuffer(float *buffer);
void FreeEdge(VertexId_CUDA *buffer);
void zero_buffer(float *buffer, int size);

void CUDA_DEVICE_SYNCHRONIZE();

class graph_engine
{

public:
	graph_engine()
	{
	}
	int getThreadNum(int num);
	void forward();
	void backward();
	void forward_COO();
	void backward_COO();
	void set_input(float *input);
	void set_output(float *output);
	void set_weight(float *weight, weight_type wt);
	float *get_output();
	float *get_input();
	//	CSC* forward_graph;
	//  CSC* backward_graph;
	void init_cuda_stream();

	void load_and_processing_1by1(float *input, float *output, float *weight, weight_type wt, VertexId_CUDA feature_size);

	void forward_one_step(float *input, float *output, float *weight, weight_type wt, VertexId_CUDA feature_size);

	void backward_one_step(float *input, float *output, float *weight, weight_type wt, VertexId_CUDA feature_size);

	void redirect_input_output(float *input, float *output, float *weight, weight_type wt, VertexId_CUDA feature_size);

	void forward_one_step_COO(float *input, float *output, float *weight,
							  weight_type wt, VertexId_CUDA feature_size);

	void forward_one_step_COO_partition(float *input_partition,
										float *output_partition, float *weight_partition,
										weight_type wt, VertexId_CUDA feature_size, int partition_id);

	void backward_one_step_COO(float *input, float *output, float *weight,
							   weight_type wt, VertexId_CUDA feature_size);

	/*Useless*/
	void load_graph(CSC_graph *forward_graph, CSC_graph *backward_graph, MetaInfo *meta);

	/*load graph for CSC graph
	qiange wang*/
	void load_graph(VertexId_CUDA f_vertices, VertexId_CUDA f_edges, VertexId_CUDA f_has_degree,
					VertexId_CUDA b_vertices, VertexId_CUDA b_edges, VertexId_CUDA b_has_degree,
					VertexId_CUDA *f_index, VertexId_CUDA *f_neighbour, VertexId_CUDA *b_index, VertexId_CUDA *b_neighbour,
					VertexId_CUDA _src_s, VertexId_CUDA _src_e, VertexId_CUDA _dst_s, VertexId_CUDA _dst_e, VertexId_CUDA _feature_size);

	/*load graph for COO graph
	qiange wang*/
	void load_graph_for_COO(VertexId_CUDA f_vertices, VertexId_CUDA f_edges, VertexId_CUDA f_has_degree,
							VertexId_CUDA *f_src, VertexId_CUDA *f_dst,
							VertexId_CUDA _src_s, VertexId_CUDA _src_e, VertexId_CUDA _dst_s, VertexId_CUDA _dst_e, VertexId_CUDA _feature_size);

	void load_graph_shards(VertexId_CUDA f_vertices, VertexId_CUDA f_edges, VertexId_CUDA f_has_degree,
						   std::vector<COO_graph>, VertexId_CUDA _feature_size);

	/*init graph for coo graph
	*/
	void init_graph_for_1by1(VertexId_CUDA f_vertices, VertexId_CUDA f_edges, VertexId_CUDA f_has_degree,
							 VertexId_CUDA b_vertices, VertexId_CUDA b_edges, VertexId_CUDA b_has_degree,
							 VertexId_CUDA *f_index, VertexId_CUDA *f_neighbour, VertexId_CUDA *b_index, VertexId_CUDA *b_neighbour);

	void test_load_graph();

	void show();

	float *_input;
	float *_output;
	float *_with_weight;
	MetaInfo *_meta;
	CSC_graph *_forward_graph;
	CSC_graph *_backward_graph;

	CSC_graph *_forward_CPU;
	CSC_graph *_backward_CPU;

	COO_graph *_graph_CPU;
	COO_graph *_graph_cuda;

	void *_cuda_stream;
	int THREAD_SIZE = -1;
	const int BLOCK_SIZE = 512;
	double overall_time;
	weight_type _wt;
};

class aggregate_engine
{
public:
	aggregate_engine()
	{
		;
	}
	void reconfig_data(size_t r0, size_t c0, size_t r1, size_t c1, weight_type wt)
	{
		size_0_message = r0;
		size_1_message = c0;
		size_0_local = r1;
		size_1_local = c1;
		wt_ = wt;
	}
	void redirect_input_output(float *msg, float *w, float *local, float *output)
	{
		remote_grad = msg;
		local_grad = local;
		final_gradient = output;
		if (wt_ != NULL_TYPE)
		{
			weight = w;
		}
	}
	void redirect_input_output(float *msg, float *w, float *local)
	{
		remote_grad = msg;
		local_grad = local;
		final_gradient = NULL;
		if (wt_ != NULL_TYPE)
		{
			weight = w;
		}
	}
	void init_intermediate_gradient();
	void init_final_gradient();
	void close_intermediate_gradient();

	size_t size_remote(size_t i)
	{
		if (0 == i)
			return size_0_message;
		else
			return size_1_message;
	}
	size_t size_local(size_t i)
	{
		if (0 == i)
			return size_0_local;
		else
			return size_1_local;
	}

	void aggregate_grad();

	float *get_grad()
	{
		return final_gradient;
	}

private:
	float *remote_grad;
	float *weight;
	float *local_grad;
	float *intermediate_gradient;
	float *final_gradient;

	size_t size_0_message;
	size_t size_1_message;
	size_t size_0_local;
	size_t size_1_local;
	weight_type wt_;
};

class gpu_processor
{
public:
	gpu_processor(int batch_size, int edge_size, int feature_size, graph_type g_t);
	void setMetaInfo(
		int src_s, int src_e,
		int dst_s, int dst_e,
		int graph_size, int partition_s,
		int partition_e, int feature_size);
	gpu_processor()
	{
		;
	}
	gpu_processor(int w, int c, int count);

	void aggregate_grad(float *a, float *b, float *c, int count);

	inline int src_s()
	{
		return meta[0];
	}
	inline int src_e()
	{
		return meta[1];
	}
	inline int dst_s()
	{
		return meta[2];
	}
	inline int dst_e()
	{
		return meta[3];
	}
	inline int graph_size()
	{
		return meta[4];
	}
	inline int partition_s()
	{
		return meta[5];
	}
	inline int partition_e()
	{
		return meta[6];
	}
	inline int feature_size()
	{
		return meta[7];
	}
	inline int batch_size()
	{
		return dst_e() - dst_s();
	}
	float *allocate_GPU(int size);
	void cp_data_from_gpu2cpu(float *dst, float *src, int size);
	void init_for_newiteration();
	void resize_graph(int new_size);

	void load_data2GPU_csc(int *index, int *neighbour,
						   float *edge_data, float *old_feature);
	void load_data2GPU_pair(int *index, int *neighbour,
							float *edge_data, float *old_feature);
	void fetch_result_fromGPU(float *to_where);
	void fetch_result_fromGPU_async(float *to_where);
	void run_sync();
	void run_async();

	void debug_all_info();
	void debug_new_feature();
	float *feature_buffer;
	float *get_grad()
	{
		return combined_grad;
	}

private:
	int *src;	   // neighboors  inside a batch
	int *dst;	   //neighboor index  inside a batch
	int *metaInfo; // contatins all meta information . stores on GPU
	int *meta;	   // contatins all meta information . stores on CPU
	float *old_feature;
	float *new_feature;
	float *edge_data;
	int grapht;
	float *combined_grad;
	float *buffer;
	int r_;
	int c_;
};
//int test();
#endif /* TEST_H_ */

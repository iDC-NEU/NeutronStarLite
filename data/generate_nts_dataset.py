# -*- coding: utf-8 -*-
# @Author   : Sanzo00
# @Email    : arrangeman@163.com
# @Time     : 2022/06/17 09:00

import os
import sys
import argparse
import numpy as np
import networkx as nx
import time
import torch
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from dgl.data import CoraFullDataset, CoauthorCSDataset, CoauthorPhysicsDataset
from dgl.data import AmazonCoBuyComputerDataset, AmazonCoBuyPhotoDataset
from ogb.nodeproppred import DglNodePropPredDataset

def extract_dataset(args):
    dataset = args.dataset

    # change dir
    if not os.path.exists(dataset):
        os.mkdir(dataset)
    os.chdir(dataset)

    if dataset in ['cora', 'citeseer', 'pubmed', 'reddit']:
        # load dataset
        data = load_data(args)
        graph = data[0]
        
        features = graph.ndata['feat']
        labels = graph.ndata['label']
        # assert(features.size(0) == len(labels))
        train_mask = graph.ndata['train_mask']
        val_mask = graph.ndata['val_mask']
        test_mask = graph.ndata['test_mask']

        if args.self_loop:
            time_stamp = time.time()
            print('before add self loop has {} edges'.format(graph.num_edges()))
            graph = dgl.remove_self_loop(graph)
            graph = dgl.add_self_loop(graph)
            # graph = dgl.to_bidirected(graph) # simple graph
            print('after add self loop has {} edges'.format(graph.num_edges()))
            print("insert self loop cost {:.2f}s".format(time.time() - time_stamp))

        edges = graph.edges()
        edge_src = edges[0].numpy().reshape((-1,1))
        edge_dst = edges[1].numpy().reshape((-1,1))
        edges_list = np.hstack((edge_src, edge_dst))

        # if args.self_loop:
        #     time_stamp = time.time()
        #     edges_list = insert_self_loop(edges_list)
        #     print("insert self loop cost {:.2f} s".format(time.time() - time_stamp))

        print("nodes: {}, edges: {}, feature dims: {}, classess: {}, label nodes: {}"
              .format(graph.number_of_nodes(), edges_list.shape, 
              list(features.shape), len(np.unique(labels)),
              train_mask.sum() + test_mask.sum() + val_mask.sum()))      
        return edges_list, features, labels, train_mask, val_mask, test_mask

    elif dataset in ['CoraFull', 'Coauthor_cs', 'Coauthor_physics', 'AmazonCoBuy_computers', 'AmazonCoBuy_photo']:
        if dataset == 'CoraFull':
            data = CoraFullDataset()
        elif dataset == 'Coauthor_cs':
            data = CoauthorCSDataset('cs')
        elif dataset == 'Coauthor_physics':
            data = CoauthorPhysicsDataset('physics')
        elif dataset == 'AmazonCoBuy_computers':
            data = AmazonCoBuyComputerDataset('computers')
        elif dataset == 'AmazonCoBuy_photo':
            data = AmazonCoBuyPhotoDataset('photo')

        graph = data[0]
        features = torch.FloatTensor(graph.ndata['feat']).numpy()
        labels = torch.LongTensor(graph.ndata['label']).numpy()
        num_nodes = graph.number_of_nodes()

        if args.self_loop:
            time_stamp = time.time()
            print('before add self loop has {} edges'.format(len(graph.all_edges()[0])))
            graph = dgl.remove_self_loop(graph)
            graph = dgl.add_self_loop(graph)
            # graph = dgl.to_bidirected(graph)
            print('after add self loop has {} edges'.format(len(graph.all_edges()[0])))
            print("insert self loop cost {:.2f}s".format(time.time() - time_stamp))


        edges = graph.edges()
        edge_src = edges[0].numpy().reshape((-1,1))
        edge_dst = edges[1].numpy().reshape((-1,1))
        edges_list = np.hstack((edge_src, edge_dst))

        train_mask, val_mask, test_mask = split_dataset(num_nodes)
        print("dataset: {} nodes: {} edges: {} feature dims: {} classess: {} label nodes: {}"
              .format(dataset, num_nodes, edges_list.shape, 
              list(features.shape), len(np.unique(labels)),
              train_mask.sum() + test_mask.sum() + val_mask.sum()))
        return edges_list, features, labels, train_mask, val_mask, test_mask
    
    elif dataset in ['ogbn-arxiv', 'ogbn-papers100M', 'ogbn-products']:
        #load dataset
        data = DglNodePropPredDataset(name=dataset)
        
        graph = data.graph[0]
        labels = data.labels
        features = graph.ndata['feat']
        
        split_idx = data.get_idx_split()
        train_nid, val_nid, test_nid = split_idx['train'], split_idx['valid'], split_idx['test']
        # print(len(train_nid) + len(val_nid) + len(test_nid))
        train_mask = np.zeros(graph.number_of_nodes(), dtype=bool)
        train_mask[train_nid] = True
        val_mask = np.zeros(graph.number_of_nodes(), dtype=bool)
        val_mask[val_nid] = True
        test_mask = np.zeros(graph.number_of_nodes(), dtype=bool)
        test_mask[test_nid] = True

        if args.self_loop:
            time_stamp = time.time()
            print('before add self loop has {} edges'.format(len(graph.all_edges()[0])))
            graph = dgl.remove_self_loop(graph)
            graph = dgl.add_self_loop(graph)
            if dataset == "ogbn-arxiv":
                graph = dgl.to_bidirected(graph)
            print('after add self loop has {} edges'.format(len(graph.all_edges()[0])))
            print("insert self loop cost {:.2f}s".format(time.time() - time_stamp))
        
        edges = graph.edges()
        edge_src = edges[0].numpy().reshape((-1,1))
        edge_dst = edges[1].numpy().reshape((-1,1))
        edges_list = np.hstack((edge_src, edge_dst))

        print("nodes: {}, edges: {}, feature dims: {}, classess: {}, label nodes: {}"
              .format(graph.number_of_nodes(), edges_list.shape, 
              list(features.shape), len(np.unique(labels)),
              train_mask.sum() + test_mask.sum() + val_mask.sum()))     
        return edges_list, features, labels, train_mask, val_mask, test_mask

    else:
        raise NotImplementedError

def split_dataset(num_nodes):
    # TODO(Sanzo00) split dataset like 80-10-10
    train_mask = np.array([False for i in range(num_nodes)])
    val_mask = np.array([False for i in range(num_nodes)])
    test_mask = np.array([False for i in range(num_nodes)])
    for x in range(100):
        train_mask[x] = True
    for x in range(100, 200):
        val_mask[x] = True
    for x in range(200, 300):
        test_mask[x] = True
    return train_mask, val_mask, test_mask


def generate_nts_dataset(args, edge_list, features, labels, train_mask, val_mask, test_mask):
    dataset = args.dataset
    pre_path = os.getcwd() + '/' + dataset

    # edgelist
    # write_to_file(pre_path + '.edgelist', edge_list, "%d")

    # edge_list binary format (gemini)
    edge2bin(pre_path + '.edge', edge_list)

    # fetures
    write_to_file(pre_path + '.feat', features, "%.4f", index=True)

    # label
    write_to_file(pre_path + '.label', labels, "%d", index=True)

    # mask
    mask_list = []
    for i in range(len(labels)):
        if train_mask[i] == True:
            mask_list.append('train')
        elif val_mask[i] == True:
            mask_list.append('val')
        elif test_mask[i] == True:
            mask_list.append('test')
        else:
            mask_list.append('unknown')
    write_to_mask(pre_path + '.mask', mask_list)


def edge2bin(name, edges):
    time_cost = time.time()
    edges = edges.flatten()
    with open(name, 'wb') as f:
        buf = [int(edge).to_bytes(4, byteorder=sys.byteorder) for edge in edges]
        f.writelines(buf)
    time_cost = time.time() - time_cost
    print("write to {} is done, cost {:.2f}s throughput {:.2f}MB/s".format(name, time_cost, os.path.getsize(name)/1024/1024/time_cost))


def write_to_mask(name, data):
    time_cost = time.time()
    with open(name, 'w') as f:
        for i, node_type in enumerate(data):
            f.write(str(i) + ' ' + node_type + '\n')
    time_cost = time.time() - time_cost
    print("write to {} is done, cost {:.2f}s throughput {:.2f}MB/s".format(name, time_cost, os.path.getsize(name)/1024/1024/time_cost))


def write_to_file(name, data, format, index=False):
    time_cost = time.time()
    if not type(data) is np.ndarray:
        data = data.numpy()
    np.savetxt(name, data, fmt=format)
    
    if index:
        in_file= open(name, 'r') 
        out_file = open(name+'.temp', 'w')
        for i, line in enumerate(in_file):
            out_file.write(str(i) + ' ' + line)
        in_file.close()
        out_file.close()
        os.remove(name)
        os.rename(name+'.temp', name)

    time_cost = time.time() - time_cost
    print("write to {} is done, cost: {:.2f}s Throughput:{:.2f}MB/s".format(name, time_cost, os.path.getsize(name)/1024/1024/time_cost))

# def insert_self_loop(edge_list):
#     time_stamp = time.time()
#     graph = nx.from_edgelist(edge_list)
#     print("graph {:.3f}".format(time.time() - time_stamp))

#     time_stamp = time.time()
#     graph = graph.to_directed()
#     print("to directed {:.3f}".format(time.time() - time_stamp))

#     print('before insert self loop', graph)    
#     for i in graph.nodes:
#         graph.add_edge(i, i)
#     print('after insert self loop', graph)    
#     return np.array(graph.edges)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Dataset')
    parser.add_argument("--dataset", type=str, default="cora", help="Dataset name (cora, citeseer, pubmed, reddit)")
    parser.add_argument("--self-loop", type=bool, default=True, help="insert self-loop (default=True)")
    args = parser.parse_args()
    print('args: ', args)
    edges_list, features, labels, train_mask, val_mask, test_mask = extract_dataset(args)
    generate_nts_dataset(args, edges_list, features, labels, train_mask, val_mask, test_mask)
# -*- coding: utf-8 -*-
# @Author  : Sanzo00
# @Email  : arrangeman@163.com
# @Time    : 2022/06/17 09:00

import os
import sys
import argparse
import numpy as np
import networkx as nx
import time
import torch
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import register_data_args, load_data, CoraFull, Coauthor, AmazonCoBuy
# 'cora', 'citeseer', 'pubmed', 'reddit', 'CoraFull', 'Coauthor_cs', 'Coauthor_physics', 'AmazonCoBuy_computers', 'AmazonCoBuy_photo'

def extract_dataset(args):
    dataset = args.dataset
    if dataset in ['cora', 'citeseer', 'pubmed', 'reddit']:
        # mkdir ./dataset
        if not os.path.exists(dataset):
            os.mkdir(dataset)
        else:
            print('./{0} is exist, if you want re-generate this dataset please remove ./{0} and try again.'.format(dataset))
            sys.exit(-1)
        os.chdir(dataset)

        # load dataset
        data = load_data(args)
        features = data.features
        labels = data.labels
        assert(features.size(0) == len(labels))
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask
        graph = data.graph
        edge_list = np.array([])
        if dataset == 'reddit':
            edges = graph.edges()
            edge_src = edges[0].numpy().reshape((-1,1))
            edge_dst = edges[1].numpy().reshape((-1,1))
            edge_list = np.hstack((edge_src, edge_dst))
        else:
            edges = graph.edges
            edge_list = np.array([[src, dst] for src, dst in edges])
        
        if args.self_loop:
          edge_list = insert_self_loop(edge_list)

        print("dataset: {} nodes: {} edges: {} feature dims: {}, label nodes: {}"
              .format(dataset, len(labels), edge_list.shape, list(features.shape), 
              train_mask.sum() + test_mask.sum() + val_mask.sum()))
        return edge_list, features, labels, train_mask, val_mask, test_mask
    else:
        raise NotImplementedError


def generate_nts_dataset(args, edge_list, features, labels, train_mask, val_mask, test_mask):
    dataset = args.dataset
    pre_path = os.getcwd() + '/' + dataset

    # edgelist
    # write_to_file(pre_path + '.edgelist', edge_list, "%d")

    # edge_list binary format (gemini)
    edge2bin(pre_path + '.edge', edge_list)

    # fetures
    write_to_file(pre_path + '.feat', features, "%f", index=True)

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


def insert_self_loop(edge_list):
    G_t = nx.from_edgelist(edge_list)
    graph = nx.Graph(G_t)
    graph = graph.to_directed()
    print('before insert self loop', graph)    
    for i in graph.nodes:
        graph.add_edge(i, i)
    print('after insert self loop', graph)    
    return np.array(graph.edges)

def edge2bin(name, edges):
    time_cost = time.time()
    with open(name, 'wb') as f:
        for arr in edges:
            for node in arr:
                f.write(int(node).to_bytes(4, byteorder=sys.byteorder))
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Dataset')
    parser.add_argument("--dataset", type=str, default="cora", help="Dataset name (cora, citeseer, pubmed, reddit)")
    parser.add_argument("--self-loop", type=bool, default=True, help="insert self-loop (default=True)")
    args = parser.parse_args()
    print('args: ', args)
    edge_list, features, labels, train_mask, val_mask, test_mask = extract_dataset(args)
    generate_nts_dataset(args, edge_list, features, labels, train_mask, val_mask, test_mask)
'''
Mainly From:
https://github.com/FFTYYY/TWIRLS/blob/main/training_procedure/load_data/load_geom.py
'''


from multiprocessing.spawn import import_main_path
import os
import re
import networkx as nx
import numpy as np
from pandas.core.indexing import need_slice
import scipy.sparse as sp
import torch as th
from sklearn.model_selection import ShuffleSplit
import pickle as pkl
import sys

from utils.data_split import random_planetoid_splits

from data.loader import loader

class geom_dataloader(loader):
    def __init__(self, ds_name, device='cuda:0', self_loop=True, 
                    digraph=False, n_cv=3, cv_id=0,
                    needs_edge=False):
        super(geom_dataloader, self).__init__(
            ds_name, 
            cross_validation=True, 
            n_cv=n_cv, 
            cv_id=cv_id,
            needs_edge=needs_edge
            )
        self.device = device
        self.digraph = digraph
        self.self_loop = self_loop
        self.root_path = 'datasets/geom_data'

    def preprocess_features(self, features):
        """Row-normalize feature matrix and convert to tuple representation"""
        rowsum = np.array(features.sum(1), dtype=np.float32)
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
        return features

    def load_geom_graph(self):
        dataset_name = self.ds_name
        digraph = self.digraph
        graph_adjacency_list_file_path = os.path.join(self.root_path, dataset_name, 'out1_graph_edges.txt')
        graph_node_features_and_labels_file_path = os.path.join(self.root_path, dataset_name, 'out1_node_feature_label.txt')
        if digraph: 
            G = nx.DiGraph()
        else:
            G = nx.Graph()
        graph_node_features_dict = {}
        graph_labels_dict = {}

        if dataset_name == 'film':
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                    feature_blank = np.zeros(932, dtype=np.uint8)
                    feature_blank[np.array(line[1].split(','), dtype=np.uint16)] = 1
                    graph_node_features_dict[int(line[0])] = feature_blank
                    graph_labels_dict[int(line[0])] = int(line[2])
        else:
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                    graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
                    graph_labels_dict[int(line[0])] = int(line[2])

        with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
            graph_adjacency_list_file.readline()
            for line in graph_adjacency_list_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 2)
                if int(line[0]) not in G:
                    G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                                label=graph_labels_dict[int(line[0])])
                if int(line[1]) not in G:
                    G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                                label=graph_labels_dict[int(line[1])])
                G.add_edge(int(line[0]), int(line[1]))

        # remove self-loops
        G.remove_edges_from(nx.selfloop_edges(G)) # yh add
        features = np.array([features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
        labels = np.array([label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])
        features = self.preprocess_features(features)

        # edge_index
        adj = nx.adjacency_matrix(G, sorted(G.nodes()))
        edge_index = th.Tensor(adj.nonzero())
        assert (np.array_equal(np.unique(labels), np.arange(len(np.unique(labels)))))

        features = th.FloatTensor(features)
        labels = th.LongTensor(labels)
        labels = labels.view(-1)
        if self.self_loop:
            self_loop_index = th.arange(edge_index.max()+1).repeat(2,1)
            edge_index = th.hstack([edge_index, self_loop_index])
        return edge_index, labels, features


    def load_vanilla_data(self, use_cache=False):
        if use_cache:
            dump_name = '_'.join(['ds=', self.ds_name, 'udgraph=',str(not self.digraph),
                                'self-loop=',str(self.self_loop)
                                # 'split=', self.ds_split
                                ])
            dump_path = './cache/' + dump_name+'.pth'
            if not os.path.exists(dump_path):
                g_edge_index, labels, features = self.load_geom_graph()
                pkl.dump((g, labels, features), open(dump_path, 'wb'))
            else:
                g_edge_index, labels, features = pkl.load(open(dump_path, 'rb'))
        else:
            g_edge_index, labels, features = self.load_geom_graph()
        
        self.edge_index = g_edge_index.long().to(self.device)
        self.features = features.to(self.device)
        self.labels = labels.to(self.device)

        self.in_feats = self.features.shape[1]
        self.n_classes = self.labels.max().item() + 1
        self.n_edges = self.edge_index.shape[-1]

    def load_a_mask(self, p=None):
        if p == None:
            splits_file_path = os.path.join("dataset/splits", "{}_split_0.6_0.2_{}.npz".format(self.ds_name, self.cv_id))
            with np.load(splits_file_path) as splits_file:
                train_mask = splits_file['train_mask']
                val_mask = splits_file['val_mask']
                test_mask = splits_file['test_mask']
            self.train_mask = th.BoolTensor(train_mask).to(self.device)
            self.val_mask = th.BoolTensor(val_mask).to(self.device)
            self.test_mask = th.BoolTensor(test_mask).to(self.device)
            return 
        else:
            (p_train, p_val, p_test) = p
            percls_trn = int(round(p_train*len(self.labels)/self.n_classes))
            val_lb = int(round(p_val*len(self.labels)))
            # import ipdb; ipdb.set_trace()
            train_mask, val_mask, test_mask = random_planetoid_splits(
                self.labels, 
                self.n_classes, 
                percls_trn, 
                val_lb, 
                seed=self.seeds[self.cv_id])
            self.train_mask = train_mask.bool()
            self.val_mask = val_mask.bool()
            self.test_mask = test_mask.bool()

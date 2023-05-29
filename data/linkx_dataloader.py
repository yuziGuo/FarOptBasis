from multiprocessing.spawn import import_main_path
from operator import imod
import sys
sys.path.insert(0,'/home/yuhe_guo/repos/GNNexpPyg')

# from load_data import load_twitch, load_fb100, load_twitch_gamer, DATAPATH
from data.linkx import dataset
from data.linkx.dataset import load_nc_dataset
from data.linkx.data_utils import load_fixed_splits
from data.spliter import index_to_mask

from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops



import os
import networkx as nx
import numpy as np
from pandas.core.indexing import need_slice
import scipy.sparse as sp
import torch as th
import pickle as pkl
import sys

# from utils.data_split import random_planetoid_splits

from data.loader import loader


class linkx_dataloader(loader):
    def __init__(self, ds_name, device='cuda:0', self_loop=True, 
                    digraph=False, n_cv=3, cv_id=0,
                    needs_edge=False):
        super(linkx_dataloader, self).__init__(
            ds_name, 
            cross_validation=True, 
            n_cv=n_cv, 
            cv_id=cv_id,
            needs_edge=needs_edge
            )
        self.device = device
        self.digraph = digraph
        self.self_loop = self_loop
        if self.ds_name != 'Penn94':
            self.linkx_dataset = load_nc_dataset(self.ds_name)
        else:
            self.linkx_dataset = load_nc_dataset('fb100', 'Penn94')

    def load_vanilla_data(self):
        dataset = self.linkx_dataset
        if not self.digraph:
            dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
        if self.self_loop:
            dataset.graph['edge_index'] = remove_self_loops(dataset.graph['edge_index'])[0]
            dataset.graph['edge_index'] = add_self_loops(dataset.graph['edge_index'])[0]
        self.edge_index = dataset.graph['edge_index'].to(self.device)
        self.features = dataset.graph['node_feat'].to(self.device)
        self.labels = dataset.label.to(self.device)
        self.n_nodes = self.labels.shape[0]

        self.in_feats = self.features.shape[1]
        # infer the number of classes for non one-hot and one-hot labels
        if len(dataset.label.shape) == 1:
            dataset.label = dataset.label.unsqueeze(1)
        self.n_classes = max(self.labels.max().item() + 1, dataset.label.shape[1])
        self.n_edges = self.edge_index.shape[-1]


    def load_a_mask(self, p=None):
        if p==None:# 
            split_idx_lst = load_fixed_splits(self.ds_name, None)
            assert self.n_cv <= len(split_idx_lst)
            split = split_idx_lst[self.cv_id]
        else:
            print('Using fixed split is encourged!')
            (p_train, p_val, p_test) = p
            split = self.linkx_dataset.get_idx_split(train_prop=p_train, valid_prop=p_val)
        train_idxs = split['train']
        val_idxs = split['valid']
        test_idxs = split['test']
        self.train_mask = index_to_mask(train_idxs, self.n_nodes).bool()
        self.val_mask = index_to_mask(val_idxs, self.n_nodes).bool()
        self.test_mask = index_to_mask(test_idxs, self.n_nodes).bool()
        return split
        

def test_penn94():
    loader = linkx_dataloader('Penn94', 'cuda:1', True)
    loader.load_data()
    loader.load_mask()
    print('Success!')

def test_twitch_gamer():
    loader = linkx_dataloader('twitch-gamer', 'cuda:1', True)
    loader.load_data()
    loader.load_mask()
    print('Success!')

def test_pokec():
    loader = linkx_dataloader('pokec', 'cuda:1', True)
    loader.load_data()
    loader.load_mask()
    print('Success!')

def test_wiki():
    loader = linkx_dataloader('wiki', 'cuda:1', True)
    loader.load_data()
    loader.load_mask(p=(0.5,0.25,0.25))
    print('Success!')

def get_splits_for_wiki():
    loader = linkx_dataloader('wiki', 'cpu', True)
    loader.load_data()
    splits = []
    for i in range(5):
        print(i)
        split = loader.load_a_mask(p=(0.5,0.25,0.25))
        splits.append({k:v.numpy() for k,v in split.items()}) 
    np.save('wiki-splits-5.npy', splits)
    print('Success!')
    

if __name__=='__main__':
    get_splits_for_wiki()
    

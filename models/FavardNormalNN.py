import imp
import torch.nn as nn
import torch.nn.init as init

import torch as th
import torch.nn.functional as F

import math


import torch.nn as nn
from layers.FavardNormalConv import FavardNormalConv

import torch as th
import torch.nn.functional as F

class FavardNormalNN(nn.Module):
    def __init__(self,
                 edge_index,
                 norm_A, 
                 in_feats,
                 n_hidden,
                 n_classes,
                 K,
                 act_fn,
                 dropout,
                 dropout2,
                 ):
        super(FavardNormalNN, self).__init__()
        self.edge_index = edge_index
        self.norm_A = norm_A
        
        self.convs = nn.ModuleList()
        for _ in range(K):
            self.convs.append(FavardNormalConv())
        self.K = K
        self.n_channel = n_hidden
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_feats, n_hidden))
        self.fcs.append(nn.Linear(n_hidden, n_classes))
        self.act_fn = act_fn
        
        self.dropout = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout2)

        self.init_alphas()
        self.init_betas_and_yitas()
    
    def init_alphas(self):
        t = th.zeros(self.K+1)
        t[0] = 1
        t = t.repeat(self.n_channel, 1)
        self.alpha_params = nn.Parameter(t.float()) 

    def init_betas_and_yitas(self):
        self.yitas = nn.Parameter(th.zeros(self.K+1).repeat(self.n_channel,1).float()) # (n_channels, K+1)
        self.sqrt_betas = nn.Parameter(th.ones(self.K+1).repeat(self.n_channel,1).float()) # (n_channels, K+1)
        return

    def forward(self, features):
        x = features

        x = self.dropout(x)
        x = self.fcs[0](x) 
        
        x = self.act_fn(x) 
        x = self.dropout2(x) 

        sqrt_betas = th.clamp(self.sqrt_betas, 1e-2)
        # sqrt_betas = th.clamp(self.sqrt_betas, 1e-1)

        h0 = x / sqrt_betas[:,0]
        rst = th.zeros_like(h0)
        rst = rst + self.alpha_params[:,0] * h0

        last_h = h0
        second_last_h = th.zeros_like(h0)
        for i, con in enumerate(self.convs, 1):
            h_i = con(self.edge_index, self.norm_A, last_h, second_last_h, self.yitas[:,i-1], sqrt_betas[:,i-1], sqrt_betas[:,i])
            rst = rst + self.alpha_params[:,i] * h_i
            second_last_h = last_h
            last_h = h_i
        
        rst = self.dropout(rst)
        rst = self.fcs[-1](rst)
        rst = F.log_softmax(rst, dim=1)
        return rst

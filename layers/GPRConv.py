'''
Official Implementation from
https://github.com/jianhao2016/GPRGNN
'''

import torch
import numpy as np

from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing


class GPR_prop(MessagePassing):
    '''
    propagation class for GPR_GNN
    '''
    def __init__(self, K, alpha):
        super(GPR_prop, self).__init__(aggr='add')
        self.K = K
        self.alpha = alpha

        # PPR-like
        TEMP = alpha*(1-alpha)**np.arange(K+1)
        TEMP[-1] = (1-alpha)**K
        self.temp = Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K+1):
            self.temp.data[k] = self.alpha*(1-self.alpha)**k
        self.temp.data[-1] = (1-self.alpha)**self.K

    def forward(self, x, edge_index, norm):
        hidden = x*(self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k+1]
            hidden = hidden + gamma*x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)
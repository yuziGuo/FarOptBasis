'''
Offitial implementation from 
https://github.com/ivam-he
'''

import torch
import torch.nn.functional as F
from torch.nn import Linear
from layers.ChebIIConv import ChebnetII_prop

class ChebNetII(torch.nn.Module):
    def __init__(self,
                 edge_index,
                 norm_A, 
                 in_feats,
                 n_hidden,
                 n_classes,
                 K,
                 act_fn,
                 dropout,
                 dropout2
                 ):
        super(ChebNetII, self).__init__()
        self.edge_index = edge_index
        self.norm_A = norm_A
        self.act_fn = act_fn

        self.lin1 = Linear(in_feats, n_hidden)
        self.lin2 = Linear(n_hidden, n_classes)
        self.prop1 = ChebnetII_prop(K)

        self.dropout = dropout
        self.dprate = dropout2
        self.reset_parameters()

    def reset_parameters(self):
        self.prop1.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, features):
        x = features
        edge_index = self.edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        x = F.relu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
        
        return F.log_softmax(x, dim=1)
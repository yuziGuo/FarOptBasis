import torch.nn as nn
from layers.ARMAConv import ARMAConv

import torch as th
import torch.nn.functional as F

class ARMANN(nn.Module):
    def __init__(self,
                 edge_index,
                 norm_A, 
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_stacks, #
                 n_layers,
                 act_fn,
                 dropout,
                 dropout2,
                 ):
        super(ARMANN, self).__init__()
        self.edge_index = edge_index
        self.norm_A = norm_A

        self.armaconv = ARMAConv(in_feats, 
                            n_hidden, 
                            n_stacks, 
                            n_layers, 
                            shared_weights=True, 
                            act=act_fn, 
                            dropout=dropout2
                            )

        self.n_layers = n_layers
        self.n_channel = n_hidden
        self.fcs = nn.Linear(n_hidden, n_classes)
        self.act_fn = act_fn
        
        self.dropout = nn.Dropout(p=dropout)
    

    def forward(self, features):
        x = features        
        rst = self.armaconv(x, self.edge_index, self.norm_A)
        rst = self.act_fn(rst)
        rst = self.dropout(rst)
        rst = self.fcs(rst)
        rst = F.log_softmax(rst, dim=1)
        return rst
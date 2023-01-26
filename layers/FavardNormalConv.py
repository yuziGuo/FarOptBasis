import torch as th
from torch_geometric.nn import MessagePassing

class FavardNormalConv(MessagePassing):
    def __init__(self, fixed=False, kwargs={}):
        super(FavardNormalConv, self).__init__()

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def forward(self, edge_index, norm_A, last_h, second_last_h, yita, sqrt_beta, _sqrt_beta):
        '''
        last_h:         N x C
        second_last_h : N x C
        yita:           C
        sqrt_beta:      C
        _sqrt_beta:     C
        '''
        rst = self.propagate(edge_index=edge_index, x=last_h, norm=norm_A)
        rst = rst - yita.unsqueeze(0)*last_h - sqrt_beta.unsqueeze(0)*second_last_h
        rst = rst / _sqrt_beta.unsqueeze(0)
        return rst
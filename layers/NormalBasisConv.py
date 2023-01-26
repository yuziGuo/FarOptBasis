import torch as th
from torch_geometric.nn import MessagePassing

class NormalBasisConv(MessagePassing):
    def __init__(self, fixed=False, kwargs={}):
        super(NormalBasisConv, self).__init__()
        self.fixed = fixed
        if self.fixed:
            n_hidden = kwargs['n_hidden']
            self.register_buffer('three_term_relations', th.zeros(n_hidden, 3), persistent=False)
            # self.three_term_relations = th.zeros(n_hidden, 3)
            self.fixed_relation_stored = False

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def forward(self, edge_index, norm_A, last_h, second_last_h):
        rst = self.propagate(edge_index=edge_index, x=last_h, norm=norm_A)
        _t = th.einsum('nh,nh->h',rst,last_h)
        rst = rst - th.einsum('h,nh->nh', _t, last_h)
        _t = th.einsum('nh,nh->h',rst,second_last_h)
        rst = rst - th.einsum('h,nh->nh', _t, second_last_h)
        rst = rst / th.clamp((th.norm(rst,dim=0)),1e-8)
        return rst


    def _forward(self, edge_index, norm_A, last_h, second_last_h):
        rst = self.propagate(edge_index=edge_index, x=last_h, norm=norm_A)
        if self.fixed is True:
            if self.fixed_relation_stored is False:
                _t = th.einsum('nh,nh->h',rst,last_h)
                rst = rst - th.einsum('h,nh->nh', _t, last_h)
                self.three_term_relations[:,0] = _t

                _t = th.einsum('nh,nh->h',rst,second_last_h)
                rst = rst - th.einsum('h,nh->nh', _t, second_last_h)
                self.three_term_relations[:,1] = _t

                _norm = th.clamp((th.norm(rst,dim=0)),1e-8)
                rst = rst / _norm
                self.three_term_relations[:,2] = _norm     

                self.fixed_relation_stored = True
            else:
                rst = (rst - th.einsum('h,nh->nh', self.three_term_relations[:,0], last_h) \
                - th.einsum('h,nh->nh', self.three_term_relations[:,1], second_last_h))/self.three_term_relations[:,2]
        else:
            _t = th.einsum('nh,nh->h',rst,last_h)
            rst = rst - th.einsum('h,nh->nh', _t, last_h)
            _t = th.einsum('nh,nh->h',rst,second_last_h)
            rst = rst - th.einsum('h,nh->nh', _t, second_last_h)
            rst = rst / th.clamp((th.norm(rst,dim=0)),1e-8)
        return rst


# n_hidden x (K) x 3 + n_hidden + in_feats x n_hidden
# in_feats: 2000 n_hidden 64/512/1024
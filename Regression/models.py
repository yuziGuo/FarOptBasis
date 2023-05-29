import torch.nn as nn
import torch as th
from torch_geometric.nn import MessagePassing

import sys
from os import path
sys.path.append(path.dirname(path.dirname(__file__)))
from layers.NormalBasisConv import NormalBasisConv
from layers.FavardNormalConv import FavardNormalConv



class BaseModel(MessagePassing):
    def __init__(self, name, edge_index, norm_A, n_channel, K) -> None:
        super(BaseModel, self).__init__()
        self.name = name
        self.edge_index = edge_index
        self.basis = name
        self.norm_A = norm_A
        self.n_channel = n_channel
        self.K = K
        self.init_alphas()

    def init_alphas(self):
        if self.basis in ['ChebII', 'Bern']:
            t = th.ones(self.K+1)
            t = t.repeat(self.n_channel, 1)
        else:
            t = th.zeros(self.K+1)
            t[0] = 1
            t = t.repeat(self.n_channel, 1)
        self.alpha_params = nn.Parameter(t.float()) 

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j
    

class Monomial(BaseModel):
    '''Multichannel version GPRGNN for regression.
    x: C x N
    self.alpha_params: K x C
    '''
    def forward(self, x):
        z = x*(self.alpha_params[:, 0])
        for k in range(self.K):
            x = self.propagate(self.edge_index, x=x, norm=self.norm_A)
            gamma = self.alpha_params[:, k+1]
            z = z + gamma*x
        return z
        

class Normal(BaseModel):
    def __init__(self, name, edge_index, norm_A, n_channel, K):
        super(Normal, self).__init__(name, edge_index, norm_A, n_channel, K)
        self.convs = nn.ModuleList()
        for _ in range(K):
            self.convs.append(NormalBasisConv())

    def forward(self, x):
        h0 = x / th.clamp((th.norm(x,dim=0)), 1e-8)
        rst = th.zeros_like(h0)
        rst = rst + self.alpha_params[:,0] * h0

        last_h = h0
        second_last_h = th.zeros_like(h0)        
        for i, con in enumerate(self.convs, 1):
            h_i = con(self.edge_index, self.norm_A, last_h, second_last_h)
            '''
            # check
            _norm = th.norm(h_i, dim=0)
            if (_norm==0).sum()>0:
                import ipdb; ipdb.set_trace()
            # end check
            '''
            rst = rst + self.alpha_params[:,i] * h_i
            second_last_h = last_h
            last_h = h_i
        return rst

class Favard(BaseModel):
    def __init__(self, name, edge_index, norm_A, n_channel, K):
        super(Favard, self).__init__(name, edge_index, norm_A, n_channel, K)
        self.convs = nn.ModuleList()
        for _ in range(K):
            self.convs.append(FavardNormalConv())
        self.init_betas_and_yitas()
        
    def init_betas_and_yitas(self):
        self.yitas = nn.Parameter(th.zeros(self.K+1).repeat(self.n_channel,1).float()) # (n_channels, K+1)
        self.sqrt_betas = nn.Parameter(th.ones(self.K+1).repeat(self.n_channel,1).float()) # (n_channels, K+1)
        return

    def forward(self, x):
        sqrt_betas = th.clamp(self.sqrt_betas, 1e-2)

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
        return rst


import torch.nn.functional as F
import math
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.utils import get_laplacian

def cheby(i,x):
    if i==0:
        return 1
    elif i==1:
        return x
    else:
        T0=1
        T1=x
        for ii in range(2,i+1):
            T2=2*x*T1-T0
            T0,T1=T1,T2
        return T2

class ChebII(BaseModel):
    def forward(self, x):
        edge_index = self.edge_index
        edge_weight = None
        coe_tmp=F.relu(self.alpha_params)
        coe=coe_tmp.clone()
        
        # import ipdb; ipdb.set_trace()
        for i in range(self.K+1):
            coe[:,i]=coe_tmp[:,0]*cheby(i,math.cos((self.K+0.5)*math.pi/(self.K+1)))
            for j in range(1,self.K+1):
                x_j=math.cos((self.K-j+0.5)*math.pi/(self.K+1))
                coe[:,i]=coe[:,i]+coe_tmp[:,j]*cheby(i,x_j)
            coe[:,i]=2*coe[:,i]/(self.K+1)
        
        #L=I-D^(-0.5)AD^(-0.5)
        edge_index1, norm1 = get_laplacian(edge_index, edge_weight,normalization='sym', dtype=x.dtype, num_nodes=x.size(self.node_dim))

        #L_tilde=L-I
        edge_index_tilde, norm_tilde= add_self_loops(edge_index1,norm1,fill_value=-1.0,num_nodes=x.size(self.node_dim))

        Tx_0=x
        Tx_1=self.propagate(edge_index_tilde,x=x,norm=norm_tilde,size=None)

        out=coe[:,0]/2*Tx_0+coe[:,1]*Tx_1

        for i in range(2,self.K+1):
            Tx_2=self.propagate(edge_index_tilde,x=Tx_1,norm=norm_tilde,size=None)
            Tx_2=2*Tx_2-Tx_0
            out=out+coe[:,i]*Tx_2
            Tx_0,Tx_1 = Tx_1, Tx_2
        return out


from scipy.special import comb

class Bern(BaseModel):
    def forward(self, x):
        edge_index = self.edge_index
        edge_weight = None
        TEMP=F.relu(self.alpha_params)

        #L=I-D^(-0.5)AD^(-0.5)
        edge_index1, norm1 = get_laplacian(edge_index, edge_weight, normalization='sym', dtype=x.dtype, num_nodes=x.size(self.node_dim))
        #2I-L
        edge_index2, norm2=add_self_loops(edge_index1,-norm1,fill_value=2.,num_nodes=x.size(self.node_dim))

        tmp=[]
        tmp.append(x)
        for i in range(self.K):
            x=self.propagate(edge_index2,x=x,norm=norm2,size=None)
            tmp.append(x)

        out=(comb(self.K,0)/(2**self.K))*TEMP[:,0]*tmp[self.K]

        for i in range(self.K):
            x=tmp[self.K-i-1]
            x=self.propagate(edge_index1,x=x,norm=norm1,size=None)
            for j in range(i):
                x=self.propagate(edge_index1,x=x,norm=norm1,size=None)

            out=out+(comb(self.K,i+1)/(2**self.K))*TEMP[:,i+1]*x
        return out
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_undirected, add_self_loops
import torch

import scipy
from scipy.sparse.linalg import eigsh
from torch_geometric.utils import to_scipy_sparse_matrix
import numpy as np
from scipy.linalg import expm

import pickle
import time

def report_time():
    print(time.strftime('%m%d-%H:%M:%S\n', time.localtime(time.time())))


def build_2Dgrid(height, width):
    N = height * width
    edge_index = []
    for i in range(0, N):
        if i + width < N:
            edge_index.append([i, i+width])
        if (i+1) % width != 0: # not the last pixel in the row
            edge_index.append([i, i+1])
    edge_index = torch.LongTensor(edge_index).T
    edge_index = to_undirected(edge_index)
    edge_index, _ = add_self_loops(edge_index)
    return edge_index

def sinm(A):
    return -0.5j*(expm(1j*A) - expm(-1j*A))

def cosm(A):
    return 0.5*(expm(1j*A) + expm(-1j*A))

def comb_filter(A):
    return np.abs(sinm(np.pi * A))

for Np in [100]:
    print(f'Pixel: {Np} x {Np}')
    report_time()
    edge_index = build_2Dgrid(Np, Np)
    A_sparse = to_scipy_sparse_matrix(edge_index)

    # normalize
    deg = np.array(A_sparse.sum(axis=0))[0]
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt = scipy.sparse.diags(deg_inv_sqrt)
    A_sparse = deg_inv_sqrt @ A_sparse @ deg_inv_sqrt
    I_sparse = scipy.sparse.diags(np.ones(Np*Np)) 
    L_sparse = I_sparse - A_sparse

    # L^2
    L_square_sparse = L_sparse @ L_sparse
    # P^2
    A_square_sparse = A_sparse @ A_sparse
    
    # Low_pass
    low_pass = expm(-10 * L_square_sparse)
    pickle.dump(low_pass, open(f'save/lowpass_Np={Np}.pkl','wb'))
    print('Low pass dumped!')
    report_time()

    # High-pass
    high_pass = I_sparse - low_pass
    pickle.dump(high_pass, open(f'save/highpass_Np={Np}.pkl','wb'))

    # Band-pass
    band_pass = expm(-10 * A_square_sparse)
    pickle.dump(band_pass, open(f'save/bandpass_Np={Np}.pkl','wb'))
    print('band pass dumped!')
    report_time()

    # Band-reject
    band_pass = pickle.load(open(f'save/bandpass_Np={Np}.pkl','rb'))
    band_reject = I_sparse - band_pass
    pickle.dump(band_reject, open(f'save/bandreject_Np={Np}.pkl','wb'))
    report_time()

    # comb
    # comb = np.abs(sinm(np.pi * L_sparse))
    # pickle.dump(comb, open(f'save/comb_Np={Np}.pkl','wb'))
    # print('comb dumped!')
    # report_time()
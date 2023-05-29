from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_undirected, add_self_loops
import torch

import pickle
import os

from torchvision.io import read_image
from torchvision.transforms import Resize
from torchvision.transforms import InterpolationMode
import torch
from torch import Tensor
from torch import clamp

import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm

global Np
Np = 100

def low_pass(signal):
    M = pickle.load(open(f'save/lowpass_Np={Np}.pkl','rb'))
    signal_ndarr = M @ signal 
    signal_tensor = torch.Tensor(signal_ndarr)
    return signal_tensor

def high_pass(signal):
    M = pickle.load(open(f'save/highpass_Np={Np}.pkl','rb'))
    signal_ndarr = M @ signal 
    signal_tensor = torch.Tensor(signal_ndarr)
    return signal_tensor

def band_pass(signal):
    M = pickle.load(open(f'save/bandpass_Np={Np}.pkl','rb'))
    signal_ndarr = M @ signal 
    signal_tensor = torch.Tensor(signal_ndarr)
    return signal_tensor

def band_reject(signal):
    M = pickle.load(open(f'save/bandreject_Np={Np}.pkl','rb'))
    signal_ndarr = M @ signal 
    signal_tensor = torch.Tensor(signal_ndarr)
    return signal_tensor

def allpass(signal):
    return signal 

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



def rgb2yuv(R,G,B):
    Y = clamp(0.299    * R + 0.587    * G + 0.114    * B, -128, 127).int();
    U = clamp(-0.14713 * R + -0.28886 * G + 0.436    * B + 64, -128, 127).int();
    V = clamp(0.615    * R + -0.51499 * G + -0.10001 * B + 64, -128, 127).int();
    return Y,U,V

def yuv2rgb(Y,U,V):
    R = 1 * Y +        0 * (U - 64) + 1.13983 * (V - 64)
    G = 1 * Y + -0.39465 * (U - 64) + -0.5806 * (V - 64)
    B = 1 * Y + 2.03211 * (U - 64) +       0 * (V - 64)
    print(R.max(),R.min())
    print(G.max(),G.min())
    print(B.max(),B.min())
    R = clamp(R, -128, 127).int()
    G = clamp(G, -128, 127).int()
    B = clamp(B, -128, 127).int()
    return R,G,B

from os import listdir

def main(need_postprocess=False):

    dataset = []
    # Np = 100
    # edge_index = build_2Dgrid(Np, Np)

    # 
    if not os.path.exists('save/'):
        os.makedirs('save')

    patterns = [
        ['band_reject', 'low_pass', 'high_pass'],
        ['high_pass', 'high_pass', 'low_pass'],
        ['high_pass', 'low_pass', 'high_pass'],
        ['low_pass', 'band_reject', 'band_reject']
    ]
    for path in tqdm(listdir('BernNetImages/')):
        print(path)
        try:
            image = Resize(size=(100,100))(read_image(f'BernNetImages/{path}'))
        except:
            RuntimeError
            continue
        if image.shape[0] == 3:
            R, G, B = image
        elif image.shape[0] == 4:
            R, G, B, _ = image
        else:
            print('Not implemented')
            exit(-1)
        R = R.int() 
        G = G.int()
        B = B.int()

        # preprocess
        R, G, B = R - 128, G -128, B - 128
        Y, U, V = rgb2yuv(R, G, B)
        feat_Y = Y.view(Np*Np,1)
        feat_Cb = U.view(Np*Np,1)-64
        feat_Cr = V.view(Np*Np,1)-64

        # filtering
        for pattern in patterns:
            [filter_Y, filter_Cb, filter_Cr] = pattern
            _feat_Y = globals()[filter_Y](feat_Y)
            _feat_Cb = globals()[filter_Cb](feat_Cb)
            _feat_Cr = globals()[filter_Cr](feat_Cr)

            dataset.append({
                'signal': torch.hstack([feat_Y, feat_Cb, feat_Cr]),
                'filtered_signal': torch.hstack([_feat_Y, _feat_Cb, _feat_Cr]),
                'true_filter': pattern
                })

        if need_postprocess:
            # postprocess
            _feat_Cb = _feat_Cb+64
            _feat_Cr = _feat_Cr+64
            _R,_G,_B = yuv2rgb(_feat_Y.view(Np,Np), _feat_Cb.view(Np,Np), _feat_Cr.view(Np,Np))
            _R, _G, _B = _R + 128, _G + 128, _B + 128

    return dataset



if __name__ == '__main__':
    dataset = main()
    pickle.dump(dataset, open('MultiChannelFilterDataset.pkl', 'wb'))




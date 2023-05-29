from models import Monomial, Normal, Favard, Favard_2, ChebII, Bern
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import to_undirected, add_self_loops

import torch as th
import pickle

def build_2Dgrid(height, width):
    N = height * width
    edge_index = []
    for i in range(0, N):
        if i + width < N:
            edge_index.append([i, i+width])
        if (i+1) % width != 0: # not the last pixel in the row
            edge_index.append([i, i+1])
    edge_index = th.LongTensor(edge_index).T
    edge_index = to_undirected(edge_index)
    edge_index, _ = add_self_loops(edge_index)
    return edge_index

def load_data(args):
    data = pickle.load(open('MultiChannelFilterDataset.pkl', 'rb'))
    idx = args.idx
    print(data[idx]['true_filter'])
    return data[idx]['signal'].float().to(args.gpu), data[idx]['filtered_signal'].float().to(args.gpu)

def build_optimizer(args, model):
    if args.basis not in  ['Favard']:
        param_groups = [{'params':[model.alpha_params], 'lr':args.lr,'weight_decay':args.wd}]
        optimizer = th.optim.Adam(param_groups)
    else:
        param_groups = [
            {'params':[model.alpha_params], 'lr':args.lr,'weight_decay':args.wd},
            {'params':[model.yitas, model.sqrt_betas], 'lr':0.05,'weight_decay':args.wd},
            ]
        optimizer = th.optim.Adam(param_groups)
    return optimizer

def build_model(args,edge_index,norm_A):
    if args.basis == 'Monomial':
        model = Monomial(args.basis, edge_index, norm_A, n_channel=3, K=10)
    if args.basis == 'Normal':
        model = Normal(args.basis, edge_index, norm_A, n_channel=3, K=10)
    if args.basis == 'ChebII':
        model = ChebII(args.basis, edge_index, norm_A, n_channel=3, K=10)
    if args.basis == 'Bern':
        model = Bern(args.basis, edge_index, norm_A, n_channel=3, K=10)
    if args.basis == 'Favard':
        model = Favard(args.basis, edge_index, norm_A, n_channel=3, K=10)
    return model.to(args.gpu)

def main(args):
    print(args.basis)
    # graph
    Np = 100
    edge_index = build_2Dgrid(Np, Np)
    edge_index = edge_index.to(args.gpu)
    edge_index, norm_A = gcn_norm(edge_index)
    
    # features
    signal, filtered_signal_ground = load_data(args)
    
    # criterion
    loss_fcn = th.nn.MSELoss()

    # model
    model = build_model(args,edge_index,norm_A)

    # optimizer
    optimizer = build_optimizer(args, model)

    # run
    if args.basis == 'Normal' or args.norm:
        _norm = th.norm(signal,dim=0)
        signal = signal / _norm
    loss_history = [th.inf]
    for epoch in range(args.n_epochs):
        model.train()
        optimizer.zero_grad()
        filtered_signal = model(signal)
        if args.basis == 'Normal' or args.norm:
            filtered_signal = filtered_signal *_norm
        loss = loss_fcn(filtered_signal, filtered_signal_ground)
        # if abs(loss_history[-1] - loss) < 0.0001:
        #     return loss, loss_history
        loss_history.append(loss.item())
        if (epoch + 1) % 50 == 0:
            print(epoch)
            print(th.round(loss, decimals=3))
        loss.backward()
        optimizer.step()
    return loss, loss_history
    


def set_args():
    import argparse
    parser = argparse.ArgumentParser(description='Regression')
    parser.add_argument("--basis", type=str, default='Normal',help='(Monomial, )')
    parser.add_argument("--gpu", type=int, default=0, help="gpu")

    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--wd", type=float, default=5e-4, help="weight decay")

    parser.add_argument("--n-epochs", type=int, default=100, help="Maximum epochs")
    parser.add_argument("--idx", type=int, default=0, help="Maximum epochs")

    parser.add_argument("--norm", action='store_true', default=False)


    args = parser.parse_args()
    return args

# python train.py  --n-epochs 500 --lr 0.05

if __name__ == '__main__':
    args = set_args()
    
    for args.idx in [0,1,2,3]:
        for basis in ['Normal','ChebII', 'Monomial', 'Bern', 'Favard']:
            args.basis = basis
            loss, loss_history = main(args)
            print(loss)
            pickle.dump(loss_history,open(f'loss_history_{basis}_sample_{args.idx}.pkl','wb'))
            print('***************')
import torch as th
import numpy as np

def index_to_mask(index, size):
    if th.is_tensor(index):
        mask = th.zeros(size, dtype=th.int)
        mask[index] = 1
    else:
        mask = np.zeros(size)
        mask[index] = 1
    return mask


def random_planetoid_splits(y, num_classes, percls_trn=20, val_lb=500, seed=12134):
    print(seed)
    index=[i for i in range(0,y.shape[0])]
    train_idx=[]
    rnd_state = np.random.RandomState(seed)
    for c in range(num_classes):
        class_idx = np.where(y == c)[0]
        if len(class_idx)<percls_trn:
            train_idx.extend(class_idx)
        else:
            train_idx.extend(rnd_state.choice(class_idx, percls_trn,replace=False))
    rest_index = [i for i in index if i not in train_idx]
    val_idx=rnd_state.choice(rest_index,val_lb,replace=False)
    test_idx=[i for i in rest_index if i not in val_idx]
    num_nodes = y.shape[-1]
    train_mask = index_to_mask(train_idx,size=num_nodes)
    val_mask = index_to_mask(val_idx,size=num_nodes)
    test_mask = index_to_mask(test_idx,size=num_nodes)
    return train_mask, val_mask, test_mask
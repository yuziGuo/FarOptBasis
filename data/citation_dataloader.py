from data.loader import loader
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T


class citation_loader(loader):
    def __init__(self, ds_name, device='cuda:0', self_loop=True):
        super(citation_loader, self).__init__(ds_name, self_loop)
        self.device = device
        self.self_loop = self_loop
        self.name = ds_name.lower()

    def load_vanilla_data(self):
        data = Planetoid(root='~/datasets/Planetoid',name=self.name,transform=T.NormalizeFeatures())
        g = data[0]

        if self.self_loop:
            g.edge_index, _ = add_remaining_self_loops(g.edge_index)
        
        self.edge_index = g.edge_index.to(self.device)
        self.features = g.x.to(self.device)
        self.labels = g.y.to(self.device)

        self.train_mask = g.train_mask.to(self.device)
        self.val_mask = g.val_mask.to(self.device)
        self.test_mask = g.test_mask.to(self.device)

        self.in_feats = self.features.shape[1]
        self.n_classes = data.num_classes
        self.n_edges = self.edge_index.shape[-1]
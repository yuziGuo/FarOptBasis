# from dgl import ops
import torch as th
import random

class loader():
    def __init__(self, ds_name='Cora', 
                        self_loop=True,
                        process_features=False,
                        cross_validation=False,
                        n_cv=-1,
                        cv_id=-1,
                        needs_edge=False,
                        seeds = None
                        # needs_edge_oracle=False
                        ):
        self.g = None
        self.feature = None
        self.labels = None
        
        self.n_classes = None
        self.n_edges = None

        self.train_mask = None
        self.val_mask = None
        self.test_mask = None

        self.cross_validation = cross_validation
        self.n_cv = n_cv
        self.cv_id = cv_id
        if cross_validation and self.n_cv > 1:
            self.cv_id = cv_id
        else:
            cv_id = 0
        
        self.ds_name = ds_name
        self.self_loop = self_loop

        self.needs_edge = needs_edge
        self.edge_labels = None
        self.edge_mask = None
    
    def set_split_seeds(self):
        self.seeds = [random.randint(0,10000) for i in range(self.n_cv)]


    def load_mask(self,p=None):
        if self.cross_validation:
            self.load_a_mask(p)
            self.cv_id += 1
            if self.needs_edge:
                self._load_edge_masks()
        else:
            # For some datasets, mask is already loaded before.
            # print('Mask is already loaded!')
            pass
        return


    def load_edge_labels(self, multiview=True):
        if multiview is False:
            labels = self.labels.float()
            labels = labels.unsqueeze(-1)
        else:
            labels = th.nn.functional.one_hot(self.labels)         
        _t = ops.u_sub_v(self.g, labels.float(), labels.float())
        _t_id_1 = (_t==0)
        _t_id_2 = (_t!=0)
        _t[_t_id_1] = 1
        _t[_t_id_2] = -1

        self.g.edata['labels_edge'] = _t  # [n_edges, 1] or [n_edges, n_class]
        
        # idxs = th.Tensor(_t.shape).uniform_(0,1) < prop
        # self.g.edata['labels_edge'] = th.ones_like(_t)
        # self.g.edata['labels_edge'][idxs] = _t[idxs]

        # self.load_edge_masks()
        # idxs = self.g.edata['train_mask_edge'] | self.g.edata['val_mask_edge']
        # # idxs = self.g.edata['train_mask_edge'] 
        # self.g.edata['labels_edge'] = th.ones_like(_t)
        # self.g.edata['labels_edge'][idxs] = _t[idxs]
        return


    def _load_edge_masks(self):
        self.g.edata['train_mask_edge'] = \
                ops.u_mul_v(self.g, self.train_mask.float(), self.train_mask.float())  
        self.g.edata['val_mask_edge'] = \
                ops.u_mul_v(self.g, self.train_mask.float(), self.val_mask.float())  \
                    + ops.u_mul_v(self.g, self.val_mask.float(), self.train_mask.float()) \
                    + ops.u_mul_v(self.g, self.val_mask.float(), self.val_mask.float())
        assert (th.unique(self.g.edata['val_mask_edge']).size()[-1]==2)
        self.g.edata['test_mask_edge'] = \
                1 - self.g.edata['train_mask_edge'] - self.g.edata['val_mask_edge'] 


        # self.g.edata['train_mask_edge'] = \
        #         ops.u_mul_v(self.g, self.train_mask.float(), self.train_mask.float()) \
        #             + ops.u_mul_v(self.g, self.train_mask.float(), self.val_mask.float())  \
        #             + ops.u_mul_v(self.g, self.val_mask.float(), self.train_mask.float())   
        # self.g.edata['val_mask_edge'] = \
        #             ops.u_mul_v(self.g, self.val_mask.float(), self.val_mask.float())
        # assert (th.unique(self.g.edata['val_mask_edge']).size()[-1]==2)
        # self.g.edata['test_mask_edge'] = \
        #         1 - self.g.edata['train_mask_edge'] - self.g.edata['val_mask_edge'] 

        self.g.edata['train_mask_edge'] = self.g.edata['train_mask_edge'].bool()
        self.g.edata['val_mask_edge'] = self.g.edata['val_mask_edge'].bool()
        self.g.edata['test_mask_edge'] = self.g.edata['test_mask_edge'].bool()
        return
        

    def load_data(self):
        self.load_vanilla_data()
        # self.process_graph()
        # import ipdb; ipdb.set_trace()

        # if self.needs_edge_oracle:
        #     self.load_edge_labels(multiview=edge_multiview)
        # self.load_mask()
        # if self.needs_edge:
        #     self.load_edge_masks()

    def load_vanilla_data(self):
        '''
            Implemented in child-classes
            load features and labels
        '''
        pass

    def process_features():
        pass

    def process_graph(self):
        self.g = self.g.remove_self_loop()
        if self.self_loop:
            self.g = self.g.add_self_loop()
        self.n_edges = self.g.number_of_edges()



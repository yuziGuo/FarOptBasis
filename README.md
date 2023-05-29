# Graph Neural Networks with Learnable and Optimal Polynomial Bases
[**Under construction**].

[Paper](https://arxiv.org/abs/2302.12432). 

This repository includes the implementation for **FavardGNN** and **OptBasisGNN**, 
two spectral graph neural networks which adapts the polynomial bases for filtering. 

## Reproducing Results.
<!-- Scripts for reproducing results of our models in Table 1.  -->

### Folder structure.
Before running the experiments, 
the folder structure is as below:
```.
├── cache
│   └── ckpts
├── data
│   ├── linkx
├── datasets
│   ├── geom_data
│   ├── linkx
│   └── Planetoid
├── layers
├── models
├── runs
│   └── placeholder.txt
└── utils
```

### Reproducing Table 1.

```
sh reproduce_favardgnn.sh
sh reproduce_optbasis.sh
```

![Image](https://pic4.zhimg.com/80/v2-0d26a237dc57435236f1ec6e7d19a9be.png)

### Reproducing Table 2.

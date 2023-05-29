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
```bash
.
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

### Reproducing Results on Geom-GCN Datasets (Tbl.1).
Run scripts in the following files under `./` path.
```bash
sh scripts/reproduce_favardgnn.sh
sh scripts/reproduce_optbasis.sh
```

![Image](https://pic4.zhimg.com/80/v2-0d26a237dc57435236f1ec6e7d19a9be.png)

### Reproducing Results on LINKX Datasets (Tbl.2).
Run scripts in the following files under `./` path.
```bash
sh scripts/reproduce_linkx.sh
```

### Reproducing Table 2.
Shift working path to `Regression/`. 
```bash
sh> cd Regression
```

**Step 1: Prepare images**
```bash
sh> unzip -d BernNetImages  BernNet-LearningFilters-image.zip
```

**Step 2: Pre-compute $ U h(\Lambda) U^T $**

Pre-compute the matrix polynomials $M = U h(\Lambda) U^T = h(L)$
where $h$ corresponds to 
$L$ is the Laplacian matrix for `100x100` grid graph, and 
and $h$ corresponds to
- `High-pass` filter;
- `Low-pass` filter;
- `Band-pass` filter; 
- `Band-reject` filter. 

```bash
sh> python preprocess_matrix_polynomials.py
```

The result of this step is saved in the `save/` folder.
```
.save/
├── bandpass_Np=100.pkl
├── bandreject_Np=100.pkl
├── highpass_Np=100.pkl
└── lowpass_Np=100.pkl
```

**Step 3: Make dataset.**
```bash
sh> python make_dataset.py
```
The result of this step is `MultiChannelFilterDataset.pkl`.


**Step 4: Run experiments!**
To reproduce *Table 5*, 
you can use the bash script below to run over all the samples.
```bash
sh> python main_all.py
```

To reproduce converging curves as in *Figure 2*, 
you can use the following script to run one or several samples and record the losses.
```bash
sh> python main_sample.py
```
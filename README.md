# Graph Neural Networks with Learnable and Optimal Polynomial Bases
[**Under construction**].

[Paper](https://arxiv.org/abs/2302.12432). 

This repository includes the implementation for **FavardGNN** and **OptBasisGNN**, 
two spectral graph neural networks which adapts the polynomial bases for filtering. 

## Reproducing Classification Results.
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

![Table 1](./scripts/reported/tbl1.png)

### Reproducing Results on LINKX Datasets (Tbl.2).
Run scripts in the following files under `./` path.
```bash
sh scripts/reproduce_linkx.sh
```
![Table 2](./scripts/reported/tbl2.png)

## Reproducing Regression Task.
Shift working path to `Regression/`. 
```bash
sh> cd Regression
```

**Step 1: Prepare images**
```bash
sh> unzip -d BernNetImages  BernNet-LearningFilters-image.zip
```

**Step 2: Pre-compute $U h(\Lambda) U^T$**
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
This step would take several hours, 
you can also 
**download** our pre-computed matrices from 
this [google drive url](https://drive.google.com/file/d/1UwNyjfTykPLhhYtW52XVJ_wexJ_LmONV/view?usp=sharing), and unzip them directly.
```bash
sh> unzip save/cachedMatrices.zip -d ./save/
sh> rm ./save/cachedMatrices.zip
```

The resulted files are:
```bash
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
The result of this step is a pickle file `MultiChannelFilterDataset.pkl`.


**Step 4: Run experiments!**
Now we run the regression task!
At this moment, the folder structure (ignoring python files) is
```
./Regression/
├── BernNet-LearningFilters-image.zip
├── MultiChannelFilterDataset.pkl
└── save
    ├── bandpass_Np=100.pkl
    ├── bandreject_Np=100.pkl
    ├── highpass_Np=100.pkl
    └── lowpass_Np=100.pkl
```


To reproduce *Table 5*, 
you can use the bash script below to run over all the samples.
```bash
sh> python main_all.py
```
<!-- ![Table 5](./scripts/reported/tbl5.png) -->
<img src="./scripts/reported/tbl5.png" alt="Table 5" width="500" height="150">

To reproduce converging curves as in *Figure 2*, 

<!-- ![sample](./scripts/reported/icml-6.pdf) -->
<!-- ![Figure 2](./scripts/reported/fig2.png) -->
<img src="./scripts/reported/fig2.png" alt="Figure 2" width="500" height="300">

you can use the following script to run one or several samples and record the losses.
```bash
sh> python main_sample.py
```
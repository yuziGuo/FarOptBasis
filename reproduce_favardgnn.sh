# pubmed
python train_models.py  --model FavardNormalNN --dataset pubmedfull --udgraph  --lr1 0.05 --lr2 0.005 --lr3 0.005 --wd1 1e-3 --wd2 1e-6 --wd3 1e-6 --n-layers 8 --dropout 0 --dropout2 0.5  --log-detail --log-detailedCh --early-stop  --n-cv 20

# chameleon
python train_models.py  --model FavardNormalNN --dataset geom-chameleon --udgraph  --lr1 0.04 --lr2 0.04 --lr3 0.02 --wd1 1e-6 --wd2 1e-5 --wd3 1e-7 --n-layers 20 --dropout 0.5 --dropout2 0.5  --log-detail --log-detailedCh --early-stop  --n-cv 20


# squirrel
python train_models.py  --model FavardNormalNN --dataset geom-squirrel --udgraph  --lr1 0.05 --lr2 0.04 --lr3 0.0005 --wd1 1e-5 --wd2 1e-7 --wd3 1e-6 --n-layers 20 --dropout 0.5 --dropout2 0.5  --log-detail --log-detailedCh --early-stop  --n-cv 20

# film
python train_models.py  --model FavardNormalNN --dataset geom-film --udgraph  --lr1 0.05 --lr2 0.05 --lr3 0.05 --wd1 1e-5 --wd2 1e-3 --wd3 1e-5 --n-layers 16 --dropout 0.7 --dropout2 0.7  --log-detail --log-detailedCh --early-stop  --n-cv 20

# citeseerfull
python train_models.py  --model FavardNormalNN --dataset citeseerfull --udgraph   --lr1 0.001 --lr2 0.001 --lr3 0.001 --wd1 1e-6 --wd2 1e-6 --wd3 1e-8 --n-layers 8 --dropout 0.7 --dropout2 0.9   --log-detail --log-detailedCh --early-stop  --n-cv 20

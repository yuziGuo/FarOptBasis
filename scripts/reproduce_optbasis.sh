# # pubmedfull
python train_models.py   --dataset pubmedfull --udgraph  --lr1 0.04 --lr2 0.005 --wd1 1e-5 --wd2 1e-3 --dropout 0.0  --dropout2 0 --n-layers 12   --log-detail --log-detailedCh --early-stop  --n-cv 20 

# # film
python train_models.py  --dataset geom-film --udgraph  --early-stop  --lr1 0.005 --lr2 0.03 --wd1 1e-3 --wd2 1e-3 --n-layers 4 --dropout 0.6  --dropout2 0.5 --n-cv 20  --log-detail --log-detailedCh


# # citeseerfull
python train_models.py   --dataset citeseerfull --udgraph  --lr1 0.04 --lr2 0.005 --wd1 1e-3 --wd2 1e-3 --dropout 0.5  --dropout2 0.5 --n-layers 2    --log-detail --log-detailedCh --early-stop  --n-cv 20


# squirrel
python train_models.py  --model NormalNN --dataset geom-squirrel --udgraph  --log-detail --log-detailedCh --early-stop --dropout 0. --dropout2 0.5 --lr1 0.001 --lr2 0.03 --n-layers 20 --wd1 1e-7 --wd2 1e-3   --n-cv 20     

# chameleon
python train_models.py  --model NormalNN --dataset geom-chameleon --udgraph  --log-detail --log-detailedCh --early-stop --dropout 0.2 --dropout2 0.6 --lr1 0.01 --lr2 0.01 --n-layers 16 --wd1 1e-4 --wd2 1e-3   --n-cv 20  --gpu 0  --es-ckpt e000   


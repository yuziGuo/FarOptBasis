# favard
python train_models_linkx.py  --gpu 1 --model FavardNormalNN --dataset twitch-gamer --udgraph  --log-detail --log-detailedCh --early-stop  --n-cv 5  --lr1 0.02 --lr2 0.04 --lr3 0.04 --wd1 1e-7 --wd2 1e-7 --wd3 1e-6 --n-layers 16 --dropout 0. --dropout2 0.9  --es-ckpt gamer.ckpt 1>far-gamer.log 2>far-gamer.err 

python train_models_linkx.py   --gpu 1   --model FavardNormalNN --dataset Penn94 --udgraph  --log-detail --log-detailedCh --early-stop  --n-cv 5  --lr1 0.03 --lr2 0.05 --lr3 0.05 --wd1 1e-5 --wd2 1e-7 --wd3 1e-8 --n-layers 20 --dropout 0.2 --dropout2 0.7  --es-ckpt penn94.ckpt 1>far-penm94.log 2>far-penn94.err 

python train_models_linkx.py --gpu 1  --model FavardNormalNN --dataset genius --udgraph  --log-detail --log-detailedCh --early-stop  --n-cv 5  --lr1 0.05 --lr2 0.001 --lr3 0.02 --wd1 1e-8 --wd2 1e-8 --wd3 1e-6 --n-layers 16 --dropout 0. --dropout2 0.2  --es-ckpt genius.ckpt 1>far-genius.log 2>far-genius.err 

# Optbasis
python train_models_linkx.py  --model NormalNN --dataset Penn94 --udgraph  --log-detail --log-detailedCh --early-stop  --n-cv 5  --lr1 0.04 --lr2 0.04 --wd1 1e-4 --wd2 1e-4 --n-layers 20 --dropout 0.1 --dropout2 0.6  --es-ckpt penn94.ckpt 1>penn94.log 2>penn94.err 

python train_models_linkx.py  --model NormalNN --dataset genius --udgraph  --log-detail --log-detailedCh --early-stop  --n-cv 5  --lr1 0.05 --lr2 0.02 --wd1 1e-8 --wd2 1e-7 --n-layers 20 --dropout 0. --dropout2 0.3 1>genius.log 2>genius.err 

python train_models_linkx.py  --model NormalNN --dataset twitch-gamer --udgraph  --log-detail --log-detailedCh --early-stop  --n-cv 5  --lr1 0.05 --lr2 0.01 --wd1 1e-7 --wd2 1e-5 --n-layers 20 --dropout 0. --dropout2 0.2 --gpu 1 --es-ckpt gamer.ckpt 1>gamer.log 2>gamer.err 


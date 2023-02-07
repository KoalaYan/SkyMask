#!/bin/bash
aggregation=(fedavg fltrust trim krum mkrum bulyan tolpegin skymask)
gpu=(4 4 5 5 5 6 6 6)
attack=(no label_flipped minmax_agnostic minsum_agnostic trim_attack krum_attack scaling)
source /data/home/yanpeishen/anaconda3/bin/activate
conda activate fl

for atk_i in {0..6};do
for agr_i in {0..7};do
export CUDA_VISIBLE_DEVICES=${gpu[agr_i]}
python main.py --aggregation ${aggregation[agr_i]} --net cnn --dataset FashionMNIST --niter 2000 --global_lr 0.5 --local_lr 0.5 --local_iter 1 --batch_size 32 --nworkers 100 --nbyz 20 --bias 0.5 --byz_type ${attack[atk_i]} &
done
wait
done
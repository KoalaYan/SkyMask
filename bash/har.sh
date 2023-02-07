#!/bin/bash
aggregation=aggregation=(fedavg fltrust trim krum mkrum bulyan tolpegin skymask)
gpu=(6 6 6 6 6 6)
attack=(no trim_attack krum_attack label_flipped minmax_agnostic minsum_agnostic)
source /data/home/yanpeishen/anaconda3/bin/activate
conda activate fl

for atk_i in {0..6};do
for agr_i in {0..6};do
export CUDA_VISIBLE_DEVICES=${gpu[agr_i]}
python serial.py --aggregation ${aggregation[agr_i]} --net LR --dataset HAR --niter 1500 --lr 0.5 --local_iter 1 --batch_size 32 --nworkers 30 --nbyz 6 --byz_type ${attack[atk_i]} &
done
wait
done
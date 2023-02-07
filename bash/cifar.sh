#!/bin/bash
aggregation=(fedavg fltrust trim krum mkrum bulyan tolpegin skymask)
gpu=(1 2 2 2 3 3 3 4)
p_list=(0.1 0.2 0.4 0.6 0.8 0.95)
attack=(no label_flipped minmax_agnostic minsum_agnostic trim_attack krum_attack scaling)
source /data/home/yanpeishen/anaconda3/bin/activate
conda activate fl

# for p_i in {1..5};do
for atk_i in {0..6};do
for agr_i in {0..7};do
export CUDA_VISIBLE_DEVICES=${gpu[agr_i]}
python main.py --aggregation ${aggregation[agr_i]} --net resnet20 --dataset CIFAR-10 --niter 500 --global_lr 0.5 --local_lr 0.5 --local_iter 1 --batch_size 64 --nworkers 100 --nbyz 20 --bias 0.5 --byz_type ${attack[atk_i]} &
done
wait
done
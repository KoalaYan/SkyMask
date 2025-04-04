# SkyMask: Attack-agnostic Robust Federated Learning with Fine-grained Learnable Masks.

This repository contains the Official Pytorch Implementation for [ECCV'24] SkyMask: Attack-agnostic Robust Federated Learning with Fine-grained Learnable Masks.

If the project is useful to you, please give us a star. ⭐️

## Framework
![arch-horizon](https://github.com/KoalaYan/SkyMask/assets/48938458/b16f445b-aba8-40a2-93cf-1023e82a1671)


## Preparation 

### Install packages

```
pip install -r requirements.txt
```

### Datasets

For Fashion-MNIST and CIFAR-10 datasets, they will be download automatically during runtime.

For Human Activity Recognition (HAR) dataset, please download `train.csv` and `test.csv` from https://www.kaggle.com/datasets/meetnagadia/human-action-recognition-har-dataset, then put them into `./data/HAR/` directory.

## Run


Below are the command-line arguments supported by the script:

| Parameter       | Type    | Default       | Description                                                                 |
|------------------|---------|---------------|-----------------------------------------------------------------------------|
| `--server_pc`    | `int`   | `100`         | The number of data samples held by the server.                             |
| `--dataset`      | `str`   | `FashionMNIST`| The dataset to use. Options: `FashionMNIST`, `CIFAR-10`, `HAR`.            |
| `--bias`         | `float` | `0.5`         | Degree of non-IID data distribution.                                       |
| `--net`          | `str`   | `cnn`         | The neural network architecture. Options: `cnn`, `resnet20`, `LR`.         |
| `--batch_size`   | `int`   | `32`          | Batch size for local training.                                                   |
| `--local_lr`     | `float` | `0.6`         | Local learning rate for federated learning.                                |
| `--global_lr`    | `float` | `0.6`         | Global learning rate for federated learning.                               |
| `--nworkers`     | `int`   | `100`         | Number of workers (clients) in federated learning.                         |
| `--niter`        | `int`   | `2500`        | Number of global iterations.                                               |
| `--nbyz`         | `int`   | `20`          | Number of Byzantine (malicious) workers.                                   |
| `--byz_type`     | `str`   | `no`          | Type of Byzantine attack. Options: `no`, `trim_attack`, `krum_attack`, etc.|
| `--aggregation`  | `str`   | `fedavg`      | Aggregation method. Options: `fedavg`, etc.                                |
| `--p`            | `float` | `0.1`         | Bias probability of 1 in server samples.                                   |
| `--local_iter`   | `int`   | `5`           | Number of local iterations per worker.                                     |
| `--thres`        | `float` | `0.5`         | Threshold for mask application.                                            |

### Example Usage

```
python main.py --aggregation skymask --net resnet20 --dataset CIFAR-10 --niter 500 --global_lr 0.5 --local_lr 0.5 --local_iter 1 --batch_size 64 --nworkers 100 --nbyz 20 --bias 0.5 --byz_type minmax_agnostic
```


## Citation
```
@inproceedings{yan2024skymask,
  title={SkyMask: Attack-agnostic robust federated learning with fine-grained learnable masks},
  author={Yan, Peishen and Wang, Hao and Song, Tao and Hua, Yang and Ma, Ruhui and Hu, Ningxin and Haghighat, Mohammad Reza and Guan, Haibing},
  booktitle={European Conference on Computer Vision},
  pages={291--308},
  year={2024},
  organization={Springer}
}
```


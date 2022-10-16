#!/bin/sh
# export CUDA_PATH=/home/lab/yangDir/cuda-9.0
# export PATH=/home/lab/yangDir/cuda-9.0/bin:$PATH
# export LD_LIBRARY_PATH=/home/lab/yangDir/cuda-9.0/lib64:$LD_LIBRARY_PATH


# python trainval_net_global_local.py --cuda --net res50 --dataset SYN_NWPU_C1 --dataset_t REAL_NWPU_C1 --gc --lc 


python trainval_net_global_local.py --cuda --device cuda:0 --net res50 --dataset synthetic_data_wdt --dataset_t xilin_wdt --gc --lc --bs 1

# python trainval_net_global_local_batch.py --cuda --device cuda:2 --net res50 --dataset synthetic_data_wdt --dataset_t xilin_wdt --gc --lc --bs 4

# python trainval_net_global_local_batch.py --cuda --device cuda:2 --net res50 --dataset SYN_NWPU_C1 --dataset_t REAL_NWPU_C1 --gc --lc --bs 4
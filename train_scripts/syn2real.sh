#!/bin/sh
# export CUDA_PATH=/home/lab/yangDir/cuda-9.0
# export PATH=/home/lab/yangDir/cuda-9.0/bin:$PATH
# export LD_LIBRARY_PATH=/home/lab/yangDir/cuda-9.0/lib64:$LD_LIBRARY_PATH


python trainval_net_global_local.py --cuda --net res50 --dataset SYN_NWPU_C1 --dataset_t REAL_NWPU_C1 --gc --lc 
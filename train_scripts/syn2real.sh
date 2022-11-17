#!/bin/sh
# export CUDA_PATH=/home/lab/yangDir/cuda-9.0
# export PATH=/home/lab/yangDir/cuda-9.0/bin:$PATH
# export LD_LIBRARY_PATH=/home/lab/yangDir/cuda-9.0/lib64:$LD_LIBRARY_PATH

# CUDA_VISIBLE_DEVICES=3
# python trainval_net_global_local.py --cuda --net res50 --dataset SYN_NWPU_C1 --dataset_t REAL_NWPU_C1 --gc --lc 


# python trainval_net_global_local.py --cuda --device cuda:0 --net res50 --dataset synthetic_data_wdt --dataset_t xilin_wdt --gc --lc --bs 1 

# python trainval_net_global_local.py --cuda --device cuda:0 --net res50 --dataset synthetic_data_wdt --dataset_t xilin_wdt --gc --lc --use_tfb --bs 1 --epochs 20 --lr 1e-3

########## global + local
# python trainval_net_global_local.py --cuda --device 3 --net res101 --dataset synthetic_data_wdt --dataset_t xilin_wdt --gc --lc --use_tfb --bs 1 --epochs 30 --lr 1e-3

########### local
# python trainval_net_local.py --cuda --device 2 --net res101 --dataset synthetic_data_wdt --dataset_t xilin_wdt --lc  --use_tfb --bs 1 --epochs 30 --lr 1e-3

########### global 
#---------------dataseed17
# python trainval_net_global.py --cuda --device 1 --net res101 --dataset synthetic_data_wdt --dataset_t xilin_wdt --gc --use_tfb --bs 1 --epochs 30 --lr 1e-3
#---------------dataseed1
# python trainval_net_global.py --cuda --device 3 --net res101 --dataset synthetic_data_wdt --dataset_t xilin_wdt --gc --use_tfb --bs 1 --epochs 30 --lr 1e-3

#---------------dataseed2
# python trainval_net_global.py --cuda --device 1 --net res101 --dataset synthetic_data_wdt --dataset_t xilin_wdt --gc --use_tfb --bs 1 --epochs 30 --lr 1e-3

# python trainval_net_global_local_batch.py --cuda --device cuda:1 --net res50 --dataset synthetic_data_wdt --dataset_t xilin_wdt --gc --lc --bs 4

# python trainval_net_global_local_batch.py --cuda --device cuda:2 --net res50 --dataset SYN_NWPU_C1 --dataset_t REAL_NWPU_C1 --gc --lc --bs 4


####----------global+local-----opt------------------------------############
# python trainval_net_global_local_opt.py --cuda --device 3 --net res50 --dataset synthetic_data_wdt --dataset_t xilin_wdt --gc --lc --use_tfb --bs 1 --epochs 30 

####----------global-----opt ------------------------------############
# python trainval_net_global_opt.py --cuda --device 2 --net res50 --dataset synthetic_data_wdt --dataset_t xilin_wdt --gc  --use_tfb --bs 1 --epochs 30 

####----------local-----opt ------------------------------############
# python trainval_net_local_opt.py --cuda --device 1 --net res50 --dataset synthetic_data_wdt --dataset_t xilin_wdt --lc  --use_tfb --bs 1 --epochs 30 
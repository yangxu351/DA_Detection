#!/bin/sh
# CUDA_VISIBLE_DEVICES=$1 python test_net_global_local.py --cuda --net vgg16 --dataset cityscape_car --gc --lc --load_name $2

python test_net_global_local.py --cuda --net res50 --dataset_test xilin_wdt --gc --lc --load_name net_res50_target_xilin_wdt_lr0.0001_eta_0.1_lcTrue_glTrue_gamma_5_session_1_epoch_50.pth
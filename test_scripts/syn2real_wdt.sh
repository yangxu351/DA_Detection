#!/bin/sh
# CUDA_VISIBLE_DEVICES=$1 python test_net_global_local.py --cuda --net vgg16 --dataset cityscape_car --gc --lc --load_name $2

# python test_net_global_local.py --cuda --net res50 --database_test xilin_wdt --gc --lc --load_name net_res50_target_xilin_wdt_lr0.0001_eta_0.1_lcTrue_glTrue_gamma_5_session_1_epoch_20.pth

# python test_net_global_local.py --cuda --net res50 --database_test xilin_wdt --gc --lc --load_name 20221016_2203/target_xilin_wdt_lr0.001_bs1_eta_0.1_lcTrue_glTrue_gamma_5_session_1_epoch_20_noflip.pth

# python test_net_local.py --cuda --net res50 --database_test xilin_wdt --gc --lc --load_name target_xilin_wdt_lr0.001_bs1_eta_0.1_lcTrue_glTrue_gamma_5_session_1_epoch_30_flipFalse.pth

# python test_net_global_local.py --cuda --net res50 --database_test xilin_wdt --gc --lc --load_name  net_res50_target_xilin_wdt_lr0.0001_eta_0.1_lcTrue_glTrue_gamma_5_session_1_epoch_50.pth

''' global + local '''
# python test_net_global_local.py --cuda --net res101 --database_test xilin_wdt --gc --lc --load_name 20221020_0107/target_xilin_wdt_lr0.001_bs1_eta_0.1_lcTrue_gcTrue_gamma_5_session_1_epoch_30_flipFalse.pth # 0.1090
# python test_net_global_local.py --cuda --net res101 --database_test xilin_wdt --gc --lc --load_name 20221021_0602/target_xilin_wdt_lr0.001_bs1_eta_0.1_lcTrue_gcTrue_gamma_5_session_1_epoch_30_flipFalse.pth # 0.094
# python test_net_global_local.py --cuda --net res101 --database_test xilin_wdt --gc --lc --load_name 20221021_0602/target_xilin_wdt_lr0.001_bs1_eta_0.1_lcTrue_gcTrue_gamma_5_session_1_epoch_20_flipFalse.pth # 0.0939

''' global + local  for aug val'''
# python test_net_global_local.py --cuda --net res101 --database_test xilin_wdt --aug --gc --lc --load_name 20221020_0107/target_xilin_wdt_lr0.001_bs1_eta_0.1_lcTrue_gcTrue_gamma_5_session_1_epoch_20_flipFalse.pth # 0.103
# python test_net_global_local.py --cuda --net res101 --database_test xilin_wdt --aug --gc --lc --load_name 20221020_0107/target_xilin_wdt_lr0.001_bs1_eta_0.1_lcTrue_gcTrue_gamma_5_session_1_epoch_10_flipFalse.pth # 0.105
#####----------change optimizer scheduler
#  python test_net_global_local.py --cuda --net res50 --database_test xilin_wdt --aug --gc --lc --load_name 20221112_0907/target_xilin_wdt_lr0.0001_bs1_eta_0.1_lcTrue_gcTrue_gamma_5_session_1_epoch_30_flipFalse.pth # 0.1114

''' local '''
# python test_net_local.py --cuda --net res101 --database_test xilin_wdt --lc --load_name 20221020_0109/local_target_xilin_wdt_lr0.001_bs1_eta_0.1_lcTrue_gamma_5_session_1_epoch_30_flipFalse.pth # 0.1737
# python test_net_local.py --cuda --net res101 --database_test xilin_wdt --lc --load_name 20221021_0605/local_target_xilin_wdt_lr0.001_bs1_eta_0.1_lcTrue_gamma_5_session_1_epoch_30_flipFalse.pth # 0.1609
# python test_net_local.py --cuda --net res101 --database_test xilin_wdt --lc --load_name 20221021_0605/local_target_xilin_wdt_lr0.001_bs1_eta_0.1_lcTrue_gamma_5_session_1_epoch_20_flipFalse.pth # 0.1633
# python test_net_local.py --cuda --net res101 --database_test xilin_wdt --lc --load_name 20221021_0605/local_target_xilin_wdt_lr0.001_bs1_eta_0.1_lcTrue_gamma_5_session_1_epoch_30_flipFalse.pth # 0.1609

'''local  for aug val'''
# python test_net_local.py --cuda --net res101 --database_test xilin_wdt --aug --lc --load_name 20221020_0109/local_target_xilin_wdt_lr0.001_bs1_eta_0.1_lcTrue_gamma_5_session_1_epoch_20_flipFalse.pth # 0.108
# python test_net_local.py --cuda --net res101 --database_test xilin_wdt --aug --lc --load_name 20221020_0109/local_target_xilin_wdt_lr0.001_bs1_eta_0.1_lcTrue_gamma_5_session_1_epoch_10_flipFalse.pth # 0.116
#####----------change optimizer scheduler
# python test_net_local.py --cuda --net res50 --database_test xilin_wdt --aug --lc --load_name 20221112_0935/local_target_xilin_wdt_lr0.0001_bs1_eta_0.1_lcTrue_gamma_5_session_1_epoch_30_flipFalse.pth # 0.1165

''' global '''
# python test_net_global.py --cuda --net res101 --database_test xilin_wdt --gc --load_name 20221020_0110/global_target_xilin_wdt_eta_0.1_efocal_False_gc_True_gamma_5_session_1_epoch_30_flipFalse.pth

# python test_net_global.py --cuda --net res101 --database_test xilin_wdt --gc --load_name 20221021_0606/global_target_xilin_wdt_eta_0.1_efocal_False_gc_True_gamma_5_session_1_epoch_20_flipFalse.pth # 0.1910

# python test_net_global.py --cuda --net res101 --database_test xilin_wdt --gc --load_name 20221021_0606/global_target_xilin_wdt_eta_0.1_efocal_False_gc_True_gamma_5_session_1_epoch_30_flipFalse.pth # 0.1930

'''global for aug val dataseed-1'''
# python test_net_global.py --cuda --net res101 --database_test xilin_wdt --aug --gc --load_name 20221021_0606/global_target_xilin_wdt_eta_0.1_efocal_False_gc_True_gamma_5_session_1_epoch_20_flipFalse.pth # 0.115

python test_net_global.py --cuda --net res101 --database_test xilin_wdt --aug --gc --load_name 20221021_0606/global_target_xilin_wdt_eta_0.1_efocal_False_gc_True_gamma_5_session_1_epoch_10_flipFalse.pth # 0.132

# python test_net_global.py --cuda --net res101 --database_test xilin_wdt --aug --gc --load_name 20221021_0606/global_target_xilin_wdt_eta_0.1_efocal_False_gc_True_gamma_5_session_1_epoch_30_flipFalse.pth # 0.110

'''global for aug val dataseed1'''
# python test_net_global.py --cuda --net res101 --database_test xilin_wdt --aug --gc --load_name 20221116_1219_data_seed1/global_target_xilin_wdt_eta_0.1_efocal_False_gc_True_gamma_5_session_1_epoch_20_flipFalse.pth  # 0.0051

'''global for aug val dataseed2'''
# python test_net_global.py --cuda --net res101 --database_test xilin_wdt --aug --gc --load_name 20221116_1221_data_seed2/global_target_xilin_wdt_eta_0.1_efocal_False_gc_True_gamma_5_session_1_epoch_30_flipFalse.pth
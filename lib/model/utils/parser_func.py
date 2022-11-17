import argparse
from model.utils.config import cfg, cfg_from_file, cfg_from_list
from lib.datasets.config_dataset import cfg_d
# from model.utils.parameters import *

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    ###########################################################=========data set
    parser.add_argument('--dataset', dest='dataset',
                        help='source training dataset: SYN_NWPU_C1, synthetic_data_wdt',
                        default=cfg_d.DATA_DIR_S, type=str)
    parser.add_argument('--dataset_dir_t', dest='dataset_dir_t',
                        help='target data dir: wind_turebine',
                        default=cfg_d.DATA_DIR_T, type=str)

    parser.add_argument('--database', dest='database',
                        help='source training database: syn_nwpu_bkg_shdw_rndsolar_sizefactor1_multimodels_negtrn_fixsigma_C1_v6',
                        default=cfg_d.DATABASE, type=str)      

    parser.add_argument('--dataset_t', dest='dataset_t',
                        help='target training dataset:REAL_NWPU_C1, xilin_wdt',
                        default=cfg_d.DATABASE_T, type=str)
    # tag: for test dataset                            
    parser.add_argument('--database_test', dest='database_test',
                        help='test dataset:REAL_NWPU_C1, xilin_wdt',
                        default=cfg_d.DATABASE_TEST, type=str)                        
    ###########################################################=========end data set
    # synthetic dataseed
    #tag: yang adds
    parser.add_argument('--data_seed', dest='data_seed',
                        help='synthetic dataseed',
                        default=cfg_d.DATA_SEED, type=int)

    parser.add_argument('--net', dest='net',
                        help='vgg16, res101 res50',
                        default='res101', type=str)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=30, type=int)
    parser.add_argument('--gamma', dest='gamma',
                        help='value of gamma',
                        default=5, type=float)
    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='number of iterations to display',
                        default=10, type=int)
    parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                        help='number of iterations to display 10000',
                        default=100, type=int)
    ###########################################################=========dir
    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models', default="models",
                        type=str)
    parser.add_argument('--log_dir', dest='log_dir',
                        help='directory to save log', default="logs",
                        type=str)
    parser.add_argument('--image_dir', dest='image_dir',
                        help='directory to load images for demo',
                        default="images")  #??????????                     
    ###########################################################=========end dir
    parser.add_argument('--load_name', dest='load_name',
                        help='time/net_res50_target_xilin_wdt_lr0.0001_eta_0.1_lcTrue_glTrue_gamma_5_session_1_epoch_50.pth', default="",
                        type=str)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=4, type=int)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA', default=True,
                        action='store_true')
    #fixme:
    # parser.add_argument('--device', dest='device', 
    #                     help=' CUDA device', default='cuda:1',
    #                     type=str)
    parser.add_argument('--device', dest='device', 
                        help=' CUDA device', default=3,
                        type=int)

    parser.add_argument('--detach', dest='detach',
                        help='whether use detach',
                        action='store_false')
    parser.add_argument('--ef', dest='ef',
                        help='whether use exponential focal loss',
                        action='store_true')
    parser.add_argument('--lc', dest='lc', 
                        help='whether use context vector for pixel level',
                        action='store_true')
    parser.add_argument('--gc', dest='gc', 
                        help='whether use context vector for global level',
                        action='store_true')
    # tag:for val aug dataset
    parser.add_argument('--aug', dest='aug', default=True,
                        help='whether use aug val',
                        action='store_true')

    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true', default=False)
    # fixme:?                    
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')

    # config optimization
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="sgd", type=str)
    # tag:                     
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate 0.001',
                        default=0.0001, type=float)
    parser.add_argument('--eta', dest='eta',
                        help='trade-off parameter between detection loss and domain-alignment loss. Used for Car datasets',
                        default=0.1, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=5, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)
    parser.add_argument('--s', dest='session',
                        help='training session',
                        default=1, type=int)
    parser.add_argument('--r', dest='resume',
                        help='resume checkpoint or not',
                        default=False, type=bool)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load model',
                        default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load model',
                        default=0, type=int)
    # log and diaplay
    parser.add_argument('--use_tfb', dest='use_tfboard',
                        help='whether use tensorboard',
                        action='store_true')
    
    args = parser.parse_args()
    return args

def set_dataset_args(args, test=False):
    if not test:
        if args.dataset == "pascal_voc":
            args.imdb_name = "voc_2007_trainval"
            args.imdbval_name = "voc_2007_test"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "pascal_voc_water":
            args.imdb_name = "voc_water_2007_trainval+voc_water_2012_trainval"
            args.imdbval_name = "voc_clipart_2007_trainval+voc_clipart_2012_trainval"
            args.imdb_name_cycle = "voc_cyclewater_2007_trainval+voc_cyclewater_2012_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "pascal_voc_cycleclipart":
            args.imdb_name = "voc_cycleclipart_2007_trainval+voc_cycleclipart_2012_trainval"
            args.imdbval_name = "voc_cycleclipart_2007_trainval+voc_cycleclipart_2012_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "pascal_voc_cyclewater":
            args.imdb_name = "voc_cyclewater_2007_trainval+voc_cyclewater_2012_trainval"
            args.imdbval_name = "voc_cyclewater_2007_trainval+voc_cyclewater_2012_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "pascal_voc_0712":
            args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
            args.imdbval_name = "voc_2007_test"
            args.imdb_name_cycle = "voc_cycleclipart_2007_trainval+voc_cycleclipart_2012_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "foggy_cityscape":
            args.imdb_name = "foggy_cityscape_trainval"
            args.imdbval_name = "foggy_cityscape_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES',
                             '30']
        elif args.dataset == "vg":
            args.imdb_name = "vg_150-50-50_minitrain"
            args.imdbval_name = "vg_150-50-50_minival"
            args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
        elif args.dataset == "cityscape":
            args.imdb_name = "cityscape_trainval"
            args.imdbval_name = "cityscape_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES',
                             '30']
        elif args.dataset == "sim10k":
            args.imdb_name = "sim10k_train"
            args.imdbval_name = "sim10k_train"
            args.imdb_name_cycle = "sim10k_cycle_train"  # "voc_cyclewater_2007_trainval+voc_cyclewater_2012_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
        elif args.dataset == "sim10k_cycle":
            args.imdb_name = "sim10k_cycle_train"
            args.imdbval_name = "sim10k_cycle_train"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
        # tag: yang adds
        elif args.dataset == "SYN_NWPU_C1":
            args.imdb_name = args.dataset + "_TRAIN"
            args.imdbval_name =args.dataset + "_VAL"
            # args.imdb_name_cycle = "sim10k_cycle_train"  # "voc_cyclewater_2007_trainval+voc_cyclewater_2012_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
        # tag: for wdt
        elif args.dataset == "synthetic_data_wdt":
            args.imdb_name = args.dataset + "_train"
            args.imdbval_name =args.dataset + "_val"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
        
        ## cityscape dataset for only car classes.
        # elif args.dataset == "cityscape_kitti":
        #     args.imdb_name = "cityscape_kitti_trainval"
        #     args.imdbval_name = "cityscape_kitti_trainval"
        #     args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES',
        #                      '30']
        if args.dataset_t == "water":
            args.imdb_name_target = "water_train"
            args.imdbval_name_target = "water_train"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES',
                             '20']
        elif args.dataset_t == "clipart":
            args.imdb_name_target = "clipart_trainval"
            args.imdbval_name_target = "clipart_test"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES',
                                    '20']
        elif args.dataset_t == "cityscape":
            args.imdb_name_target = "cityscape_trainval"
            args.imdbval_name_target = "cityscape_trainval"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES',
                                    '30']
        ## cityscape dataset for only car classes.
        elif args.dataset_t == "cityscape_car":
            args.imdb_name_target = "cityscape_car_trainval"
            args.imdbval_name_target = "cityscape_car_trainval"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES',
                                    '20']
        # elif args.dataset_t == "kitti":
        #     args.imdb_name_target = "kitti_trainval"
        #     args.imdbval_name_target = "kitti_trainval"
        #     args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES',
        #                             '20']
        elif args.dataset_t == "foggy_cityscape":
            args.imdb_name_target = "foggy_cityscape_trainval"
            args.imdbval_name_target = "foggy_cityscape_trainval"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES',
                                    '30']
        # tag: yang adds
        elif args.dataset_t == "REAL_NWPU_C1":
            args.imdb_name_target = args.dataset_t + "_TEST"
            args.imdbval_name_target = args.dataset_t + "_TEST"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']  
        # tag: for wdt
        #fixme:???? whether set imdb and imdbval the same??????
        elif args.dataset_t == "xilin_wdt":
            args.imdb_name_target = args.dataset_t + "_train"
            args.imdbval_name_target = args.dataset_t + "_val"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']                                                        
    else:
        if args.dataset == "pascal_voc":
            args.imdb_name = "voc_2007_trainval"
            args.imdbval_name = "voc_2007_test"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
        elif args.dataset == "pascal_voc_0712":
            args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
            args.imdbval_name = "voc_2007_test"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
        elif args.dataset == "sim10k":
            args.imdb_name = "sim10k_val"
            args.imdbval_name = "sim10k_val"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES',
                             '30']
        # tag: yang adds                             
        elif args.database_test == "xilin_wdt":
            args.imdb_name =  args.database_test + "_val"
            if args.aug:
                args.imdbval_name = args.database_test + "_aug_val"
            else:
                args.imdbval_name = args.database_test + "_val"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES',
                             '30']
        elif args.dataset == "cityscape":
            args.imdb_name = "cityscape_val"
            args.imdbval_name = "cityscape_val"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
        elif args.dataset == "foggy_cityscape":
            args.imdb_name = "foggy_cityscape_test"
            args.imdbval_name = "foggy_cityscape_test"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
        elif args.dataset == "cityscape_kitti":
            args.imdb_name = "cityscape_kitti_val"
            args.imdbval_name = "cityscape_kitti_val"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
        elif args.dataset == "water":
            args.imdb_name = "water_test"
            args.imdbval_name = "water_test"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES',
                                '20']
        elif args.dataset == "clipart":
            args.imdb_name = "clipart_trainval"
            args.imdbval_name = "clipart_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES',
                             '20']
        elif args.dataset == "cityscape_car":
            args.imdb_name = "cityscape_car_val"
            args.imdbval_name = "cityscape_car_val"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES',
                                    '20']
    args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

    return args

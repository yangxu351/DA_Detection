'''
https://github.com/JPM-Tech/Object-Detection/blob/main/Scripts/converters/convert-yolo-to-xml.py
creater xuyang_ustb@163.com
xview process
in order to generate xivew 2-d background with synthetic airplances
'''
import glob
import numpy as np
import random
import argparse
import os
import sys
sys.path.append('.')
import pandas as pd
import shutil
from PIL import Image

from lib.datasets.object_score_util import get_bbox_coords_from_annos_with_object_score as gbc

IMG_FORMAT = '.jpg'
TXT_FORMAT = '.txt'
XML_FORMAT = '.xml'


def generate_xml_from_real_annotation(args, data_folder='REAL_NWPU_C1', data_cat='NWPU_C1'):
    #tag: adjust for wdt

    yolo_dila_annos_files = []
    if '0' in data_folder: # background
        imgs_dir = os.path.join(args.real_base_dir, 'all_negative_image_set')
        annos_dir = os.path.join(args.real_base_dir, 'all_negative_label_set')
    else:
        annos_dir = args.real_yolo_annos_dir
        imgs_dir = args.real_img_dir

    yolo_dila_annos_files = glob.glob(os.path.join(annos_dir, f'*{TXT_FORMAT}'))
    print('annos', len(yolo_dila_annos_files))
    
    if not os.path.exists(args.real_voc_annos_dir):
        os.makedirs(args.real_voc_annos_dir)
    else:
        shutil.rmtree(args.real_voc_annos_dir)
        os.makedirs(args.real_voc_annos_dir)

    print('valid annos', len(yolo_dila_annos_files))
    cnt = 0
    for ix, f in enumerate(yolo_dila_annos_files):
        img_name = os.path.basename(f).replace(TXT_FORMAT, IMG_FORMAT)
        img_f = os.path.join(imgs_dir, img_name)
        if not gbc.is_non_zero_file(f):
            df = pd.DataFrame([])
        else:
            df = pd.read_csv(f, header=None, sep=' ').to_numpy()
            print('dila_anno', df.shape)
            #xcycwh are float relative values, xminyminxmaxymax are absolute values
            min_ws = np.clip((df[:, 1] - df[:, 3]/2)*args.tile_size, 0, args.tile_size-1).astype(np.int32)
            min_hs = np.clip((df[:, 2] - df[:, 4]/2)*args.tile_size, 0, args.tile_size-1).astype(np.int32)
            max_ws = np.clip((df[:, 1] + df[:, 3]/2)*args.tile_size, 0, args.tile_size-1).astype(np.int32)
            max_hs = np.clip((df[:, 2] + df[:, 4]/2)*args.tile_size, 0, args.tile_size-1).astype(np.int32)
        
        orig_img = Image.open(img_f)
        image_width = orig_img.width
        image_height = orig_img.height
        
        cnt += 1
        xml_file = open(os.path.join(args.real_voc_annos_dir, img_name.replace(IMG_FORMAT, '.xml')), 'w')
        xml_file.write('<annotation>\n')
        # xml_file.write('\t<folder>'+ folder +'</folder>\n')
        xml_file.write('\t<filename>' + img_name + '</filename>\n')
        xml_file.write('\t<path>' + img_f + '</path>\n')
        xml_file.write('\t<source>\n')
        xml_file.write('\t\t<database>'+ data_folder + '</database>\n')
        xml_file.write('\t</source>\n')
        xml_file.write('\t<size>\n')
        xml_file.write('\t\t<width>' + str(image_width) + '</width>\n')
        xml_file.write('\t\t<height>' + str(image_height) + '</height>\n')
        xml_file.write('\t\t<depth>3</depth>\n') # assuming a 3 channel color image (RGB)
        xml_file.write('\t</size>\n')
        xml_file.write('\t<segmented>'+ str(1) +'</segmented>\n')

        for j in range(df.shape[0]):
            x_min, y_min, x_max, y_max = min_ws[j], min_hs[j], max_ws[j], max_hs[j]
            if x_min >= x_max or y_min >= y_max:
                continue
            #tag: adjust for wdt
            if 'NWPU' in data_folder:
                cat_id = int(df[j, 5] + 1) # change the cat id, start from 1, new cat_id=0--> background
                new_data_cat = data_cat[:-1] + str(cat_id) # NWPU_C*
            else:
                cat_id = int(df[j,  0] + 1) # new cat_id=0--> background
                new_data_cat = data_cat
            # write each object to the file
            xml_file.write('\t<object>\n')
            xml_file.write('\t\t<name>' + new_data_cat + '</name>\n')
            xml_file.write('\t\t<pose>Unspecified</pose>\n')
            xml_file.write('\t\t<truncated>0</truncated>\n')
            xml_file.write('\t\t<difficult>0</difficult>\n')
            xml_file.write('\t\t<bndbox>\n')
            xml_file.write('\t\t\t<xmin>' + str(x_min) + '</xmin>\n')
            xml_file.write('\t\t\t<ymin>' + str(y_min) + '</ymin>\n')
            xml_file.write('\t\t\t<xmax>' + str(x_max) + '</xmax>\n')
            xml_file.write('\t\t\t<ymax>' + str(y_max) + '</ymax>\n')
            xml_file.write('\t\t</bndbox>\n')
            xml_file.write('\t</object>\n')
            
        # Close the annotation tag once all the objects have been written to the file
        xml_file.write('</annotation>\n')
        xml_file.close() # Close the file
    print('cnt', cnt)
    print('finished!!!')
    

def draw_bbx_on_rgb_images():
    img_path = args.real_img_dir
    print('img_path', img_path)
    img_files = np.sort(glob.glob(os.path.join(img_path, '*{}'.format(IMG_FORMAT))))
    img_names = [os.path.basename(f) for f in img_files]
    print('images: ', len(img_names))
    annos_path = args.real_voc_annos_dir

    bbox_folder_name = 'xml_bbox'
    save_bbx_path = os.path.join(args.real_base_dir, bbox_folder_name)
    if not os.path.exists(save_bbx_path):
        os.makedirs(save_bbx_path)
    else:
        shutil.rmtree(save_bbx_path)
        os.makedirs(save_bbx_path)

    for ix, f in enumerate(img_files[:10]):
        xml_file = os.path.join(annos_path, img_names[ix].replace(IMG_FORMAT, XML_FORMAT))
        if not os.path.exists(xml_file):
            continue
        # print('xml_file', xml_file)
        gbc.plot_img_with_bbx_from_xml(f, xml_file, save_bbx_path)


def split_real_nwpu_background_trn_val(seed=17, data_folder='real_nwpu_c0', data_cat='NWPU_C0'):
    data_folder = data_folder.upper()
    data_dir = args.workdir_data_txt.format(data_folder)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    bkg_img_dir = os.path.join(args.real_base_dir, 'all_negative_image_set')
    bkg_files = np.sort(glob.glob(os.path.join(bkg_img_dir, '*'+IMG_FORMAT)))
    num_bkg_all = len(bkg_files)
    bkg_xml_dir = args.real_voc_annos_dir
    print('num_files', num_bkg_all)
    
    data_txt = open(os.path.join(data_dir, f'path.data'), 'w')
    data_txt.write(f'img_dir={bkg_img_dir}\n')
    data_txt.write(f'lbl_dir={bkg_xml_dir}\n')
    data_txt.write(f'class_cat={data_cat}')
    data_txt.close()

    trn_txt = open(os.path.join(data_dir, 'train_seed{}.txt'.format(seed)), 'w')
    val_txt = open(os.path.join(data_dir, 'val_seed{}.txt'.format(seed)), 'w')
    
    np.random.seed(seed)
    num_bkg_val = int(num_bkg_all*args.val_percent)
    val_bkg_indexes = np.random.choice(num_bkg_all, num_bkg_val, replace=False)

    num_trn = num_bkg_all - num_bkg_val
    print('num_trn', num_trn)

    for j, f in enumerate(bkg_files):
    #        print('all_files[i]', all_files[j])
        if j in val_bkg_indexes:
            val_txt.write('%s\n' % os.path.basename(f))
        else:
            trn_txt.write('%s\n' % os.path.basename(f))
    val_txt.close()
    trn_txt.close()


def split_real_data_train_val(data_seed = 0, data_cat='WindTurbine', aug=False):

    files = glob.glob(os.path.join(args.real_voc_annos_dir, '*.xml'))
    files_name = sorted([os.path.basename(f).split(".")[0] for f in files])
    files_num = len(files_name)
    #tag: for validation
    # for f in files:
    #     file_name = os.path.basename(f)
    #     print(file_name)
    random.seed(data_seed)
    val_index = random.sample(range(0, files_num), k=int(files_num*args.val_percent))
    # print('val_index', val_index)
    train_img_files = []
    val_img_files = []
    train_lbl_files = []
    val_lbl_files = []
    for index, f in enumerate(files):
        img_name = os.path.basename(f).replace('.xml', '.jpg')
        if index in val_index:
            val_lbl_files.append(f)
            val_img_files.append(os.path.join(args.real_img_dir, img_name))
        else:
            train_lbl_files.append(f)
            train_img_files.append(os.path.join(args.real_img_dir, img_name))
    # tag: for validation
    # print('len val files', len(val_lbl_files))
    print('val_file_names', [os.path.basename(f).split('.')[0] for f in val_lbl_files])

    try:
        if not os.path.exists(args.workdir_data_txt):
            os.makedirs(args.workdir_data_txt)
        all_f = open(os.path.join(args.workdir_data_txt, "all.txt"), "w")
        train_img_f = open(os.path.join(args.workdir_data_txt, f"train_img_seed{data_seed}.txt"), "w")
        val_img_f = open(os.path.join(args.workdir_data_txt, f"val_img_seed{data_seed}.txt"), "w")
        train_img_f.write("\n".join(train_img_files))
        val_img_f.write("\n".join(val_img_files))
        train_lbl_f = open(os.path.join(args.workdir_data_txt, f"train_lbl_seed{data_seed}.txt"), "w")
        val_lbl_f = open(os.path.join(args.workdir_data_txt, f"val_lbl_seed{data_seed}.txt"), "w")
        train_lbl_f.write("\n".join(train_lbl_files))
        val_lbl_f.write("\n".join(val_lbl_files))
        all_f.write("\n".join(val_img_files+train_img_files))

        data_txt = open(os.path.join(args.workdir_data_txt, f'path_seed{data_seed}.data'), 'w')
        data_txt.write(f'img_dir={args.real_img_dir}\n')
        data_txt.write(f'lbl_dir={args.real_voc_annos_dir}\n')
        data_txt.write(f'class_cat={data_cat}')
        data_txt.close()
    except FileExistsError as e:
        print(e)
        exit(1)
    if aug:
        # tag: ori + aug files
        files = glob.glob(os.path.join(args.real_voc_annos_aug_dir, '*.xml'))
        val_aug_img_files = []
        val_aug_lbl_files = []
        print('len val aug files', len(files))
        for f in files:
            val_aug_lbl_files.append(f)
            img_name = os.path.basename(f).replace('.xml', '.jpg')
            val_aug_img_files.append(os.path.join(args.real_img_aug_dir, img_name))
        val_img_f = open(os.path.join(args.workdir_data_txt, f"val_img_aug_seed{data_seed}.txt"), "w")
        val_lbl_f = open(os.path.join(args.workdir_data_txt, f"val_lbl_aug_seed{data_seed}.txt"), "w")
        data_txt = open(os.path.join(args.workdir_data_txt, f'path_seed{data_seed}_aug.data'), 'w')
        try:
            val_img_f.write("\n".join(val_aug_img_files))
            val_lbl_f.write("\n".join(val_aug_lbl_files))
            data_txt.write(f'img_dir={args.real_img_aug_dir}\n')
            data_txt.write(f'lbl_dir={args.real_voc_annos_aug_dir}\n')
            data_txt.write(f'class_cat={data_cat}')
            data_txt.close()
        except FileExistsError as e:
            print(e)
            exit(1)

def create_data_combine_real_C_val_bkg(seed=17, data_cats=['real_nwpu_c1','real_nwpu_c0']):
    
    data_dir = args.workdir_data_txt.format(data_cats[0])
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    test_img_txt = open(os.path.join(data_dir, 'test_img_seed{}.txt'.format(seed)), 'w')
    test_lbl_txt = open(os.path.join(data_dir, 'test_lbl_seed{}.txt'.format(seed)), 'w')
    
    for data_cat in data_cats:
        data_cat = data_cat.upper()
        if 'C0' in data_cat:
            bkg_args = get_args(data_cat)
            img_dir = os.path.join(bkg_args.real_base_dir, 'all_negative_image_set')
            xml_dir = bkg_args.real_voc_annos_dir
            img_files = np.sort(glob.glob(os.path.join(img_dir, '*'+IMG_FORMAT)))
            num_all = len(img_files)
            np.random.seed(seed)
            num_bkg_val = int(num_all*bkg_args.val_percent)
            val_bkg_indexes = np.random.choice(num_all, num_bkg_val, replace=False)
            img_files = img_files[val_bkg_indexes]
            print('bkg num_files', len(img_files))
        else:
            img_dir = args.real_img_dir
            print('img_dir', img_dir)
            img_files = np.sort(glob.glob(os.path.join(img_dir, '*'+IMG_FORMAT)))
            num_all = len(img_files)
            xml_dir = args.real_voc_annos_dir
            print('num_files', num_all)
        for f in img_files:
            test_img_txt.write('%s\n' % f)
            test_lbl_txt.write('%s\n' % os.path.join(xml_dir, os.path.basename(f).replace(IMG_FORMAT, XML_FORMAT)))      
    test_img_txt.close()
    test_lbl_txt.close()

    data_txt = open(os.path.join(data_dir, 'data_list.data'), 'w')
    data_txt.write(f"test_img_file={os.path.join(data_dir, f'test_img_seed{seed}.txt')}\n")
    data_txt.write(f"test_lbl_file={os.path.join(data_dir, f'test_lbl_seed{seed}.txt')}\n")
    data_txt.write(f'class_set={data_cat}')
    data_txt.close()


def parse_data_cfg(path):
    # Parses the data configuration file
    if not os.path.exists(path) and os.path.exists('data' + os.sep + path):  # add data/ prefix if omitted
        path = 'data' + os.sep + path

    with open(path, 'r') as f:
        lines = f.readlines()

    options = dict()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, val = line.split('=')
        options[key.strip()] = val.strip()

    return options

def get_args(data_folder='REAL_NWPU_C1', data_cat='NWPU_C1'):
    parser = argparse.ArgumentParser()
    if 'NWPU' in data_folder:
        parser.add_argument("--real_base_dir", type=str,
                            help="base path of real data",
                            default=f'/data/users/yang/data/{data_folder}')

        parser.add_argument("--real_img_dir", type=str,
                            help="Path to folder containing real images ",
                            default='{}/{}_imgs_{}_all')

        parser.add_argument("--real_yolo_annos_dir", type=str, default='{}/{}_labels_xcycwh_all',
                            help="Path to folder containing yolo format annotations")  
                            
        parser.add_argument("--real_voc_annos_dir", type=str, default='{}/{}_all_annos_xml',
                            help="syn annos in voc format .xml \{real_base_dir\}/{cat}_all_xml_annos")     
                        
    else: #'wdt'#tag: adjust for wdt
        parser.add_argument("--workdir_data_txt", type=str, default=f'data/real_syn_wdt_vockit/{data_folder}',
                            help="syn related txt files data/real_syn_wdt_vockit/\{xilin_wdt\}")

        parser.add_argument("--real_base_dir", type=str,default='/data/users/yang/data/wind_turbine', help="base path of synthetic data")
        parser.add_argument("--real_img_dir", type=str, default='{}/{}_crop', help="Path to folder containing real images")
        parser.add_argument("--real_yolo_annos_dir", type=str, default='{}/{}_crop_label_xcycwh', help="Path to folder containing real annos of yolo format")
        parser.add_argument("--real_voc_annos_dir", type=str, default='{}/{}_crop_label_xml_annos', help="Path to folder containing real annos of yolo format")
        # tag: yang adds
        parser.add_argument("--real_img_aug_dir", type=str, default='{}/{}_crop_aug', help="Path to folder containing real images")
        parser.add_argument("--real_voc_annos_aug_dir", type=str, default='{}/{}_crop_label_xml_annos_aug', help="Path to folder containing real annos of voc format")

    #fixme ---***** min_region ***** change
    parser.add_argument("--tile_size", type=int, default=608, help="image size")
    parser.add_argument("--class_num", type=int, default=1, help="class number")
    parser.add_argument("--val_percent", type=float, default=0.3, help="train:val=0.7:0.3")
    
    args = parser.parse_args()
    args.real_img_dir = args.real_img_dir.format(args.real_base_dir, data_folder, args.tile_size)
    args.real_yolo_annos_dir = args.real_yolo_annos_dir.format(args.real_base_dir, data_folder)
    args.real_voc_annos_dir = args.real_voc_annos_dir.format(args.real_base_dir, data_folder)
    # tag: yang adds
    args.real_img_aug_dir = args.real_img_aug_dir.format(args.real_base_dir, data_folder, args.tile_size)
    args.real_voc_annos_aug_dir = args.real_voc_annos_aug_dir.format(args.real_base_dir, data_folder)

    return args


if __name__ == '__main__':

    '''
    generate txt and bbox for syn_background data
    bbox annotation meet certain conditions: px_thres, whr_thres
    '''
    ################################# 
    ######
    # data_cat = 'REAL_NWPU_C1'
    # # data_cat = 'REAL_NWPU_C0'
    # args = get_args(data_cat)
    # generate_xml_from_real_annotation(args, data_cat)

    ######### for wdt
    #tag: adjust for wdt
    # data_folder = 'xilin_wdt'
    # data_cat = 'WindTurbine'
    # args = get_args(data_folder, data_cat)
    # generate_xml_from_real_annotation(args, data_folder, data_cat)

    ''' split real dataset of wdt into train val'''
    data_folder = 'xilin_wdt'
    data_cat = 'WindTurbine'
    args = get_args(data_folder, data_cat)
    data_seed = 0
    aug = True
    # aug = False
    split_real_data_train_val(data_seed, data_cat, aug=aug)

    

    '''
    draw bbox on rgb images for syn_background data
    '''
    # data_cat = 'REAL_NWPU_C1'
    # syn_args = get_args(data_cat)
    # draw_bbx_on_rgb_images()


    ''' split background (NWPU_C0) data into train val '''
    # from datasets.config_dataset import cfg_d
    # seed = cfg_d.DATA_SEED
    # data_cat = 'REAL_NWPU_C0'
    # args = get_args(data_cat)
    # split_real_nwpu_background_trn_val(seed, data_cat)


    ''' combine real nwpu C* with background val C0 '''
    # from datasets.config_dataset import cfg_d
    # seed = cfg_d.DATA_SEED
    # data_cats=['REAL_NWPU_C1','REAL_NWPU_C0'] # backgroud C0 is the last
    # args = get_args(data_cats[0])
    # create_data_combine_real_C_val_bkg(seed, data_cats)
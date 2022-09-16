import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model.utils.config import cfg

if __name__ == '__main__':
    img_path = '/data/users/yang/data/SYN_NWPU_C1/syn_nwpu_bkg_shdw_rndsolar_sizefactor1_multimodels_negtrn_fixsigma_C1_v6/color_all_images_step304/color_C1_nwpu_background_sd309_1.png'

    save_dir = 'vis_util/save_figures'
    im = cv2.imread(img_path)

    fig, axes = plt.subplots(2,1)
    new_im = im - cfg.PIXEL_MEANS
    new_im  = np.clip(new_im,  0, 255)
    axes[0].imshow(im)
    axes[1].imshow(new_im)
    
    name = os.path.basename(img_path)
    name = name.split('sd')[-1]
    plt.savefig(os.path.join(save_dir, name))


# THCudaCheck FAIL file=/opt/conda/conda-bld/pytorch_1535493744281/work/aten/src/THC/THCGeneral.cpp line=663 error=11 : invalid argument
# [session 1][epoch  1][iter    0/10000] loss: 14.7387, lr: 1.00e-03
#                         fg/bg=(3/125), time cost: 0.550988
#                         rpn_cls: 0.6557, rpn_box: 0.9490, rcnn_cls: 13.0735, rcnn_box 0.0605 dloss s: 0.0295 dloss t: 0.0051 dloss s pixel: 0.1249 dloss t pixel: 0.1259 eta: 0.1000
# [session 1][epoch  1][iter  100/10000] loss: nan, lr: 1.00e-03
#                         fg/bg=(128/0), time cost: 22.918086
#                         rpn_cls: nan, rpn_box: nan, rcnn_cls: nan, rcnn_box nan dloss s: nan dloss t: nan dloss s pixel: 0.1250 dloss t pixel: 0.1250 eta: 0.1000
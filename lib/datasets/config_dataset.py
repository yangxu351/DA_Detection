from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__D = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg_d = __D

# __D.DATA_SEED = 17
# tag: yang changed
# __D.DATA_SEED = 0
# __D.DATA_SEED = 1
__D.DATA_SEED = 2
# tag: for WDT
__D.REAL_DATA_SEED = 0
#
# Training options
#with regard to pascal, the directories under the path will be ./VOC2007, ./VOC2012"
__D.PASCAL = "/VOCdevkit"
__D.PASCALCLIP = ""
__D.PASCALWATER = "/VOCdevkit"

#For these datasets, the directories under the path will be Annotations  ImageSets  JPEGImages."
__D.CLIPART = "/clipart"
__D.WATER = "/watercolor"
__D.SIM10K = "Sim10k/VOC2012"
__D.CITYSCAPE_CAR = "/VOC2007"
__D.CITYSCAPE = "VOC2007"
__D.FOGGYCITY = "VOC2007"

# __D.BASE_DATA_DIR = "/data/users/yang/data"
# __D.DEV_DATA_DIR = "data/real_syn_nwpu_vockit"
# #Tag:
# __D.DATASET = 'SYN_NWPU_C1'
# __D.DATABASE = 'syn_nwpu_bkg_shdw_rndsolar_sizefactor1_multimodels_negtrn_fixsigma_C1_v6'
# __D.DATASET_T = 'REAL_NWPU_C1'
# __D.DATABASE_T = 'REAL_NWPU_C1'

#Tag:  for wind turbines
__D.BASE_DATA_DIR = "/data/users/yang/data"
__D.DEV_DATA_DIR = "data/real_syn_wdt_vockit"

__D.DATA_DIR_S = 'synthetic_data_wdt'
__D.DATABASE = 'syn_wdt_rnd_sky_rnd_solar_rnd_cam_p3_shdw_step40'

__D.DATA_DIR_T = 'wind_turbine'
__D.DATABASE_T = 'xilin_wdt'
__D.DATABASE_TEST = 'xilin_wdt'

__D.DATA_CAT = 'WindTurbine'

def _merge_a_into_b(a, b):
  """Merge config dictionary a into config dictionary b, clobbering the
  options in b whenever they are also specified in a.
  """
  if type(a) is not edict:
    return

  for k, v in a.items():
    # a must specify keys that are in b
    if k not in b:
      raise KeyError('{} is not a valid config key'.format(k))

    # the types must match, too
    old_type = type(b[k])
    if old_type is not type(v):
      if isinstance(b[k], np.ndarray):
        v = np.array(v, dtype=b[k].dtype)
      else:
        raise ValueError(('Type mismatch ({} vs. {}) '
                          'for config key: {}').format(type(b[k]),
                                                       type(v), k))

    # recursively merge dicts
    if type(v) is edict:
      try:
        _merge_a_into_b(a[k], b[k])
      except:
        print(('Error under config key: {}'.format(k)))
        raise
    else:
      b[k] = v


def cfg_from_file(filename):
  """Load a config file and merge it into the default options."""
  import yaml
  with open(filename, 'r') as f:
    yaml_cfg = edict(yaml.load(f))

  _merge_a_into_b(yaml_cfg, __D)


def cfg_from_list(cfg_list):
  """Set config keys via list (e.g., from command line)."""
  from ast import literal_eval
  assert len(cfg_list) % 2 == 0
  for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
    key_list = k.split('.')
    d = __D
    for subkey in key_list[:-1]:
      assert subkey in d
      d = d[subkey]
    subkey = key_list[-1]
    assert subkey in d
    try:
      value = literal_eval(v)
    except:
      # handle the case when v is a string literal
      value = v
    assert type(value) == type(d[subkey]), \
      'type {} does not match original type {}'.format(
        type(value), type(d[subkey]))
    d[subkey] = value

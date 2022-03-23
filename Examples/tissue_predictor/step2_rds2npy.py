import numpy as np
import pandas as pd
import rpy2.robjects as ro
from utils import embed

readRDS = ro.r['readRDS']

data_dir = './processed/'
infer_token = readRDS(data_dir + 'infer_token.rds')
infer_token = np.asarray(infer_token, dtype=object)
infer_token = [np.asarray(token) for token in infer_token]
infer_token = np.asarray(infer_token, dtype=object)

inst_len = 50
inst_stride = 10

infer_bag = []
for token in infer_token:
  one_hot_bag = embed(token, inst_len, inst_stride)
  infer_bag.append(one_hot_bag)
infer_bag = np.asarray(infer_bag, dtype=object)
np.save(data_dir + 'infer_bag.npy', infer_bag)  

geo = 'chunkTX'
tx = 'all'

geo_name = geo + '_' + tx
infer_geo = readRDS(data_dir + 'infer_' + geo_name + '_out.rds')
infer_geo = np.asarray(infer_geo)

if geo == 'chunkTX':
  infer_geo = infer_geo.reshape((-1, 7, 35))
  infer_geo = infer_geo.transpose([0, 2, 1])
  infer_geo[:, :, -1] = np.log(infer_geo[:, :, -1] + 1)

if tx == 'all':
    geo_idx = readRDS(data_dir + 'infer_' + geo_name + '_xid.rds')
    geo_idx = np.asarray(geo_idx)
    uni_count = np.unique(geo_idx.astype(np.int32), return_counts=True)[1]
    infer_geo = np.split(infer_geo, np.cumsum(uni_count), axis=0)
    infer_geo = np.asarray(infer_geo, dtype=object)[:-1]

np.save(data_dir + 'infer_geo.npy', infer_geo)

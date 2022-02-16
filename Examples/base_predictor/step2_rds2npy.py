import numpy as np
import pandas as pd
import rpy2.robjects as ro

readRDS = ro.r['readRDS']

data_dir = './processed/'
infer_token = readRDS(data_dir + 'infer_token.rds')
infer_token = np.asarray(infer_token)
np.save(data_dir + 'infer_token.npy', infer_token)

geo = 'chunkTX'
tx = 'all' # 'long'

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
    infer_geo = np.asarray(infer_geo)[:-1]

np.save(data_dir + 'infer_' + geo_name + '.npy', infer_geo)

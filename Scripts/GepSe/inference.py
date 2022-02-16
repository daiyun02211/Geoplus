import time
import itertools
import numpy as np
import tensorflow as tf
from nets import DeepPromise, DeepRiPe, GepSe, iGepSe
from utils import create_folder
from prettytable import PrettyTable

tfk = tf.keras
tfdd = tf.data.Dataset


def infer_model(c):

    print('Loading data!')
    infer_seq = np.load(c.data_dir + 'infer_token.npy', allow_pickle=True)
    infer_seq = np.eye(4)[infer_seq - 1].astype(np.float32)

    if c.tx:
        infer_geo = np.load(c.data_dir + 'infer_' + c.geo_enc + '_' + c.tx + '.npy',
                            allow_pickle=True)
    if c.tx == 'all':
        infer_dataset = tfdd.from_generator(lambda: itertools.zip_longest(infer_seq, infer_geo),
                                            output_types=(tf.float32, tf.float32),
                                            output_shapes=(tf.TensorShape([501, 4]),
                                                           tf.TensorShape([None, c.window, 7]),
                                                           ))
        infer_dataset = infer_dataset.batch(1)
    elif c.tx == 'long':
        infer_geo = infer_geo.astype(np.float32)
        infer_dataset = tfdd.from_tensor_slices((infer_seq, infer_geo))
        infer_dataset = infer_dataset.batch(128)
    elif not c.tx:
        infer_dataset = tfdd.from_tensor_slices(infer_seq)
        infer_dataset = infer_dataset.batch(256)
    else:
        raise('Currently only all TXs (all), longest TXs (long) and no TXs (None) are available.')

    print('Creating model!')
    if isinstance(c.model_name, str):
        if not c.tx:
            dispatcher = {'DeepPromise': DeepPromise}
        elif c.tx == 'all':
            dispatcher = {'iGepSe': iGepSe}
        elif c.tx == 'long':
            dispatcher = {'GepSe': GepSe,
                          'DeepRiPe': DeepRiPe}
        try:
            model_funname = dispatcher[c.model_name]
        except KeyError:
            raise ValueError('Invalid model name')

    model = model_funname()
    model.load_weights(c.cp_path)

    pred = []
    for data in infer_dataset:
        p = model(data, training=False)
        pred.append(p)
    pred = np.concatenate(pred, axis=0).reshape(1, -1)
    return pred
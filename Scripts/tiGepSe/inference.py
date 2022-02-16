import time
import itertools
import numpy as np
import tensorflow as tf
from nets import tiGepSe
from utils import create_folder
from prettytable import PrettyTable

tfk = tf.keras
tfdd = tf.data.Dataset


def infer_model(c):

    print('Loading data!')
    infer_seq = np.load(c.data_dir + 'infer_bag.npy', allow_pickle=True)
    infer_geo = np.load(c.data_dir + 'infer_geo.npy', allow_pickle=True)

    infer_dataset = tfdd.from_generator(lambda: itertools.zip_longest(infer_seq, infer_geo),
                                        output_types=(tf.float32, tf.float32),
                                        output_shapes=(tf.TensorShape([None, c.len, 4]),
                                                       tf.TensorShape([None, 35, 7]),
                                                       ))
    infer_dataset = infer_dataset.batch(1)

    print('Creating model!')
    model = tiGepSe()
    model.load_weights(c.cp_path)

    pred = []
    for data in infer_dataset:
        p = model(data, training=False)
        pred.append(p)
    pred = np.concatenate(pred, axis=0).reshape(1, -1)
    return pred
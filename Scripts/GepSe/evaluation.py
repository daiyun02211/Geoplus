import time
import itertools
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_auc_score, average_precision_score
from nets import DeepPromise, DeepRiPe, GepSe, iGepSe
from utils import create_folder
from prettytable import PrettyTable

tfk = tf.keras
tfdd = tf.data.Dataset


def eval_model(c):

    print('Loading data!')
    valid_seq = np.load(c.data_dir + 'fold' + str(c.valid_idx) + '_token.npy', allow_pickle=True)
    valid_seq = np.eye(4)[valid_seq - 1].astype(np.float32)

    valid_out = np.load(c.data_dir + 'fold' + str(c.valid_idx) + '_label.npy', allow_pickle=True)
    valid_out = valid_out.astype(np.int32).reshape(-1, 1)

    if c.tx:
        valid_geo = np.load(c.data_dir + 'fold' + str(c.valid_idx) + '_' + c.geo_enc + '_' + t + '.npy',
                            allow_pickle=True)

    if c.tx == 'all':
        valid_dataset = tfdd.from_generator(lambda: itertools.zip_longest(valid_seq, valid_geo),
                                            output_types=(tf.float32, tf.float32),
                                            output_shapes=(tf.TensorShape([c.num_bp, 4]),
                                                           tf.TensorShape([None, c.window, 7]),
                                                           ))
        valid_dataset = valid_dataset.batch(1)
    elif c.tx == 'long':
        valid_geo = valid_geo.astype(np.float32)
        valid_dataset = tfdd.from_tensor_slices((valid_seq, valid_geo))
        valid_dataset = valid_dataset.batch(128)
    elif not c.tx:
        valid_dataset = tfdd.from_tensor_slices(valid_seq)
        valid_dataset = valid_dataset.batch(256)
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

    results = []
    pred = []
    for data in valid_dataset:
        p = model(data, training=False)
        pred.append(p)
    pred = np.concatenate(pred, axis=0)

    accuracy_scores = []
    f1_scores = []
    recall_scores = []
    precision_scores = []
    MCCs = []
    auROCs = []
    auPRCs = []

    thres = 0.5
    acc = accuracy_score(y_true=valid_out, y_pred=pred > thres)
    f1 = f1_score(y_true=valid_out, y_pred=pred > thres)
    rec = recall_score(y_true=valid_out, y_pred=pred > thres)
    pre = precision_score(y_true=valid_out, y_pred=pred > thres)
    mcc = matthews_corrcoef(y_true=valid_out, y_pred=pred > thres)
    auc = roc_auc_score(y_true=valid_out, y_score=pred)
    ap = average_precision_score(y_true=valid_out, y_score=pred)

    if c.cv:
        results.append(np.array([acc, f1, rec, pre, mcc, auc, ap]).reshape(1, -1))
    else:
        table = PrettyTable()
        column_names = ['Accuracy', 'Recall', 'Precision', 'F1', 'MCC', 'AUC', 'AP']
        table.add_column(column_names[0], np.round(accuracy_scores, 4))
        table.add_column(column_names[1], np.round(recall_scores, 4))
        table.add_column(column_names[2], np.round(precision_scores, 4))
        table.add_column(column_names[3], np.round(f1_scores, 4))
        table.add_column(column_names[4], np.round(MCCs, 4))
        table.add_column(column_names[5], np.round(auROCs, 4))
        table.add_column(column_names[6], np.round(auPRCs, 4))
        print(table)

    if c.cv:
        return results
    else:
        return None
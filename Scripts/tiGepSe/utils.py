import os
import argparse
import numpy as np
import pandas as pd
from prettytable import PrettyTable


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        print('Create directory: ', folder_path)
        os.makedirs(folder_path)


def str2bool(verbose):
    if isinstance(verbose, bool):
        return verbose
    if verbose.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif verbose.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Expect boolean value')


def merge_cv_results(results):
    results = np.concatenate(results)
    num_fold = results.shape[0]
    results = np.concatenate([results, np.mean(results, axis=0).reshape(1,-1)])
    results = np.concatenate([results, np.std(results, axis=0).reshape(1,-1)])
    results = np.round(results, 4)
    df = pd.DataFrame(results, columns=['Accuracy', 'recall', 'precision', 'f1', 'MCC', 'AUC', 'AP'])
    idx_column = list(range(1, num_fold+1))
    idx_column.append('Avg')
    idx_column.append('Std')
    df.insert(0, 'Fold', idx_column)
    table = PrettyTable()
    for col in df.columns.values:
        table.add_column(col, df[col])
    print(table)
    return df

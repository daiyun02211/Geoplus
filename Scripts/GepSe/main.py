import argparse
import numpy as np
import pandas as pd

from training import train_model
from evaluation import eval_model
from inference import infer_model
from utils import merge_cv_results


parser = argparse.ArgumentParser(description="Train and evaluate GepSe and iGepSe")
parser.add_argument('--mode', default='infer', type=str,
                    help='train, eval or infer')
parser.add_argument('--data_dir', default='./Examples/base_predictor/processed/', type=str,
                    help='Path to processed data directory')
parser.add_argument('--geo_enc', default='chunkTX', type=str,
                    help='One of [chunkTX, gridTX and onehotTX]')
parser.add_argument('--window', default=None, type=int,
                    help='The window size of the generated Geo2vec encodings')
parser.add_argument('--tx', default=None, type=str,
                    help='The transcript selection approach: None, long or all')
parser.add_argument('--cv', default=5, type=int,
                    help='The number of cross-validation folds')
parser.add_argument('--up', default=None, type=int,
                    help='The number of upsampling')
parser.add_argument('--epoch', default=20, type=int,
                    help='The number of epoch')
parser.add_argument('--lr_init', default=1e-4, type=float,
                    help='Initial learning rate')
parser.add_argument('--lr_decay', default=1e-5, type=float,
                    help='Decayed learning rate')
parser.add_argument('--cp_dir', default=None, type=str,
                    help='Path to checkpoint directory')
parser.add_argument('--save_dir', default=None, type=str,
                    help='Path to save directory')

args = parser.parse_args()


if not args.window:
    if args.geo_enc == 'chunk':
        args.window = 35
    elif args.geo_enc == 'grid':
        args.window = 80
    elif args.geo_enc == 'onehot':
        args.window = 251

if not args.tx:
    args.model_name = 'DeepPromise'
elif args.tx == 'long':
    if args.geo_enc == 'onehot':
        args.model_name = 'DeepRiPe'
    else:
        args.model_name = 'GepSe'
elif args.tx == 'all':
    args.model_name = 'iGepSe'
else:
    raise('Error: --tx have to be one of [None, long and all]')


if __name__ == '__main__':
    if args.geo_enc == 'grid':
        args.geo_enc = args.geo_enc + str(args.window)

    if args.mode == 'train':
        for i in np.arange(1, args.cv + 1):
            if args.cp_dir:
                args.cp_path = args.cp_dir + 'f' + str(i) + '.ckpt'
            else:
                args.cp_path = None
            args.valid_idx = i
            args.train_idx = list(range(1, args.cv + 1))
            args.train_idx.remove(args.valid_idx)
            train_model(args)
    elif args.mode == 'eval':
        valid_results = []
        for i in np.arange(1, args.cv + 1):
            if args.cp_dir:
                args.cp_path = args.cp_dir + 'f' + str(i) + '.ckpt'
            else:
                raise('Error: cp_dir is required for evaluation.')
            args.valid_idx = i
            results = eval_model(args)
            valid_results.append(results[0])
        valid_results = merge_cv_results(valid_results)
    elif args.mode == 'infer':
        infer_results = []
        for i in np.arange(1, args.cv + 1):
            if args.cp_dir:
                args.cp_path = args.cp_dir + 'f' + str(i) + '.ckpt'
            else:
                raise('Error: cp_dir is required for inference.')
            preds = infer_model(args)
            infer_results.append(preds)
        infer_results = np.concatenate(infer_results).mean(axis=0)
        if args.save_dir:
            infer_results = pd.DataFrame(infer_results, columns=['y_pred'])
            infer_results.to_csv(args.save_dir + 'y_pred.csv', index=False)
    else:
        raise('Error: mode should be one of [train, eval, infer]')

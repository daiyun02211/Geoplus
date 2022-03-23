import argparse
import numpy as np
import pandas as pd

from training import train_model
from evaluation import eval_model
from inference import infer_model
from utils import merge_cv_results


parser = argparse.ArgumentParser(description="Train and evaluate tiGepSe")
parser.add_argument('--mode', default='infer', type=str,
                    help='train, eval or infer')
parser.add_argument('--data_dir', default='./Examples/tissue_predictor/', type=str,
                    help='Path to processed data directory.')
parser.add_argument('--tissue', required=True, type=str,
                    help='Tissue type')
parser.add_argument('--len', default=50, type=int,
                    help='Instance length')
parser.add_argument('--cv', default=5, type=int,
                    help='The number of cross-validation folds')
parser.add_argument('--epoch', default=20, type=int,
                    help='The number of epoch')
parser.add_argument('--lr_init', default=1e-4, type=float,
                    help='Initial learning rate')
parser.add_argument('--lr_decay', default=1e-5, type=float,
                    help='Decayed learning rate')
parser.add_argument('--cp_dir', default='./Weights/tissue_predictor/', type=str,
                    help='Path to checkpoint directory')
parser.add_argument('--save_dir', default=None, type=str,
                    help='Path to save directory')

args = parser.parse_args()


if __name__ == '__main__':
    args.data_dir = args.data_dir + args.tissue + '/'
    if args.mode != 'train':
        if not args.cp_dir:
            raise("Error: cp_dir is required for evaluation or inference.")

    pred_results = []
    if args.cv:
        for i in np.arange(1, args.cv+1):
            if args.cp_dir:
                args.cp_path = args.cp_dir + args.tissue + '/f' + str(i) + '.h5'
            args.valid_idx = i
            args.train_idx = list(range(1, args.cv + 1))
            args.train_idx.remove(args.valid_idx)

            if args.mode == 'train':
                train_model(args)
            elif args.mode == 'eval':
                results = eval_model(args)
                pred_results.append(results[0])
            elif args.mode == 'infer':
                preds = infer_model(args)
                pred_results.append(preds)
            else:
                raise ('Error: mode should be one of [train, eval, infer]')

    if args.mode == 'eval':
        pred_results = merge_cv_results(pred_results)
    elif args.mode == 'infer':
        pred_results = np.concatenate(pred_results).mean(axis=0)
        if args.save_dir:
            pred_results = pd.DataFrame(pred_results, columns=['y_pred'])
            pred_results.to_csv(args.save_dir + 'y_pred.csv', index=False)



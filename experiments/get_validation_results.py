'''Parses history files from (pre-)training scripts to combine all results on 
validation set in two files (one for CLS, MLM).'''

import os
import argparse
import pandas as pd


def get_validation_results(results_dir, cls_max, mlm_max):
    '''Parses history files from (pre-)training scripts to combine all results
    on validation set in two files (one for CLS, MLM).'''

    cls_results, mlm_results = [], []
    for exp_dir in os.listdir(results_dir):

        try: 
            results = pd.read_csv(f'{results_dir}/{exp_dir}/history.csv')
        except FileNotFoundError:
            print(f"No history file found for: {exp_dir}")
        results['exp_name'] = exp_dir

        if exp_dir.startswith('CLS'):
            cls_results.append(
                results.sort_values(by=cls_max, ascending=False).head(1)
            )
        
        elif exp_dir.startswith('MLM'):
            mlm_results.append(
                results.sort_values(by=mlm_max, ascending=False).head(1)
            )

    cls_results = pd.concat(cls_results, ignore_index=True)
    mlm_results = pd.concat(mlm_results, ignore_index=True)
    cls_results.to_csv(f'{results_dir}/cls_results.csv')
    mlm_results.to_csv(f'{results_dir}/mlm_results.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results_dir', type=str)
    parser.add_argument('--cls_max', type=str, default='F1 (macro)|valid')
    parser.add_argument('--mlm_max', type=str, default='Accuracy|valid')
    args = parser.parse_args()
    get_validation_results(args.results_dir, args.cls_max, args.mlm_max)
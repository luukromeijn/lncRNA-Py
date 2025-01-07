'''Prediction scripts for lncRNA-LR and lncRNA-RF.'''

import argparse
import torch
from lncrnapy.data import Data

def evaluate(feature_table, model_type, dataset_name, data_dir, result_dir):

    # Loading the data
    data = Data(hdf_filepath=f'{data_dir}/tables/{feature_table}')

    # Loading the model & making prediction
    model = torch.load(f'{data_dir}/models/lncRNA-{model_type}.pt')
    features = model.feature_names_in_
    data.df['pred'] = model.predict(data.df[features])
    data.to_hdf(f'{result_dir}/lncrna-{model_type.lower()}_' +
                f'{dataset_name.lower()}.h5')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('feature_table', type=str)
    parser.add_argument('model_type', type=str, choices=['LR', 'RF'])
    parser.add_argument('dataset_name', type=str)
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--result_dir', type=str, default='results/scores')
    args = parser.parse_args()
    evaluate(args.feature_table, args.model_type, args.dataset_name, 
             args.data_dir, args.result_dir)
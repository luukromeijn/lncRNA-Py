'''Prediction scripts for lncRNA-LR and lncRNA-RF.'''

import argparse
import torch
from lncrnapy.data import Data
from lncrnapy.evaluate import lncRNA_classification_report

def evaluate(fasta_pcrna, fasta_ncrna, feature_table, model_type, dataset_name,
             data_dir):

    # Loading the data
    data = Data([f'{data_dir}/sequences/{fasta_pcrna}',
                f'{data_dir}/sequences/{fasta_ncrna}'],
                f'{data_dir}/tables/{feature_table}')

    # Loading the model & making prediction
    model = torch.load(f'{data_dir}/models/lncRNA-{model_type}.pt')
    features = model.feature_names_in_
    y_pred = model.predict(data.df[features])
    print(lncRNA_classification_report(
        data.df['label'], y_pred, f'lncRNA-{model_type}', dataset_name, 
        f'lncRNA-{model_type}_{dataset_name.lower()}'
    ))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fasta_pcrna', type=str)
    parser.add_argument('fasta_ncrna', type=str)
    parser.add_argument('feature_table', type=str)
    parser.add_argument('model_type', type=str, choices=['LR', 'RF'])
    parser.add_argument('dataset_name', type=str)
    parser.add_argument('data_dir', type=str)
    args = parser.parse_args()
    evaluate(args.fasta_pcrna, args.fasta_ncrna, args.feature_table, 
             args.model_type, args.dataset_name, args.data_dir)
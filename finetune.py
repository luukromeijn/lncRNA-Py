'''Experimental script for fine-tuning'''

import argparse
import torch
from rhythmnblues.data import Data
from rhythmnblues import utils
from rhythmnblues.modules import Classifier
from rhythmnblues.train import train_classifier
from rhythmnblues.train.loggers import LoggerList, EarlyStopping, LoggerPlot, LoggerWrite


# NOTE device specific!!!!
data_dir = '/data/s2592800/data'
results_dir = '/data/s2592800/results'

epochs = 100
n_samples_per_epoch = 10000
batch_size = 1

def finetune(model_file, learning_rate, dropout, weight_decay, freeze_motifs,
             freeze_network, weighted_loss):
    
    exp_name = f'CLS_lr{learning_rate}_d{dropout}_wd{weight_decay}'
    exp_name = exp_name + '--freeze_motifs' if freeze_motifs else exp_name
    exp_name = exp_name + '--freeze_network' if freeze_network else exp_name
    exp_name = exp_name + '--weighted_loss' if weighted_loss else exp_name
    print(exp_name)

    model = torch.load(f'{data_dir}/models/{model_file}')

    train_data = Data([f'{data_dir}/sequences/finetune_gencode_pcrna.fasta',
                       f'{data_dir}/sequences/finetune_gencode_ncrna.fasta'])
    valid_data = Data([f'{data_dir}/sequences/valid_gencode_pcrna.fasta',
                       f'{data_dir}/sequences/valid_gencode_ncrna.fasta'])

    train_data.set_tensor_features('4D-DNA')
    valid_data.set_tensor_features('4D-DNA')

    model = Classifier(model.base_arch, dropout, batch_size).to(utils.DEVICE)

    if freeze_motifs:
        for child in model.base_arch.motif_embedder.motif_encoder.children():
            for param in child.parameters():
                param.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, 
                                 weight_decay=weight_decay)
    model, history = train_classifier(
        model, train_data, valid_data, epochs, n_samples_per_epoch, batch_size,
        None, optimizer, logger=LoggerList(
            LoggerPlot(f'{results_dir}/CLS_{exp_name}'),
            LoggerWrite(f'{results_dir}/CLS_{exp_name}/history.csv'),
            EarlyStopping('F1 (macro)|valid', 
                          filepath=f'{data_dir}/models/{exp_name}.pt')
        )
    )



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', type=str)
    parser.add_argument('learning_rate', type=float)
    parser.add_argument('dropout', type=float)
    parser.add_argument('weight_decay', type=float)
    parser.add_argument('--freeze_motifs', default=False, action='store_true')
    parser.add_argument('--freeze_network', default=False, action='store_true')
    parser.add_argument('--weighted_loss', default=False, action='store_true')
    args = parser.parse_args()
    finetune(args.model_file, args.learning_rate, args.dropout, args.weight_decay,
             args.freeze_motifs, args.freeze_network, args.weighted_loss)
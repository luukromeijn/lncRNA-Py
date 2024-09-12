'''Hyperparameter tuning for classification.'''

import argparse
import torch
from rhythmnblues.data import Data
from rhythmnblues.modules import Classifier
from rhythmnblues.modules.cnn import MotifResNet
from rhythmnblues.modules.bert import MotifBERT
from rhythmnblues.train import train_classifier
from rhythmnblues.train.loggers import LoggerList, LoggerPlot, LoggerWrite
from rhythmnblues import utils

epochs = 100
samples_per_epoch = 10000
data_dir = '/exports/sascstudent/lromeijn/data'
results_dir = '/exports/sascstudent/lromeijn/results'
# print(utils.DEVICE)

# For duranium
data_dir = '/data/s2592800/data'
results_dir = '/data/s2592800/results'
utils.DEVICE = torch.device('cuda:2')

def train(arch, learning_rate, batch_size):


    exp_name = (f'CLS_{arch}_lr{learning_rate}_bs{batch_size}')
    print(exp_name)

    # Loading and configuring data
    train = Data([f'{data_dir}/sequences/train_pcrna.fasta',
                  f'{data_dir}/sequences/train_ncrna.fasta'])
    valid = Data([f'{data_dir}/sequences/valid_pcrna.fasta',
                  f'{data_dir}/sequences/valid_ncrna.fasta'])
    train.set_tensor_features('4D-DNA')
    valid.set_tensor_features('4D-DNA')

    if arch == 'MotifResNet':
        arch = MotifResNet(256, 12, [1,1,1,1])
    elif arch == 'MotifBERT':
        arch = MotifBERT(256, 12)
    else:
        raise ValueError('Unknown architecture type.')
    
    model = Classifier(arch, pred_batch_size=8).to(utils.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    logger = LoggerList(
        LoggerPlot(f'{results_dir}/{exp_name}'),
        LoggerWrite(f'{results_dir}/{exp_name}/history.csv')
    )
    
    model, _ = train_classifier(model, train, valid, epochs, batch_size, None,
                                optimizer, samples_per_epoch, logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('arch', type=str)
    parser.add_argument('learning_rate', type=float) 
    parser.add_argument('batch_size', type=int)
    args = parser.parse_args()
    train(args.arch, args.learning_rate, args.batch_size)
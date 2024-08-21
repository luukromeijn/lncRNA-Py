'''Hyperparameter tuning for classification.'''

import argparse
import torch
import numpy as np
from rhythmnblues.data import Data
from rhythmnblues.features import BytePairEncoding
from rhythmnblues.modules import Classifier, BERT, PyTorchBERT
from rhythmnblues.train import train_classifier
from rhythmnblues.train.loggers import LoggerList, LoggerPlot, LoggerWrite
from rhythmnblues import utils

epochs = 100
samples_per_epoch = 10000
# data_dir = '/exports/sascstudent/lromeijn/data'
# results_dir = '/exports/sascstudent/lromeijn/results'
# print(utils.DEVICE)

# For duranium
data_dir = '/data/s2592800/data'
results_dir = '/data/s2592800/data/results'
results_dir = 'results'
utils.DEVICE = torch.device('cuda:2')

def train(vocab_size, context_length):

    exp_name = (f'CLS_PyTorch_vs{vocab_size}_cl{context_length}')
    print(exp_name)

    # Loading and configuring data
    bpe = BytePairEncoding(f'{data_dir}/features/{vocab_size}.bpe', 
                           context_length)
    train = Data([f'{data_dir}/sequences/train_pcrna.fasta',
                  f'{data_dir}/sequences/train_ncrna.fasta'])
    valid = Data([f'{data_dir}/sequences/valid_pcrna.fasta', 
                  f'{data_dir}/sequences/valid_ncrna.fasta'])
    train.calculate_feature(bpe)
    valid.calculate_feature(bpe)
    train.set_tensor_features(bpe.name, torch.long)
    valid.set_tensor_features(bpe.name, torch.long)

    model = Classifier(PyTorchBERT(bpe.vocab_size), pred_batch_size=8).to(utils.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss = torch.nn.BCEWithLogitsLoss()
    logger = LoggerList(
        LoggerPlot(f'{results_dir}/{exp_name}'),
        LoggerWrite(f'{results_dir}/{exp_name}/history.csv')
    )
    
    model, _ = train_classifier(model, train, valid, epochs, 8, loss, 
                                optimizer, samples_per_epoch, logger)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('vocab_size', type=int) # 256 512 1024 2048 4096
    parser.add_argument('context_length', type=int) # 512 768 1024 1536 2048
    args = parser.parse_args()
    train(args.vocab_size, args.context_length)
    
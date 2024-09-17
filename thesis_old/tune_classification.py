'''Hyperparameter tuning for classification.'''

import argparse
import torch
from rhythmnblues.data import Data
from rhythmnblues.features import BytePairEncoding
from rhythmnblues.modules import Classifier, BERT
from rhythmnblues.train import train_classifier
from rhythmnblues.train.loggers import LoggerList, LoggerPlot, LoggerWrite
from rhythmnblues import utils

epochs = 100
samples_per_epoch = 10000
data_dir = '/exports/sascstudent/lromeijn/data'
results_dir = '/exports/sascstudent/lromeijn/results'
print(utils.DEVICE)

# For duranium
data_dir = '/data/s2592800/data'
results_dir = 'results'
utils.DEVICE = torch.device('cuda:2')

def train(learning_rate, batch_size, dropout, weighted_loss, vocab_size):

    exp_name = (f'CLS_lr{learning_rate}_bs{batch_size}_d{dropout}' + 
                f'_wl{weighted_loss}_vs{vocab_size}')
    print(exp_name)

    # Loading and configuring data
    bpe = BytePairEncoding(f'{data_dir}/features/{vocab_size}.bpe')
    train = Data(hdf_filepath=f'{data_dir}/tables/train_{vocab_size}.h5')
    valid = Data(hdf_filepath=f'{data_dir}/tables/valid_{vocab_size}.h5')
    train.set_tensor_features(bpe.name, torch.long)
    valid.set_tensor_features(bpe.name, torch.long)

    model = Classifier(BERT(bpe.vocab_size), dropout).to(utils.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if weighted_loss > 0:
        loss = torch.nn.BCEWithLogitsLoss(pos_weight=train.pos_weight())
    else:
        loss = torch.nn.BCEWithLogitsLoss()
    logger = LoggerList(
        LoggerPlot(f'{results_dir}/{exp_name}'),
        LoggerWrite(f'{results_dir}/{exp_name}/history.csv')
    )
    
    model, _ = train_classifier(model, train, valid, epochs, batch_size, loss, 
                                optimizer, samples_per_epoch, logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('learning_rate', type=float) # 0.0001 0.00001 0.000001
    parser.add_argument('batch_size', type=int) # 8 16 32 64 128 NOTE: make it failsafe
    parser.add_argument('dropout', type=float) # 0 0.1 0.5
    parser.add_argument('weighted_loss', type=int) # 0 1 # TODO: later
    parser.add_argument('vocab_size', type=int) # 4096
    args = parser.parse_args()
    train(args.learning_rate, args.batch_size, args.dropout, 
          args.weighted_loss, args.vocab_size)
    
'''
python -u -m thesis.tune_classification 0.0001 8 0 0 4096
python -u -m thesis.tune_classification 0.0001 8 0.1 0 4096
python -u -m thesis.tune_classification 0.0001 8 0.5 0 4096
python -u -m thesis.tune_classification 0.0001 16 0 0 4096
python -u -m thesis.tune_classification 0.0001 16 0.1 0 4096
python -u -m thesis.tune_classification 0.0001 16 0.5 0 4096
python -u -m thesis.tune_classification 0.0001 32 0 0 4096
python -u -m thesis.tune_classification 0.0001 32 0.1 0 4096
python -u -m thesis.tune_classification 0.0001 32 0.5 0 4096
python -u -m thesis.tune_classification 0.0001 64 0 0 4096
python -u -m thesis.tune_classification 0.0001 64 0.1 0 4096
python -u -m thesis.tune_classification 0.0001 64 0.5 0 4096
python -u -m thesis.tune_classification 0.00001 8 0 0 4096
python -u -m thesis.tune_classification 0.00001 8 0.1 0 4096
python -u -m thesis.tune_classification 0.00001 8 0.5 0 4096
python -u -m thesis.tune_classification 0.00001 16 0 0 4096
python -u -m thesis.tune_classification 0.00001 16 0.1 0 4096
python -u -m thesis.tune_classification 0.00001 16 0.5 0 4096
python -u -m thesis.tune_classification 0.00001 32 0 0 4096
python -u -m thesis.tune_classification 0.00001 32 0.1 0 4096
python -u -m thesis.tune_classification 0.00001 32 0.5 0 4096
python -u -m thesis.tune_classification 0.00001 64 0 0 4096
python -u -m thesis.tune_classification 0.00001 64 0.1 0 4096
python -u -m thesis.tune_classification 0.00001 64 0.5 0 4096
python -u -m thesis.tune_classification 0.000001 8 0 0 4096
python -u -m thesis.tune_classification 0.000001 8 0.1 0 4096
python -u -m thesis.tune_classification 0.000001 8 0.5 0 4096
python -u -m thesis.tune_classification 0.000001 16 0 0 4096
python -u -m thesis.tune_classification 0.000001 16 0.1 0 4096
python -u -m thesis.tune_classification 0.000001 16 0.5 0 4096
python -u -m thesis.tune_classification 0.000001 32 0 0 4096
python -u -m thesis.tune_classification 0.000001 32 0.1 0 4096
python -u -m thesis.tune_classification 0.000001 32 0.5 0 4096
python -u -m thesis.tune_classification 0.000001 64 0 0 4096
python -u -m thesis.tune_classification 0.000001 64 0.1 0 4096
python -u -m thesis.tune_classification 0.000001 64 0.5 0 4096
'''
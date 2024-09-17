''''Pre-trains and fine-tunes RNA models using different encoding methods.'''

import argparse
import torch
from rhythmnblues.data import Data
from rhythmnblues.features import BytePairEncoding, KmerTokenizer
from rhythmnblues.modules import (BERT, MotifBERT, MaskedMotifModel, 
                                  MaskedTokenModel, Classifier)
from rhythmnblues.train import (train_masked_token_modeling, 
                                train_masked_motif_modeling, train_classifier)
from rhythmnblues.train.loggers import (LoggerList, LoggerPlot, LoggerWrite, 
                                        EarlyStopping)
from rhythmnblues import utils

# Device specific
data_dir = '/data/s2592800/data'
results_dir = '/data/s2592800/results'

# Constants
pretrain_epochs = 1
finetune_epochs = 1
samples_per_epoch = 10000
context_length = 768
d_model = 1024
d_ff = 4096
N = 8
batch_size = 8
finetune_lr = 0.00001
mlm_metrics = ['Loss', 'Accuracy']


def experiment(encoding_method, arg1, arg2, weighted_loss):

    if encoding_method not in ['nuc', 'kmer', 'bpe', 'motif']:
        raise ValueError("Invalid encoding method, should be one of: [nuc, " +
                         "kmer, bpe, motif].")
                         
    exp_name = encoding_method
    exp_name = exp_name + f'-{arg1}' if arg1 is not None else exp_name
    exp_name = exp_name + f'-{arg2}' if arg2 is not None else exp_name
    exp_name = exp_name + '-weighted' if weighted_loss else exp_name
    print(exp_name)

    # Loading the data
    data_pretrain = Data([f'{data_dir}/sequences/pretrain_human_pcrna.fasta',
                          f'{data_dir}/sequences/pretrain_human_ncrna.fasta'])
    data_finetune = Data([f'{data_dir}/sequences/finetune_gencode_pcrna.fasta',
                          f'{data_dir}/sequences/finetune_gencode_pcrna.fasta'])
    data_validate = Data([f'{data_dir}/sequences/valid_gencode_pcrna.fasta',
                          f'{data_dir}/sequences/valid_gencode_pcrna.fasta'])
    
    # Encoding the data
    if encoding_method in ['nuc', 'kmer', 'bpe']:
        if encoding_method == 'nuc':
            tokenizer = KmerTokenizer(1, context_length)
        elif encoding_method == 'kmer':
            tokenizer = KmerTokenizer(arg1, context_length)
        else:
            tokenizer = BytePairEncoding(
                f'{data_dir}/features/{arg1}.bpe', context_length)
        for dataset in [data_pretrain, data_finetune, data_validate]:
            dataset.calculate_feature(tokenizer)
            dataset.set_tensor_features(tokenizer.name, torch.long)
    elif encoding_method == 'motif':
        for dataset in [data_pretrain, data_finetune, data_validate]:
            dataset.set_tensor_features('4D-DNA')
        # Force the same context length for motif encoding
        utils.LEN_4D_DNA = (context_length-1)*arg2
        
    if encoding_method in ['nuc', 'kmer', 'bpe']:
        base_arch = BERT(tokenizer.vocab_size, d_model, d_ff, N)
        model_type = MaskedTokenModel
        pretrain_function = train_masked_token_modeling
    elif encoding_method == 'motif':
        base_arch = MotifBERT(arg1, arg2, d_model=d_model, 
                              d_ff=d_ff, N=N)
        model_type = MaskedMotifModel
        pretrain_function = train_masked_motif_modeling
    model = model_type(base_arch, pred_batch_size=batch_size).to(utils.DEVICE)
    
    # Pre-training the model
    model, history = pretrain_function(
        model, data_pretrain, data_validate.sample(N=2500, random_state=42), 
        pretrain_epochs, batch_size, warmup_steps=32000, 
        n_samples_per_epoch=samples_per_epoch, logger=LoggerList(
            LoggerPlot(f'{results_dir}/MLM_{exp_name}', mlm_metrics),
            LoggerWrite(f'{results_dir}/MLM_{exp_name}/history.csv', 
                        mlm_metrics),
            EarlyStopping('Accuracy|valid', 
                          filepath=f'{data_dir}/models/MLM_{exp_name}.pt')
        )
    )

    # Loading the (early-stopped) model and preparing for classification
    model = torch.load(f'{data_dir}/models/MLM_{exp_name}.pt')
    model = Classifier(model.base_arch, pred_batch_size=batch_size)
    model = model.to(utils.DEVICE)

    # Setting loss and optimizer according to hyperparameters
    if weighted_loss:
        loss = torch.nn.BCEWithLogitsLoss(pos_weight=data_finetune.pos_weight())
    else:
        loss = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=finetune_lr)

    # Fine-tuning
    model, history = train_classifier(
        model, data_finetune, data_validate, finetune_epochs, batch_size, loss,
        optimizer, samples_per_epoch, logger=LoggerList(
            LoggerPlot(f'{results_dir}/CLS_{exp_name}', mlm_metrics),
            LoggerWrite(f'{results_dir}/CLS_{exp_name}/history.csv', 
                        mlm_metrics),
            EarlyStopping('F1 (macro)|valid', 
                          filepath=f'{data_dir}/models/CLS_{exp_name}.pt')
        ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('encoding_method', type=str)
    parser.add_argument('arg1', nargs='?', default=None, type=int)
    parser.add_argument('arg2', nargs='?', default=None, type=int)
    parser.add_argument('--weighted_loss', action='store_true')
    args = parser.parse_args()
    experiment(args.encoding_method, args.arg1, args.arg2, args.weighted_loss)
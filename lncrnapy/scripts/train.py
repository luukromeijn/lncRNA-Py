'''Trains (or fine-tunes) a model (optionally pre-trained) for lncRNA 
classification.

Please call `python -m lncrnapy.scripts.train --help` for usage info.'''

import argparse
import torch
from lncrnapy import utils
from lncrnapy.data import Data
from lncrnapy.features import KmerTokenizer, BytePairEncoding
from lncrnapy.train.loggers import EarlyStopping
from lncrnapy.train.loggers import LoggerList, LoggerPlot, LoggerWrite
from lncrnapy.modules import BERT, CSEBERT, MaskedConvModel, MaskedTokenModel
from lncrnapy.modules import Classifier
from lncrnapy.train import train_classifier


def train(
        fasta_pcrna_train, fasta_ncrna_train, fasta_pcrna_valid, 
        fasta_ncrna_valid, exp_prefix, pretrained_model, encoding_method, 
        epochs, n_samples_per_epoch, batch_size, learning_rate, weight_decay, 
        d_model, N, d_ff, h, dropout, hidden_cls_layers, n_kernels, kernel_size, 
        bpe_file, k, context_length, data_dir, results_dir, model_dir, 
        weighted_loss, random_reading_frame, freeze_network, freeze_kernels, 
        input_linear, input_relu,
    ):
    '''lncRNA classification function as called by training script. Run
    `lncrnapy.scripts.train --help` for usage info.'''

    exp_name = f'{exp_prefix}_{encoding_method}'

    # Import data
    train_data = Data([f'{data_dir}/{fasta_pcrna_train}',
                       f'{data_dir}/{fasta_ncrna_train}']) 
    valid_data = Data([f'{data_dir}/{fasta_pcrna_valid}',
                       f'{data_dir}/{fasta_ncrna_valid}'])
    
    # If specified, load the pre-trained model and update hyperparameters
    if len(pretrained_model) > 0:
        exp_name += '_finetuned'
        if encoding_method in ['nuc', 'kmer', 'bpe']:
            base_arch = MaskedTokenModel.from_pretrained(pretrained_model)
        else:
            base_arch = MaskedConvModel.from_pretrained(pretrained_model)
        base_arch = base_arch.base_arch.to(utils.DEVICE)
        d_model = base_arch.d_model
        N = base_arch.N
        d_ff = base_arch.d_ff
        h = base_arch.h
        if type(base_arch) == CSEBERT:
            n_kernels = base_arch.n_kernels
            kernel_size = base_arch.kernel_size
            input_linear = base_arch.input_linear
            input_relu = base_arch.input_relu
    
    # Encoding the data
    if encoding_method in ['nuc', 'kmer', 'bpe']:
        if encoding_method == 'nuc':
            tokenizer = KmerTokenizer(1, context_length)
        elif encoding_method == 'kmer':
            tokenizer = KmerTokenizer(k, context_length)
            exp_name += f'_k{k}'
        else:
            tokenizer = BytePairEncoding(f'{data_dir}/{bpe_file}', 
                                         context_length)
            exp_name += f'_vs{tokenizer.vocab_size}'
        for dataset in [train_data, valid_data]:
            dataset.calculate_feature(tokenizer)
            dataset.set_tensor_features(tokenizer.name, torch.long)
    elif encoding_method == 'cse':
        len_4d_dna = (context_length-1)*kernel_size
        for dataset in [train_data, valid_data]:
            dataset.set_tensor_features('4D-DNA', len_4d_dna=len_4d_dna)
        exp_name += f'_nm{n_kernels}_sm{kernel_size}'

    # Initializing the base arch. (if no pre-trained model is provided) & model
    if len(pretrained_model) == 0:
        if encoding_method in ['nuc', 'kmer', 'bpe']:
            base_arch = BERT(tokenizer.vocab_size, d_model, N, d_ff, h)
        elif encoding_method == 'cse':
            base_arch = CSEBERT(n_kernels, kernel_size, d_model, N, d_ff, h,
                 input_linear=input_linear, input_relu=input_relu)
    pooling = 'CLS'
    if freeze_kernels:
        base_arch.freeze_kernels()
    if freeze_network:
        base_arch.freeze()
        pooling = 'mean'
    model = Classifier(base_arch.config, base_arch, dropout, pooling, 
                       hidden_cls_layers, batch_size)

    # Model/experiment name processing
    exp_name += f'_dm{d_model}_N{N}'
    exp_name = f'{exp_name}_dff{d_ff}' if d_ff is not None else exp_name
    exp_name = f'{exp_name}_h{h}' if h is not None else exp_name
    exp_name += f'_bs{batch_size}_lr{learning_rate}_wd{weight_decay}' 
    exp_name += f'_cl{context_length}_d{dropout}'
    exp_name =f'{exp_name}--no_weighted_loss' if not weighted_loss else exp_name
    exp_name = f'{exp_name}--no_rrf' if not random_reading_frame else exp_name
    exp_name = f'{exp_name}--freeze_network' if freeze_network else exp_name
    exp_name = f'{exp_name}--freeze_kernels' if freeze_kernels else exp_name
    exp_name = f'{exp_name}--in_lin' if input_linear else exp_name
    exp_name = f'{exp_name}--no_in_relu' if not input_relu else exp_name

    # Pre-training the model
    model = model.to(utils.DEVICE) # Send model to GPU
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, 
                                 weight_decay=weight_decay)
    model, history = train_classifier(
        model, train_data, valid_data, epochs, n_samples_per_epoch, batch_size,
        optimizer, weighted_loss, random_reading_frame, logger=LoggerList(
            LoggerPlot(f'{results_dir}/{exp_name}'),
            LoggerWrite(f'{results_dir}/{exp_name}/history.csv'),
            EarlyStopping('F1 (macro)|valid', 
                          filepath=f'{model_dir}/{exp_name}')
        )
    )


description = 'Trains (or fine-tunes) a model (optionally pre-trained) for ' \
              'lncRNA classification.'

args = {
    'fasta_pcrna_train': {
        'type': str,
        'help': 'Path to FASTA file with pcRNA training sequences. (str)'
    }, 
    'fasta_ncrna_train': {
        'type': str,
        'help': 'Path to FASTA file with ncRNA training sequences. (str)'
    }, 
    'fasta_pcrna_valid': {
        'type': str,
        'help': 'Path to FASTA file with pcRNA sequences used for validating '
                'the model after every epoch. (str)'
    },  
    'fasta_ncrna_valid': {
        'type': str,
        'help': 'Path to FASTA file with ncRNA sequences used for validating '
                'the model after every epoch. (str)'
    }, 
    '--exp_prefix': {
        'type': str, 
        'default': 'CLS',
        'help': 'Added prefix to model/experiment name. (str)',
    },
    '--pretrained_model': {
        'type': str, 
        'default': "",
        'help': 'If specified, fine-tunes this pre-trained model instead of '
                'training one from scratch. Note that this causes model-related'
                ' hyperparameters, such as d_model and N, to be ignored. '
                'Specified by id of a model hosted on the HuggingFace Hub, or a'
                ' path to a local directory containing model weights. (str)=""',
    },
    '--encoding_method': {
        'type': str,
        'choices': ['cse', 'bpe', 'kmer', 'nuc'],
        'default': 'cse',
        'help': 'Sequence encoding method. (str="cse")'
    },
    '--epochs': {
        'type': int,
        'default': 100,
        'help': 'Number of epochs to train for. (int=100)'
    },
    '--n_samples_per_epoch': {
        'type': int,
        'default': 10000,
        'help': 'Number of training samples per epoch. (int=10000)'
    },
    '--batch_size': {
        'type': int,
        'default': 8,
        'help': 'Number of samples per optimization step. (int=8)'
    },
    '--learning_rate': {
        'type': float,
        'default': 1e-5,
        'help': 'Learning rate used by Adam optimizer. (float=1e-5)'
    },
    '--weight_decay': {
        'type': float,
        'default': 0,
        'help': 'Weight decay used by Adam optimizer. (float=0.0)'
    },
    '--d_model': {
        'type': int,
        'default': 768,
        'help': 'BERT embedding dimensionality. (int=768)'
    },
    '--N': {
        'type': int,
        'default': 12,
        'help': 'Number of BERT transformer blocks. (int=12)'
    },
    '--d_ff': {
        'type': int,
        'default': None,
        'help': 'Number of nodes in BERT FFN sublayers (int=4*d_model)'
    },
    '--h': {
        'type': int,
        'default': None,
        'help': 'Number of BERT self-attention heads (int=int(d_model/64))'
    },
    '--dropout': {
        'type': float,
        'default': 0,
        'help': 'Dropout probability in CLS output head. (float=0)'
    },
    '--hidden_cls_layers': {
        'type': int,
        'nargs': '+',
        'default': [],
        'help': 'Space-separated list with number of hidden nodes in ReLU-'
                'activated classification head layers. (int=[])'
    },
    '--n_kernels': {
        'type': int,
        'default': 768,
        'help': 'Specifies number of kernels when convolutional sequence '
                'encoding is used. (int=768)'
    },
    '--kernel_size': {
        'type': int,
        'default': 9,
        'help': 'Specifies kernel size when convolutional sequence encoding is '
            	'used. (int=9)'
    },
    '--bpe_file': {
        'type': str,
        'default': "",
        'help': 'Filepath to BPE model generated with BPE script. Required when'
                ' Byte Pair Encoding is used. (str="")'
    },
    '--k': {
        'type': int,
        'default': 6,
        'help': 'Specifies k when k-mer encoding is used. (int=6)'
    },
    '--context_length': {
        'type': int,
        'default': 768,
        'help': 'Number of input positions. For cse/k-mer encoding, this '
                'translates to a maximum of (768-1)*k input nucleotides. '
                '(int=768)'
    },
    '--data_dir': {
        'type': str,
        'default': '.',
        'help': 'Parent directory to use for any of the paths specified in '
                'these arguments. (str="")'
    }, 
    '--results_dir': {
        'type': str,
        'default': '.',
        'help': 'Parent directory to use for the results folder of this script.'
                ' (str="")'
    }, 
    '--model_dir': {
        'type': str,
        'default': '.',
        'help': 'Directory where to save the trained model to. Model with '
                'highest macro F1-score on the validation dataset is saved. '
                ' (str=f"{data_dir}/models")'
    }, 
    '--no_weighted_loss': {
        'action': 'store_true',
        'default': False,
        'help': 'Applies correction to pcRNA/ncRNA class imbalance. (bool)'
    }, 
    '--no_random_reading_frame': {
        'action': 'store_true',
        'default': False,
        'help': 'Turns off sampling in random reading frame for convolutional '
                'sequence encoding. (bool)'
    }, 
    '--freeze_network': {
        'action': 'store_true',
        'default': False,
        'help': 'Freezes all weights from the pre-trained model and bases the '
                'clasification on the mean embeddings of this model. This only '
                'works with the --pretrained_model flag. (bool)'
    }, 
    '--freeze_kernels': {
        'action': 'store_true',
        'default': False,
        'help': 'Freezes all convolutional sequence encoding weights from the '
                'pre-trained model. Only works with the --pretrained_model '
                'flag. (bool)'
    }, 
    '--input_linear': {
        'action': 'store_true',
        'default': None,
        'help': 'Forces linear projection of kernels onto d_model dimensions in '
                'convolutional sequence encoding. (bool)'
    }, 
    '--no_input_relu': {
        'action': 'store_true',
        'default': False,
        'help': 'Turns off ReLU activation of kernels in convolutional sequence'
                ' encoding. (bool)'
    }, 
}


if __name__ == '__main__':

    # Parsing arguments
    p = argparse.ArgumentParser(description=description)
    for arg in args:
        p.add_argument(arg, **args[arg])
    p = p.parse_args()
    
    # Argument checks and preprocessing
    if p.encoding_method == 'bpe' and len(p.bpe_file) == 0:
        raise ValueError(
            "Please use --bpe_file flag to specify BPE model file."
        )
    p.model_dir = f'{p.data_dir}/models' if p.model_dir=='.' else p.model_dir
    
    train( # Call
        fasta_pcrna_train=p.fasta_pcrna_train, 
        fasta_ncrna_train=p.fasta_ncrna_train, 
        fasta_pcrna_valid=p.fasta_pcrna_valid, 
        fasta_ncrna_valid=p.fasta_ncrna_valid, 
        exp_prefix=p.exp_prefix, pretrained_model=p.pretrained_model, 
        encoding_method=p.encoding_method, 
        epochs=p.epochs, n_samples_per_epoch=p.n_samples_per_epoch, 
        batch_size=p.batch_size, learning_rate=p.learning_rate, 
        weight_decay=p.weight_decay, 
        d_model=p.d_model, N=p.N, d_ff=p.d_ff, h=p.h, dropout=p.dropout, 
        hidden_cls_layers=p.hidden_cls_layers, n_kernels=p.n_kernels, 
        kernel_size=p.kernel_size, bpe_file=p.bpe_file, k=p.k, 
        context_length=p.context_length, data_dir=p.data_dir, 
        results_dir=p.results_dir, model_dir=p.model_dir, 
        weighted_loss=(not p.no_weighted_loss),
        random_reading_frame=(not p.no_random_reading_frame),
        freeze_network=p.freeze_network, freeze_kernels=p.freeze_kernels, 
        input_linear=p.input_linear, 
        input_relu=(not p.no_input_relu),
    )


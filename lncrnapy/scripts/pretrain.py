'''Pre-training script for a Nucleotide Language Model. Several encoding methods
and hyperparameter settings are supported.

Please call `python -m lncrnapy.scripts.pretrain --help` for usage info.'''

import argparse
import torch
from functools import partial
from lncrnapy import utils
from lncrnapy.data import Data
from lncrnapy.features import KmerTokenizer, BytePairEncoding
from lncrnapy.train.loggers import EarlyStopping
from lncrnapy.train.loggers import LoggerList, LoggerPlot, LoggerWrite
from lncrnapy.modules import BERT, CSEBERT
from lncrnapy.modules import MaskedTokenModel, MaskedConvModel
from lncrnapy.train import train_masked_token_modeling
from lncrnapy.train import train_masked_conv_modeling


def pretrain(
        fasta_train, fasta_valid, exp_prefix, encoding_method, epochs, 
        n_samples_per_epoch, batch_size, warmup_steps, d_model, N, d_ff, h, 
        dropout, n_kernels, kernel_size, bpe_file, k, p_mlm, p_mask, p_random, 
        context_length, data_dir, results_dir, model_dir, mask_size, 
        random_reading_frame, input_linear, input_relu, 
        output_linear, output_relu, 
    ):
    '''Pre-training function as used in pre-training script. Run 
    `lncrnapy.scripts.pretrain --help` for usage info.'''

    exp_name = f'{exp_prefix}_{encoding_method}'

    # Import data, subsample valid dataset to save time and resources
    train_data = Data(f'{data_dir}/{fasta_train}') 
    valid_data = Data(f'{data_dir}/{fasta_valid}').sample(N=2500,
                                                          random_state=42)

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

    # Initializing the model
    if encoding_method in ['nuc', 'kmer', 'bpe']:
        base_arch = BERT(tokenizer.vocab_size, d_model, N, d_ff, h)
        model = MaskedTokenModel(base_arch.config, base_arch, dropout, 
                                 batch_size)
        pretrain_function = train_masked_token_modeling
    elif encoding_method == 'cse':
        base_arch = CSEBERT(
            n_kernels, kernel_size, d_model, N, d_ff, h, 
            input_linear=input_linear, input_relu=input_relu
        )
        model = MaskedConvModel(base_arch.config, base_arch, dropout, 
                                pred_batch_size=batch_size,
                                output_linear=output_linear, 
                                output_relu=output_relu)
        pretrain_function = partial(train_masked_conv_modeling, 
                                    mask_size=mask_size, 
                                    random_reading_frame=random_reading_frame) 

    # Model/experiment name processing
    exp_name += f'_dm{d_model}_N{N}'
    exp_name = f'{exp_name}_dff{d_ff}' if d_ff is not None else exp_name
    exp_name = f'{exp_name}_h{h}' if h is not None else exp_name
    exp_name += f'_bs{batch_size}_ws{warmup_steps}'
    exp_name += f'_cl{context_length}_d{dropout}'
    exp_name = f'{exp_name}_ms{mask_size}' if mask_size != 1 else exp_name
    exp_name = f'{exp_name}--no_rrf' if not random_reading_frame else exp_name
    exp_name = f'{exp_name}--in_lin' if input_linear else exp_name
    exp_name = f'{exp_name}--no_in_relu' if not input_relu else exp_name
    exp_name = f'{exp_name}--no_out_lin' if not output_linear else exp_name
    exp_name = f'{exp_name}--out_relu' if output_relu else exp_name

    # Pre-training the model
    model = model.to(utils.DEVICE) # Send model to GPU
    model, history = pretrain_function(
        model, train_data, valid_data, epochs, n_samples_per_epoch, batch_size,
        p_mlm, p_mask, p_random, warmup_steps, logger=LoggerList(
            LoggerPlot(f'{results_dir}/{exp_name}', ['Loss', 'Accuracy']),
            LoggerWrite(f'{results_dir}/{exp_name}/history.csv', 
                        ['Loss', 'Accuracy']),
            EarlyStopping('Accuracy|valid', 
                          filepath=f'{model_dir}/{exp_name}')
        )
    )


description = 'Pre-training script for a Nucleotide Language Model. Several ' \
              'encoding methods and hyperparameter settings are supported.'

# Arguments
args = {
    'fasta_train': {
        'type': str, 
        'help': 'Path to FASTA file with pre-training sequences. (str)',
    },
    'fasta_valid': {
        'type': str, 
        'help': 'Path to FASTA file with sequences to use for validating model '
                'performance after every epoch. (str)',
    },
    '--exp_prefix': {
        'type': str, 
        'default': 'MLM',
        'help': 'Added prefix to model/experiment name. (str="MLM")',
    },
    '--encoding_method': {
        'type': str,
        'choices': ['cse', 'bpe', 'kmer', 'nuc'],
        'default': 'cse',
        'help': 'Sequence encoding method. (str="cse")'
    },
    '--epochs': {
        'type': int,
        'default': 500,
        'help': 'Number of epochs to pre-train for. (int=500)'
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
    '--warmup_steps': {
        'type': int,
        'default': 32000,
        'help': 'Number of optimization steps in which learning rate increases '
                'linearly. After this amount of steps, the learning rate '
                'decreases proportional to the inverse square root of the '
                'step number. (int=8)'
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
        'help': 'Dropout probability in MLM output head. (float=0)'
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
    '--p_mlm': {
        'type': float,
        'default': 0.15,
        'help': 'Selection probability per token/nucleotide in MLM. '
                '(float=0.15)'
    },
    '--p_mask': {
        'type': float,
        'default': 0.8,
        'help': 'Mask probability for selected token/nucleotide. (float=0.8)'
    },
    '--p_random': {
        'type': float,
        'default': 0.1,
        'help': 'Random replacement chance per token/nucleotide. (float=0.1)'
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
        'help': 'Directory where to save pre-trained model to. Model with the '
                'highest accuracy on the validation dataset is saved. '
                '(str=f"{data_dir}/models")'
    }, 
    '--mask_size': {
        'type': int,
        'default': 1,
        'help': 'Number of contiguous nucleotides that make up a mask. (int=1)'
    }, 
    '--no_random_reading_frame': {
        'action': 'store_true',
        'default': False,
        'help': 'Turns off sampling in random reading frame for convolutional '
                'sequence encoding (bool)'
    }, 
    '--input_linear': {
        'action': 'store_true',
        'default': None,
        'help': 'Forces linear projection of kernels onto d_model dimensions in'
                ' convolutional sequence encoding. (bool)'
    }, 
    '--no_input_relu': {
        'action': 'store_true',
        'default': False,
        'help': 'Turns off ReLU activation of kernels in convolutional sequence'
                ' encoding. (bool)'
    }, 
    '--no_output_linear': {
        'action': 'store_true',
        'default': False,
        'help': 'Forces linear projection of embeddings onto n_kernels ' 
                'dimensions before masked convolution output layer. (bool)'
    }, 
    '--output_relu': {
        'action': 'store_true',
        'default': False,
        'help': 'Forces ReLU activation of embeddings before masked convolution' 
                ' output layer. (bool)'
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
    
    pretrain( # Call
        fasta_train=p.fasta_train, fasta_valid=p.fasta_valid, 
        exp_prefix=p.exp_prefix, encoding_method=p.encoding_method, 
        epochs=p.epochs, n_samples_per_epoch=p.n_samples_per_epoch, 
        batch_size=p.batch_size, warmup_steps=p.warmup_steps, d_model=p.d_model,
        N=p.N, d_ff=p.d_ff, h=p.h, dropout=p.dropout, n_kernels=p.n_kernels, 
        kernel_size=p.kernel_size, bpe_file=p.bpe_file, k=p.k, p_mlm=p.p_mlm, 
        p_mask=p.p_mask, p_random=p.p_random, context_length=p.context_length, 
        data_dir=p.data_dir, results_dir=p.results_dir, model_dir=p.model_dir,
        mask_size=p.mask_size, 
        random_reading_frame=(not p.no_random_reading_frame),
        input_linear=p.input_linear, 
        input_relu=(not p.no_input_relu), 
        output_linear=(not p.no_output_linear), 
        output_relu=p.output_relu, 
    )
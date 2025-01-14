'''Retrieves MLM accuracy per sequence by specified model for input dataset.'''

import argparse
import torch
from lncrnapy import utils
from lncrnapy.data import Data
from lncrnapy.features import KmerTokenizer, BytePairEncoding, MLMAccuracy
from lncrnapy.modules import CSEBERT, MaskedConvModel, MaskedTokenModel


def mlm_accuracy(
        fasta_file, model_file, output_file, encoding_method, bpe_file, k, 
        batch_size, context_length, data_dir, results_dir 
    ):
    '''MLM accuracy (per seq.) function as called by this script.'''

    # Import data
    fasta_file = [f'{data_dir}/{filepath}' for filepath in fasta_file]
    fasta_file = fasta_file[0] if len(fasta_file) == 1 else fasta_file
    data = Data(fasta_file) 
    
    # Loading the model
    if encoding_method in ['nuc', 'kmer', 'bpe']:
        model = MaskedTokenModel.from_pretrained(model_file)
    else:
        model = MaskedConvModel.from_pretrained(model_file)
    model.pred_batch_size = batch_size
    model = model.to(utils.DEVICE)
    if type(model.base_arch) == CSEBERT:
        kernel_size = model.base_arch.kernel_size
    print("Model loaded.")

    # Encoding the data
    if encoding_method in ['nuc', 'kmer', 'bpe']:
        if encoding_method == 'nuc':
            tokenizer = KmerTokenizer(1, context_length)
        elif encoding_method == 'kmer':
            tokenizer = KmerTokenizer(k, context_length)
        else:
            tokenizer = BytePairEncoding(f'{data_dir}/{bpe_file}',
                                         context_length)
        data.calculate_feature(tokenizer)
        data.set_tensor_features(tokenizer.name, torch.long)
    elif encoding_method == 'cse':
        len_4d_dna = (context_length-1)*kernel_size
        data.set_tensor_features('4D-DNA', len_4d_dna=len_4d_dna)

    # Retrieving and saving the MLM accuracy
    data.calculate_feature(MLMAccuracy(model))
    data.to_hdf(f'{results_dir}/{output_file}')


args = {
    'fasta_file': {
        'type': str,
        'nargs': '+',
        'help': 'Path to FASTA file of RNA sequences or pair of paths to two '
                'FASTA files containing protein- and non-coding RNAs, '
                'respectively. (str)'
    }, 
    '--model_file': {
        'type': str, 
        'default': 'luukromeijn/lncRNA-BERT-CSE-k9-pretrained',
        'help': '(Pre-)trained model, specified by id of a model hosted on'
                ' the HuggingFace Hub, or a path to a local directory '
                'containing model weights. '
                '(str="luukromeijn/lncRNA-BERT-CSE-k9-pretrained")'
        
    },
    '--output_file': {
        'type': str, 
        'default': 'mlm_accuracy.h5',
        'help': 'Name of hdf output file. (str)',
    },
    '--encoding_method': {
        'type': str,
        'choices': ['cse', 'bpe', 'kmer', 'nuc'],
        'default': 'cse',
        'help': 'Sequence encoding method. (str="cse")'
    },
    '--bpe_file': {
        'type': str,
        'default': "",
        'help': 'Filepath to BPE model generated with BPE script. Required when'
                ' Byte Pair Encoding is used. (str="")'
    },
    '--k': {
        'type': int,
        'default': 9,
        'help': 'Specifies k when k-mer encoding is used. (int=9)'
    },
    '--batch_size': {
        'type': int,
        'default': 8,
        'help': 'Number of samples per prediction step. (int=8)'
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
}


if __name__ == '__main__':

    # Parsing arguments
    p = argparse.ArgumentParser()
    for arg in args:
        p.add_argument(arg, **args[arg])
    p = p.parse_args()
    
    # Argument checks and preprocessing
    if p.encoding_method == 'bpe' and len(p.bpe_file) == 0:
        raise ValueError(
            "Please use --bpe_file flag to specify BPE model file."
        )
    
    mlm_accuracy( # Call
        p.fasta_file, p.model_file, p.output_file, p.encoding_method, 
        p.bpe_file, p.k, p.batch_size, p.context_length, p.data_dir, 
        p.results_dir,
    )
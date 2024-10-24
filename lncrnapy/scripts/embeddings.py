'''Retrieves sequence embeddings by specified model for input dataset.

Please call `python -m lncrnapy.scripts.embeddings --help` for usage info.
'''

import argparse
import torch
from lncrnapy import utils
from lncrnapy.data import Data
from lncrnapy.features import KmerTokenizer, BytePairEncoding
from lncrnapy.modules import CSEBERT
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap


dim_red_functions = {
    'tsne': TSNE(),
    'pca': PCA(),
    'umap': umap.UMAP(),
}


def embeddings(
        fasta_file, model_file, output_file, output_plot_file, encoding_method, 
        bpe_file, k, pooling, dim_red, batch_size, context_length, data_dir, 
        results_dir, model_dir, 
    ):
    '''RNA sequence embedding function as called by embeddings script. Run
    `lncrnapy.scripts.embeddings --help` for usage info.'''

    # Import data
    fasta_file = [f'{data_dir}/{filepath}' for filepath in fasta_file]
    fasta_file = fasta_file[0] if len(fasta_file) == 1 else fasta_file
    data = Data(fasta_file) 
    
    # Loading the model
    model = torch.load(f'{model_dir}/{model_file}', utils.DEVICE)
    model.pred_batch_size = batch_size
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
    elif encoding_method == 'conv':
        len_4d_dna = (context_length-1)*kernel_size
        data.set_tensor_features('4D-DNA', len_4d_dna=len_4d_dna)

    # Retrieving and saving the embeddings (+ dimensionality reduction)
    dim_red = dim_red_functions[dim_red] if dim_red != 'None' else None
    model.latent_space(data, inplace=True, pooling=pooling, dim_red=dim_red)
    data.to_hdf(f'{results_dir}/{output_file}')                                 # NOTE: I guess .csv option would be nice here
    if output_plot_file is not None:
        data.plot_feature_scatter('L0', 'L1', 
                                  filepath=f'{results_dir}/{output_plot_file}')


args = {
    'fasta_file': {
        'type': str,
        'nargs': '+',
        'help': 'Path to FASTA file of RNA sequences or pair of paths to two '
                'FASTA files containing protein- and non-coding RNAs, '
                'respectively. (str)'
    }, 
    'model_file': {
        'type': str, 
        'help': '(Pre-)trained model to get sequence embeddings from. (str)',
    },
    '--output_file': {
        'type': str, 
        'default': 'embeddings.h5',
        'help': 'Name of hdf output file. (str)',
    },
    '--output_plot_file': {
        'type': str, 
        'default': None,
        'help': 'If specified, plots the first two dimensions of the (reduced) '
                'sequence embeddings and saves them to this file. (str)',
    },
    '--encoding_method': {
        'type': str,
        'choices': ['conv', 'bpe', 'kmer', 'nuc'],
        'default': 'conv',
        'help': 'Sequence encoding method. (str="conv")'
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
    '--pooling': {
        'type': str,
        'default': 'mean',
        'choices': ['CLS', 'mean', 'max'],
        'help': 'Type of pooling to apply. If "CLS", will extract embeddings '
                'from CLS token. (str="mean")'
    },
    '--dim_red': {
        'type': str,
        'default': 'tsne',
        'choices': ['tsne', 'pca', 'umap', 'None'],
        'help': 'Type of dimensionality reduction to apply to retrieved '
                'embeddings. If None, will not reduce dimensions. (str=None)'
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
    '--model_dir': {
        'type': str,
        'default': '.',
        'help': 'Directory where to and load the (pre-)trained model from. ' 
                '(str=f"{data_dir}/models")'
    },
}


if __name__ == '__main__':

    # Parsing arguments
    p = argparse.ArgumentParser(description=
        'Retrieves sequence embeddings by specified model for input dataset.'
    )
    for arg in args:
        p.add_argument(arg, **args[arg])
    p = p.parse_args()
    
    # Argument checks and preprocessing
    if p.encoding_method == 'bpe' and len(p.bpe_file) == 0:
        raise ValueError(
            "Please use --bpe_file flag to specify BPE model file."
        )
    p.model_dir = f'{p.data_dir}/models' if p.model_dir=='.' else p.model_dir
    
    embeddings( # Call
        p.fasta_file, p.model_file, p.output_file, p.output_plot_file, 
        p.encoding_method, p.bpe_file, p.k, p.pooling, p.dim_red, p.batch_size, 
        p.context_length, p.data_dir, p.results_dir, p.model_dir,
    )
'''Plots dataset embedding while using simulated frameshifts to asses 
sensitivity to those. Good models should be robust against them and frameshifted
sequences should be embedded similarly.'''

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lncrnapy import utils
from lncrnapy.data import Data
from lncrnapy.features import KmerTokenizer, BytePairEncoding
from lncrnapy.modules import CSEBERT, MaskedConvModel, MaskedTokenModel
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap


dim_red_functions = {
    'tsne': TSNE(),
    'pca': PCA(),
    'umap': umap.UMAP(),
}


def frameshifts(
        fasta_file, model_file, max_shift, output_file, output_plot_file, 
        encoding_method, bpe_file, k, pooling, dim_red, batch_size, 
        context_length, random_state, coding_only, data_dir, results_dir
    ):

    np.random.seed(random_state)

    # Import data
    fasta_file = [f'{data_dir}/{filepath}' for filepath in fasta_file]
    fasta_file = fasta_file[0] if len(fasta_file) == 1 else fasta_file
    data = Data(fasta_file)
    choose_from = data.num_coding_noncoding()[0] if coding_only else len(data)
    data.df['label'] = -1
    data.df['rf'] = -1
    for i, idx in enumerate(np.random.choice(choose_from, 10, replace=False)):
        to_add = data.df.iloc[[idx]*max_shift]
        to_add['sequence'] = [to_add['sequence'].iloc[j][j:] 
                              for j in range(max_shift)]
        to_add['label'] = i 
        to_add['rf'] = [j % 3 for j in range(max_shift)]
        data.df = pd.concat([data.df, to_add], axis=0, ignore_index=True)
    
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

    # Retrieving and saving the embeddings (+ dimensionality reduction)
    dim_red = dim_red_functions[dim_red] if dim_red != 'None' else None
    model.latent_space(data, inplace=True, pooling=pooling, dim_red=dim_red)
    data.to_hdf(output_file)

    if output_plot_file is not None:
        data = data.df
        plt.scatter('L0', 'L1', data=data[data['label']==-1], s=1, c='#BFBFBF')
        colors = {i:plt.rcParams['axes.prop_cycle'].by_key()['color'][i] 
                for i in range(10)}
        markers = {0:'X', 1:'^', 2:'o'}
        for i in range(10):
            for j in range(3):
                plt.scatter('L0', 'L1', c=colors[i], marker=markers[j],
                            data=data[(data['label']==i) & (data['rf']==j)])
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(f'{results_dir}/{output_plot_file}')


args = {
    'fasta_file': {
        'type': str,
        'nargs': '+',
        'help': 'Path to FASTA file of RNA sequences or pair of paths to two '
                'FASTA files containing protein- and non-coding RNAs, '
                'respectively. (str)'
    }, 
    'max_shift': {
        'type': int, 
        'help': 'Number of simulated frameshifts. (int)',
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
        'default': 'frameshift_sensitivity.h5',
        'help': 'Name of hdf output file. (str)',
    },
    '--output_plot_file': {
        'type': str, 
        'default': None,
        'help': 'Name of png output file. (str)',
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
    '--random_state': {
        'type': int,
        'default': 42,
        'help': 'Seed for sequence selection. (int=42)'
    },
    '--coding_only': {
        'action': 'store_true',
        'default': False,
        'help': 'Select only pcRNAs to assess sensitivity.'
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
    
    frameshifts( # Call
        p.fasta_file, p.model_file, p.max_shift, p.output_file, 
        p.output_plot_file, p.encoding_method, p.bpe_file, p.k, p.pooling, 
        p.dim_red, p.batch_size, p.context_length, p.random_state, 
        p.coding_only, p.data_dir, p.results_dir
    )
'''Fits a Byte Pair Encoding (BPE) model to a dataset.

Please call `python -m lncrnapy.scripts.bpe --help` for usage info.'''

import argparse
from lncrnapy.data import Data
from lncrnapy.features import BytePairEncoding


def bpe(fasta_train, vocab_size, bpe_file, data_dir):
    '''BPE fitting function as used by BPE script.'''

    train_data = Data(f'{data_dir}/{fasta_train}') # Load data
    BytePairEncoding(train_data, vocab_size=vocab_size, # Fit model
                     export_path=f'{data_dir}/{bpe_file}')
    

description = 'Fits a Byte Pair Encoding (BPE) model to a dataset.'

args = {
    'fasta_train': {
        'type': str,
        'help': 'Path to FASTA file of RNA sequences to be used for fitting the'
                ' BPE model. (str)'
    }, 
    'vocab_size': {
        'type': int, 
        'help': 'Pre-defined number of tokens in vocabulary. (str)',
    },
    '--bpe_file': {
        'type': str, 
        'default': None,
        'help': 'Name of BPE output file. (str=f"features/{vocab_size}.bpe")',
    },
    '--data_dir': {
        'type': str,
        'default': '.',
        'help': 'Parent directory to use for any of the paths specified in '
                'these arguments. (str="")'
    }, 
}


if __name__ == '__main__':

    # Parsing arguments
    p = argparse.ArgumentParser(description=description)
    for arg in args:
        p.add_argument(arg, **args[arg])
    p = p.parse_args()
    p.bpe_file = (f'features/{p.vocab_size}.bpe' if p.bpe_file is None 
                                                 else p.bpe_file)
    
    bpe( # Call
        p.fasta_train, p.vocab_size, p.bpe_file, p.data_dir
    )
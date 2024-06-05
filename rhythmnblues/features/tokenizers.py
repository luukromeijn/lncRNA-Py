'''Tokenization methods required for deep learning language models.'''

import itertools
import numpy as np
from rhythmnblues import utils


class TokenizerBase:
    '''Base class for tokenizers, only for some shared attributes.

    Attributes
    ----------
    `context_length`: `int`
        Number of tokens this tokenizer generates per sample.
    `tokens`: `dict[str:int]`
        Mapping of sequences (or token indicators such as 'CLS') to the integer
        values that these tokens are represented by.
    `name`: `list[str]`
        List of column names for the generated tokens.'''

    def __init__(self, context_length, method_name):
        '''Initializes `TokenizerBase` object.
        
        Arguments
        ---------
        `context_length`: `int`
            Number of tokens this tokenizer generates per sample.
        `method_name`: `str`
            Name of the tokenization method (used to generate unique column
            names.)'''
        
        self.context_length = context_length
        self.tokens = {'CLS': 0, 'SEP': 1, 'PAD': 2, 'UNK': 3}
        self.name = [f'T{i} {method_name}' for i in range(self.context_length)]
        

class KmerTokenizer(TokenizerBase):
    '''Tokenizer based on k-mers, every k-mer is given its own token.
    
    Attributes
    ----------
    `k`: `int`
        Length of k-mers.'''

    def __init__(self, k, context_length=512):
        '''Initializes `KmerTokenizer` object.
        
        Arguments
        ---------
        `k`: `int`
            Length of k-mers.
        `context_length`: `int`
            Number of tokens this tokenizer generates per sample.'''
        
        super().__init__(context_length, f'({k}-mer)')
        self.k = k
        self.tokens.update(
            {''.join(list(kmer)):(i + len(self.tokens)) for i, kmer in 
             enumerate(itertools.product('ACGT', repeat=self.k))}
        )

    def calculate(self, data):
        '''Calculates the token representations of all sequences in `data`.'''
        data.check_columns(['sequence'])
        tokens = []
        for _, row in utils.progress(data.df.iterrows()):
            tokens.append(self.calculate_kmer_tokens(row['sequence']))
        return np.stack(tokens)
    
    def calculate_kmer_tokens(self, sequence):
        '''Tokenizes `sequence`.'''

        # Initialize all tokens as 'PAD' except first ('CLS')
        tokens = self.tokens['PAD']*np.ones(self.context_length, dtype=int)
        tokens[0] = self.tokens['CLS']

        # Loop through k-mers and add tokens at correct index
        for t, i in enumerate(
            range(self.k, 
                  min(len(sequence)+1, ((self.context_length-1)*self.k)+1), 
                  self.k)
            ):
            kmer = sequence[i-self.k:i]
            try:
                tokens[t+1] = self.tokens[kmer]
            except KeyError: # In case of non-canonical bases (e.g. N)
                tokens[t+1] = self.tokens['UNK'] 
        if t+2 < self.context_length:
            tokens[t+2] = self.tokens['SEP'] # Add seperator if still space left
            
        return tokens
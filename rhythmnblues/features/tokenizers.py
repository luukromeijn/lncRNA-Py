'''Tokenization methods required for deep learning language models.'''

import pandas as pd
import io
import itertools
import numpy as np
import sentencepiece as spm
from rhythmnblues import utils


class TokenizerBase:
    '''Base class for tokenizers, only for some shared attributes.

    Attributes
    ----------
    `context_length`: `int`
        Number of tokens this tokenizer generates per sample.
    `vocab_size`: `int`
        The number of unique tokens known by the model.
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
        self.tokens = utils.TOKENS.copy()
        self.name = [f'T{i} {method_name}' for i in range(self.context_length)]
    
    @property
    def vocab_size(self):
        '''The number of unique tokens known by the model.'''
        return len(self.tokens)
        

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
    

class BytePairEncoding(TokenizerBase):
    '''Byte Pair Encoding from sentenciepiece, applied to nucleotide sequences.

    References
    ----------
    BPE: Sennrich et al. (2016) https://doi.org/10.18653/v1/P16-1162
    DNABERT-2: Zhou et al. (2023) https://doi.org/10.48550/arXiv.2306.15006'''

    def __init__(self, data, context_length=512, vocab_size=768, 
                 max_sentence_length=8000, export_path=None):
        '''Initializes `BytePairEncoding` object.
        
        Arguments
        ---------
        `data`: `Data`|`str`
            Data object to fit BPE tokenizer on, or path to existing model.
        `context_length`: `int`
            Number of tokens this tokenizer generates per sample (default 512).
        `vocab_size`: `int`
            Number of unique tokens known to the model (includes CLS, PAD, etc.)
            Disregarded if `data` is of type `str` (default is 768).
        `max_sentence_length`: `int`
            Maximum length of sequences to consider for fitting the BPE model.
            Disregarded if `data` is of type `str` (default is 8000).
        `export_path`: `str`
            If specified, saves BPE model to this path (default is None).'''
        
        super().__init__(context_length, 'BPE')
        self.encoder = spm.SentencePieceProcessor

        if type(data) == str: # Import model if filepath is provided
            self.encoder = self.encoder(model_file=data)
        
        else: # Fit model on data if data is provided
            stream = (io.BytesIO() if export_path is None 
                                   else open(export_path,'wb'))
            sequences = iter(data.df['sequence'].tolist())
            spm.SentencePieceTrainer.train(
                sentence_iterator=sequences, model_writer=stream,
                vocab_size=vocab_size, model_type='bpe', 
                control_symbols=['MASK'], bos_id=self.tokens['CLS'], # MASK=0
                eos_id=self.tokens['SEP'], pad_id=self.tokens['PAD'], 
                unk_id=self.tokens['UNK'], add_dummy_prefix=False, 
                character_coverage=1.0, max_sentence_length=max_sentence_length
            )
            if export_path is None:
                self.encoder = self.encoder(model_proto=stream.getvalue())
            else:
                stream.close() # Important to close stream before file reading 
                self.encoder = self.encoder(model_file=export_path)

    def calculate(self, data):
        '''Calculates the BPE tokens for all rows in `data`.'''
        return np.array(
            [[self.tokens['CLS']] + # CLS token
             encoding[:self.context_length-1] + # Actual encoding
             # Separator/padding token (if space allows it)
             [self.tokens['SEP']]*int(len(encoding)<self.context_length-1) +
             [self.tokens['PAD']]*(self.context_length-2-len(encoding))
             # This is where we call the encoding function
             for encoding in self.encoder.encode(data.df['sequence'].tolist())]
        )

    @property
    def vocab_size(self):
        # NOTE: We verified that the vocab_size parameter of sentencepiece 
        # includes the 'special' tokens like CLS, PAD, etc.
        return self.encoder.vocab_size()
    
    def get_piece_length_stats(self):
        '''Returns the avg, std, min, and max length of word pieces in the BPE 
        vocabulary.'''
        lengths = [len(self.encoder.IdToPiece(i)) 
                   for i in range(self.vocab_size)]
        return pd.DataFrame(
            [[self.vocab_size, 
              np.average(lengths), 
              np.std(lengths), 
              np.min(lengths), 
              np.max(lengths)]], 
            columns=['Vocab size', 'avg', 'std', 'min', 'max']
        )
    

class BytePairEncodingLength(BytePairEncoding):
    '''Calculates the full Byte Pair Encoding length, without special tokens 
    (e.g. CLS), assuming no context length cut-off.'''

    def __init__(self, data, vocab_size=768, max_sentence_length=8000, 
                 export_path=None):
        '''Initializes `BytePairEncodingLength` object.
        
        Arguments
        ---------
        `data`: `Data`|`str`
            Path to existing BPE tokenizer model.
        `vocab_size`: `int`
            Number of unique tokens known to the model (includes CLS, PAD, etc.)
            Disregarded if `data` is of type `str` (default is 768).
        `max_sentence_length`: `int`
            Maximum length of sequences to consider for fitting the BPE model.
            Disregarded if `data` is of type `str` (default is 8000).
        `export_path`: `str`
            If specified, saves BPE model to this path (default is None).'''
        
        super().__init__(
            data, context_length=0, vocab_size=vocab_size, 
            max_sentence_length=max_sentence_length, export_path=export_path
        )
        self.context_length = None
        self.name = 'BPE length'

    def calculate(self, data):
        '''Calculates theoretical BPE length for every row in `data`.'''
        return np.array([len(encoding) for encoding 
                         in self.encoder.encode(data.df['sequence'].tolist())])


class TokenLocalization:
    '''Adds special MASK token to tokenized input sequence, with the location of
    this MASK as new feature.'''

    def __init__(self, tokenizer):
        '''Initializes `TokenLocalization` object for given `tokenizer`, which
        should inherit from `TokenizerBase`.'''
        self.context_length = tokenizer.context_length
        self.apply_to = tokenizer.name
        self.name = 'TL'

    def calculate(self, data):
        '''Adds MASK token and localization target to every row in `data`.'''
        data.check_columns(self.apply_to)
        idx = np.random.randint(0, self.context_length, len(data))
        for i, j in utils.progress(enumerate(idx)):
            data.df[self.apply_to[j]].iat[i] = utils.TOKENS['MASK']
        return idx
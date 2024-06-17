'''Initializes a Byte Pair Encoder and encodes the train/validation sets.'''

import argparse
from rhythmnblues.data import Data
from rhythmnblues.features import BytePairEncoding

data_dir = '/exports/sascstudent/lromeijn/data'

# For duranium
data_dir = '/data/s2592800/data'

def encode(vocab_size):

    trainset = Data([f'{data_dir}/sequences/train_pcrna.fasta', 
                     f'{data_dir}/sequences/train_ncrna.fasta'])
    validset = Data([f'{data_dir}/sequences/valid_pcrna.fasta', 
                     f'{data_dir}/sequences/valid_ncrna.fasta'])
    
    bpe = BytePairEncoding(
        trainset, vocab_size=vocab_size, max_sentence_length=10000, 
        export_path=f'{data_dir}/features/{vocab_size}.bpe'
    )

    trainset.calculate_feature(bpe)
    validset.calculate_feature(bpe)

    trainset.to_hdf(f'{data_dir}/tables/train_{vocab_size}.h5')
    validset.to_hdf(f'{data_dir}/tables/valid_{vocab_size}.h5')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('vocab_size', type=int)
    args = parser.parse_args()
    encode(args.vocab_size)
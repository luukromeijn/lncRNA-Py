import argparse
from rhythmnblues.data import Data
from rhythmnblues.features import BytePairEncoding


data_dir = '/exports/sascstudent/lromeijn/data' # SHARK
# data_dir = '/data1/s2592800/data' # ALICE


def encode_simple(vocab_size, context_length):
    '''Given an existing BPE model, encode the train/valid set.'''

    train = Data([f'{data_dir}/sequences/train_pcrna.fasta',
                f'{data_dir}/sequences/train_ncrna.fasta'])
    valid = Data([f'{data_dir}/sequences/valid_pcrna.fasta',
                f'{data_dir}/sequences/valid_ncrna.fasta'])
    
    bpe = BytePairEncoding(f'{data_dir}/features/{vocab_size}', context_length)

    train.calculate_feature(bpe)
    valid.calculate_feature(bpe)

    train.to_hdf(
        f'{data_dir}/tables/train_vs{vocab_size}_cl{context_length}.bpe'
    )
    valid.to_hdf(
        f'{data_dir}/tables/train_vs{vocab_size}_cl{context_length}.bpe'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('vocab_size', type=int)
    parser.add_argument('context_length', type=int)
    args = parser.parse_args()
    encode_simple(args.vocab_size, args.context_length)
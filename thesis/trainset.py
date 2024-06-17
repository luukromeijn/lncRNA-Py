'''Creates a combination of GENCODE/NONCODE/RefSeq to be used as training set
as well as a validation set of 7500 sequences.
'''

import copy
import pandas as pd
from rhythmnblues.data import Data
from rhythmnblues.features import Length
import pandas as pd

data_dir = '/exports/sascstudent/lromeijn/data'

# For duranium
data_dir = '/data/s2592800/data'

valid_size = 7500
seq_dir = f'{data_dir}/sequences'

gencode = Data([f'{seq_dir}/gencode.v45.pc_transcripts.fa', 
                f'{seq_dir}/gencode.v45.lncRNA_transcripts.fa'])
noncode = Data(f'{seq_dir}/NONCODE.lncRNA.fa')
noncode.df['label'] = 'ncrna'
refseq = Data([f'{seq_dir}/refseq223_pcrna.fasta', 
               f'{seq_dir}/refseq223_ncrna.fasta'])

train_data = []
valid_data = []

for dataset in [gencode, noncode, refseq]:
    dataset.calculate_feature(Length())
    dataset.filter_outliers('length', [100, 10000])
    train, valid = dataset.train_test_split(int(valid_size/3), random_state=42)
    train_data.append(train)
    valid_data.append(valid)

trainset = copy.deepcopy(train_data[0])
trainset.df = pd.concat([data.df for data in train_data], ignore_index=True)

validset = copy.deepcopy(valid_data[0])
validset.df = pd.concat([data.df for data in valid_data], ignore_index=True)

trainset.to_fasta([f'{seq_dir}/train_pcrna.fasta',
                   f'{seq_dir}/train_ncrna.fasta'])
validset.to_fasta([f'{seq_dir}/valid_pcrna.fasta',
                   f'{seq_dir}/valid_ncrna.fasta'])
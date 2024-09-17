'''Creates the (pre-)training and test sets for this study.'''

import copy
import numpy as np
import pandas as pd
from rhythmnblues.data import Data
from rhythmnblues.features import Length

seq_dir = '/data/s2592800/data/sequences'

# Loading the data
gencode = Data([f'{seq_dir}/gencode.v46.pc_transcripts.fa',
                f'{seq_dir}/gencode.v46.lncRNA_transcripts.fa'])
noncode = Data(f'{seq_dir}/NONCODE.lncRNA.fa')
noncode.df['label'] = 'ncRNA'
refseq = Data([f'{seq_dir}/refseq225_human_pcrna.fasta',
               f'{seq_dir}/refseq225_human_ncrna.fasta'])

# Remove sequences < 100 bp
for dataset in [gencode, noncode, refseq]:
    dataset.calculate_feature(Length())
    dataset.filter_outliers('length', [100, np.inf]) 

# Splitting GENCODE (90% train, 5% validation, 5% test)
finetune, test = gencode.train_test_split(0.1,random_state=42)
test, valid = test.train_test_split(0.5,random_state=42)

# Merging the remaining part of GENCODE with RefSeq and NONCODE
pretrain = copy.deepcopy(finetune)
pretrain.df = pd.concat([finetune.df, noncode.df, refseq.df], ignore_index=True)

for i, dataset in enumerate([pretrain, finetune, valid, test]):
    print(['pretrain:', 'finetune:', 'valid:', 'test:'][i])
    print("(pcRNA, ncRNA):", dataset.num_coding_noncoding())

# Export files
pretrain.to_fasta([f'{seq_dir}/pretrain_human_pcrna.fasta',
                   f'{seq_dir}/pretrain_human_ncrna.fasta'])
finetune.to_fasta([f'{seq_dir}/finetune_gencode_pcrna.fasta',
                   f'{seq_dir}/finetune_gencode_ncrna.fasta'])
valid.to_fasta(   [f'{seq_dir}/valid_gencode_pcrna.fasta',
                   f'{seq_dir}/valid_gencode_ncrna.fasta'])
test.to_fasta(    [f'{seq_dir}/test_gencode_pcrna.fasta',
                   f'{seq_dir}/test_gencode_ncrna.fasta'])
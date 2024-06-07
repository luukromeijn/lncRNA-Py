from rhythmnblues.data import Data
import numpy as np

tables_folder = '/exports/sascstudent/lromeijn/data/tables/copy'
datasets = [
    'gencode', 
    'refseq', 
    'noncode-refseq', 
    'cpat_train', 
    'cpat_test'
    ]

for dataset_name in datasets:
    
    full_file = f'{tables_folder}/{dataset_name}'
    dataset = Data(hdf_filepath=f'{full_file}.h5')
    train, test = dataset.train_test_split(test_size=0.1, random_state=42)
    train.to_hdf(f'{full_file}_train.h5')
    test.to_hdf(f'{full_file}_test.h5')
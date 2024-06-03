import pytest
import torch
from rhythmnblues.data import Data
from rhythmnblues.features import Length
from rhythmnblues import utils

utils.DEVICE = torch.device('cpu') 

@pytest.fixture(scope="class")
def data():
    '''Simple data object with only id, sequence, and label'''
    return Data(['tests/data/pcrna.fasta', 'tests/data/ncrna.fasta'])

@pytest.fixture(scope="class")
def data_hdf():
    '''Data object with id, sequence, label, and length as feature'''
    return Data(['tests/data/pcrna.fasta', 'tests/data/ncrna.fasta'], 
                'tests/data/test.h5')

@pytest.fixture(scope="class")
def data_unlabelled():
    '''Unlabelled data object with id, sequence, and length as feature'''
    data = Data('tests/data/pcrna.fasta')
    data.calculate_feature(Length())
    return data
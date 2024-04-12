import pytest
from rhythmnblues.data import Data

@pytest.fixture(scope="class")
def data():
    '''Simple data object with only id, sequence, and label'''
    return Data('tests/data/pcrna.fasta', 'tests/data/ncrna.fasta')

@pytest.fixture(scope="class")
def data_hdf():
    '''Data object with id, sequence, label, and length as feature'''
    return Data('tests/data/pcrna.fasta', 'tests/data/ncrna.fasta', 
                'tests/data/test.h5')
import pytest
import numpy as np
from lncrnapy.selection import *
from lncrnapy.selection.methods import FeatureSelectionBase

@pytest.mark.parametrize('method', [
    NoSelection, TTestSelection, RegressionSelection, ForestSelection, 
    PermutationSelection, RFESelection, MDSSelection
])
def test_select_features_no_error(method, data):
    cols = list('abcdef')
    data.df[cols] = np.random.random((len(data),6))
    method = method(5)
    method.select_features(data, cols)
    # What if k < available features?

def test_k_most_important_indices_nan():
    '''Assert that nan are put last'''
    test = np.arange(0,100, dtype=np.float32)
    test[0] = np.nan
    fs = FeatureSelectionBase('dummy', 'dummy', 100)
    indices = fs.k_most_important_indices(test)
    assert np.isnan(test[indices[-1]])

def test_k_most_important_indices_large_k():
    '''Assert that nan are put last'''
    test = np.arange(0,100, dtype=np.float32)
    fs = FeatureSelectionBase('dummy', 'dummy', 101)
    with pytest.raises(RuntimeError): # Raise error for k=101
        fs.k_most_important_indices(test)
    fs = FeatureSelectionBase('dummy', 'dummy', 100) # 100 should be fine
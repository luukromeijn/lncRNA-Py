import pytest
import numpy as np
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
from Bio import SeqIO
from torch.utils.data import DataLoader
from rhythmnblues.data import Data, plot_cross_dataset_violins
from rhythmnblues.features import Length

def test_init_fasta(data):
    for col, test_col in zip(data.df.columns, ['id', 'sequence', 'label']):
        assert col == test_col
    assert (
        data.df[data.df['id']=='test']['sequence'].iloc[0] == 
        'GTTAACTTGCCGTCAGCCTTTTCNTTGACCTCTTCTTTCTGTTCATGTGTATTTGCTGTC'
    )
    assert len(data.df[data.df['label']=='pcrna']) == 10
    assert len(data.df[data.df['label']=='ncrna']) == 9

def test_init_hdf(data_hdf):
    assert (
        data_hdf.df[data_hdf.df['id']=='test']['sequence'].iloc[0] == 
        'GTTAACTTGCCGTCAGCCTTTTCNTTGACCTCTTCTTTCTGTTCATGTGTATTTGCTGTC'
    )
    assert len(data_hdf.df[data_hdf.df['label']=='pcrna']) == 10
    assert len(data_hdf.df[data_hdf.df['label']=='ncrna']) == 9
    assert 'length' in data_hdf.df.columns

def test_init_unlabelled(data_unlabelled):
    assert data_unlabelled.check_columns(['label'], 'bool') == False
    assert len(data_unlabelled) == 10

def test_to_hdf(data, tmp_path):
    data.to_hdf(str(tmp_path) + '/test.h5')

def test_to_fasta_split(data, tmp_path):
    tmp_path = str(tmp_path)
    data.to_fasta([tmp_path + '/pc.fasta', tmp_path + '/nc.fasta'])
    new_data = Data([tmp_path + '/pc.fasta', tmp_path + '/nc.fasta'])
    assert data.num_coding_noncoding() == new_data.num_coding_noncoding()

def test_to_fasta_one(data, tmp_path):
    tmp_path = str(tmp_path) + '/all.fasta'
    data.to_fasta(fasta_filepath=tmp_path)
    assert len([s for s in SeqIO.parse(tmp_path, 'fasta')]) == 19

def test_num_coding_noncoding(data):
    coding, noncoding = data.num_coding_noncoding()
    assert coding == 10
    assert noncoding == 9
    
def test_calculate_feature(data):
    extractor = Length()
    data.calculate_feature(Length())
    assert extractor.name in data.df.columns

def test_all_features(data_hdf):
    for col, col_test in zip(data_hdf.all_features(), ['length']):
        assert col == col_test

def test_check_columns(data):
    assert data.check_columns(['length'], behaviour='bool') == False
    assert data.check_columns(['sequence'], behaviour='bool') == True
    data.check_columns(['sequence'], behaviour='error')
    with pytest.raises(RuntimeError):
        data.check_columns(['length'], behaviour='error')
    with pytest.raises(ValueError):
        data.check_columns(['sequence'], behaviour='blabla')

def test_train_test_split(data):
    train, test = data.train_test_split(0.2)
    assert type(train) == Data
    assert type(test) == Data

def test_coding_noncoding_split(data):
    pc, nc = data.coding_noncoding_split()
    assert type(pc) == Data
    assert len(pc.df[pc.df['label']=='pcrna']) == 10 
    assert len(pc.df[pc.df['label']=='ncrna']) == 0 
    assert type(nc) == Data
    assert len(nc.df[nc.df['label']=='pcrna']) == 0 
    assert len(nc.df[nc.df['label']=='ncrna']) == 9 

def test_test_features(data_hdf):
    result = data_hdf.test_features(['length'])

def test_plot_feature_boxplot(data_hdf, data_unlabelled):
    for data in [data_hdf, data_unlabelled]:
        assert type(data.plot_feature_boxplot('length')) == Figure

def test_plot_feature_density(data_hdf, data_unlabelled):
    for data in [data_hdf, data_unlabelled]:
        assert type(data.plot_feature_density('length')) == Figure

def test_plot_feature_violin(data_hdf, data_unlabelled):
    for data in [data_hdf, data_unlabelled]:
        assert type(data.plot_feature_violin('length')) == Figure

def test_plot_feature_space(data_hdf, data_unlabelled):
    for data in [data_hdf, data_unlabelled]:
        data.df['length 2'] = Length().calculate(data)
        assert type(data.plot_feature_scatter('length', 'length 2')) == Figure

def test_plot_feature_space_pca(data):
    dummy_features = [str(i) for i in range(20)]
    data.df[dummy_features] = np.random.random((len(data.df), 20))
    assert type(data.plot_feature_space(dummy_features, dim_red=PCA()))==Figure

def test_plot_feature_correlation(data):
    dummy_features = [str(i) for i in range(20)]
    data.df[dummy_features] = np.random.random((len(data.df), 20))
    assert type(data.plot_feature_correlation(dummy_features))==Figure

def test_plot_cross_dataset_violins(data_hdf):
    assert type(plot_cross_dataset_violins([data_hdf],['a'],'length')) == Figure

def test_filter_outliers(data):
    data.df['test'] = 10
    data.df['test'].iloc[0] = 20
    data.filter_outliers('test', [10,21])
    assert len(data.df) == 19
    data.filter_outliers('test', 3)
    assert len(data.df) == 18

def test_filter_sequence_quality(data):
    # Given that we know only one sequence contains an N-base (the first ncrna)
    data.filter_sequence_quality(0.1)
    assert len(data.df) == 19
    data.filter_sequence_quality(0)
    assert len(data.df) == 18

@pytest.mark.parametrize('pc,nc,N,replace',[
    (0, 0, None, False),
    (None, None, 0, False),
    (5, 5, None, False),
    (None, None, 10, False),
    (None, None, 20, False),
    (None, None, 20, True),
    (10, 10, None, True),
])
def test_sample(data, pc, nc, N, replace):
    if N is not None and N > len(data.df) and not replace:
        with pytest.raises(ValueError):
            subset = data.sample(pc, nc, N, replace)
    else:
        subset = data.sample(pc, nc, N, replace)
        coding, noncoding = subset.num_coding_noncoding()
        if N is not None:
            assert N == coding + noncoding
        else:
            assert pc == coding
            assert nc == noncoding

def test_set_tensor_features(data):
    with pytest.raises(AttributeError):
        data[0]
    data.calculate_feature(Length())
    data.set_tensor_features(['length'])
    assert data[0][0].dim() == 1 # Single element indexed -> 1 dimension
    assert data[0][0].size(0) == 1 # Length feature
    assert data[0][1].dim() == 1 # Single element indexed -> 1 dimension
    assert data[0][1].size(0) == 1 # Target label
    assert data[[0,1]][0].dim() == 2 # Two elements indexed -> 2 dimensions
    assert data[[0,1]][0].size(0) == 2 # Two elements
    assert data[[0,1]][0].size(1) == 1 # Length feature
    assert data[[0,1]][1].dim() == 2 # Two elements indexed -> 2 dimensions
    assert data[[0,1]][1].size(0) == 2 # Two elements
    assert data[[0,1]][1].size(1) == 1 # Target label
    for x, y in DataLoader(data, batch_size=3):
        assert len(x) == 3
        assert len(y) == 3
        break
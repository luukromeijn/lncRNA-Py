import pytest
import torch
from lncrnapy.features import (KmerFreqs, KmerTokenizer, ORFCoordinates, 
                                   Standardizer)
from lncrnapy.modules import Classifier, Regressor
from lncrnapy.modules import MaskedTokenModel, MaskedConvModel
from lncrnapy.modules import MycoAICNN, BERT, CSEBERT, CSEResNet
from lncrnapy.train.loggers import *
from lncrnapy.train import train_classifier, train_regressor
from lncrnapy.train import train_masked_token_modeling
from lncrnapy.train import train_masked_conv_modeling
from lncrnapy import utils

loggers = [
    ('LoggerBase',  lambda path: LoggerBase()),
    ('LoggerPrint', lambda path: LoggerPrint()),
    ('LoggerWrite', lambda path: LoggerWrite(str(path) + '/test.csv')),
    ('LoggerPlot',  lambda path: LoggerPlot(path)),
    ('LoggerList',  lambda path: LoggerList(LoggerPrint(), LoggerPlot(path)))
]

@pytest.fixture(scope="function", params=loggers, ids=[n for n,_ in loggers])
def logger(request, tmp_path):
    _, fixture_func = request.param
    return fixture_func(tmp_path)

def test_logger_start(logger):
    logger.start({'Test1': lambda x: x})
    assert len(logger.columns) == 4
    assert logger.columns[0] == 'Loss|train'
    assert logger.columns[1] == 'Test1|train'
    assert logger.columns[2] == 'Loss|valid'
    assert logger.columns[3] == 'Test1|valid'

def test_logger_log(logger):
    logger.start({})
    logger.log([0,0], 'model_dummy')

def test_train_classifier_cnn(data):
    kmers = KmerFreqs(3)
    model = Classifier(MycoAICNN())
    data.calculate_feature(kmers)
    data.set_tensor_features(kmers.name)
    train_classifier(model, data, data, 1)

def test_train_classifier_bert(data):
    kmers = KmerTokenizer(3)
    model = Classifier(BERT(kmers.vocab_size, N=1))
    data.calculate_feature(kmers)
    data.set_tensor_features(kmers.name, torch.long)
    train_classifier(model, data, data, 1)

def test_train_classifier_csebert(data):
    model = Classifier(CSEBERT(12, N=1))
    data.set_tensor_features('4D-DNA')
    train_classifier(model, data, data, 1)

def test_train_masked_token_modeling(data):
    kmers = KmerTokenizer(3)
    model = MaskedTokenModel(BERT(kmers.vocab_size, N=1))
    data.calculate_feature(kmers)
    data.set_tensor_features(kmers.name, torch.long)
    train_masked_token_modeling(model, data, data, 1)

def test_train_masked_conv_modeling(data):
    model = MaskedConvModel(CSEBERT(12, N=1))
    data.set_tensor_features('4D-DNA')
    train_masked_conv_modeling(model, data, data, 1)

def test_train_regression(data):
    kmers = KmerTokenizer(3)
    model = Regressor(BERT(kmers.vocab_size, N=1), n_features=2)
    orf = ORFCoordinates()
    data.calculate_feature(kmers)
    data.calculate_feature(orf)
    s_orf = Standardizer(data, orf.name)
    data.calculate_feature(s_orf)
    data.set_tensor_features(kmers.name, torch.long, s_orf.name)
    train_regressor(model, data, data, 1, standardizer=s_orf)

def test_utils_device():
    '''The code assumes that device is device object (and not a string!)'''
    assert type(utils.DEVICE) == torch.device
    assert type(utils.DEVICE) != str
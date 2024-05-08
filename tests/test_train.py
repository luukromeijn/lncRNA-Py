import pytest
from rhythmnblues.features import KmerFreqs
from rhythmnblues.modules import MycoAICNN, Model
from rhythmnblues.train.loggers import *
from rhythmnblues.train import train_classifier

loggers = [
    ('LoggerBase',  lambda path: LoggerBase()),
    ('LoggerPrint', lambda path: LoggerPrint()),
    ('LoggerWrite', lambda path: LoggerWrite(str(path) + '/test.csv')),
    ('LoggerPlot',  lambda path: LoggerPlot(path)),
]

@pytest.fixture(scope="function", params=loggers, ids=[n for n,_ in loggers])
def logger(request, tmp_path):
    _, fixture_func = request.param
    return fixture_func(tmp_path)

def test_logger_set_columns(logger):
    logger.set_columns({'Test1': lambda x: x})
    assert len(logger.columns) == 4
    assert logger.columns[0] == 'Loss|train'
    assert logger.columns[1] == 'Test1|train'
    assert logger.columns[2] == 'Loss|valid'
    assert logger.columns[3] == 'Test1|valid'

def test_logger_log(logger):
    logger.set_columns({})
    logger.log([0,0])

def test_train_classifier(data):
    kmers = KmerFreqs(3)
    model = Model(MycoAICNN())
    data.calculate_feature(kmers)
    data.set_tensor_features(kmers.name)
    train_classifier(model, data, data, 1)
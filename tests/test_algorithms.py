import pytest
from rhythmnblues.algorithms import Algorithm
from rhythmnblues.algorithms.traditional import *
from rhythmnblues.features import *
from rhythmnblues.modules import Classifier
from rhythmnblues.modules import MycoAICNN

algorithms = [
    ('CPAT',   lambda data: CPAT('tests/data/fickett_paper.txt', data)),
    ('CNCI',   lambda data: CNCI(data)),
    ('CNIT',   lambda data: CNIT(data)),
    ('PLEK',   lambda data: PLEK()),
    ('CPC2',   lambda data: CPC2('tests/data/fickett_paper.txt')),
    ('FEELnc', lambda data: FEELnc(data)),
    ('CPPred', lambda data: CPPred('tests/data/fickett_paper.txt', data)),
    ('DeepCPP',lambda data: DeepCPP('tests/data/fickett_paper.txt', data, 
                                    'tests/data/zhang_ref.txt'))
]

@pytest.fixture(scope="class", params=algorithms, ids=[n for n,_ in algorithms])
def algorithm(request, data):
    _, fixture_func = request.param
    return fixture_func(data)

def test_algorithm_feature_extraction(algorithm, data):
    alg_instance = algorithm
    alg_instance.feature_extraction(data)

def test_algorithm_fit_predict(algorithm, data):
    alg_instance = algorithm
    alg_instance.fit(data)
    alg_instance.predict(data)

@pytest.mark.parametrize('alg',[CPC, iSeeRNA])
def test_fit_predict_no_blast(data, alg):
    # Bypasses blastx by skipping feature extraction
    algorithm = alg('dummy_string')
    data.df[algorithm.used_features] = (
        np.random.random((len(data), len(algorithm.used_features)))
    )
    algorithm.fit(data)
    algorithm.predict(data)

def test_algorithm_base_single_extractor_no_feature_names():
    alg = Algorithm('dummy', Length()) 
    assert type(alg.feature_extractors) == list
    assert type(alg.used_features) == list
    assert len(alg.used_features) == 1
    assert alg.used_features[0] == 'length'

def test_algorithm_base_multi_extractors_and_feature_names():
    alg = Algorithm('dummy', [Length(), KmerFreqs(3)], ['length', 'ATG']) 
    assert type(alg.feature_extractors) == list
    assert len(alg.feature_extractors) == 2
    assert type(alg.used_features) == list
    assert len(alg.used_features) == 2
    assert alg.used_features[1] == 'ATG'

def test_algorithm_base_deep(data):
    alg = Algorithm(Classifier(MycoAICNN()), KmerFreqs(3))
    with pytest.raises(AttributeError):
        alg.fit(data)
    pred = alg.predict(data)
    assert len(pred) == len(data)
    assert pred[0] in ['pcRNA', 'ncRNA']
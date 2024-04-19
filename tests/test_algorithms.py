import pytest
from rhythmnblues.algorithms import *
from rhythmnblues.feature_extraction import *

algorithms = [
    ('CPAT',   lambda data: CPAT('tests/data/fickett_paper.txt', data)),
    ('CNCI',   lambda data: CNCI(data)),
    ('CNIT',   lambda data: CNIT(data)),
    ('PLEK',   lambda data: PLEK()),
    ('CPC2',   lambda data: CPC2('tests/data/fickett_paper.txt')),
    ('FEELnc', lambda data: FEELnc(data))
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

def test_cpc_fit_predict(data):
    # Bypasses blastx by skipping feature extraction
    algorithm = CPC('dummy_string')
    data.df[algorithm.used_features] = (
        np.random.random((len(data), len(algorithm.used_features)))
    )
    algorithm.fit(data)
    algorithm.predict(data)
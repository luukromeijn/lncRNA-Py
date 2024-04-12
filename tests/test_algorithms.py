import pytest
from rhythmnblues.algorithms import *

algorithms = [
    ('cpat', lambda data: CPAT('tests/data/fickett_paper.txt', data)),
    ('cnci', lambda data: CNCI(data)),
    ('cnit', lambda data: CNIT(data)),
    ('plek', lambda data: PLEK()),
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